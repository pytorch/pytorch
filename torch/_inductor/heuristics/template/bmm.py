"""Template heuristics for batched matrix multiplication."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._inductor.heuristics.registry import register_template_heuristic
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

from ... import config
from ...autows_utils import has_two_ctas, meta_ws_enabled
from ...kernel.bmm import (
    BLACKWELL_BMM_MAX_AUTOTUNE_CONFIGS,
    blackwell_ws_persistent_tma_bmm_template,
    is_blackwell_bmm_2cta_compatible,
)
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...utils import can_use_tma, expr_fits_within_32bit, get_num_sms, has_free_symbols
from ...virtualized import V
from .base import TemplateConfigHeuristics
from .triton import mm_allow_tf32
from .triton_addmm import AddMMConfigMixin


if TYPE_CHECKING:
    from collections.abc import Generator


@register_template_heuristic(
    blackwell_ws_persistent_tma_bmm_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="bmm",
)
class CUDABlackwellBMMTemplateConfigHeuristic(TemplateConfigHeuristics):
    """Bounded configs for the Blackwell persistent-TMA BMM template."""

    bmm_configs = BLACKWELL_BMM_MAX_AUTOTUNE_CONFIGS

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")

        mat1, mat2 = kernel_inputs.mat1mat2()
        # aten.bmm is rank-3 by contract; matmul on higher ranks is reshaped to
        # bmm and anything with a 2D operand lowers to mm before reaching here.
        if len(mat1.get_size()) != 3 or len(mat2.get_size()) != 3:
            raise AssertionError("Blackwell BMM requires rank-3 operands")

        mat1_size = mat1.get_size()
        mat2_size = mat2.get_size()
        batch, batch_b = mat1_size[0], mat2_size[0]
        strides = (*mat1.get_stride(), *mat2.get_stride())
        # A backed dynamic batch is a runtime argument: only the grid and the
        # tile loop bound depend on it. The matrix dims and strides are
        # constexprs, so they must be static (strides can be symbolic for a
        # slice of a dynamic tensor), and CUDA tensor-map dimensions must be
        # positive.
        if (
            has_free_symbols((*mat1_size[1:], *mat2_size[1:], *strides))
            or has_free_symbols((mat1.get_layout().offset, mat2.get_layout().offset))
            or free_unbacked_symbols(batch)
            or free_unbacked_symbols(batch_b)
            or not V.graph.sizevars.statically_known_equals(batch, batch_b)
        ):
            return
        static_batch = isinstance(batch, (int, sympy.Integer))
        if static_batch:
            batch = batch_b = int(batch)
            if batch <= 0:
                return

        m, k = map(int, mat1_size[1:])
        k_b, n = map(int, mat2_size[1:])
        if min(m, n, k) <= 0:
            return
        # The TMA store indexes in int32. A dynamic batch is checked on its
        # hint; a larger batch recompiles (codegen guards the 32-bit index)
        # and falls back to other choices.
        if not static_batch and not all(
            expr_fits_within_32bit(numel)
            for numel in (batch * m * k, batch * k * n, batch * m * n)
        ):
            return

        # Ordinary operands use rank-3 TMA descriptors; stride-zero broadcast
        # operands use one shared rank-2 descriptor. In either case every
        # matrix-leading stride and batch base must retain TMA's 16-byte
        # alignment.
        if not can_use_tma(mat1, mat2):
            return

        if (static_batch and batch != batch_b) or k != k_b:
            raise NotImplementedError(
                "Blackwell BMM does not broadcast logical batches"
            )

        a_row_major = mat1.get_stride()[2] == 1
        a_col_major = mat1.get_stride()[1] == 1

        b_row_major = mat2.get_stride()[2] == 1
        b_col_major = mat2.get_stride()[1] == 1

        # e.g. a batch-innermost layout: TMA needs a contiguous matrix dim.
        if not (a_row_major or a_col_major) or not (b_row_major or b_col_major):
            return

        a_broadcast = int(mat1.get_stride()[0]) == 0
        b_broadcast = int(mat2.get_stride()[0]) == 0
        # Host descriptors are built from the base buffer, so they cannot carry a
        # storage offset, and one aliased kernel arg cannot hold a descriptor and
        # anything else (the other operand, or a baddbmm bias).
        host_tma_nodes = [
            node
            for node, broadcast in ((mat1, a_broadcast), (mat2, b_broadcast))
            if not broadcast
        ]
        # Count by name, not identity: baddbmm(a, a, b) passes the same node as
        # bias and mat1.
        input_names = [node.get_name() for node in kernel_inputs.nodes()]
        host_side_tma = (
            config.triton.enable_host_side_tma
            and all(node.get_layout().offset == 0 for node in host_tma_nodes)
            and all(input_names.count(node.get_name()) == 1 for node in host_tma_nodes)
        )

        output_layout = kernel_inputs.output_layout()
        flatten_output = len(output_layout.size) == 2
        # The grid reads a dynamic batch from the rank-3 output's size.
        if not static_batch and flatten_output:
            return
        rank3_output = len(output_layout.size) == 3
        tma_store = config.triton.enable_template_tma_store and can_use_tma(
            output_layout=output_layout
        )
        descriptor_options = {
            **({"BATCH_SIZE": batch} if static_batch else {}),
            "LOGICAL_M": m,
            "LOGICAL_N": n,
            "DESCRIPTOR_K": k,
            "A_BATCH_STRIDE": int(mat1.get_stride()[0]),
            "B_BATCH_STRIDE": int(mat2.get_stride()[0]),
            "K_BATCH_OFFSET": 0,
            "A_M_STRIDE": int(mat1.get_stride()[1]),
            "A_K_STRIDE": int(mat1.get_stride()[2]),
            "B_K_STRIDE": int(mat2.get_stride()[1]),
            "B_N_STRIDE": int(mat2.get_stride()[2]),
            "OUTPUT_BATCH_ROWS": m,
            "NUM_SMS": get_num_sms(),
            "HOST_SIDE_TMA": host_side_tma,
            "A_ROW_MAJOR": a_row_major,
            "B_ROW_MAJOR": b_row_major,
            "A_BROADCAST_BATCH": a_broadcast,
            "B_BROADCAST_BATCH": b_broadcast,
            "VIRTUAL_BATCH": False,
            "FLATTEN_OUTPUT": flatten_output,
        }
        use_meta_ws = meta_ws_enabled()
        for candidate in self.bmm_configs:
            # Meta autoWS data partitioning offsets the batch coordinate of a
            # rank-3 A descriptor load instead of M, so only a broadcast
            # (rank-2) A may be partitioned.
            if candidate.data_partition_factor > 1 and not a_broadcast:
                continue
            # A flattened output has no batch boundary, so an M tail tile would
            # overwrite the leading rows of the next batch.
            if flatten_output and m % candidate.block_m != 0:
                continue
            two_ctas = use_meta_ws and candidate.two_ctas and has_two_ctas()
            if two_ctas and not is_blackwell_bmm_2cta_compatible(
                output_batch_rows=m,
                block_m=candidate.block_m,
                flatten_output=flatten_output,
                tma_store=tma_store,
            ):
                continue
            template_kwargs = {
                "BLOCK_M": candidate.block_m,
                "BLOCK_N": candidate.block_n,
                "BLOCK_K": candidate.block_k,
                "K_TILES": math.ceil(k / candidate.block_k),
                "GROUP_M": 8,
                "num_stages": candidate.num_stages,
                "num_warps": candidate.num_warps,
                "EPILOGUE_SUBTILE": candidate.epilogue_subtile,
                "USE_META_WS": use_meta_ws,
                "WARP_SPECIALIZE": True,
                "FLATTEN": not use_meta_ws,
                "DATA_PARTITION_FACTOR": candidate.data_partition_factor,
                "SEPARATE_EPILOGUE_STORE": candidate.separate_epilogue_store,
                "TWO_CTAS": two_ctas,
                "RANK3_TMA_OUTPUT": tma_store and rank3_output,
                "tma_store": tma_store,
                "tma_load_for_template_epilogue": config.triton.enable_tma_load_for_template_epilogue,
                **descriptor_options,
            }
            if two_ctas:
                # The kernel strides by NUM_SMS, so it must be the even worker
                # count the grid launches; an odd count would skip tiles.
                template_kwargs["NUM_SMS"] = get_num_sms(two_ctas=True)
                template_kwargs["ctas_per_cga"] = (2, 1, 1)
            yield template_kwargs

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")
        m, n, k = kernel_inputs.mnk_symbolic()
        return {"ALLOW_TF32": mm_allow_tf32(m, n, k, kernel_inputs.device_type)}


@register_template_heuristic(
    blackwell_ws_persistent_tma_bmm_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="baddbmm",
)
class CUDABlackwellBaddbmmTemplateConfigHeuristic(
    AddMMConfigMixin, CUDABlackwellBMMTemplateConfigHeuristic
):
    """Blackwell BMM configs with the baddbmm bias applied in the epilogue."""
