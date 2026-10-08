"""Template heuristics for batched matrix multiplication."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.heuristics.registry import register_template_heuristic

from ... import config
from ...autows_utils import has_two_ctas, meta_ws_enabled
from ...kernel.bmm import (
    BLACKWELL_BMM_MAX_AUTOTUNE_CONFIGS,
    blackwell_ws_persistent_tma_bmm_template,
    is_blackwell_bmm_2cta_compatible,
)
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...utils import can_use_tma, get_num_sms, has_free_symbols
from .base import TemplateConfigHeuristics
from .triton import mm_allow_tf32


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
        sizes = (*mat1_size, *mat2_size)
        # The current bounded configs require concrete dimensions, and CUDA
        # tensor-map dimensions must be positive.
        if has_free_symbols(sizes):
            return

        batch, m, k = map(int, mat1_size)
        batch_b, k_b, n = map(int, mat2_size)
        if min(batch, m, n, k) <= 0:
            return

        # Ordinary operands use rank-3 TMA descriptors; stride-zero broadcast
        # operands use one shared rank-2 descriptor. In either case every
        # matrix-leading stride and batch base must retain TMA's 16-byte
        # alignment.
        if not can_use_tma(mat1, mat2):
            return

        if batch != batch_b or k != k_b:
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
        # storage offset, and one aliased kernel arg cannot hold two descriptors.
        host_side_tma = (
            config.triton.enable_host_side_tma
            and all(
                node.get_layout().offset == 0
                for node, broadcast in ((mat1, a_broadcast), (mat2, b_broadcast))
                if not broadcast
            )
            and (a_broadcast or b_broadcast or mat1.get_name() != mat2.get_name())
        )

        output_layout = kernel_inputs.output_layout()
        flatten_output = len(output_layout.size) == 2
        tma_store = (
            flatten_output
            and config.triton.enable_template_tma_store
            and can_use_tma(output_layout=output_layout)
        )
        descriptor_options = {
            "NUM_SMS": get_num_sms(),
            "HOST_SIDE_TMA": host_side_tma,
            "A_ROW_MAJOR": a_row_major,
            "B_ROW_MAJOR": b_row_major,
            "A_BROADCAST_BATCH": a_broadcast,
            "B_BROADCAST_BATCH": b_broadcast,
            "FLATTEN_OUTPUT": flatten_output,
            "tma_store": tma_store,
            # The grid needs the logical problem, not the flattened output size.
            "call_sizes": (batch, m, n),
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
