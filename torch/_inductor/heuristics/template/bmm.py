"""Template heuristics for batched matrix multiplication."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.heuristics.registry import register_template_heuristic

from ... import config
from ...autows_utils import meta_ws_enabled
from ...kernel.bmm import (
    BLACKWELL_BMM_MAX_AUTOTUNE_CONFIGS,
    blackwell_ws_persistent_tma_bmm_template,
    is_blackwell_bmm_2cta_compatible,
)
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...utils import can_use_tma, get_num_sms, has_free_symbols
from .base import TemplateConfigHeuristics


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
        if len(mat1.get_size()) != 3 or len(mat2.get_size()) != 3:
            raise NotImplementedError("Blackwell BMM requires rank-3 operands")

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

        if not (a_row_major or a_col_major) or not (b_row_major or b_col_major):
            raise NotImplementedError(
                "Blackwell BMM requires one contiguous matrix dimension"
            )

        output_layout = kernel_inputs.output_layout()
        flatten_output = len(output_layout.size) == 2
        rank3_output = len(output_layout.size) == 3
        can_tma_store = config.triton.enable_template_tma_store and can_use_tma(
            output_layout=output_layout
        )
        descriptor_options = {
            "BATCH_SIZE": batch,
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
            "A_ROW_MAJOR": a_row_major,
            "B_ROW_MAJOR": b_row_major,
            "A_BROADCAST_BATCH": int(mat1.get_stride()[0]) == 0,
            "B_BROADCAST_BATCH": int(mat2.get_stride()[0]) == 0,
            "VIRTUAL_BATCH": False,
            "FLATTEN_OUTPUT": flatten_output,
        }
        use_meta_ws = meta_ws_enabled()
        for candidate in self.bmm_configs:
            two_ctas = use_meta_ws and candidate.two_ctas
            tma_store = can_tma_store and (
                flatten_output or (two_ctas and rank3_output)
            )
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
                "RANK3_TMA_OUTPUT": two_ctas and rank3_output,
                "tma_store": tma_store,
                **descriptor_options,
            }
            if two_ctas:
                template_kwargs["ctas_per_cga"] = (2, 1, 1)
            yield template_kwargs

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        return {"ALLOW_TF32": False}
