import functools
import logging
from typing import Any, Dict, List, Optional

import torch
from torch._inductor.virtualized import V

from .. import config as inductor_config
from ..codegen.cuda.gemm_template import CUTLASSGemmTemplate
from ..lowering import register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
)
from ..utils import (
    use_aten_gemm_kernels,
    use_cutlass_template,
    use_max_autotune,
    use_triton_template,
)
from .mm_common import (
    addmm_epilogue,
    int8_mm_configs,
    mm_args,
    mm_configs,
    mm_grid,
    mm_options,
)

log = logging.getLogger(__name__)
aten = torch.ops.aten

scaled_mm_template = TritonTemplate(
    name="scaled_mm",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)


aten__scaled_mm = ExternKernelChoice(torch._scaled_mm, "at::_scaled_mm")


@register_lowering(aten._scaled_mm.default, type_promotion_kind=None)
def tuned_scaled_mm(
    mat1,
    mat2,
    *,
    bias,
    out_dtype,
    scale_a,
    scale_b,
    scale_result,
    use_fast_accum,
    layout=None
):
    m, n, k, layout, mat1, mat2, scale_a, scale_b = mm_args(
        mat1, mat2, scale_a, scale_b, layout=layout
    )

    choices = (
        [
            aten__scaled_mm.bind(
                (
                    mat1,
                    mat2,
                    bias,
                    out_dtype,
                    scale_a,
                    scale_b,
                    scale_result,
                    use_fast_accum,
                ),
                layout,
            )
        ]
        if use_aten_gemm_kernels()
        else []
    )

    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            scaled_mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2, scale_a, scale_b),
                layout=layout,
                **mm_options(config, m, n, k, layout),
            )

    return autotune_select_algorithm("scaled_mm", choices, [mat1, mat2], layout)
