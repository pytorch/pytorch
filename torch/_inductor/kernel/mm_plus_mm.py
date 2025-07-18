# mypy: allow-untyped-defs

import torch

from .. import ir
from ..lowering import lowerings
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
)
from ..utils import use_aten_gemm_kernels, use_triton_template
from ..virtualized import V
from .mm_common import mm_args, mm_grid, mm_options


aten = torch.ops.aten

aten_mm_plus_mm = ExternKernelChoice(
    torch.ops.inductor._mm_plus_mm, "torch::inductor::_mm_plus_mm"
)

mm_plus_mm_template = TritonTemplate(
    name="mm_plus_mm",
    grid=mm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C", "D")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K1 = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    # K2 = {{size("C", 1)}}
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
    stride_cm = {{stride("C", 0)}}
    stride_ck = {{stride("C", 1)}}
    stride_dk = {{stride("D", 0)}}
    stride_dn = {{stride("D", 1)}}

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
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if (((stride_am == 1 and stride_ak == M) or (stride_am == K1 and stride_ak == 1))
        and ((stride_cm == 1 and stride_ck == M) or (stride_cm == K1 and stride_ck == 1))):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M

    if (((stride_bk == 1 and stride_bn == K1) or (stride_bk == N and stride_bn == 1))
        and ((stride_dk == 1 and stride_dn == K1) or (stride_dk == N and stride_dn == 1))):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    C = C + (ram[:, None] * stride_cm + rk[None, :] * stride_ck)
    D = D + (rk[:, None] * stride_dk + rbn[None, :] * stride_dn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k1 in range(K1, 0, -BLOCK_K):
        # First matmul with A @ B
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k1, other=0.)
            b = tl.load(B, mask=rk[:, None] < k1, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    for k2 in range(K1, 0, -BLOCK_K):

        # Second matmul with C @ D
        if EVEN_K:
            c = tl.load(C)
            d = tl.load(D)
        else:
            c = tl.load(C, mask=rk[None, :] < k2, other=0.)
            d = tl.load(D, mask=rk[:, None] < k2, other=0.)
        acc += tl.dot(c, d, allow_tf32=ALLOW_TF32)
        C += BLOCK_K * stride_ck
        D += BLOCK_K * stride_dk


    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
    cache_codegen_enabled_for_template=True,
)


def tuned_mm_plus_mm(mat1, mat2, mat3, mat4, *, layout=None):
    """
    Computes mm(mat1, mat2) + mm(mat3, mat4)
    """
    m1, n1, k1, layout1, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    m2, n2, _, layout2, mat3, mat4 = mm_args(mat3, mat4, layout=layout)
    device_type = ir.get_device_type(mat1)

    # Optimization is optional, because we can always just not do the fusion
    if (
        m1 * n1 == 0
        or m2 * n2 == 0
        or not V.graph.sizevars.statically_known_list_equals(
            mat1.get_size(), mat3.get_size()
        )
        or not V.graph.sizevars.statically_known_list_equals(
            mat2.get_size(), mat4.get_size()
        )
    ):
        # TODO(jansel): support different K values when this is fixed:
        # https://github.com/triton-lang/triton/issues/967
        return lowerings[aten.add](
            lowerings[aten.mm](mat1, mat2), lowerings[aten.mm](mat3, mat4)
        )

    assert layout1 == layout2
    # options to tune from
    choices = (
        [aten_mm_plus_mm.bind((mat1, mat2, mat3, mat4), layout1)]
        if use_aten_gemm_kernels()
        else []
    )

    mm_configs = V.choices.get_mm_plus_mm_configs(device_type)
    if use_triton_template(layout1):
        for config in mm_configs():
            # see https://github.com/triton-lang/triton/issues/1298
            # BLOCK_K = K causes llvm error
            if V.graph.sizevars.statically_known_lt(config.kwargs["BLOCK_K"], k1):
                mm_plus_mm_template.maybe_append_choice(
                    choices,
                    input_nodes=(mat1, mat2, mat3, mat4),
                    layout=layout1,
                    **mm_options(config, m1, n1, k1, layout1),
                )

    return autotune_select_algorithm(
        "mm_plus_mm", choices, [mat1, mat2, mat3, mat4], layout1
    )
