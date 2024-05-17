import logging

import sympy

import torch

from .. import config as inductor_config
from ..lowering import (
    add_layout_constraint,
    constrain_to_fx_strides,
    empty,
    register_lowering,
)
from ..select_algorithm import (
    autotune_select_algorithm,
    # ExternKernelChoice,
    TritonTemplate,
)
from ..utils import use_triton_template  # use_max_autotune,
from .mm import _is_static_problem  # TODO(yangsiyu) move to mm_common
from .mm_common import mm_args, mm_grid, scaled_mm_configs

log = logging.getLogger(__name__)
aten = torch.ops.aten


scaled_mm_w_bias_template = TritonTemplate(
    name="scaled_mm_w_bias",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B", "A_inverse_scale", "B_inverse_scale", "bias_ptr")}}
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
        if FP8_FAST_ACCUM:
            acc = tl.dot(a, b, acc, out_dtype=ACC_TYPE, allow_tf32=ALLOW_TF32)
        else:
            acc += tl.dot(a, b, out_dtype=ACC_TYPE, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    if SCALING:
        if SCALING_ROWWISE:
            inv_a_scale_row = tl.load(A_inverse_scale + rm, mask=rm < M)
            inv_b_scale_row = tl.load(B_inverse_scale + rn, mask=rn < N)
            inv_scale_row = inv_a_scale_row[:, None] * inv_b_scale_row[None, :]
            acc *= inv_scale_row
        else:
            # for tensor-wise scaling, the scales are scalars
            inv_a_scale = tl.load(A_inverse_scale)
            inv_b_scale = tl.load(B_inverse_scale)
            inv_scale = inv_a_scale * inv_b_scale
            acc *= inv_scale

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + rn, mask=rn < N)
        acc += bias

    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)


# aten__scaled_mm = ExternKernelChoice(
#     torch._scaled_mm,
#     "at::_scaled_mm",
#     op_overload=aten._scaled_mm.default,
#     use_fallback_kernel=True,  # only FallbackKernel can handle multi-output now
# )
# also changed select_algorithm to use_fallback_kernel if True without checking other conditions


def scaled_mm_options(
    config,
    sym_m,
    sym_n,
    sym_k,
    layout,
    bias,
    scale_a,
    scale_b,
    use_fast_accum,
    b_prologue_cast_type=None,
):
    # copied from mm_common.mm_options()
    even_k_symbolic = (
        sympy.gcd(sym_k, config.kwargs["BLOCK_K"]) == config.kwargs["BLOCK_K"]
    )
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32 and (
        not inductor_config.force_same_precision
        or ((sym_m % 16) == 0 and (sym_n % 16) == 0 and (sym_k % 8) == 0)
    )

    if scale_a or scale_b:
        assert len(scale_a.get_size()) == len(
            scale_b.get_size()
        ), "Expect inverse scale_a and scale_b to be both scalars (tensor-wise scaling) or tensors (rowwise scaling)"
    is_scaling = not (scale_a is None and scale_b is None)
    return dict(
        GROUP_M=8,
        EVEN_K=even_k_symbolic,
        ALLOW_TF32=allow_tf32,
        ACC_TYPE="tl.float32",  # should acc always be in float32?
        B_PROLOGUE_CAST_TYPE=b_prologue_cast_type,
        FP8_FAST_ACCUM=use_fast_accum,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        SCALING=is_scaling,
        SCALING_ROWWISE=is_scaling
        and len(scale_a.get_size()) != 0,  # tensor-wise scaling if scalar scale
        HAS_BIAS=bias is not None,
        **config.kwargs,
    )


@register_lowering(aten._scaled_mm.default, type_promotion_kind=None)
def tuned_scaled_mm(
    mat_a,
    mat_b,
    bias=None,
    out_dtype=None,
    scale_a=None,
    scale_b=None,
    scale_result=None,
    use_fast_accum=True,
    # out, amax  # if using _scaled_mm_out_cuda
    layout=None,
):
    add_layout_constraint(aten._scaled_mm.default, constrain_to_fx_strides)
    m, n, k, layout, mat_a, mat_b = mm_args(
        mat_a, mat_b, layout=layout, out_dtype=out_dtype
    )

    # choices = (
    #     [aten__scaled_mm.bind((mat_a, mat_b), layout)] if use_aten_gemm_kernels() else []
    # )

    choices = []  # only using Triton kernels for now

    static_shape, is_nonzero = _is_static_problem([mat_a, mat_b], layout)
    if is_nonzero and use_triton_template(layout, enable_float8=True):
        for config in scaled_mm_configs(m, n, k):
            kwargs = scaled_mm_options(
                config, m, n, k, layout, bias, scale_a, scale_b, use_fast_accum
            )
            log.info(f"Siyu DEBUG scaled_mm_options output {kwargs}")

            if bias is None:
                bias_placeholder = empty((), dtype=torch.float32, device=mat_a.get_device())

            # possibly appends a TritonTemplateCaller to choices
            scaled_mm_w_bias_template.maybe_append_choice(
                choices,
                input_nodes=(mat_a, mat_b, scale_a, scale_b, bias_placeholder if bias is None else bias),
                layout=layout,
                **kwargs,
            )

    if bias is None:
        bias_placeholder = empty((), dtype=torch.float32, device=mat_a.get_device())
    input_nodes = [mat_a, mat_b, scale_a, scale_b, bias_placeholder if bias is None else bias]
    output = autotune_select_algorithm("scaled_mm_w_bias", choices, input_nodes, layout)

    output_amax_placeholder = empty((1), dtype=torch.float32, device=mat_a.get_device())
    return output, output_amax_placeholder
