import logging
from typing import Any, Dict, List, Optional, Tuple

import sympy

import torch

from .. import config as inductor_config
from ..ir import ChoiceCaller, Layout, StorageBox, TensorBox
from ..lowering import add_layout_constraint, constrain_to_fx_strides, register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    NoValidChoicesError,
    realize_inputs,
    TritonTemplate,
)
from ..utils import use_aten_gemm_kernels, use_triton_template
from .mm import _is_static_problem  # TODO(yangsiyu) move to mm_common
from .mm_common import mm_args, mm_grid, scaled_mm_configs


log = logging.getLogger(__name__)
aten = torch.ops.aten


scaled_mm_template = TritonTemplate(
    name="scaled_mm",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B", "A_inverse_scale", "B_inverse_scale")}}
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
        if USE_FAST_ACCUM:
            acc = tl.dot(a, b, acc, out_dtype=ACC_TYPE)
        else:
            acc += tl.dot(a, b, out_dtype=ACC_TYPE)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

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

    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)


# Inductor does not allow optional tensor input arguments currently (pass None as an
# input node to template choices), but since for _scaled_mm there is only one such arg
# (bias), work around by having a second template when bias is provided.
scaled_mm_bias_template = TritonTemplate(
    name="scaled_mm_bias",
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
        if USE_FAST_ACCUM:
            acc = tl.dot(a, b, acc, out_dtype=ACC_TYPE)
        else:
            acc += tl.dot(a, b, out_dtype=ACC_TYPE)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

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

    # bias
    bias = tl.load(bias_ptr + rn, mask=rn < N)
    acc += bias

    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)


aten__fp8_mm = ExternKernelChoice(torch._scaled_mm, "at::_scaled_mm")


def are_compatible_scales(size_a: List[int], size_b: List[int]) -> bool:
    # Same sized scales are compatable
    if len(size_a) == len(size_b):
        return True

    # Both need to be scalars or len(1) tensors
    if len(size_a) <= 1 and len(size_b) <= 1:
        return True

    return False


def scaled_mm_options(  # type: ignore[no-untyped-def]
    config,  # triton.Config
    sym_m: sympy.core.numbers.Integer,
    sym_n: sympy.core.numbers.Integer,
    sym_k: sympy.core.numbers.Integer,
    layout: Layout,
    scale_a: StorageBox,
    scale_b: StorageBox,
    use_fast_accum: bool,
    b_prologue_cast_type: Optional[str] = None,
) -> Dict[str, Any]:
    even_k_symbolic = (
        sympy.gcd(sym_k, config.kwargs["BLOCK_K"]) == config.kwargs["BLOCK_K"]
    )

    size_a, size_b = scale_a.get_size(), scale_b.get_size()
    assert are_compatible_scales(size_a, size_b), (
        "Expect scale_a and scale_b to be either both scalars (including single-element tensors) "
        f"or 1-dimensional tensors with the same size. Got scale_a: {len(size_a)} and scale_b: {len(size_b)}."
    )
    return dict(
        GROUP_M=8,
        EVEN_K=even_k_symbolic,
        ACC_TYPE="tl.float32",
        B_PROLOGUE_CAST_TYPE=b_prologue_cast_type,
        USE_FAST_ACCUM=use_fast_accum,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        # tensor-wise scaling if scalar scales
        SCALING_ROWWISE=len(scale_a.get_size()) == 2,
        **config.kwargs,
    )


add_layout_constraint(aten._scaled_mm.default, constrain_to_fx_strides)


@register_lowering(aten._scaled_mm.default, type_promotion_kind=None)  # type: ignore[misc]
def tuned_scaled_mm(
    mat_a: TensorBox,
    mat_b: TensorBox,
    scale_a: TensorBox,
    scale_b: TensorBox,
    bias: Optional[TensorBox] = None,
    scale_result: Optional[TensorBox] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
    layout: Optional[Layout] = None,
) -> TensorBox:
    m, n, k, layout, mat_a, mat_b = mm_args(
        mat_a, mat_b, layout=layout, out_dtype=out_dtype
    )
    scale_a, scale_b = realize_inputs(scale_a, scale_b)

    input_nodes: Tuple[Any, ...]
    # workaround for Inductor not supporting optional tensor input arguments
    if bias is None:
        input_nodes = (mat_a, mat_b, scale_a, scale_b)
        triton_template = scaled_mm_template
    else:
        bias = realize_inputs(bias)
        input_nodes = (mat_a, mat_b, scale_a, scale_b, bias)
        triton_template = scaled_mm_bias_template

    aten_choice = aten__fp8_mm.bind(
        input_nodes, layout, out_dtype=out_dtype, use_fast_accum=use_fast_accum
    )

    choices: List[ChoiceCaller] = []
    if use_aten_gemm_kernels():
        choices.append(aten_choice)

    _, is_nonzero = _is_static_problem([mat_a, mat_b], layout)
    if is_nonzero and use_triton_template(layout, enable_float8=True):
        for config in scaled_mm_configs(m, n, k):
            if k == 16 and config.kwargs["BLOCK_M"] >= 64:
                continue  # Triton crashes in this case
            kwargs = scaled_mm_options(
                config, m, n, k, layout, scale_a, scale_b, use_fast_accum
            )
            # possibly appends a TritonTemplateCaller to choices
            triton_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=layout,
                **kwargs,
            )

    if (
        len(choices) == 0
        and not use_aten_gemm_kernels()
        and inductor_config.autotune_fallback_to_aten
    ):
        log.warning("No choices for scaled_mm, using ATen backend as fallback")
        return aten_choice.output_node()

    try:
        return autotune_select_algorithm("scaled_mm", choices, input_nodes, layout)
    except NoValidChoicesError:
        if not inductor_config.autotune_fallback_to_aten:
            raise
        log.warning(
            "All choices for scaled_mm were invalid, using ATen backend as fallback"
        )
        return aten_choice.output_node()
