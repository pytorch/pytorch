import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sympy

import torch
from torch._inductor.codegen.rocm.ck_universal_gemm_template import CKGemmTemplate
from torch.utils._triton import has_triton_tma_device

from .. import config as inductor_config
from ..codegen.common import WorkspaceArg, WorkspaceZeroMode
from ..config import triton as triton_config
from ..ir import _IntLike, ChoiceCaller, Layout, StorageBox, TensorBox
from ..lowering import add_layout_constraint, constrain_to_fx_strides, register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    NoValidChoicesError,
    realize_inputs,
    TritonTemplate,
)
from ..utils import use_aten_gemm_kernels, use_ck_gemm_template, use_triton_template
from .mm_common import (
    _is_static_problem,
    mm_args,
    mm_grid,
    persistent_grid,
    persistent_mm_configs,
    scaled_mm_configs,
)


_TMA_SIZE = 128
log = logging.getLogger(__name__)
aten = torch.ops.aten

load_scales = r"""
@triton.jit
def load_scales(a_scale_ptr, b_scale_ptr, SCALING_ROWWISE: tl.constexpr):
    if SCALING_ROWWISE:
        # For row-wise scaling, we'll return the pointers
        return a_scale_ptr, b_scale_ptr
    else:
        # For per-tensor scaling, we'll load the scalar values
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr)
        return a_scale, b_scale
"""


apply_scaling = r"""
@triton.jit
def apply_scaling(
    accumulator,
    a_scale,
    b_scale,
    SCALING_ROWWISE: tl.constexpr,
    offs_cm,
    offs_cn,
    M,
    N,
    stride_a_scale_m,
    stride_b_scale_n,
):
    if SCALING_ROWWISE:
        # For row-wise scaling, we need to load the scales for each row/column
        a_scales = tl.load(
            a_scale + (offs_cm * stride_a_scale_m),
            mask=offs_cm < M,
            other=0.0,
        )
        b_scales = tl.load(
            b_scale + (offs_cn * stride_b_scale_n),
            mask=offs_cn < N,
            other=0.0,
        )
        acc_scale = a_scales[:, None] * b_scales[None, :]
    else:
        # For per-tensor scaling, we can directly use the loaded scalar values
        acc_scale = a_scale * b_scale

    return accumulator * acc_scale
"""


device_tma = r"""
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

    if SCALING_ROWWISE:
        stride_a_scale_m = 1
        stride_b_scale_n = 1
    else:
        stride_a_scale_m = 0
        stride_b_scale_n = 0

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    workspace_base = ws_ptr + start_pid * 3 * TMA_SIZE
    a_desc_ptr = workspace_base
    b_desc_ptr = workspace_base + TMA_SIZE
    c_desc_ptr = workspace_base + 2 * TMA_SIZE

    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=a_desc_ptr,
        global_address=A,
        load_size=[BLOCK_M, BLOCK_K],
        global_size=[M, K],
        element_ty=A.dtype.element_ty,
    )
    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=b_desc_ptr,
        global_address=B,
        load_size=[BLOCK_N, BLOCK_K],
        global_size=[N, K],
        element_ty=B.dtype.element_ty,
    )

    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_M * num_pid_n
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    a_scale, b_scale = load_scales(A_inverse_scale, B_inverse_scale, SCALING_ROWWISE)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N

        offs_k = ki * BLOCK_K

        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K],  A.dtype.element_ty
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K],  B.dtype.element_ty
        )
        if USE_FAST_ACCUM:
            accumulator = tl.dot(a, b.T, accumulator)
        else:
            accumulator += tl.dot(a, b.T)

        if ki == k_tiles - 1:
            # Apply inverse scaling
            offs_cm = offs_am + tl.arange(0, BLOCK_M)
            offs_cn = offs_bn + tl.arange(0, BLOCK_N)
            # Apply scaling
            accumulator = apply_scaling(
                accumulator,
                a_scale,
                b_scale,
                SCALING_ROWWISE,
                offs_cm,
                offs_cn,
                M,
                N,
                stride_a_scale_m,
                stride_b_scale_n,
            )

            idx_m = offs_cm[:, None]
            idx_n = offs_cn[None, :]
            mask = (idx_m < M) & (idx_n < N)
            # inductor generates a suffix
            {{store_output(("idx_m", "idx_n"), "accumulator", "mask", indent_width=12)}}
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
"""


scaled_mm_device_tma_template = TritonTemplate(
    name="scaled_mm_device_tma",
    grid=persistent_grid,
    source=device_tma + load_scales + apply_scaling,
)


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


aten__fp8_mm = ExternKernelChoice(
    torch._scaled_mm, "at::_scaled_mm_out", op_overload=aten._scaled_mm.out
)


def are_compatible_scales(size_a: Sequence[int], size_b: Sequence[int]) -> bool:
    # Same sized scales are compatable
    if len(size_a) == len(size_b):
        return True

    # Both need to be scalars or len(1) tensors
    if len(size_a) <= 1 and len(size_b) <= 1:
        return True

    return False


def check_supported_striding(mat_a: TensorBox, mat_b: TensorBox) -> None:
    def is_row_major(stride: Sequence[_IntLike]) -> bool:
        return stride[1] == 1

    def is_col_major(stride: Sequence[_IntLike]) -> bool:
        return stride[0] == 1

    def has_zero_dim(size: Sequence[_IntLike]) -> bool:
        return bool(size[0] == 0 or size[1] == 0)

    # Check mat_a (self) stride requirements
    torch._check(
        is_row_major(mat_a.get_stride()) or has_zero_dim(mat_a.get_size()),
        lambda: f"mat_a must be row_major, got stride {mat_a.get_stride()}",
    )

    # Check mat_b stride requirements
    torch._check(
        is_col_major(mat_b.get_stride()) or has_zero_dim(mat_b.get_size()),
        lambda: f"mat_b must be col_major, got stride {mat_b.get_stride()}",
    )


def scaled_mm_options_device_tma(  # type: ignore[no-untyped-def]
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
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
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
        TMA_SIZE=_TMA_SIZE,
        NUM_SMS=NUM_SMS,
        **config.kwargs,
    )


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


def get_workspace_size(
    num_sms: int, TMA_SIZE: int = _TMA_SIZE, NUM_TMA_DESCRIPTORS: int = 3
) -> int:
    """Device side TMA requires a workspace buffer to be allocated in global memory."""
    return num_sms * NUM_TMA_DESCRIPTORS * TMA_SIZE


def get_workspace_arg(num_sms: int, device: torch.device) -> WorkspaceArg:
    """Builds and returns a WorkspaceArg for the device side TMA workspace buffer."""
    size = get_workspace_size(num_sms)
    zero_mode = WorkspaceZeroMode.from_bool(False)
    return WorkspaceArg(
        count=size,
        zero_mode=zero_mode,
        device=device,
        outer_name=WorkspaceArg.unique_name(),
    )


def use_persistent_tma(k: sympy.core.numbers.Integer, has_bias: bool) -> bool:
    available = has_triton_tma_device() and triton_config.enable_persistent_tma_matmul
    # _determine_swizzle_mode_2d requires BLOCK_K to be at least 32 contiguous bytes
    # When K is 16, BLOCK_K = 16 and is not valid
    min_k = k >= 32
    return available and min_k and not has_bias


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

    check_supported_striding(mat_a, mat_b)

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

    static_shape, is_nonzero = _is_static_problem(layout)

    if is_nonzero and use_triton_template(layout, enable_float8=True):
        if use_persistent_tma(k, bias is not None):
            for config in persistent_mm_configs(m, n, k):
                kwargs = scaled_mm_options_device_tma(
                    config, m, n, k, layout, scale_a, scale_b, use_fast_accum
                )
                input_nodes = (mat_a, mat_b, scale_a, scale_b)
                scaled_mm_device_tma_template.maybe_append_choice(
                    choices,
                    input_nodes=input_nodes,
                    layout=layout,
                    workspace_arg=get_workspace_arg(
                        kwargs["NUM_SMS"], mat_a.get_device()
                    ),
                    **kwargs,
                )
        else:
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

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, input_nodes)

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
