# mypy: allow-untyped-defs
import functools
import logging
from typing import Any, Optional

import sympy

import torch
from torch._dynamo.utils import counters
from torch._inductor.autoheuristic.autoheuristic import AutoHeuristicSelectAlgorithm
from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    context_add_strides,
    context_add_using_tf32,
    mm_operations,
)
from torch._inductor.codegen.cpp_gemm_template import CppGemmTemplate
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.torch_version import TorchVersion

from .. import config as inductor_config, ir
from ..codegen.cuda.gemm_template import CUTLASS2xGemmTemplate, CUTLASS3xGemmTemplate
from ..codegen.rocm.ck_tile_universal_gemm_template import CKTileGemmTemplate
from ..codegen.rocm.ck_universal_gemm_template import CKGemmTemplate
from ..codegen.subgraph import SubgraphTemplate
from ..ir import FlexibleLayout, is_triton
from ..lowering import (
    add_layout_constraint,
    constrain_to_fx_strides,
    lowerings as L,
    register_lowering,
)
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
    TritonTemplate,
)
from ..utils import (
    _use_cutlass_for_op,
    get_k_splits,
    get_tma_workspace_arg,
    use_aten_gemm_kernels,
    use_ck_gemm_template,
    use_ck_tile_gemm_template,
    use_cpp_gemm_template,
    use_cutlass_template,
    use_decompose_k_choice,
    use_triton_template,
    use_triton_tma_template,
)
from .mm_common import (
    _is_static_problem,
    addmm_epilogue,
    mm_args,
    mm_config_kwargs,
    mm_grid,
    mm_options,
    persistent_mm_grid,
    persistent_mm_options,
    scale_mm_epilogue,
    scaled_mm_options,
)


try:
    import triton

    triton_version = TorchVersion(triton.__version__)
    has_triton = True
except ImportError:
    triton_version = TorchVersion("0.0.0")
    has_triton = False

log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

mm_template = TritonTemplate(
    name="mm",
    grid=mm_grid,
    source=(
        r"""
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
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and M >= BLOCK_M:
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and N >= BLOCK_N:
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        {% if not EVEN_K %}
        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
        {% endif %}
        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        {{load_input("A", "a", ("idx_m", "idx_n"), mask=None if EVEN_K else "a_mask", indent_width=8)}}

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        {{load_input("B", "b", ("idx_m", "idx_n"), mask=None if EVEN_K else "b_mask", indent_width=8)}}

        {% if USE_FAST_ACCUM %}
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        {% else %}
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        {% endif %}

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
"""
        if (torch.version.hip is None) or triton_version >= "3.3.0"
        # FIXME: To get around rocm failures like https://github.com/pytorch/pytorch/actions/runs/13123783322/job/36617154943
        # The only difference between the two templates is M >= BLOCK_M and N >= BLOCK_N checking.
        # See more details in https://github.com/pytorch/pytorch/pull/146293
        else r"""
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
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        {% if not EVEN_K %}
        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
        {% endif %}
        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        {{load_input("A", "a", ("idx_m", "idx_n"), mask=None if EVEN_K else "a_mask", indent_width=8)}}

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        {{load_input("B", "b", ("idx_m", "idx_n"), mask=None if EVEN_K else "b_mask", indent_width=8)}}
        {% if USE_FAST_ACCUM %}
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        {% else %}
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
        {% endif %}

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
"""
    ),
    cache_codegen_enabled_for_template=True,
    prologue_loads_all_inputs=True,
)

persistent_tma_mm_template = TritonTemplate(
    name="mm_persistent_tma",
    grid=persistent_mm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = grid_m * grid_n
    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    width = GROUP_M * grid_n
    rk_for_mask = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    workspace_base = ws_ptr + start_pid * 2 * TMA_SIZE
    a_desc_ptr = workspace_base
    b_desc_ptr = workspace_base + TMA_SIZE

    {%- if TMA_EXPERIMENTAL_API %}
    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=a_desc_ptr,
        global_address=A,
        load_size=[BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],
        global_size=[M, K] if A_ROW_MAJOR else [K, M],
        element_ty=A.dtype.element_ty,
    )
    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=b_desc_ptr,
        global_address=B,
        load_size=[BLOCK_K, BLOCK_N] if B_ROW_MAJOR else [BLOCK_N, BLOCK_K],
        global_size=[K, N] if B_ROW_MAJOR else [N, K],
        element_ty=B.dtype.element_ty,
    )

    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)

    a_desc = a_desc_ptr
    b_desc = b_desc_ptr
    {%- else %}
    a_desc = triton.language.make_tensor_descriptor(
        base=A,
        shape=[M, K] if A_ROW_MAJOR else [K, M],
        strides=[K, 1] if A_ROW_MAJOR else [M, 1],
        block_shape=[BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],
    )
    b_desc = triton.language.make_tensor_descriptor(
        base=B,
        shape=[K, N] if B_ROW_MAJOR else [N, K],
        strides=[N, 1] if B_ROW_MAJOR else [K, 1],
        block_shape=[BLOCK_K, BLOCK_N] if B_ROW_MAJOR else [BLOCK_N, BLOCK_K],
    )
    {%- endif %}

    pid_m = 0
    pid_n = 0
    rm = 0
    rn = 0

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            # re-order program ID for better L2 performance
            group_id = tile_id // width
            group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
            pid_m = group_id * GROUP_M + (tile_id % group_size)
            pid_n = (tile_id % width) // (group_size)

            rm = pid_m * BLOCK_M
            rn = pid_n * BLOCK_N

        rk = ki * BLOCK_K

        {%- if TMA_EXPERIMENTAL_API %}
        a = tl._experimental_descriptor_load(
            a_desc,
            [rm, rk] if A_ROW_MAJOR else [rk, rm],
            [BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],
            A.dtype.element_ty,
        )
        b = tl._experimental_descriptor_load(
            b_desc,
            [rk, rn] if B_ROW_MAJOR else [rn, rk],
            [BLOCK_K, BLOCK_N] if B_ROW_MAJOR else [BLOCK_N, BLOCK_K],
            B.dtype.element_ty,
        )
        {%- else %}
        a = tl.load_tensor_descriptor(
            a_desc,
            [rm, rk] if A_ROW_MAJOR else [rk, rm],
        )
        b = tl.load_tensor_descriptor(
            b_desc,
            [rk, rn] if B_ROW_MAJOR else [rn, rk],
        )
        {%- endif %}
        acc += tl.dot(
            a if A_ROW_MAJOR else a.T,
            b if B_ROW_MAJOR else b.T,
            allow_tf32=ALLOW_TF32,
        )

        if ki == k_tiles - 1:
            # rematerialize rm and rn to save registers
            rcm = rm + tl.arange(0, BLOCK_M)
            rcn = rn + tl.arange(0, BLOCK_N)
            idx_m = rcm[:, None]
            idx_n = rcn[None, :]
            mask = (idx_m < M) & (idx_n < N)

            # inductor generates a suffix
            {{store_output(("idx_m", "idx_n"), "acc", "mask", indent_width=12)}}
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

""",
)

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

    workspace_base = ws_ptr + start_pid * 2 * TMA_SIZE
    a_desc_ptr = workspace_base
    b_desc_ptr = workspace_base + TMA_SIZE

    {%- if TMA_EXPERIMENTAL_API %}
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

    a_desc = a_desc_ptr
    b_desc = a_desc_ptr
    {%- else %}
    a_desc = triton.language.make_tensor_descriptor(
        base=A,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = triton.language.make_tensor_descriptor(
        base=B,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    {%- endif %}

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

        {%- if TMA_EXPERIMENTAL_API %}
        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K],  A.dtype.element_ty
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K],  B.dtype.element_ty
        )
        {%- else %}
        a = tl.load_tensor_descriptor(a_desc, [offs_am, offs_k])
        b = tl.load_tensor_descriptor(b_desc, [offs_bn, offs_k])
        {%- endif %}
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
    grid=persistent_mm_grid,
    source=device_tma + load_scales + apply_scaling,
)


# prevent duplication registration of extern functions
@functools.cache
def lazy_register_extern_choice(fn):
    return ExternKernelChoice(fn)


aten_mm = ExternKernelChoice(torch.mm, "at::mm_out")

aten_addmm = ExternKernelChoice(
    torch.addmm, "at::addmm_out", op_overload=aten.addmm.default
)

aten__int_mm = ExternKernelChoice(torch._int_mm, "at::_int_mm_out")

aten__sparse_semi_structured_mm = ExternKernelChoice(
    torch._sparse_semi_structured_mm,
    "at::_sparse_semi_structured_mm",
    has_out_variant=False,
)

aten__fp8_mm = ExternKernelChoice(
    torch._scaled_mm, "at::_scaled_mm_out", op_overload=aten._scaled_mm.out
)


def _is_int8_mat(mat):
    return mat.get_dtype() in (torch.int8, torch.uint8)


def _is_large_block_for_cpu(m, n, k):
    # Thresholds are experimentally determined to reduce Triton CPU compile times
    return m * n > 2**13


@functools.lru_cache
def using_b200() -> bool:
    """Returns true if the device is a NVIDIA B200, otherwise returns false."""
    if not torch.cuda.is_available():
        return False
    # compute capability 10.0 or 10.0a is NVIDIA B200
    device_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return device_properties.major == 10


def bias_addmm(inp, mat1, mat2, *, out=None, alpha=1, beta=1):
    """
    Giving torch.addmm a 1D tensor calls a different (faster) cublasLt
    kernel under the hood.  There are a few shapes where this is slower,
    but they are rare.
    """
    if inp.stride(0) == 0 or inp.size(0) == 1:
        return torch.addmm(inp[0], mat1, mat2, out=out, alpha=alpha, beta=beta)
    return torch.addmm(inp, mat1, mat2, out=out, alpha=alpha, beta=beta)


def check_supported_striding(mat_a, mat_b) -> None:
    def is_row_major(stride) -> bool:
        return V.graph.sizevars.statically_known_equals(stride[1], 1)

    def is_col_major(stride) -> bool:
        return V.graph.sizevars.statically_known_equals(stride[0], 1)

    def has_zero_dim(size) -> bool:
        return bool(
            V.graph.sizevars.statically_known_equals(size[0], 0)
            or V.graph.sizevars.statically_known_equals(size[1], 0)
        )

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


aten_bias_addmm = ExternKernelChoice(bias_addmm, None)


def decomposeK(a, b, k_splits):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    k_parts = k // k_splits
    B = k_splits
    a_reshaped = torch.permute(a.reshape(m, B, k_parts), (1, 0, 2))
    b_reshaped = b.reshape(B, k_parts, n)
    result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    reduced_buf = torch.sum(result, 0)
    return reduced_buf.to(a.dtype)


@register_lowering(aten.mm, type_promotion_kind=None)
def tuned_mm(mat1, mat2, *, layout=None):
    """
    Lowering for autotuning aten.mm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    device_type = ir.get_device_type(mat1)
    name = "mm"

    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten.mm_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten.mm: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )

    aten_layout = layout
    if not (inductor_config.max_autotune or inductor_config.max_autotune_gemm):
        aten_layout = FlexibleLayout(
            device=layout.device, dtype=layout.dtype, size=layout.size
        )

    # options to tune from
    choices = (
        [aten_mm.bind((mat1, mat2), aten_layout)] if use_aten_gemm_kernels() else []
    )
    static_shape, is_nonzero = _is_static_problem(layout)

    mm_configs = V.choices.get_base_mm_configs(device_type)
    persistent_mm_configs = V.choices.get_persistent_mm_configs(device_type)
    extra_mm_configs = V.choices.get_extra_mm_configs(device_type)

    dtype = mat1.get_dtype()
    if is_nonzero and use_triton_template(layout):
        for config in mm_configs(
            m,
            n,
            k,
            **mm_config_kwargs(device_type, _is_large_block_for_cpu, dtype.itemsize),
        ):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
            )

        if use_triton_tma_template(mat1, mat2):
            for config in persistent_mm_configs(
                m,
                n,
                k,
                **mm_config_kwargs(
                    device_type, _is_large_block_for_cpu, dtype.itemsize
                ),
            ):
                persistent_tma_mm_template.maybe_append_choice(
                    choices,
                    input_nodes=(mat1, mat2),
                    layout=layout,
                    workspace_arg=get_tma_workspace_arg(
                        num_tma_descriptors=2,
                        device=mat1.get_device(),
                    ),
                    **mm_options(config, m, n, k, layout),
                    **persistent_mm_options(mat1, mat2),
                )

        from torch._inductor.ir import get_free_symbols

        # Only do split-k optimization if K is much larger than m, n and m, n are small
        # and if there aren't any unbacked symbols
        unbacked_symbols = any(
            len(get_free_symbols(itr, unbacked_only=True)) > 0
            for itr in (
                mat1.get_size(),
                mat1.get_stride(),
                mat2.get_size(),
                mat2.get_stride(),
            )
        )
        if use_decompose_k_choice(m, n, k) and not unbacked_symbols:
            from torch._dispatch.python import enable_python_dispatcher

            from ..decomposition import select_decomp_table

            k_splits = get_k_splits(m, n, k)
            for k_split in k_splits:
                if not V.graph.sizevars.statically_known_true(
                    sympy.Eq(sympy.Mod(k, k_split), 0)
                ):
                    continue

                with enable_python_dispatcher():
                    decompositions = select_decomp_table()

                    decompose_k_subgraph_template = SubgraphTemplate(
                        name=f"decompose_k_mm_{k_split}_split",
                        make_fx_graph=make_fx(
                            functools.partial(decomposeK, k_splits=k_split),
                            decompositions,
                        ),
                    )

                decompose_k_subgraph_template.maybe_append_choice(
                    choices,
                    input_nodes=(mat1, mat2),
                    layout=layout,
                )

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op("mm")
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(choices, layout, [mat1, mat2])

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, [mat1, mat2])
    if is_nonzero and use_ck_tile_gemm_template(layout, m, n, k):
        CKTileGemmTemplate.add_choices(choices, layout, [mat1, mat2])

    if use_cpp_gemm_template(layout, mat1, mat2):
        CppGemmTemplate.add_choices(
            choices,
            layout,
            [mat1, mat2],
        )

    input_nodes = [mat1, mat2]
    if (
        is_nonzero
        and use_triton_template(layout)
        and torch._inductor.config.run_autoheuristic(name)
        and is_triton(mat1)
    ):
        always_included = []
        if use_aten_gemm_kernels():
            always_included.append("extern_mm")
        num_choices_before_extra_configs = len(choices)
        for config in extra_mm_configs(
            m, n, k, **mm_config_kwargs(device_type, _is_large_block_for_cpu)
        ):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
            )

        # using AutoHeuristic for ranking
        ah_choices = mm_autoheuristic(
            mat1,
            mat2,
            m,
            n,
            k,
            choices,
            name,
            input_nodes,
            mm_operations(),
            None,
            top_k=10,
            always_included=always_included,
        )
        if not torch._inductor.config.collect_autoheuristic(name):
            # if we are collecting data, we do not want to modify choices
            if ah_choices is not None and len(ah_choices) > 0:
                # the order in which autoheuristic returns choices is not the same as
                # as the order of choices, which affects things like epilogue fusion.
                # once epilogue fusion benchmarks choices in sorted order, I think we can
                # just use the order returned by autoheuristic
                choices = [choice for choice in choices if choice in ah_choices]
            else:
                choices = choices[:num_choices_before_extra_configs]

    for k in inductor_config.external_matmul:
        choices.append(lazy_register_extern_choice(k).bind((mat1, mat2), layout))

    return autotune_select_algorithm(name, choices, [mat1, mat2], layout)


@register_lowering(aten._int_mm, type_promotion_kind=None)
def tuned_int_mm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=torch.int32
    )

    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten._int_mm_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten._int_mm: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )

    device_type = ir.get_device_type(mat1)

    static_shape, is_nonzero = _is_static_problem(layout)
    use_cutlass = static_shape and is_nonzero and use_cutlass_template(layout, m, n, k)

    choices = (
        [aten__int_mm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []
    )

    if use_cutlass and _use_cutlass_for_op("int_mm"):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, [mat1, mat2], fuseable=True, non_fuseable=True
        )

    int8_mm_configs = V.choices.get_int8_mm_configs(device_type)

    if is_nonzero and use_triton_template(layout, enable_int32=True):
        for config in int8_mm_configs(
            m, n, k, **mm_config_kwargs(device_type, _is_large_block_for_cpu)
        ):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
            )

    return autotune_select_algorithm("int_mm", choices, [mat1, mat2], layout)


@register_lowering(aten.addmm, type_promotion_kind=None)
def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    device_type = ir.get_device_type(mat1)
    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    static_shape, is_nonzero = _is_static_problem(layout)

    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten.addmm_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten.addmm: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )

    if (not is_nonzero) or (
        not (inductor_config.max_autotune or inductor_config.max_autotune_gemm)
    ):
        # Use a FlexibleLayout if we are not autotuning.
        # This allows padding strides for the output.
        from torch._inductor.ir import FixedLayout, FlexibleLayout

        if isinstance(layout, FixedLayout):
            layout = FlexibleLayout(
                device=layout.device, dtype=layout.dtype, size=layout.size
            )
        choices = (
            [
                aten_addmm.bind(
                    (inp, mat1, mat2),
                    layout,
                    alpha=alpha,
                    beta=beta,
                )
            ]
            if use_aten_gemm_kernels()
            else []
        )
        return autotune_select_algorithm("addmm", choices, [inp, mat1, mat2], layout)

    choices = (
        [
            aten_addmm.bind(
                (inp_expanded, mat1, mat2),
                layout,
                alpha=alpha,
                beta=beta,
            )
        ]
        if use_aten_gemm_kernels()
        else []
    )

    if (
        use_aten_gemm_kernels()
        and inp_expanded.get_stride()[0] == 0
        and inp_expanded.get_device().type == "cuda"
        and inductor_config.triton.autotune_cublasLt
    ):
        # unexpand inp to make sure fused addmm from cublasLt is used
        choices.insert(
            0,
            aten_bias_addmm.bind(
                (inp_expanded, mat1, mat2), layout, alpha=alpha, beta=beta
            ),
        )

    mm_configs = V.choices.get_base_mm_configs(device_type)
    persistent_mm_configs = V.choices.get_persistent_mm_configs(device_type)

    dtype = mat1.get_dtype()
    if is_nonzero and use_triton_template(layout):
        for config in mm_configs(
            m,
            n,
            k,
            **mm_config_kwargs(device_type, _is_large_block_for_cpu, dtype.itemsize),
        ):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(inp_expanded, mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
                prefix_args=1,
                epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta),
                epilogue_fn_hash=str(["addmm_epilogue", layout.dtype, alpha, beta]),
            )

        if use_triton_tma_template(mat1, mat2):
            for config in persistent_mm_configs(
                m,
                n,
                k,
                **mm_config_kwargs(
                    device_type, _is_large_block_for_cpu, dtype.itemsize
                ),
            ):
                persistent_tma_mm_template.maybe_append_choice(
                    choices,
                    input_nodes=(inp_expanded, mat1, mat2),
                    layout=layout,
                    workspace_arg=get_tma_workspace_arg(
                        num_tma_descriptors=2,
                        device=mat1.get_device(),
                    ),
                    **mm_options(config, m, n, k, layout),
                    **persistent_mm_options(mat1, mat2),
                    prefix_args=1,
                    epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta),
                )

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op("addmm")
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            [mat1, mat2, inp_expanded],
            alpha=alpha,
            beta=beta,
            input_reorder=[2, 0, 1],
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(
            choices,
            layout,
            [mat1, mat2, inp_expanded],
            alpha=alpha,
            beta=beta,
            input_reorder=[2, 0, 1],
        )

    if use_cpp_gemm_template(layout, mat1, mat2):
        CppGemmTemplate.add_choices(
            choices,
            layout,
            [inp_expanded, mat1, mat2],
            alpha=alpha,
            beta=beta,
            has_bias=True,
        )

    return autotune_select_algorithm(
        "addmm", choices, [inp_expanded, mat1, mat2], layout
    )


@register_lowering(aten._sparse_semi_structured_mm, type_promotion_kind=None)
def tuned_sparse_semi_structured_mm(
    mat1, mat1_meta, mat2, *, out_dtype=None, layout=None
):
    from torch._inductor.select_algorithm import realize_inputs

    mat1, mat1_meta, mat2 = realize_inputs(mat1, mat1_meta, mat2)
    m1, k1 = mat1.get_size()
    m2, _ = mat1_meta.get_size()
    k2, n = mat2.get_size()
    m = V.graph.sizevars.check_equals_and_simplify(m1, m2)
    k = V.graph.sizevars.check_equals_and_simplify(2 * k1, k2)

    if layout is None:
        from torch._inductor.ir import FixedLayout

        layout = FixedLayout(
            mat2.get_device(),
            out_dtype if out_dtype else mat2.get_dtype(),
            [m, n],
            [n, 1],
        )
    else:
        assert out_dtype is None, "out_dtype is ignored if layout is specified."

    choices = (
        [
            aten__sparse_semi_structured_mm.bind(
                (mat1, mat1_meta, mat2), layout, out_dtype=out_dtype
            )
        ]
        if use_aten_gemm_kernels()
        else []
    )

    if (
        m * n != 0
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op("sparse_semi_structured_mm")
    ):
        CUTLASS2xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, [mat1, mat2, mat1_meta], fuseable=True, non_fuseable=True
        )

    return autotune_select_algorithm(
        "sparse_semi_structured_mm", choices, [mat1, mat1_meta, mat2], layout
    )


add_layout_constraint(aten._scaled_mm.default, constrain_to_fx_strides)


@register_lowering(aten._scaled_mm.default, type_promotion_kind=None)  # type: ignore[misc]
def tuned_scaled_mm(
    mat_a,
    mat_b,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    layout=None,
):
    """
    Performs an optimized matrix multiplication where scaling factors are applied
    to the inputs and/or output.

    Args:
        mat1 (Tensor): First input matrix
        mat2 (Tensor): Second input matrix
        scale1 (Tensor): Scale factor applied to mat1 (supports broadcasting)
        scale2 (Tensor): Scale factor applied to mat2 (supports broadcasting)
        bias (Tensor, optional): Optional bias tensor to add to the result
        layout: Layout hint for optimization

    Returns:
        Tensor: The result of the scaled matrix multiplication
    """
    m, n, k, layout, mat_a, mat_b = mm_args(
        mat_a, mat_b, layout=layout, out_dtype=out_dtype
    )
    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten._scaled_mm.default_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten._scaled_mm.default: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat_a.get_dtype(),
        mat_b.get_dtype(),
        layout,
    )

    device_type = ir.get_device_type(mat_a)
    check_supported_striding(mat_a, mat_b)

    scale_a_real, scale_b_real = realize_inputs(scale_a, scale_b)

    input_nodes: tuple[Any, ...]

    if not bias:
        input_nodes = (mat_a, mat_b, scale_a_real, scale_b_real)
    else:
        bias_real = realize_inputs(bias)
        input_nodes = (mat_a, mat_b, scale_a_real, scale_b_real, bias_real)

    aten_choice = aten__fp8_mm.bind(
        input_nodes, layout, out_dtype=out_dtype, use_fast_accum=use_fast_accum
    )

    choices = []
    if use_aten_gemm_kernels():
        choices.append(aten_choice)

    # We dont have triton lowerings for the MX variants yet
    if scale_a.dtype != torch.float32:
        return autotune_select_algorithm("scaled_mm", choices, input_nodes, layout)

    _, is_nonzero = _is_static_problem(layout)

    scaled_mm_configs = V.choices.get_scaled_mm_configs(device_type)
    scaled_persistent_mm_configs = V.choices.get_scaled_persistent_mm_configs(
        device_type
    )

    if is_nonzero and use_triton_template(layout, enable_float8=True):
        triton_input_nodes: tuple[Any, ...]
        if bias and len(mat_b.get_size()) == len(bias.get_size()) + 1:
            # Need to unsqueeze bias from [N] -> [1, N]
            triton_bias = L[aten.unsqueeze](bias, 0)
        else:
            triton_bias = bias

        if len(scale_a.get_size()) == 0 or len(scale_b.get_size()) == 0:
            assert len(scale_a.get_size()) == len(scale_b.get_size())
            # Need to unsqueeze scale from [] -> [1, 1]
            triton_scale_a = L[aten.unsqueeze](L[aten.unsqueeze](scale_a, 0), 1)
            triton_scale_b = L[aten.unsqueeze](L[aten.unsqueeze](scale_b, 0), 1)
        else:
            triton_scale_a = scale_a
            triton_scale_b = scale_b

        if bias:
            triton_input_nodes = (
                mat_a,
                mat_b,
                triton_scale_a,
                triton_scale_b,
                triton_bias,
            )
            suffix_args = 3
        else:
            triton_input_nodes = (mat_a, mat_b, triton_scale_a, triton_scale_b)
            suffix_args = 2

        # TODO (paulzhan): There is no template that exists for bias and TMA
        # Don't run tma template currently if bias exists
        if use_triton_tma_template(mat_a, mat_b) and not bias:
            for config in scaled_persistent_mm_configs(m, n, k):
                kwargs = scaled_mm_options(
                    config,
                    m,
                    n,
                    k,
                    layout,
                    scale_a,
                    scale_b,
                    use_fast_accum,
                    device_tma=True,
                )
                scaled_mm_device_tma_template.maybe_append_choice(
                    choices,
                    input_nodes=triton_input_nodes,
                    layout=layout,
                    workspace_arg=get_tma_workspace_arg(
                        num_tma_descriptors=2,
                        device=mat_a.get_device(),
                    ),
                    **kwargs,
                )

        for config in scaled_mm_configs(m, n, k):
            if V.graph.sizevars.guard_or_false(sympy.Le(k, 16)):
                # Triton crashes however uncommon for real workloads
                continue

            # On NVIDIA B200 GPUs, K dim must be >= 32 for tcgen05.mma.kind::f8f6f4.* PTX instruction to be valid
            # source: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape
            if using_b200() and V.graph.sizevars.guard_or_false(sympy.Lt(k, 32)):
                continue

            kwargs = scaled_mm_options(
                config, m, n, k, layout, scale_a, scale_b, use_fast_accum
            )
            # possibly appends a TritonTemplateCaller to choices
            mm_template.maybe_append_choice(
                choices,
                input_nodes=triton_input_nodes,
                layout=layout,
                **kwargs,
                suffix_args=suffix_args,
                epilogue_fn=scale_mm_epilogue(),
                epilogue_fn_hash="scale_mm_epilogue",
            )

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op("scaled_mm")
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            input_nodes,  # type: ignore[arg-type]
            use_fast_accum=use_fast_accum,  # type: ignore[arg-type]
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, input_nodes)

    return autotune_select_algorithm("scaled_mm", choices, input_nodes, layout)


@functools.cache
def _is_sm7x_or_older_gpu(index: Optional[int]) -> bool:
    props = torch.cuda.get_device_properties(index or 0)
    return props.major <= 7


def dims_are_int(dims):
    return all(isinstance(dim, int) for dim in dims)


def mm_autoheuristic(
    mat1,
    mat2,
    m,
    n,
    k,
    choices,
    name,
    input_nodes,
    ops,
    precondition,
    top_k: Optional[int] = None,
    always_included=None,
):
    m, n, k = get_size_hints(mat1, mat2, m, n, k)
    if not dims_are_int([m, n, k]):
        return None
    mat1_stride, mat2_stride = get_size_hints_strides(mat1, mat2)

    def get_context(m, k, n, mat1, mat2, mat1_stride, mat2_stride):
        context = AHContext()
        context.add_feature("m", m)
        context.add_feature("k", k)
        context.add_feature("n", n)
        context.add_feature("mat1_dtype", mat1.layout.dtype, is_categorical=True)
        context.add_feature("mat2_dtype", mat2.layout.dtype, is_categorical=True)
        context_add_strides(context, "mat1", mat1_stride)
        context_add_strides(context, "mat2", mat2_stride)
        context.add_feature(
            "mat1_iscontig", mat1.layout.is_contiguous(), is_categorical=True
        )
        context.add_feature(
            "mat2_iscontig", mat2.layout.is_contiguous(), is_categorical=True
        )
        if name == "mm":
            context_add_using_tf32(context, mat1.layout.dtype)
        return context

    def fallback():
        return None

    context = get_context(m, k, n, mat1, mat2, mat1_stride, mat2_stride)
    autoheuristic = AutoHeuristicSelectAlgorithm(
        fallback=fallback,
        choices=choices,
        input_nodes=input_nodes,
        context=context,
        name=name,
        augment_context=ops,
        precondition=precondition,
    )

    if top_k is not None:
        # TODO: is there a cleaner way to ensure aten.mm is always included?
        return autoheuristic.get_top_k_choices_caller(
            top_k, always_included=always_included
        )

    return autoheuristic.get_choice_caller()


def get_size_hints(mat1, mat2, m, n, k):
    if not isinstance(m, int) or not isinstance(k, int):
        (m, k) = V.graph.sizevars.size_hints(
            mat1.get_size(),
            fallback=torch._inductor.config.unbacked_symint_fallback,
        )

    if not isinstance(n, int) or not isinstance(k, int):
        (k, n) = V.graph.sizevars.size_hints(
            mat2.get_size(),
            fallback=torch._inductor.config.unbacked_symint_fallback,
        )
    return m, n, k


def get_size_hints_strides(mat1, mat2):
    mat1_stride = mat1.layout.stride
    mat2_stride = mat2.layout.stride
    strides = [mat1_stride, mat2_stride]
    strides_hints = []
    for stride in strides:
        if not isinstance(stride, int):
            stride = V.graph.sizevars.size_hints(
                stride,
                fallback=torch._inductor.config.unbacked_symint_fallback,
            )
        strides_hints.append(stride)
    return strides_hints[0], strides_hints[1]
