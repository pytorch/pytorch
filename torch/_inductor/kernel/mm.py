# mypy: allow-untyped-defs
import functools
import logging
from typing import Any, Optional, Union

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
from torch._inductor.remote_gemm_autotune_cache import gen_best_config
from torch._inductor.virtualized import ops, V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.functional import ScalingType  # type: ignore[attr-defined]
from torch.torch_version import TorchVersion

from .. import config as inductor_config, distributed_autotune
from ..codegen.cuda.gemm_template import CUTLASS2xGemmTemplate, CUTLASS3xGemmTemplate
from ..codegen.rocm.ck_tile_universal_gemm_template import CKTileGemmTemplate
from ..codegen.rocm.ck_universal_gemm_template import CKGemmTemplate
from ..codegen.subgraph import SubgraphChoiceCaller, SubgraphTemplate
from ..ir import Buffer, ChoiceCaller, is_triton, Layout
from ..kernel_inputs import MMKernelInputs
from ..lowering import (
    lowerings,
    make_pointwise,
    make_reduction,
    register_lowering,
    transform_args,
)
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    KernelTemplate,
    realize_inputs,
    TritonTemplate,
)
from ..utils import (
    _use_cutlass_for_op,
    ceildiv,
    use_aten_gemm_kernels,
    use_ck_gemm_template,
    use_ck_tile_gemm_template,
    use_cpp_gemm_template,
    use_cutlass_template,
    use_decompose_k_choice,
    use_triton_blackwell_tma_template,
    use_triton_template,
    use_triton_tma_template,
)
from .mm_common import (
    _is_static_problem,
    mm_args,
    mm_grid,
    persistent_mm_grid,
    use_native_matmul,
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
    pid = tl.program_id(0).to(INDEX_DTYPE)
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
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
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
        {{load_input("A", "a", ("idx_m", "idx_n"), mask=None if EVEN_K else "a_mask",
                     indent_width=8, index_shape=("BLOCK_M", "BLOCK_K"))}}

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        {{load_input("B", "b", ("idx_m", "idx_n"), mask=None if EVEN_K else "b_mask",
                     indent_width=8, index_shape=("BLOCK_K", "BLOCK_N"))}}

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
    {{store_output(("idx_m", "idx_n"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
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
    pid = tl.program_id(0).to(INDEX_DTYPE)
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
        {{load_input("A", "a", ("idx_m", "idx_n"), mask=None if EVEN_K else "a_mask",
                     indent_width=8, index_shape=("BLOCK_M", "BLOCK_K"))}}

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        {{load_input("B", "b", ("idx_m", "idx_n"), mask=None if EVEN_K else "b_mask",
                     indent_width=8, index_shape=("BLOCK_K", "BLOCK_N"))}}
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
    {{store_output(("idx_m", "idx_n"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
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

    start_pid = tl.program_id(0).to(INDEX_DTYPE)
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

    {%- if TMA_EXPERIMENTAL_API %}
    workspace_base = ws_ptr + start_pid * 2 * TMA_SIZE
    a_desc_ptr = workspace_base
    b_desc_ptr = workspace_base + TMA_SIZE

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

    {%- else %}
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
    a_desc = triton.language.make_tensor_descriptor(
        base=A,
        shape=[M, K] if A_ROW_MAJOR else [K, M],
        strides=[stride_am, 1] if A_ROW_MAJOR else [stride_ak, 1],
        block_shape=[BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],
    )
    b_desc = triton.language.make_tensor_descriptor(
        base=B,
        shape=[K, N] if B_ROW_MAJOR else [N, K],
        strides=[stride_bk, 1] if B_ROW_MAJOR else [stride_bn, 1],
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
            a_desc_ptr,
            [rm, rk] if A_ROW_MAJOR else [rk, rm],
            [BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],
            A.dtype.element_ty,
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr,
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
            # inductor generates a suffix
            {%- if TMA_EXPERIMENTAL_API %}
            # rematerialize rm and rn to save registers
            rcm = rm + tl.arange(0, BLOCK_M)
            rcn = rn + tl.arange(0, BLOCK_N)
            idx_m = rcm[:, None]
            idx_n = rcn[None, :]
            mask = (idx_m < M) & (idx_n < N)
            {{store_output(("idx_m", "idx_n"), "acc", "mask", indent_width=12, val_shape=("BLOCK_M", "BLOCK_N"))}}
            {%- else %}
            {{store_output(("rm", "rn"), "acc", indent_width=12, val_shape=("BLOCK_M", "BLOCK_N"), block_indexing=True)}}
            {%- endif %}
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

""",
)

load_scales = r"""
@triton.jit
def load_scales(scale_ptr, SCALE_RECIPE: tl.constexpr):
    if SCALE_RECIPE == 0:
        return tl.load(scale_ptr)  # For tensor-wise scaling, we'll load the scalar values
    else:
        return scale_ptr  # For all other scaling recipes, we'll return the pointers
"""


apply_scaling = r"""
@triton.jit
def apply_scaling(
    accumulator,
    a_scale,
    b_scale,
    SCALE_RECIPE_A: tl.constexpr,
    SCALE_RECIPE_B: tl.constexpr,
    offs_cm,
    offs_cn,
    M,
    N,
    stride_a_scale_m,
    stride_b_scale_n,
):
    if SCALE_RECIPE_A == 1 and SCALE_RECIPE_B == 1:  # (ScalingType.RowWise, ScalingType.RowWise)
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
    else:  # (ScalingType.TensorWise, ScalingType.TensorWise)
        # For per-tensor scaling, we can directly use the loaded scalar values
        acc_scale = a_scale * b_scale

    return accumulator * acc_scale
"""


scaled_mm_device_tma_epilogue_scaling = r"""
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

    if SCALE_RECIPE_A == 1:  # ScalingType.RowWise
        stride_a_scale_m = 1
    else:
        stride_a_scale_m = 0

    if SCALE_RECIPE_B == 1:  # ScalingType.RowWise
        stride_b_scale_n = 1
    else:
        stride_b_scale_n = 0

    start_pid = tl.program_id(axis=0).to(INDEX_DTYPE)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    {%- if TMA_EXPERIMENTAL_API %}
    workspace_base = ws_ptr + start_pid * 2 * TMA_SIZE
    a_desc_ptr = workspace_base
    b_desc_ptr = workspace_base + TMA_SIZE

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

    {%- else %}
    stride_am = {{stride("A", 0)}}
    stride_bn = {{stride("B", 1)}}
    a_desc = triton.language.make_tensor_descriptor(
        base=A,
        shape=[M, K],
        strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = triton.language.make_tensor_descriptor(
        base=B,
        shape=[N, K],
        strides=[stride_bn, 1],
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
    a_scale = load_scales(A_inverse_scale, SCALE_RECIPE_A)
    b_scale = load_scales(B_inverse_scale, SCALE_RECIPE_B)

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
                SCALE_RECIPE_A,
                SCALE_RECIPE_B,
                offs_cm,
                offs_cn,
                M,
                N,
                stride_a_scale_m,
                stride_b_scale_n,
            )

            # inductor generates a suffix
            {%- if TMA_EXPERIMENTAL_API %}
            idx_m = offs_cm[:, None]
            idx_n = offs_cn[None, :]
            mask = (idx_m < M) & (idx_n < N)
            {{store_output(("idx_m", "idx_n"), "accumulator", "mask", indent_width=12, val_shape=("BLOCK_M", "BLOCK_N"))}}
            {%- else %}
            {{store_output(
                ("offs_am", "offs_bn"),
                "accumulator",
                indent_width=12,
                val_shape=("BLOCK_M", "BLOCK_N"),
                block_indexing=True,
            )}}
            {%- endif %}
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
"""


scaled_mm_device_tma_epilogue_scaling_template = TritonTemplate(
    name="scaled_mm_device_tma_epilogue_scaling",
    grid=persistent_mm_grid,
    source=scaled_mm_device_tma_epilogue_scaling + load_scales + apply_scaling,
)

blockwise1xTILESIZE_scaling = r"""
@triton.jit
def blockwise1xTILESIZE_scaling(
    pid,
    scale,
    ki,
    lhs_size,
    lhs_blocks,
    k_blocks,
    BLOCK_lhs: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MIN_BLOCK_TILE_K: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    row_offs_scale = pid * BLOCK_lhs + tl.arange(0, BLOCK_lhs)
    col_offs_scale = ki * tl.cdiv(BLOCK_K, TILE_SIZE) + tl.arange(0, (BLOCK_K + TILE_SIZE - 1) // TILE_SIZE)
    ptrs = scale + row_offs_scale[:, None] * k_blocks + col_offs_scale[None, :]
    mask = (row_offs_scale[:, None] < lhs_size) & (col_offs_scale[None, :] < k_blocks)
    scale_block = tl.load(ptrs, mask=mask, other=1.0)

    scale_expanded = scale_block[:, :, None]
    scale_expanded = tl.broadcast_to(
        scale_expanded,
        (BLOCK_lhs, (BLOCK_K + TILE_SIZE - 1) // TILE_SIZE, MIN_BLOCK_TILE_K)
    )
    scale_expanded = scale_expanded.reshape(
        BLOCK_lhs,
        ((BLOCK_K + TILE_SIZE - 1) // TILE_SIZE) * MIN_BLOCK_TILE_K
    )

    return scale_expanded
"""

blockwise128x128_scaling = r"""
@triton.jit
def blockwise128x128_scaling(
    pid,
    scale,
    ki,
    lhs_blocks,
    k_blocks,
    BLOCK_lhs: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MIN_BLOCK_TILE_lhs: tl.constexpr,
    MIN_BLOCK_TILE_K: tl.constexpr,
):
    row_offs_scale = pid * tl.cdiv(BLOCK_lhs, 128) + tl.arange(0, (BLOCK_lhs + 128 - 1) // 128)
    col_offs_scale = ki * tl.cdiv(BLOCK_K, 128) + tl.arange(0, (BLOCK_K + 128 - 1) // 128)
    ptrs = scale + row_offs_scale[:, None] * k_blocks + col_offs_scale[None, :]
    mask = (row_offs_scale[:, None] < lhs_blocks) & (col_offs_scale[None, :] < k_blocks)
    scale_block = tl.load(ptrs, mask=mask, other=1.0)

    scale_expanded = scale_block[:, :, None, None]
    scale_expanded = tl.broadcast_to(
        scale_expanded,
        ((BLOCK_lhs + 128 - 1) // 128, (BLOCK_K + 128 - 1) // 128, MIN_BLOCK_TILE_lhs, MIN_BLOCK_TILE_K)
    )
    scale_expanded = scale_expanded.reshape(
        ((BLOCK_lhs + 128 - 1) // 128) * MIN_BLOCK_TILE_lhs,
        ((BLOCK_K + 128 - 1) // 128) * MIN_BLOCK_TILE_K
    )

    return scale_expanded
"""

scaled_mm_device_tma_main_loop_scaling = r"""
{{def_kernel("A", "B", "A_inverse_scale", "B_inverse_scale")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    stride_am = {{stride("A", 0)}}
    stride_bn = {{stride("B", 1)}}

    start_pid = tl.program_id(axis=0).to(INDEX_DTYPE)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = triton.language.make_tensor_descriptor(
        base=A,
        shape=[M, K],
        strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = triton.language.make_tensor_descriptor(
        base=B,
        shape=[N, K],
        strides=[stride_bn, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )

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
    a_scale = load_scales(A_inverse_scale, SCALE_RECIPE_A)
    b_scale = load_scales(B_inverse_scale, SCALE_RECIPE_B)

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

        a = tl.load_tensor_descriptor(a_desc, [offs_am, offs_k])
        b = tl.load_tensor_descriptor(b_desc, [offs_bn, offs_k])

        am_blocks = tl.cdiv(M, TILE_SIZE_A)
        ak_blocks = tl.cdiv(K, TILE_SIZE_A)
        bn_blocks = tl.cdiv(N, TILE_SIZE_B)
        bk_blocks = tl.cdiv(K, TILE_SIZE_B)

        {%- if SCALE_RECIPE_A == 5 %}  # ScalingType.Blockwise128x128
        scale_a_block = blockwise128x128_scaling(
            pid_m,
            a_scale,
            ki,
            am_blocks,
            ak_blocks,
            BLOCK_M,
            BLOCK_K,
            MIN_BLOCK_TILE_AM,
            MIN_BLOCK_TILE_AK,
        )
        {%- else %}  # ScalingType.Blockwise1xTILESIZE
        scale_a_block = blockwise1xTILESIZE_scaling(
            pid_m,
            a_scale,
            ki,
            M,
            am_blocks,
            ak_blocks,
            BLOCK_M,
            BLOCK_K,
            MIN_BLOCK_TILE_AK,
            TILE_SIZE_A,
        )
        {%- endif %}

        {%- if SCALE_RECIPE_A == 5 %}  # ScalingType.Blockwise128x128
        scale_b_block = blockwise128x128_scaling(
            pid_n,
            b_scale,
            ki,
            bn_blocks,
            bk_blocks,
            BLOCK_N,
            BLOCK_K,
            MIN_BLOCK_TILE_BN,
            MIN_BLOCK_TILE_BK,
        )
        {%- else %}  # ScalingType.Blockwise1xTILESIZE
        scale_b_block = blockwise1xTILESIZE_scaling(
            pid_n,
            b_scale,
            ki,
            N,
            bn_blocks,
            bk_blocks,
            BLOCK_N,
            BLOCK_K,
            MIN_BLOCK_TILE_BK,
            TILE_SIZE_B,
        )
        {%- endif %}

        a_scaled = a * scale_a_block
        b_scaled = b * scale_b_block
        accumulator = tl.dot(a_scaled, b_scaled.T, accumulator)

        if ki == k_tiles - 1:
            offs_cm = offs_am + tl.arange(0, BLOCK_M)
            offs_cn = offs_bn + tl.arange(0, BLOCK_N)

            # inductor generates a suffix
            {{store_output(
                ("offs_am", "offs_bn"),
                "accumulator",
                indent_width=12,
                val_shape=("BLOCK_M", "BLOCK_N"),
                block_indexing=True,
            )}}
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
"""

scaled_mm_device_tma_main_loop_scaling_template = TritonTemplate(
    name="scaled_mm_device_tma_main_loop_scaling",
    grid=persistent_mm_grid,
    source=scaled_mm_device_tma_main_loop_scaling
    + load_scales
    + blockwise1xTILESIZE_scaling
    + blockwise128x128_scaling,
)

_compute_blackwell_pid = r"""
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, grid_m, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % GROUP_M)
    pid_n = (tile_id % num_pid_in_group) // GROUP_M
    return pid_m, pid_n
"""

_blackwell_ws_persistent_device_tma = r"""
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

    # Note: We require TMA_EXPERIMENTAL_API == False, which
    # we will check before invoking this template.
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
    a_desc = triton.language.make_tensor_descriptor(
        base=A,
        shape=[M, K] if A_ROW_MAJOR else [K, M],
        strides=[stride_am, 1] if A_ROW_MAJOR else [stride_ak, 1],
        block_shape=[BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],
    )
    b_desc = triton.language.make_tensor_descriptor(
        base=B,
        shape=[K, N] if B_ROW_MAJOR else [N, K],
        strides=[stride_bk, 1] if B_ROW_MAJOR else [stride_bn, 1],
        block_shape=[BLOCK_K, BLOCK_N] if B_ROW_MAJOR else [BLOCK_N, BLOCK_K],
    )

    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_M * grid_n

    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, grid_m, GROUP_M, NUM_SMS
        )
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            a = tl.load_tensor_descriptor(
                a_desc,
                [offs_am, offs_k] if A_ROW_MAJOR else [offs_k, offs_am],
            )
            b = tl.load_tensor_descriptor(
                b_desc,
                [offs_k, offs_bn] if B_ROW_MAJOR else [offs_bn, offs_k],
            )
            accumulator += tl.dot(
                a if A_ROW_MAJOR else a.T,
                b if B_ROW_MAJOR else b.T,
                allow_tf32=ALLOW_TF32,
            )

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, grid_m, GROUP_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_M
        offs_cn = pid_n * BLOCK_N
        {%- if EPILOGUE_SUBTILE %}
        tl.static_assert(BLOCK_N % 2 == 0)
        acc = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        {{store_output(
            ("offs_cm", "offs_cn"),
            "acc0",
            indent_width=8,
            val_shape=("BLOCK_M", "BLOCK_N // 2"),
            block_indexing=True
        )}}
        offs_cn2 = offs_cn + BLOCK_N // 2
        {{store_output(
            ("offs_cm", "offs_cn2"),
            "acc1",
            indent_width=8,
            val_shape=("BLOCK_M", "BLOCK_N // 2"),
            block_indexing=True
        )}}
        {%- else %}
        {{store_output(
            ("offs_cm", "offs_cn"),
            "accumulator",
            indent_width=8,
            val_shape=("BLOCK_M", "BLOCK_N"),
            block_indexing=True
        )}}
        {%- endif %}
"""

blackwell_ws_persistent_device_tma_mm_template = TritonTemplate(
    name="blackwell_ws_persistent_device_tma",
    grid=persistent_mm_grid,
    source=_blackwell_ws_persistent_device_tma + _compute_blackwell_pid,
)


# prevent duplication registration of extern functions
@functools.cache
def lazy_register_extern_choice(fn):
    return ExternKernelChoice(fn)


aten_mm = ExternKernelChoice(torch.mm, "at::mm_out", op_overload=aten.mm.out)
aten_mm_dtype = ExternKernelChoice(
    torch.mm,
    "at::_mm_dtype_out_cuda",
    name="mm_dtype",
    op_overload=aten.mm.dtype_out,
)

aten_addmm = ExternKernelChoice(
    torch.addmm, "at::addmm_out", op_overload=aten.addmm.out
)

aten__int_mm = ExternKernelChoice(
    torch._int_mm, "at::_int_mm_out", op_overload=aten._int_mm.out
)

aten__sparse_semi_structured_mm = ExternKernelChoice(
    torch._sparse_semi_structured_mm,
    "at::_sparse_semi_structured_mm",
    has_out_variant=False,
    op_overload=aten._sparse_semi_structured_mm.default,
)

aten__fp8_mm = ExternKernelChoice(
    torch._scaled_mm, "at::_scaled_mm_out", op_overload=aten._scaled_mm.out
)


def _is_int8_mat(mat):
    return mat.get_dtype() in (torch.int8, torch.uint8)


def bias_addmm(inp, mat1, mat2, *, out=None, alpha=1, beta=1):
    """
    Giving torch.addmm a 1D tensor calls a different (faster) cublasLt
    kernel under the hood.  There are a few shapes where this is slower,
    but they are rare.
    """
    if (inp.stride(0) == 0 and inp.size(0) != 0) or inp.size(0) == 1:
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


class DecomposeKSugraphTemplate(SubgraphTemplate):
    def __init__(self):
        super().__init__(
            name="decompose_k",
        )

    def generate(  # type: ignore[override]
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        k_split: int,
    ) -> SubgraphChoiceCaller:
        from torch._dispatch.python import enable_python_dispatcher

        from ..decomposition import select_decomp_table

        name = f"decompose_k_mm_{k_split}_split"
        description = f"{k_split=}"

        with enable_python_dispatcher():
            decompositions = select_decomp_table()
            fn = make_fx(
                functools.partial(decomposeK, k_splits=k_split),
                decompositions,
            )

            return super().generate(
                name=name,
                input_nodes=input_nodes,
                layout=layout,
                make_fx_graph=fn,
                description=description,
            )


decompose_k_subgraph_template = DecomposeKSugraphTemplate()


class ContiguousTemplate(SubgraphTemplate):
    def __init__(self, name: str, description: str, fn: Any):
        self.name = name
        self.description = description
        self.fn = fn
        super().__init__(
            name=name,
        )

    def generate(  # type: ignore[override]
        self,
        input_nodes: list[Buffer],
        layout: Layout,
    ) -> SubgraphChoiceCaller:
        from torch._dispatch.python import enable_python_dispatcher

        from ..decomposition import select_decomp_table

        with enable_python_dispatcher():
            decompositions = select_decomp_table()
            fn = make_fx(
                self.fn,
                decompositions,
            )

            return super().generate(
                name=self.name,
                input_nodes=input_nodes,
                layout=layout,
                make_fx_graph=fn,
                description=self.description,
            )


def contiguous_mm(a, b):
    return torch.mm(a, b.contiguous())


def contiguous_addmm(inp, a, b):
    return torch.addmm(inp, a, b.contiguous())


mm_contiguous_subgraph_template = ContiguousTemplate(
    "contiguous_mm", "contiguous mm", contiguous_mm
)
addmm_contiguous_subgraph_template = ContiguousTemplate(
    "contiguous_addmm", "contiguous addmm", contiguous_addmm
)


@register_lowering(aten.mm, type_promotion_kind=None)
def tuned_mm(mat1, mat2, out_dtype=None, *, layout=None):
    """
    Lowering for autotuning aten.mm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    if out_dtype is not None:
        input_dtype = mat1.get_dtype()
        torch._check(
            mat2.get_dtype() == input_dtype,
            lambda: "input dtypes must be the same",
        )
        torch._check(
            mat1.get_device().type == "cuda",
            lambda: "out_dtype is only supported for CUDA",
        )
        torch._check(
            out_dtype == input_dtype
            or (
                out_dtype == torch.float32
                and input_dtype in (torch.float16, torch.bfloat16)
            ),
            lambda: "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs",
        )

    # Lower matmul-related operations (e.g., torch.matmul / torch.bmm / torch.addmm)
    # into native matmul IR using `ops.dot`. When we see a matmul pattern
    # (C[y, x] = A[y, r] * B[r, x]), the core idea is to emulate a broadcasted
    # multiply followed by a sum.
    #
    # For example, given `C = torch.matmul(A, B)`, this can be rewritten as:
    #
    #     Prod = A.unsqueeze(-1) * B.unsqueeze(0)
    #     C = Prod.sum(dim=1)
    #
    # Instead of explicitly using `ops.mul` and `ops.reduction("sum")`, we lower
    # these into `ops.dot` (pointwise) and `ops.reduction("dot")`. These IR nodes
    # are semantically equivalent to the `ops.mul` + `ops.reduction("sum")`
    # combination, but are lowered to `tl.dot` during the code generation phase.
    if use_native_matmul(mat1, mat2):
        mat1 = lowerings[aten.unsqueeze](mat1, -1)
        mat2 = lowerings[aten.unsqueeze](mat2, 0)
        args, kwargs = transform_args(
            args=[mat1, mat2],
            kwargs={},
            broadcast=True,
            type_promotion_kind=None,
            convert_input_to_bool=False,
        )  # Handles broadcasting the arguments

        if inductor_config.triton.codegen_upcast_to_fp32 and mat1.dtype in [
            torch.float16,
            torch.bfloat16,
        ]:

            def _to_dtype(x):
                return ops.to_dtype(x, mat1.dtype, use_compute_types=False)

            args = [make_pointwise(_to_dtype)(x) for x in args]

        mul_pointwise = make_pointwise(ops.dot)(*args)
        dot_reduction = make_reduction("dot")(mul_pointwise, 1)

        return dot_reduction

    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=out_dtype
    )
    static_shape, is_nonzero = _is_static_problem(layout)
    name = "mm"

    # Create MMKernelInputs for standard MM at the top
    kernel_inputs = MMKernelInputs([mat1, mat2], out_dtype=out_dtype)

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

    choices: list[ChoiceCaller] = []
    static_shape, is_nonzero = _is_static_problem(layout)

    aten_handler: ExternKernelChoice = aten_mm
    aten_extra_kwargs: dict[str, Any] = {}
    if out_dtype is not None:
        aten_handler = aten_mm_dtype
        aten_extra_kwargs = {"out_dtype": out_dtype}

    templates_to_use: list[Union[ExternKernelChoice, KernelTemplate]] = []
    kwarg_overrides: dict[str, dict[str, Any]] = {}
    if use_aten_gemm_kernels():
        templates_to_use.append(aten_handler)
        if aten_extra_kwargs:
            kwarg_overrides[aten_handler.uid] = aten_extra_kwargs

    if (
        out_dtype is None
        and is_nonzero
        and use_triton_template(layout, check_max_autotune=True)
    ):
        if use_decompose_k_choice(m, n, k):
            templates_to_use.append(decompose_k_subgraph_template)
        # Triton Templates typically perform very poorly for large K.
        # Its highly unlikely that if we want to use decompose_k, then
        # Triton will ever win.
        #
        # To be conservative we increase this threshold for N/M by 2.
        is_exhaustive = inductor_config.max_autotune_gemm_search_space == "exhaustive"
        if is_exhaustive or not use_decompose_k_choice(m, n, k, threshold_multiple=2):
            templates_to_use.append(mm_template)

            if use_triton_tma_template(mat1, mat2, output_layout=layout):
                templates_to_use.append(persistent_tma_mm_template)

            if use_triton_blackwell_tma_template(mat1, mat2, output_layout=layout):
                templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)

        templates_to_use.append(mm_contiguous_subgraph_template)

    choices.extend(
        V.choices.get_template_configs(
            kernel_inputs,
            templates_to_use,
            "mm",
            kwarg_overrides=kwarg_overrides,
        )
    )

    if (
        out_dtype is None
        and is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op("mm")
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, kernel_inputs.nodes()
        )

    if out_dtype is None and is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, kernel_inputs.nodes())
    if out_dtype is None and is_nonzero and use_ck_tile_gemm_template(layout, m, n, k):
        CKTileGemmTemplate.add_choices(choices, layout, kernel_inputs.nodes())

    if out_dtype is None and use_cpp_gemm_template(layout, mat1, mat2):
        CppGemmTemplate.add_choices(
            choices,
            layout,
            kernel_inputs.nodes(),
        )

    input_nodes = [mat1, mat2]
    if (
        out_dtype is None
        and is_nonzero
        and use_triton_template(layout)
        and torch._inductor.config.run_autoheuristic(name)
        and is_triton(mat1)
    ):
        always_included = []
        if use_aten_gemm_kernels():
            always_included.append("extern_mm")
        num_choices_before_extra_configs = len(choices)
        choices.extend(
            V.choices.get_template_configs(
                # TODO(coconutruben): remove once we deprecate ah
                # mm-extra is a hack to keep the ah functionality alive
                # while we transition to the unified kwargs retrieval
                kernel_inputs,
                [mm_template],
                "mm-ah",
            )
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

    if out_dtype is None:
        for k in inductor_config.external_matmul:
            choices.append(
                lazy_register_extern_choice(k).bind(kernel_inputs.nodes(), layout)
            )

    best_config_future = None
    if out_dtype is None and torch._inductor.config.remote_gemm_autotune_cache:
        # Purposely not awaiting the future here - this kicks off the best config lookup at lowering time
        # The future will be awaited at scheduling time in select_algorithm.py
        best_config_future = gen_best_config(mat1, mat2)

    if box := distributed_autotune.maybe_autotune_remote(
        name, choices, kernel_inputs.nodes(), layout
    ):
        return box

    return autotune_select_algorithm(
        name,
        choices,
        kernel_inputs.nodes(),
        layout,
        best_config_future=best_config_future,
    )


@register_lowering(aten._int_mm, type_promotion_kind=None)
def tuned_int_mm(mat1, mat2, *, layout=None):
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=torch.int32
    )
    name = "int_mm"
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

    static_shape, is_nonzero = _is_static_problem(layout)
    use_cutlass = static_shape and is_nonzero and use_cutlass_template(layout, m, n, k)
    choices: list[ChoiceCaller] = []

    # Create MMKernelInputs for Int MM
    kernel_inputs = MMKernelInputs([mat1, mat2], out_dtype=torch.int32)

    # Collect all templates for unified call
    templates_to_use: list[Union[ExternKernelChoice, KernelTemplate]] = []
    if use_aten_gemm_kernels():
        templates_to_use.append(aten__int_mm)

    if is_nonzero and use_triton_template(
        layout, enable_int32=True, check_max_autotune=False
    ):
        templates_to_use.append(mm_template)

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(kernel_inputs, templates_to_use, name)
    )

    if use_cutlass and _use_cutlass_for_op(name):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, kernel_inputs.nodes(), fuseable=True, non_fuseable=True
        )

    return autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)


@register_lowering(aten.addmm, type_promotion_kind=None)
def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    """
    Lowering for autotuning aten.addmm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    if use_native_matmul(mat1, mat2):
        if beta == 0:
            arg1 = 0
        else:
            arg1 = lowerings[aten.mul](beta, inp)

        if alpha == 0:
            arg2 = 0
        else:
            arg2 = lowerings[aten.mul](alpha, lowerings[aten.mm](mat1, mat2))

        return lowerings[aten.add](arg1, arg2)

    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    static_shape, is_nonzero = _is_static_problem(layout)
    name = "addmm"

    # Create MMKernelInputs for AddMM at the top
    kernel_inputs = MMKernelInputs(
        [inp_expanded, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
    )
    choices: list[ChoiceCaller] = []

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
        # TODO(coconutruben): combine this with the main flow of addmm through
        # a subgraph or something as inp vs inp_expanded causes some slight numeric
        # differences
        kernel_inputs = MMKernelInputs(
            [inp, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
        )
        choices.extend(
            V.choices.get_template_configs(
                kernel_inputs,
                [aten_addmm],
                name,
            )
        )
        return autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)

    # Collect all templates for unified call
    templates_to_use: list[Union[ExternKernelChoice, KernelTemplate]] = []
    if use_aten_gemm_kernels():
        templates_to_use.extend([aten_bias_addmm, aten_addmm])

    if is_nonzero and use_triton_template(layout, check_max_autotune=False):
        templates_to_use.append(mm_template)

        if use_triton_tma_template(mat1, mat2, output_layout=layout):
            templates_to_use.append(persistent_tma_mm_template)

        if use_triton_blackwell_tma_template(mat1, mat2, output_layout=layout):
            templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)

        templates_to_use.append(addmm_contiguous_subgraph_template)

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(kernel_inputs, templates_to_use, name)
    )

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op(name)
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            # reorder here because CUTLASS expects (x, w, bias) but torch
            # is bias, x, w
            kernel_inputs.nodes(reorder=[1, 2, 0]),
            alpha=alpha,
            beta=beta,
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(
            choices,
            layout,
            # reorder here because CK expects (x, w, bias) but torch
            # is bias, x, w
            kernel_inputs.nodes(reorder=[1, 2, 0]),
            alpha=alpha,
            beta=beta,
            input_reorder=[2, 0, 1],
        )

    if use_cpp_gemm_template(layout, mat1, mat2):
        CppGemmTemplate.add_choices(
            choices,
            layout,
            kernel_inputs.nodes(),
            alpha=alpha,
            beta=beta,
            has_bias=True,
        )

    return autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)


@register_lowering(aten._sparse_semi_structured_mm, type_promotion_kind=None)
def tuned_sparse_semi_structured_mm(
    mat1, mat1_meta, mat2, *, out_dtype=None, layout=None
):
    from torch._inductor.select_algorithm import realize_inputs

    # TODO(coconturuben): support V.choices.get_mm_configs for sparse_semi_structured_mm
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
        "sparse_semi_structured_mm", choices, (mat1, mat1_meta, mat2), layout
    )


scaling_pairs = [
    (ScalingType.TensorWise, ScalingType.TensorWise),
    (ScalingType.RowWise, ScalingType.RowWise),
    (ScalingType.BlockWise1x128, ScalingType.BlockWise128x128),
    (ScalingType.BlockWise1x128, ScalingType.BlockWise1x128),
]


epilogue_scaling_types = [ScalingType.TensorWise, ScalingType.RowWise]
main_loop_scaling_types = [ScalingType.BlockWise1x128, ScalingType.BlockWise128x128]


def _is_tensorwise_scaling(sz: Any) -> bool:
    return (len(sz) == 0) or all(
        V.graph.sizevars.statically_known_equals(d, 1) for d in sz
    )


def _is_rowwise_scaling(sz: Any, transpose: bool) -> bool:
    idx = 0 if transpose else -1
    return V.graph.sizevars.statically_known_equals(sz[idx], 1)


def _is_blockwise1xTILESIZE_scaling(
    sz: Any, tensor_sz: Any, tile_size: int, transpose: bool
) -> bool:
    lhs = 1 if transpose else 0
    rhs = 0 if transpose else 1
    return V.graph.sizevars.statically_known_equals(
        sz[lhs], tensor_sz[lhs]
    ) and V.graph.sizevars.statically_known_equals(
        sz[rhs], ceildiv(tensor_sz[rhs], tile_size)
    )


def _is_blockwise128x128_scaling(sz: Any, tensor_sz: Any) -> bool:
    return V.graph.sizevars.statically_known_equals(
        sz[0], ceildiv(tensor_sz[0], 128)
    ) and V.graph.sizevars.statically_known_equals(sz[1], ceildiv(tensor_sz[1], 128))


def is_desired_scaling(
    t: Any,
    scale_size: torch.Tensor,
    scaling_type: ScalingType,
    transpose: bool = False,
) -> bool:
    match scaling_type:
        case ScalingType.TensorWise:
            return _is_tensorwise_scaling(scale_size)
        case ScalingType.RowWise:
            return _is_rowwise_scaling(scale_size, transpose)
        case ScalingType.BlockWise1x128:
            return _is_blockwise1xTILESIZE_scaling(
                scale_size, t.get_size(), 128, transpose
            )
        case ScalingType.BlockWise128x128:
            return _is_blockwise128x128_scaling(scale_size, t.get_size())
        case _:
            raise AssertionError(f"Unsupported scaling type {scaling_type}")


def get_tile_size(scale_option) -> int:
    match scale_option:
        case ScalingType.BlockWise128x128:
            return 128
        case ScalingType.BlockWise1x128:
            return 128
        case _:
            raise AssertionError(
                f"Unsupported scaling type {scale_option} in get_tile_size"
            )


def get_scaling_options(
    mat_a: Any,
    mat_b: Any,
    scale_a_size: torch.Tensor,
    scale_b_size: torch.Tensor,
) -> tuple[ScalingType, ScalingType]:
    for scale_option_a, scale_option_b in scaling_pairs:
        if is_desired_scaling(
            mat_a, scale_a_size, scale_option_a
        ) and is_desired_scaling(mat_b, scale_b_size, scale_option_b, transpose=True):
            return scale_option_a, scale_option_b

    raise AssertionError(
        f"Inductor Triton does not support scale_a.shape = {scale_a_size}, scale_b.shape = {scale_b_size}"
    )  # verify that shapes are supported by at least one existing pairing


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
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
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
    name = "scaled_mm"
    check_supported_striding(mat_a, mat_b)

    scale_a_real, scale_b_real = realize_inputs(scale_a, scale_b)

    input_nodes: list[Any]

    if not bias:
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real]
    else:
        bias_real = realize_inputs(bias)
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real, bias_real]

    # Create MMKernelInputs for Scaled MM (matrices are at indices 0, 1)
    kernel_inputs = MMKernelInputs(
        input_nodes, mat1_idx=0, mat2_idx=1, out_dtype=out_dtype
    )

    choices: list[ChoiceCaller] = []

    # Collect all templates for unified call
    templates_to_use: list[Union[ExternKernelChoice, KernelTemplate]] = []
    kwarg_overrides = {}

    if use_aten_gemm_kernels():
        templates_to_use.append(aten__fp8_mm)
        kwarg_overrides[aten__fp8_mm.uid] = dict(
            out_dtype=out_dtype, use_fast_accum=use_fast_accum
        )

    _, is_nonzero = _is_static_problem(layout)

    if (
        # We dont have triton lowerings for the MX variants yet
        scale_a.dtype == torch.float32
        and is_nonzero
        and use_triton_template(layout, enable_float8=True, check_max_autotune=False)
    ):
        overriders = dict(USE_FAST_ACCUM=use_fast_accum)

        # TODO (paulzhan): There is no template that exists for bias and TMA
        # Don't run tma template currently if bias exist
        if use_triton_tma_template(mat_a, mat_b, output_layout=layout) and not bias:
            scale_a_size, scale_b_size = scale_a_real.shape, scale_b_real.shape

            scale_option_a, scale_option_b = get_scaling_options(
                mat_a, mat_b, scale_a_size, scale_b_size
            )
            overriders["SCALE_RECIPE_A"] = scale_option_a.value
            overriders["SCALE_RECIPE_B"] = scale_option_b.value

            if (
                scale_option_a in epilogue_scaling_types
                and scale_option_b in epilogue_scaling_types
            ):
                templates_to_use.append(scaled_mm_device_tma_epilogue_scaling_template)
                kwarg_overrides[scaled_mm_device_tma_epilogue_scaling_template.uid] = (
                    overriders
                )
            elif (
                scale_option_a in main_loop_scaling_types
                and scale_option_b in main_loop_scaling_types
            ):
                overriders["TILE_SIZE_A"] = get_tile_size(scale_option_a)
                overriders["TILE_SIZE_B"] = get_tile_size(scale_option_b)

                templates_to_use.append(scaled_mm_device_tma_main_loop_scaling_template)
                kwarg_overrides[scaled_mm_device_tma_main_loop_scaling_template.uid] = (
                    overriders
                )
            else:
                raise AssertionError(
                    "Inductor Triton does not support scaling options that are present "
                    + "in both epilogue scaling and main loop scaling"
                )

        if (
            use_triton_blackwell_tma_template(mat_a, mat_b, output_layout=layout)
            and not bias
        ):
            templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)
            kwarg_overrides[blackwell_ws_persistent_device_tma_mm_template.uid] = (
                overriders
            )

        templates_to_use.append(mm_template)
        kwarg_overrides[mm_template.uid] = overriders

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(
            kernel_inputs,
            templates_to_use,
            name,
            kwarg_overrides=kwarg_overrides,
        )
    )

    # Early return for MX variants
    if scale_a.dtype != torch.float32:
        return autotune_select_algorithm(name, choices, input_nodes, layout)

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op(name)
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            kernel_inputs.nodes(),  # type: ignore[arg-type]
            use_fast_accum=use_fast_accum,  # type: ignore[arg-type]
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, kernel_inputs.nodes())

    return autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)


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
