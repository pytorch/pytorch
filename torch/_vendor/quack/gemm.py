# Copyright (c) 2025-2026, QuACK team.
# GEMM compilation via TVM-FFI with fake tensors and NamedTuple args.

from typing import Optional

from torch import Tensor

import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.runtime import make_ptr

from .cache_utils import jit_cache
from .compile_utils import make_fake_tensor as fake_tensor
from .cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from .gemm_default_epi import (
    GemmDefaultEpiMixin,
    GemmDefaultSm80,
    GemmDefaultSm90,
    GemmDefaultSm100,
    GemmDefaultSm120,
)
from .rounding import RoundingMode
from .gemm_tvm_ffi_utils import (
    get_majors,
    get_dtypes,
    perm3d,
    make_scheduler_args,
    make_varlen_args,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_fake_gemm_tensors,
    compile_gemm_kernel,
)


@jit_cache
def _compile_gemm(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    rowvec_dtype,
    colvec_dtype,
    colvec_ndim,
    alpha_mode,
    beta_mode,
    add_to_output,
    concat_layout,
    varlen_m,
    varlen_k,
    gather_A,
    use_tma_gather,
    has_batch_idx_permute,
    device_capacity,
    rounding_mode,
    sr_seed_mode,
    has_trace_ptr,
    num_warps,
):
    sm_to_cls = {
        8: GemmDefaultSm80,
        9: GemmDefaultSm90,
        10: GemmDefaultSm100,
        11: GemmDefaultSm100,
        12: GemmDefaultSm120,
    }
    GemmCls = sm_to_cls[device_capacity[0]]
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        varlen_m=varlen_m,
        varlen_k=varlen_k,
        gather_A=gather_A,
    )

    def fake_scalar(mode, dtype=Float32):
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(1.0 if dtype == Float32 else 0)
        else:
            return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=4)

    mRowVec = fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4)
    if colvec_ndim == 2:
        mColVec = fake_tensor(colvec_dtype, (l, m), leading_dim=1, divisibility=4)
    elif colvec_ndim == 1:  # m is total_m in this case
        mColVec = fake_tensor(colvec_dtype, (m,), leading_dim=0, divisibility=4)
    else:
        mColVec = None

    epi_args = GemmCls.EpilogueArguments(
        alpha=fake_scalar(alpha_mode),
        beta=fake_scalar(beta_mode),
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
        add_to_output=add_to_output,
        rounding_mode=rounding_mode,
        sr_seed=fake_scalar(sr_seed_mode, dtype=Int32),
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] <= 9), has_batch_idx_permute, l
    )
    aidx_len = m if varlen_m else (k if varlen_k else None)
    varlen_args = make_fake_varlen_args(varlen_m, varlen_k, gather_A, aidx_len)
    return compile_gemm_kernel(
        GemmCls,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        gather_A,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        has_trace_ptr=has_trace_ptr,
        use_tma_gather=use_tma_gather,
        concat_layout=concat_layout or None,
        num_warps=num_warps,
    )


def gemm(
    # (l, m, k) or (total_m, k) if varlen_m or (m, total_k) if varlen_k or (whatever, k) if gather_A_varlen_m or (m, whatever) if gather_A_varlen_k
    A: Tensor,
    B: Tensor,  # (l, n, k) or (n, total_k) if varlen_k
    D: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    cluster_K: int = 1,
    tile_K: int | None = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[Tensor] = None,  # (l, n)
    colvec_bias: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    cu_seqlens_k: Optional[Tensor] = None,  # (l+1,) cumulative sum of k values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) or (total_k,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (l,) permutation of batch indices for scheduler
    add_to_output: bool = False,
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int | Tensor = 0,
    use_tma_gather: bool = False,
    concat_layout: dict | None = None,
    trace_ptr=None,  # Optional Int64 from TraceSession.ptr
    num_warps: Optional[int] = None,
) -> None:
    varlen_m = cu_seqlens_m is not None
    varlen_k = cu_seqlens_k is not None
    varlen = varlen_m or varlen_k
    gather_A = A_idx is not None
    assert not (varlen_m and varlen_k), "Only one of cu_seqlens_m and cu_seqlens_k"
    if gather_A:
        assert varlen, "gather_A requires varlen"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    if add_to_output:
        assert not varlen_m, "Add to output not supported with varlen_m"
    if varlen_m:
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
    if varlen_k:
        assert A.stride(-2) == 1, "varlen_k requires A to be m-major"
        assert B.stride(-2) == 1, "varlen_k requires B to be n-major"

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [8, 9, 10, 11, 12], (
        "Only SM8x, SM90, SM100, SM110, and SM120 are supported"
    )
    if use_tma_gather:
        assert device_capacity[0] in [10, 11], "TMA gather currently requires SM100/SM110"
    if rounding_mode == RoundingMode.RS:
        assert device_capacity[0] == 10, "Stochastic rounding (RoundingMode.RS) requires SM100"
    if is_dynamic_persistent and device_capacity[0] <= 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler for SM8x and SM90 requires a semaphore in GMEM"
        )
    if device_capacity[0] == 8:
        if add_to_output:
            C = D
            add_to_output = False

    A_p, B_p, D_p, C_p = perm3d(A, B, D, C, varlen_m=varlen_m, varlen_k=varlen_k)
    a_major, b_major, d_major, c_major = get_majors(A_p, B_p, D_p, C_p)
    a_dtype, b_dtype, d_dtype, c_dtype = get_dtypes(A, B, D, C)

    alpha_mode = 2 if isinstance(alpha, Tensor) else (1 if alpha != 1.0 else 0)
    beta_mode = 2 if isinstance(beta, Tensor) else (1 if beta != 1.0 else 0)
    colvec_ndim = colvec_bias.ndim if colvec_bias is not None else 0
    concat_layout = tuple(sorted(concat_layout)) if concat_layout else ()

    sr_seed_mode = (
        2 if isinstance(sr_seed, Tensor) else (1 if rounding_mode == RoundingMode.RS else 0)
    )
    tile_shape_mnk = (tile_M, tile_N) if tile_K is None else (tile_M, tile_N, tile_K)
    compiled_fn = _compile_gemm(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        tile_shape_mnk,
        (cluster_M, cluster_N, cluster_K),
        pingpong,
        persistent,
        is_dynamic_persistent,
        torch2cute_dtype_map[rowvec_bias.dtype] if rowvec_bias is not None else None,
        torch2cute_dtype_map[colvec_bias.dtype] if colvec_bias is not None else None,
        colvec_ndim,
        alpha_mode,
        beta_mode,
        add_to_output,
        concat_layout,
        varlen_m,
        varlen_k,
        gather_A,
        use_tma_gather,
        batch_idx_permute is not None,
        device_capacity,
        rounding_mode,
        sr_seed_mode,
        trace_ptr is not None,
        num_warps,
    )

    from .cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return

    def scalar_arg(scalar, mode, dtype=Float32):
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(scalar)
        else:
            return scalar.data_ptr()

    cluster_size = cluster_M * cluster_N * cluster_K
    max_active_clusters = (
        get_max_active_clusters(cluster_size, device_capacity=device_capacity) if persistent else 0
    )

    epi_args = GemmDefaultEpiMixin.EpilogueArguments(
        alpha=scalar_arg(alpha, alpha_mode),
        beta=scalar_arg(beta, beta_mode),
        mRowVecBroadcast=rowvec_bias,
        mColVecBroadcast=colvec_bias,
        add_to_output=None,
        rounding_mode=None,
        sr_seed=scalar_arg(sr_seed, sr_seed_mode, dtype=Int32),
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        tile_count_semaphore,
        batch_idx_permute,
    )
    varlen_args = make_varlen_args(cu_seqlens_m, cu_seqlens_k, A_idx)

    if device_capacity[0] in [10, 11]:
        compiled_fn(
            A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None, None, trace_ptr
        )
    else:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, trace_ptr)
