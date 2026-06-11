# Copyright (c) 2025, Tri Dao.
# Shared utilities for TVM-FFI GEMM compilation.

from functools import partial


import cutlass.cute as cute
from cutlass import Int32, Int64, Float32
from cutlass.cute.runtime import make_ptr

from .compile_utils import make_fake_tensor as fake_tensor
from .cute_dsl_utils import torch2cute_dtype_map
from .tile_scheduler import TileSchedulerOptions
from .varlen_utils import VarlenArguments, VarlenNArguments


def div_for_dtype(dtype):
    """16-byte alignment: divisibility in elements = 128 // dtype_width_bits."""
    return 128 // dtype.width


def perm3d_single(t, varlen_m=False):
    """Permute a single 3D tensor from (L, *, *) to (*, *, L), skipping for varlen_m or 2D."""
    return t.permute(1, 2, 0) if t is not None and t.ndim == 3 and not varlen_m else t


def perm3d(A, B, D, C, varlen_m=False, varlen_k=False, varlen_n=False):
    """Permute 3D tensors from (L, *, *) to (*, *, L)."""

    def _perm(t):
        return t.permute(1, 2, 0) if t is not None and t.ndim == 3 else t

    if varlen_m:
        return A, _perm(B), D, C
    elif varlen_k:
        return A, B, _perm(D), _perm(C)
    elif varlen_n:
        return _perm(A), B, D, C
    else:
        return _perm(A), _perm(B), _perm(D), _perm(C)


def get_major(t, dim0, dim1):
    return dim1 if t.stride(1) == 1 else dim0


def get_majors(A_p, B_p, D_p, C_p):
    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n")
    c_major = get_major(C_p, "m", "n") if C_p is not None else None
    return a_major, b_major, d_major, c_major


def get_dtypes(A, B, D, C):
    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype]
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None
    return a_dtype, b_dtype, d_dtype, c_dtype


def make_scheduler_args(
    max_active_clusters, max_swizzle_size, tile_count_semaphore, batch_idx_permute=None
):
    return TileSchedulerOptions(
        max_active_clusters=Int32(max_active_clusters),
        raster_order=None,
        max_swizzle_size=max_swizzle_size,
        tile_count_semaphore=(
            tile_count_semaphore.data_ptr() if tile_count_semaphore is not None else None
        ),
        batch_idx_permute=batch_idx_permute,
    )


def make_fake_scheduler_args(has_semaphore, has_batch_idx_permute, l_sym):
    return TileSchedulerOptions(
        max_active_clusters=Int32(1),
        max_swizzle_size=Int32(8),
        tile_count_semaphore=(
            make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=4) if has_semaphore else None
        ),
        batch_idx_permute=(
            fake_tensor(Int32, (l_sym,), leading_dim=0, divisibility=4)
            if has_batch_idx_permute
            else None
        ),
    )


def make_varlen_args(cu_seqlens_m, cu_seqlens_k, A_idx=None, *, cu_seqlens_n=None):
    if cu_seqlens_m is None and cu_seqlens_k is None and cu_seqlens_n is None:
        return None
    if cu_seqlens_n is not None:
        return VarlenNArguments(
            mCuSeqlensM=cu_seqlens_m,
            mCuSeqlensK=cu_seqlens_k,
            mAIdx=A_idx,
            mCuSeqlensN=cu_seqlens_n,
        )
    return VarlenArguments(
        mCuSeqlensM=cu_seqlens_m,
        mCuSeqlensK=cu_seqlens_k,
        mAIdx=A_idx,
    )


def make_fake_varlen_args(varlen_m, varlen_k, gather_A=False, aidx_len=None, *, varlen_n=False):
    if not varlen_m and not varlen_k and not varlen_n:
        return None
    num_seqlens = cute.sym_int()
    cu_seqlens_m = (
        fake_tensor(Int32, (num_seqlens,), leading_dim=0, divisibility=4) if varlen_m else None
    )
    cu_seqlens_k = (
        fake_tensor(Int32, (num_seqlens,), leading_dim=0, divisibility=4) if varlen_k else None
    )
    a_idx = (
        fake_tensor(Int32, (aidx_len,), leading_dim=0, divisibility=4)
        if gather_A is True
        else None
    )
    if varlen_n:
        return VarlenNArguments(
            mCuSeqlensM=cu_seqlens_m,
            mCuSeqlensK=cu_seqlens_k,
            mAIdx=a_idx,
            mCuSeqlensN=fake_tensor(Int32, (num_seqlens,), leading_dim=0, divisibility=4),
        )
    return VarlenArguments(
        mCuSeqlensM=cu_seqlens_m,
        mCuSeqlensK=cu_seqlens_k,
        mAIdx=a_idx,
    )


def make_fake_gemm_tensors(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    varlen_m=False,
    varlen_k=False,
    varlen_n=False,
    gather_A=False,
):
    """Create fake tensors for mA, mB, mD, mC with shared sym_ints.
    Pass dtype=None to get None for that tensor (e.g. optional C).
    Returns (mA, mB, mD, mC, m, n, k, l).
    When varlen_m, m is total_m (flattened M of D/C). When varlen_k, k is total_k.
    """
    a_leading = 1 if a_major == "k" else 0
    b_leading = 1 if b_major == "k" else 0
    d_leading = 1 if d_major == "n" else 0
    c_leading = 1 if c_major == "n" else 0
    m, n, k, l = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    div_a = div_for_dtype(a_dtype)
    div_b = div_for_dtype(b_dtype)
    div_d = div_for_dtype(d_dtype) if d_dtype is not None else 1
    div_c = div_for_dtype(c_dtype) if c_dtype is not None else 1
    if varlen_m:
        # m is total_m in this case: the flattened M dimension of D/C
        m = cute.sym_int()
        a_m = cute.sym_int() if gather_A else m
        mA = fake_tensor(a_dtype, (a_m, k), leading_dim=a_leading, divisibility=div_a)
        mB = fake_tensor(b_dtype, (n, k, l), leading_dim=b_leading, divisibility=div_b)
        mD = fake_tensor(d_dtype, (m, n), leading_dim=d_leading, divisibility=div_d)
        mC = fake_tensor(c_dtype, (m, n), leading_dim=c_leading, divisibility=div_c)
    elif varlen_k:
        # k is total_k in this case: the flattened K dimension of A/B
        k = cute.sym_int()
        a_k = cute.sym_int() if gather_A else k
        mA = fake_tensor(a_dtype, (m, a_k), leading_dim=a_leading, divisibility=div_a)
        mB = fake_tensor(b_dtype, (n, k), leading_dim=b_leading, divisibility=div_b)
        mD = fake_tensor(d_dtype, (m, n, l), leading_dim=d_leading, divisibility=div_d)
        mC = fake_tensor(c_dtype, (m, n, l), leading_dim=c_leading, divisibility=div_c)
    elif varlen_n:
        n = cute.sym_int()
        mA = fake_tensor(a_dtype, (m, k, l), leading_dim=a_leading, divisibility=div_a)
        mB = fake_tensor(b_dtype, (n, k), leading_dim=b_leading, divisibility=div_b)
        mD = fake_tensor(d_dtype, (m, n), leading_dim=d_leading, divisibility=div_d)
        mC = fake_tensor(c_dtype, (m, n), leading_dim=c_leading, divisibility=div_c)
    else:
        mA = fake_tensor(a_dtype, (m, k, l), leading_dim=a_leading, divisibility=div_a)
        mB = fake_tensor(b_dtype, (n, k, l), leading_dim=b_leading, divisibility=div_b)
        mD = fake_tensor(d_dtype, (m, n, l), leading_dim=d_leading, divisibility=div_d)
        mC = fake_tensor(c_dtype, (m, n, l), leading_dim=c_leading, divisibility=div_c)
    return mA, mB, mD, mC, m, n, k, l


def compile_gemm_kernel(
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
    post_init=None,
    mSFA=None,
    mSFB=None,
    has_trace_ptr=False,
    use_tma_gather=False,
    concat_layout=None,
    num_warps=None,
):
    """Build GemmCls instance, apply SM90 partial, and cute.compile with TVM-FFI."""
    if device_capacity[0] == 8:
        sm8x_kwargs = {"is_persistent": persistent, "num_warps": num_warps}
        sm8x_kwargs["arch"] = device_capacity[0] * 10 + device_capacity[1]
        GemmCls = partial(GemmCls, **sm8x_kwargs)
    elif device_capacity[0] in [9, 12]:
        GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)
    elif device_capacity[0] in [10, 11]:
        GemmCls = partial(
            GemmCls,
            use_clc_persistence=is_dynamic_persistent,
            use_tma_gather=use_tma_gather,
        )
    gemm_obj = GemmCls(
        Float32,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        gather_A=gather_A,
        concat_layout=concat_layout,
    )
    if post_init:
        post_init(gemm_obj)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    sf_args = () if device_capacity[0] in (8, 9, 12) else (mSFA, mSFB)
    # Trace pointer: Optional[Int64]. Compile with Int64(0) when tracing is
    # requested, None otherwise. TVM-FFI caches each variant separately.
    trace_ptr = Int64(0) if has_trace_ptr else None
    return cute.compile(
        gemm_obj,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        stream,
        *sf_args,
        trace_ptr,
        options="--enable-tvm-ffi",
    )
