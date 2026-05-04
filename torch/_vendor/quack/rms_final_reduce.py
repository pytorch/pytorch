# Copyright (c) 2025-2026, Tri Dao.
# Given a 2D array of partial squared sums, compute rstd[m] = rsqrt(sum_n(x[m,n]) * scale + eps).
# This is the second kernel in a gemm_rms fused pipeline where the first GEMM kernel
# writes per-tile partial sums of squares.

import math
from typing import Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

import torch
from torch import Tensor

from . import copy_utils
from .compile_utils import make_fake_tensor as fake_tensor
from .reduce import row_reduce
from .reduction_base import ReductionBase
from .cache_utils import jit_cache
from .cute_dsl_utils import torch2cute_dtype_map


class RmsFinalReduce(ReductionBase):
    """Reduce partial squared sums and compute rstd: rstd[m] = rsqrt(sum_n(x[m,n]) * scale + eps).

    Inherits from ReductionBase for tiled copy, reduction buffer, and cluster support.
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        super().__init__(dtype, N, stage=1)

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        self.cluster_n = 1

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mRstd: cute.Tensor,
        scale: Float32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        vecsize = math.gcd(self.N, 128 // self.dtype.width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(mX, mRstd, scale, eps, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mRstd: cute.Tensor,
        scale: Float32,
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tv_layout = tiled_copy.layout_tv_tiled

        smem = cutlass.utils.SmemAllocator()
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        thr_copy = tiled_copy.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]

        tXrX = cute.make_rmem_tensor_like(tXgX)
        cute.filter_zeros(tXrX).fill(0)

        is_even_N = const_expr(shape[1] == tiler_mn[1])
        tXpX = (
            copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )

        row = tXcX[0][0]
        if row < shape[0]:
            copy_utils.copy(tXgX, tXrX, pred=tXpX)
        x = tXrX.load().to(Float32)

        sum_x = row_reduce(
            x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
        )
        rstd = cute.math.rsqrt(sum_x * scale + eps, fastmath=True)
        if tXcX[0][1] == 0 and row < shape[0]:
            mRstd[row] = rstd


@jit_cache
def _compile_rms_final_reduce(dtype, N):
    batch_sym = cute.sym_int()
    div = math.gcd(N, 128 // dtype.width)
    x_cute = fake_tensor(dtype, (batch_sym, N), div)
    rstd_cute = fake_tensor(Float32, (batch_sym,))
    return cute.compile(
        RmsFinalReduce(dtype, N),
        x_cute,
        rstd_cute,
        Float32(0),  # scale
        Float32(0),  # eps
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _rms_final_reduce_out(
    x: Tensor,
    rstd: Tensor,
    scale: float,
    eps: float,
) -> None:
    """Compute rstd[m] = rsqrt(sum_n(x[m, n]) * scale + eps)."""
    x_dtype = torch2cute_dtype_map[x.dtype]
    N = x.shape[1]
    compiled_fn = _compile_rms_final_reduce(x_dtype, N)
    compiled_fn(x, rstd, scale, eps)


def rms_final_reduce(
    x: Tensor,  # (M, N) partial squared sums
    scale: float,  # typically 1.0 / total_columns
    eps: float = 1e-6,
) -> Tensor:
    """Compute rstd[m] = rsqrt(sum_n(x[m, n]) * scale + eps)."""
    assert x.ndim == 2
    M = x.shape[0]
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)

    from .cache_utils import COMPILE_ONLY

    if COMPILE_ONLY:
        return rstd

    _rms_final_reduce_out(x, rstd, scale, eps)
    return rstd
