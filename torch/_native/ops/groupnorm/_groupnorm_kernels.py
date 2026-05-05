"""GroupNorm CuTE DSL forward and backward kernels.

Computes row-wise LayerNorm (mean subtraction, variance normalization, affine)
on a [M, K] tensor where M = N * group and K = (C // group) * HxW.
Weight and bias are [C] tensors indexed by channel: c = g*cpg + k//HxW.
"""

# pyre-ignore-all-errors
# pyrefly: ignore-errors
# ruff: noqa: S101

import math
import operator
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, const_expr, Float16, Float32, Int32, Int64
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op, T

from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.copy_utils import copy, predicate_k
from quack.layout_utils import expand
from quack.reduce import row_reduce
from quack.reduction_base import ReductionBase
from quack.utils import fill_oob

import torch
from torch import Tensor


_TORCH2CUTE_DTYPE = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


@dsl_user_op
def atomic_add_f32(
    val: Float32,
    gmem_ptr: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """Atomically add a Float32 value to global memory.

    Not provided by quack (only int variants exist there); needed for the
    weight/bias gradient accumulation in the backward kernel.
    """
    from cutlass import CUDA_VERSION

    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
        return nvvm.atomicrmw(
            res=T.f32(),
            op=nvvm.AtomicOpKind.FADD,
            ptr=gmem_ptr.llvm_ptr,
            a=val.ir_value(loc=loc, ip=ip),
        )
    else:
        return nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.FADD,
            ptr=gmem_ptr.llvm_ptr,
            a=val.ir_value(loc=loc, ip=ip),
        )


@cute.jit
def _load_affine_param(
    mP: cute.Tensor,
    tXrP: cute.Tensor,
    tXcX: cute.Tensor,
    group_idx: Int32,
    cpg: cutlass.Constexpr[int],
    HxW: cutlass.Constexpr[int],
    K: Int32,
    cluster_y_offset: Int32,
):
    """Load [C] affine param into fragment register using channel indexing.

    tXrP has shape ((vec, rest_v), rest_m, rest_k).
    tXcX has shape (rest_v, rest_m, rest_k) — vec dimension already sliced out.
    Within a vec group, elements are at consecutive columns starting from
    the base column given by tXcX[rest_v, 0, rest_k][1].
    """
    vecsize = const_expr(tXrP.shape[0][0])
    zero = mP.element_type(0)
    for rest_v in cutlass.range_constexpr(tXrP.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXrP.shape[2]):
            base_col = cluster_y_offset + tXcX[rest_v, 0, rest_k][1]
            for vec_i in cutlass.range_constexpr(vecsize):
                col_k = base_col + vec_i
                if col_k < K:
                    c_idx = group_idx * cpg + col_k // HxW
                    tXrP[(vec_i, rest_v), 0, rest_k] = mP[c_idx]
                else:
                    tXrP[(vec_i, rest_v), 0, rest_k] = zero


@cute.jit
def _atomic_accumulate_dw_db(
    vals: cute.TensorSSA,
    mOut: cute.Tensor,
    tXcX: cute.Tensor,
    group_idx: Int32,
    cpg: cutlass.Constexpr[int],
    HxW: cutlass.Constexpr[int],
    K: Int32,
    cluster_y_offset: Int32,
    vec_size: cutlass.Constexpr[int],
    rest_v_size: cutlass.Constexpr[int],
    rest_k_size: cutlass.Constexpr[int],
    threads_per_row: cutlass.Constexpr[int],
):
    """Reduce over HxW locally per channel, then atomic-add cpg values.

    vals is a Float32 TensorSSA with the same element count as the fragment.
    mOut is [C] float32 global tensor.

    Instead of one atomic per element (K/num_threads atomics per thread),
    this first bins contributions by channel (cpg bins), reducing over the
    HxW spatial dimension, then emits only cpg atomics per thread.
    """
    # Per-channel accumulators: allocate as a register tensor of size cpg
    channel_acc = cute.make_rmem_tensor(cute.make_layout(cpg), Float32)
    channel_acc.fill(0.0)

    # Accumulate each fragment element into the appropriate channel bin
    for rest_v in cutlass.range_constexpr(rest_v_size):
        for rest_k in cutlass.range_constexpr(rest_k_size):
            base_col = cluster_y_offset + tXcX[rest_v, 0, rest_k][1]
            for vec_i in cutlass.range_constexpr(vec_size):
                col_k = base_col + vec_i
                flat_idx = (rest_v * rest_k_size + rest_k) * vec_size + vec_i
                val = vals[flat_idx]
                if col_k < K:
                    c_in_group = col_k // HxW
                    for c in cutlass.range_constexpr(cpg):
                        if c_in_group == c:
                            channel_acc[c] = channel_acc[c] + val

    # Warp-reduce only within the same row (threads_per_row lanes), then
    # one thread per row does the atomic. This avoids mixing groups that
    # share a warp when tiler_mn[0] > 1.
    c_base = group_idx * cpg
    for c in cutlass.range_constexpr(cpg):
        acc_val = cute.arch.warp_reduction(
            channel_acc[c], operator.add,
            threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
        )
        if cute.arch.lane_idx() % min(threads_per_row, cute.arch.WARP_SIZE) == 0:
            atomic_add_f32(acc_val, mOut.iterator + c_base + c)


class GroupNormFwd(ReductionBase):
    """Row-wise LayerNorm kernel for GroupNorm forward with fused affine."""

    def __init__(self, dtype: type[cutlass.Numeric], K: int, cpg: int, HxW: int):
        super().__init__(dtype, N=K, stage=1, reduction_dtype=Float32)
        self.K = K
        self.cpg = cpg
        self.HxW = HxW
        self.reload_from = None if K <= 8192 else "smem"

    def _threads_per_row(self):
        K = self.K
        for limit, threads in [
            (64, 8),
            (128, 16),
            (3072, 32),
            (6144, 64),
            (16384, 128),
        ]:
            if K <= limit:
                return threads
        return 256

    def _num_threads(self):
        return 128 if self.K <= 16384 else 256

    def _set_cluster_n(self):
        K = self.K
        if const_expr(self.dtype.width == 16):
            thresholds = [
                (16 * 1024, 1),
                (32 * 1024, 2),
                (64 * 1024, 4),
                (128 * 1024, 8),
            ]
        else:
            thresholds = [
                (32 * 1024, 1),
                (64 * 1024, 2),
                (128 * 1024, 4),
                (256 * 1024, 8),
            ]
        for limit, cluster in thresholds:
            if K <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mO: cute.Tensor,
        mMean: cute.Tensor | None,
        mRstd: cute.Tensor | None,
        eps: Float32,
        group: Int32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(*(t.element_type.width for t in [mX, mO] if t is not None))
        )
        vecsize = math.gcd(self.K, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        if const_expr(mMean is not None):
            mMean = expand(mMean, dim=1, size=self.K)
        if const_expr(mRstd is not None):
            mRstd = expand(mRstd, dim=1, size=self.K)
        self.kernel(
            mX,
            mW,
            mB,
            mO,
            mMean,
            mRstd,
            eps,
            group,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mO: cute.Tensor,
        mMean: cute.Tensor | None,
        mRstd: cute.Tensor | None,
        eps: Float32,
        group: Int32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = (
            const_expr(0)
            if const_expr(self.cluster_n == 1)
            else cute.arch.block_idx()[1]
        )
        tv_layout = tiled_copy.layout_tv_tiled

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, gO, gMean, gRstd, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) if mT is not None else None
            for mT in (mX, mO, mMean, mRstd, idX)
        ]

        thr_copy_X = tiled_copy.get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_X.partition_D(gO)
        tXrMean = (
            thr_copy_X.partition_D(gMean) if const_expr(mMean is not None) else None
        )
        tXrRstd = (
            thr_copy_X.partition_D(gRstd) if const_expr(mRstd is not None) else None
        )
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrX, tXrO = [cute.make_fragment_like(t) for t in (tXgX, tXgO)]
        tXrW = cute.make_fragment_like(tXgX) if const_expr(mW is not None) else None
        tXrB = cute.make_fragment_like(tXgX) if const_expr(mB is not None) else None

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_K = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_K
            else None
        )
        copy_ = partial(copy, pred=tXpX)

        row = tXcX[0][0]
        if row < shape[0]:
            copy_(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()

        # Load weight/bias from [C] using channel indexing while async copy runs
        cluster_y_offset = cluster_y * tiler_mn[1]
        if const_expr(mW is not None):
            group_idx = row % group
            _load_affine_param(
                mW, tXrW, tXcX, group_idx, self.cpg, self.HxW, shape[1],
                cluster_y_offset,
            )
        if const_expr(mB is not None):
            group_idx = row % group
            _load_affine_param(
                mB, tXrB, tXcX, group_idx, self.cpg, self.HxW, shape[1],
                cluster_y_offset,
            )

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)

        # Compute mean via row reduction
        sum_x = row_reduce(
            x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )
        mean_val = sum_x / shape[1]

        # Compute variance as E[x^2] - mean^2.
        # OOB positions are zero (from predicated cp.async), so x^2 = 0 contributes
        # nothing, while (x-mean)^2 would incorrectly contribute mean^2.
        if const_expr(self.cluster_n > 1):
            cute.arch.cluster_wait()

        sum_x_sq = row_reduce(
            x * x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )
        rstd = cute.math.rsqrt(
            sum_x_sq / shape[1] - mean_val * mean_val + eps, fastmath=True
        )

        # Store mean and rstd statistics
        if const_expr(mMean is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrMean[0] = mean_val
        if const_expr(mRstd is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = rstd

        # Reload x from smem if needed (large K)
        if const_expr(self.reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)

        # Normalize and apply affine: y = (x - mean) * rstd * w + b
        y = (x - mean_val) * rstd
        if const_expr(mW is not None):
            y *= tXrW.load().to(cute.Float32)
        if const_expr(mB is not None):
            y += tXrB.load().to(cute.Float32)
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy_(tXrO, tXgO)


def _groupnorm_fwd(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    out: Tensor,
    mean: Tensor,
    rstd: Tensor,
    cpg: int,
    HxW: int,
    group: int,
    eps: float = 1e-5,
) -> None:
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_types, "Unsupported dtype"
    assert x.dim() == 2, "Input must be 2D [M, K]"

    _, K = x.shape
    C = cpg * group
    dtype = _TORCH2CUTE_DTYPE[x.dtype]
    out_dtype = _TORCH2CUTE_DTYPE[out.dtype]
    weight_dtype = _TORCH2CUTE_DTYPE[weight.dtype] if weight is not None else None
    bias_dtype = _TORCH2CUTE_DTYPE[bias.dtype] if bias is not None else None
    compile_key = (dtype, out_dtype, weight_dtype, bias_dtype, K, cpg, HxW, group)
    if compile_key not in _groupnorm_fwd.compile_cache:
        batch_sym = cute.sym_int()
        all_dtypes = [dtype, out_dtype]
        div = math.gcd(K, *(128 // dt.width for dt in all_dtypes))
        x_cute, out_cute = [
            fake_tensor(dt, (batch_sym, K), div) for dt in [dtype, out_dtype]
        ]
        weight_cute = fake_tensor(weight_dtype, (C,)) if weight_dtype else None
        bias_cute = fake_tensor(bias_dtype, (C,)) if bias_dtype else None
        mean_cute = fake_tensor(Float32, (batch_sym,))
        rstd_cute = fake_tensor(Float32, (batch_sym,))
        _groupnorm_fwd.compile_cache[compile_key] = cute.compile(
            GroupNormFwd(dtype, K, cpg, HxW),
            x_cute,
            weight_cute,
            bias_cute,
            out_cute,
            mean_cute,
            rstd_cute,
            Float32(0),
            group,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _groupnorm_fwd.compile_cache[compile_key](
        x, weight, bias, out, mean, rstd, eps, group,
    )


_groupnorm_fwd.compile_cache = {}


class GroupNormBackward(ReductionBase):
    """Persistent backward kernel for GroupNorm on [M, K] tensors.

    Like RMSNormBackward but with an extra mean-subtraction correction term.
    dx = (wdy - x_hat * mean(x_hat * wdy) - mean(wdy)) * rstd
    where x_hat = (x - mean) * rstd.
    """

    def __init__(self, dtype: type[cutlass.Numeric], K: int, cpg: int, HxW: int):
        super().__init__(dtype, N=K, stage=2, reduction_dtype=Float32)
        self.K = K
        self.cpg = cpg
        self.HxW = HxW
        self.reload_wdy = None if K <= 16 * 1024 else "smem"
        if self.K > 128 * 1024 and self.dtype.width >= 32:
            raise ValueError(
                "GroupNormBackward does not support K > 128k with dtype >= 32 bits"
            )

    def _num_threads(self):
        return 128 if self.K <= 4096 else 256

    def _threads_per_row(self):
        K = self.K
        for limit, threads in [(64, 8), (128, 16), (256, 32), (512, 64), (4096, 128)]:
            if K <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        K = self.K
        for limit, cluster in [
            (8 * 1024, 1),
            (16 * 1024, 2),
            (32 * 1024, 4),
            (64 * 1024, 8),
        ]:
            if K <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mdO: cute.Tensor,
        mMean: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: cute.Tensor | None,
        mdB: cute.Tensor | None,
        sm_count: Int32,
        group: Int32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(*(t.element_type.width for t in [mX, mdO, mdX] if t is not None))
        )
        vecsize = math.gcd(self.K, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        num_blocks = sm_count
        self.kernel(
            mX,
            mW,
            mdO,
            mMean,
            mRstd,
            mdX,
            mdW,
            mdB,
            group,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[num_blocks, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mdO: cute.Tensor,
        mMean: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: cute.Tensor | None,
        mdB: cute.Tensor | None,
        group: Int32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_start, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        cluster_y = (
            const_expr(0)
            if const_expr(self.cluster_n == 1)
            else cute.arch.block_idx()[1]
        )
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        M, _K = shape[0], shape[1]
        is_even_K = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)

        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_ordered_layout(
            (tiler_mn[0], tiler_mn[1], 2), order=(1, 0, 2)
        )
        sX = smem.allocate_tensor(mX.element_type, smem_layout, byte_alignment=16)
        sdO = smem.allocate_tensor(mdO.element_type, smem_layout, byte_alignment=16)
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout, is_persistent=True,
        )
        if const_expr(mbar_ptr is not None):
            mbar_full_ptr, mbar_empty_ptr = mbar_ptr, mbar_ptr + 2
        else:
            mbar_full_ptr, mbar_empty_ptr = None, None

        thr_copy_X = tiled_copy.get_slice(tidx)

        gX, gdO, gdX, cX = [
            cute.local_tile(mT, tiler_mn, (None, cluster_y))
            for mT in (mX, mdO, mdX, idX)
        ]

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgdO = thr_copy_X.partition_S(gdO)
        tXsdO = thr_copy_X.partition_D(sdO)
        tXgdX = thr_copy_X.partition_D(gdX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None, None]

        tXrX, tXrdO, tXrdX = [
            cute.make_fragment_like(thr[None, None, None, 0])
            for thr in (tXgX, tXgdO, tXgdX)
        ]

        tXpX = (
            None
            if is_even_K
            else predicate_k(thr_copy_X.partition_S(cX[None, None, 0]), limit=shape[1])
        )
        copy_ = partial(copy, pred=tXpX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE

        self._initialize_cluster(tidx, mbar_ptr, num_warps, is_persistent=True)

        # Weight register fragment (loaded from [C] per row)
        tXrW = None
        if const_expr(mW is not None):
            tXrW = cute.make_fragment_like(tXrX)

        cluster_y_offset = cluster_y * tiler_mn[1]

        # Prefetch the first batch
        row = tXcX[None, None, None, bidx_start][0][0]
        if row < M:
            copy_(
                tXgX[None, None, None, bidx_start],
                tXsX[None, None, None, 0],
                is_async=True,
            )
            copy_(
                tXgdO[None, None, None, bidx_start],
                tXsdO[None, None, None, 0],
                is_async=True,
            )
        else:
            if const_expr(tiler_mn[0] > 1):
                fill_oob(
                    tXsX[None, None, None, 0], None, fill_value=mX.element_type.zero
                )
                fill_oob(
                    tXsdO[None, None, None, 0], None, fill_value=mdO.element_type.zero
                )
        cute.arch.cp_async_commit_group()

        if const_expr(self.cluster_n > 1):
            cute.arch.cluster_wait()

        stage = Int32(0)
        producer_phase = Int32(1)
        consumer_phase = Int32(0)
        for bidx in cutlass.range(bidx_start, cute.ceil_div(M, tiler_mn[0]), gdim):
            row = tXcX[None, None, None, bidx][0][0]
            if row + gdim * tiler_mn[0] < M:
                copy_(
                    tXgX[None, None, None, bidx + gdim],
                    tXsX[None, None, None, stage ^ 1],
                    is_async=True,
                )
                copy_(
                    tXgdO[None, None, None, bidx + gdim],
                    tXsdO[None, None, None, stage ^ 1],
                    is_async=True,
                )
            else:
                if const_expr(tiler_mn[0] > 1):
                    fill_oob(
                        tXsX[None, None, None, stage ^ 1],
                        None,
                        fill_value=mX.element_type.zero,
                    )
                    fill_oob(
                        tXsdO[None, None, None, stage ^ 1],
                        None,
                        fill_value=mdO.element_type.zero,
                    )
            cute.arch.cp_async_commit_group()
            mean_val = cutlass.Float.zero
            rstd = cutlass.Float.zero
            if row < M or tiler_mn[0] == 1:
                mean_val = mMean[row]
                rstd = mRstd[row]

            # Load weight from [C] using channel indexing
            if const_expr(mW is not None):
                group_idx = row % group
                _load_affine_param(
                    mW, tXrW, tXcX[None, None, None, bidx],
                    group_idx, self.cpg, self.HxW, shape[1],
                    cluster_y_offset,
                )

            cute.arch.cp_async_wait_group(1)
            cute.autovec_copy(tXsX[None, None, None, stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
            dout = tXrdO.load().to(cute.Float32)
            x_hat = (x - mean_val) * rstd
            wdy = dout
            if const_expr(mW is not None):
                wdy *= tXrW.load().to(Float32)

            # Two reductions needed: mean(wdy) and mean(x_hat * wdy)
            # Use mbar protocol for first, cluster_wait between, mbar for second
            if const_expr(self.cluster_n > 1):
                cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)
            mean_wdy = (
                row_reduce(
                    wdy,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, stage],
                    (mbar_full_ptr + stage if const_expr(self.cluster_n > 1) else None),
                    phase=consumer_phase,
                    init_val=0.0,
                )
                / shape[1]
            )

            if const_expr(self.cluster_n > 1):
                cute.arch.cluster_wait()

            mean_xhat_wdy = (
                row_reduce(
                    x_hat * wdy,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, stage],
                    (mbar_full_ptr + stage if const_expr(self.cluster_n > 1) else None),
                    phase=consumer_phase,
                    init_val=0.0,
                    hook_fn=cute.arch.cluster_wait
                    if const_expr(self.cluster_n > 1)
                    else None,
                )
                / shape[1]
            )

            # Signal buffer is free for next iteration
            if const_expr(self.cluster_n > 1):
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                lane_idx = cute.arch.lane_idx()
                if lane_idx < self.cluster_n:
                    cute.arch.mbarrier_arrive(
                        mbar_empty_ptr + stage, peer_cta_rank_in_cluster=lane_idx
                    )

            if const_expr(self.reload_wdy == "smem"):
                cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
                dout = tXrdO.load().to(cute.Float32)
                wdy = dout
                if const_expr(mW is not None):
                    _load_affine_param(
                        mW, tXrW, tXcX[None, None, None, bidx],
                        group_idx, self.cpg, self.HxW, shape[1],
                        cluster_y_offset,
                    )
                    wdy *= tXrW.load().to(Float32)

            dx = (wdy - x_hat * mean_xhat_wdy - mean_wdy) * rstd
            tXrdX.store(dx.to(tXrdX.element_type))
            if row < M or tiler_mn[0] == 1:
                copy_(tXrdX, tXgdX[None, None, None, bidx])
            if row < M:
                if const_expr(mdW is not None):
                    _atomic_accumulate_dw_db(
                        dout * x_hat, mdW, tXcX[None, None, None, bidx],
                        group_idx, self.cpg, self.HxW, shape[1],
                        cluster_y_offset,
                        const_expr(tXrdX.shape[0][0]),
                        const_expr(tXrdX.shape[0][1]),
                        const_expr(tXrdX.shape[2]),
                        threads_per_row,
                    )
                if const_expr(mdB is not None):
                    _atomic_accumulate_dw_db(
                        dout, mdB, tXcX[None, None, None, bidx],
                        group_idx, self.cpg, self.HxW, shape[1],
                        cluster_y_offset,
                        const_expr(tXrdX.shape[0][0]),
                        const_expr(tXrdX.shape[0][1]),
                        const_expr(tXrdX.shape[2]),
                        threads_per_row,
                    )

            stage ^= 1
            if stage == 0:
                consumer_phase ^= 1
                producer_phase ^= 1

        if const_expr(self.cluster_n > 1):
            stage ^= 1
            if stage == 0:
                producer_phase ^= 1
            cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)


def _get_sm_count(K: int, device: torch.device) -> int:
    sm_count_multiple = (
        16
        if K <= 256
        else (8 if K <= 1024 else (4 if K <= 2048 else (2 if K <= 4096 else 1)))
    )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    sm_count = (
        sm_count * sm_count_multiple
        if K <= 8192
        else sm_count // 2
        if K <= 16384
        else sm_count * 2
    )
    return sm_count


def _groupnorm_bwd(
    x: Tensor,
    weight: Tensor | None,
    dout: Tensor,
    mean: Tensor,
    rstd: Tensor,
    dx: Tensor,
    dw: Tensor | None,
    db: Tensor | None = None,
    sm_count: int | None = None,
    cpg: int = 1,
    HxW: int = 1,
    group: int = 1,
) -> None:
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_types, "Unsupported dtype"
    if weight is not None:
        assert weight.is_cuda, "Weight tensor must be on CUDA device"
        assert weight.dtype in supported_types

    K = x.size(1)
    C = cpg * group
    assert sm_count is not None
    dtype, dout_dtype, dx_dtype, weight_dtype = [
        _TORCH2CUTE_DTYPE[t.dtype] if t is not None else None
        for t in [x, dout, dx, weight]
    ]
    compile_key = (
        K,
        cpg,
        HxW,
        group,
        dtype,
        dout_dtype,
        dx_dtype,
        weight_dtype,
        dw is not None,
        db is not None,
    )
    if compile_key not in _groupnorm_bwd.compile_cache:
        batch_sym = cute.sym_int()
        all_dtypes = [dtype, dout_dtype, dx_dtype]
        div = math.gcd(K, *(128 // dt.width for dt in all_dtypes if dt is not None))
        x_cute, dout_cute, dx_cute = [
            fake_tensor(dt, (batch_sym, K), div) for dt in [dtype, dout_dtype, dx_dtype]
        ]
        weight_cute = fake_tensor(weight_dtype, (C,)) if weight_dtype else None
        mean_cute = fake_tensor(Float32, (batch_sym,))
        rstd_cute = fake_tensor(Float32, (batch_sym,))
        dw_cute = fake_tensor(Float32, (C,)) if dw is not None else None
        db_cute = fake_tensor(Float32, (C,)) if db is not None else None
        _groupnorm_bwd.compile_cache[compile_key] = cute.compile(
            GroupNormBackward(dtype, K, cpg, HxW),
            x_cute,
            weight_cute,
            dout_cute,
            mean_cute,
            rstd_cute,
            dx_cute,
            dw_cute,
            db_cute,
            sm_count,
            group,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _groupnorm_bwd.compile_cache[compile_key](
        x, weight, dout, mean, rstd, dx, dw, db, sm_count, group,
    )


_groupnorm_bwd.compile_cache = {}
