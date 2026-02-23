"""
Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
RMSNorm CuTE DSL kernel classes from quack
"""
# pyre-ignore-all-errors
# pyrefly: ignore-errors
# ruff: noqa: S101

import math
from functools import partial
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, const_expr, Float16, Float32, Int32, Int64

import torch
from torch import Tensor

from ._cute_utils import (
    allocate_reduction_buffer_and_mbar,
    copy,
    expand,
    fill_oob,
    get_tiled_copy,
    initialize_cluster,
    make_fake_tensor as fake_tensor,
    predicate_k,
    row_reduce,
)


_TORCH2CUTE_DTYPE = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


class RMSNorm:
    def __init__(self, dtype: type[cutlass.Numeric], N: int):
        self.dtype = dtype
        self.N = N
        self.stage = 1
        self.reduction_dtype = Float32
        self.reload_from = None if N <= 8192 else "smem"
        self.delay_w_load = False

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [
            (64, 8),
            (128, 16),
            (3072, 32),
            (6144, 64),
            (16384, 128),
        ]:
            if N <= limit:
                return threads
        return 256

    def _num_threads(self):
        return 128 if self.N <= 16384 else 256

    def _set_cluster_n(self):
        N = self.N
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
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mB: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        eps: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(
                *(
                    t.element_type.width
                    for t in [mX, mRes, mW, mB, mO, mResO]
                    if t is not None
                )
            )
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = get_tiled_copy(
            self.dtype,
            self.N,
            self.cluster_n,
            self._threads_per_row(),
            self._num_threads(),
            vecsize=vecsize,
        )
        num_threads = tiled_copy.size
        mW, mB = [
            expand(mT, dim=0, size=tiler_mn[0]) if const_expr(mT is not None) else None
            for mT in (mW, mB)
        ]
        if const_expr(mRstd is not None):
            mRstd = expand(mRstd, dim=1, size=self.N)
        self.kernel(
            mX,
            mW,
            mB,
            mRes,
            mO,
            mResO,
            mRstd,
            eps,
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
        mW: Optional[cute.Tensor],
        mB: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        eps: Float32,
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
        if const_expr(mRes is not None):
            sRes = smem.allocate_tensor(
                mRes.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
        reduction_buffer, mbar_ptr = allocate_reduction_buffer_and_mbar(
            smem, self.reduction_dtype, self.stage, self.cluster_n, tv_layout
        )

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, gRes, gO, gResO, gRstd, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) if mT is not None else None
            for mT in (mX, mRes, mO, mResO, mRstd, idX)
        ]
        gW, gB = [
            cute.local_tile(mT, tiler_mn, (0, cluster_y))
            if const_expr(mT is not None)
            else None
            for mT in (mW, mB)
        ]

        thr_copy_X = tiled_copy.get_slice(tidx)

        tXgW = thr_copy_X.partition_S(gW) if const_expr(mW is not None) else None
        tXgB = thr_copy_X.partition_S(gB) if const_expr(mB is not None) else None
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        if const_expr(mRes is not None):
            tXgRes = thr_copy_X.partition_S(gRes)
            tXsRes = thr_copy_X.partition_D(sRes)
        tXgO = thr_copy_X.partition_D(gO)
        if const_expr(mResO is not None):
            tXgResO = thr_copy_X.partition_D(gResO)
        tXrRstd = (
            thr_copy_X.partition_D(gRstd) if const_expr(mRstd is not None) else None
        )
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrW = cute.make_fragment_like(tXgW) if const_expr(mW is not None) else None
        tXrB = cute.make_fragment_like(tXgB) if const_expr(mB is not None) else None
        tXrX, tXrO = [cute.make_fragment_like(t) for t in (tXgX, tXgO)]
        if const_expr(mRes is not None):
            tXrRes = cute.make_fragment_like(tXgRes)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        initialize_cluster(tidx, mbar_ptr, num_warps, self.cluster_n, self.stage)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        copy_ = partial(copy, pred=tXpX)

        row = tXcX[0][0]
        if row < shape[0]:
            copy_(tXgX, tXsX, is_async=True)
            if const_expr(mRes is not None):
                copy_(tXgRes, tXsRes, is_async=True)
        cute.arch.cp_async_commit_group()

        if const_expr(not self.delay_w_load):
            if const_expr(mW is not None):
                copy_(tXgW, tXrW)
            if const_expr(mB is not None):
                copy_(tXgB, tXrB)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        if const_expr(mRes is not None):
            cute.autovec_copy(tXsRes, tXrRes)
            x += tXrRes.load().to(cute.Float32)
        if const_expr(mResO is not None):
            tXrResO = cute.make_fragment_like(tXgResO)
            tXrResO.store(x.to(tXrResO.element_type))
            if row < shape[0]:
                copy_(tXrResO, tXgResO)

        sum_sq_x = row_reduce(
            x * x,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )
        rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=True)
        if const_expr(mRstd is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = rstd
        if const_expr(self.delay_w_load):
            if const_expr(mW is not None):
                copy_(tXgW, tXrW)
            if const_expr(mB is not None):
                copy_(tXgB, tXrB)
        if const_expr(self.reload_from == "smem" or self.reload_from == "gmem"):
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                if const_expr(mRes is not None):
                    cute.autovec_copy(tXsRes, tXrRes)
            else:
                copy_(tXgX, tXrX)
                if const_expr(mRes is not None):
                    copy_(tXgRes, tXrRes)
            x = tXrX.load().to(cute.Float32)
            if const_expr(mRes is not None):
                x += tXrRes.load().to(cute.Float32)
        x_hat = x * rstd
        y = x_hat
        if const_expr(mW is not None):
            y *= tXrW.load().to(cute.Float32)
        if const_expr(mB is not None):
            y += tXrB.load().to(cute.Float32)
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy_(tXrO, tXgO)


_FWD_SCHEMA = (
    "(Tensor x, Tensor? weight, Tensor(a2!) out, Tensor? bias,"
    " Tensor(a4!)? rstd, Tensor? residual,"
    " Tensor(a6!)? residual_out, float eps=1e-6) -> ()"
)


@torch.library.custom_op(
    "cutedsl_norms::_rmsnorm_fwd",
    mutates_args=("out", "rstd", "residual_out"),
    device_types="cuda",
    schema=_FWD_SCHEMA,
)
def _rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor],
    out: Tensor,
    bias: Optional[Tensor] = None,
    rstd: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> None:
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_types, "Unsupported dtype"
    if weight is not None:
        assert weight.dtype in supported_types, (
            "Weight must be float32, float16 or bfloat16"
        )
    if residual is not None:
        assert residual.dtype in supported_types, (
            "Residual must be float16, bfloat16, or float32"
        )

    _, N = x.shape
    dtype, out_dtype, weight_dtype, bias_dtype, res_dtype, res_out_dtype = [
        _TORCH2CUTE_DTYPE[t.dtype] if t is not None else None
        for t in [x, out, weight, bias, residual, residual_out]
    ]
    compile_key = (
        dtype,
        out_dtype,
        res_dtype,
        weight_dtype,
        bias_dtype,
        res_out_dtype,
        N,
        rstd is not None,
    )
    if compile_key not in _rmsnorm_fwd.compile_cache:
        batch_sym = cute.sym_int()
        all_dtypes = [
            dtype,
            out_dtype,
            res_dtype,
            weight_dtype,
            bias_dtype,
            res_out_dtype,
        ]
        div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
        x_cute, out_cute, res_cute, res_out_cute = [
            fake_tensor(dt, (batch_sym, N), div)
            for dt in [dtype, out_dtype, res_dtype, res_out_dtype]
        ]
        weight_cute, bias_cute = [
            fake_tensor(dt, (N,), div) for dt in [weight_dtype, bias_dtype]
        ]
        rstd_cute = fake_tensor(Float32, (batch_sym,)) if rstd is not None else None
        _rmsnorm_fwd.compile_cache[compile_key] = cute.compile(
            RMSNorm(dtype, N),
            x_cute,
            weight_cute,
            bias_cute,
            res_cute,
            out_cute,
            res_out_cute,
            rstd_cute,
            Float32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _rmsnorm_fwd.compile_cache[compile_key](
        x, weight, bias, residual, out, residual_out, rstd, eps
    )


_rmsnorm_fwd.compile_cache = {}


class RMSNormBackward:
    def __init__(self, dtype: cutlass.Numeric, N: int):
        self.dtype = dtype
        self.N = N
        self.stage = 2
        self.reduction_dtype = Float32
        self.reload_wdy = None if N <= 16 * 1024 else "smem"
        if self.N > 128 * 1024 and self.dtype.width >= 32:
            raise ValueError(
                "RMSNormBackward does not support N > 128k with dtype >= 32 bits"
            )

    def _num_threads(self):
        return 128 if self.N <= 4096 else 256

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (256, 32), (512, 64), (4096, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        N = self.N
        for limit, cluster in [
            (8 * 1024, 1),
            (16 * 1024, 2),
            (32 * 1024, 4),
            (64 * 1024, 8),
        ]:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mdResO: Optional[cute.Tensor],
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: Optional[cute.Tensor],
        mdRes: Optional[cute.Tensor],
        mdB: Optional[cute.Tensor],
        sm_count: Int32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(
                *(
                    t.element_type.width
                    for t in [mX, mW, mdO, mdResO, mdX, mdRes]
                    if t is not None
                )
            )
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = get_tiled_copy(
            self.dtype,
            self.N,
            self.cluster_n,
            self._threads_per_row(),
            self._num_threads(),
            vecsize=vecsize,
        )
        num_threads = tiled_copy.size
        mW = expand(mW, dim=0, size=tiler_mn[0]) if const_expr(mW is not None) else None
        num_blocks = sm_count
        self.kernel(
            mX,
            mW,
            mdO,
            mdResO,
            mRstd,
            mdX,
            mdW,
            mdB,
            mdRes,
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
        mW: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mdResO: Optional[cute.Tensor],
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: Optional[cute.Tensor],
        mdB: Optional[cute.Tensor],
        mdRes: Optional[cute.Tensor],
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
        M, _N = shape[0], shape[1]
        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)

        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_ordered_layout(
            (tiler_mn[0], tiler_mn[1], 2), order=(1, 0, 2)
        )
        sX = smem.allocate_tensor(mX.element_type, smem_layout, byte_alignment=16)
        sdO = smem.allocate_tensor(mdO.element_type, smem_layout, byte_alignment=16)
        reduction_buffer, mbar_ptr = allocate_reduction_buffer_and_mbar(
            smem,
            self.reduction_dtype,
            self.stage,
            self.cluster_n,
            tv_layout,
            is_persistent=True,
        )
        if const_expr(mbar_ptr is not None):
            mbar_full_ptr, mbar_empty_ptr = mbar_ptr, mbar_ptr + 2
        else:
            mbar_full_ptr, mbar_empty_ptr = None, None

        thr_copy_X = tiled_copy.get_slice(tidx)

        gX, gdO, gdResO, gdX, gdRes, cX = [
            cute.local_tile(mT, tiler_mn, (None, cluster_y)) if mT is not None else None
            for mT in (mX, mdO, mdResO, mdX, mdRes, idX)
        ]
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y)) if mW is not None else None
        gdW, gdB = [
            cute.local_tile(mT, (1, tiler_mn[1]), (bidx_start, cluster_y))
            if const_expr(mT is not None)
            else None
            for mT in (mdW, mdB)
        ]

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgdO = thr_copy_X.partition_S(gdO)
        tXsdO = thr_copy_X.partition_D(sdO)
        tXgdX = thr_copy_X.partition_D(gdX)
        if const_expr(mdResO is not None):
            tXgdResO = thr_copy_X.partition_S(gdResO)
        if const_expr(mdRes is not None):
            tXgdRes = thr_copy_X.partition_D(gdRes)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None, None]

        tXrX, tXrdO, tXrdX = [
            cute.make_fragment_like(thr[None, None, None, 0])
            for thr in (tXgX, tXgdO, tXgdX)
        ]
        tXrdResO = None
        if const_expr(mdResO is not None):
            tXrdResO = cute.make_fragment_like(tXgdResO[None, None, None, 0])
        tXrdRes = None
        if const_expr(mdRes is not None):
            tXrdRes = cute.make_fragment_like(tXgdRes[None, None, None, 0])

        tXpX = (
            None
            if is_even_N
            else predicate_k(thr_copy_X.partition_S(cX[None, None, 0]), limit=shape[1])
        )
        copy_ = partial(copy, pred=tXpX)

        tXgdW, tXrdW = None, None
        tXgdB, tXrdB = None, None
        if const_expr(mdW is not None):
            tXgdW = thr_copy_X.partition_S(gdW)
            tXrdW = cute.make_fragment_like(tXgdW, Float32)
        if const_expr(mdB is not None):
            tXgdB = thr_copy_X.partition_S(gdB)
            tXrdB = cute.make_fragment_like(tXgdB, Float32)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE

        initialize_cluster(
            tidx,
            mbar_ptr,
            num_warps,
            self.cluster_n,
            self.stage,
            is_persistent=True,
        )

        tXrW = None
        if const_expr(mW is not None):
            tXgW = thr_copy_X.partition_S(gW)
            tXrW = cute.make_fragment_like(tXgW)
            if const_expr(not is_even_N):
                tXrW.fill(0.0)
            copy_(tXgW, tXrW)

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

        if const_expr(mdW is not None):
            tXrdW.fill(0.0)
        if const_expr(mdB is not None):
            tXrdB.fill(0.0)
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
            rstd = cutlass.Float.zero
            if row < M or tiler_mn[0] == 1:
                rstd = mRstd[row]
            if const_expr(mdResO is not None):
                if row < M or tiler_mn[0] == 1:
                    copy_(tXgdResO[None, None, None, bidx], tXrdResO)
                elif tiler_mn[0] > 1:
                    tXrdResO.fill(0.0)
            cute.arch.cp_async_wait_group(1)
            cute.autovec_copy(tXsX[None, None, None, stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
            dout = tXrdO.load().to(cute.Float32)
            x_hat = x * rstd
            wdy = dout
            if const_expr(mW is not None):
                wdy *= tXrW.load().to(Float32)
            if const_expr(self.cluster_n > 1):
                cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)
            mean_xhat_wdy = (
                row_reduce(
                    x_hat * wdy,
                    threads_per_row,
                    reduction_buffer[None, None, stage],
                    (mbar_full_ptr + stage if const_expr(self.cluster_n > 1) else None),
                    phase=consumer_phase,
                    init_val=0.0,
                )
                / shape[1]
            )

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
                    wdy *= tXrW.load().to(Float32)

            dx = (wdy - x_hat * mean_xhat_wdy) * rstd
            if const_expr(mdResO is not None):
                dx += tXrdResO.load().to(cute.Float32)
            tXrdX.store(dx.to(tXrdX.element_type))
            if row < M or tiler_mn[0] == 1:
                copy_(tXrdX, tXgdX[None, None, None, bidx])
            if const_expr(mdRes is not None):
                tXrdRes.store(dx.to(tXrdRes.element_type))
                if row < M or tiler_mn[0] == 1:
                    copy_(tXrdRes, tXgdRes[None, None, None, bidx])
            if const_expr(mdW is not None):
                tXrdW.store(tXrdW.load() + dout * x_hat)
            if const_expr(mdB is not None):
                tXrdB.store(tXrdB.load() + dout)

            stage ^= 1
            if stage == 0:
                consumer_phase ^= 1
                producer_phase ^= 1

        if const_expr(tiler_mn[0] > 1):
            if const_expr(mdW is not None):
                sdW = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdW = thr_copy_X.partition_D(sdW)
                cute.arch.barrier()
                row = tXcX[None, None, None, 0][0][0]
                if row > 0:
                    cute.autovec_copy(tXrdW, tXsdW)
                cute.arch.barrier()
                if row == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdW_other = cute.make_fragment_like(tXrdW)
                        tXsdW_other = cute.make_tensor(
                            tXsdW.iterator + i * sdW.stride[0], tXsdW.layout
                        )
                        cute.autovec_copy(tXsdW_other, tXrdW_other)
                        tXrdW.store(tXrdW.load() + tXrdW_other.load())
                    copy_(tXrdW, tXgdW)
                cute.arch.barrier()
            if const_expr(mdB is not None):
                sdB = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdB = thr_copy_X.partition_D(sdB)
                cute.arch.barrier()
                row = tXcX[None, None, None, 0][0][0]
                if row > 0:
                    cute.autovec_copy(tXrdB, tXsdB)
                cute.arch.barrier()
                if row == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdB_other = cute.make_fragment_like(tXrdB)
                        tXsdB_other = cute.make_tensor(
                            tXsdB.iterator + i * sdB.stride[0], tXsdB.layout
                        )
                        cute.autovec_copy(tXsdB_other, tXrdB_other)
                        tXrdB.store(tXrdB.load() + tXrdB_other.load())
                    copy_(tXrdB, tXgdB)
        else:
            if const_expr(mdW is not None):
                copy_(tXrdW, tXgdW)
            if const_expr(mdB is not None):
                copy_(tXrdB, tXgdB)

        if const_expr(self.cluster_n > 1):
            stage ^= 1
            if stage == 0:
                producer_phase ^= 1
            cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)


def _get_sm_count(N: int, device: torch.device) -> int:
    sm_count_multiple = (
        16
        if N <= 256
        else (8 if N <= 1024 else (4 if N <= 2048 else (2 if N <= 4096 else 1)))
    )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    sm_count = (
        sm_count * sm_count_multiple
        if N <= 8192
        else sm_count // 2
        if N <= 16384
        else sm_count * 2
    )
    return sm_count


_BWD_SCHEMA = (
    "(Tensor x, Tensor? weight, Tensor dout, Tensor rstd,"
    " Tensor(a4!) dx, Tensor(a5!)? dw_partial,"
    " Tensor(a6!)? db_partial, Tensor? dresidual_out,"
    " Tensor(a8!)? dresidual, int? sm_count) -> ()"
)


@torch.library.custom_op(
    "cutedsl_norms::_rmsnorm_bwd",
    mutates_args={"dx", "dw_partial", "db_partial", "dresidual"},
    device_types="cuda",
    schema=_BWD_SCHEMA,
)
def _rmsnorm_bwd(
    x: Tensor,
    weight: Optional[Tensor],
    dout: Tensor,
    rstd: Tensor,
    dx: Tensor,
    dw_partial: Optional[Tensor],
    db_partial: Optional[Tensor] = None,
    dresidual_out: Optional[Tensor] = None,
    dresidual: Optional[Tensor] = None,
    sm_count: Optional[int] = None,
) -> None:
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_types, "Unsupported dtype"
    if weight is not None:
        assert weight.dim() == 1, "Weight must be 1D"
        assert x.shape[-1] == weight.shape[0], (
            "Last dimension of input must match weight dimension"
        )
        assert weight.is_cuda, "Weight tensor must be on CUDA device"
        assert weight.dtype in supported_types, (
            "Weight must be float32, float16 or bfloat16"
        )
    if dresidual_out is not None:
        assert dresidual_out.shape == x.shape
        assert dresidual_out.is_cuda
        assert dresidual_out.dtype in supported_types, (
            "Residual must be float16, bfloat16, or float32"
        )
    if dresidual is not None:
        assert dresidual.shape == x.shape
        assert dresidual.is_cuda
        assert dresidual.dtype in supported_types, (
            "Residual must be float16, bfloat16, or float32"
        )

    N = x.size(1)
    if dw_partial is None and db_partial is None:
        assert sm_count is not None
    else:
        sm_count = (
            dw_partial.shape[0] if dw_partial is not None else db_partial.shape[0]
        )
    dtype, dout_dtype, dx_dtype, weight_dtype, dres_dtype, dres_out_dtype = [
        _TORCH2CUTE_DTYPE[t.dtype] if t is not None else None
        for t in [x, dout, dx, weight, dresidual, dresidual_out]
    ]
    compile_key = (
        N,
        dtype,
        dout_dtype,
        dx_dtype,
        weight_dtype,
        db_partial is not None,
        dres_dtype,
        dres_out_dtype,
    )
    if compile_key not in _rmsnorm_bwd.compile_cache:
        batch_sym, batch_partial_sym = cute.sym_int(), cute.sym_int()
        all_dtypes = [dtype, dout_dtype, dx_dtype, dres_dtype, dres_out_dtype]
        div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
        x_cute, dout_cute, dx_cute, dres_out_cute, dres_cute = [
            fake_tensor(dt, (batch_sym, N), div)
            for dt in [dtype, dout_dtype, dx_dtype, dres_out_dtype, dres_dtype]
        ]
        weight_cute = fake_tensor(weight_dtype, (N,), div)
        rstd_cute = fake_tensor(Float32, (batch_sym,))
        dw_partial_cute = (
            fake_tensor(Float32, (batch_partial_sym, N), div)
            if dw_partial is not None
            else None
        )
        db_partial_cute = (
            fake_tensor(Float32, (batch_partial_sym, N), div)
            if db_partial is not None
            else None
        )
        _rmsnorm_bwd.compile_cache[compile_key] = cute.compile(
            RMSNormBackward(dtype, N),
            x_cute,
            weight_cute,
            dout_cute,
            dres_out_cute,
            rstd_cute,
            dx_cute,
            dw_partial_cute,
            dres_cute,
            db_partial_cute,
            sm_count,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _rmsnorm_bwd.compile_cache[compile_key](
        x,
        weight,
        dout,
        dresidual_out,
        rstd,
        dx,
        dw_partial,
        dresidual,
        db_partial,
        sm_count,
    )


_rmsnorm_bwd.compile_cache = {}
