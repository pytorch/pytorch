# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import math
from typing import Optional, Tuple, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

import torch
from torch import Tensor

from . import utils
from . import copy_utils
from . import layout_utils
from .compile_utils import make_fake_tensor as fake_tensor
from .reduce import row_reduce
from .reduction_base import ReductionBase
from .cache_utils import jit_cache
from .cute_dsl_utils import torch2cute_dtype_map
from cutlass.base_dsl.arch import Arch


def _ensure_contiguous(t):
    """Ensure last-dim stride is 1. Under torch.compile use unconditional .contiguous()
    (dynamo can't inspect strides on fake tensors); otherwise check first to avoid copies.
    """
    if torch.compiler.is_compiling():
        return t.contiguous()
    return t if t.stride(-1) == 1 else t.contiguous()


class RMSNorm(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, is_layernorm: bool = False):
        super().__init__(dtype, N, stage=2 if is_layernorm else 1)
        self.is_layernorm = is_layernorm
        self.reload_from = None if N <= (16384 if is_layernorm else 8192) else "smem"
        self.delay_w_load = False

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        # SM8x (Ampere/Ada) lacks cluster support
        if arch < Arch.sm_90:
            self.cluster_n = 1
            return
        # SM12x supports cluster up to 8
        max_cluster = 8 if arch.major == 12 else 16
        N = self.N
        # cluster_n = 4 is faster and cluster_n = 2 for N=64k for some reason
        # Similarly cluster_n = 8 is faster for N=128k
        if arch.major == 12 and const_expr(self.dtype.width >= 32):
            # SM12x 99 KB SMEM: fp32 needs tighter clustering (conservative for residual case)
            thresholds = [(8 * 1024, 1), (16 * 1024, 2), (32 * 1024, 4), (64 * 1024, 8)]
        elif const_expr(self.dtype.width == 16):
            thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
        else:
            thresholds = [(32 * 1024, 1), (64 * 1024, 2), (128 * 1024, 4), (256 * 1024, 8)]
        for limit, cluster in thresholds:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = max_cluster

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (b, N) or (b, H, N)
        mW: Optional[cute.Tensor],  # (N,) or (H, N)
        mB: Optional[cute.Tensor],  # (N,) or (H, N)
        mRes: Optional[cute.Tensor],  # (b, N) or (b, H, N)
        mO: cute.Tensor,  # (b, N) or (b, H, N)
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        eps: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(*(t.element_type.width for t in [mX, mRes, mW, mB, mO, mResO] if t is not None))
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        mW, mB = [
            layout_utils.expand(mT, dim=0, size=tiler_mn[0]) if const_expr(mT is not None) else None
            for mT in (mW, mB)
        ]
        mRstd, mMean = [
            layout_utils.expand(mT, dim=cute.rank(mT), size=self.N)
            if const_expr(mT is not None)
            else None
            for mT in (mRstd, mMean)
        ]
        num_heads = mX.shape[1] if const_expr(cute.rank(mX) == 3) else 1
        self.kernel(
            mX, mW, mB, mRes, mO, mResO, mRstd, mMean, eps, tiler_mn, tiled_copy, threads_per_row
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, num_heads],
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
        mMean: Optional[cute.Tensor],
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, bidz = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        if const_expr(mRes is not None):
            sRes = smem.allocate_tensor(
                mRes.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        # Slice per head
        if const_expr(cute.rank(mX) == 3):
            mX, mW, mB, mRes, mO, mResO, mRstd, mMean = [
                mT[None, bidz, None] if const_expr(mT is not None) else None
                for mT in (mX, mW, mB, mRes, mO, mResO, mRstd, mMean)
            ]

        shape = (cute.size(mX, mode=[0]), cute.size(mX, mode=[1]))
        idX = cute.make_identity_tensor(shape)
        # Slice for CTAs
        gX, gRes, gO, gResO, gRstd, gMean, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) if mT is not None else None
            for mT in (mX, mRes, mO, mResO, mRstd, mMean, idX)
        ]
        gW, gB = [
            cute.local_tile(mT, tiler_mn, (0, cluster_y)) if const_expr(mT is not None) else None
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
        tXrRstd = thr_copy_X.partition_D(gRstd) if const_expr(mRstd is not None) else None
        tXrMean = thr_copy_X.partition_D(gMean) if const_expr(mMean is not None) else None
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tXrW = cute.make_rmem_tensor_like(tXgW) if const_expr(mW is not None) else None
        tXrB = cute.make_rmem_tensor_like(tXgB) if const_expr(mB is not None) else None
        tXrX, tXrO = [cute.make_rmem_tensor_like(t) for t in (tXgX, tXgO)]
        if const_expr(mRes is not None):
            tXrRes = cute.make_rmem_tensor_like(tXgRes)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        # Each copy will use the same predicate
        copy = partial(copy_utils.copy, pred=tXpX)

        row = tXcX[0][0]
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
            if const_expr(mRes is not None):
                copy(tXgRes, tXsRes, is_async=True)
        cute.arch.cp_async_commit_group()

        if const_expr(not self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
            if const_expr(mB is not None):
                copy(tXgB, tXrB)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        if const_expr(mRes is not None):
            cute.autovec_copy(tXsRes, tXrRes)
            x += tXrRes.load().to(cute.Float32)
        if const_expr(mResO is not None):
            tXrResO = cute.make_rmem_tensor_like(tXgResO)
            tXrResO.store(x.to(tXrResO.element_type))
            if row < shape[0]:
                copy(tXrResO, tXgResO)

        mean, rstd = None, None
        if const_expr(self.is_layernorm):
            # LayerNorm: compute mean first, then variance
            sum_x = row_reduce(
                x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            )
            mean = sum_x / shape[1]
            if const_expr(mMean is not None):
                # Only the thread corresponding to column 0 writes out the mean to gmem
                if (
                    tXcX[0][1] == 0
                    and row < shape[0]
                    and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
                ):
                    tXrMean[0] = mean
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                x = tXrX.load().to(cute.Float32)
                if const_expr(mRes is not None):
                    cute.autovec_copy(tXsRes, tXrRes)
                    x += tXrRes.load().to(cute.Float32)
            elif const_expr(self.reload_from == "gmem"):
                copy(tXgX, tXrX)
                x = tXrX.load().to(cute.Float32)
                if const_expr(mRes is not None):
                    copy(tXgRes, tXrRes)
                    x += tXrRes.load().to(cute.Float32)
            sum_sq_x_sub_mean = row_reduce(
                (x - mean) * (x - mean),
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 1],
                mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
            )
            rstd = cute.math.rsqrt(sum_sq_x_sub_mean / shape[1] + eps, fastmath=True)
        else:
            # RMSNorm: compute sum of squares directly
            mean = const_expr(0.0)
            sum_sq_x = row_reduce(
                x * x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                init_val=0.0,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            )
            rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=True)
        if const_expr(mRstd is not None):
            # Only the thread corresponding to column 0 writes out the rstd to gmem
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = rstd
        if const_expr(self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
            if const_expr(mB is not None):
                copy(tXgB, tXrB)
        if const_expr(self.reload_from == "smem" or self.reload_from == "gmem"):
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                if const_expr(mRes is not None):
                    cute.autovec_copy(tXsRes, tXrRes)
            else:
                copy(tXgX, tXrX)
                if const_expr(mRes is not None):
                    copy(tXgRes, tXrRes)
            x = tXrX.load().to(cute.Float32)
            if const_expr(mRes is not None):
                x += tXrRes.load().to(cute.Float32)
        x_hat = (x - mean) * rstd if const_expr(self.is_layernorm) else x * rstd
        y = x_hat
        if const_expr(mW is not None):
            y *= tXrW.load().to(cute.Float32)
        if const_expr(mB is not None):
            y += tXrB.load().to(cute.Float32)
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy(tXrO, tXgO)


def _rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor],
    out: Tensor,
    bias: Optional[Tensor] = None,
    rstd: Optional[Tensor] = None,
    mean: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
    eps: float = 1e-6,
    is_layernorm: bool = False,
) -> None:
    """RMSNorm/LayerNorm forward pass.
    Args:
        x: Input tensor of shape (M, N)
        weight: Optional weight tensor of shape (N,) or (H, N) for per-head weight
        eps: Small value for numerical stability
        is_layernorm: If True, compute LayerNorm instead of RMSNorm
    Returns:
        Normalized output tensor of same shape as x
    """
    # Don't need to check is_cuda since torch.library ensures that
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_types, "Unsupported dtype"
    if weight is not None:
        assert weight.dtype in supported_types, "Weight must be float32, float16 or bfloat16"
    if residual is not None:
        assert residual.dtype in supported_types, "Residual must be float16, bfloat16, or float32"

    N = x.size(-1)
    per_head = (weight is not None and weight.dim() == 2) or (bias is not None and bias.dim() == 2)
    dtype, out_dtype, weight_dtype, bias_dtype, res_dtype, res_out_dtype = [
        torch2cute_dtype_map[t.dtype] if t is not None else None
        for t in [x, out, weight, bias, residual, residual_out]
    ]
    _compile_rmsnorm_fwd(
        dtype,
        out_dtype,
        res_dtype,
        weight_dtype,
        bias_dtype,
        res_out_dtype,
        N,
        rstd is not None,
        mean is not None,
        is_layernorm,
        per_head,
    )(x, weight, bias, residual, out, residual_out, rstd, mean, eps)


@jit_cache
def _compile_rmsnorm_fwd(
    dtype,
    out_dtype,
    res_dtype,
    weight_dtype,
    bias_dtype,
    res_out_dtype,
    N,
    has_rstd,
    has_mean,
    is_layernorm,
    per_head,
):
    batch_sym = cute.sym_int()
    head_sym = cute.sym_int() if per_head else None
    batch_shape = (batch_sym, head_sym) if per_head else (batch_sym,)
    all_dtypes = [dtype, out_dtype, res_dtype, weight_dtype, bias_dtype, res_out_dtype]
    div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
    x_cute, out_cute, res_cute, res_out_cute = [
        fake_tensor(dt, (*batch_shape, N), div)
        for dt in [dtype, out_dtype, res_dtype, res_out_dtype]
    ]
    weight_shape = (head_sym, N) if per_head else (N,)
    weight_cute, bias_cute = [
        fake_tensor(dt, weight_shape, div) for dt in [weight_dtype, bias_dtype]
    ]
    rstd_cute = fake_tensor(Float32, batch_shape) if has_rstd else None
    mean_cute = fake_tensor(Float32, batch_shape) if has_mean else None
    return cute.compile(
        RMSNorm(dtype, N, is_layernorm=is_layernorm),
        x_cute,
        weight_cute,
        bias_cute,
        res_cute,
        out_cute,
        res_out_cute,
        rstd_cute,
        mean_cute,
        Float32(0),  # eps, just for compilation
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    # Need to wrap to handle the case where residual_out is a alias of x, which makes torch.library
    # and torch.compile unhappy. Also allocate memory for out and residual_out if they are None
    # so that _layer_norm_fwd_impl doesn't have to return them.
    out_dtype = x.dtype if out_dtype is None else out_dtype
    out = torch.empty_like(x, dtype=out_dtype)
    rstd = torch.empty(*x.shape[:-1], device=x.device, dtype=torch.float32) if store_rstd else None
    if residual is not None and residual_dtype is None:
        residual_dtype = residual.dtype
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty_like(
            x, dtype=residual_dtype if residual_dtype is not None else x.dtype
        )
    else:
        residual_out = None
    if x.numel() > 0:
        _rmsnorm_fwd(x, weight, out, bias, rstd, None, residual, residual_out, eps, False)
    # residual_out is None if residual is None and residual_dtype == input_dtype and dropout_p == 0.0
    if residual_out is None:
        residual_out = x
    return out, residual_out, rstd


def rmsnorm_ref(x, w=None, bias=None, residual=None, eps=1e-6):
    x_f32 = x.float()
    if residual is not None:
        residual_f32 = residual.float()
        x_f32 = x_f32 + residual_f32
    x_norm = x_f32 / (torch.sqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + eps))
    out = x_norm * w if w is not None else x_norm
    if bias is not None:
        out = out + bias.float()
    if residual is None:
        return out.to(x.dtype)
    else:
        return out.to(x.dtype), x_f32.to(residual.dtype)


def rmsnorm_bwd_ref(x, w, dout, rstd, eps=1e-6):
    """Reference implementation for RMSNorm backward pass."""
    x_f32 = x.float()
    x_hat = x_f32 * rstd.unsqueeze(1)
    if w is not None:
        wdy = dout * w
    else:
        wdy = dout
    c1 = (x_hat * wdy).mean(dim=-1, keepdim=True)
    dx = (wdy - x_hat * c1) * rstd.unsqueeze(1)

    # dL/dW
    if w is not None:
        dw = (dout * x_hat).sum(dim=0)
        return dx.to(x.dtype), dw.to(w.dtype)
    else:
        return dx.to(x.dtype), None


class RMSNormBackward(ReductionBase):
    def __init__(self, dtype: cutlass.Numeric, N: int):
        # 2 stages for double buffering when computing mean of x_hat * wdy
        super().__init__(dtype, N, stage=2, reduction_dtype=Float32)
        self.reload_wdy = None if N <= 16 * 1024 else "smem"
        if self.N > 128 * 1024 and self.dtype.width >= 32:
            # Not enough smem
            raise ValueError("RMSNormBackward does not support N > 128k with dtype >= 32 bits")

    def _num_threads(self):
        return 128 if self.N <= 4096 else 256

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (256, 32), (512, 64), (4096, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        # SM8x (Ampere/Ada) lacks cluster support
        if arch < Arch.sm_90:
            self.cluster_n = 1
            return
        # SM12x supports cluster up to 8
        max_cluster = 8 if arch.major == 12 else 16
        N = self.N
        if arch.major == 12 and const_expr(self.dtype.width >= 32):
            # SM12x 99 KB SMEM: fp32 bwd double-buffers 2 tensors, needs much tighter clustering
            thresholds = [(1024, 1), (8 * 1024, 2), (16 * 1024, 4), (32 * 1024, 8)]
        else:
            thresholds = [(8 * 1024, 1), (16 * 1024, 2), (32 * 1024, 4), (64 * 1024, 8)]
        for limit, cluster in thresholds:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = max_cluster

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
            max(*(t.element_type.width for t in [mX, mW, mdO, mdResO, mdX, mdRes] if t is not None))
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        mW = (
            layout_utils.expand(mW, dim=0, size=tiler_mn[0]) if const_expr(mW is not None) else None
        )
        num_blocks = sm_count
        num_heads = mX.shape[1] if const_expr(cute.rank(mX) == 3) else 1
        self.kernel(
            mX, mW, mdO, mdResO, mRstd, mdX, mdW, mdB, mdRes, tiler_mn, tiled_copy, threads_per_row
        ).launch(
            grid=[num_blocks, self.cluster_n, num_heads],
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
        bidx_start, _, bidz = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        # Slice per head
        if const_expr(cute.rank(mX) == 3):
            mX, mW, mdO, mdResO, mdX, mdW, mdB, mdRes = [
                mT[None, bidz, None] if const_expr(mT is not None) else None
                for mT in (mX, mW, mdO, mdResO, mdX, mdW, mdB, mdRes)
            ]
            mRstd = mRstd[None, bidz]

        shape = mX.shape
        M, N = shape[0], shape[1]
        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)

        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_ordered_layout((tiler_mn[0], tiler_mn[1], 2), order=(1, 0, 2))
        sX = smem.allocate_tensor(mX.element_type, smem_layout, byte_alignment=16)
        sdO = smem.allocate_tensor(mdO.element_type, smem_layout, byte_alignment=16)
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout, is_persistent=True
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
            cute.make_rmem_tensor_like(thr[None, None, None, 0]) for thr in (tXgX, tXgdO, tXgdX)
        ]
        tXrdResO = None
        if const_expr(mdResO is not None):
            tXrdResO = cute.make_rmem_tensor_like(tXgdResO[None, None, None, 0])
        tXrdRes = None
        if const_expr(mdRes is not None):
            tXrdRes = cute.make_rmem_tensor_like(tXgdRes[None, None, None, 0])

        # This doesn't change across iterations
        tXpX = (
            None
            if is_even_N
            else copy_utils.predicate_k(thr_copy_X.partition_S(cX[None, None, 0]), limit=shape[1])
        )
        # Each copy will use the same number of elements as X
        copy = partial(copy_utils.copy, pred=tXpX)

        tXgdW, tXrdW = None, None
        tXgdB, tXrdB = None, None
        if const_expr(mdW is not None):
            tXgdW = thr_copy_X.partition_S(gdW)
            # Always compute partial weight gradients in fp32
            tXrdW = cute.make_rmem_tensor_like(tXgdW, Float32)
        if const_expr(mdB is not None):
            tXgdB = thr_copy_X.partition_S(gdB)
            # Always compute partial bias gradients in fp32
            tXrdB = cute.make_rmem_tensor_like(tXgdB, Float32)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE

        self._initialize_cluster(tidx, mbar_ptr, num_warps, is_persistent=True)

        tXrW = None
        if const_expr(mW is not None):
            tXgW = thr_copy_X.partition_S(gW)
            tXrW = cute.make_rmem_tensor_like(tXgW)
            # Need this, otherwise rW can have arbitrary values that changes the reduction
            if const_expr(not is_even_N):
                tXrW.fill(0.0)
            copy(tXgW, tXrW)

        # Prefetch the first batch
        row = tXcX[None, None, None, bidx_start][0][0]
        if row < M:
            copy(tXgX[None, None, None, bidx_start], tXsX[None, None, None, 0], is_async=True)
            copy(tXgdO[None, None, None, bidx_start], tXsdO[None, None, None, 0], is_async=True)
        else:
            if const_expr(tiler_mn[0] > 1):
                # Fill with zero, otherwise smem will be uninitialized, and we could read this back
                # later into registers, causing wrong dW.
                utils.fill_oob(tXsX[None, None, None, 0], None, fill_value=mX.element_type.zero)
                utils.fill_oob(tXsdO[None, None, None, 0], None, fill_value=mdO.element_type.zero)
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
            if row + gdim * tiler_mn[0] < M:  # Prefetch the next batch
                copy(
                    tXgX[None, None, None, bidx + gdim],
                    tXsX[None, None, None, stage ^ 1],
                    is_async=True,
                )
                copy(
                    tXgdO[None, None, None, bidx + gdim],
                    tXsdO[None, None, None, stage ^ 1],
                    is_async=True,
                )
            else:
                if const_expr(tiler_mn[0] > 1):
                    utils.fill_oob(
                        tXsX[None, None, None, stage ^ 1], None, fill_value=mX.element_type.zero
                    )
                    utils.fill_oob(
                        tXsdO[None, None, None, stage ^ 1], None, fill_value=mdO.element_type.zero
                    )
            cute.arch.cp_async_commit_group()
            rstd = cutlass.Float.zero
            if row < M or tiler_mn[0] == 1:
                rstd = mRstd[row]
            if const_expr(mdResO is not None):
                if row < M or tiler_mn[0] == 1:
                    copy(tXgdResO[None, None, None, bidx], tXrdResO)
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
                # Need this fence since the STAS from the producer is using the async proxy.
                cute.arch.fence_view_async_shared()
                # It's faster to have 1 lane per warp to signal the mbar, rather than all lanes
                # Requires adjusting the thread_count when initializing the mbar
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
                copy(tXrdX, tXgdX[None, None, None, bidx])
            if const_expr(mdRes is not None):
                tXrdRes.store(dx.to(tXrdRes.element_type))
                if row < M or tiler_mn[0] == 1:
                    copy(tXrdRes, tXgdRes[None, None, None, bidx])
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
                # reduction of dw_partial within the same threadblock
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
                        tXrdW_other = cute.make_rmem_tensor_like(tXrdW)
                        tXsdW_other = cute.make_tensor(
                            tXsdW.iterator + i * sdW.stride[0], tXsdW.layout
                        )
                        cute.autovec_copy(tXsdW_other, tXrdW_other)
                        tXrdW.store(tXrdW.load() + tXrdW_other.load())
                    copy(tXrdW, tXgdW)
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
                        tXrdB_other = cute.make_rmem_tensor_like(tXrdB)
                        tXsdB_other = cute.make_tensor(
                            tXsdB.iterator + i * sdB.stride[0], tXsdB.layout
                        )
                        cute.autovec_copy(tXsdB_other, tXrdB_other)
                        tXrdB.store(tXrdB.load() + tXrdB_other.load())
                    copy(tXrdB, tXgdB)
        else:
            # dw is already in fp32, so we can directly copy to global memory
            if const_expr(mdW is not None):
                copy(tXrdW, tXgdW)
            if const_expr(mdB is not None):
                copy(tXrdB, tXgdB)

        if const_expr(self.cluster_n > 1):  # Prevent cluster from exiting early
            # Assume state contains that next useful buffer
            # So we only need to advance to num_stages - 1 times to last used buffer
            stage ^= 1
            if stage == 0:
                producer_phase ^= 1
            cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)


def _get_sm_count(N: int, device: torch.device) -> int:
    # This should be tuned on how many CTAs can be launched on each SM
    sm_count_multiple = (
        16 if N <= 256 else (8 if N <= 1024 else (4 if N <= 2048 else (2 if N <= 4096 else 1)))
    )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    # By right, if we're using cluster, this should be cluster_count not sm_count.
    # But for cluster >= 4, due to quantization we would need to query active max cluster.
    # Instead we just do sm_count * 2, which is reasonably larger than active_cluster_count to
    # avoid wave quantization.
    sm_count = (
        sm_count * sm_count_multiple if N <= 8192 else sm_count // 2 if N <= 16384 else sm_count * 2
    )

    return sm_count


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
    """RMSNorm backward pass.
    Args:
        x: Input tensor of shape (M, N) or (M, H, N) for per-head
        weight: Optional weight tensor of shape (N,) or (H, N) for per-head
        dout: Upstream gradients tensor of shape (M, N) or (M, H, N)
        rstd: Reciprocal standard deviation tensor of shape (M,) or (M, H)
    Returns:
        Tuple of (dx, dw) where:
        - dx: Input gradients tensor of same shape as x
        - dw: Weight gradients tensor of same shape as weight (or None if weight is None)
    """
    assert x.dim() in (2, 3), "Input must be 2D or 3D"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_types, "Unsupported dtype"
    per_head = x.dim() == 3
    if weight is not None:
        assert weight.is_cuda, "Weight tensor must be on CUDA device"
        assert weight.dtype in supported_types, "Weight must be float32, float16 or bfloat16"
    if dresidual_out is not None:
        assert dresidual_out.shape == x.shape
        assert dresidual_out.is_cuda
        assert dresidual_out.dtype in supported_types, (
            "Residual must be float16, bfloat16, or float32"
        )
    if dresidual is not None:
        assert dresidual.shape == x.shape
        assert dresidual.is_cuda
        assert dresidual.dtype in supported_types, "Residual must be float16, bfloat16, or float32"

    N = x.size(-1)
    if dw_partial is None and db_partial is None:
        assert sm_count is not None
    else:
        sm_count = dw_partial.shape[0] if dw_partial is not None else db_partial.shape[0]
    dtype, dout_dtype, dx_dtype, weight_dtype, dres_dtype, dres_out_dtype = [
        torch2cute_dtype_map[t.dtype] if t is not None else None
        for t in [x, dout, dx, weight, dresidual, dresidual_out]
    ]
    _compile_rmsnorm_bwd(
        N,
        dtype,
        dout_dtype,
        dx_dtype,
        weight_dtype,
        db_partial is not None,
        dres_dtype,
        dres_out_dtype,
        dw_partial is not None,
        per_head,
    )(x, weight, dout, dresidual_out, rstd, dx, dw_partial, dresidual, db_partial, sm_count)


@jit_cache
def _compile_rmsnorm_bwd(
    N,
    dtype,
    dout_dtype,
    dx_dtype,
    weight_dtype,
    has_db_partial,
    dres_dtype,
    dres_out_dtype,
    has_dw_partial,
    per_head=False,
):
    batch_sym, batch_partial_sym = cute.sym_int(), cute.sym_int()
    head_sym = cute.sym_int() if per_head else None
    batch_shape = (batch_sym, head_sym) if per_head else (batch_sym,)
    all_dtypes = [dtype, dout_dtype, dx_dtype, dres_dtype, dres_out_dtype]
    div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
    x_cute, dout_cute, dx_cute, dres_out_cute, dres_cute = [
        fake_tensor(dt, (*batch_shape, N), div)
        for dt in [dtype, dout_dtype, dx_dtype, dres_out_dtype, dres_dtype]
    ]
    weight_shape = (head_sym, N) if per_head else (N,)
    weight_cute = fake_tensor(weight_dtype, weight_shape, div)
    rstd_cute = fake_tensor(Float32, batch_shape)
    dw_shape = (batch_partial_sym, head_sym, N) if per_head else (batch_partial_sym, N)
    dw_partial_cute = fake_tensor(Float32, dw_shape, div) if has_dw_partial else None
    db_partial_cute = fake_tensor(Float32, dw_shape, div) if has_db_partial else None
    return cute.compile(
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
        0,  # sm_count, just for compilation
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def rmsnorm_bwd(
    x: Tensor,
    weight: Optional[Tensor],
    dout: Tensor,
    rstd: Tensor,
    dresidual_out: Optional[Tensor] = None,  # grad wrt residual_out
    has_bias: bool = False,
    has_residual: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    device = x.device
    N = x.size(-1)
    per_head = x.dim() == 3
    dx = torch.empty_like(x)
    if dresidual_out is not None and dresidual_out.dtype != dx.dtype:
        dresidual = torch.empty_like(x, dtype=dresidual_out.dtype)
    else:
        dresidual = None
    sm_count = _get_sm_count(N, device)
    if per_head:
        H = x.size(1)
        sm_count = max(round(sm_count / H), 1)
    else:
        H = None
    if weight is not None:
        # Always store partial gradients in fp32 for numerical accuracy
        dw_shape = (sm_count, H, N) if per_head else (sm_count, N)
        dw_partial = torch.empty(dw_shape, device=device, dtype=torch.float32)
    else:
        dw_partial = None
    db_shape = (sm_count, H, N) if per_head else (sm_count, N)
    db_partial = torch.empty(db_shape, device=device, dtype=torch.float32) if has_bias else None

    if x.numel() > 0:
        _rmsnorm_bwd(
            x, weight, dout, rstd, dx, dw_partial, db_partial, dresidual_out, dresidual, sm_count
        )
        # we have summed the partial gradients in fp32, now we convert back to the weight dtype
        dw = dw_partial.sum(dim=0).to(weight.dtype) if weight is not None else None
        db = db_partial.sum(dim=0).to(weight.dtype) if has_bias else None
    else:
        dw = torch.zeros_like(weight) if weight is not None else None
        db = torch.zeros_like(weight) if has_bias else None
    # dresidual is the same as dx in this case
    if has_residual and dresidual is None:
        dresidual = dx
    return dx, dw, db, dresidual


class RMSNormFunction(torch.autograd.Function):
    """Autograd wrapper for rmsnorm.

    All input reshaping (flattening batch dims, per-head layout) is done in the
    rmsnorm() wrapper BEFORE calling .apply(). This function receives already-
    flattened tensors so that tensor ranks never change between recompilations,
    which is required for torch.compile compatibility.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        residual=None,
        out_dtype=None,
        residual_dtype=None,
        eps=1e-6,
        prenorm=False,
    ):
        x = _ensure_contiguous(x)
        if residual is not None:
            residual = _ensure_contiguous(residual)
        need_grad = any(ctx.needs_input_grad[:3])
        out, residual_out, rstd = rmsnorm_fwd(
            x,
            weight,
            bias=bias,
            residual=residual,
            out_dtype=out_dtype,
            residual_dtype=residual_dtype,
            eps=eps,
            store_rstd=need_grad,
        )
        ctx.save_for_backward(x if residual is None else residual_out, weight, rstd)
        ctx.has_bias = bias is not None
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        if residual_out is None or not prenorm:
            return out
        else:
            return out, residual_out

    @staticmethod
    def backward(ctx, dout, *args):
        x, weight, rstd = ctx.saved_tensors
        dout = _ensure_contiguous(dout)
        if ctx.prenorm and ctx.has_residual:
            dresidual_out = _ensure_contiguous(args[0])
        else:
            dresidual_out = None
        dx, dw, db, dresidual = rmsnorm_bwd(
            x,
            weight,
            dout,
            rstd,
            dresidual_out,
            ctx.has_bias,
            has_residual=ctx.has_residual,
        )
        return dx, dw, db, dresidual, *([None] * 4)


def rmsnorm(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    prenorm: bool = False,
) -> Tensor:
    """RMSNorm with automatic differentiation support.

    Args:
        x: Input tensor of shape (M, N) or (B, S, H, D) for per-head mode
        weight: Optional weight tensor of shape (N,) or (H, D) for per-head mode
        eps: Small value for numerical stability

    Returns:
        Normalized output tensor of same shape as x
    """
    x_shape_og = x.shape
    per_head = (weight is not None and weight.dim() == 2) or (bias is not None and bias.dim() == 2)
    last_shape = x_shape_og[-1:] if not per_head else x_shape_og[-2:]
    # Flatten batch dims before entering autograd.Function so tensor ranks
    # are determined by per_head (which dynamo guards on via the if-branch),
    # not by the original input shape. This ensures torch.compile can
    # recompile the backward subgraph correctly when switching between
    # per_head=False and per_head=True.
    x_flat = x.reshape(-1, *last_shape)
    res_flat = residual.reshape(-1, *last_shape) if residual is not None else None
    result = RMSNormFunction.apply(
        x_flat, weight, bias, res_flat, out_dtype, residual_dtype, eps, prenorm
    )
    if isinstance(result, tuple):
        return tuple(r.reshape(x_shape_og) for r in result)
    return result.reshape(x_shape_og)


class QuackRMSNorm(torch.nn.RMSNorm):
    """RMSNorm module that behaves like torch.nn.RMSNorm.

    This class provides a drop-in replacement for torch.nn.RMSNorm that uses
    the torch._vendor.quack.rmsnorm implementation under the hood.

    Args:
        dim (int): The dimension to normalize over
        eps (float, optional): A small constant for numerical stability. Default: 1e-6

    Attributes:
        weight (torch.nn.Parameter): The learnable weight parameter
        eps (float): A small constant for numerical stability
    """

    def __init__(
        self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True, device=None, dtype=None
    ):
        super().__init__(dim, eps, elementwise_affine, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm to the input tensor.

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Normalized tensor
        """
        return rmsnorm(x, self.weight, eps=self.eps)


def layernorm_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    eps: float = 1e-6,
    return_rstd: bool = False,
    return_mean: bool = False,
):
    """LayerNorm forward pass using the unified RMSNorm/LayerNorm kernel.

    Args:
        x: Input tensor of shape (M, N)
        weight: Weight tensor of shape (N,). Must be float32.
        bias: Optional bias tensor of shape (N,). Must be float32.
        eps: Small value for numerical stability
        return_rstd: Whether to return the reciprocal standard deviation
        return_mean: Whether to return the mean

    Returns:
        Normalized output tensor of same shape as x
        If return_rstd is True, also returns rstd tensor of shape (M,)
        If return_mean is True, also returns mean tensor of shape (M,)
    """
    assert x.dim() == 2, "Input must be 2D"
    assert weight.dim() == 1, "Weight must be 1D"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    assert weight.dtype == torch.float32, "Weight must be float32"
    if bias is not None:
        assert bias.dim() == 1, "Bias must be 1D"
        assert bias.dtype == torch.float32, "Bias must be float32"

    M, N = x.shape
    device = x.device
    out = torch.empty_like(x)
    rstd = torch.empty(M, device=device, dtype=torch.float32) if return_rstd else None
    mean = torch.empty(M, device=device, dtype=torch.float32) if return_mean else None

    _rmsnorm_fwd(x, weight, out, bias, rstd, mean, None, None, eps, True)

    if return_rstd and return_mean:
        return out, rstd, mean
    elif return_rstd:
        return out, rstd
    elif return_mean:
        return out, mean
    return out


def layernorm_ref(x: Tensor, w: Tensor, eps: float = 1e-6) -> Tensor:
    """Reference implementation for LayerNorm."""
    x_f32 = x.float()
    return torch.nn.functional.layer_norm(x_f32, w.shape, w, None, eps).to(x.dtype)


def layernorm_rstd_ref(x: torch.Tensor, eps: float = 1e-6):
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = ((x_f32 - mean) ** 2).mean(dim=-1)
    return 1.0 / torch.sqrt(var + eps)


def layernorm_mean_ref(x: torch.Tensor) -> torch.Tensor:
    return x.float().mean(dim=-1)
