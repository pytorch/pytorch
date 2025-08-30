# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

# pyre-ignore-all-errors
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from torch._inductor.kernel.quack import utils

from .reduction_base import ReductionBase, torch2cute_dtype_map

# import from: https://github.com/Dao-AILab/quack/blob/main/quack/rmsnorm.py


class RMSNorm(ReductionBase):
    def __init__(self, dtype: cutlass.Numeric, N: int):
        super().__init__(dtype, N, stage=1)
        self.reload_from = None if N <= 16384 else "smem"
        self.delay_w_load = False

    def _calculate_threads_per_row(self):
        """Calculate the number of threads per row for the RMSNorm kernel."""
        N = self.N
        if N <= 64:
            return 8
        elif N <= 128:
            return 16
        elif N <= 3072:
            return 32
        elif N <= 6144:
            return 64
        elif N <= 16384:
            return 128
        else:
            return 256

    def _set_cluster_n(self):
        """
        Set the number of clusters for the RMSNorm kernel.
        Stored in self.cluster_n.
        """
        N = self.N

        # cluster_n = 4 is faster and cluster_n = 2 for N=64k for some reason
        # Similarly cluster_n = 8 is faster for N=128k
        if cutlass.const_expr(self.dtype.width == 16):
            # 16-bit types (fp16, bf16)
            if N <= 16 * 1024:
                cluster_n = 1
            elif N <= 32 * 1024:
                cluster_n = 2
            elif N <= 64 * 1024:
                cluster_n = 4
            elif N <= 128 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16
        else:
            # 32-bit types (fp32)
            if N <= 32 * 1024:
                cluster_n = 1
            elif N <= 64 * 1024:
                cluster_n = 2
            elif N <= 128 * 1024:
                cluster_n = 4
            elif N <= 256 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16

        self.cluster_n = cluster_n

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
    ):
        semistatic_shape = (
            *mX.shape[:-1],
            self.N,
        )  # Set last dimension to be statically N
        new_stride = lambda t: (
            cute.assume(t.stride[0], divby=128 // t.element_type.width),
            t.stride[1],
        )
        mX, mO = [
            cute.make_tensor(
                t.iterator, cute.make_layout(semistatic_shape, stride=new_stride(t))
            )
            for t in (mX, mO)
        ]
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW = cute.make_tensor(mW.iterator, mW_expanded_layout)
        if cutlass.const_expr(mRstd is not None):
            mRstd_expanded_layout = cute.append(
                mRstd.layout, cute.make_layout((self.N,), stride=(0,))
            )
            mRstd = cute.make_tensor(mRstd.iterator, mRstd_expanded_layout)
        self.kernel(
            mX, mW, mO, mRstd, eps, tv_layout, tiler_mn, self.reload_from
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=(
                [1, self.cluster_n, 1]
                if cutlass.const_expr(self.cluster_n > 1)
                else None
            ),
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        eps: cute.Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
        reload_from: cutlass.Constexpr = None,
        delay_w_load: cutlass.Constexpr = False,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

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
        # slice for CTAs
        # We use domain_offset_i64 to deal with tensors larger than 2^31 elements
        mX, mO = [
            utils.domain_offset_i64((bidx * tiler_mn[0], 0), mT) for mT in (mX, mO)
        ]
        gX, gO = [cute.local_tile(mT, tiler_mn, (0, cluster_y)) for mT in (mX, mO)]
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y))
        gRstd = (
            cute.local_tile(mRstd, tiler_mn, (bidx, cluster_y))
            if cutlass.const_expr(mRstd is not None)
            else None
        )

        # declare the atoms which will be used later for memory copy
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=128
        )
        copy_atom_load_X_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=128
        )
        num_bits_per_copy_W = cutlass.const_expr(
            min(128, 128 // mX.element_type.width * mW.element_type.width)
        )
        copy_atom_load_W = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mW.element_type,
            num_bits_per_copy=num_bits_per_copy_W,
        )
        num_bits_per_copy_O = cutlass.const_expr(
            min(128, 128 // mX.element_type.width * mO.element_type.width)
        )
        copy_atom_store_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mO.element_type,
            num_bits_per_copy=num_bits_per_copy_O,
        )

        thr_copy_X = cute.make_tiled_copy(
            copy_atom_load_X_async, tv_layout, tiler_mn
        ).get_slice(tidx)

        tXgW = thr_copy_X.partition_S(gW)
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_X.partition_D(gO)
        tXrRstd = (
            thr_copy_X.partition_D(gRstd)
            if cutlass.const_expr(mRstd is not None)
            else None
        )
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tXrW = cute.make_fragment_like(tXgW)
        tXrW.fill(0.0)
        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        tXpX = utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
        row = tXcX[0][0]
        if row < shape[0]:
            cute.copy(copy_atom_load_X_async, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()

        tXpW = utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
        if cutlass.const_expr(not delay_w_load):
            cute.copy(copy_atom_load_W, tXgW, tXrW, pred=tXpW)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        threads_per_row = tv_layout.shape[0][0]
        sum_sq_x = utils.row_reduce(
            x * x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=(
                cute.arch.cluster_wait
                if cutlass.const_expr(self.cluster_n > 1)
                else None
            ),
        )
        rstd = utils.rsqrt(sum_sq_x / shape[1] + eps)
        if cutlass.const_expr(mRstd is not None):
            # Only the thread corresponding to column 0 writes out the rstd to gmem
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = rstd
        if cutlass.const_expr(delay_w_load):
            cute.copy(copy_atom_load_W, tXgW, tXrW, pred=tXpW)
        if cutlass.const_expr(reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)
        elif cutlass.const_expr(reload_from == "gmem"):
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
            x = tXrX.load().to(cute.Float32)
        x_hat = x * rstd
        w = tXrW.load().to(cute.Float32)
        y = x_hat * w
        tXrO.store(y.to(tXrO.element_type))
        tXpO = utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
        if row < shape[0]:
            cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tXpO)


def _rmsnorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    return_rstd: bool = False,
) -> torch.Tensor:
    """RMSNorm forward pass.
    Args:
        x: Input tensor of shape (M, N)
        weight: Weight tensor of shape (N,)
        eps: Small value for numerical stability
        return_rstd: Whether to return the reciprocal standard deviation
    Returns:
        Normalized output tensor of same shape as x
        If return_rstd is True, also returns rstd tensor of shape (M,)
    """
    assert x.dim() == 2, "Input must be 2D"
    assert weight.dim() == 1, "Weight must be 1D"
    assert (
        x.shape[-1] == weight.shape[0]
    ), "Last dimension of input must match weight dimension"
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ], "Unsupported dtype"

    assert weight.dtype in [
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ], "Weight must be float32, float16 or bfloat16"

    M, N = x.shape
    device = x.device
    out = torch.empty_like(x)
    rstd = torch.empty(M, device=device, dtype=torch.float32) if return_rstd else None
    dtype = torch2cute_dtype_map[x.dtype]
    # convert_from_dlpack = lambda x: (
    #     from_dlpack(x.detach(), assumed_align=16).mark_compact_shape_dynamic(
    #         mode=0, divisibility=128 // dtype.width
    #     )
    # )
    convert_from_dlpack = lambda x: (
        from_dlpack(x.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    )
    x_tensor, out_tensor = [convert_from_dlpack(t) for t in (x, out)]
    # handle weight divisibility based on weight dtype
    weight_dtype = torch2cute_dtype_map[weight.dtype]
    weight_tensor = utils.convert_from_dlpack(
        weight.detach(), leading_dim=0, divisibility=128 // weight_dtype.width
    )
    rstd_tensor = (
        from_dlpack(rstd.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if rstd is not None
        else None
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, N, rstd is not None, weight.dtype)
    if compile_key not in _rmsnorm_fwd.compile_cache:
        rmsnorm_op = RMSNorm(dtype, N)
        _rmsnorm_fwd.compile_cache[compile_key] = cute.compile(
            rmsnorm_op, x_tensor, weight_tensor, out_tensor, rstd_tensor, current_stream
        )
    _rmsnorm_fwd.compile_cache[compile_key](
        x_tensor, weight_tensor, out_tensor, rstd_tensor, current_stream, eps
    )
    return (out, rstd) if return_rstd else out


_rmsnorm_fwd.compile_cache = {}


def rmsnorm_ref(x, w, eps=1e-6):
    x_f32 = x.float()
    return (
        x_f32 / (torch.sqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + eps)) * w
    ).to(x.dtype)


def rstd_ref(x, eps=1e-6):
    x_f32 = x.float()
    return 1.0 / torch.sqrt(torch.mean(x_f32 * x_f32, dim=-1) + eps)


def rmsnorm_bwd_ref(x, w, dout, rstd, eps=1e-6):
    """Reference implementation for RMSNorm backward pass."""
    x_f32 = x.float()
    x_hat = x_f32 * rstd.unsqueeze(1)
    wdy = dout * w
    c1 = (x_hat * wdy).mean(dim=-1, keepdim=True)
    dx = (wdy - x_hat * c1) * rstd.unsqueeze(1)

    # dL/dW
    dw = (dout * x_hat).sum(dim=0)
    return dx.to(x.dtype), dw.to(w.dtype)


class RMSNormBackward(ReductionBase):
    def __init__(self, dtype: cutlass.Numeric, N: int):
        # 2 stages for double buffering when computing mean of x_hat * wdy
        super().__init__(dtype, N, stage=2, reduction_dtype=Float32)
        self.reload_wdy = None if N <= 16 * 1024 else "smem"
        if self.N > 128 * 1024 and self.dtype.width >= 32:
            # Not enough smem
            raise ValueError(
                "RMSNormBackward does not support N > 128k with dtype >= 32 bits"
            )

    def _get_num_threads(self):
        return 128 if self.N <= 4096 else 256

    def _calculate_threads_per_row(self):
        N = self.N
        return (
            8
            if N <= 64
            else (
                16
                if N <= 128
                else (
                    32
                    if N <= 256
                    else (64 if N <= 512 else (128 if N <= 4096 else 256))
                )
            )
        )

    def _set_cluster_n(self):
        N = self.N
        cluster_n = (
            1
            if N <= 8 * 1024
            else (
                2
                if N <= 16 * 1024
                else (4 if N <= 32 * 1024 else (8 if N <= 64 * 1024 else 16))
            )
        )
        self.cluster_n = cluster_n

    def _smem_size_in_bytes(self, tiler_mn, num_warps):
        return (
            # Multiply by 2 since we need space for X and dOut,
            # and multiply by another 2 due to double buffering
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)) * 2 * 2
            + self.stage
            * num_warps
            * self.cluster_n
            * (self.reduction_dtype.width // 8)
            + self.stage
            * (cutlass.Int64.width // 8)
            * 2  # mult 2 as we need 2 mbar per stage
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mdOut: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: cute.Tensor,
        sm_count: Int32,
        stream: cuda.CUstream,
    ):
        semistatic_shape = (
            *mX.shape[:-1],
            self.N,
        )  # Set last dimension to be statically N
        new_stride = lambda t: (
            cute.assume(t.stride[0], divby=128 // t.element_type.width),
            t.stride[1],
        )
        mX, mdOut, mdX = [
            cute.make_tensor(
                t.iterator, cute.make_layout(semistatic_shape, stride=new_stride(t))
            )
            for t in (mX, mdOut, mdX)
        ]
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE

        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW = cute.make_tensor(mW.iterator, mW_expanded_layout)

        num_blocks = sm_count
        self.kernel(mX, mW, mdOut, mRstd, mdX, mdW, tv_layout, tiler_mn).launch(
            grid=[num_blocks, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mdOut: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_start, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        shape = mX.shape
        M, N = shape[0], shape[1]
        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)

        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_ordered_layout(
            (tiler_mn[0], tiler_mn[1], 2), order=(1, 0, 2)
        )
        sX = smem.allocate_tensor(mX.element_type, smem_layout, byte_alignment=16)
        sdOut = smem.allocate_tensor(mdOut.element_type, smem_layout, byte_alignment=16)
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout, is_persistent=True
        )
        if cutlass.const_expr(mbar_ptr is not None):
            mbar_full_ptr, mbar_empty_ptr = mbar_ptr, mbar_ptr + 2
        else:
            mbar_full_ptr, mbar_empty_ptr = None, None

        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=128
        )
        copy_atom_load_X_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=128
        )
        num_bits_per_copy_W = cutlass.const_expr(
            min(128, 128 // mX.element_type.width * mW.element_type.width)
        )
        copy_atom_load_W = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mW.element_type,
            num_bits_per_copy=num_bits_per_copy_W,
        )
        num_bits_per_copy_dX = cutlass.const_expr(
            min(128, 128 // mX.element_type.width * mdX.element_type.width)
        )
        copy_atom_store_dX = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mdX.element_type,
            num_bits_per_copy=num_bits_per_copy_dX,
        )
        num_bits_per_copy_dW = cutlass.const_expr(
            min(128, 128 // mX.element_type.width * mdW.element_type.width)
        )
        copy_atom_store_dW = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mdW.element_type,
            num_bits_per_copy=num_bits_per_copy_dW,
        )

        thr_copy_X = cute.make_tiled_copy(
            copy_atom_load_X, tv_layout, tiler_mn
        ).get_slice(tidx)

        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y))
        tXgW = thr_copy_X.partition_S(gW)
        tXrW = cute.make_fragment_like(tXgW)
        # Need this, otherwise rW can have arbitrary values that changes the reduction
        if not is_even_N:
            tXrW.fill(0.0)

        gW_coord = cute.local_tile(idX, tiler_mn, (0, cluster_y))
        tXpW = (
            utils.predicate_k(thr_copy_X.partition_S(gW_coord), limit=shape[1])
            if not is_even_N
            else None
        )
        cute.copy(copy_atom_load_W, tXgW, tXrW, pred=tXpW)
        weight = tXrW.load().to(cute.Float32)

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE

        self._initialize_cluster(tidx, mbar_ptr, num_warps, is_persistent=True)

        dw_coord = cute.local_tile(idX, tiler_mn, (0, cluster_y))
        tXpdW = (
            utils.predicate_k(thr_copy_X.partition_S(dw_coord), limit=shape[1])
            if not is_even_N
            else None
        )

        gdW = cute.local_tile(mdW, (1, tiler_mn[1]), (bidx_start, cluster_y))
        tXgdW = thr_copy_X.partition_S(gdW)
        # Always compute partial weight gradients in fp32
        tXrdW = cute.make_fragment_like(tXgdW, Float32)

        gX = cute.local_tile(mX, tiler_mn, (None, cluster_y))
        gdOut = cute.local_tile(mdOut, tiler_mn, (None, cluster_y))
        gdX = cute.local_tile(mdX, tiler_mn, (None, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (None, cluster_y))
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgdOut = thr_copy_X.partition_S(gdOut)
        tXsdOut = thr_copy_X.partition_D(sdOut)
        tXgdX = thr_copy_X.partition_D(gdX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None, None]
        # This doesn't change across iterations
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX[None, None, 0]), limit=shape[1])
            if not is_even_N
            else None
        )

        tXrX, tXrdOut, tXrdX = [
            cute.make_fragment_like(thr[None, None, None, 0])
            for thr in (tXgX, tXgdOut, tXgdX)
        ]

        # Prefetch the first batch
        row = tXcX[None, None, None, bidx_start][0][0]
        if row < M:
            tXgX_cur = utils.coord_offset_i64(bidx_start, tXgX, dim=3)[
                None, None, None, 0
            ]
            tXgdOut_cur = utils.coord_offset_i64(bidx_start, tXgdOut, dim=3)[
                None, None, None, 0
            ]
            cute.copy(
                copy_atom_load_X_async,
                tXgX_cur,
                tXsX[None, None, None, 0],
                pred=tXpX,
            )
            cute.copy(
                copy_atom_load_X_async,
                tXgdOut_cur,
                tXsdOut[None, None, None, 0],
                pred=tXpX,
            )
        elif tiler_mn[0] > 1:
            # Fill with zero, otherwise smem will be uninitialized, and we could read this back
            # later into registers, causing wrong dW.
            utils.fill_oob(
                tXsX[None, None, None, 0], None, fill_value=mX.element_type.zero
            )
            utils.fill_oob(
                tXsdOut[None, None, None, 0], None, fill_value=mdOut.element_type.zero
            )
        cute.arch.cp_async_commit_group()

        if cutlass.const_expr(self.cluster_n > 1):
            cute.arch.cluster_wait()

        threads_per_row = tv_layout.shape[0][0]
        tXrdW.fill(0.0)
        stage = Int32(0)
        producer_phase = Int32(1)
        consumer_phase = Int32(0)
        for bidx in cutlass.range(bidx_start, cute.ceil_div(M, tiler_mn[0]), gdim):
            row = tXcX[None, None, None, bidx][0][0]
            rstd = cutlass.Float.zero
            if row + gdim * tiler_mn[0] < M:  # Prefetch the next batch
                tXgX_cur = utils.coord_offset_i64(bidx + gdim, tXgX, dim=3)[
                    None, None, None, 0
                ]
                tXgdOut_cur = utils.coord_offset_i64(bidx + gdim, tXgdOut, dim=3)[
                    None, None, None, 0
                ]
                cute.copy(
                    copy_atom_load_X_async,
                    tXgX_cur,
                    tXsX[None, None, None, stage ^ 1],
                    pred=tXpX,
                )
                cute.copy(
                    copy_atom_load_X_async,
                    tXgdOut_cur,
                    tXsdOut[None, None, None, stage ^ 1],
                    pred=tXpX,
                )
            elif tiler_mn[0] > 1:
                utils.fill_oob(
                    tXsX[None, None, None, stage ^ 1],
                    None,
                    fill_value=mX.element_type.zero,
                )
                utils.fill_oob(
                    tXsdOut[None, None, None, stage ^ 1],
                    None,
                    fill_value=mdOut.element_type.zero,
                )
            cute.arch.cp_async_commit_group()
            if row < M or tiler_mn[0] == 1:
                rstd = mRstd[row]
            cute.arch.cp_async_wait_group(1)
            cute.autovec_copy(tXsX[None, None, None, stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            cute.autovec_copy(tXsdOut[None, None, None, stage], tXrdOut)
            dout = tXrdOut.load().to(cute.Float32)
            x_hat = x * rstd
            wdy = dout * weight
            if cutlass.const_expr(self.cluster_n > 1):
                cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)
            mean_xhat_wdy = (
                utils.row_reduce(
                    x_hat * wdy,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, stage],
                    (
                        mbar_full_ptr + stage
                        if cutlass.const_expr(self.cluster_n > 1)
                        else None
                    ),
                    phase=consumer_phase,
                    init_val=0.0,
                )
                / shape[1]
            )

            if cutlass.const_expr(self.cluster_n > 1):
                # It's faster to have 1 lane per warp to signal the mbar, rather than all lanes
                # Requires adjusting the thread_count when initializing the mbar
                cute.arch.sync_warp()
                lane_idx = cute.arch.lane_idx()
                if lane_idx < self.cluster_n:
                    cute.arch.mbarrier_arrive(
                        mbar_empty_ptr + stage, peer_cta_rank_in_cluster=lane_idx
                    )

            if cutlass.const_expr(self.reload_wdy == "smem"):
                cute.autovec_copy(tXsdOut[None, None, None, stage], tXrdOut)
                dout = tXrdOut.load().to(cute.Float32)
                wdy = dout * weight

            dx = (wdy - x_hat * mean_xhat_wdy) * rstd
            tXrdX.store(dx.to(tXrdOut.element_type))
            if row < M or tiler_mn[0] == 1:
                tXgdX_cur = utils.coord_offset_i64(bidx, tXgdX, dim=3)[
                    None, None, None, 0
                ]
                cute.copy(copy_atom_store_dX, tXrdX, tXgdX_cur, pred=tXpX)
            # Accumulate weight gradients in fp32
            tXrdW.store(tXrdW.load() + dout * x_hat)

            stage ^= 1
            if stage == 0:
                consumer_phase ^= 1
                producer_phase ^= 1

        if cutlass.const_expr(self.cluster_n > 1):  # Prevent cluster from exiting early
            cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)

        if cutlass.const_expr(tiler_mn[0] > 1):
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
                for i in cutlass.range_constexpr(1, cutlass.const_expr(tiler_mn[0])):
                    tXrdW_other = cute.make_fragment_like(tXrdW)
                    tXsdW_other = cute.make_tensor(
                        tXsdW.iterator + i * sdW.stride[0], tXsdW.layout
                    )
                    cute.autovec_copy(tXsdW_other, tXrdW_other)
                    tXrdW.store(tXrdW.load() + tXrdW_other.load())
                cute.copy(copy_atom_store_dW, tXrdW, tXgdW, pred=tXpdW)
        else:
            # dw is already in fp32, so we can directly copy to global memory
            cute.copy(copy_atom_store_dW, tXrdW, tXgdW, pred=tXpdW)


def _rmsnorm_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    dout: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm backward pass.
    Args:
        x: Input tensor of shape (M, N)
        weight: Weight tensor of shape (N,)
        dout: Upstream gradients tensor of shape (M, N)
        rstd: Reciprocal standard deviation tensor of shape (M,)
    Returns:
        Tuple of (dx, dw) where:
        - dx: Input gradients tensor of same shape as x
        - dw: Weight gradients tensor of same shape as weight
    """
    assert x.dim() == 2, "Input must be 2D"
    assert weight.dim() == 1, "Weight must be 1D"
    assert (
        x.shape[-1] == weight.shape[0]
    ), "Last dimension of input must match weight dimension"
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ], "Unsupported dtype"

    assert weight.dtype in [
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ], "Weight must be float32, float16 or bfloat16"

    M, N = x.shape
    dx = torch.empty_like(x)

    device = x.device

    # This should be tuned on how many CTAs can be launched on each SM
    sm_count_multiple = (
        16
        if N <= 256
        else (8 if N <= 1024 else (4 if N <= 2048 else (2 if N <= 4096 else 1)))
    )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    # By right, if we're using cluster, this should be cluster_count not sm_count.
    # But for cluster >= 4, due to quantization we would need to query active max cluster.
    # Instead we just do sm_count * 2, which is reasonably larger than active_cluster_count to
    # avoid wave quantization.
    sm_count = (
        sm_count * sm_count_multiple
        if N <= 8192
        else sm_count // 2
        if N <= 16384
        else sm_count * 2
    )

    # Always store partial gradients in fp32 for numerical accuracy
    dw_partial = torch.empty(sm_count, N, device=device, dtype=torch.float32)

    dtype = torch2cute_dtype_map[x.dtype]

    convert_from_dlpack = lambda x: (
        from_dlpack(x.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    )
    x_tensor, dout_tensor, dx_tensor = [
        convert_from_dlpack(tensor) for tensor in (x, dout, dx)
    ]

    # Handle weight div based on weight dtype
    weight_dtype = torch2cute_dtype_map[weight.dtype]
    weight_tensor = utils.convert_from_dlpack(
        weight.detach(), leading_dim=0, divisibility=128 // weight_dtype.width
    )

    dw_partial_tensor = from_dlpack(
        dw_partial, assumed_align=16
    ).mark_compact_shape_dynamic(mode=0)
    rstd_tensor = from_dlpack(rstd.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (dtype, N, weight.dtype)
    if compile_key not in _rmsnorm_backward.compile_cache:
        rmsnorm_backward_op = RMSNormBackward(dtype, N)
        _rmsnorm_backward.compile_cache[compile_key] = cute.compile(
            rmsnorm_backward_op,
            x_tensor,
            weight_tensor,
            dout_tensor,
            rstd_tensor,
            dx_tensor,
            dw_partial_tensor,
            sm_count,
            current_stream,
        )

    _rmsnorm_backward.compile_cache[compile_key](
        x_tensor,
        weight_tensor,
        dout_tensor,
        rstd_tensor,
        dx_tensor,
        dw_partial_tensor,
        sm_count,
        current_stream,
    )
    # we have summed the partial gradients in fp32, now we convert back to the weight dtype
    dw = dw_partial.sum(dim=0).to(weight.dtype)
    return dx, dw


_rmsnorm_backward.compile_cache = {}


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x_shape_start = x.shape

        # Flatten input
        x = x.view(-1, x.shape[-1])

        out, rstd = _rmsnorm_fwd(x, weight, eps, return_rstd=True)
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        ctx.x_shape_start = x_shape_start

        return out.reshape(x_shape_start)

    @staticmethod
    def backward(ctx, dout):
        x, weight, rstd = ctx.saved_tensors
        x_shape_start = ctx.x_shape_start
        # Reshape dout to match the flattened shape used in forward
        dout = dout.view(-1, dout.shape[-1])
        dx, dw = _rmsnorm_backward(x, weight, dout, rstd)
        dx = dx.view(x_shape_start)
        # dx is returned for input gradient,
        # dw is returned for weight gradient,
        # None for eps gradient
        return dx, dw, None


@torch.library.custom_op("ads_mkl::quack_rmsnorm", mutates_args=())
def rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rmsnorm_fwd(x, weight, eps, return_rstd=True)


@rmsnorm.register_fake
def _(x, weight, eps=1e-5):
    rms = torch.sqrt(eps + torch.sum(x * x, dim=1) / x.shape[1])
    rstd = 1 / rms
    return (x * rstd[:, None] * weight[None, :]).to(torch.bfloat16), rstd


@torch.library.custom_op("ads_mkl::quack_rmsnorm_backward", mutates_args=())
def wrapped_rmsnorm_backward(
    x: torch.Tensor, weight: torch.Tensor, dout: torch.Tensor, rstd: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _rmsnorm_backward(x, weight, dout, rstd)


@wrapped_rmsnorm_backward.register_fake
def _(x: torch.Tensor, weight: torch.Tensor, dout: torch.Tensor, rstd: torch.Tensor):
    return dout.clone().detach(), weight.clone().detach()


def rmsnorm_backward(ctx, dout, _):
    x, weight, rstd = ctx.saved_tensors
    dx, dw = wrapped_rmsnorm_backward(x, weight, dout, rstd)
    # dw is returned for weight gradient, None for eps gradient
    return dx, dw, None


def setup_context(ctx, inputs, output):
    x, weight, eps = inputs
    out, rstd = output
    ctx.save_for_backward(x, weight, rstd)
    ctx.eps = eps


rmsnorm.register_autograd(rmsnorm_backward, setup_context=setup_context)
