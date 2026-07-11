# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_bwd_preprocess_kernel.h
# from Cutlass C++ to Cute-DSL.
#
# Computes D_i = (dO_i * O_i).sum(dim=-1), optionally adjusted for LSE gradient:
#   D'_i = D_i - dLSE_i
# This works because in the backward pass:
#   dS_ij = P_ij * (dP_ij - D_i)                     [standard]
# When LSE is differentiable, d(loss)/d(S_ij) gets an extra term dLSE_i * P_ij
# (since d(LSE_i)/d(S_ij) = P_ij), giving:
#   dS_ij = P_ij * (dP_ij - D_i) + dLSE_i * P_ij
#         = P_ij * (dP_ij - (D_i - dLSE_i))
# So the main backward kernel is unchanged; we just replace D with D' = D - dLSE here.
import math
import operator
from functools import partial
from typing import Callable, Type, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import Arch, BaseDSL

from ..quack import copy_utils, layout_utils

from . import utils
from .seqlen_info import SeqlenInfo
from ..quack.cute_dsl_utils import ParamsBase
from .tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from .pack_gqa import pack_gqa_layout


class FlashAttentionBackwardPreprocess:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int,
        tile_m: int = 128,
        num_threads: int = 256,
        use_padded_offsets: bool = True,
        nheads_major: bool = False,
        pack_gqa: bool = False,
        qhead_per_kvhead: int = 1,
        nheads_kv: int = 1,
    ):
        """
        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int
        :param num_threads: number of threads
        :type num_threads: int
        """
        self.use_pdl = BaseDSL._get_dsl().get_arch_enum() >= Arch.sm_90a
        self.dtype = dtype
        self.tile_m = tile_m
        # padding head_dim to a multiple of 32 as k_block_size
        hdim_multiple_of = 32
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.num_threads = num_threads
        self.use_padded_offsets = use_padded_offsets
        self.nheads_major = nheads_major
        self.pack_gqa = pack_gqa
        self.qhead_per_kvhead = qhead_per_kvhead
        self.nheads_kv = nheads_kv

    @staticmethod
    def can_implement(dtype, head_dim, tile_m, num_threads) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int
        :param num_threads: number of threads
        :type num_threads: int

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if num_threads < tile_m:  # For multiplying lse with log2
            return False
        return True

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        # We want kBlockKGmem to be a power of 2 so that when we do the summing,
        # it's just between threads in the same warp
        gmem_k_block_size = (
            128
            if self.head_dim_v_padded % 128 == 0
            else (
                64
                if self.head_dim_v_padded % 64 == 0
                else (32 if self.head_dim_v_padded % 32 == 0 else 16)
            )
        )
        num_copy_elems = 128 // self.dtype.width
        threads_per_row = gmem_k_block_size // num_copy_elems
        self.gmem_tiled_copy_O = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, self.num_threads, num_copy_elems
        )
        universal_copy_bits = 128
        num_copy_elems_dQaccum = universal_copy_bits // Float32.width
        assert (
            self.tile_m * self.head_dim_padded // num_copy_elems_dQaccum
        ) % self.num_threads == 0
        self.gmem_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(
            Float32, self.num_threads, num_copy_elems_dQaccum
        )

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,  # (batch, seqlen, nheads, head_dim_v) or (total_q, nheads, head_dim_v)
        mdO: cute.Tensor,  # same shape as mO
        mPdPsum: cute.Tensor,  # (batch, nheads, seqlen_padded) or (nheads, total_q_padded)
        mLSE: Optional[cute.Tensor],  # (batch, nheads, seqlen) or (nheads, total_q)
        mLSElog2: Optional[cute.Tensor],  # same shape as mPdPsum
        # (batch, nheads, seqlen_padded * head_dim_v) or (nheads, total_q_padded * head_dim_v)
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],  # (batch + 1,)
        mSeqUsedQ: Optional[cute.Tensor],  # (batch,)
        mdLSE: Optional[cute.Tensor],  # (batch, nheads, seqlen) or (nheads, total_q)
        mRowMax: Optional[cute.Tensor],  # (b, s, n, h) or (t, n, h)
        mScaleP: Optional[cute.Tensor],  # == mRowMax
        softmax_scale: Float32,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mO.element_type == mdO.element_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mO.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mPdPsum.element_type not in [Float32]):
            raise TypeError("PdPsum tensor must be Float32")
        if const_expr(mdQaccum is not None):
            assert self.nheads_major is False
            assert self.pack_gqa is False
            assert self.use_padded_offsets is True
            if const_expr(mdQaccum.element_type not in [Float32]):
                raise TypeError("dQaccum tensor must be Float32")
        if const_expr(mLSE is not None):
            if const_expr(mLSE.element_type not in [Float32]):
                raise TypeError("LSE tensor must be Float32")
        if const_expr(mLSElog2 is not None):
            if const_expr(mLSElog2.element_type not in [Float32]):
                raise TypeError("LSElog2 tensor must be Float32")
        if const_expr(mdLSE is not None):
            if const_expr(mdLSE.element_type not in [Float32]):
                raise TypeError("dLSE tensor must be Float32")
        if const_expr(mScaleP is not None):
            assert self.nheads_major is True
            assert self.pack_gqa is True
            assert mRowMax is not None
            if const_expr(mScaleP.element_type not in [Float32]):
                raise TypeError("ScaleP tensor must be Float32")
            if const_expr(mRowMax.element_type not in [Float32]):
                raise TypeError("RowMax tensor must be Float32")

        self._setup_attributes()

        # (b, s, h, d)  -> (s, d, h, b)  or
        # (total, h, d) -> (total, d, h)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mO, mdO = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=QO_layout_transpose))
            for mX in (mO, mdO)
        ]

        if const_expr(not self.nheads_major):
            # (batch, nheads, seqlen) -> (seqlen, nheads, batch) or
            # (nheads, total_q) -> (total_q, nheads)
            transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        else:
            # (batch, seqlen, nheads) -> (seqlen, nheads, batch) or
            # (total_q, nheads) -> (total_q, nheads)
            transpose = [1, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 1]
        mPdPsum, mLSE, mLSElog2, mdLSE, mdQaccum = [
            layout_utils.select(mX, transpose) if mX is not None else None
            for mX in (mPdPsum, mLSE, mLSElog2, mdLSE, mdQaccum)
        ]

        # (b, s, n, h) => (s, n, h, b) or
        # (total, n, h) == (total, n, h)
        rowmax_layout_transpose = [1, 2, 3, 0] if const_expr(mCuSeqlensQ is None) else [0, 1, 2]
        if const_expr(mRowMax is not None):
            mRowMax = layout_utils.select(mRowMax, rowmax_layout_transpose)
        if const_expr(mScaleP is not None):
            mScaleP = layout_utils.select(mScaleP, rowmax_layout_transpose)

        # pack gqa
        if const_expr(self.pack_gqa):
            mO, mdO, mRowMax, mScaleP = [
                pack_gqa_layout(mX, self.qhead_per_kvhead, self.nheads_kv, head_idx=2)
                if mX is not None
                else None
                for mX in (mO, mdO, mRowMax, mScaleP)
            ]
            mPdPsum, mLSE, mLSElog2, mdLSE = [
                pack_gqa_layout(mX, self.qhead_per_kvhead, self.nheads_kv, head_idx=1)
                if mX is not None
                else None
                for mX in (mPdPsum, mLSE, mLSElog2, mdLSE)
            ]

        # mO: (s, d, h, b) or (total, d, h)
        if const_expr(mCuSeqlensQ is not None):
            TileScheduler = SingleTileVarlenScheduler
            num_head = mO.shape[2]
            num_batch = mCuSeqlensQ.shape[0] - 1
        else:
            TileScheduler = SingleTileScheduler
            num_head = mO.shape[2]
            num_batch = mO.shape[3]

        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mO.shape[0], self.tile_m),
            num_head=num_head,
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=0,
            headdim_v=mO.shape[1],
            total_q=cute.size(mO.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mO.shape[0]) * cute.size(mO.shape[3]),
            tile_shape_mn=(self.tile_m, 1),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            mO,
            mdO,
            mPdPsum,
            mLSE,
            mLSElog2,
            mdQaccum,
            mCuSeqlensQ,
            mSeqUsedQ,
            mdLSE,
            mRowMax,
            mScaleP,
            softmax_scale_log2,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_dQaccum,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mPdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mdLSE: Optional[cute.Tensor],
        mRowMax: Optional[cute.Tensor],
        mScaleP: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        # This kernel is launched with use_pdl=True, so the GPU may start executing it in
        # "prologue" mode while the previous stream kernel is still running. We must wait
        # before touching any upstream GMEM outputs (mO, mdO, mLSE); otherwise we risk
        # reading a partially-written dout, which silently corrupts dpsum = sum(O * dO) and
        # propagates to dQ/dK via dS = P * (dP - dpsum).
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        if work_tile.is_valid_tile:
            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////
            seqlen_static = mO.shape[0] if const_expr(not self.pack_gqa) else mO.shape[0][1]
            seqlen = SeqlenInfo.create(
                batch_idx, seqlen_static, mCuSeqlensQ, mSeqUsedQ, tile=self.tile_m
            )
            # (seqlen, dv)
            mO_cur, mdO_cur = [
                seqlen.offset_batch(mX, batch_idx, dim=3)[None, None, head_idx] for mX in (mO, mdO)
            ]
            mPdPsum_cur = seqlen.offset_batch(
                mPdPsum, batch_idx, dim=2, padded=self.use_padded_offsets
            )[None, head_idx]
            headdim_v = mO_cur.shape[1]
            seqlen_q = (
                seqlen.seqlen
                if const_expr(not self.pack_gqa)
                else seqlen.seqlen * self.qhead_per_kvhead
            )
            seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)
            seqlen_limit = seqlen_q - m_block * self.tile_m

            lse = None
            if const_expr(mLSE is not None):
                mLSE_cur = seqlen.offset_batch(mLSE, batch_idx, dim=2)[None, head_idx]
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                lse = Float32.inf
                if tidx < seqlen_limit:
                    lse = gLSE[tidx]

            blk_shape = (self.tile_m, self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, blk_shape, (m_block, 0))
            gdO = cute.local_tile(mdO_cur, blk_shape, (m_block, 0))
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            # (CPY_Atom, CPY_M, CPY_K)
            tOgO = gmem_thr_copy_O.partition_S(gO)
            tOgdO = gmem_thr_copy_O.partition_S(gdO)
            cO = cute.make_identity_tensor(blk_shape)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cO)
            tOpO = None
            if const_expr(self.check_hdim_v_oob):
                tOpO = copy_utils.predicate_k(tOcO, limit=headdim_v)
            # Each copy will use the same predicate
            copy = partial(copy_utils.copy, pred=tOpO)

            tOrO = cute.make_rmem_tensor_like(tOgO)
            tOrdO = cute.make_rmem_tensor_like(tOgdO)
            if const_expr(self.check_hdim_v_oob):
                tOrO.fill(0.0)
                tOrdO.fill(0.0)
            assert tOgO.shape == tOgdO.shape
            for m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
                # Instead of using tOcO, we using t0OcO and subtract the offset from the limit.
                # This is bc the entries of t0OcO are known at compile time.
                if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                    copy(tOgO[None, m, None], tOrO[None, m, None])
                    copy(tOgdO[None, m, None], tOrdO[None, m, None])
            # O and dO loads are done; signal that the next kernel can start.
            # Correctness is ensured by griddepcontrol_wait() in bwd_sm90 before it reads our outputs.
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()
            # Sum across the "k" dimension
            pdpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
                cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1)
            )
            threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
            assert cute.arch.WARP_SIZE % threads_per_row == 0
            pdpsum = utils.warp_reduce(pdpsum, operator.add, width=threads_per_row)
            PdP_sum = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
            PdP_sum.store(pdpsum)

            # If dLSE is provided, compute D' = D - dLSE (see module docstring for derivation).
            gdLSE = None
            if const_expr(mdLSE is not None):
                mdLSE_cur = seqlen.offset_batch(mdLSE, batch_idx, dim=2)[None, head_idx]
                gdLSE = cute.local_tile(mdLSE_cur, (self.tile_m,), (m_block,))

            # Write PdPsum from rmem -> gmem
            gPdPsum = cute.local_tile(mPdPsum_cur, (self.tile_m,), (m_block,))
            # Only the thread corresponding to column 0 writes out the PdPsum to gmem
            if tOcO[0, 0, 0][1] == 0:
                for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                    row = tOcO[0, m, 0][0]
                    PdPsum_val = 0.0
                    if row < seqlen_limit:
                        PdPsum_val = PdP_sum[m]
                        if const_expr(mdLSE is not None):
                            PdPsum_val -= gdLSE[row]
                    gPdPsum[row] = PdPsum_val

            # Clear dQaccum
            if const_expr(mdQaccum is not None):
                mdQaccum_cur = seqlen.offset_batch(
                    mdQaccum,
                    batch_idx,
                    dim=2,
                    padded=self.use_padded_offsets,
                    multiple=self.head_dim_padded,
                )[None, head_idx]
                blkdQaccum_shape = (self.tile_m * self.head_dim_padded,)
                gdQaccum = cute.local_tile(mdQaccum_cur, blkdQaccum_shape, (m_block,))
                gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
                tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
                zero = cute.make_rmem_tensor_like(tdQgdQaccum)
                zero.fill(0.0)
                cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum)

            LOG2_E = math.log2(math.e)
            lse_log2 = lse * LOG2_E if lse != -Float32.inf else 0.0
            if const_expr(mLSElog2 is not None):
                mLSElog2_cur = seqlen.offset_batch(
                    mLSElog2, batch_idx, dim=2, padded=self.use_padded_offsets
                )[None, head_idx]
                gLSElog2 = cute.local_tile(mLSElog2_cur, (self.tile_m,), (m_block,))
                LOG2_E = math.log2(math.e)
                if tidx < seqlen_q_rounded - m_block * self.tile_m:
                    gLSElog2[tidx] = lse_log2

            if const_expr(mRowMax is not None):
                assert mLSE is not None
                # (s, n)
                mRowMax_cur, mScaleP_cur = [
                    seqlen.offset_batch(mX, batch_idx, dim=3)[None, None, head_idx]
                    for mX in (mRowMax, mScaleP)
                ]
                # (tile_m, n)
                gRowMax, gScaleP = [
                    cute.local_tile(mX, (self.tile_m,), (m_block, None))
                    for mX in (mRowMax_cur, mScaleP_cur)
                ]

                assert self.tile_m <= self.num_threads
                if const_expr(self.tile_m == self.num_threads) or tidx < self.tile_m:
                    for n in cutlass.range(gRowMax.shape[1], unroll=4):
                        row_max = gRowMax[tidx, n]
                        scale = 0.0
                        if row_max != -Float32.inf and lse != -Float32.inf:
                            scale = softmax_scale_log2 * row_max - lse_log2
                            scale = cute.math.exp2(scale, fastmath=True)
                        gScaleP[tidx, n] = scale
