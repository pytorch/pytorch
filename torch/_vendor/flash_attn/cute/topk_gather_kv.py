from typing import Type, Optional
from dataclasses import dataclass
import operator

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync
from cutlass import Int32, Uint32, const_expr, Boolean

from . import utils
from .utils import warp_reduce
from ..quack.cute_dsl_utils import ParamsBase

import math


@dataclass
class CpasyncGatherKVManager(ParamsBase):
    mIndexTopk: cute.Tensor
    sBitmask: Optional[cute.Tensor]

    cta_rank_in_cluster: Int32
    thread_idx: Int32
    warp_idx: Int32

    topk_length: Int32
    seqlen_k_limit: Int32
    tile_n: Int32
    num_threads: cutlass.Constexpr[Int32]
    hdim: cutlass.Constexpr[Int32]
    hdim_v: cutlass.Constexpr[Int32]
    num_hdimv_splits: cutlass.Constexpr[Int32]
    cta_group_size: cutlass.Constexpr[Int32]

    gmem_threads_per_row: cutlass.Constexpr[Int32]
    topk_indices_per_thread: Int32
    async_copy_elems: Int32

    gmem_tiled_copy_KV: cute.TiledCopy
    gmem_thr_copy_KV: cute.TiledCopy

    rTopk: cute.Tensor
    rTopkHalf: cute.Tensor
    # for bitmask
    rTopk_NonInterleaved: cute.Tensor

    pipeline_bitmask: Optional[pipeline.PipelineAsync]
    cpasync_barrier: Optional[pipeline.NamedBarrier]

    disable_bitmask: cutlass.Constexpr[Boolean]

    @staticmethod
    def create(
        mIndexTopk: cute.Tensor,
        cta_rank_in_cluster: Int32,
        thread_idx: Int32,
        warp_idx: Int32,
        topk_length: Int32,
        seqlen_k_limit: Int32,
        tile_n: cutlass.Constexpr[Int32],
        hdim: cutlass.Constexpr[Int32],
        hdim_v: cutlass.Constexpr[Int32],
        num_hdimv_splits: cutlass.Constexpr[Int32],
        num_threads: cutlass.Constexpr[Int32],
        dtype: Type[cutlass.Numeric],
        cta_group_size: cutlass.Constexpr[Int32],
        cpasync_barrier: Optional[pipeline.NamedBarrier] = None,
        disable_bitmask: cutlass.Constexpr[Boolean] = False,
        sBitmask: Optional[cute.Tensor] = None,
        pipeline_bitmask: Optional[pipeline.PipelineAsync] = None,
    ):
        assert tile_n % num_threads == 0
        assert num_threads == 128
        assert hdim % 64 == 0
        assert (hdim_v // num_hdimv_splits // cta_group_size) % 64 == 0
        assert num_threads % cute.arch.WARP_SIZE == 0
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // dtype.width
        dtype_bytes = dtype.width // 8
        # assumes hdim is never part of transposed operand
        gmem_k_block_size = math.gcd(
            hdim,
            hdim_v // num_hdimv_splits // cta_group_size,
            128 // dtype_bytes,
        )
        assert gmem_k_block_size % async_copy_elems == 0
        gmem_threads_per_row = gmem_k_block_size // async_copy_elems
        assert cute.arch.WARP_SIZE % gmem_threads_per_row == 0
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        thr_layout = cute.make_ordered_layout(
            (num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_KV = cute.make_tiled_copy_tv(atom_async_copy, thr_layout, val_layout)
        gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(thread_idx)
        topk_indices_per_thread = tile_n // num_threads

        rTopk = cute.make_rmem_tensor((topk_indices_per_thread,), Int32)
        rTopkHalf = cute.make_rmem_tensor((topk_indices_per_thread,), Int32)
        rTopk_NonInterleaved = cute.make_rmem_tensor((topk_indices_per_thread,), Int32)

        return CpasyncGatherKVManager(
            mIndexTopk,
            sBitmask,
            cta_rank_in_cluster,
            thread_idx,
            warp_idx,
            topk_length,
            seqlen_k_limit,
            tile_n,
            num_threads,
            hdim,
            hdim_v,
            num_hdimv_splits,
            cta_group_size,
            gmem_threads_per_row,
            topk_indices_per_thread,
            async_copy_elems,
            gmem_tiled_copy_KV,
            gmem_thr_copy_KV,
            rTopk,
            rTopkHalf,
            rTopk_NonInterleaved,
            pipeline_bitmask,
            cpasync_barrier,
            disable_bitmask,
        )

    @cute.jit
    def load_index_topk(
        self,
        n_block: Int32,
        transpose: bool,
    ):
        entries_per_thread = self.topk_indices_per_thread
        rTopk = self.rTopk if const_expr(transpose) else self.rTopkHalf

        for i in cutlass.range_constexpr(entries_per_thread):
            row = (
                i * self.num_threads
                + (self.thread_idx % self.gmem_threads_per_row)
                * (self.num_threads // self.gmem_threads_per_row)
                + (self.thread_idx // self.gmem_threads_per_row)
            )
            # need this if not offset in load_X
            # if const_expr(not transpose):
            #     row += self.cta_rank_in_cluster * (self.tile_n//self.cta_group_size)
            #     row = row % self.tile_n
            row_idx = n_block * self.tile_n + row
            rTopk[i] = self.mIndexTopk[row_idx]

            if const_expr(not transpose and not self.disable_bitmask):
                row_non_interleaved = i * self.num_threads + self.thread_idx
                row_idx_non_interleaved = n_block * self.tile_n + row_non_interleaved
                self.rTopk_NonInterleaved[0] = self.mIndexTopk[row_idx_non_interleaved]

    @cute.jit
    def compute_bitmask(
        self,
        producer_state_bitmask,
    ):
        assert self.pipeline_bitmask is not None, "pipeline_bitmask not provided"
        assert self.cpasync_barrier is not None, "cpasync barrier not provided"

        lane_idx = cute.arch.lane_idx()
        assert cute.size(self.rTopk_NonInterleaved) == 1
        bitmask = Uint32(0)

        # Step 1. Construct per-thread bitmask
        topk_idx = self.rTopk_NonInterleaved[0]
        is_valid = topk_idx >= 0 and topk_idx < self.seqlen_k_limit
        if is_valid:
            bitmask = Uint32(1 << lane_idx)

        # Step 2. Warp shuffle bitwise OR = add since indices are exclusive.
        bitmask = warp_reduce(bitmask, operator.add)

        self.pipeline_bitmask.producer_acquire(producer_state_bitmask)
        # store to smem and sync threads
        if lane_idx == 0:
            self.sBitmask[self.warp_idx, producer_state_bitmask.index] = bitmask
        self.cpasync_barrier.arrive_and_wait()

        self.pipeline_bitmask.producer_commit(producer_state_bitmask)
        producer_state_bitmask.advance()
        return producer_state_bitmask

    @cute.jit
    def compute_X_ptr(
        self,
        mX: cute.Tensor,
        transpose: bool,
        d_offset: int = 0,
    ):
        entries_per_thread = self.topk_indices_per_thread
        tPrXPtr = cute.make_rmem_tensor((entries_per_thread,), cutlass.Int64)
        tPrRowValid = cute.make_rmem_tensor((entries_per_thread,), cutlass.Int32)
        rTopk = self.rTopk if const_expr(transpose) else self.rTopkHalf

        for i in cutlass.range_constexpr(entries_per_thread):
            topk_idx = rTopk[i]
            if const_expr(not self.disable_bitmask):
                row_valid = topk_idx >= 0 and topk_idx < self.seqlen_k_limit
                tPrRowValid[i] = row_valid
            if const_expr(not transpose):
                tPrXPtr[i] = utils.elem_pointer(mX, (topk_idx, d_offset)).toint()
            else:
                tPrXPtr[i] = utils.elem_pointer(mX, (d_offset, topk_idx)).toint()

        return tPrXPtr, tPrRowValid

    @cute.jit
    def load_X(
        self,
        mX: cute.Tensor,
        sX: cute.Tensor,
        transpose: bool,
        K_or_V: str,
        d_offset: int = 0,
    ):
        assert K_or_V in ("K", "V")
        cta_tile_n = self.tile_n if const_expr(transpose) else self.tile_n // self.cta_group_size
        head_dim = self.hdim if const_expr(K_or_V == "K") else self.hdim_v // self.num_hdimv_splits
        if const_expr(transpose):
            head_dim = head_dim // self.cta_group_size
        order = (1, 0) if const_expr(transpose) else (0, 1)

        sX_nd_layout = cute.make_ordered_layout((cta_tile_n, head_dim), order=order)
        sX_nd = cute.composition(sX, sX_nd_layout)

        cX = cute.make_identity_tensor((cta_tile_n, head_dim))
        tXsX = self.gmem_thr_copy_KV.partition_D(sX_nd)
        tXcX = self.gmem_thr_copy_KV.partition_S(cX)

        tPrXPtr, tPrRowValid = self.compute_X_ptr(mX, transpose, d_offset)

        if const_expr(not transpose):
            offset = self.cta_rank_in_cluster * (self.gmem_threads_per_row // self.cta_group_size)
        else:
            offset = 0

        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            if const_expr(not self.disable_bitmask):
                row_valid = utils.shuffle_sync(
                    tPrRowValid[m // self.gmem_threads_per_row],
                    (m + offset) % self.gmem_threads_per_row,
                    width=self.gmem_threads_per_row,
                )
                should_load = cute.make_fragment_like(tXsX[(0, None), m, 0], Boolean)
                should_load.fill(Boolean(row_valid))
            x_ptr_i64 = utils.shuffle_sync(
                tPrXPtr[m // self.gmem_threads_per_row],
                (m + offset) % self.gmem_threads_per_row,
                width=self.gmem_threads_per_row,
            )
            x_gmem_ptr = cute.make_ptr(
                mX.element_type, x_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            mX_cur = cute.make_tensor(x_gmem_ptr, cute.make_layout((head_dim,)))
            mX_cur_copy = cute.tiled_divide(mX_cur, (self.async_copy_elems,))

            for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                ki = tXcX[0, 0, k][1] // self.async_copy_elems
                mX_cur_copy_ki = mX_cur_copy[None, ki]
                tXsX_k = tXsX[None, m, k]
                mX_cur_copy_ki = cute.make_tensor(mX_cur_copy_ki.iterator, tXsX_k.layout)
                cute.copy(
                    self.gmem_tiled_copy_KV,
                    mX_cur_copy_ki,
                    tXsX_k,
                    pred=should_load if const_expr(not self.disable_bitmask) else None,
                )
