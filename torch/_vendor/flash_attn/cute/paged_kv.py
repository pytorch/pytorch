from typing import Type
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import Int32, const_expr

from . import utils
from ..quack.cute_dsl_utils import ParamsBase
from cutlass.cute import FastDivmodDivisor

import math


@dataclass
class PagedKVManager(ParamsBase):
    mPageTable: cute.Tensor
    mK_paged: cute.Tensor
    mV_paged: cute.Tensor
    thread_idx: Int32

    page_size_divmod: FastDivmodDivisor
    seqlen_k: Int32
    leftpad_k: Int32
    n_block_size: Int32
    num_threads: cutlass.Constexpr[Int32]
    head_dim_padded: cutlass.Constexpr[Int32]
    head_dim_v_padded: cutlass.Constexpr[Int32]

    arch: cutlass.Constexpr[Int32]
    v_gmem_transposed: cutlass.Constexpr[bool]

    gmem_threads_per_row: cutlass.Constexpr[Int32]
    page_entry_per_thread: Int32
    async_copy_elems: Int32

    gmem_tiled_copy_KV: cute.TiledCopy
    gmem_thr_copy_KV: cute.TiledCopy
    tPrPage: cute.Tensor
    tPrPageOffset: cute.Tensor
    tKpK: cute.Tensor
    tVpV: cute.Tensor

    @staticmethod
    def create(
        mPageTable: cute.Tensor,
        mK_paged: cute.Tensor,
        mV_paged: cute.Tensor,
        page_size_divmod: FastDivmodDivisor,
        bidb: Int32,
        bidh: Int32,
        thread_idx: Int32,
        seqlen_k: Int32,
        leftpad_k: Int32,
        n_block_size: cutlass.Constexpr[Int32],
        head_dim_padded: cutlass.Constexpr[Int32],
        head_dim_v_padded: cutlass.Constexpr[Int32],
        num_threads: cutlass.Constexpr[Int32],
        dtype: Type[cutlass.Numeric],
        arch: cutlass.Constexpr[int] = 100,
    ):
        # SM100 transposes V in gmem to (dv, page_size, num_pages);
        # SM90 keeps V as (page_size, dv, num_pages), same layout as K.
        v_gmem_transposed = arch != 90
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // dtype.width
        dtype_bytes = dtype.width // 8
        gmem_k_block_size = math.gcd(
            head_dim_padded,
            head_dim_v_padded,
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
        page_entry_per_thread = n_block_size // num_threads

        tPrPage = cute.make_rmem_tensor((page_entry_per_thread,), Int32)
        tPrPageOffset = cute.make_rmem_tensor((page_entry_per_thread,), Int32)

        mPageTable = mPageTable[bidb, None]
        mK_paged = mK_paged[None, None, bidh, None]
        mV_paged = mV_paged[None, None, bidh, None]

        cK = cute.make_identity_tensor((n_block_size, head_dim_padded))
        tKcK = gmem_thr_copy_KV.partition_S(cK)
        tKpK = utils.predicate_k(tKcK, limit=mK_paged.shape[1])

        if const_expr(head_dim_padded == head_dim_v_padded):
            tVpV = tKpK
        else:
            cV = cute.make_identity_tensor((n_block_size, head_dim_v_padded))
            tVcV = gmem_thr_copy_KV.partition_S(cV)
            # When V is transposed in gmem, dv is shape[0]; otherwise dv is shape[1] (same as K)
            V_limit = cute.size(mV_paged.shape[0 if v_gmem_transposed else 1])
            tVpV = utils.predicate_k(tVcV, limit=V_limit)

        return PagedKVManager(
            mPageTable,
            mK_paged,
            mV_paged,
            thread_idx,
            page_size_divmod,
            seqlen_k,
            leftpad_k,
            n_block_size,
            num_threads,
            head_dim_padded,
            head_dim_v_padded,
            arch,
            v_gmem_transposed,
            gmem_threads_per_row,
            page_entry_per_thread,
            async_copy_elems,
            gmem_tiled_copy_KV,
            gmem_thr_copy_KV,
            tPrPage,
            tPrPageOffset,
            tKpK,
            tVpV,
        )

    @cute.jit
    def load_page_table(self, n_block: Int32):
        for i in cutlass.range(self.page_entry_per_thread, unroll=1):
            row = (
                i * self.num_threads
                + (self.thread_idx % self.gmem_threads_per_row)
                * (self.num_threads // self.gmem_threads_per_row)
                + (self.thread_idx // self.gmem_threads_per_row)
            )
            row_idx = n_block * self.n_block_size + row

            page_idx, page_offset = divmod(row_idx + self.leftpad_k, self.page_size_divmod)

            is_valid = (
                (i + 1) * self.num_threads <= self.n_block_size or row < self.n_block_size
            ) and row_idx < self.seqlen_k
            page = self.mPageTable[page_idx] if is_valid else 0

            self.tPrPage[i] = page
            self.tPrPageOffset[i] = page_offset

    @cute.jit
    def compute_X_ptr(self, K_or_V: str, d_offset: int = 0):
        tPrXPtr = cute.make_rmem_tensor((self.page_entry_per_thread,), cutlass.Int64)
        mX = self.mK_paged if const_expr(K_or_V == "K") else self.mV_paged
        # K is always (page_size, d, num_pages). V matches K when not transposed,
        # but is (dv, page_size, num_pages) when transposed (SM100).
        transposed = const_expr(K_or_V == "V" and self.v_gmem_transposed)
        for i in cutlass.range(self.page_entry_per_thread, unroll=1):
            page = self.tPrPage[i]
            page_offset = self.tPrPageOffset[i]
            if const_expr(transposed):
                tPrXPtr[i] = utils.elem_pointer(mX, (d_offset, page_offset, page)).toint()
            else:
                tPrXPtr[i] = utils.elem_pointer(mX, (page_offset, d_offset, page)).toint()
        return tPrXPtr

    @cute.jit
    def _flatten_smem_sm100(self, sX: cute.Tensor, K_or_V: str):
        """Flatten SM100 smem ((a,b), cta_split, k) to (a,(b,k)); transpose V to (d,page_size)."""
        sX_pi = cute.make_tensor(
            sX.iterator,
            cute.make_layout(
                (sX.shape[0][0], (sX.shape[0][1], sX.shape[2])),
                stride=(sX.stride[0][0], (sX.stride[0][1], sX.stride[2])),
            ),
        )
        if const_expr(K_or_V == "V"):
            sX_pi = cute.make_tensor(sX_pi.iterator, cute.select(sX_pi.layout, mode=[1, 0]))
        return sX_pi

    @cute.jit
    def _copy_row_async(
        self,
        tXsX: cute.Tensor,
        tXcX: cute.Tensor,
        mX_paged_cur_copy: cute.Tensor,
        m: Int32,
        should_load: cute.Tensor,
    ):
        """Issue cp.async copies for one row across all k-tiles."""
        for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
            ki = tXcX[0, 0, k][1] // self.async_copy_elems
            mX_paged_cur_copy_ki = mX_paged_cur_copy[None, ki]
            tXsX_k = tXsX[None, m, k]
            mX_paged_cur_copy_ki = cute.make_tensor(mX_paged_cur_copy_ki.iterator, tXsX_k.layout)
            cute.copy(
                self.gmem_tiled_copy_KV,
                mX_paged_cur_copy_ki,
                tXsX_k,
                pred=should_load,
            )

    @cute.jit
    def load_KV(self, n_block: Int32, sX: cute.Tensor, K_or_V: str):
        assert K_or_V in ("K", "V")

        tPrXPtr = self.compute_X_ptr(K_or_V)

        if const_expr(self.arch == 90):
            # SM90: sX is already stage-sliced by caller (sK[None, None, stage]).
            # Flatten hierarchical modes to get (n_block_size, head_dim).
            sX_pi = cute.group_modes(sX, 0, 1)
            # SM90 does NOT transpose V here (it's transposed via utils.transpose_view before MMA)
        else:
            sX_pi = self._flatten_smem_sm100(sX, K_or_V)

        head_dim = self.head_dim_v_padded if const_expr(K_or_V == "V") else self.head_dim_padded
        cX = cute.make_identity_tensor((self.n_block_size, head_dim))
        tXsX = self.gmem_thr_copy_KV.partition_D(sX_pi)
        tXcX = self.gmem_thr_copy_KV.partition_S(cX)
        tXc0X = self.gmem_thr_copy_KV.get_slice(0).partition_S(cX)

        seqlenk_row_limit = (
            self.seqlen_k - n_block * self.n_block_size - tXcX[0][0] if n_block >= 0 else 0
        )
        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            row_valid = tXc0X[0, m, 0][0] < seqlenk_row_limit
            should_load = cute.make_fragment_like(tXsX[(0, None), m, 0], cute.Boolean)
            should_load.fill(row_valid)

            x_ptr_i64 = utils.shuffle_sync(
                tPrXPtr[m // self.gmem_threads_per_row],
                m % self.gmem_threads_per_row,
                width=self.gmem_threads_per_row,
            )
            x_gmem_ptr = cute.make_ptr(
                self.mK_paged.element_type, x_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            mX_paged_cur = cute.make_tensor(x_gmem_ptr, cute.make_layout((head_dim,)))
            mX_paged_cur_copy = cute.tiled_divide(mX_paged_cur, (self.async_copy_elems,))
            self._copy_row_async(tXsX, tXcX, mX_paged_cur_copy, m, should_load)
