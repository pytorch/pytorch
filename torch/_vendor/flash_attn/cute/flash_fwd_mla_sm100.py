# Copyright (c) 2026, Colfax International.

import math
from functools import partial
from typing import Callable, Optional


import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64, Int32, Uint32, Boolean, const_expr
from cutlass.cute import FastDivmodDivisor
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.utils import ClcDynamicPersistentTileScheduler

from ..quack import copy_utils

from .pack_gqa import pack_gqa_layout, make_packgqa_tiled_tma_atom
from .paged_kv import PagedKVManager
from . import utils as fa_utils
from .seqlen_info import SeqlenInfoQK
from .block_info import BlockInfo
from .mask import AttentionMask
from . import blackwell_helpers as fa_sm100_utils
from .softmax import SoftmaxSm100
from .tile_scheduler import (
    ClcState,
    SchedulingMode,
    TileSchedulerArguments,
    TileSchedulerProtocol,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    ParamsBase,
)
from .fa_logging import fa_log, fa_printf
from .utils import smid, get_batch_from_cu_tensor

from .topk_gather_kv import CpasyncGatherKVManager


from .named_barrier import NamedBarrierFwdSm100_MLA2CTA


class FlashAttentionMLAForwardSm100:
    def __init__(
        self,
        is_causal: bool = False,
        use_cpasync_load_KV: bool = False,
        topk_length: int = 2048,
        is_topk_gather: bool = True,
        pack_gqa: bool = False,
        qhead_per_kvhead: int = 1,
        nheads_kv: int = 1,
        hdim: int = 64,
        hdimv: int = 512,
        has_seqused_q: bool = False,
        has_cu_seqlens_q: bool = False,
        disable_bitmask: bool = False,
        use_clc_scheduler: bool = True,
        has_qk: bool = True,
    ):
        self.is_causal = is_causal
        self.is_local = False
        self.pack_gqa = pack_gqa
        self.qhead_per_kvhead = qhead_per_kvhead
        assert qhead_per_kvhead <= 128
        self.nheads_kv = nheads_kv
        self.use_tma_O = True
        self.use_cpasync_load_KV = use_cpasync_load_KV
        self.use_tma_KV = not use_cpasync_load_KV
        self.topk_length = topk_length
        self.is_topk_gather = is_topk_gather
        if is_topk_gather:
            assert pack_gqa
            assert qhead_per_kvhead == 128, "require MQA 128 for DSA path"
            assert use_cpasync_load_KV
        # user-provided option if topk indices guaranteed in bounds
        self.disable_bitmask = disable_bitmask
        self.has_qk = has_qk

        # ==== tile scheduler ====
        self.is_persistent = False
        self.use_clc_scheduler = use_clc_scheduler
        self.sched_stages = 1
        self.scheduling_mode = (
            SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC
        )

        self.is_varlen_q = has_seqused_q or has_cu_seqlens_q
        self.use_packed_varlen_sched = has_cu_seqlens_q and qhead_per_kvhead == 128 and pack_gqa
        self.use_varlen_scheduler = self.is_varlen_q and not self.use_packed_varlen_sched

        if const_expr(self.use_varlen_scheduler):
            self.TileScheduler = SingleTileVarlenScheduler
        elif self.use_clc_scheduler:
            self.TileScheduler = SingleTileLPTScheduler
        else:
            self.TileScheduler = SingleTileScheduler

        fa_log(
            1,
            f"TileScheduler={self.TileScheduler.__name__}, scheduling_mode={self.scheduling_mode.name}",
        )

        # ==== thread info ====
        self.num_softmax_threads = 128
        self.num_epilogue_threads = 128
        self.num_load_threads = 32
        self.num_mma_threads = 32
        self.num_empty_threads = 32 if use_cpasync_load_KV else 64
        self.num_relay_threads = 32 if use_cpasync_load_KV else 0
        self.num_cpasync_load_threads = 128 if use_cpasync_load_KV else 0
        self.num_threads = (
            self.num_softmax_threads
            + self.num_epilogue_threads
            + self.num_load_threads
            + self.num_mma_threads
            + self.num_empty_threads
            + self.num_relay_threads
            + self.num_cpasync_load_threads
        )
        self.num_warps = self.num_threads // 32
        assert self.num_warps == 12 or self.num_warps == 16
        self.softmax_warp_indices = (0, 1, 2, 3)
        self.epilogue_warp_indices = (4, 5, 6, 7)
        self.load_warp_id = 8
        self.mma_warp_id = 9
        self.clc_scheduler_warp_id = 10
        self.relay_warp_id = 11
        self.empty_warp_ids = tuple(
            w
            for w, active in [
                (self.relay_warp_id, not use_cpasync_load_KV),
                (self.clc_scheduler_warp_id, not self.use_clc_scheduler),
            ]
            if active
        )
        self.cpasync_load_warp_indices = (12, 13, 14, 15)

        # ==== register usage ====
        if self.num_warps == 16:
            self.num_regs_load = 112
            self.num_regs_mma = 112
            self.num_regs_softmax = 192
            self.num_regs_epilogue = 128
            self.num_regs_cpasync = 80 if self.use_cpasync_load_KV else 0
            self.num_regs_other = 48
        else:
            self.num_regs_load = 168 - 40
            self.num_regs_mma = 168 - 40
            self.num_regs_softmax = 168 + 80
            self.num_regs_epilogue = 168 - 40
            self.num_regs_cpasync = 0
            self.num_regs_other = 48

        self.num_regs_per_thread = 168 if self.num_warps == 12 else 128
        self.num_regs_total = 504 if self.num_warps == 12 else 512

        assert (
            self.num_regs_mma
            + self.num_regs_softmax
            + self.num_regs_epilogue
            + self.num_regs_cpasync
            <= self.num_regs_total
        )

        # ==== 2cta info ====
        self.use_2cta_instrs = True
        self.cta_group = tcgen05.CtaGroup.TWO
        self.cta_group_size = 2
        self.cluster_shape_mn = (2, 1)
        self.cluster_shape_mnk = (2, 1, 1)

        # ==== problem shape info ====
        self.hdim = hdim
        self.hdimv = hdimv
        self.cta_tile_m = 64
        self.cluster_tile_m = self.cta_group_size * self.cta_tile_m
        self.tile_n = 128
        assert (
            pack_gqa is False
            or self.cluster_tile_m % qhead_per_kvhead == 0
            or qhead_per_kvhead % self.cluster_tile_m == 0
        )
        self.num_hdimv_splits = 2  # split hdimv in half for our Qv @ V^T and P @ V mmas.
        assert hdimv % 32 == 0
        assert self.topk_length % self.tile_n == 0 or not self.is_topk_gather
        self.epi_tile = (self.cta_tile_m, self.hdimv // self.num_hdimv_splits)
        self.tile_P = (self.cta_tile_m, self.tile_n)

        # ==== MMA info ====
        self.mma_tiler_QK = (
            self.cluster_tile_m,
            self.tile_n,
            self.hdim,
        )
        self.mma_tiler_QvV = (
            self.cluster_tile_m,
            self.tile_n,
            self.hdimv // self.num_hdimv_splits,
        )
        self.mma_tiler_PVt = (
            self.cluster_tile_m,
            self.hdimv // self.num_hdimv_splits,
            self.tile_n,
        )
        self.major_mode_Q = tcgen05.OperandMajorMode.K
        self.major_mode_Qvi = tcgen05.OperandMajorMode.K
        self.major_mode_K = tcgen05.OperandMajorMode.K
        self.major_mode_Vi = tcgen05.OperandMajorMode.K
        self.major_mode_Vti = tcgen05.OperandMajorMode.MN
        self.major_mode_P = tcgen05.OperandMajorMode.K
        self.operand_source_Q = tcgen05.OperandSource.SMEM
        self.operand_source_Qvi = tcgen05.OperandSource.SMEM
        self.operand_source_P = tcgen05.OperandSource.SMEM

        # ==== pipeline info ====
        self.num_stages_Q = 1
        self.num_stages_K = 1
        self.num_stages_Qv = 2
        self.num_stages_V = 4
        self.num_stages_S = 2
        # self.num_stages_P = 1 if has_qk else 2
        self.num_stages_P = 1
        self.num_stages_Oi = 1
        self.num_stages_sm_stats = 2
        self.num_stages_bitmask = 2
        assert self.num_stages_S == 2, "mainloops expect 2 stages for S"

        # ==== dtype info ====
        self.dtype_acc = Float32

        # ==== TMEM info ====
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_cols_S = self.tile_n // self.cta_group_size
        self.tmem_cols_Oi = (self.hdimv // self.num_hdimv_splits) // self.cta_group_size
        self.tmem_offset_S = [
            self.tmem_cols_S * stage for stage in range(self.num_stages_S)
        ]  # allocate 64 TMEM columns for each stage of S
        self.tmem_offset_O0 = self.tmem_cols_S * self.num_stages_S
        self.tmem_offset_O1 = self.tmem_offset_O0 + self.tmem_cols_Oi
        self.tmem_offsets_O = [self.tmem_offset_O0, self.tmem_offset_O1]
        self.total_tmem = self.tmem_offset_O1 + self.tmem_cols_Oi
        assert self.total_tmem <= self.tmem_alloc_cols, (
            f"Total TMEM columns allocated {self.total_tmem} exceeds capacity {self.tmem_alloc_cols}"
        )

    def _get_shared_storage_cls(self):
        self.buffer_align_bytes = 1024

        def smem_struct_align(dtype, staged_layout, disabled=False):
            if disabled:
                return cute.struct.MemRange[dtype, 0]
            return cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(staged_layout)],
                self.buffer_align_bytes,
            ]

        def mbar_struct(num_stages):
            return cute.struct.MemRange[Int64, 2 * num_stages]

        (sQ_struct, sK_struct, sQv_struct, sV_struct, sP_struct) = (
            smem_struct_align(dtype, layout, disabled)
            for dtype, layout, disabled in [
                (self.dtype_Q, self.sQ_layout_staged, not self.has_qk),
                (self.dtype_K, self.sK_layout_staged, not self.has_qk),
                (self.dtype_Qv, self.sQv_layout_staged, False),
                (self.dtype_V, self.sV_layout_staged, False),
                (self.dtype_P, self.sP_layout_staged, False),
            ]
        )
        sStats_struct = cute.struct.MemRange[Float32, cute.cosize(self.sStats_layout)]
        sScale_struct = cute.struct.MemRange[Float32, cute.cosize(self.sScale_layout)]
        sBitmask_struct = cute.struct.MemRange[Uint32, cute.cosize(self.sBitmask_layout)]

        (
            mbar_ptr_Q_struct,
            mbar_ptr_K_struct,
            mbar_ptr_Qv_struct,
            mbar_ptr_V_struct,
            mbar_ptr_S_struct,
            mbar_ptr_P_struct,
            mbar_ptr_O0_struct,
            mbar_ptr_O1_struct,
            mbar_sm_stats_struct,
            mbar_bitmask_struct,
        ) = (
            mbar_struct(n)
            for n in [
                self.num_stages_Q,
                self.num_stages_K,
                self.num_stages_Qv,
                self.num_stages_V,
                self.num_stages_S,
                self.num_stages_P,
                self.num_stages_Oi,
                self.num_stages_Oi,
                self.num_stages_sm_stats,
                self.num_stages_bitmask,
            ]
        )
        tmem_dealloc_mbar_struct = Int64
        tmem_holding_buf_struct = Int32

        self.sched_stages = 1
        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0

        @cute.struct
        class SharedStorage:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_Qv: mbar_ptr_Qv_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_S: mbar_ptr_S_struct
            mbar_ptr_P: mbar_ptr_P_struct
            mbar_ptr_O0: mbar_ptr_O0_struct
            mbar_ptr_O1: mbar_ptr_O1_struct
            mbar_ptr_K_cpasync: mbar_ptr_K_struct
            mbar_ptr_V_cpasync: mbar_ptr_V_struct
            mbar_ptr_sm_stats: mbar_sm_stats_struct
            mbar_ptr_bitmask: mbar_bitmask_struct
            tmem_dealloc_mbar: tmem_dealloc_mbar_struct
            tmem_holding_buf: tmem_holding_buf_struct
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.MemRange[Int32, clc_response_size]
            sO_empty_mbar_ptr: cutlass.Int64

            sRowMax: sStats_struct
            sRowSum: sStats_struct
            sScale: sScale_struct
            sBitmask: sBitmask_struct
            sQv: sQv_struct
            sQ: sQ_struct
            sK: sK_struct
            sV: sV_struct
            sP: sP_struct

        # print("smem bytes = ", SharedStorage.size_in_bytes())

        return SharedStorage

    # fmt: off
    @cute.jit
    def __call__(
        self,
        mQ: Optional[cute.Tensor],    # (b, s_q, h, d)     or (total_q, h, d)    if there is cu_seqlens_q
        mQv: cute.Tensor,             # (b, s_q, h, dv)    or (total_q, h, d)    if there is cu_seqlens_q
        mK: Optional[cute.Tensor],    # (b, s_k, h_k, d)   or (total_k, h_k, d)  if there is cu_seqlens_k  or (num_pages, page_size, h_k, d)  if there is page_table
        mV: cute.Tensor,              # (b, s_k, h_k, dv)  or (total_k, h_k, dv) if there is cu_seqlens_k  or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,              # (b, s_q, h, dv)    or (total_q, h, dv)   if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],  # (b, s_q, h)        or (total_q, h)       if there is cu_seqlens_q
        softmax_scale: Float32,
        mP: Optional[cute.Tensor] = None,           # (b, s_q, h, topk)            or (total_q, h, topk)           if there is cu_seqlens_q
        mRowMax: Optional[cute.Tensor] = None,      # (b, s_q, topk // tile_n, h)  or (total_q, topk // tile_n, h) if there is cu_seqlens_q
        mCuSeqlensQ: Optional[cute.Tensor] = None,  # (b + 1)
        mCuSeqlensK: Optional[cute.Tensor] = None,  # (b + 1)
        mSeqUsedQ: Optional[cute.Tensor] = None,    # (b)
        mSeqUsedK: Optional[cute.Tensor] = None,    # (b)
        mIndexTopk: Optional[cute.Tensor] = None,   # (b, s_q, topk)  or (total_q, topk) if there is cu_seqlens_q
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # fmt: on
        self.store_P = mP is not None
        self.store_row_max = mRowMax is not None

        if const_expr(self.has_qk):
            assert mQ is not None and mK is not None, "has_qk requires mQ and mK"
        else:
            assert mQ is None and mK is None, "not has_qk disallows mQ and mK"

        # ==== dtype info ====
        self.dtype_Q = mQ.element_type if self.has_qk else cutlass.BFloat16
        self.dtype_K = mK.element_type if self.has_qk else cutlass.BFloat16
        self.dtype_Qv = mQv.element_type
        self.dtype_V = mV.element_type
        self.dtype_P = mV.element_type
        self.dtype_O = mO.element_type

        if const_expr(self.store_P):
            assert mP.element_type == self.dtype_P

        # ==== Prepare Tensors ====
        new_stride = lambda mX: (
            *(cute.assume(s, divby=128 // mX.element_type.width) for s in mX.stride[:-1]),
            mX.stride[-1],
        )
        mQ, mQv, mK, mV, mO, mP = [
            cute.make_tensor(mX.iterator, cute.make_layout(mX.shape, stride=new_stride(mX)))
            if mX is not None
            else None
            for mX in (mQ, mQv, mK, mV, mO, mP)
        ]

        # (b, s, h, d)  -> (s, d, h, b)  or
        # (total, h, d) -> (total, d, h) or
        # (num_pages, page_size, h_k, d) -> (page_size, d, h_k, num_pages)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mQ, mQv, mO, mP = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=QO_layout_transpose))
            if mX is not None
            else None
            for mX in (mQ, mQv, mO, mP)
        ]
        mK, mV = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=KV_layout_transpose))
            if mX is not None
            else None
            for mX in (mK, mV)
        ]
        # (s_k, dv, h_k, b)  -> (dv, s_k, h_k, b) or
        # (total_k, dv, h_k) -> (dv, total_k, h_k)
        V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        mVt = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))
        # (b, s_q, topk) -> (topk, s_q, b) or (total_q, topk) -> (topk, total_q)
        topk_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mIndexTopk = (
            cute.make_tensor(
                mIndexTopk.iterator, cute.select(mIndexTopk.layout, mode=topk_layout_transpose)
            )
            if mIndexTopk is not None
            else None
        )
        # (b, s_q, h) -> (s_q, h, b) or (total_q, h) -> (total_q, h)
        LSE_layout_transpose = [1, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 1]
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if mLSE is not None
            else None
        )
        # (b, s, topk//128, h) => (s, topk//128, h, b) or
        # (total, topk//128, h) == (total, topk//128, h)
        rowmax_layout_transpose = [1, 2, 3, 0] if const_expr(mCuSeqlensQ is None) else [0, 1, 2]
        if const_expr(mRowMax is not None):
            mRowMax = cute.make_tensor(
                mRowMax.iterator, cute.select(mRowMax.layout, mode=rowmax_layout_transpose)
            )

        topk_length_dynamic = mIndexTopk.shape[0] if mIndexTopk is not None else None

        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        self.p_layout = cutlass.utils.LayoutEnum.ROW_MAJOR
        if const_expr(self.store_P):
            assert cutlass.utils.LayoutEnum.from_tensor(mP) == self.p_layout

        mO_og = mO
        mP_og = mP
        if const_expr(self.pack_gqa):
            mQ, mQv, mO, mP, mRowMax = [
                pack_gqa_layout(mX, self.qhead_per_kvhead, self.nheads_kv, head_idx=2)
                if mX is not None
                else None
                for mX in (mQ, mQv, mO, mP, mRowMax)
            ]
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, self.nheads_kv, head_idx=1)

        # ==== Prepare MMAs ====
        # (local_var, dtype_a, major_a, major_b, mma_tiler, operand_source_a)
        # fmt: off
        _mma_specs = [
            ("tiled_mma_QK",    self.dtype_Q,  self.major_mode_Q,   self.major_mode_K,   self.mma_tiler_QK,    self.operand_source_Q),
            ("tiled_mma_QvV", self.dtype_Qv, self.major_mode_Qvi, self.major_mode_Vi,  self.mma_tiler_QvV, self.operand_source_Qvi),
            ("tiled_mma_PVt",  self.dtype_P,  self.major_mode_P,   self.major_mode_Vti, self.mma_tiler_PVt,  self.operand_source_P),
        ]
        tiled_mma_QK, tiled_mma_QvV, tiled_mma_PVt = (
            sm100_utils.make_trivial_tiled_mma(
                dtype_a, major_a, major_b, self.dtype_acc, self.cta_group, mma_tiler[:2], operand_source_a,
            )
            for _, dtype_a, major_a, major_b, mma_tiler, operand_source_a in _mma_specs
        )
        # fmt: on

        # ==== Prepare SMEM layouts and TMAs ====
        # (attr, make_fn, tiled_mma, mma_tiler, dtype, num_stages)
        # fmt: off
        _smem_layout_specs = [
            ("sQ_layout",  sm100_utils.make_smem_layout_a, tiled_mma_QK,  self.mma_tiler_QK,  self.dtype_Q,  self.num_stages_Q),
            ("sK_layout",  sm100_utils.make_smem_layout_b, tiled_mma_QK,  self.mma_tiler_QK,  self.dtype_K,  self.num_stages_K),
            ("sP_layout",  sm100_utils.make_smem_layout_a, tiled_mma_PVt, self.mma_tiler_PVt, self.dtype_P,  self.num_stages_P),
            ("sQv_layout", sm100_utils.make_smem_layout_a, tiled_mma_QvV, self.mma_tiler_QvV, self.dtype_Qv, self.num_stages_Qv),
            ("sV_layout",  sm100_utils.make_smem_layout_b, tiled_mma_QvV, self.mma_tiler_QvV, self.dtype_V,  self.num_stages_V),
            ("sVt_layout", sm100_utils.make_smem_layout_b, tiled_mma_PVt, self.mma_tiler_PVt, self.dtype_V,  self.num_stages_V),
        ]
        for attr, make_fn, tiled_mma, mma_tiler, dtype, num_stages in _smem_layout_specs:
            ab_kwarg = "a_dtype" if make_fn is sm100_utils.make_smem_layout_a else "b_dtype"
            staged = make_fn(
                tiled_mma=tiled_mma,
                mma_tiler_mnk=mma_tiler,
                num_stages=num_stages,
                **{ab_kwarg: dtype},
            )
            setattr(self, f"{attr}_staged", staged)
            setattr(self, attr, cute.select(staged, mode=[0, 1, 2]))
        # fmt: on

        self.sStats_layout = cute.make_layout((self.cta_tile_m, self.cta_group_size))
        self.sScale_layout = cute.make_layout((self.cta_tile_m, self.num_stages_sm_stats))
        self.sBitmask_layout = cute.make_layout((self.tile_n // 32, self.num_stages_bitmask))

        # fmt: off
        for attr, dtype, layout in [
            ("tma_copy_bytes_Q",   self.dtype_Q,  self.sQ_layout),
            ("tma_copy_bytes_K",   self.dtype_K,  self.sK_layout),
            ("tma_copy_bytes_Qvi", self.dtype_Qv, self.sQv_layout),
            ("tma_copy_bytes_Vi",  self.dtype_V,  self.sV_layout),
        ]:
            setattr(self, attr, cute.size_in_bytes(dtype, layout) * self.cta_group_size)
        # fmt: on

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_QK.thr_id.shape,)
        )
        cta_shape = cta_layout_vmnk.shape

        def make_tma(make_fn, mX, smem_layout, mma_tiler, tiled_mma):
            return make_fn(tma_load_op, mX, smem_layout, mma_tiler, tiled_mma, cta_shape)

        A, B = cute.nvgpu.make_tiled_tma_atom_A, cute.nvgpu.make_tiled_tma_atom_B

        # (atom_name, tensor_name, make_fn, m, smem_layout, mma_tiler, tiled_mma, kv_only)
        # fmt: off
        _tma_specs = [
            ("tma_atom_Q",  "tma_tensor_Q",  A, mQ,  self.sQ_layout,  self.mma_tiler_QK,  tiled_mma_QK,  False),
            ("tma_atom_Qv", "tma_tensor_Qv", A, mQv, self.sQv_layout, self.mma_tiler_QvV, tiled_mma_QvV, False),
            ("tma_atom_K",  "tma_tensor_K",  B, mK,  self.sK_layout,  self.mma_tiler_QK,  tiled_mma_QK,  True),
            ("tma_atom_V",  "tma_tensor_V",  B, mV,  self.sV_layout,  self.mma_tiler_QvV, tiled_mma_QvV, True),
            ("tma_atom_Vt", "tma_tensor_Vt", B, mVt, self.sVt_layout, self.mma_tiler_PVt, tiled_mma_PVt, True),
        ]
        _tmas = {}
        for atom_name, tensor_name, make_fn, m, smem_layout, mma_tiler, tiled_mma, kv_only in _tma_specs:
            _tmas[atom_name], _tmas[tensor_name] = (
                make_tma(make_fn, m, smem_layout, mma_tiler, tiled_mma)
                if const_expr((not kv_only or self.use_tma_KV) and m is not None)
                else (None, None)
            )

        (tma_atom_Q,  tma_tensor_Q,
         tma_atom_Qv, tma_tensor_Qv,
         tma_atom_K,  tma_tensor_K,
         tma_atom_V,  tma_tensor_V,
         tma_atom_Vt, tma_tensor_Vt) = _tmas.values()
        # fmt: on

        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        self.ragged_tma_O = (
            self.use_tma_O
            and self.is_varlen_q
            and self.pack_gqa
            and self.cta_tile_m % self.qhead_per_kvhead == 0
        )
        make_tiled_tma_atom_fn = (
            partial(make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.qhead_per_kvhead, head_idx=2)
            if const_expr(self.ragged_tma_O)
            else cpasync.make_tiled_tma_atom
        )

        # ==== Set up P smem -> gmem tma store ====

        # S<3,4,3> o 0 o ((8,8),(64,2),(1,1)):((64,512),(1,4096),(0,0))
        sP_layout_out = sm100_utils.make_smem_layout_epi(
            self.dtype_P, self.p_layout, self.tile_P, self.num_stages_P
        )

        if const_expr(self.store_P):
            # TODO: add asserts
            mP_tma = mP_og if const_expr(self.ragged_tma_O) else mP
            if const_expr(self.ragged_tma_O):
                mP_tma = copy_utils.create_ragged_tensor_for_tma(
                    mP_tma, ragged_dim=0, ptr_shift=True
                )
            tma_atom_P, tma_tensor_P = make_tiled_tma_atom_fn(
                tma_store_op, mP_tma, cute.select(sP_layout_out, mode=[0, 1]), self.tile_P
            )
        else:
            tma_atom_P = None
            tma_tensor_P = None

        # ==== Set up Oi smem -> gmem tma store ====

        self.overlap_sO_sV = True
        if const_expr(self.overlap_sO_sV):
            num_stages_sO = self.num_stages_V
        else:
            num_stages_sO = self.num_hdimv_splits
        sO_layout = sm100_utils.make_smem_layout_epi(
            self.dtype_O, self.o_layout, self.epi_tile, num_stages_sO
        )

        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.ragged_tma_O) else mO
            if const_expr(self.ragged_tma_O):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(
                    mO_tma, ragged_dim=0, ptr_shift=True
                )

            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                tma_store_op, mO_tma, cute.select(sO_layout, mode=[0, 1]), self.epi_tile
            )
        else:
            tma_atom_O = None
            tma_tensor_O = None

        # ==== Set up Oi rmem -> gmem copy ====
        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype_O,
            num_bits_per_copy=universal_copy_bits,
        )
        thread_layout_O_r2g = cute.make_layout((64, 2), stride=(1, 64))
        value_layout_O_r2g = cute.make_layout(
            (1, self.hdimv // self.num_hdimv_splits // self.cta_group_size)
        )
        tiled_copy_O_r2g = cute.make_tiled_copy_tv(
            atom=atom_universal_copy,
            thr_layout=thread_layout_O_r2g,
            val_layout=value_layout_O_r2g,
        )

        # ==== Allocate shared memory ====
        SharedStorage = self._get_shared_storage_cls()

        # ==== Tile scheduler ====

        TileScheduler = self.TileScheduler

        batch_size_for_sched = (
            cute.size(mQv.shape[3]) if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1) if self.use_varlen_scheduler
            else 1
        )

        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQv.shape[0]), self.cluster_tile_m),
            num_head=cute.size(mQv.shape[2]),
            num_batch=batch_size_for_sched,
            num_splits=1,  # todo: split_kv
            seqlen_k=cute.size(mV.shape[0])
            if const_expr(mPageTable is None)
            else cute.size(mV.shape[0]) * cute.size(mPageTable.shape[1]),
            headdim=self.hdim,
            headdim_v=self.hdimv,
            total_q=cute.size(mQv.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQv.shape[0]) * cute.size(mQv.shape[3]),
            tile_shape_mn=(self.cta_tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype_K.width // 8,
            is_persistent=self.is_persistent,
            # lpt=self.is_causal or self.is_local,
            lpt=False,
            is_split_kv=False,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=True,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        fa_printf(1, "grid = {}", grid_dim)

        # ==== Named Barrier ====
        self.cpasync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.Cpasync),
            num_threads=self.num_cpasync_load_threads,
        )
        self.softmax_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.Softmax),
            num_threads=self.num_softmax_threads,
        )
        self.epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.Epilogue),
            num_threads=self.num_epilogue_threads,
        )
        # softmax -> correction
        self.sm_stats_barrier_full = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.SoftmaxStatsFull),
            num_threads=self.num_softmax_threads + self.num_epilogue_threads,
        )
        self.sm_stats_barrier_empty = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.SoftmaxStatsEmpty),
            num_threads=self.num_softmax_threads + self.num_epilogue_threads,
        )

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        # ==== Launch kernel ====
        self.kernel(
            tma_tensor_Q,
            tma_tensor_Qv,
            tma_tensor_K if self.use_tma_KV else mK,
            tma_tensor_V if self.use_tma_KV else mV,
            tma_tensor_Vt if self.use_tma_KV else mVt,
            tma_tensor_O if self.use_tma_O else mO,
            tma_tensor_P,
            mLSE,
            mRowMax,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mIndexTopk,
            mPageTable,
            tma_atom_Q,
            tma_atom_Qv,
            tma_atom_K,
            tma_atom_V,
            tma_atom_Vt,
            tma_atom_O,
            tma_atom_P,
            tiled_copy_O_r2g,
            self.sQ_layout_staged,
            self.sK_layout_staged,
            self.sQv_layout_staged,
            self.sV_layout_staged,
            self.sVt_layout_staged,
            self.sP_layout_staged,
            self.sStats_layout,
            self.sScale_layout,
            self.sBitmask_layout,
            sO_layout,
            sP_layout_out,
            tiled_mma_QK,
            tiled_mma_QvV,
            tiled_mma_PVt,
            softmax_scale,
            softmax_scale_log2,
            topk_length_dynamic,
            tile_sched_params,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=(
                self.num_threads,
                1,
                1,
            ),
            cluster=self.cluster_shape_mnk,
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: Optional[cute.Tensor],
        mQv: cute.Tensor,
        mK: Optional[cute.Tensor],
        mV: cute.Tensor,
        mVt: cute.Tensor,
        mO: cute.Tensor,
        mP: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mRowMax: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mIndexTopk: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Qv: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_Vt: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        tma_atom_P: Optional[cute.CopyAtom],
        tiled_copy_O_r2g: cute.TiledCopy,
        sQ_layout_staged: cute.ComposedLayout,
        sK_layout_staged: cute.ComposedLayout,
        sQv_layout_staged: cute.ComposedLayout,
        sV_layout_staged: cute.ComposedLayout,
        sVt_layout_staged: cute.ComposedLayout,
        sP_layout_staged: cute.ComposedLayout,
        sStats_layout: cute.Layout,
        sScale_layout: cute.Layout,
        sBitmask_layout: cute.Layout,
        sO_layout: cute.ComposedLayout,
        sP_layout_out: cute.ComposedLayout,
        tiled_mma_QK: cute.TiledMma,
        tiled_mma_QvV: cute.TiledMma,
        tiled_mma_PVt: cute.TiledMma,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        topk_length_dynamic: Optional[Int32],
        tile_sched_params: ParamsBase,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_QvV.thr_id.shape,)
        )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        mma_tile_coord_v = cta_rank_in_cluster % cute.size(tiled_mma_QvV.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # ==== Allocate SMEM ====
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # ==== TMEM stuff ====
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.TmemPtr),
            num_threads=self.num_mma_threads + self.num_softmax_threads + self.num_epilogue_threads,
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # ==== Prefetch TMA descriptors ====
        if warp_idx == self.load_warp_id:
            if const_expr(self.has_qk):
                cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_Qv)
            if const_expr(self.use_tma_KV):
                if const_expr(self.has_qk):
                    cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_Vt)
            if const_expr(self.use_tma_O):
                cpasync.prefetch_descriptor(tma_atom_O)

        # ==== Construct pipelines ====
        tma_warp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        mma_warp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        sm_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_softmax_threads)
        epi_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epilogue_threads)
        sm_threads_cluster = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_softmax_threads * self.cta_group_size
        )
        epi_threads_cluster = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_epilogue_threads * self.cta_group_size
        )

        TmaUmma = pipeline.PipelineTmaUmma
        AsyncUmma = pipeline.PipelineAsyncUmma
        UmmaAsync = pipeline.PipelineUmmaAsync
        Async = pipeline.PipelineAsync

        def make_pipeline(cls, mbar_ptr, num_stages, producer, consumer, tx_count=None):
            return cls.create(
                barrier_storage=mbar_ptr.data_ptr(),
                num_stages=num_stages,
                producer_group=producer,
                consumer_group=consumer,
                defer_sync=True,
                **({"cta_layout_vmnk": cta_layout_vmnk} if cls is not Async else {}),
                **({"tx_count": tx_count} if tx_count is not None else {}),
            )

        # Unconditional pipelines
        # fmt: off
        pipeline_Q = None
        if const_expr(self.has_qk):
            pipeline_Q    = make_pipeline(TmaUmma,   storage.mbar_ptr_Q,        self.num_stages_Q,        tma_warp,           mma_warp,           self.tma_copy_bytes_Q)
        pipeline_Qv       = make_pipeline(TmaUmma,   storage.mbar_ptr_Qv,       self.num_stages_Qv,       tma_warp,           mma_warp,           self.tma_copy_bytes_Qvi)
        pipeline_S        = make_pipeline(UmmaAsync, storage.mbar_ptr_S,        self.num_stages_S,        mma_warp,           sm_threads_cluster)
        pipeline_P        = make_pipeline(AsyncUmma, storage.mbar_ptr_P,        self.num_stages_P,        sm_threads_cluster, mma_warp)
        pipeline_O0       = make_pipeline(UmmaAsync, storage.mbar_ptr_O0,       self.num_stages_Oi,       mma_warp,           epi_threads_cluster)
        pipeline_O1       = make_pipeline(UmmaAsync, storage.mbar_ptr_O1,       self.num_stages_Oi,       mma_warp,           epi_threads_cluster)
        pipeline_sm_stats = make_pipeline(Async,     storage.mbar_ptr_sm_stats, self.num_stages_sm_stats, sm_threads,         epi_threads)

        # K/V pipelines: type and producer depend on use_tma_KV
        if const_expr(self.use_tma_KV):
            pipeline_K = None
            if const_expr(self.has_qk):
                pipeline_K     = make_pipeline(TmaUmma, storage.mbar_ptr_K, self.num_stages_K, tma_warp, mma_warp, self.tma_copy_bytes_K)
            pipeline_V         = make_pipeline(TmaUmma, storage.mbar_ptr_V, self.num_stages_V, tma_warp, mma_warp, self.tma_copy_bytes_Vi)
            pipeline_K_cpasync = pipeline_V_cpasync = pipeline_bitmask = None
        else:
            cpasync_load_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_cpasync_load_threads)
            relay_warps_cluster  = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.cta_group_size)
            relay_threads        = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_relay_threads)
            pipeline_K = pipeline_K_cpasync = None
            if const_expr(self.has_qk):
                pipeline_K         = make_pipeline(AsyncUmma, storage.mbar_ptr_K,         self.num_stages_K, relay_warps_cluster,  mma_warp)
                pipeline_K_cpasync = make_pipeline(Async,     storage.mbar_ptr_K_cpasync, self.num_stages_K, cpasync_load_threads, relay_threads)
            pipeline_V             = make_pipeline(AsyncUmma, storage.mbar_ptr_V,         self.num_stages_V, relay_warps_cluster,  mma_warp)
            pipeline_V_cpasync     = make_pipeline(Async,     storage.mbar_ptr_V_cpasync, self.num_stages_V, cpasync_load_threads, relay_threads)
            pipeline_bitmask   = (
                make_pipeline(Async, storage.mbar_ptr_bitmask, self.num_stages_bitmask, cpasync_load_threads, sm_threads)
                if const_expr(self.is_topk_gather and not self.disable_bitmask) else None
            )
        # fmt: on

        sO_empty_mbar_ptr = None
        if const_expr(self.use_tma_O and self.overlap_sO_sV):
            sO_empty_mbar_ptr = storage.sO_empty_mbar_ptr
            if warp_idx == 0:
                cute.arch.mbarrier_init(sO_empty_mbar_ptr, 1)

        pipeline.pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        # ==== Get SMEM tensors ====
        # fmt: off
        sQ, sK, sQv, sV, sVt, sP, sP_out = (
            store.get_tensor(layout.outer, swizzle=layout.inner)
            if const_expr(store._size > 0) else None
            for store, layout in [
                (storage.sQ,  sQ_layout_staged),
                (storage.sK,  sK_layout_staged),
                (storage.sQv, sQv_layout_staged),
                (storage.sV,  sV_layout_staged),
                (storage.sV,  sVt_layout_staged),  # sVt reuses sV storage
                (storage.sP,  sP_layout_staged),
                (storage.sP,  sP_layout_out),
            ]
        )
        # fmt: on
        sRowMax = storage.sRowMax.get_tensor(sStats_layout)
        sRowSum = storage.sRowSum.get_tensor(sStats_layout)
        sScale = storage.sScale.get_tensor(sScale_layout)
        sBitmask = None
        if const_expr(self.is_topk_gather):
            sBitmask = storage.sBitmask.get_tensor(sBitmask_layout)

        if const_expr(self.overlap_sO_sV):
            sO_iterator = sV.iterator
            assert cute.cosize(sO_layout) <= cute.cosize(sV_layout_staged)
        else:
            sO_iterator = sQv.iterator
            assert cute.cosize(sO_layout) <= cute.cosize(sQv_layout_staged)
        sO = cute.make_tensor(
            cute.recast_ptr(sO_iterator, sO_layout.inner, self.dtype_O), sO_layout.outer
        )

        # ==== Get thread MMAs and accumulator fragments ====
        thr_mma_QK = tiled_mma_QK.get_slice(mma_tile_coord_v)
        thr_mma_QvV = tiled_mma_QvV.get_slice(mma_tile_coord_v)
        thr_mma_PVt = tiled_mma_PVt.get_slice(mma_tile_coord_v)

        acc_shape_S = thr_mma_QvV.partition_shape_C(self.mma_tiler_QvV[:2])
        tStS_fake = thr_mma_QvV.make_fragment_C(cute.append(acc_shape_S, self.num_stages_S))

        acc_shape_Oi = thr_mma_PVt.partition_shape_C(self.mma_tiler_PVt[:2])
        tOtO0_fake = thr_mma_PVt.make_fragment_C(acc_shape_Oi)
        tOtO1_fake = thr_mma_PVt.make_fragment_C(acc_shape_Oi)

        block_info = BlockInfo(
            self.cta_tile_m * self.cta_group_size,
            self.tile_n,
            is_causal=self.is_causal,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQv.shape[0] if const_expr(not self.pack_gqa) else mQv.shape[0][1],
            seqlen_k_static=mV.shape[0]
            if const_expr(mPageTable is None)
            else mV.shape[0] * mPageTable.shape[1],
            tile_m=self.cta_tile_m,
            tile_n=self.tile_n,
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.cta_tile_m * self.cta_group_size,
            self.tile_n,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

        if const_expr(self.use_clc_scheduler):
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()

            clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            num_clc_consumer_warps_per_cta = self.num_threads // cute.arch.WARP_SIZE
            num_clc_consumer_warps = num_clc_consumer_warps_per_cta * self.cta_group_size
            clc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cute.arch.WARP_SIZE * num_clc_consumer_warps
            )
            clc = ClcState.create(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=clc_mbar_ptr,
                    num_stages=self.sched_stages,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=16,
                    cta_layout_vmnk=cta_layout_vmnk,
                ),
                consumer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.sched_stages
                ),
                producer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.sched_stages
                ),
            )
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, clc=clc)
        else:
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), (
            f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"
        )

        pipeline.pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        if const_expr(self.use_clc_scheduler):
            if warp_idx == self.clc_scheduler_warp_id:
                if const_expr(self.num_regs_other < self.num_regs_per_thread):
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                if is_leader_cta:
                    self.clc_scheduler_warp(tile_scheduler)
                else:
                    self.empty_warp(tile_scheduler)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i] and warp_idx != self.clc_scheduler_warp_id:
                    if const_expr(self.num_regs_other < self.num_regs_per_thread):
                        cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    if const_expr(self.num_regs_other < self.num_regs_per_thread):
                        cute.arch.setmaxregister_decrease(self.num_regs_other)

        if const_expr(self.use_cpasync_load_KV):
            if warp_idx == self.relay_warp_id:
                if const_expr(self.num_regs_load < self.num_regs_per_thread):
                    cute.arch.setmaxregister_decrease(self.num_regs_load)
                self.relay(
                    pipeline_K,
                    pipeline_V,
                    pipeline_K_cpasync,
                    pipeline_V_cpasync,
                    sO_empty_mbar_ptr,
                    topk_length_dynamic,
                    block_info,
                    SeqlenInfoCls,
                    tile_scheduler=tile_scheduler,
                    mCuSeqlensQ=mCuSeqlensQ,
                )

            if warp_idx in self.cpasync_load_warp_indices:
                if const_expr(self.num_regs_cpasync < self.num_regs_per_thread):
                    cute.arch.setmaxregister_decrease(self.num_regs_cpasync)
                self.load_cpasync(
                    mIndexTopk,
                    mK,
                    mV,
                    mVt,
                    sK,
                    sV,
                    sVt,
                    sBitmask,
                    pipeline_K,
                    pipeline_V,
                    pipeline_K_cpasync,
                    pipeline_V_cpasync,
                    pipeline_bitmask,
                    sO_empty_mbar_ptr,
                    topk_length_dynamic,
                    block_info,
                    SeqlenInfoCls,
                    tile_scheduler=tile_scheduler,
                    mPageTable=mPageTable,
                    mCuSeqlensQ=mCuSeqlensQ,
                )

        if warp_idx == self.load_warp_id:
            if const_expr(self.num_regs_load < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                mQ,
                mK,
                mQv,
                mV,
                mVt,
                sQ,
                sK,
                sQv,
                sV,
                sVt,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_Qv,
                tma_atom_V,
                tma_atom_Vt,
                pipeline_Q,
                pipeline_K,
                pipeline_Qv,
                pipeline_V,
                sO_empty_mbar_ptr,
                thr_mma_QK,
                thr_mma_QvV,
                thr_mma_PVt,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
                mPageTable=mPageTable,
                mCuSeqlensQ=mCuSeqlensQ,
            )

        if warp_idx == self.mma_warp_id:
            if const_expr(self.num_regs_mma < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_mma)
            # ==== Allocate TMEM ====
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)
            tOtO0 = cute.make_tensor(tmem_ptr + self.tmem_offset_O0, tOtO0_fake.layout)
            tOtO1 = cute.make_tensor(tmem_ptr + self.tmem_offset_O1, tOtO1_fake.layout)
            self.mma(
                sQ,
                sK,
                sQv,
                sV,
                sVt,
                sP,
                tStS,
                tOtO0,
                tOtO1,
                tiled_mma_QK,
                tiled_mma_QvV,
                tiled_mma_PVt,
                pipeline_Q,
                pipeline_K,
                pipeline_Qv,
                pipeline_V,
                pipeline_S,
                pipeline_P,
                pipeline_O0,
                pipeline_O1,
                sO_empty_mbar_ptr,
                is_leader_cta,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
                mCuSeqlensQ=mCuSeqlensQ,
            )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if warp_idx in self.softmax_warp_indices:
            if const_expr(self.num_regs_softmax > self.num_regs_per_thread):
                cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)
            self.softmax_loop(
                softmax_scale,
                softmax_scale_log2,
                mLSE,
                mRowMax,
                sRowMax,
                sRowSum,
                sScale,
                sBitmask,
                sP,
                tStS,
                thr_mma_QvV,
                pipeline_S,
                pipeline_P,
                pipeline_sm_stats,
                pipeline_bitmask,
                sO_empty_mbar_ptr,
                AttentionMaskCls,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
                tma_atom_P=tma_atom_P,
                mP=mP,
                sP_out=sP_out,
                mCuSeqlensQ=mCuSeqlensQ,
            )
            tmem_alloc_barrier.arrive()

        if warp_idx in self.epilogue_warp_indices:
            if const_expr(self.num_regs_epilogue < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_epilogue)
            elif const_expr(self.num_regs_epilogue > self.num_regs_per_thread):
                cute.arch.setmaxregister_increase(self.num_regs_epilogue)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            tOtO0 = cute.make_tensor(tmem_ptr + self.tmem_offset_O0, tOtO0_fake.layout)
            tOtO1 = cute.make_tensor(tmem_ptr + self.tmem_offset_O1, tOtO1_fake.layout)
            self.correction_loop(
                softmax_scale_log2,
                mO,
                mLSE,
                tma_atom_O,
                sRowMax,
                sRowSum,
                sScale,
                sO,
                tOtO0,
                tOtO1,
                pipeline_O0,
                pipeline_O1,
                pipeline_sm_stats,
                sO_empty_mbar_ptr,
                tiled_copy_O_r2g,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
                mCuSeqlensQ=mCuSeqlensQ,
            )
            tmem_alloc_barrier.arrive()

    @cute.jit
    def clc_scheduler_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.advance_to_next_work()
            # cluster_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if cute.arch.thread_idx()[0] == self.clc_scheduler_warp_id * cute.arch.WARP_SIZE:
                fa_printf(
                    3,
                    "[CLC] query sm={} cta={} (m_blk={},h={},b={},s={}) valid={}\n",
                    smid(),
                    cute.arch.block_idx()[0],
                    work_tile.tile_idx[0],
                    work_tile.tile_idx[1],
                    work_tile.tile_idx[2],
                    work_tile.tile_idx[3],
                    work_tile.is_valid_tile,
                )
        tile_scheduler.producer_tail()

    @cute.jit
    def empty_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def relay(
        self,
        pipeline_K: Optional[pipeline.PipelineAsyncUmma],
        pipeline_V: pipeline.PipelineAsyncUmma,
        pipeline_K_cpasync: Optional[pipeline.PipelineAsync],
        pipeline_V_cpasync: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
    ):
        # ==== Make pipeline states ====
        # pipeline_{K,V0,V1} producer
        # pipeline_{K,V0,V1}_cpasync consumer
        Producer, Consumer = pipeline.PipelineUserType.Producer, pipeline.PipelineUserType.Consumer
        relay_K_fn = None
        if const_expr(self.has_qk):
            producer_state_K = pipeline.make_pipeline_state(Producer, stages=self.num_stages_K)
            consumer_state_K = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_K)
            relay_K_fn = partial(self.relay_inner, pipeline_K_cpasync, pipeline_K)

        producer_state_V = pipeline.make_pipeline_state(Producer, stages=self.num_stages_V)
        consumer_state_V = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_V)
        relay_V_fn = partial(self.relay_inner, pipeline_V_cpasync, pipeline_V)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cluster_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(self.use_packed_varlen_sched):
                batch_idx = get_batch_from_cu_tensor(cluster_m_block, mCuSeqlensQ)
                if const_expr(not self.is_topk_gather):
                    cluster_m_block -= mCuSeqlensQ[batch_idx]

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                n_block_max = self.topk_length // self.tile_n
                # n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min

            # ==== Prologue ====
            # relay K, V0, V1
            if const_expr(self.has_qk):
                consumer_state_K, producer_state_K = relay_K_fn(consumer_state_K, producer_state_K)
            for _ in cutlass.range_constexpr(self.num_hdimv_splits):
                consumer_state_V, producer_state_V = relay_V_fn(consumer_state_V, producer_state_V)

            # ==== Mainloop ====
            for _ in cutlass.range(num_n_blocks - 1, unroll=2):
                # relay K, V0, V1, Vt0, Vt1
                if const_expr(self.has_qk):
                    consumer_state_K, producer_state_K = relay_K_fn(
                        consumer_state_K, producer_state_K
                    )
                for _ in cutlass.range_constexpr(2 * self.num_hdimv_splits):
                    consumer_state_V, producer_state_V = relay_V_fn(
                        consumer_state_V, producer_state_V
                    )

            # ==== Epilogue ===
            # relay Vt0, Vt1
            for _ in cutlass.range_constexpr(self.num_hdimv_splits):
                consumer_state_V, producer_state_V = relay_V_fn(consumer_state_V, producer_state_V)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        if const_expr(self.has_qk):
            pipeline_K.producer_tail(producer_state_K)
        pipeline_V.producer_tail(producer_state_V)

    @cute.jit
    def relay_inner(
        self,
        pipeline_cpasync: pipeline.PipelineAsync,
        pipeline_mma: pipeline.PipelineAsyncUmma,
        consumer_state: pipeline.PipelineState,
        producer_state: pipeline.PipelineState,
    ):
        pipeline_cpasync.consumer_wait(consumer_state)
        with cute.arch.elect_one():
            pipeline_mma.producer_commit(producer_state)
        consumer_state.advance()
        producer_state.advance()
        return consumer_state, producer_state

    @cute.jit
    def load_cpasync(
        self,
        mIndexTopk: cute.Tensor,
        mK: Optional[cute.Tensor],
        mV: cute.Tensor,
        mVt: cute.Tensor,
        sK: Optional[cute.Tensor],
        sV: cute.Tensor,
        sVt: cute.Tensor,
        sBitmask: Optional[cute.Tensor],
        pipeline_K: Optional[pipeline.PipelineAsyncUmma],
        pipeline_V: pipeline.PipelineAsyncUmma,
        pipeline_K_cpasync: Optional[pipeline.PipelineAsync],
        pipeline_V_cpasync: pipeline.PipelineAsync,
        pipeline_bitmask: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mPageTable: Optional[cute.Tensor] = None,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
    ):
        # ==== cpasync load warpgroup ====
        # Description: loads tiles of K, V, V0, V1 from gmem to smem using cpasync
        # produces: K, V, V0, V1, bitmask
        # consumes: -

        # cpasync load is used for both topk gather and paged KV with page_size != tile_n
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        tidx = cute.arch.thread_idx()[0] % self.num_cpasync_load_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (
            self.num_cpasync_load_threads // 32
        )

        # ==== Make pipeline states ====
        # producer: acquire PipelineAsyncUmma <- mma
        # producer: commit  PipelineAsync     -> relay
        Producer = pipeline.PipelineUserType.Producer
        if const_expr(self.has_qk):
            producer_state_K = pipeline.make_pipeline_state(Producer, stages=self.num_stages_K)
        producer_state_V = pipeline.make_pipeline_state(Producer, stages=self.num_stages_V)
        if const_expr(self.is_topk_gather and not self.disable_bitmask):
            producer_state_bitmask = pipeline.make_pipeline_state(
                Producer, stages=self.num_stages_bitmask
            )
        if const_expr(self.use_tma_O):
            producer_phase_O = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cluster_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(self.use_packed_varlen_sched):
                batch_idx = get_batch_from_cu_tensor(cluster_m_block, mCuSeqlensQ)
                if const_expr(not self.is_topk_gather):
                    cluster_m_block -= mCuSeqlensQ[batch_idx]
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                n_block_max = self.topk_length // self.tile_n
                # n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min

            if const_expr(self.is_topk_gather):
                # ==== Topk gather path ====
                # cluster_m_block == m_idx under MQA 128 assumption
                m_idx = cluster_m_block
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mIndexTopk_cur = mIndexTopk[None, m_idx, batch_idx]
                else:
                    offset_q = seqlen.offset_q if const_expr(not self.use_packed_varlen_sched) else 0
                    mIndexTopk_cur = mIndexTopk[None, m_idx + offset_q]

                if const_expr(self.is_causal):
                    m_local_idx = (
                        m_idx - seqlen.offset_q if const_expr(self.use_packed_varlen_sched) else m_idx
                    )
                    seqlen_k_limit = m_local_idx + 1 + seqlen.seqlen_k - seqlen.seqlen_q
                else:
                    seqlen_k_limit = seqlen.seqlen_k
                cpasync_gather_kv_manager = CpasyncGatherKVManager.create(
                    mIndexTopk_cur,
                    cta_rank_in_cluster,
                    tidx,
                    warp_idx,
                    self.topk_length,
                    seqlen_k_limit,
                    self.tile_n,
                    self.hdim,
                    self.hdimv,
                    self.num_hdimv_splits,
                    self.num_cpasync_load_threads,
                    mV.element_type,
                    self.cta_group_size,
                    self.cpasync_barrier,
                    self.disable_bitmask,
                    sBitmask,
                    pipeline_bitmask,
                )

                # (seqlen_k, hdim) or (seqlen_k, hdimv)
                if const_expr(self.has_qk):
                    mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
                # (hdimv, seqlen_k)
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mVt_cur = mVt[None, None, head_idx_kv, batch_idx]
                else:
                    mVt_cur = cute.domain_offset((0, seqlen.offset_k), mVt[None, None, head_idx_kv])

                hdimv_split_per_cta = self.hdimv // self.num_hdimv_splits // self.cta_group_size
                mVt_cur = cute.tiled_divide(mVt_cur, (hdimv_split_per_cta,))
                mVt_cur = cute.logical_divide(mVt_cur, (1, self.cta_group_size, 1))
                mVt_cur = mVt_cur[(0, None), (cta_rank_in_cluster, None), (0, None)]
                mVt_cur = cute.group_modes(mVt_cur, 0, 2)  # ((hdimv//4, 2), seqlen_k)

                load_K = None
                if const_expr(self.has_qk):
                    load_K = partial(
                        self.cpasync_gather_load_KV,
                        cpasync_gather_kv_manager,
                        pipeline_K,
                        pipeline_K_cpasync,
                        sK,
                        False,
                        "K",
                        mK_cur,
                    )
                load_V = partial(
                    self.cpasync_gather_load_KV,
                    cpasync_gather_kv_manager,
                    pipeline_V,
                    pipeline_V_cpasync,
                    sV,
                    False,
                    "V",
                    mV_cur,
                )
                load_Vt = partial(
                    self.cpasync_gather_load_KV,
                    cpasync_gather_kv_manager,
                    pipeline_V,
                    pipeline_V_cpasync,
                    sVt,
                    True,
                    "V",
                    mVt_cur,
                )

                # process n_blocks in decreasing order
                n_block = n_block_max - 1

                # ==== Prologue ====
                # K, V0, V1
                cpasync_gather_kv_manager.load_index_topk(n_block, transpose=False)
                if const_expr(self.has_qk):
                    producer_state_K = load_K(producer_state_K)
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    producer_state_V = load_V(producer_state_V, d_offset=split * self.hdimv // 2)
                if const_expr(not self.disable_bitmask):
                    producer_state_bitmask = cpasync_gather_kv_manager.compute_bitmask(
                        producer_state_bitmask
                    )

                if const_expr(self.use_tma_O and self.overlap_sO_sV):
                    cute.arch.mbarrier_wait(sO_empty_mbar_ptr, phase=producer_phase_O)
                    producer_phase_O ^= 1

                # ==== Mainloop ====
                for _ in cutlass.range(num_n_blocks - 1, unroll=2):
                    # K, V0, V1
                    cpasync_gather_kv_manager.load_index_topk(n_block - 1, transpose=False)
                    if const_expr(self.has_qk):
                        producer_state_K = load_K(producer_state_K)
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        producer_state_V = load_V(
                            producer_state_V, d_offset=split * self.hdimv // 2
                        )
                    if const_expr(not self.disable_bitmask):
                        producer_state_bitmask = cpasync_gather_kv_manager.compute_bitmask(
                            producer_state_bitmask
                        )
                    # Vt0, Vt1
                    cpasync_gather_kv_manager.load_index_topk(n_block, transpose=True)
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        producer_state_V = load_Vt(
                            producer_state_V, d_offset=split * hdimv_split_per_cta
                        )
                    # advance n_block
                    n_block -= 1

                # ==== Epilogue ====

                # Vt0, Vt1 for n_block = 0
                cpasync_gather_kv_manager.load_index_topk(0, transpose=True)
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    producer_state_V = load_Vt(
                        producer_state_V, d_offset=split * hdimv_split_per_cta
                    )
            else:
                # ==== Paged KV cp.async path (page_size != tile_n) ====
                page_size_divmod = FastDivmodDivisor(cute.size(mV.shape[0]))
                hdimv_split = self.hdimv // self.num_hdimv_splits
                hdimv_split_per_cta = hdimv_split // self.cta_group_size

                # CTA-split Vt: (dv, page_size, h_k, num_pages) -> ((dv/4, 2), page_size, h_k, num_pages)
                mVt_cta = cute.tiled_divide(mVt, (hdimv_split_per_cta,))
                mVt_cta = cute.logical_divide(mVt_cta, (1, self.cta_group_size, 1, 1, 1))
                mVt_cta = mVt_cta[
                    (0, None), (cta_rank_in_cluster, None), (0, None), (0, None), (0, None)
                ]
                mVt_cta = cute.group_modes(mVt_cta, 0, 2)

                # PagedKVManager for K (hdim=64): uses "K" mode only
                if const_expr(self.has_qk):
                    paged_kv_K = PagedKVManager.create(
                        mPageTable,
                        mK,
                        mK,
                        page_size_divmod,
                        batch_idx,
                        head_idx_kv,
                        tidx,
                        seqlen.seqlen_k,
                        0,
                        self.tile_n,
                        self.hdim,
                        self.hdim,
                        self.num_cpasync_load_threads,
                        mK.element_type,
                        arch=100,
                    )
                # PagedKVManager for V/Vt: "K" mode → V (non-transposed), "V" mode → Vt (transposed)
                paged_kv_V = PagedKVManager.create(
                    mPageTable,
                    mV,
                    mVt_cta,
                    page_size_divmod,
                    batch_idx,
                    head_idx_kv,
                    tidx,
                    seqlen.seqlen_k,
                    0,
                    self.tile_n,
                    hdimv_split,
                    hdimv_split_per_cta,
                    self.num_cpasync_load_threads,
                    mV.element_type,
                    arch=100,
                )

                if const_expr(self.has_qk):
                    load_K = partial(
                        self.cpasync_paged_load_KV,
                        paged_kv_K,
                        pipeline_K,
                        pipeline_K_cpasync,
                        sK,
                        False,
                        "K",
                        cta_rank_in_cluster,
                    )
                load_V = partial(
                    self.cpasync_paged_load_KV,
                    paged_kv_V,
                    pipeline_V,
                    pipeline_V_cpasync,
                    sV,
                    False,
                    "K",
                    cta_rank_in_cluster,
                )
                load_Vt = partial(
                    self.cpasync_paged_load_KV,
                    paged_kv_V,
                    pipeline_V,
                    pipeline_V_cpasync,
                    sVt,
                    True,
                    "V",
                    cta_rank_in_cluster,
                )

                n_block_first = n_block_max - 1
                n_block = n_block_first
                safe_n_block_first = n_block_first if num_n_blocks > 0 else 0

                # ==== Prologue ====
                if const_expr(self.has_qk):
                    paged_kv_K.load_page_table(safe_n_block_first)
                    producer_state_K = load_K(n_block_first, producer_state_K)
                paged_kv_V.load_page_table(safe_n_block_first)
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    producer_state_V = load_V(
                        n_block_first, producer_state_V, d_offset=split * self.hdimv // 2
                    )

                if const_expr(self.use_tma_O and self.overlap_sO_sV):
                    cute.arch.mbarrier_wait(sO_empty_mbar_ptr, phase=producer_phase_O)
                    producer_phase_O ^= 1

                # ==== Mainloop ====
                for n_block_idx in cutlass.range(num_n_blocks - 1, unroll=2):
                    n_block = n_block_first - n_block_idx
                    # K, V0, V1 for next block in descending order
                    if const_expr(self.has_qk):
                        paged_kv_K.load_page_table(n_block - 1)
                        producer_state_K = load_K(n_block - 1, producer_state_K)
                    paged_kv_V.load_page_table(n_block - 1)
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        producer_state_V = load_V(
                            n_block - 1, producer_state_V, d_offset=split * self.hdimv // 2
                        )
                    # Vt0, Vt1 for current block
                    paged_kv_V.load_page_table(n_block)
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        producer_state_V = load_Vt(
                            n_block, producer_state_V, d_offset=split * hdimv_split_per_cta
                        )

                # ==== Epilogue ====
                paged_kv_V.load_page_table(n_block_min)
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    producer_state_V = load_Vt(
                        n_block_min, producer_state_V, d_offset=split * hdimv_split_per_cta
                    )

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        # note: don't use producer tail with pipeline_X_cpasync since we never use its producer_acquire.
        if const_expr(self.is_topk_gather and not self.disable_bitmask):
            # pipeline_bitmask invokes producer acquire in gather kv manager,
            # so we should call its producer tail.
            pipeline_bitmask.producer_tail(producer_state_bitmask)

    @cute.jit
    def cpasync_gather_load_KV(
        self,
        cpasync_gather_kv_manager: CpasyncGatherKVManager,
        pipeline_mma: pipeline.PipelineAsyncUmma,
        pipeline_cpasync: pipeline.PipelineAsync,
        sX: cute.Tensor,
        transpose: bool,
        K_or_V: str,
        mX: cute.Tensor,
        producer_state: pipeline.PipelineState,
        d_offset: int = 0,
    ):
        stage = producer_state.index
        pipeline_mma.producer_acquire(producer_state)
        cpasync_gather_kv_manager.load_X(
            mX, sX[None, None, None, stage], transpose, K_or_V, d_offset
        )
        cute.arch.cp_async_commit_group()
        pipeline_cpasync.sync_object_full.arrive_cp_async_mbarrier(stage)
        producer_state.advance()
        return producer_state

    @cute.jit
    def cpasync_paged_load_KV(
        self,
        paged_kv_manager: PagedKVManager,
        pipeline_mma: pipeline.PipelineAsyncUmma,
        pipeline_cpasync: pipeline.PipelineAsync,
        sX: cute.Tensor,
        transpose: bool,
        K_or_V: str,
        cta_rank_in_cluster: Int32,
        n_block: Int32,
        producer_state: pipeline.PipelineState,
        d_offset: int = 0,
    ):
        """Load one tile of K or V from paged gmem to smem using cp.async.

        Uses PagedKVManager for page table lookups and pointer computation,
        with smem reshaping via cute.composition (same approach as CpasyncGatherKVManager).

        For non-transposed tensors (K, V0, V1): K_or_V="K", transpose=False
        For transposed tensors (Vt0, Vt1):      K_or_V="V", transpose=True
        """
        stage = producer_state.index
        pipeline_mma.producer_acquire(producer_state)

        # NOTE: load_page_table() must be called by the caller BEFORE this method.
        # Calling it here (through a @cute.jit boundary) causes MLIR SSA verification
        # errors because the rmem tensor writes inside load_page_table's dynamic
        # cutlass.range loop cross region boundaries. This matches the SM90/SM100
        # pattern where load_page_table is called directly in the loop body.

        # Compute row pointers from cached page table entries
        tPrXPtr = paged_kv_manager.compute_X_ptr(K_or_V, d_offset)

        # Reshape smem to flat 2D using composition (matches CpasyncGatherKVManager.load_X)
        head_dim = (
            paged_kv_manager.head_dim_v_padded
            if const_expr(K_or_V == "V")
            else paged_kv_manager.head_dim_padded
        )
        cta_tile_n = self.tile_n if const_expr(transpose) else self.tile_n // self.cta_group_size
        order = (1, 0) if const_expr(transpose) else (0, 1)

        sX_stage = sX[None, None, None, stage]
        sX_nd_layout = cute.make_ordered_layout((cta_tile_n, head_dim), order=order)
        sX_nd = cute.composition(sX_stage, sX_nd_layout)

        cX = cute.make_identity_tensor((cta_tile_n, head_dim))
        tXsX = paged_kv_manager.gmem_thr_copy_KV.partition_D(sX_nd)
        tXcX = paged_kv_manager.gmem_thr_copy_KV.partition_S(cX)
        tXc0X = paged_kv_manager.gmem_thr_copy_KV.get_slice(0).partition_S(cX)

        base_offset = n_block * self.tile_n
        if const_expr(not transpose):
            base_offset += cta_tile_n * cta_rank_in_cluster
        seqlenk_row_limit = (
            paged_kv_manager.seqlen_k - base_offset - tXcX[0][0] if n_block >= 0 else 0
        )

        if const_expr(not transpose):
            offset = cta_rank_in_cluster * (
                paged_kv_manager.gmem_threads_per_row // self.cta_group_size
            )
        else:
            offset = 0

        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            row_valid = tXc0X[0, m, 0][0] < seqlenk_row_limit
            should_load = cute.make_fragment_like(tXsX[(0, None), m, 0], cute.Boolean)
            should_load.fill(row_valid)

            x_ptr_i64 = fa_utils.shuffle_sync(
                tPrXPtr[m // paged_kv_manager.gmem_threads_per_row],
                (m + offset) % paged_kv_manager.gmem_threads_per_row,
                width=paged_kv_manager.gmem_threads_per_row,
            )
            x_gmem_ptr = cute.make_ptr(
                paged_kv_manager.mK_paged.element_type,
                x_ptr_i64,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            mX_cur = cute.make_tensor(x_gmem_ptr, cute.make_layout((head_dim,)))
            mX_cur_copy = cute.tiled_divide(mX_cur, (paged_kv_manager.async_copy_elems,))

            for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                ki = tXcX[0, 0, k][1] // paged_kv_manager.async_copy_elems
                mX_cur_copy_ki = mX_cur_copy[None, ki]
                tXsX_k = tXsX[None, m, k]
                mX_cur_copy_ki = cute.make_tensor(mX_cur_copy_ki.iterator, tXsX_k.layout)
                cute.copy(
                    paged_kv_manager.gmem_tiled_copy_KV,
                    mX_cur_copy_ki,
                    tXsX_k,
                    pred=should_load,
                )

        cute.arch.cp_async_commit_group()
        pipeline_cpasync.sync_object_full.arrive_cp_async_mbarrier(stage)
        producer_state.advance()
        return producer_state

    @cute.jit
    def load(
        self,
        mQ: Optional[cute.Tensor],
        mK: Optional[cute.Tensor],
        mQv: cute.Tensor,
        mV: cute.Tensor,
        mVt: cute.Tensor,
        sQ: Optional[cute.Tensor],
        sK: Optional[cute.Tensor],
        sQv: cute.Tensor,
        sV: cute.Tensor,
        sVt: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_Qv: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_Vt: cute.CopyAtom,
        pipeline_Q: Optional[pipeline.PipelineAsync],
        pipeline_K: Optional[pipeline.PipelineAsync],
        pipeline_Qv: pipeline.PipelineAsync,
        pipeline_V: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        thr_mma_QK: cute.ThrMma,
        thr_mma_QvV: cute.ThrMma,
        thr_mma_PVt: cute.ThrMma,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mPageTable: Optional[cute.Tensor] = None,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
    ):
        # ==== Load warp ====
        # Description: loads tiles of Q, Qv, K, V, V0, V1 from gmem to smem using TMA
        # produces: Q, Qv, K, V, V0, V1
        # consumes: -

        # ==== Make pipeline states ====
        Producer = pipeline.PipelineUserType.Producer
        if const_expr(self.has_qk):
            producer_state_Q = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Q)
        producer_state_Qv = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Qv)
        if const_expr(self.use_tma_KV):
            if const_expr(self.has_qk):
                producer_state_K = pipeline.make_pipeline_state(Producer, stages=self.num_stages_K)
            producer_state_V = pipeline.make_pipeline_state(Producer, stages=self.num_stages_V)
        if const_expr(self.use_tma_O):
            producer_phase_O = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cluster_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(self.use_packed_varlen_sched):
                batch_idx = get_batch_from_cu_tensor(cluster_m_block, mCuSeqlensQ)
                cluster_m_block -= mCuSeqlensQ[batch_idx]
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                n_block_max = self.topk_length // self.tile_n
                # n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min
            even_n_blocks = num_n_blocks % 2 == 0 and num_n_blocks > 0
            num_n_block_groups = cute.ceil_div(num_n_blocks, self.num_stages_S)

            # ==== Partition GMEM tensors ====
            # (seqlen_q, hdim or hdimv//2)
            if const_expr(self.has_qk):
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                # (mma_tile_m, hdim or hdimv//2)
                gQ = cute.local_tile(
                    mQ_cur,
                    (self.mma_tiler_QK[0], self.mma_tiler_QK[2]),
                    (cluster_m_block, 0),
                )
                tSgQ = thr_mma_QK.partition_A(gQ)
                tQsQ, tQgQ = cpasync.tma_partition(
                    atom=tma_atom_Q,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sQ, 0, 3),
                    gmem_tensor=cute.group_modes(tSgQ, 0, 3),
                )
            mQv_cur = seqlen.offset_batch_Q(mQv, batch_idx, dim=3)[None, None, head_idx]
            gQv = cute.local_tile(
                mQv_cur,
                (self.mma_tiler_QvV[0], self.mma_tiler_QvV[2]),
                (cluster_m_block, None),
            )
            tSgQv = thr_mma_QvV.partition_A(gQv)
            tQvsQv, tQvgQv = cpasync.tma_partition(
                atom=tma_atom_Qv,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sQv, 0, 3),
                gmem_tensor=cute.group_modes(tSgQv, 0, 3),
            )

            if const_expr(self.use_tma_KV):
                if const_expr(mPageTable is None):
                    mPageTable_cur = None
                    # Non-paged: select batch, tile over seqlen_k
                    if const_expr(self.has_qk):
                        # (seqlen_k, hdim)
                        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        # (tile_n, hdim, num_n_blocks)
                        gK = cute.local_tile(
                            mK_cur,
                            (self.mma_tiler_QK[1], self.mma_tiler_QK[2]),
                            (None, 0),
                        )
                    # (seqlen_k, hdimv)
                    mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
                    # (hdimv, seqlen_k)
                    if const_expr(not seqlen.has_cu_seqlens_k):
                        mVt_cur = mVt[None, None, head_idx_kv, batch_idx]
                    else:
                        mVt_cur = cute.domain_offset(
                            (0, seqlen.offset_k), mVt[None, None, head_idx_kv]
                        )
                    # (tile_n, hdimv//4, num_n_blocks, num_d_blocks=4)
                    gV = cute.local_tile(
                        mV_cur,
                        (self.mma_tiler_QvV[1], self.mma_tiler_QvV[2]),
                        (None, None),
                    )
                    # (tile_n, hdimv//4, num_d_blocks=4, num_n_blocks)
                    gV = cute.make_tensor(gV.iterator, cute.select(gV.layout, mode=[0, 1, 3, 2]))
                    # (hdimv//4, tile_n, num_d_blocks=4, num_n_blocks)
                    gVt = cute.local_tile(
                        mVt_cur,
                        (self.mma_tiler_PVt[1], self.mma_tiler_PVt[2]),
                        (None, None),
                    )
                else:
                    mPageTable_cur = mPageTable[batch_idx, None]
                    # Paged KV: keep pages dim, index by page_idx at load time
                    # TMA path assumes page_size == tile_n
                    if const_expr(self.has_qk):
                        # (page_size, hdim, num_pages)
                        mK_cur = mK[None, None, head_idx_kv, None]
                        # (tile_n, hdim, num_pages)
                        gK = cute.local_tile(
                            mK_cur,
                            (self.mma_tiler_QK[1], self.mma_tiler_QK[2]),
                            (0, 0, None),
                        )
                    # (page_size, hdimv, num_pages)
                    mV_cur = mV[None, None, head_idx_kv, None]
                    # (hdimv, page_size, num_pages)
                    mVt_cur = mVt[None, None, head_idx_kv, None]
                    # (tile_n, hdimv//4, num_d_blocks=4, num_pages)
                    gV = cute.local_tile(
                        mV_cur,
                        (self.mma_tiler_QvV[1], self.mma_tiler_QvV[2]),
                        (0, None, None),
                    )
                    # (hdimv//4, tile_n, num_d_blocks=4, num_pages)
                    gVt = cute.local_tile(
                        mVt_cur,
                        (self.mma_tiler_PVt[1], self.mma_tiler_PVt[2]),
                        (None, 0, None),
                    )

                if const_expr(self.has_qk):
                    tSgK = thr_mma_QK.partition_B(gK)
                    tKsK, tKgK = cpasync.tma_partition(
                        atom=tma_atom_K,
                        cta_coord=0,
                        cta_layout=cute.make_layout(1),
                        smem_tensor=cute.group_modes(sK, 0, 3),
                        gmem_tensor=cute.group_modes(tSgK, 0, 3),
                    )

                tSgV = thr_mma_QvV.partition_B(gV)
                tOgVt = thr_mma_PVt.partition_B(gVt)
                tVsV, tVgV = cpasync.tma_partition(
                    atom=tma_atom_V,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sV, 0, 3),
                    gmem_tensor=cute.group_modes(tSgV, 0, 3),
                )
                tVtsVt, tVtgVt = cpasync.tma_partition(
                    atom=tma_atom_Vt,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sVt, 0, 3),
                    gmem_tensor=cute.group_modes(tOgVt, 0, 3),
                )

            if const_expr(self.has_qk):
                load_Q = partial(self.load_inner, tma_atom_Q, tQgQ, tQsQ, pipeline_Q)
            load_Qv = partial(self.load_inner, tma_atom_Qv, tQvgQv, tQvsQv, pipeline_Qv)

            if const_expr(self.use_tma_KV):
                if const_expr(self.has_qk):
                    load_K = partial(self.load_inner, tma_atom_K, tKgK, tKsK, pipeline_K)
                load_V = partial(self.load_inner, tma_atom_V, tVgV, tVsV, pipeline_V)
                load_Vt = partial(self.load_inner, tma_atom_Vt, tVtgVt, tVtsVt, pipeline_V)

            # ==== Load stationary operands ====

            # copy Q, Qvi gmem -> smem
            if const_expr(self.has_qk):
                producer_state_Q = load_Q(producer_state_Q)
            for dv_split in cutlass.range_constexpr(2):
                producer_state_Qv = load_Qv(producer_state_Qv, block=dv_split)

            if const_expr(self.use_tma_KV):
                # ==== Prologue ====
                n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                block = self._get_block_idx(n_block_first, mPageTable_cur)
                # copy K gmem -> smem
                if const_expr(self.has_qk):
                    producer_state_K = load_K(producer_state_K, block=block)
                # copy Vi gmem -> smem
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    producer_state_V = load_V(producer_state_V, block=block, split=split)

                if const_expr(self.use_tma_O and self.overlap_sO_sV):
                    cute.arch.mbarrier_wait(sO_empty_mbar_ptr, phase=producer_phase_O)
                    producer_phase_O ^= 1

                # ==== Main loop ====
                for n_block_group in cutlass.range(num_n_block_groups - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.num_stages_S):
                        n_block = n_block_max - 1 - n_block_group * self.num_stages_S - stage
                        block_next = self._get_block_idx(n_block - 1, mPageTable_cur)
                        block = self._get_block_idx(n_block, mPageTable_cur)
                        if const_expr(self.has_qk):
                            # copy K gmem -> smem
                            producer_state_K = load_K(producer_state_K, block=block_next)
                        # copy Vi gmem -> smem
                        for split in cutlass.range_constexpr(self.num_hdimv_splits):
                            producer_state_V = load_V(
                                producer_state_V, block=block_next, split=split
                            )
                        # copy Vti gmem -> smem
                        for split in cutlass.range_constexpr(self.num_hdimv_splits):
                            producer_state_V = load_Vt(producer_state_V, block=block, split=split)

                # ==== Epilogue ====
                num_final_n_blocks = self.num_stages_S if even_n_blocks else self.num_stages_S - 1
                for stage in cutlass.range(num_final_n_blocks, unroll_full=True):
                    n_block = num_final_n_blocks - 1 - stage
                    block = self._get_block_idx(n_block, mPageTable_cur)
                    if n_block > 0:
                        block_next = self._get_block_idx(n_block - 1, mPageTable_cur)
                        if const_expr(self.has_qk):
                            # copy K gmem -> smem
                            producer_state_K = load_K(producer_state_K, block=block_next)
                        # copy Vi gmem -> smem
                        for split in cutlass.range_constexpr(self.num_hdimv_splits):
                            producer_state_V = load_V(
                                producer_state_V, block=block_next, split=split
                            )
                    # copy Vti gmem -> smem
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        producer_state_V = load_Vt(producer_state_V, block=block, split=split)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        if const_expr(self.has_qk):
            pipeline_Q.producer_tail(producer_state_Q)
        pipeline_Qv.producer_tail(producer_state_Qv)
        if const_expr(self.use_tma_KV):
            if const_expr(self.has_qk):
                pipeline_K.producer_tail(producer_state_K)
            pipeline_V.producer_tail(producer_state_V)

    @cute.jit
    def _get_block_idx(
        self,
        n_block,
        mPageTable_cur: Optional[cute.Tensor],
    ):
        if const_expr(mPageTable_cur is not None):
            return mPageTable_cur[n_block]
        else:
            return n_block

    @cute.jit
    def load_inner(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        load_pipeline: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        block: Optional[Int32] = None,
        split: Optional[Int32] = None,
    ):
        if const_expr(split is not None):
            tXgX = tXgX[(None, split, None)]
        if const_expr(block is not None):
            tXgX = tXgX[(None, block)]
        if const_expr(cute.rank(tXsX) != 1):
            assert cute.rank(tXsX) == 2, f"wrong rank for tXsX, got {cute.rank(tXsX)}"
            stage = producer_state.index
            tXsX = tXsX[(None, stage)]
        load_pipeline.producer_acquire(producer_state)
        tma_bar_ptr = load_pipeline.producer_get_barrier(producer_state)
        cute.copy(tma_atom, tXgX, tXsX, tma_bar_ptr=tma_bar_ptr)
        producer_state.advance()
        return producer_state

    @cute.jit
    def mma(
        self,
        sQ: Optional[cute.Tensor],
        sK: Optional[cute.Tensor],
        sQv: cute.Tensor,
        sV: cute.Tensor,
        sVt: cute.Tensor,
        sP: cute.Tensor,
        tStS: cute.Tensor,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        tiled_mma_QK: cute.TiledMma,
        tiled_mma_QvV: cute.TiledMma,
        tiled_mma_PVt: cute.TiledMma,
        pipeline_Q: Optional[pipeline.PipelineAsync],
        pipeline_K: Optional[pipeline.PipelineAsync],
        pipeline_Qv: pipeline.PipelineAsync,
        pipeline_V: pipeline.PipelineAsync,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_O0: pipeline.PipelineAsync,
        pipeline_O1: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        is_leader_cta: Boolean,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
    ):
        # ==== mma warp ====
        # Description: Computes Q @ K^T, Qv @ V^T, and P @ V
        # Produces: S, O
        # Consumes: Q, K, Qv, V, P

        pipelines_O = [pipeline_O0, pipeline_O1]
        tOtOs = [tOtO0, tOtO1]

        use_ptx_gemm_QK = True
        use_ptx_gemm_QvV = True
        use_ptx_gemm_PVt = True

        # Operands for S = Q @ K^T
        if const_expr(self.has_qk):
            tSrQ = tiled_mma_QK.make_fragment_A(sQ)
            tSrK = tiled_mma_QK.make_fragment_B(sK)

        # Operands for S += Qv @ V^T
        tSrQv = tiled_mma_QvV.make_fragment_A(sQv)
        tSrV = tiled_mma_QvV.make_fragment_B(sV)

        # Operands for Oi = P @ Vi
        tOrP = tiled_mma_PVt.make_fragment_A(sP)
        tOrVt = tiled_mma_PVt.make_fragment_B(sVt)

        # GEMM functions
        if const_expr(self.has_qk):
            if const_expr(use_ptx_gemm_QK):
                gemm_QK = [
                    partial(
                        fa_sm100_utils.gemm_ptx_partial,
                        tiled_mma_QK.op,
                        self.tmem_offset_S[stage],
                        zero_init=True,
                        cta_group=self.cta_group_size,
                    )
                    for stage in range(self.num_stages_S)
                ]
            else:
                gemm_QK = [
                    partial(
                        fa_sm100_utils.gemm,
                        tiled_mma_QK,
                        tStS[None, None, None, stage],
                        zero_init=True,
                    )
                    for stage in range(self.num_stages_S)
                ]
        if const_expr(use_ptx_gemm_QvV):
            gemm_QvV = [
                partial(
                    fa_sm100_utils.gemm_ptx_partial,
                    tiled_mma_QvV.op,
                    self.tmem_offset_S[stage],
                    cta_group=self.cta_group_size,
                )
                for stage in range(self.num_stages_S)
            ]
        else:
            gemm_QvV = [
                partial(
                    fa_sm100_utils.gemm,
                    tiled_mma_QvV,
                    tStS[None, None, None, stage],
                )
                for stage in range(self.num_stages_S)
            ]

        if const_expr(use_ptx_gemm_PVt):
            gemm_PVt = [
                partial(
                    fa_sm100_utils.gemm_ptx_partial,
                    tiled_mma_PVt.op,
                    self.tmem_offsets_O[split],
                    cta_group=self.cta_group_size,
                )
                for split in range(self.num_hdimv_splits)
            ]
        else:
            gemm_PVt = [
                partial(
                    fa_sm100_utils.gemm,
                    tiled_mma_PVt,
                    tOtOs[split],
                )
                for split in range(self.num_hdimv_splits)
            ]

        Consumer, Producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer
        if const_expr(self.has_qk):
            consumer_state_Q = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_Q)
            consumer_state_K = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_K)
        consumer_state_Qv = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_Qv)
        consumer_state_V = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_V)
        producer_state_S = pipeline.make_pipeline_state(Producer, stages=self.num_stages_S)
        consumer_state_P = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_P)
        producer_state_O0 = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Oi)
        producer_state_O1 = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Oi)

        mma_fn = self.mma_inner
        if const_expr(self.has_qk):
            mma_QK = partial(
                mma_fn, gemm_QK, pipeline_K, tSrQ, sQ, tSrK, sK, use_ptx=use_ptx_gemm_QK
            )
        mma_QvV = partial(
            mma_fn, gemm_QvV, pipeline_V, tSrQv, sQv, tSrV, sV, use_ptx=use_ptx_gemm_QvV
        )
        mma_PVt = partial(
            mma_fn, gemm_PVt, pipeline_V, tOrP, sP, tOrVt, sVt, use_ptx=use_ptx_gemm_PVt
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        O_should_accumulate = False
        while work_tile.is_valid_tile:
            cluster_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(self.use_packed_varlen_sched):
                batch_idx = get_batch_from_cu_tensor(cluster_m_block, mCuSeqlensQ)
                cluster_m_block -= mCuSeqlensQ[batch_idx]

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                # n_block_max = self.topk_length // self.tile_n
                n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min
            even_n_blocks = num_n_blocks % 2 == 0 and num_n_blocks > 0
            num_n_block_groups = cute.ceil_div(num_n_blocks, self.num_stages_S)

            if is_leader_cta:
                if const_expr(self.has_qk):
                    pipeline_Q.consumer_wait(consumer_state_Q)

                consumer_wait_state_Qv = consumer_state_Qv.clone()
                for _ in cutlass.range_constexpr(self.num_hdimv_splits):
                    pipeline_Qv.consumer_wait(consumer_wait_state_Qv)
                    consumer_wait_state_Qv.advance()

                producer_states_O = [producer_state_O0, producer_state_O1]

                # ==== Prologue ====
                pipeline_S.producer_acquire(producer_state_S)
                if const_expr(self.has_qk):
                    # S = Q @ K^T
                    consumer_state_K = mma_QK(consumer_state_K, acc_stage=0)
                # S += Qvi @ Vi^T
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    consumer_state_V = mma_QvV(
                        consumer_state_V,
                        acc_stage=0,
                        a_stage=split,
                        zero_init=split == 0 and not self.has_qk,
                    )
                pipeline_S.producer_commit(producer_state_S)
                producer_state_S.advance()

                # ==== Mainloop ====
                for _ in cutlass.range(num_n_block_groups - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.num_stages_S):
                        next_stage = const_expr((stage + 1) % self.num_stages_S)
                        pipeline_S.producer_acquire(producer_state_S)
                        if const_expr(self.has_qk):
                            # S = Q @ K^T
                            consumer_state_K = mma_QK(consumer_state_K, acc_stage=next_stage)
                        # S += Qvi @ Vi^T
                        for split in cutlass.range_constexpr(self.num_hdimv_splits):
                            consumer_state_V = mma_QvV(
                                consumer_state_V,
                                acc_stage=next_stage,
                                a_stage=split,
                                zero_init=split == 0 and not self.has_qk,
                            )
                        pipeline_S.producer_commit(producer_state_S)
                        producer_state_S.advance()
                        # Oi += P @ Vi
                        pipeline_P.consumer_wait(consumer_state_P)
                        for split in cutlass.range_constexpr(self.num_hdimv_splits):
                            producer_state_Oi = producer_states_O[split]
                            pipelines_O[split].producer_acquire(producer_state_Oi)
                            consumer_state_V = mma_PVt(
                                consumer_state_V,
                                acc_stage=split,
                                a_stage=consumer_state_P.index,
                                zero_init=not O_should_accumulate,
                            )
                            pipelines_O[split].producer_commit(producer_state_Oi)
                            producer_state_Oi.advance()
                            producer_states_O[split] = producer_state_Oi
                        pipeline_P.consumer_release(consumer_state_P)
                        consumer_state_P.advance()
                        O_should_accumulate = True

                # ==== Epilogue ====
                num_final_n_blocks = self.num_stages_S if even_n_blocks else self.num_stages_S - 1
                for stage in cutlass.range_constexpr(self.num_stages_S):
                    n_block = num_final_n_blocks - 1 - stage
                    if const_expr(stage == 0):
                        if n_block > 0:
                            pipeline_S.producer_acquire(producer_state_S)
                            if const_expr(self.has_qk):
                                # S = Q @ K^T
                                consumer_state_K = mma_QK(consumer_state_K, acc_stage=stage + 1)
                            # S += Qvi @ Vi^T
                            for split in cutlass.range_constexpr(self.num_hdimv_splits):
                                consumer_state_V = mma_QvV(
                                    consumer_state_V,
                                    acc_stage=stage + 1,
                                    a_stage=split,
                                    zero_init=split == 0 and not self.has_qk,
                                )
                            pipeline_S.producer_commit(producer_state_S)
                            producer_state_S.advance()
                    if n_block >= 0:
                        # Oi += P @ Vi
                        pipeline_P.consumer_wait(consumer_state_P)
                        for split in cutlass.range_constexpr(self.num_hdimv_splits):
                            producer_state_Oi = producer_states_O[split]
                            pipelines_O[split].producer_acquire(producer_state_Oi)
                            consumer_state_V = mma_PVt(
                                consumer_state_V,
                                acc_stage=split,
                                a_stage=consumer_state_P.index,
                                zero_init=not O_should_accumulate,
                            )
                            pipelines_O[split].producer_commit(producer_state_Oi)
                            producer_state_Oi.advance()
                            producer_states_O[split] = producer_state_Oi
                        pipeline_P.consumer_release(consumer_state_P)
                        consumer_state_P.advance()
                    O_should_accumulate = True

                producer_state_O0, producer_state_O1 = producer_states_O

                if const_expr(self.has_qk):
                    pipeline_Q.consumer_release(consumer_state_Q)
                    consumer_state_Q.advance()

                # if we overlap sOi with sQvi for tma store, need to acquire signal
                if const_expr(self.use_tma_O and not self.overlap_sO_sV):
                    pipeline_O0.producer_tail(producer_state_O0.clone())
                    pipeline_O1.producer_tail(producer_state_O1.clone())

                for _ in cutlass.range_constexpr(self.num_hdimv_splits):
                    pipeline_Qv.consumer_release(consumer_state_Qv)
                    consumer_state_Qv.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
            O_should_accumulate = False

        pipeline_S.producer_tail(producer_state_S)
        pipeline_O0.producer_tail(producer_state_O0)
        pipeline_O1.producer_tail(producer_state_O1)

    @cute.jit
    def mma_inner(
        self,
        gemm,
        load_pipeline,
        tCrA,
        sA,
        tCrB,
        sB,
        consumer_state: pipeline.PipelineState,
        acc_stage: Optional[Int32] = None,
        a_stage: Int32 = 0,
        zero_init: Optional[bool] = None,
        use_ptx: bool = True,
    ):
        if const_expr(acc_stage is not None):
            gemm = gemm[acc_stage]

        tCrA_cur = tCrA[None, None, None, a_stage]
        sA_cur = sA[None, None, None, a_stage]
        b_stage = consumer_state.index
        tCrB_cur = tCrB[None, None, None, b_stage]
        sB_cur = sB[None, None, None, b_stage]

        load_pipeline.consumer_wait(consumer_state)

        kwargs = dict(tCrA=tCrA_cur, tCrB=tCrB_cur)
        if const_expr(use_ptx):
            kwargs |= dict(sA=sA_cur, sB=sB_cur)
        if const_expr(zero_init is not None):
            kwargs["zero_init"] = zero_init
        gemm(**kwargs)

        load_pipeline.consumer_release(consumer_state)
        consumer_state.advance()
        return consumer_state

    @cute.jit
    def softmax_loop(
        self,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        mLSE: Optional[cute.Tensor],
        mRowMax: Optional[cute.Tensor],
        sRowMax: cute.Tensor,
        sRowSum: cute.Tensor,
        sScale: cute.Tensor,
        sBitmask: Optional[cute.Tensor],
        sP: cute.Tensor,
        tStS: cute.Tensor,
        thr_mma_S: cute.ThrMma,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        pipeline_bitmask: Optional[pipeline.PipelineAsync],
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        AttentionMaskCls: Callable,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        tma_atom_P: Optional[cute.CopyAtom] = None,
        mP: Optional[cute.Tensor] = None,
        sP_out: Optional[cute.Tensor] = None,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
    ):
        # ==== softmax warpgroup ====
        # Description: computes softmax on S and writes the result to P
        # Produces: P, softmax stats
        # Consumes: S, bitmask (for topk sparsity)

        tidx = cute.arch.thread_idx()[0] % self.num_softmax_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (
            self.num_softmax_threads // 32
        )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

        tSAcc = tStS[(None, None), 0, 0, 0]
        tSAcc_staged = [tStS[(None, None), 0, 0, stage] for stage in range(self.num_stages_S)]

        cS = cute.make_identity_tensor(self.mma_tiler_QK[:2])  # (128, 128)
        tScS = thr_mma_S.partition_C(cS)[(None, None), 0, 0]  # (64, 128)

        # S tmem -> rmem copy objects
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.dtype_acc,
        )
        tmem_load_tiled = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc)
        tmem_load_thr = tmem_load_tiled.get_slice(tidx)
        # S tmem -> rmem copy operands
        tStS_t2r = tmem_load_thr.partition_S(tSAcc)  # (((32, 32), 1), 1, 2)
        tStS_t2r_staged = [
            tmem_load_thr.partition_S(tSAcc_staged[stage]) for stage in range(self.num_stages_S)
        ]
        tScS_t2r = tmem_load_thr.partition_D(tScS)
        tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, self.dtype_acc)

        # P rmem -> smem copy objects
        universal_copy_bits = 128
        smem_store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype_P,
            num_bits_per_copy=universal_copy_bits,
        )
        smem_store_tiled = cute.make_tiled_copy_D(smem_store_atom, tmem_load_tiled)
        smem_store_thr = smem_store_tiled.get_slice(tidx)
        # P rmem -> smem copy operands
        sP_mnp_layout = cute.make_ordered_layout(
            self.tile_P + (self.num_stages_P,), order=(0, 1, 2)
        )
        sP_mnp = cute.composition(sP, sP_mnp_layout)
        sP_smem_view = smem_store_thr.partition_D(sP_mnp)

        Consumer, Producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer
        consumer_state_S = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_S)
        producer_state_P = pipeline.make_pipeline_state(Producer, stages=self.num_stages_P)
        producer_state_sm_stats = pipeline.make_pipeline_state(Producer, stages=self.num_stages_sm_stats)
        consumer_state_bitmask = None
        if const_expr(self.is_topk_gather and not self.disable_bitmask):
            consumer_state_bitmask = pipeline.make_pipeline_state(
                Consumer, stages=self.num_stages_bitmask
            )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cluster_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(self.use_packed_varlen_sched):
                batch_idx = get_batch_from_cu_tensor(cluster_m_block, mCuSeqlensQ)
                cluster_m_block -= mCuSeqlensQ[batch_idx]
            cta_m_block = cluster_m_block * self.cta_group_size + cta_rank_in_cluster
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                n_block_max = self.topk_length // self.tile_n
                # n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min
            even_n_blocks = num_n_blocks % 2 == 0 and num_n_blocks > 0
            num_n_block_groups = cute.ceil_div(num_n_blocks, self.num_stages_S)

            gRowMax = None
            if const_expr(mRowMax is not None):
                # (seqlen_q, {seqlen_k_rounded, topk} / tile_n)
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mRowMax_cur = mRowMax[None, None, head_idx, batch_idx]
                else:
                    q_offset = (
                        seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    )
                    mRowMax_cur = cute.domain_offset((q_offset, 0), mRowMax[None, None, head_idx])
                # (cta_tile_m, {seqlen_k_rounded, topk} / tile_n)
                gRowMax = cute.local_tile(mRowMax_cur, (self.cta_tile_m,), (cta_m_block, None))

            store_P = None
            if const_expr(self.store_P):
                # (seqlen_q, seqlen_k)
                mP_cur = seqlen.offset_batch_Q(mP, batch_idx, dim=3, ragged=self.ragged_tma_O)[
                    None, None, head_idx
                ]
                # (cta_tile_m, tile_n, num_n_blocks)
                gP = cute.local_tile(mP_cur, self.tile_P, (cta_m_block, None))
                store_P, tPsP, tPgP = copy_utils.tma_get_copy_fn(
                    tma_atom_P,
                    0,
                    cute.make_layout(1),
                    sP_out,
                    gP,
                )

            mask = AttentionMaskCls(seqlen)
            mask_fn = partial(
                mask.apply_mask_sm100,
                m_block=cluster_m_block,
                thr_mma=thr_mma_S,
                thr_tmem_load=tmem_load_thr,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                r2p=False,  # TODO: fix r2p for 2cta
            )
            disable_mask = self.disable_bitmask and self.is_topk_gather

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0 if const_expr(self.dtype_Q.width == 16) else 0.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            softmax_step_fn = partial(
                self.softmax_step,
                softmax,
                sRowMax,
                sScale,
                sBitmask,
                tStS_t2r_staged,
                tSrS_t2r,
                sP_smem_view,
                tmem_load_thr,
                smem_store_thr,
                pipeline_S,
                pipeline_P,
                pipeline_sm_stats,
                pipeline_bitmask,
                tidx,
                warp_idx,
                store_P=store_P,
                gRowMax=gRowMax,
            )

            ### first iteration ###
            n_block = n_block_max - 1
            (
                consumer_state_S,
                producer_state_P,
                producer_state_sm_stats,
                consumer_state_bitmask,
            ) = softmax_step_fn(
                consumer_state_S,
                producer_state_P,
                producer_state_sm_stats,
                consumer_state_bitmask,
                0,
                n_block,
                mask_fn=partial(mask_fn, mask_seqlen=True)
                if not const_expr(disable_mask)
                else None,
                is_first=True,
            )
            n_block -= 1

            ### Separate iterations with causal masking
            # note: For square mma tile, can mask at most 1 n_block_group
            if const_expr((self.is_causal or self.is_local) and not self.is_topk_gather):
                n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                    seqlen, cluster_m_block, n_block_min
                )
                num_masked_n_blocks = n_block_max - 1 - n_block_min_causal_local_mask
                num_masked_n_block_groups = min(
                    num_n_block_groups - 1, cute.ceil_div(num_masked_n_blocks, self.num_stages_S)
                )
                num_n_block_groups -= num_masked_n_block_groups
                for _ in cutlass.range(num_masked_n_block_groups, unroll=1):
                    for stage in cutlass.range_constexpr(self.num_stages_S):
                        (
                            consumer_state_S,
                            producer_state_P,
                            producer_state_sm_stats,
                            consumer_state_bitmask,
                        ) = softmax_step_fn(
                            consumer_state_S,
                            producer_state_P,
                            producer_state_sm_stats,
                            consumer_state_bitmask,
                            1 - stage,
                            n_block,
                            mask_fn=partial(mask_fn, mask_seqlen=False),
                        )
                        n_block -= 1

            ### Mainloop ###
            for n_block_group in cutlass.range(num_n_block_groups - 1, unroll=1):
                for stage in cutlass.range_constexpr(self.num_stages_S):
                    (
                        consumer_state_S,
                        producer_state_P,
                        producer_state_sm_stats,
                        consumer_state_bitmask,
                    ) = softmax_step_fn(
                        consumer_state_S,
                        producer_state_P,
                        producer_state_sm_stats,
                        consumer_state_bitmask,
                        1 - stage,
                        n_block,
                        mask_fn=partial(mask_fn, mask_seqlen=False)
                        if const_expr(self.is_topk_gather and not self.disable_bitmask)
                        else None,
                    )
                    n_block -= 1

            ### last iteration if even ###
            # always mask to simplify logic
            if even_n_blocks:
                (
                    consumer_state_S,
                    producer_state_P,
                    producer_state_sm_stats,
                    consumer_state_bitmask,
                ) = softmax_step_fn(
                    consumer_state_S,
                    producer_state_P,
                    producer_state_sm_stats,
                    consumer_state_bitmask,
                    1,
                    n_block,
                    mask_fn=partial(mask_fn, mask_seqlen=False)
                    if not const_expr(disable_mask)
                    else None,
                )
                n_block -= 1

            # write row max and sum to smem
            sRowSum[tidx % self.cta_tile_m, warp_idx // self.cta_group_size] = softmax.row_sum[0]
            if const_expr(mLSE is not None):
                if tidx < self.cta_tile_m:
                    sRowMax[tidx, 0] = softmax.row_max[0]
            self.sm_stats_barrier_full.arrive()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
            self.sm_stats_barrier_empty.arrive_and_wait()

        pipeline_P.producer_tail(producer_state_P)
        pipeline_sm_stats.producer_tail(producer_state_sm_stats)

    @cute.jit
    def softmax_step(
        self,
        softmax: SoftmaxSm100,
        sRowMax: cute.Tensor,
        sScale: cute.Tensor,
        sBitmask: Optional[cute.Tensor],
        tStS_t2r_staged: cute.Tensor,
        tSrS_t2r: cute.Tensor,
        sP_smem_view: cute.Tensor,
        tmem_load_thr: cute.CopyAtom,
        smem_store_thr: cute.CopyAtom,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        pipeline_bitmask: Optional[pipeline.PipelineAsync],
        tidx: Int32,
        warp_idx: Int32,
        consumer_state_S: pipeline.PipelineState,
        producer_state_P: pipeline.PipelineState,
        producer_state_sm_stats: pipeline.PipelineState,
        consumer_state_bitmask: Optional[pipeline.PipelineState],
        stage: cutlass.Constexpr[Int32],
        n_block: Int32,
        mask_fn: Optional[Callable] = None,
        is_first: Boolean = False,
        store_P: Optional[Callable] = None,
        gRowMax: Optional[cute.Tensor] = None,
    ):
        leader_warp = warp_idx == 0
        tSrP = cute.make_rmem_tensor(tSrS_t2r.shape, self.dtype_P)
        rP_smem_view = smem_store_thr.retile(tSrP)

        pipeline_S.consumer_wait(consumer_state_S)
        cute.copy(tmem_load_thr, tStS_t2r_staged[stage], tSrS_t2r)
        cute.arch.fence_view_async_tmem_load()
        pipeline_S.consumer_release(consumer_state_S)

        rBitmask = None
        if const_expr(self.is_topk_gather and not self.disable_bitmask):
            assert pipeline_bitmask is not None
            assert consumer_state_bitmask is not None
            pipeline_bitmask.consumer_wait(consumer_state_bitmask)
            rBitmask = cute.make_rmem_tensor((self.tile_n // 64,), dtype=Uint32)
            bitmask_col_offset = self.tile_n // 64 if warp_idx >= 2 else 0
            for i in cutlass.range_constexpr(cute.size(rBitmask)):
                rBitmask[i] = sBitmask[bitmask_col_offset + i, consumer_state_bitmask.index]

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block, rBitmask=rBitmask)

        # compute threadwise row_max
        row_max = softmax.compute_row_max_local(tSrS_t2r.load(), is_first)
        # 2-thread reduce row_max through smem
        assert self.cta_tile_m * self.cta_group_size == 128
        sRowMax[tidx % self.cta_tile_m, warp_idx // self.cta_group_size] = row_max
        self.softmax_barrier.arrive_and_wait()
        # must release after barrier sync
        if const_expr(self.is_topk_gather and not self.disable_bitmask):
            pipeline_bitmask.consumer_release(consumer_state_bitmask)
        row_max0 = sRowMax[tidx % self.cta_tile_m, 0]
        row_max1 = sRowMax[tidx % self.cta_tile_m, 1]
        row_max = max(row_max0, row_max1)

        row_max, acc_scale = softmax.update_row_max_from_local(row_max, is_first)

        if const_expr(gRowMax is not None):
            if tidx < self.cta_tile_m:
                gRowMax[tidx, n_block] = row_max

        # note: acc_scales agree for paired threads
        pipeline_sm_stats.producer_acquire(producer_state_sm_stats)
        if warp_idx < self.cta_group_size:
            sScale[tidx % self.cta_tile_m, producer_state_sm_stats.index] = acc_scale
        pipeline_sm_stats.producer_commit(producer_state_sm_stats)

        # x -> scale_log2*x-rowmax
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)

        # x -> exp2(x)
        softmax.apply_exp2_convert(tSrS_t2r, tSrP)

        if const_expr(self.store_P):
            if leader_warp:
                cute.arch.cp_async_bulk_wait_group(self.num_stages_P - 1, read=True)
            self.softmax_barrier.arrive_and_wait()

        pipeline_P.producer_acquire(producer_state_P)
        cute.copy(
            smem_store_thr, rP_smem_view, sP_smem_view[None, None, None, producer_state_P.index]
        )
        cute.arch.fence_view_async_shared()
        pipeline_P.producer_commit(producer_state_P)
        # unconditionally necessary for sRowMax read to complete before next iter's store
        self.softmax_barrier.arrive_and_wait()

        if const_expr(self.store_P):
            if leader_warp:
                store_P(src_idx=producer_state_P.index, dst_idx=n_block)
                cute.arch.cp_async_bulk_commit_group()

        consumer_state_S.advance()
        producer_state_P.advance()
        producer_state_sm_stats.advance()
        if const_expr(self.is_topk_gather and not self.disable_bitmask):
            consumer_state_bitmask.advance()

        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)

        return consumer_state_S, producer_state_P, producer_state_sm_stats, consumer_state_bitmask

    @cute.jit
    def correction_loop(
        self,
        softmax_scale_log2: Float32,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_O: Optional[cute.CopyAtom],
        sRowMax: cute.Tensor,
        sRowSum: cute.Tensor,
        sScale: cute.Tensor,
        sO: cute.Tensor,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        pipeline_O0: pipeline.PipelineAsync,
        pipeline_O1: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        tiled_copy_O_r2g: cute.TiledCopy,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
    ):
        ### ==== correction/epilogue warpgroup ====
        # Correction: copy scale smem -> rmem, copy O tmem -> rmem, rescale O, store O rmem -> tmem
        # Epilogue:   copy O tmem -> rmem, do final scaling of O, store O rmem -> gmem,
        #             optionally store LSE
        # Produces: -
        # Consumes: O, softmax stats

        tidx = cute.arch.thread_idx()[0] % self.num_epilogue_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (
            self.num_epilogue_threads // 32
        )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        leader_warp = warp_idx == 0

        tOtO0 = tOtO0[(None, None), 0, 0]  # (64, (128, 2))
        tOtO1 = tOtO1[(None, None), 0, 0]  # (64, (128, 2))
        tOtOs = [tOtO0, tOtO1]

        # tuneable parameter
        corr_tile_size = math.gcd(32, self.tmem_cols_Oi)

        tmem_load_atom_O = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.dtype_acc,
        )
        tmem_store_atom_O = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.dtype_acc,
        )
        thr_tmem_load_O = tcgen05.make_tmem_copy(tmem_load_atom_O, tOtO0).get_slice(tidx)
        thr_tmem_store_O = tcgen05.make_tmem_copy(tmem_store_atom_O, tOtO0).get_slice(tidx)

        # ((32,1),1,4)
        tOtOs_t2r = [
            thr_tmem_load_O.partition_S(tOtOs[split]) for split in range(self.num_hdimv_splits)
        ]
        tOtOs_r2t = [
            thr_tmem_store_O.partition_D(tOtOs[split]) for split in range(self.num_hdimv_splits)
        ]

        cOi = cute.make_identity_tensor((self.cta_tile_m, self.hdimv // self.num_hdimv_splits))
        thr_tiled_copy_O_r2g = tiled_copy_O_r2g.get_slice(tidx)
        tOicOi = thr_tiled_copy_O_r2g.partition_S(cOi)

        tOicOi_t2r = thr_tmem_load_O.partition_D(tOicOi[(None, None), 0, 0])

        pipelines_O = [pipeline_O0, pipeline_O1]

        Consumer = pipeline.PipelineUserType.Consumer
        consumer_state_O0 = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_Oi)
        consumer_state_O1 = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_Oi)
        consumer_state_sm_stats = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_sm_stats)

        do_correction_rescale = partial(
            self.correction_rescale,
            thr_tmem_load_O,
            thr_tmem_store_O,
            tOicOi_t2r,
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cluster_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(self.use_packed_varlen_sched):
                batch_idx = get_batch_from_cu_tensor(cluster_m_block, mCuSeqlensQ)
                cluster_m_block -= mCuSeqlensQ[batch_idx]
            cta_m_block = cluster_m_block * self.cta_group_size + cta_rank_in_cluster

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                n_block_max = self.topk_length // self.tile_n
                # n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min

            consumer_states_O = [consumer_state_O0, consumer_state_O1]

            # acquire first signal and release immediately
            pipeline_sm_stats.consumer_wait(consumer_state_sm_stats)
            pipeline_sm_stats.consumer_release(consumer_state_sm_stats)
            consumer_state_sm_stats.advance()

            for _ in cutlass.range(num_n_blocks - 1, unroll=1):
                pipeline_sm_stats.consumer_wait(consumer_state_sm_stats)
                scale = sScale[tidx % self.cta_tile_m, consumer_state_sm_stats.index]
                should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                pipeline_sm_stats.consumer_release(consumer_state_sm_stats)
                consumer_state_sm_stats.advance()

                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    consumer_state_Oi = consumer_states_O[split]
                    pipelines_O[split].consumer_wait(consumer_state_Oi)
                    if should_rescale:
                        do_correction_rescale(
                            tOtOs_t2r[split],
                            tOtOs_r2t[split],
                            scale,
                        )
                    pipelines_O[split].consumer_release(consumer_state_Oi)
                    consumer_state_Oi.advance()
                    consumer_states_O[split] = consumer_state_Oi

            # (seqlen_q, hdimv)
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=self.ragged_tma_O)[
                None, None, head_idx
            ]
            # (cta_tile_m, hdimv//2, 2)
            gO = cute.local_tile(
                mO_cur,
                (self.cta_tile_m, self.hdimv // self.num_hdimv_splits),
                (cta_m_block, None),
            )
            tOgO = thr_tiled_copy_O_r2g.partition_D(gO)
            # ((32, 1), 1, 4)
            tOrOs_t2r = [
                cute.make_rmem_tensor(tOicOi_t2r.shape, self.dtype_acc)
                for split in range(self.num_hdimv_splits)
            ]
            tOrOs_r2g_f32 = [
                thr_tiled_copy_O_r2g.retile(tOrOs_t2r[split])
                for split in range(self.num_hdimv_splits)
            ]
            tOrOs_r2g = [
                cute.make_rmem_tensor_like(tOrOs_r2g_f32[split], self.dtype_O)
                for split in range(self.num_hdimv_splits)
            ]
            if const_expr(self.use_tma_O):
                tOsO = thr_tiled_copy_O_r2g.partition_D(sO)
                store_O, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_O,
                    0,
                    cute.make_layout(1),
                    sO,
                    gO,
                )

            self.sm_stats_barrier_full.arrive_and_wait()

            row_sum0 = sRowSum[tidx % self.cta_tile_m, 0]
            row_sum1 = sRowSum[tidx % self.cta_tile_m, 1]
            row_sum = row_sum0 + row_sum1
            acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
            scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)

            row_max = 0.0
            if const_expr(mLSE is not None):
                if tidx < self.cta_tile_m:
                    row_max = sRowMax[tidx, 0]

            self.sm_stats_barrier_empty.arrive()

            seqlen_q = (
                seqlen.seqlen_q
                if const_expr(not self.pack_gqa)
                else seqlen.seqlen_q * self.qhead_per_kvhead
            )

            # compute and store lse to gmem
            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    lse_offset = (
                        seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    )
                    mLSE_cur = cute.domain_offset((lse_offset,), mLSE[None, head_idx])
                gLSE = cute.local_tile(mLSE_cur, (self.cta_tile_m,), (cta_m_block,))
                if tidx < self.cta_tile_m:
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2 + cute.math.log2(row_sum, fastmath=True))
                        * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    if tidx < seqlen_q - cta_m_block * self.cta_tile_m:
                        gLSE[tidx] = lse

            row_idx = cta_m_block * self.cta_tile_m + tOicOi[0][0]

            for split in cutlass.range_constexpr(self.num_hdimv_splits):
                consumer_state_Oi = consumer_states_O[split]
                pipelines_O[split].consumer_wait(consumer_state_Oi)
                # copy Oi tmem -> rmem
                cute.copy(
                    thr_tmem_load_O,
                    tOtOs_t2r[split],
                    tOrOs_t2r[split],
                )

                # scale and downcast Oi
                tOrOs_r2g[split].store((tOrOs_r2g_f32[split].load() * scale).to(self.dtype_O))

                if const_expr(not self.use_tma_O):
                    # copy Oi rmem -> gmem
                    if row_idx < seqlen_q:
                        cute.copy(
                            thr_tiled_copy_O_r2g,
                            tOrOs_r2g[split],
                            tOgO[None, None, None, split],
                        )
                else:
                    # copy Oi rmem -> smem
                    if const_expr(self.overlap_sO_sV):
                        # last slot for Vti is always 2, 3
                        sO_idx = 2 + split
                    else:
                        sO_idx = split
                    cute.copy(
                        thr_tiled_copy_O_r2g,
                        tOrOs_r2g[split],
                        tOsO[None, None, None, sO_idx],
                    )
                    cute.arch.fence_view_async_shared()
                    self.epi_barrier.arrive_and_wait()
                    # tma store Oi smem -> gmem
                    if leader_warp:
                        store_O(src_idx=sO_idx, dst_idx=split)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(1 - split, read=True)
                        if const_expr(split == 1 and self.overlap_sO_sV):
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(sO_empty_mbar_ptr)

            consumer_state_O0, consumer_state_O1 = consumer_states_O

            cute.arch.fence_view_async_tmem_load()
            pipeline_O0.consumer_release(consumer_state_O0)
            pipeline_O1.consumer_release(consumer_state_O1)
            consumer_state_O0.advance()
            consumer_state_O1.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def correction_rescale(
        self,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tOcO_t2r: cute.Tensor,
        tOtO_t2r: cute.Tensor,
        tOtO_r2t: cute.Tensor,
        scale: Float32,
    ):
        tOrO_t2r_frg = cute.make_rmem_tensor_like(tOcO_t2r[None, None, 0], self.dtype_acc)

        for i in cutlass.range_constexpr(cute.size(tOtO_t2r, mode=[2])):
            tOtO_t2r_cur = tOtO_t2r[None, None, i]
            tOtO_r2t_cur = tOtO_r2t[None, None, i]

            cute.copy(thr_tmem_load, tOtO_t2r_cur, tOrO_t2r_frg)
            for j in cutlass.range(0, cute.size(tOrO_t2r_frg), 2, unroll_full=True):
                tOrO_t2r_frg[j], tOrO_t2r_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_t2r_frg[j], tOrO_t2r_frg[j + 1]), (scale, scale)
                )
            cute.copy(thr_tmem_store, tOrO_t2r_frg, tOtO_r2t_cur)
        cute.arch.fence_view_async_tmem_store()
