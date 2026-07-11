# Copyright (c) 2026, Colfax International.

import math
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64, Int32, Boolean, const_expr
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.utils import ClcDynamicPersistentTileScheduler

from ..quack import copy_utils, layout_utils

from .pack_gqa import pack_gqa_layout
from .seqlen_info import SeqlenInfoQK
from .block_info import BlockInfo
from . import blackwell_helpers as fa_sm100_utils
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
from .utils import smid, elem_pointer, get_batch_from_cu_tensor
from .copy_utils import tiled_copy_2d, atomic_add_fp32x4

from .topk_gather_kv import CpasyncGatherKVManager


from .named_barrier import NamedBarrierBwdSm100_MLA2CTA


class FlashAttentionSparseMLABackwardSm100:
    def __init__(
        self,
        is_causal: bool = False,
        topk_length: int = 2048,
        qhead_per_kvhead: int = 1,
        nheads_kv: int = 1,
        hdim: int = 64,
        hdimv: int = 512,
        has_seqused_q: bool = False,
        disable_bitmask: bool = False,
        use_clc_scheduler: bool = True,
    ):
        use_cpasync_load_KV = True
        self.is_causal = is_causal
        self.is_local = False
        self.pack_gqa = True
        self.qhead_per_kvhead = qhead_per_kvhead
        self.nheads_kv = nheads_kv
        self.has_seqused_q = has_seqused_q
        self.use_tma_O = True
        self.use_cpasync_load_KV = True
        self.use_tma_KV = False
        self.topk_length = topk_length
        self.is_topk_gather = True
        assert qhead_per_kvhead == 128 or qhead_per_kvhead == 64

        # user-provided option if topk indices guaranteed in bounds
        self.disable_bitmask = disable_bitmask

        # ==== tile scheduler ====
        self.static_persistent = False
        self.use_clc_scheduler = use_clc_scheduler
        self.sched_stages = 1
        self.scheduling_mode = (
            SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC
        )

        if const_expr(has_seqused_q):
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
        self.num_empty_threads = 0
        self.num_relay_threads = 32
        self.num_cpasync_load_threads = 128
        self.num_threads = 512
        self.num_warps = self.num_threads // 32
        self.softmax_warp_indices = (0, 1, 2, 3)
        self.epilogue_warp_indices = (4, 5, 6, 7)
        self.load_warp_id = 8
        self.mma_warp_id = 9
        self.clc_scheduler_warp_id = 10
        self.relay_warp_id = 11
        self.cpasync_load_warp_indices = (12, 13, 14, 15)
        self.empty_warp_ids = ()

        # ==== register usage ====
        assert self.num_warps == 16

        self.num_regs_load = 128
        self.num_regs_mma = 128
        self.num_regs_softmax = 128
        self.num_regs_epilogue = 128
        self.num_regs_cpasync = 128
        self.num_regs_other = 128

        # self.num_regs_load = 128 - 32
        # self.num_regs_mma = 128 - 32
        # self.num_regs_softmax = 128 + 32
        # self.num_regs_epilogue = 128 + 32
        # self.num_regs_cpasync = 128 - 32
        # self.num_regs_other = 48

        self.num_regs_per_thread = 128
        self.num_regs_total = 512

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
        self.hdim = hdim  # ignored
        self.hdimv = hdimv
        self.tile_m = qhead_per_kvhead
        self.tile_n = 64
        self.cta_tiler_mn = (self.tile_m // self.cta_group_size, self.tile_n)
        self.cluster_tile_n = self.cta_group_size * self.tile_n
        self.num_hdimv_splits = 2  # split hdimv in half for our Qv @ V^T and P @ V mmas.

        self.tile_P = (self.tile_m, self.tile_n)
        self.tile_Pt = (self.tile_n, self.tile_m)
        self.tile_dS = (self.tile_m, self.tile_n)
        self.tile_dSt = (self.tile_n, self.tile_m)
        self.tile_dV = (self.tile_n, 32)

        # ==== MMA info ====
        # dP.T = V    @ dO.T , N x M x dv
        # dV  += P.T  @ dO   , N x dv x M
        # dV  += dS.T @ Qv   , N x dv x M
        self.mma_tiler_VdO = (
            self.cluster_tile_n,
            self.tile_m,
            self.hdimv // self.num_hdimv_splits,
        )
        self.mma_tiler_PtdOt = (
            self.cluster_tile_n,
            self.hdimv // self.num_hdimv_splits,
            self.tile_m,
        )
        self.mma_tiler_dStQvt = (
            self.cluster_tile_n,
            self.hdimv // self.num_hdimv_splits,
            self.tile_m,
        )
        # note: store P.T, dS.T as tile_n major (i.e., as P and dS)
        self.major_mode_V = tcgen05.OperandMajorMode.K
        self.major_mode_dO = tcgen05.OperandMajorMode.K
        self.major_mode_Pt = tcgen05.OperandMajorMode.MN
        self.major_mode_dOt = tcgen05.OperandMajorMode.MN
        self.major_mode_dSt = tcgen05.OperandMajorMode.MN
        self.major_mode_Qvt = tcgen05.OperandMajorMode.MN
        self.operand_source_V = tcgen05.OperandSource.SMEM
        self.operand_source_Pt = tcgen05.OperandSource.SMEM
        self.operand_source_dSt = tcgen05.OperandSource.SMEM

        # ==== pipeline info ====
        # stationary: dOi
        # mainloop:
        # *) P, scaleP => Pt
        # *) dSt
        # *) Vi, i = {0, 1}
        # *) dOti => Qvi => dVi, i = {0, 1}

        # redundant names for ease-of-use
        self.num_stages_V = 2
        self.num_stages_dO = 2
        self.num_stages_P = 1
        self.num_stages_Pt = 1
        self.num_stages_dS = 1
        self.num_stages_dSt = 1
        self.num_stages_dOt = 2
        self.num_stages_Qv = 2
        self.num_stages_Qvt = 2

        self.num_stages_dP = 1
        self.num_stages_dPt = 1
        self.num_stages_dV = 2  # == hdimv splits, for Umma <-> Async
        self.num_epi_stages_dV = 8  # == 2 splits x 4 slots/split

        self.num_stages_scaleP = 1
        self.num_stages_dPsum = 1

        # ==== dtype info ====
        self.dtype_acc = Float32

        # ==== TMEM info ====
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_cols_dP = self.tile_m // self.cta_group_size
        self.tmem_cols_dVi = (self.hdimv // self.num_hdimv_splits) // self.cta_group_size
        self.tmem_offset_dV0 = 0
        self.tmem_offset_dV1 = self.tmem_offset_dV0 + self.tmem_cols_dVi
        self.tmem_offsets_dV = [self.tmem_offset_dV0, self.tmem_offset_dV1]
        self.tmem_offset_dP = self.tmem_offset_dV1 + self.tmem_cols_dVi
        self.total_tmem = self.tmem_offset_dP + self.tmem_cols_dP
        assert self.total_tmem <= self.tmem_alloc_cols, (
            f"Total TMEM columns allocated {self.total_tmem} exceeds capacity {self.tmem_alloc_cols}"
        )

    def _get_shared_storage_cls(self):
        self.buffer_align_bytes = 1024

        def smem_struct_align(dtype, staged_layout):
            return cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(staged_layout)],
                self.buffer_align_bytes,
            ]

        def mbar_struct(num_stages):
            return cute.struct.MemRange[Int64, 2 * num_stages]

        # sV, sdO, sP = sPt, sdSt = sdS, sdOt = sQvt = sdV
        (
            sV_struct,
            sdO_struct,
            sP_struct,
            sdS_struct,
            sQvt_struct,
            sScaleP_struct,
            sdPsum_struct,
        ) = (
            smem_struct_align(dtype, layout)
            for dtype, layout in [
                (self.dtype, self.sV_layout_staged),
                (self.dtype, self.sdO_layout_staged),
                (self.dtype, self.sPt_layout_staged),
                (self.dtype, self.sdSt_layout_staged),
                (self.dtype, self.sQvt_layout_staged),
                (self.dtype_scale, self.sScaleP_layout_staged),
                (self.dtype_scale, self.sdPsum_layout_staged),
            ]
        )

        (
            mbar_ptr_V_struct,  # load V
            mbar_ptr_dO_struct,  # load dO
            mbar_ptr_dOt_Qvt_struct,  # load dOt => Qvt
            mbar_ptr_dSt_struct,  # store dS
            mbar_ptr_P_struct,  # load P
            mbar_ptr_Pt_struct,  # store Pt
            mbar_ptr_dPt_struct,  # dP mma
            mbar_ptr_dV_struct,  # dV mma
            mbar_ptr_scaleP_struct,  # load scaleP
            mbar_ptr_dPsum_struct,  # load dPsum
        ) = (
            mbar_struct(n)
            for n in [
                self.num_stages_V,
                self.num_stages_dO,
                self.num_stages_Qvt,
                self.num_stages_dSt,
                self.num_stages_P,
                self.num_stages_Pt,
                self.num_stages_dPt,
                self.num_stages_dV,
                self.num_stages_scaleP,
                self.num_stages_dPsum,
            ]
        )
        tmem_dealloc_mbar_struct = Int64
        tmem_holding_buf_struct = Int32

        self.sched_stages = 1
        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0

        @cute.struct
        class SharedStorage:
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_V_cpasync: mbar_ptr_V_struct
            mbar_ptr_dO: mbar_ptr_dO_struct
            mbar_ptr_dOt_Qvt: mbar_ptr_dOt_Qvt_struct
            mbar_ptr_P: mbar_ptr_P_struct
            mbar_ptr_Pt: mbar_ptr_Pt_struct
            mbar_ptr_dSt: mbar_ptr_dSt_struct
            mbar_ptr_dPt: mbar_ptr_dPt_struct
            mbar_ptr_dV: mbar_ptr_dV_struct
            mbar_ptr_dV_epi: mbar_ptr_dV_struct
            mbar_ptr_scaleP: mbar_ptr_scaleP_struct
            mbar_ptr_dPsum: mbar_ptr_dPsum_struct
            tmem_dealloc_mbar: tmem_dealloc_mbar_struct
            tmem_holding_buf: tmem_holding_buf_struct
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.MemRange[Int32, clc_response_size]

            sScaleP: sScaleP_struct
            sdPsum: sdPsum_struct
            sV: sV_struct
            sdO: sdO_struct
            sP: sP_struct
            sdS: sdS_struct
            sQv: sQvt_struct

        # print("smem bytes = ", SharedStorage.size_in_bytes())

        return SharedStorage

    # fmt: off
    @cute.jit
    def __call__(
        self,
        mdO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mV: cute.Tensor,   # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k
        mQv: cute.Tensor,  # == mdO
        mP: cute.Tensor,   # (b, s_q, h, topk) or (total_q, h, topk)
        mdV: cute.Tensor,  # == mV
        mdS: cute.Tensor,  # == mP
        mIndexTopk: cute.Tensor,  # (b, s_q, topk) or (total_q, topk) if there is cu_seqlens_q
        softmax_scale: Float32,
        mScaleP: Optional[cute.Tensor] = None,      # (b, s_q, topk//128, h) or (total_q, topk//128, h)
        mdPsum: Optional[cute.Tensor] = None,       # (b, s_q, h) or (total_q, h) if there is cu_seqlens_q
        mCuSeqlensQ: Optional[cute.Tensor] = None,  # (b + 1)
        mCuSeqlensK: Optional[cute.Tensor] = None,  # (b + 1)
        mSeqUsedQ: Optional[cute.Tensor] = None,    # (b)
        mSeqUsedK: Optional[cute.Tensor] = None,    # (b)
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # fmt: on
        # ==== dtype info ====
        self.dtype = mdO.element_type
        self.dtype_dV = mdV.element_type
        self.dtype_scale = Float32
        self.dtype_index = mIndexTopk.element_type
        assert self.dtype.width == 16
        assert self.dtype_dV.width == 32
        assert self.dtype_index == Int32
        if const_expr(mScaleP is not None):
            assert mScaleP.element_type == self.dtype_scale
        if const_expr(mdPsum is not None):
            assert mdPsum.element_type == self.dtype_scale

        # ==== Prepare Tensors ====
        new_stride = lambda mX: (
            *(cute.assume(s, divby=128 // mX.element_type.width) for s in mX.stride[:-1]),
            mX.stride[-1],
        )
        mQv, mV, mdV, mdO, mP, mdS, mScaleP, mdPsum = [
            cute.make_tensor(mX.iterator, cute.make_layout(mX.shape, stride=new_stride(mX)))
            if mX is not None
            else None
            for mX in (mQv, mV, mdV, mdO, mP, mdS, mScaleP, mdPsum)
        ]
        # (b, s, h, d)  -> (s, d, h, b)  or
        # (total, h, d) -> (total, d, h)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mQv, mdO, mP, mdS = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=QO_layout_transpose))
            if mX is not None
            else None
            for mX in (mQv, mdO, mP, mdS)
        ]
        mV, mdV = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=KV_layout_transpose))
            if mX is not None
            else None
            for mX in (mV, mdV)
        ]

        # (b, s, topk//128, h) -> (s, topk//128, h, b) or
        # (total, topk//128, h) -> (total, topk//128, h)
        ScaleP_layout_transpose = [1, 2, 3, 0] if const_expr(mCuSeqlensQ is None) else [0, 1, 2]
        mScaleP = cute.make_tensor(
            mScaleP.iterator, cute.select(mScaleP.layout, mode=ScaleP_layout_transpose)
        )

        # (b, s, h) -> (s, h, b) or
        # (total, h) -> (total, h)
        dPsum_layout_transpose = [1, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 1]
        mdPsum = cute.make_tensor(
            mdPsum.iterator, cute.select(mdPsum.layout, mode=dPsum_layout_transpose)
        )

        # (b, s_q, topk) -> (topk, s_q, b) or (total_q, topk) -> (topk, total_q)
        topk_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mIndexTopk = cute.make_tensor(
            mIndexTopk.iterator, cute.select(mIndexTopk.layout, mode=topk_layout_transpose)
        )
        topk_length_dynamic = mIndexTopk.shape[0]

        if const_expr(self.pack_gqa):
            mQv, mdO, mP, mdS, mScaleP = [
                pack_gqa_layout(mX, self.qhead_per_kvhead, self.nheads_kv, head_idx=2)
                if mX is not None
                else None
                for mX in (mQv, mdO, mP, mdS, mScaleP)
            ]
            if const_expr(mdPsum is not None):
                mdPsum = pack_gqa_layout(mdPsum, self.qhead_per_kvhead, self.nheads_kv, head_idx=1)

        # ((h/h_k, s_q), dv, h_k, b) -> (dv, (h/h_k, s_q), h_k, b)
        # or ((h/h_k, total_q), dv, h_k) -> (dv, (h/h_k, total_q), h_k)
        mma_operand_layout_transpose = (
            [1, 0, 2, 3] if const_expr(mCuSeqlensQ is None) else [1, 0, 2]
        )
        mQvt, mdOt = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=mma_operand_layout_transpose))
            for mX in (mQv, mdO)
        ]

        # fmt: off
        # ==== Prepare MMAs ====
        # (local_var, dtype_a, major_a, major_b, mma_tiler, operand_source_a)
        _mma_specs = [
            ("tiled_mma_VdO",    self.dtype, self.major_mode_V,   self.major_mode_dO,  self.mma_tiler_VdO,    self.operand_source_V),
            ("tiled_mma_PtdOt",  self.dtype, self.major_mode_Pt,  self.major_mode_dOt, self.mma_tiler_PtdOt,  self.operand_source_Pt),
            ("tiled_mma_dStQvt", self.dtype, self.major_mode_dSt, self.major_mode_Qvt, self.mma_tiler_dStQvt, self.operand_source_dSt),
        ]
        tiled_mma_VdO, tiled_mma_PtdOt, tiled_mma_dStQvt = (
            sm100_utils.make_trivial_tiled_mma(
                dtype_a, major_a, major_b, self.dtype_acc, self.cta_group, mma_tiler[:2], operand_source_a,
            )
            for _, dtype_a, major_a, major_b, mma_tiler, operand_source_a in _mma_specs
        )

        # ==== Prepare SMEM layouts and TMAs ====
        # (attr, make_fn, tiled_mma, mma_tiler, dtype, num_stages)
        _smem_layout_specs = [
            ("sV_layout",   sm100_utils.make_smem_layout_a, tiled_mma_VdO,    self.mma_tiler_VdO,    self.dtype, self.num_stages_V),
            ("sdO_layout",  sm100_utils.make_smem_layout_b, tiled_mma_VdO,    self.mma_tiler_VdO,    self.dtype, self.num_stages_dO),
            ("sPt_layout",  sm100_utils.make_smem_layout_a, tiled_mma_PtdOt,  self.mma_tiler_PtdOt,  self.dtype, self.num_stages_Pt),
            ("sdOt_layout", sm100_utils.make_smem_layout_b, tiled_mma_PtdOt,  self.mma_tiler_PtdOt,  self.dtype, self.num_stages_dOt),
            ("sdSt_layout", sm100_utils.make_smem_layout_a, tiled_mma_dStQvt, self.mma_tiler_dStQvt, self.dtype, self.num_stages_dSt),
            ("sQvt_layout", sm100_utils.make_smem_layout_b, tiled_mma_dStQvt, self.mma_tiler_dStQvt, self.dtype, self.num_stages_Qvt),
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

        # Prepare additional SMEM load layouts
        self.P_layout_major = cutlass.utils.LayoutEnum.from_tensor(mP)
        self.sP_layout_staged = sm100_utils.make_smem_layout_epi(
            self.dtype, self.P_layout_major, self.tile_P, self.num_stages_P
        )
        self.sP_layout = cute.select(self.sP_layout_staged, mode=[0, 1])
        self.sScaleP_layout_staged = cute.make_layout((self.tile_m, self.num_stages_scaleP))
        self.sScaleP_layout = cute.select(self.sScaleP_layout_staged, mode=[0])
        self.sdPsum_layout_staged = cute.make_layout((self.tile_m, self.num_stages_dPsum))
        self.sdPsum_layout = cute.select(self.sdPsum_layout_staged, mode=[0])

        # ==== TMA load ====
        for attr, dtype, layout in [
            ("tma_copy_bytes_V",   self.dtype, self.sV_layout),
            ("tma_copy_bytes_dO",  self.dtype, self.sdO_layout),
            ("tma_copy_bytes_dOt", self.dtype, self.sdOt_layout),
            ("tma_copy_bytes_Qvt", self.dtype, self.sQvt_layout),
        ]:
            setattr(self, attr, cute.size_in_bytes(dtype, layout) * self.cta_group_size)

        assert self.tma_copy_bytes_dOt == self.tma_copy_bytes_Qvt
        self.tma_copy_bytes_P = cute.size_in_bytes(self.dtype, self.sP_layout)
        self.tma_copy_bytes_scaleP = cute.size_in_bytes(self.dtype_scale, self.sScaleP_layout)
        self.tma_copy_bytes_dPsum = cute.size_in_bytes(self.dtype_scale, self.sdPsum_layout)

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_VdO.thr_id.shape,)
        )
        cta_shape = cta_layout_vmnk.shape

        def make_tma(make_fn, mX, smem_layout, mma_tiler, tiled_mma):
            return make_fn(tma_load_op, mX, smem_layout, mma_tiler, tiled_mma, cta_shape)

        A, B = cute.nvgpu.make_tiled_tma_atom_A, cute.nvgpu.make_tiled_tma_atom_B

        # (atom_name, tensor_name, make_fn, m, smem_layout, mma_tiler, tiled_mma)
        _tma_specs = [
            ("tma_atom_dO",  "tma_tensor_dO",  B, mdO,  self.sdO_layout,  self.mma_tiler_VdO,    tiled_mma_VdO),
            ("tma_atom_dOt", "tma_tensor_dOt", B, mdOt, self.sdOt_layout, self.mma_tiler_PtdOt,  tiled_mma_PtdOt),
            ("tma_atom_Qvt", "tma_tensor_Qvt", B, mQvt, self.sQvt_layout, self.mma_tiler_dStQvt, tiled_mma_dStQvt),
        ]
        _tmas = {}
        for atom_name, tensor_name, make_fn, m, smem_layout, mma_tiler, tiled_mma in _tma_specs:
            _tmas[atom_name], _tmas[tensor_name] = (
                make_tma(make_fn, m, smem_layout, mma_tiler, tiled_mma)
            )

        (tma_atom_dO,  tma_tensor_dO,
         tma_atom_dOt, tma_tensor_dOt,
         tma_atom_Qvt, tma_tensor_Qvt) = _tmas.values()

        # Make TMA load for P separately
        tma_atom_P, tma_tensor_P = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mP,
            self.sP_layout,
            self.tile_P,
        )

        # ==== TMA store ====
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        self.dS_layout_major = cutlass.utils.LayoutEnum.from_tensor(mdS)
        self.dV_layout_major = cutlass.utils.LayoutEnum.from_tensor(mdV)
        # (tile_m, tile_n, dS_stages) = (nheads, 64, dS_stage)
        sdS_layout_staged = sm100_utils.make_smem_layout_epi(
            self.dtype, self.dS_layout_major, self.tile_dS, self.num_stages_dSt
        )
        # (tile_n, 32, dV_epi_stages) = (64, 32, 4 x 2)
        sdV_layout_staged = sm100_utils.make_smem_layout_epi(
            self.dtype_dV, self.dV_layout_major, self.tile_dV, self.num_epi_stages_dV
        )
        tma_atom_dS, tma_tensor_dS = cpasync.make_tiled_tma_atom(
            tma_store_op, mdS, cute.select(sdS_layout_staged, mode=[0, 1]), self.tile_dS
        )
        # fmt: on

        # ==== Allocate shared memory ====
        SharedStorage = self._get_shared_storage_cls()

        # ==== Tile scheduler ====
        TileScheduler = self.TileScheduler

        fa_printf(1, "mdO = {}", mdO.layout)
        batch_size_for_sched = cute.size(mdO.shape[3]) if const_expr(mCuSeqlensQ is None) else 1
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mdO.shape[0]), self.tile_m),
            num_head=cute.size(mdO.shape[2]),
            num_batch=batch_size_for_sched,
            num_splits=1,
            seqlen_k=cute.size(mV.shape[0]),
            headdim=self.hdim,
            headdim_v=self.hdimv,
            total_q=cute.size(mdO.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mdO.shape[0]) * cute.size(mdO.shape[3]),
            tile_shape_mn=self.cta_tiler_mn,
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype.width // 8,
            is_persistent=self.static_persistent,
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
            barrier_id=int(NamedBarrierBwdSm100_MLA2CTA.Cpasync),
            num_threads=self.num_cpasync_load_threads,
        )
        self.softmax_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100_MLA2CTA.Softmax),
            num_threads=self.num_softmax_threads,
        )
        self.epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100_MLA2CTA.Epilogue),
            num_threads=self.num_epilogue_threads,
        )

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        # ==== Launch kernel ====
        block_dim = (self.num_threads, 1, 1)
        self.kernel(
            mV,
            mdV,
            tma_tensor_dO,
            tma_tensor_dOt,
            tma_tensor_Qvt,
            tma_tensor_P,
            tma_tensor_dS,
            mScaleP,
            mdPsum,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mIndexTopk,
            tma_atom_dO,
            tma_atom_dOt,
            tma_atom_Qvt,
            tma_atom_P,
            tma_atom_dS,
            self.sV_layout_staged,
            self.sdO_layout_staged,
            self.sdOt_layout_staged,
            self.sQvt_layout_staged,
            self.sP_layout_staged,  # load P
            sdS_layout_staged,  # store dS
            sdV_layout_staged,
            self.sPt_layout_staged,  # mma Pt
            self.sdSt_layout_staged,  # mma dSt
            self.sScaleP_layout_staged,
            self.sdPsum_layout_staged,
            tiled_mma_VdO,
            tiled_mma_PtdOt,
            tiled_mma_dStQvt,
            softmax_scale,
            softmax_scale_log2,
            topk_length_dynamic,
            tile_sched_params,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=block_dim,
            cluster=self.cluster_shape_mnk,
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mV: cute.Tensor,
        mdV: cute.Tensor,
        mdO: cute.Tensor,
        mdOt: cute.Tensor,
        mQvt: cute.Tensor,
        mP: cute.Tensor,
        mdS: cute.Tensor,
        mScaleP: Optional[cute.Tensor],
        mdPsum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mIndexTopk: Optional[cute.Tensor],
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOt: cute.CopyAtom,
        tma_atom_Qvt: cute.CopyAtom,
        tma_atom_P: cute.CopyAtom,
        tma_atom_dS: cute.CopyAtom,
        sV_layout_staged: cute.ComposedLayout,
        sdO_layout_staged: cute.ComposedLayout,
        sdOt_layout_staged: cute.ComposedLayout,
        sQvt_layout_staged: cute.ComposedLayout,
        sP_layout_staged: cute.ComposedLayout,
        sdS_layout_staged: cute.ComposedLayout,
        sdV_layout_staged: cute.ComposedLayout,
        sPt_layout_staged: cute.ComposedLayout,
        sdSt_layout_staged: cute.ComposedLayout,
        sScaleP_layout_staged: cute.Layout,
        sdPsum_layout_staged: cute.Layout,
        tiled_mma_VdO: cute.TiledMma,
        tiled_mma_PtdOt: cute.TiledMma,
        tiled_mma_dStQvt: cute.TiledMma,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        topk_length_dynamic: Optional[Int32],
        tile_sched_params: ParamsBase,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_VdO.thr_id.shape,)
        )
        mma_tile_coord_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_leader_cta = mma_tile_coord_v == 0

        # ==== Allocate SMEM ====
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # ==== Prepare TMEM allocator ====
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100_MLA2CTA.TmemPtr),
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
            cpasync.prefetch_descriptor(tma_atom_dO)
            cpasync.prefetch_descriptor(tma_atom_dOt)
            cpasync.prefetch_descriptor(tma_atom_Qvt)
            cpasync.prefetch_descriptor(tma_atom_P)
            cpasync.prefetch_descriptor(tma_atom_dS)

        # ==== Construct pipelines ====
        tma_warp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        mma_warp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        sm_warps = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_softmax_threads // 32)
        store_warp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        sm_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_softmax_threads)
        epi_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epilogue_threads)
        sm_threads_cluster = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_softmax_threads * self.cta_group_size
        )
        epi_threads_cluster = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_epilogue_threads * self.cta_group_size
        )
        cpasync_load_threads = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_cpasync_load_threads
        )
        relay_warps_cluster = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.cta_group_size)
        relay_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_relay_threads)

        TmaUmma = pipeline.PipelineTmaUmma
        TmaAsync = pipeline.PipelineTmaAsync
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
                **(
                    {"cta_layout_vmnk": cta_layout_vmnk}
                    if cls is not Async and cls is not TmaAsync
                    else {}
                ),
                **({"tx_count": tx_count} if tx_count is not None else {}),
            )

        # Unconditional pipelines
        # fmt: off
        # TmaUmma: dO, dOt & Qvt
        pipeline_dO = make_pipeline(TmaUmma, storage.mbar_ptr_dO, self.num_stages_dO, tma_warp, mma_warp, self.tma_copy_bytes_dO)
        pipeline_dOt_Qvt = make_pipeline(TmaUmma, storage.mbar_ptr_dOt_Qvt, self.num_stages_Qvt, tma_warp, mma_warp, self.tma_copy_bytes_dOt)
        # TmaAsync: P, scaleP
        pipeline_P = make_pipeline(TmaAsync, storage.mbar_ptr_P, self.num_stages_P, tma_warp, sm_warps, self.tma_copy_bytes_P)
        pipeline_scaleP = make_pipeline(TmaAsync, storage.mbar_ptr_scaleP, self.num_stages_scaleP, tma_warp, sm_warps, self.tma_copy_bytes_scaleP)
        pipeline_dPsum = make_pipeline(TmaAsync, storage.mbar_ptr_dPsum, self.num_stages_dPsum, tma_warp, sm_warps, self.tma_copy_bytes_dPsum)
        # AsyncUmma: Pt => dV mma, dSt => dV mma
        pipeline_Pt = make_pipeline(AsyncUmma, storage.mbar_ptr_Pt, self.num_stages_Pt, sm_threads_cluster, mma_warp)
        pipeline_dSt = make_pipeline(AsyncUmma, storage.mbar_ptr_dSt, self.num_stages_dSt, sm_threads_cluster, mma_warp)
        # UmmaAsync: dPt, dV
        pipeline_dPt = make_pipeline(UmmaAsync, storage.mbar_ptr_dPt, self.num_stages_dPt, mma_warp, sm_threads_cluster)
        pipeline_dV = make_pipeline(UmmaAsync, storage.mbar_ptr_dV, self.num_stages_dV, mma_warp, epi_threads_cluster)
        # Async: dV_epi
        pipeline_dV_epi = make_pipeline(Async, storage.mbar_ptr_dV_epi, self.num_stages_dV, tma_warp, store_warp)

        pipeline_V         = make_pipeline(AsyncUmma, storage.mbar_ptr_V,         self.num_stages_V, relay_warps_cluster,  mma_warp)
        pipeline_V_cpasync = make_pipeline(Async,     storage.mbar_ptr_V_cpasync, self.num_stages_V, cpasync_load_threads, relay_threads)
        # fmt: on

        pipeline.pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        # ==== Get SMEM tensors ====
        # fmt: off
        sV, sdO, sP, sPt, sdOt, sdS, sdSt, sQvt = (
            store.get_tensor(layout.outer, swizzle=layout.inner)
            for store, layout in [
                (storage.sV,  sV_layout_staged),
                (storage.sdO, sdO_layout_staged),
                (storage.sP,  sP_layout_staged),    # P & Pt overlap
                (storage.sP,  sPt_layout_staged),   # P & Pt overlap
                (storage.sQv, sdOt_layout_staged),  # {dOt, Qvt, dV} overlap
                (storage.sdS, sdS_layout_staged),   # dS & dSt overlap
                (storage.sdS, sdSt_layout_staged),  # dS & dSt overlap
                (storage.sQv, sQvt_layout_staged),  # {dOt, Qvt, dV} overlap
            ]
        )
        sdV = cute.make_tensor(
            cute.recast_ptr(sdOt.iterator, sdV_layout_staged.inner, self.dtype_acc), sdV_layout_staged.outer
        )
        assert cute.cosize(sdV) * self.dtype_acc.width // self.dtype.width == cute.cosize(sdOt)

        sScaleP = storage.sScaleP.get_tensor(sScaleP_layout_staged)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout_staged)
        # fmt: on

        # ==== Get thread MMAs and accumulator fragments ====
        thr_mma_VdO = tiled_mma_VdO.get_slice(mma_tile_coord_v)
        thr_mma_PtdOt = tiled_mma_PtdOt.get_slice(mma_tile_coord_v)
        thr_mma_dStQvt = tiled_mma_dStQvt.get_slice(mma_tile_coord_v)

        acc_shape_dPt = thr_mma_VdO.partition_shape_C(self.mma_tiler_VdO[:2])
        acc_shape_dVi = thr_mma_PtdOt.partition_shape_C(self.mma_tiler_PtdOt[:2])
        tdPtdP_fake = thr_mma_VdO.make_fragment_C(acc_shape_dPt)
        tdVtdV0_fake = thr_mma_PtdOt.make_fragment_C(acc_shape_dVi)
        tdVtdV1_fake = thr_mma_PtdOt.make_fragment_C(acc_shape_dVi)
        # tdPtdP = cute.make_tensor(tdPtdP.iterator + self.tmem_offset_dP, tdPtdP.layout)
        # tdVtdV0 = cute.make_tensor(tdVtdV0.iterator + self.tmem_offset_dV0, tdVtdV0.layout)
        # tdVtdV1 = cute.make_tensor(tdVtdV1.iterator + self.tmem_offset_dV1, tdVtdV1.layout)

        block_info = BlockInfo(
            self.tile_m * self.cta_group_size,
            self.tile_n,
            is_causal=self.is_causal,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mdO.shape[0] if const_expr(not self.pack_gqa) else mdO.shape[0][1],
            seqlen_k_static=mV.shape[0],
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
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
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                if is_leader_cta:
                    self.clc_scheduler_warp(tile_scheduler)
                else:
                    self.empty_warp(tile_scheduler)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i] and warp_idx != self.clc_scheduler_warp_id:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)

        if const_expr(self.use_cpasync_load_KV):
            if warp_idx == self.relay_warp_id:
                if const_expr(self.num_regs_load < self.num_regs_per_thread):
                    cute.arch.setmaxregister_decrease(self.num_regs_load)
                self.relay(
                    pipeline_V,
                    pipeline_V_cpasync,
                    topk_length_dynamic,
                    block_info,
                    SeqlenInfoCls,
                    tile_scheduler=tile_scheduler,
                )

            if warp_idx in self.cpasync_load_warp_indices:
                if const_expr(self.num_regs_cpasync < self.num_regs_per_thread):
                    cute.arch.setmaxregister_decrease(self.num_regs_cpasync)
                self.load_cpasync(
                    mIndexTopk,
                    mV,
                    sV,
                    pipeline_V,
                    pipeline_V_cpasync,
                    topk_length_dynamic,
                    block_info,
                    SeqlenInfoCls,
                    mCuSeqlensQ,
                    tile_scheduler=tile_scheduler,
                )

        if warp_idx == self.load_warp_id:
            if const_expr(self.num_regs_load < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                mdO,
                mP,
                mdOt,
                mQvt,
                mScaleP,
                mdPsum,
                sdO,
                sP,
                sdOt,
                sQvt,
                sScaleP,
                sdPsum,
                tma_atom_dO,
                tma_atom_P,
                tma_atom_dOt,
                tma_atom_Qvt,
                pipeline_dO,
                pipeline_P,
                pipeline_dOt_Qvt,
                pipeline_Pt,
                pipeline_dV_epi,
                pipeline_scaleP,
                pipeline_dPsum,
                thr_mma_VdO,
                thr_mma_PtdOt,
                thr_mma_dStQvt,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                mCuSeqlensQ,
                tile_scheduler=tile_scheduler,
            )

        if warp_idx == self.mma_warp_id:
            if const_expr(self.num_regs_mma < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_mma)
            # ==== Allocate TMEM ====
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_offset_dP, tdPtdP_fake.layout)
            tdVtdV0 = cute.make_tensor(tmem_ptr + self.tmem_offset_dV0, tdVtdV0_fake.layout)
            tdVtdV1 = cute.make_tensor(tmem_ptr + self.tmem_offset_dV1, tdVtdV1_fake.layout)
            self.mma(
                sV,
                sdO,
                sPt,
                sdOt,
                sdSt,
                sQvt,
                tdPtdP,
                tdVtdV0,
                tdVtdV1,
                tiled_mma_VdO,
                tiled_mma_PtdOt,
                tiled_mma_dStQvt,
                pipeline_V,
                pipeline_dO,
                pipeline_dPt,
                pipeline_Pt,
                pipeline_dOt_Qvt,
                pipeline_dSt,
                pipeline_dV,
                is_leader_cta,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                mCuSeqlensQ,
                tile_scheduler=tile_scheduler,
            )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if warp_idx in self.softmax_warp_indices:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_offset_dP, tdPtdP_fake.layout)
            self.compute_loop(
                softmax_scale,
                softmax_scale_log2,
                thr_mma_VdO,
                tdPtdP,
                sP,
                sdS,
                sScaleP,
                sdPsum,
                mdS,
                tma_atom_dS,
                pipeline_P,
                pipeline_Pt,
                pipeline_dPt,
                pipeline_dSt,
                pipeline_scaleP,
                pipeline_dPsum,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                mCuSeqlensQ,
                tile_scheduler=tile_scheduler,
            )
            tmem_alloc_barrier.arrive()

        if warp_idx in self.epilogue_warp_indices:
            if const_expr(self.num_regs_epilogue < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_epilogue)
            elif const_expr(self.num_regs_epilogue > self.num_regs_per_thread):
                cute.arch.setmaxregister_increase(self.num_regs_epilogue)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            tdVtdV0 = cute.make_tensor(tmem_ptr + self.tmem_offset_dV0, tdVtdV0_fake.layout)
            tdVtdV1 = cute.make_tensor(tmem_ptr + self.tmem_offset_dV1, tdVtdV1_fake.layout)
            self.dVacc_store(
                mIndexTopk,
                mdV,
                sdV,
                tdVtdV0,
                tdVtdV1,
                thr_mma_PtdOt,
                pipeline_dV,
                pipeline_dV_epi,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                mCuSeqlensQ,
                tile_scheduler=tile_scheduler,
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
            cta_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
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
        pipeline_V: pipeline.PipelineAsyncUmma,
        pipeline_V_cpasync: pipeline.PipelineAsync,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== Make pipeline states ====
        # pipeline_V producer
        # pipeline_V_cpasync consumer
        producer_state_V = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_V
        )
        consumer_state_V = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_V
        )
        relay_V_fn = partial(self.relay_inner, pipeline_V_cpasync, pipeline_V)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            # seqlen = SeqlenInfoCls(batch_idx)

            num_n_block_groups = self.topk_length // self.cluster_tile_n
            # num_n_block_groups = topk_length_dynamic // self.cluster_tile_n

            # ==== Mainloop ====
            for _ in cutlass.range(num_n_block_groups, unroll=1):
                for _ in cutlass.range_constexpr(self.num_hdimv_splits):
                    consumer_state_V, producer_state_V = relay_V_fn(
                        consumer_state_V, producer_state_V
                    )

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

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
        mV: cute.Tensor,
        sV: cute.Tensor,
        pipeline_V: pipeline.PipelineAsyncUmma,
        pipeline_V_cpasync: pipeline.PipelineAsync,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        mCuSeqlensQ: Optional[cute.Tensor],
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== cpasync load warpgroup ====
        # Description: loads tiles of V from gmem to smem using cpasync
        # produces: V
        # consumes: -

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        tidx = cute.arch.thread_idx()[0] % self.num_cpasync_load_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (
            self.num_cpasync_load_threads // 32
        )

        # ==== Make pipeline states ====
        # producer: acquire PipelineAsyncUmma <- mma
        # producer: commit  PipelineAsync     -> relay
        producer_state_V = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_V
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(mCuSeqlensQ is not None):
                batch_idx = get_batch_from_cu_tensor(m_block, mCuSeqlensQ)
            seqlen = SeqlenInfoCls(batch_idx)

            num_n_block_groups = self.topk_length // self.cluster_tile_n
            # num_n_block_groups = topk_length_dynamic // self.cluster_tile_n

            if const_expr(seqlen.has_cu_seqlens_q):
                # m_block means absolute m_idx
                mIndexTopk_cur = mIndexTopk[None, m_block]
            else:
                mIndexTopk_cur = mIndexTopk[None, m_block, batch_idx]

            if const_expr(self.is_causal):
                m_local_idx = (
                    m_block - seqlen.offset_q if const_expr(seqlen.has_cu_seqlens_q) else m_block
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
                self.cluster_tile_n,
                self.hdim,
                self.hdimv,
                self.num_hdimv_splits,
                self.num_cpasync_load_threads,
                mV.element_type,
                self.cta_group_size,
                self.cpasync_barrier,
                self.disable_bitmask,
            )

            # (seqlen_k, hdimv)
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx]

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

            # ==== Mainloop ====
            for n_block_group in cutlass.range(num_n_block_groups, unroll=1):
                cpasync_gather_kv_manager.load_index_topk(n_block_group, transpose=False)
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    producer_state_V = load_V(producer_state_V, d_offset=split * self.hdimv // 2)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        pipeline_V.producer_tail(producer_state_V)

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
    def load(
        self,
        mdO: cute.Tensor,
        mP: cute.Tensor,
        mdOt: cute.Tensor,
        mQvt: cute.Tensor,
        mScaleP: Optional[cute.Tensor],
        mdPsum: Optional[cute.Tensor],
        sdO: cute.Tensor,
        sP: cute.Tensor,
        sdOt: cute.Tensor,
        sQvt: cute.Tensor,
        sScaleP: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_P: cute.CopyAtom,
        tma_atom_dOt: cute.CopyAtom,
        tma_atom_Qvt: cute.CopyAtom,
        pipeline_dO: pipeline.PipelineAsync,  # TmaUmma
        pipeline_P: pipeline.PipelineAsync,  # TmaAsync
        pipeline_dOt_Qvt: pipeline.PipelineAsync,  # TmaUmma
        pipeline_Pt: pipeline.PipelineAsync,  # AsyncUmma
        pipeline_dV_epi: pipeline.PipelineAsync,  # Async
        pipeline_scaleP: pipeline.PipelineAsync,  # TmaAsync
        pipeline_dPsum: pipeline.PipelineAsync,  # TmaAsync
        thr_mma_VdO: cute.ThrMma,
        thr_mma_PtdOt: cute.ThrMma,
        thr_mma_dStQvt: cute.ThrMma,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        mCuSeqlensQ: Optional[cute.Tensor],
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== Load warp ====
        # Description: loads tiles of dO, P, dOt, Qvt from gmem to smem using TMA
        # produces: dO, P, dOt, Qvt
        # consumes: -
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        lane_idx = cute.arch.lane_idx()

        # ==== Make pipeline states ====
        Producer = pipeline.PipelineUserType.Producer
        producer_state_dO = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dO)
        producer_state_P = pipeline.make_pipeline_state(Producer, stages=self.num_stages_P)
        producer_state_dOt_Qvt = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dOt)
        producer_state_dV_epi = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dV)
        producer_state_scaleP = pipeline.make_pipeline_state(
            Producer, stages=self.num_stages_scaleP
        )
        producer_state_dPsum = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dPsum)

        copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(mCuSeqlensQ is not None):
                batch_idx = get_batch_from_cu_tensor(m_block, mCuSeqlensQ)
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(mCuSeqlensQ is not None):
                m_block -= seqlen.offset_q
            num_n_block_groups = self.topk_length // self.cluster_tile_n
            # num_n_block_groups = topk_length_dynamic // self.cluster_tile_n

            # ==== Partition GMEM tensors ====
            # (seqlen_q, topk or hdimv)
            mP_cur = seqlen.offset_batch_Q(mP, batch_idx, dim=3)[None, None, head_idx]
            mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx]

            # (hdimv, seqlen_q)
            offset = (
                (0, seqlen.offset_q) if const_expr(not self.pack_gqa) else (0, (0, seqlen.offset_q))
            )
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdOt_cur = mdOt[None, None, head_idx, batch_idx]
                mQvt_cur = mQvt[None, None, head_idx, batch_idx]
            else:
                mdOt_cur = cute.domain_offset(offset, mdOt[None, None, head_idx])
                mQvt_cur = cute.domain_offset(offset, mQvt[None, None, head_idx])

            gScaleP = None
            if const_expr(mScaleP is not None):
                mScaleP_cur = seqlen.offset_batch_Q(mScaleP, batch_idx, dim=3)[None, None, head_idx]
                # (tile_m, topk//128)
                gScaleP = cute.local_tile(mScaleP_cur, (self.tile_m,), (m_block, None))
            gdPsum = None
            if const_expr(mdPsum is not None):
                mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2)[None, head_idx]
                # (tile_m)
                gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (m_block,))

            # (tile_m, tile_n, n_blocks)
            gP = cute.local_tile(
                mP_cur,
                (self.tile_m, self.tile_n),
                (m_block, None),
            )
            # (tile_m, hdimv//2, 2)
            gdO = cute.local_tile(
                mdO_cur,
                (self.mma_tiler_VdO[1], self.mma_tiler_VdO[2]),
                (m_block, None),
            )
            # (hdimv//2, tile_m, 2)
            gdOt = cute.local_tile(
                mdOt_cur,
                (self.mma_tiler_PtdOt[1], self.mma_tiler_PtdOt[2]),
                (None, m_block),
            )
            gQvt = cute.local_tile(
                mQvt_cur,
                (self.mma_tiler_dStQvt[1], self.mma_tiler_dStQvt[2]),
                (None, m_block),
            )

            tdPgdO = thr_mma_VdO.partition_B(gdO)
            tdVgdOt = thr_mma_PtdOt.partition_B(gdOt)
            tdVgQvt = thr_mma_dStQvt.partition_B(gQvt)

            # (V, REST)
            tPsP, tPgP = cpasync.tma_partition(
                atom=tma_atom_P,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sP, 0, 2),
                gmem_tensor=cute.group_modes(gP, 0, 2),
            )
            tdOsdO, tdOgdO = cpasync.tma_partition(
                atom=tma_atom_dO,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sdO, 0, 3),
                gmem_tensor=cute.group_modes(tdPgdO, 0, 3),
            )
            tdOtsdOt, tdOtgdOt = cpasync.tma_partition(
                atom=tma_atom_dOt,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sdOt, 0, 3),
                gmem_tensor=cute.group_modes(tdVgdOt, 0, 3),
            )
            tQvtsQvt, tQvtgQvt = cpasync.tma_partition(
                atom=tma_atom_Qvt,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sQvt, 0, 3),
                gmem_tensor=cute.group_modes(tdVgQvt, 0, 3),
            )

            load_P = partial(self.load_inner, tma_atom_P, tPgP, tPsP, pipeline_P)
            load_dO = partial(self.load_inner, tma_atom_dO, tdOgdO, tdOsdO, pipeline_dO)
            load_dOt = partial(self.load_inner, tma_atom_dOt, tdOtgdOt, tdOtsdOt, pipeline_dOt_Qvt)
            load_Qvt = partial(self.load_inner, tma_atom_Qvt, tQvtgQvt, tQvtsQvt, pipeline_dOt_Qvt)
            load_scaleP = partial(
                self.load_inner, copy_atom_stats, gScaleP, sScaleP, pipeline_scaleP, bulk_copy=True
            )
            load_dPsum = partial(
                self.load_inner, copy_atom_stats, gdPsum, sdPsum, pipeline_dPsum, bulk_copy=True
            )

            # ==== Load stationary operands ====
            for split in cutlass.range_constexpr(self.num_hdimv_splits):
                producer_state_dO = load_dO(producer_state_dO, block=split)

            producer_state_dPsum = load_dPsum(producer_state_dPsum)

            # ==== Mainloop ====
            for n_block_group in cutlass.range(num_n_block_groups, unroll=1):
                n_block = 2 * n_block_group + cta_rank_in_cluster
                # load ScaleP
                if const_expr(mScaleP is not None):
                    producer_state_scaleP = load_scaleP(producer_state_scaleP, block=n_block_group)
                pipeline_Pt.producer_acquire(producer_state_P)
                producer_state_P = load_P(producer_state_P, block=n_block)
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    pipeline_dV_epi.producer_acquire(producer_state_dV_epi)
                    producer_state_dV_epi.advance()
                    producer_state_dOt_Qvt = load_dOt(producer_state_dOt_Qvt, block=split)
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    producer_state_dOt_Qvt = load_Qvt(producer_state_dOt_Qvt, block=split)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        pipeline_P.producer_tail(producer_state_P)
        pipeline_dO.producer_tail(producer_state_dO)
        pipeline_dOt_Qvt.producer_tail(producer_state_dOt_Qvt)

    @cute.jit
    def load_inner(
        self,
        copy_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        load_pipeline: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        block: Optional[Int32] = None,
        bulk_copy: bool = False,
    ):
        if const_expr(block is not None):
            tXgX = tXgX[(None, block)]
        if const_expr(cute.rank(tXsX) != 1):
            assert cute.rank(tXsX) == 2, f"wrong rank for tXsX, got {cute.rank(tXsX)}"
            stage = producer_state.index
            tXsX = tXsX[(None, stage)]

        load_pipeline.producer_acquire(producer_state)
        mbar_ptr = load_pipeline.producer_get_barrier(producer_state)
        if const_expr(bulk_copy):
            with cute.arch.elect_one():
                cute.copy(copy_atom, tXgX, tXsX, mbar_ptr=mbar_ptr)
        else:
            cute.copy(copy_atom, tXgX, tXsX, tma_bar_ptr=mbar_ptr)
        producer_state.advance()
        return producer_state

    @cute.jit
    def mma(
        self,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sPt: cute.Tensor,
        sdOt: cute.Tensor,
        sdSt: cute.Tensor,
        sQvt: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV0: cute.Tensor,
        tdVtdV1: cute.Tensor,
        tiled_mma_VdO: cute.TiledMma,
        tiled_mma_PtdOt: cute.TiledMma,
        tiled_mma_dStQvt: cute.TiledMma,
        pipeline_V: pipeline.PipelineAsync,  # AsyncUmma
        pipeline_dO: pipeline.PipelineAsync,  # TmaUmma
        pipeline_dPt: pipeline.PipelineAsync,  # UmmaAsync
        pipeline_Pt: pipeline.PipelineAsync,  # AsyncUmma
        pipeline_dOt_Qvt: pipeline.PipelineAsync,  # TmaUmma
        pipeline_dSt: pipeline.PipelineAsync,  # AsyncUmma
        pipeline_dV: pipeline.PipelineAsync,  # UmmaAsync
        is_leader_cta: Boolean,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        mCuSeqlensQ: Optional[cute.Tensor],
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== mma warp ====
        # Description: Computes dP = V @ dO^T, dV = P^T @ dO, and dV += dS^T @ Qv
        # i.e. dP = gemm(V, dO), dV += gemm(P.T, dO.T), dV += gemm(dS.T, Qv.T)
        # Produces: dP, dV
        # Consumes: V, dO, P.T, dO.T, dS.T, Qv.T
        lane_idx = cute.arch.lane_idx()

        tdVtdVs = [tdVtdV0, tdVtdV1]

        # Set accumulate = True for dS^T@Qv since we are accumulating on the P^T@dO result
        tiled_mma_dStQvt.set(tcgen05.Field.ACCUMULATE, True)

        # Operands for dP=V@dO^T
        tdPrV = tiled_mma_VdO.make_fragment_A(sV)
        tdPrdO = tiled_mma_VdO.make_fragment_B(sdO)

        # Operands for dVi=P^T@dOi
        tdVrPt = tiled_mma_PtdOt.make_fragment_A(sPt)
        tdVrdOt = tiled_mma_PtdOt.make_fragment_B(sdOt)

        # Operands for dVi+=dS^T@Qvi
        tdVrdSt = tiled_mma_dStQvt.make_fragment_A(sdSt)
        tdVrQvt = tiled_mma_dStQvt.make_fragment_B(sQvt)

        use_ptx_gemm_VdO = False
        use_ptx_gemm_PtdOt = False
        use_ptx_gemm_dStQvt = False

        # GEMM functions
        if const_expr(use_ptx_gemm_VdO):
            gemm_VdO = partial(
                fa_sm100_utils.gemm_ptx_partial,
                tiled_mma_VdO.op,
                self.tmem_offset_dP,
                zero_init=True,
                cta_group=self.cta_group_size,
            )
        else:
            gemm_VdO = partial(
                fa_sm100_utils.gemm,
                tiled_mma_VdO,
                tdPtdP,
            )
        if const_expr(use_ptx_gemm_PtdOt):
            gemm_PtdOt = [
                partial(
                    fa_sm100_utils.gemm_ptx_partial,
                    tiled_mma_PtdOt.op,
                    self.tmem_offsets_dV[split],
                    zero_init=True,
                    cta_group=self.cta_group_size,
                )
                for split in range(self.num_hdimv_splits)
            ]
        else:
            gemm_PtdOt = [
                partial(
                    fa_sm100_utils.gemm,
                    tiled_mma_PtdOt,
                    tdVtdVs[split],
                    zero_init=True,
                )
                for split in range(self.num_hdimv_splits)
            ]
        if const_expr(use_ptx_gemm_dStQvt):
            gemm_dStQvt = [
                partial(
                    fa_sm100_utils.gemm_ptx_partial,
                    tiled_mma_dStQvt.op,
                    self.tmem_offsets_dV[split],
                    zero_init=False,
                    cta_group=self.cta_group_size,
                )
                for split in range(self.num_hdimv_splits)
            ]
        else:
            gemm_dStQvt = [
                partial(
                    fa_sm100_utils.gemm,
                    tiled_mma_dStQvt,
                    tdVtdVs[split],
                    zero_init=False,
                )
                for split in range(self.num_hdimv_splits)
            ]

        Consumer, Producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer
        consumer_state_V = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_V)
        consumer_state_dO = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_dO)
        consumer_state_Pt = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_Pt)
        consumer_state_dOt_Qvt = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_dOt)
        consumer_state_dSt = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_dSt)
        producer_state_dPt = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dPt)
        producer_state_dV = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dV)

        mma_VdO = partial(
            self.mma_inner, gemm_VdO, pipeline_V, tdPrV, sV, tdPrdO, sdO, swap_AB_stage=True, use_ptx=use_ptx_gemm_VdO
        )
        mma_PtdOt = partial(
            self.mma_inner, gemm_PtdOt, pipeline_dOt_Qvt, tdVrPt, sPt, tdVrdOt, sdOt, use_ptx=use_ptx_gemm_PtdOt
        )
        mma_dStQvt = partial(
            self.mma_inner, gemm_dStQvt, pipeline_dOt_Qvt, tdVrdSt, sdSt, tdVrQvt, sQvt, use_ptx=use_ptx_gemm_dStQvt
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            # if const_expr(mCuSeqlensQ is not None):
            #     batch_idx = get_batch_from_cu_tensor(m_block, mCuSeqlensQ)
            # seqlen = SeqlenInfoCls(batch_idx)
            # num_n_block_groups = self.topk_length // self.cluster_tile_n
            num_n_block_groups = topk_length_dynamic // self.cluster_tile_n

            if is_leader_cta:
                # ==== Prologue ====
                consumer_wait_state_dO = consumer_state_dO.clone()
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    pipeline_dO.consumer_wait(consumer_wait_state_dO)
                    consumer_wait_state_dO.advance()

                # ==== Mainloop ====
                for _ in cutlass.range(num_n_block_groups, unroll=1):
                    # 1. dP = V @ dO^T
                    # mma inner waits for V
                    pipeline_dPt.producer_acquire(producer_state_dPt)
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        consumer_state_V = mma_VdO(
                            consumer_state_V, a_stage=split, zero_init=split == 0
                        )
                    pipeline_dPt.producer_commit(producer_state_dPt)
                    producer_state_dPt.advance()

                    # 2. dV = P^T @ dO
                    # mma inner waits for dOt
                    pipeline_Pt.consumer_wait(consumer_state_Pt)
                    producer_acquire_state_dV = producer_state_dV.clone()
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        pipeline_dV.producer_acquire(producer_acquire_state_dV)
                        producer_acquire_state_dV.advance()
                        consumer_state_dOt_Qvt = mma_PtdOt(consumer_state_dOt_Qvt, acc_stage=split)
                    pipeline_Pt.consumer_release(consumer_state_Pt)
                    consumer_state_Pt.advance()

                    # 3. dV += dS^T @ Qv
                    # mma inner waits for Qvt
                    pipeline_dSt.consumer_wait(consumer_state_dSt)
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        consumer_state_dOt_Qvt = mma_dStQvt(consumer_state_dOt_Qvt, acc_stage=split)
                        pipeline_dV.producer_commit(producer_state_dV)
                        producer_state_dV.advance()
                    pipeline_dSt.consumer_release(consumer_state_dSt)
                    consumer_state_dSt.advance()

                # ==== Epilogue ====
                for _ in cutlass.range_constexpr(self.num_hdimv_splits):
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        pipeline_dPt.producer_tail(producer_state_dPt)
        pipeline_dV.producer_tail(producer_state_dV)

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
        swap_AB_stage: bool = False,
        use_ptx: bool = True,
    ):
        if const_expr(acc_stage is not None):
            gemm = gemm[acc_stage]

        smem_stage = consumer_state.index

        if const_expr(not swap_AB_stage):
            a_stage = a_stage
            b_stage = smem_stage
        else:
            a_stage = smem_stage
            b_stage = a_stage

        tCrA_cur = tCrA[None, None, None, a_stage]
        sA_cur = sA[None, None, None, a_stage]
        tCrB_cur = tCrB[None, None, None, b_stage]
        sB_cur = sB[None, None, None, b_stage]

        kwargs = dict(tCrA=tCrA_cur, tCrB=tCrB_cur)
        if const_expr(use_ptx):
            kwargs |= dict(sA=sA_cur, sB=sB_cur)
        if const_expr(zero_init is not None):
            kwargs["zero_init"] = zero_init

        load_pipeline.consumer_wait(consumer_state)
        gemm(**kwargs)
        load_pipeline.consumer_release(consumer_state)
        consumer_state.advance()
        return consumer_state

    @cute.jit
    def compute_loop(
        self,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        thr_mma_VdO: cute.ThrMma,
        tdPtdP: cute.Tensor,
        sP: cute.Tensor,
        sdS: cute.Tensor,
        sScaleP: cute.Tensor,
        sdPsum: cute.Tensor,
        mdS: cute.Tensor,
        tma_atom_dS: cute.CopyAtom,
        pipeline_P: pipeline.PipelineAsync,  # TmaAsync
        pipeline_Pt: pipeline.PipelineAsync,  # AsyncUmma
        pipeline_dPt: pipeline.PipelineAsync,  # UmmaAsync
        pipeline_dSt: pipeline.PipelineAsync,  # AsyncUmma
        pipeline_scaleP: pipeline.PipelineAsync,  # TmaAsync
        pipeline_dPsum: pipeline.PipelineAsync,  # TmaAsync
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        mCuSeqlensQ: Optional[cute.Tensor],
        tile_scheduler: TileSchedulerProtocol,
    ):
        tidx = cute.arch.thread_idx()[0] % self.num_softmax_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (
            self.num_softmax_threads // 32
        )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        leader_warp = warp_idx == 0

        # 256b // 32 = 8 values, 128 mqa // 2 => tmem_rep = 8
        tmem_rep = self.tile_m // self.cta_group_size // 8
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(tmem_rep)),
            self.dtype_acc,
        )
        # ((64,(64,2)),1,1):((65536,(1,4194304 = 65536*64)),0,0)
        tdPtdP = tdPtdP[(None, None), 0, 0]
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tdPtdP)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N)
        # (((64,16),1),2,1):(((1,65536),0),1048576,0)>, 1048576/65536 = 16
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)

        cdP = cute.make_identity_tensor(self.mma_tiler_VdO[:2])  # (128, 128)
        tdPcdP = thr_mma_VdO.partition_C(cdP)[(None, None), 0, 0]  # (64,128):(1@0,1@1)
        # (((2,2,8),1),2,1):(((1@1,8@0,8@1),0),16@0,0)
        tdPcdP_t2r = thr_copy_t2r.partition_D(tdPcdP)
        assert tdPcdP_t2r.shape[0][1] == 1, f"unexpected tdPcdP_t2r shape, got {tdPcdP_t2r.shape}"

        smem_load_op = cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4)  # ldsm x num_matrices = ldsm x 4
        smem_store_op = cute.nvgpu.warp.StMatrix8x8x16bOp(True, 4)  # stsm x num_matrices = stsm x 4
        smem_load_atom = cute.make_copy_atom(smem_load_op, self.dtype)
        smem_store_atom = cute.make_copy_atom(smem_store_op, self.dtype)
        tiled_copy_r2s = cute.make_tiled_copy_D(smem_store_atom, tiled_copy_t2r)
        tiled_copy_s2r = cute.make_tiled_copy_D(smem_load_atom, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)

        sPt_load_layout = cute.make_ordered_layout(
            self.tile_Pt + (self.num_stages_P,), order=(1, 0, 2)
        )
        # (tile_n, tile_m, stages_P)
        sPt = cute.composition(sP, sPt_load_layout)
        sdSt = cute.composition(sdS, sPt_load_layout)

        # (R2S, R2S_M, R2S_N, PIPE_D)
        # ((8,4),2,1,1):((1,1024),16,0,0)
        tSR_sPt = thr_copy_s2r.partition_S(sPt)
        tRS_sdSt = thr_copy_r2s.partition_D(sdSt)

        # ((2,2),(2,8,1),stage):((0,0),(1,8,0),_)
        tPsScaleP_nm = self.broadcast_tensor_nm_view(sScaleP, thr_mma_VdO, thr_copy_t2r)
        tdPsdPsum_nm = self.broadcast_tensor_nm_view(sdPsum, thr_mma_VdO, thr_copy_t2r)

        Consumer, Producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer

        consumer_state_P = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_P)
        consumer_state_dPt = pipeline.make_pipeline_state(Consumer, stages=1)
        consumer_state_scaleP = pipeline.make_pipeline_state(
            Consumer, stages=self.num_stages_scaleP
        )
        consumer_state_dPsum = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_dPsum)

        producer_state_Pt = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Pt)
        producer_state_dSt = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dSt)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(mCuSeqlensQ is not None):
                batch_idx = get_batch_from_cu_tensor(m_block, mCuSeqlensQ)
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(mCuSeqlensQ is not None):
                m_block -= seqlen.offset_q
            num_n_block_groups = self.topk_length // self.cluster_tile_n
            # num_n_block_groups = topk_length_dynamic // self.cluster_tile_n

            mdS_cur = seqlen.offset_batch_Q(mdS, batch_idx, dim=3)[None, None, head_idx]

            gdS = cute.local_tile(mdS_cur, (self.tile_m, self.tile_n), (m_block, None))
            store_dS, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dS,
                0,
                cute.make_layout(1),
                sdS,
                gdS,
            )

            pipeline_dPsum.consumer_wait(consumer_state_dPsum)

            tdPsdPsum_cur = tdPsdPsum_nm[0, None, consumer_state_dPsum.index]
            tdPrdPsum_cur_f32 = cute.make_rmem_tensor(tdPsdPsum_cur.shape, dtype=self.dtype_scale)
            cute.autovec_copy(tdPsdPsum_cur, tdPrdPsum_cur_f32)

            for n_block_group in cutlass.range(num_n_block_groups, unroll=1):
                n_block = 2 * n_block_group + cta_rank_in_cluster

                pipeline_P.consumer_wait(consumer_state_P)
                # todo: ablate wait -> try_wait
                pipeline_scaleP.consumer_wait(consumer_state_scaleP)

                # (((2,2,8),1),2,1):(((1,2,4),0),32,0)
                rPt = cute.make_rmem_tensor(tdPcdP_t2r.shape, self.dtype)
                # (S2R, S2R_M, S2R_N)
                rPt_copy_view = tiled_copy_s2r.retile(rPt)
                tSR_sPt_cur = tSR_sPt[None, None, None, consumer_state_P.index]
                cute.copy(tiled_copy_s2r, tSR_sPt_cur, rPt_copy_view)

                # ((2,2),(2,8,1)):((2,32),(1,4,0))
                rP_nm = layout_utils.reshape_acc_to_mn(rPt[(None, 0), None, None])

                tPsScaleP_cur = tPsScaleP_nm[0, None, consumer_state_scaleP.index]
                tPrScaleP_cur_f32 = cute.make_rmem_tensor(
                    tPsScaleP_cur.shape, dtype=self.dtype_scale
                )
                tPrScaleP_cur = cute.make_rmem_tensor(tPsScaleP_cur.shape, dtype=self.dtype)
                cute.autovec_copy(tPsScaleP_cur, tPrScaleP_cur_f32)
                tPrScaleP_cur.store(tPrScaleP_cur_f32.load().to(self.dtype))

                # scale P
                for n in cutlass.range_constexpr(cute.size(rP_nm.shape[0])):
                    rP_cur = rP_nm[n, None]
                    rP_cur.store(rP_cur.load() * tPrScaleP_cur.load())
                cute.arch.sync_warp()

                cute.copy(tiled_copy_r2s, rPt_copy_view, tSR_sPt_cur)
                cute.arch.fence_view_async_shared()
                self.softmax_barrier.arrive_and_wait()

                pipeline_scaleP.consumer_release(consumer_state_scaleP)
                consumer_state_scaleP.advance()

                pipeline_Pt.producer_commit(producer_state_Pt)
                producer_state_Pt.advance()

                # note: mma also signals Pt free, signal acquired in tma warp
                pipeline_P.consumer_release(consumer_state_P)
                consumer_state_P.advance()

                pipeline_dPt.consumer_wait(consumer_state_dPt)

                # (((2,2,8),1),2,1):(((1,2,4),0),32,0)
                tdPrdP_t2r = cute.make_rmem_tensor(tdPcdP_t2r.shape, self.dtype_acc)
                cute.copy(tiled_copy_t2r, tdPtdP_t2r, tdPrdP_t2r)
                cute.arch.fence_view_async_tmem_load()
                self.softmax_barrier.arrive_and_wait()

                pipeline_dPt.consumer_release(consumer_state_dPt)
                consumer_state_dPt.advance()

                # dS = P o (dP - dPsum)
                rdP_nm = layout_utils.reshape_acc_to_mn(tdPrdP_t2r[(None, 0), None, None])
                for n in cutlass.range_constexpr(cute.size(rdP_nm.shape[0])):
                    rdP_cur = rdP_nm[n, None]
                    rdP_cur.store(rdP_cur.load() - tdPrdPsum_cur_f32.load())

                rPt.store(rPt.load() * (tdPrdP_t2r.load() * softmax_scale).to(self.dtype))

                # wait for tma store to free dSt buffer
                if leader_warp:
                    cute.arch.cp_async_bulk_wait_group(1 - self.num_stages_dSt, read=True)
                self.softmax_barrier.arrive_and_wait()

                # note: dS guaranteed free as mma operand
                pipeline_dSt.producer_acquire(producer_state_dSt)

                tRS_sdSt_cur = tRS_sdSt[None, None, None, producer_state_dSt.index]
                cute.copy(tiled_copy_r2s, rPt_copy_view, tRS_sdSt_cur)

                cute.arch.fence_view_async_shared()
                self.softmax_barrier.arrive_and_wait()
                pipeline_dSt.producer_commit(producer_state_dSt)

                # tma store
                if leader_warp:
                    store_dS(src_idx=producer_state_dSt.index, dst_idx=n_block)
                    cute.arch.cp_async_bulk_commit_group()

                producer_state_dSt.advance()

            pipeline_dPsum.consumer_release(consumer_state_dPsum)
            consumer_state_dPsum.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        # producer tails

    @cute.jit
    def broadcast_tensor_nm_view(
        self,
        sX: cute.Tensor,  # (tile_m, num_stages)
        thr_mma: cute.ThrMma,
        thr_copy_t2r: cute.ThrCopy,
    ):
        assert cute.size(sX.shape[0]) == self.tile_m
        num_stages = sX.shape[1] if const_expr(cute.rank(sX) > 1) else 1
        sX_2D_cluster = cute.make_tensor(
            sX.iterator,
            cute.make_layout(
                (self.tile_m, self.cluster_tile_n, num_stages),
                stride=(1, 0, self.tile_m),
            ),
        )
        sXt_2D_cluster = layout_utils.transpose_view(sX_2D_cluster)
        sXt_2D = thr_mma.partition_C(sXt_2D_cluster)[(None, None), 0, 0, None]
        tXsXt_2D = thr_copy_t2r.partition_D(sXt_2D)[(None, 0), None, None, None]
        tXsXt_nm = layout_utils.make_acc_tensor_mn_view(tXsXt_2D)
        return tXsXt_nm

    @cute.jit
    def dVacc_store(
        self,
        mIndexTopk: cute.Tensor,
        mdV: cute.Tensor,
        sdV: cute.Tensor,
        tdVtdV0: cute.Tensor,
        tdVtdV1: cute.Tensor,
        thr_mma_PtdOt: cute.ThrMma,
        pipeline_dV: pipeline.PipelineAsync,  # UmmaAsync
        pipeline_dV_epi: pipeline.PipelineAsync,  # Async
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        mCuSeqlensQ: Optional[cute.Tensor],
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== dVaccum store warpgroup ====
        # produces: -
        # consumes: dV

        tdVtdV0 = tdVtdV0[(None, None), 0, 0]
        tdVtdV1 = tdVtdV1[(None, None), 0, 0]

        num_epi_warps = self.num_epilogue_threads // 32
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        tidx = cute.arch.thread_idx()[0] % self.num_epilogue_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % num_epi_warps
        leader_warp = warp_idx == 0
        wg_half = warp_idx // 2

        consumer_state_dV = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_dV
        )

        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.dtype_acc,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tdVtdV0)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tdVtdV0_t2r = thr_copy_t2r.partition_S(tdVtdV0)
        tdVtdV1_t2r = thr_copy_t2r.partition_S(tdVtdV1)
        tdVtdVs_t2r = [tdVtdV0_t2r, tdVtdV1_t2r]

        cdVmma = cute.make_identity_tensor(self.mma_tiler_PtdOt[:2])
        tdVcdVmma = thr_mma_PtdOt.partition_C(cdVmma)[(None, None), 0, 0]
        tdVcdVmma_t2r = thr_copy_t2r.partition_D(tdVcdVmma)

        # 64 threads x 4 values to tile over tile_dV = (64, 32)
        tiled_copy_r2s = tiled_copy_2d(self.dtype_acc, 4, 64)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx % 64)

        # ((4,1),1,8,(1,8)):((1,0),0,4,(0,2048))
        tRS_sdV = thr_copy_r2s.partition_D(sdV)

        tiled_copy_s2r = copy_utils.tiled_copy_2d(self.dtype_acc, 8, self.num_epilogue_threads, 4)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        # (V, M, N, STAGE)
        tSR_sdV = thr_copy_s2r.partition_S(sdV)

        cdV = cute.make_identity_tensor(cute.product_each(sdV.shape[:2]))
        # (V, M, N)
        tdVcdV = thr_copy_s2r.partition_S(cdV)

        gmem_rows_per_thread = cute.size(tSR_sdV.shape[1])

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(mCuSeqlensQ is not None):
                batch_idx = get_batch_from_cu_tensor(m_block, mCuSeqlensQ)
            seqlen = SeqlenInfoCls(batch_idx)
            num_n_block_groups = self.topk_length // self.cluster_tile_n
            # num_n_block_groups = topk_length_dynamic // self.cluster_tile_n

            # (seqlen_k, hdimv)
            mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[None, None, head_idx]

            # (topk, dv)
            if const_expr(seqlen.has_cu_seqlens_q):
                # m_block means absolute m_idx
                mIndexTopk_cur = mIndexTopk[None, m_block]
            else:
                mIndexTopk_cur = mIndexTopk[None, m_block, batch_idx]

            # ==== Mainloop ====
            for n_block_group in cutlass.range(num_n_block_groups, unroll=1):
                n_block = 2 * n_block_group + cta_rank_in_cluster

                rIdxTopK = cute.make_rmem_tensor((gmem_rows_per_thread,), dtype=self.dtype_index)
                for j in cutlass.range_constexpr(gmem_rows_per_thread):
                    n_idx = n_block * self.tile_n + tdVcdV[0, j, 0][0]
                    rIdxTopK[j] = mIndexTopk_cur[n_idx]

                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    tdVtdV_t2r = tdVtdVs_t2r[split]

                    pipeline_dV.consumer_wait(consumer_state_dV)

                    # TODO: record meaning of hard-coded values
                    num_cols_per_store = self.tile_dV[1] * 2
                    num_epi_subtiles = (self.hdimv // self.num_hdimv_splits) // num_cols_per_store
                    assert num_cols_per_store == 64
                    assert num_epi_subtiles == 4
                    assert cute.size(tdVtdV_t2r.shape[2]) == num_epi_subtiles

                    tdVrdV_cur_shape = tdVcdVmma_t2r[None, None, 0].shape
                    tRS_rdV_cur_shape = tRS_sdV[None, None, None, 0].shape
                    assert cute.size(tdVrdV_cur_shape) == cute.size(tRS_rdV_cur_shape)

                    tdVrdV_out_shape = tSR_sdV[None, None, None, 0].shape + (2,)

                    for i in cutlass.range_constexpr(num_epi_subtiles):
                        tdVrdV_cur = cute.make_rmem_tensor(tdVrdV_cur_shape, self.dtype_acc)
                        cute.copy(tiled_copy_t2r, tdVtdV_t2r[None, None, i], tdVrdV_cur)

                        tRS_rdV_cur = cute.make_tensor(tdVrdV_cur.iterator, tRS_rdV_cur_shape)

                        stage = 4 * split + 2 * wg_half + (i % 2)
                        cute.copy(tiled_copy_r2s, tRS_rdV_cur, tRS_sdV[None, None, None, stage])
                        cute.arch.fence_view_async_shared()
                        self.epi_barrier.arrive_and_wait()

                        tSR_rdV = cute.make_rmem_tensor(tdVrdV_out_shape, dtype=self.dtype_acc)

                        for w in cutlass.range_constexpr(2):
                            stage_out = 4 * split + 2 * w + (i % 2)
                            cute.copy(
                                tiled_copy_s2r,
                                tSR_sdV[None, None, None, stage_out],
                                tSR_rdV[None, None, None, w],
                            )

                        for j in cutlass.range_constexpr(gmem_rows_per_thread):
                            gmem_n_idx = rIdxTopK[j]
                            for w in cutlass.range_constexpr(2):
                                dv_offset = (
                                    self.hdimv // self.num_hdimv_splits * split  # 256 * split
                                    + (self.hdimv // self.num_hdimv_splits // 2) * w  # 128 * w
                                    + 32 * i
                                )
                                dv_offset += tdVcdV[0, j, 0][1]
                                gmem_coord = (gmem_n_idx, dv_offset)
                                dV_gmem_ptr = elem_pointer(mdV_cur, gmem_coord)

                                a = tSR_rdV[0, j, 0, w]
                                b = tSR_rdV[1, j, 0, w]
                                c = tSR_rdV[2, j, 0, w]
                                d = tSR_rdV[3, j, 0, w]
                                atomic_add_fp32x4(a, b, c, d, dV_gmem_ptr)

                    cute.arch.fence_view_async_tmem_load()
                    self.epi_barrier.arrive_and_wait()
                    pipeline_dV.consumer_release(consumer_state_dV)

                    if leader_warp:
                        with cute.arch.elect_one():
                            pipeline_dV_epi.consumer_release(consumer_state_dV)

                    consumer_state_dV.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
