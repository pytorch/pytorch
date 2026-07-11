# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

import math
from typing import Tuple, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32, Int64, Float32

from cutlass.utils import ClcDynamicPersistentTileScheduler
from .tile_scheduler import (
    ClcState,
    compute_sm100_fmha_grid as compute_grid,
    compute_sm100_fmha_grid_clc as compute_grid_clc,
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
    Sm100FmhaStaticTileScheduler as FmhaStaticTileScheduler,
    Sm100FmhaStaticTileSchedulerParams as FmhaStaticTileSchedulerParams,
    Sm100FmhaClcDynamicTileScheduler as FmhaClcDynamicTileScheduler,
    Sm100FmhaClcDynamicTileSchedulerParams as FmhaClcDynamicTileSchedulerParams,
)
from .mask import (
    Sm100FusedMask as FusedMask,
)
from .tile_scheduler import SM100_TMEM_CAPACITY_COLUMNS
from .flash_fwd_sm100 import DescaleTensors, _TUNING_CONFIG
from .utils import ex2_emulation_2, as_bshkrd_tensor, AuxData


class BlackwellFusedMultiHeadAttentionForward:
    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int = 1,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: int = 2,
        is_persistent: bool = True,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors: bool = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        use_2cta_instrs: bool = False,
        use_clc_scheduler: bool = False,
    ):
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        assert head_dim == 256 and head_dim_v == 256, (
            "SM100 dedicated kernel only supports (head_dim, head_dim_v) = (256, 256)"
        )
        assert score_mod is None, "SM100 forward with head_dim=256 does not support score_mod"
        assert mask_mod is None, "SM100 forward with head_dim=256 does not support mask_mod"
        assert not has_aux_tensors, "SM100 forward with head_dim=256 does not support aux tensors"
        assert not paged_kv_non_tma, (
            "SM100 hd256 2CTA supports TMA paged KV only (page_size must equal tile_n=128)"
        )
        assert not pack_gqa, "SM100 forward with head_dim=256 does not support pack_gqa"
        assert not is_split_kv, "SM100 forward with head_dim=256 does not support SplitKV"
        assert q_subtile_factor == 1, (
            "SM100 forward with head_dim=256 does not support q_subtile_factor"
        )
        assert m_block_size == 128 and n_block_size == 128, (
            "SM100 dedicated kernel only supports tile_m=128 and tile_n=128"
        )
        # q_stage / persistence / scheduler knobs are accepted for interface parity,
        # but this dedicated kernel uses fixed internal settings.

        qk_acc_dtype = cutlass.Float32
        pv_acc_dtype = cutlass.Float32
        mma_tiler = (128, 128, head_dim)
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.qhead_per_kvhead = qhead_per_kvhead
        self.mma_tiler = mma_tiler
        assert mma_tiler[0] == 128 and mma_tiler[1] == 128, "Only 128x128 tile impl is supported"
        assert mma_tiler[2] == 256, "Only 256 is supported for 128x128 tile impl"
        self.cta_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.qk_mma_tiler = (
            2 * mma_tiler[0],
            mma_tiler[1],
            min(self.cta_tiler[2], 128),
        )
        self.pv_mma_tiler = self.qk_mma_tiler
        self.pv_block_tiler = (
            self.pv_mma_tiler[0] // 2,
            self.pv_mma_tiler[1],
            self.pv_mma_tiler[2],
        )
        self.iterations_qk = self.cta_tiler[2] // self.qk_mma_tiler[2]
        self.iterations_pv = self.cta_tiler[2] // self.pv_mma_tiler[1]
        self.cluster_shape_mn = (2, 1)
        self.tmem_warp_shape_mn = (4, 1)
        # Dedicated hd256 kernel uses fixed scheduling policy.
        self.is_persistent = False
        self.is_causal = is_causal
        self.is_local = is_local
        self.use_semantic_trip_range = is_causal or is_local
        self.use_clc_scheduler = False

        self.softmax_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.empty_warp_id = (10, 11)
        self.sched_warp_id = self.empty_warp_id[0] if use_clc_scheduler else None
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax_warp_ids,  # this is to get a round num threads
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                *self.empty_warp_id,
            )
        )

        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )

        self.tmem_s_offset = 0
        self.tmem_o_offset = 256
        self.tmem_p_offset = self.tmem_s_offset

        _tune_key = (True, is_causal, 256, False)  # hd256: always 2cta, no sm103 variant
        _tune = _TUNING_CONFIG.get(_tune_key, {})
        self.num_regs_softmax = _tune.get("num_regs_softmax", 256)
        self.num_regs_correction = _tune.get("num_regs_correction", 160)
        self.num_regs_other = 32  # fixed for hd256; not derived from 512 budget like other kernels
        self.ex2_emu_freq = _tune.get("ex2_emu_freq", 4)
        self.ex2_emu_res = _tune.get("ex2_emu_res", 3)
        self.ex2_emu_start_frg = _tune.get("ex2_emu_start_frg", 0)

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.q_stage = self.iterations_qk
        self.kv_stage = 4
        self.qk_acc_stage = 2
        self.mma_corr_stage = 1
        if cutlass.const_expr(self.use_clc_scheduler):
            self.num_clc_stage = 1
            self.num_clc_response_bytes = 16

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        descale_tensors: Optional[DescaleTensors] = None,
        blocksparse_tensors: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
        stream: cuda.CUstream = None,
    ):
        # Keep parity with FlashAttentionForwardSm100.__call__ interface.
        # (TODO@wangsiyu) Implement these features.
        assert mSeqUsedQ is None and mSeqUsedK is None, (
            "SM100 forward with head_dim=256 does not support seqused_q/seqused_k"
        )
        assert learnable_sink is None, (
            "SM100 forward with head_dim=256 does not support learnable_sink"
        )
        assert blocksparse_tensors is None, (
            "SM100 forward with head_dim=256 does not support block sparsity"
        )
        assert aux_data.tensors is None, (
            "SM100 forward with head_dim=256 does not support aux_tensors"
        )
        assert aux_data.scalars is None, (
            "SM100 forward with head_dim=256 does not support aux_scalars"
        )
        assert not self.is_local, (
            "SM100 forward with head_dim=256 does not support local attention yet"
        )
        assert window_size_left is None and window_size_right is None, (
            "SM100 forward with head_dim=256 does not support runtime window_size overrides"
        )
        assert descale_tensors is None, (
            "SM100 forward with head_dim=256 does not support descale_tensors"
        )

        q_tensor, k_tensor, v_tensor, o_tensor = mQ, mK, mV, mO
        lse_tensor = mLSE
        cum_seqlen_q = mCuSeqlensQ
        cum_seqlen_k = mCuSeqlensK

        q_rank = len(mQ.shape)
        k_rank = len(mK.shape)
        if cutlass.const_expr(cum_seqlen_q is not None):
            # Varlen path accepts either legacy 5D tensors or standard 3D tensors.
            if cutlass.const_expr(q_rank == 5):
                s_q = mQ.shape[1]
                h_q = mQ.shape[2] * mQ.shape[3]
                d = mQ.shape[4]
            elif cutlass.const_expr(q_rank == 3):
                s_q = mQ.shape[0]
                h_q = mQ.shape[1]
                d = mQ.shape[2]
            else:
                raise RuntimeError(f"hd256 forward varlen expects q rank 3 or 5, got rank {q_rank}")
        else:
            # Non-varlen path accepts either legacy 5D tensors or standard 4D tensors.
            if cutlass.const_expr(q_rank == 5):
                s_q = mQ.shape[1]
                h_q = mQ.shape[2] * mQ.shape[3]
                d = mQ.shape[4]
            elif cutlass.const_expr(q_rank == 4):
                s_q = mQ.shape[1]
                h_q = mQ.shape[2]
                d = mQ.shape[3]
            else:
                raise RuntimeError(
                    f"hd256 forward non-varlen expects q rank 4 or 5, got rank {q_rank}"
                )

        if cutlass.const_expr(cum_seqlen_k is not None):
            if cutlass.const_expr(k_rank == 5):
                s_k = mK.shape[1]
                h_k = mK.shape[2]
            elif cutlass.const_expr(k_rank == 3):
                s_k = mK.shape[0]
                h_k = mK.shape[1]
            else:
                raise RuntimeError(f"hd256 forward varlen expects k rank 3 or 5, got rank {k_rank}")
        else:
            if cutlass.const_expr(k_rank == 5):
                s_k = mK.shape[1]
                h_k = mK.shape[2]
            elif cutlass.const_expr(k_rank == 4):
                s_k = mK.shape[1]
                h_k = mK.shape[2]
            else:
                raise RuntimeError(
                    f"hd256 forward non-varlen expects k rank 4 or 5, got rank {k_rank}"
                )
        if cutlass.const_expr(cum_seqlen_q is not None):
            b = mCuSeqlensQ.shape[0] - 1
        elif cutlass.const_expr(cum_seqlen_k is not None):
            b = mCuSeqlensK.shape[0] - 1
        else:
            b = mQ.shape[0]

        scale_softmax = softmax_scale
        scale_softmax_log2 = softmax_scale * math.log2(math.exp(1.0))
        scale_output = 1.0
        s_lse = s_q
        h_r = h_q // h_k
        s_q64 = Int64(s_q)
        s_k64 = Int64(s_k)
        s_lse64 = Int64(s_lse)
        h_r64 = Int64(h_r)
        h_k64 = Int64(h_k)
        b64 = Int64(b)
        s_q_total = (
            q_tensor.shape[1]
            if cum_seqlen_q is not None and q_rank == 5
            else (q_tensor.shape[0] if cum_seqlen_q is not None else s_q64)
        )
        s_k_total = (
            k_tensor.shape[1]
            if cum_seqlen_k is not None and k_rank == 5
            else (k_tensor.shape[0] if cum_seqlen_k is not None else s_k64)
        )
        b_lse = b64 if cum_seqlen_q is None else 1
        stride_b_lse = h_r64 * h_k64 * s_lse64 if cum_seqlen_q is None else 0

        varlen_q = cum_seqlen_q is not None
        varlen_k = cum_seqlen_k is not None
        q_norm = as_bshkrd_tensor(q_tensor, h_k, h_r, varlen_q)
        o_norm = as_bshkrd_tensor(o_tensor, h_k, h_r, varlen_q)

        # Forward layout: (s, d, ((h_r, h_k), b)). Stride picks from canonical
        # positions 1=S, 4=D, 3=H_r, 2=H_k, 0=B.
        q = cute.make_tensor(
            q_norm.iterator,
            cute.make_layout(
                (s_q_total, d, ((h_r, h_k), b)),
                stride=(
                    q_norm.stride[1],
                    q_norm.stride[4],
                    ((q_norm.stride[3], q_norm.stride[2]), q_norm.stride[0]),
                ),
            ),
        )
        if cutlass.const_expr(mPageTable is not None):
            # Paged: input k/v are rank-4 (num_pages, page_size, h_k, d); the kernel
            # consumes K as (page_size, d, h_k, num_pages) and V as
            # (d, page_size, h_k, num_pages).
            # cute.select reorders modes while preserving input strides
            page_size = k_tensor.shape[1]
            max_seqlen_k_paged = Int32(mPageTable.shape[1] * page_size)
            k = cute.make_tensor(k_tensor.iterator, cute.select(k_tensor.layout, mode=[1, 3, 2, 0]))
            v = cute.make_tensor(v_tensor.iterator, cute.select(v_tensor.layout, mode=[3, 1, 2, 0]))
            page_table = cute.make_tensor(
                mPageTable.iterator,
                cute.make_layout(
                    (b, mPageTable.shape[1]),
                    stride=(mPageTable.stride[0], mPageTable.stride[1]),
                ),
            )
        else:
            # K/V have no h_r dim; pass h_r=1 to the normalizer and override the
            # h_r stride to 0 below to broadcast across the query-grouped heads.
            k_norm = as_bshkrd_tensor(k_tensor, h_k, 1, varlen_k)
            v_norm = as_bshkrd_tensor(v_tensor, h_k, 1, varlen_k)
            # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
            k = cute.make_tensor(
                k_norm.iterator,
                cute.make_layout(
                    (s_k_total, d, ((h_r, h_k), b)),
                    stride=(
                        k_norm.stride[1],
                        k_norm.stride[4],
                        ((0, k_norm.stride[2]), k_norm.stride[0]),
                    ),
                ),
            )
            # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
            v = cute.make_tensor(
                v_norm.iterator,
                cute.make_layout(
                    (d, s_k_total, ((h_r, h_k), b)),
                    stride=(
                        v_norm.stride[4],
                        v_norm.stride[1],
                        ((0, v_norm.stride[2]), v_norm.stride[0]),
                    ),
                ),
            )
            page_table = None
            max_seqlen_k_paged = None
        # (s, d, ((h_r, h_k), b))
        o = cute.make_tensor(
            o_norm.iterator,
            cute.make_layout(
                (s_q_total, d, ((h_r, h_k), b)),
                stride=(
                    o_norm.stride[1],
                    o_norm.stride[4],
                    ((o_norm.stride[3], o_norm.stride[2]), o_norm.stride[0]),
                ),
            ),
        )
        if cutlass.const_expr(lse_tensor is not None):
            # (s, ((h_r, h_k), b))
            lse_layout = cute.make_layout(
                (s_lse64, ((h_r, h_k), b_lse)),
                stride=(1, ((s_lse64, h_r64 * s_lse64), stride_b_lse)),
            )
            lse = cute.make_tensor(lse_tensor.iterator, lse_layout)
        else:
            lse = None

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type
        self.tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.q_dtype.width

        if cutlass.const_expr(self.use_clc_scheduler):
            self.tile_sched_params, grid = compute_grid_clc(
                (s_q, o.shape[1], o.shape[2]) if cum_seqlen_q is not None else o.shape,
                self.cta_tiler,
                (*self.cluster_shape_mn, 1),
            )
        else:
            self.tile_sched_params, grid = compute_grid(
                (s_q, o.shape[1], o.shape[2]) if cum_seqlen_q is not None else o.shape,
                self.cta_tiler,
                self.is_persistent,
            )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")

        # check type consistency
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        # the intermediate tensor p is from tmem & k-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.pv_block_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.kv_stage,
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.qk_acc_stage,
        )
        p_tmem_layout = cute.select(p_tmem_layout_staged, mode=[0, 1, 2])
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.v_dtype,
            self.kv_stage,
        )
        # TMA load for Q
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        self.tma_copy_q_bytes = q_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_kv_bytes = k_copy_size * cute.size(qk_tiled_mma.thr_id.shape)

        @cute.struct
        class SharedStorage:
            # TMA G2S load barriers: LOAD warp (producer) -> MMA warp (consumer)
            load_q_mbar_ptr: cute.struct.MemRange[
                Int64, self.q_stage * 2
            ]  # load_q_{producer,consumer}
            load_kv_mbar_ptr: cute.struct.MemRange[
                Int64, self.kv_stage * 2
            ]  # load_kv_{producer,consumer}
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            # Softmax -> Correction signaling barriers (row_max/row_sum vec ready)
            s_corr_mbar_ptr: cute.struct.MemRange[
                Int64, self.qk_acc_stage * 2
            ]  # s_corr_{producer,consumer}
            sum_mbar_ptr: cute.struct.MemRange[Int64, 2]
            # MMA -> Correction ownership barriers for O_partial tokens (online rescale/finalize)
            mma_corr_mbar_ptr: cute.struct.MemRange[
                Int64, self.mma_corr_stage * 2
            ]  # mma_corr_{producer,consumer}
            # A CTA-wide "TMEM lifetime" barrier used to safely deallocate TMEM after all users finish.
            tmem_dealloc_mbar: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # CLC pipeline barriers and response buffer
            clc_mbar_ptr: cute.struct.MemRange[Int64, 2]
            clc_response: cute.struct.MemRange[Int32, 4]

        self.shared_storage = SharedStorage

        grid = cute.round_up(grid, self.cluster_shape_mnk)
        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            o,
            cum_seqlen_q,
            cum_seqlen_k,
            lse,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            page_table,
            max_seqlen_k_paged,
            window_size_left,
            window_size_right,
            self.cluster_layout_vmnk,
            q_smem_layout_staged,
            k_smem_layout_staged,
            p_tmem_layout,
            v_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        mO_qdl: cute.Tensor,
        cum_seqlen_q: Optional[cute.Tensor],
        cum_seqlen_k: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        mPageTable: Optional[cute.Tensor],
        max_seqlen_k: Optional[Int32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: FmhaStaticTileSchedulerParams | FmhaClcDynamicTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        #
        # Prefetch tma desc
        #
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        p_mma_producer, p_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        s_corr_producer, s_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        sum_producer, sum_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.sum_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_corr_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.correction_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.correction_warp_ids[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        tmem.allocate(self.tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
        # Initialize CLC state if using dynamic scheduler
        if cutlass.const_expr(self.use_clc_scheduler):
            clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            cluster_size = cute.size(self.cluster_shape_mnk)
            num_clc_consumer_threads = self.threads_per_warp * (
                1  # sched_warp (CTA 0 only)
                + cluster_size
                * (
                    len(self.softmax_warp_ids)
                    + len(self.correction_warp_ids)
                    + 1  # mma_warp
                    + 1  # load_warp
                )
            )
            clc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_clc_consumer_threads
            )
            clc_response_ptr = storage.clc_response.data_ptr()
            clc = ClcState.create(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_sched_params.clc_hw_params(),
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=storage.clc_mbar_ptr.data_ptr(),
                    num_stages=self.num_clc_stage,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=self.num_clc_response_bytes,
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                ),
                consumer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_clc_stage
                ),
                producer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_clc_stage
                ),
            )
        else:
            clc = None
            clc_response_ptr = None

        # Cluster arrive after barrier init
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        # K and V now use separate memory since we removed the transform stage
        sV = smem.allocate_tensor(
            element_type=self.v_dtype,
            layout=v_smem_layout_staged.outer,
            swizzle=v_smem_layout_staged.inner,
            byte_alignment=128,
        )

        sSum = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp),
            byte_alignment=128,
        )
        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        pv_thr_mma = pv_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tStS = qk_thr_mma.make_fragment_C(cute.append(qk_acc_shape, self.qk_acc_stage))
        pv_acc_shape = pv_thr_mma.partition_shape_C((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                self.iterations_pv,
                stride=self.pv_mma_tiler[1] // self.tmem_warp_shape_mn[1],
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tOtO_staged = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO_layout)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        for _i in cutlass.range_constexpr(len(self.empty_warp_id)):
            if warp_idx == self.empty_warp_id[_i]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        if cutlass.const_expr(self.use_clc_scheduler):
            tile_sched = FmhaClcDynamicTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
                clc,
            )
        else:
            blk_idx = cute.arch.block_idx()
            tile_sched = FmhaStaticTileScheduler(
                tile_sched_params, blk_idx[0], blk_idx, cute.arch.grid_dim()
            )
        work_tile = tile_sched.initial_work_tile_info()

        # Cluster wait
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx  # (q_tile_idx, 0, (head_idx, batch_idx))
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                continue_cond = False
                batch_coord = curr_block_coord[2][1]
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                cuseqlen_q = Int32(0)
                cuseqlen_k = Int32(0)
                block_offset = (
                    Int32(0),
                    Int32(0),
                    Int32(0),
                    ((Int32(0), Int32(0)), Int32(0)),
                )
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    block_offset = (
                        cuseqlen_q,
                        cuseqlen_k,
                        Int32(0),
                        ((Int32(0), Int32(0)), Int32(0)),
                    )
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )
                if not continue_cond:
                    mQ_qdl_ = cute.domain_offset(cute.select(block_offset, mode=[0, 2, 3]), mQ_qdl)
                    # Local tile partition global tensors
                    q_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
                    )
                    # (bM, bK, loopM, loopK, loopL)
                    gQ_qdl = cute.flat_divide(mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2]))
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        block_in_cluster_coord_vmnk[2],
                        q_cta_layout,
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    kv_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                    )
                    if cutlass.const_expr(mPageTable is None):
                        # Dense path: domain_offset K/V by batch block, select batch via mma_block_coord[2].
                        mK_kdl_ = cute.domain_offset(
                            cute.select(block_offset, mode=[1, 2, 3]), mK_kdl
                        )
                        mV_dkl_ = cute.domain_offset(
                            cute.select(block_offset, mode=[2, 1, 3]), mV_dkl
                        )
                        gK_kdl = cute.flat_divide(
                            mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
                        )
                        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_k,
                            block_in_cluster_coord_vmnk[1],
                            kv_cta_layout,
                            cute.group_modes(sK, 0, 3),
                            cute.group_modes(tSgK_kdl, 0, 3),
                        )
                        gV_dkl = cute.flat_divide(
                            mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
                        )
                        tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_v,
                            block_in_cluster_coord_vmnk[1],
                            kv_cta_layout,
                            cute.group_modes(sV, 0, 3),
                            cute.group_modes(tSgV_dkl, 0, 3),
                        )
                        # ((atom_v, rest_v), RestN, RestK)
                        tKgK = tKgK_kdl[None, None, None, mma_block_coord[2]]
                        tVgV = tVgV_dkl[None, None, None, mma_block_coord[2]]
                    else:
                        # Paged path: slice K/V by KV head, keep num_pages dim for page_idx-based TMA.
                        head_kv_coord = curr_block_coord[2][0] // self.qhead_per_kvhead
                        mK_kdl_ = mK_kdl[None, None, head_kv_coord, None]
                        mV_dkl_ = mV_dkl[None, None, head_kv_coord, None]
                        gK_kdl = cute.flat_divide(
                            mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
                        )
                        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_k,
                            block_in_cluster_coord_vmnk[1],
                            kv_cta_layout,
                            cute.group_modes(sK, 0, 3),
                            cute.group_modes(tSgK_kdl, 0, 3),
                        )
                        gV_dkl = cute.flat_divide(
                            mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
                        )
                        tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_v,
                            block_in_cluster_coord_vmnk[1],
                            kv_cta_layout,
                            cute.group_modes(sV, 0, 3),
                            cute.group_modes(tSgV_dkl, 0, 3),
                        )
                        tKgK = tKgK_kdl
                        tVgV = tVgV_dkl
                    # ((atom_v, rest_v), RestK)
                    tQgQ = tQgQ_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]

                    seqlen_kv_loop_start, seqlen_kv_loop_steps = (
                        FusedMask.get_trip_start_count_via_block_info(
                            mma_block_coord,
                            self.qk_mma_tiler,
                            seqlen_q,
                            seqlen_k,
                            self.is_causal,
                            self.is_local,
                            window_size_left,
                            window_size_right,
                        )
                    )
                    seqlen_kv_loop_end = seqlen_kv_loop_start + seqlen_kv_loop_steps
                    # Q
                    for iter in cutlass.range(self.iterations_qk, unroll=1):
                        q_handle = load_q_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_q,
                            tQgQ[None, iter],
                            tQsQ[None, q_handle.index],
                            tma_bar_ptr=q_handle.barrier,
                        )

                    # K0
                    kv_coord = seqlen_kv_loop_start
                    k_page_idx = (
                        mPageTable[batch_coord, kv_coord]
                        if cutlass.const_expr(mPageTable is not None)
                        else None
                    )
                    for iter in cutlass.range(self.iterations_qk, unroll=1):
                        k_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, kv_coord, iter]
                            if cutlass.const_expr(mPageTable is None)
                            else tKgK[None, 0, iter, k_page_idx],
                            tKsK[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                    kv_coord += 1
                    # v_page_idx_prev carries K[i-1]'s page index for use as V[i-1]'s page
                    # (K and V for the same KV block share the same physical page).
                    # Also serves as the Vend page index when seqlen_kv_loop_steps == 1.
                    v_page_idx_prev = (
                        k_page_idx if cutlass.const_expr(mPageTable is not None) else None
                    )
                    # Prefetch K1 page after K0 TMA dispatch to hide L2 latency.
                    if cutlass.const_expr(mPageTable is not None):
                        if seqlen_kv_loop_steps > 1:
                            k_page_idx = mPageTable[batch_coord, kv_coord]

                    for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # Ki: k_page_idx was prefetched at end of previous iteration
                        # (or in the prologue for i==1); L2 latency already hidden.
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            k_handle = load_kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_k,
                                tKgK[None, kv_coord, iter]
                                if cutlass.const_expr(mPageTable is None)
                                else tKgK[None, 0, iter, k_page_idx],
                                tKsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )
                        # Vi-1: reuse v_page_idx_prev (= K[i-1]'s page), no extra GMEM read.
                        for iter in cutlass.range(self.iterations_pv, unroll=1):
                            v_handle = load_kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v,
                                tVgV[None, iter, kv_coord - 1]
                                if cutlass.const_expr(mPageTable is None)
                                else tVgV[None, iter, 0, v_page_idx_prev],
                                tVsV[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                            )
                        v_page_idx_prev = (
                            k_page_idx if cutlass.const_expr(mPageTable is not None) else None
                        )
                        kv_coord += 1
                        # Prefetch next K page while V TMA is in flight.
                        if cutlass.const_expr(mPageTable is not None):
                            if kv_coord < seqlen_kv_loop_end:
                                k_page_idx = mPageTable[batch_coord, kv_coord]
                    # Vend: reuse v_page_idx_prev (= K[end-1]'s page), no extra GMEM read.
                    for iter in cutlass.range(self.iterations_pv, unroll=1):
                        v_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, iter, seqlen_kv_loop_end - 1]
                            if cutlass.const_expr(mPageTable is None)
                            else tVgV[None, iter, 0, v_page_idx_prev],
                            tVsV[None, v_handle.index],
                            tma_bar_ptr=v_handle.barrier,
                        )

                work_tile = tile_sched.advance_to_next_work()
                # End of persistent scheduler loop
            load_kv_producer.tail()
            load_q_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            is_leader_cta = cta_rank_in_cluster % 2 == 0

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                continue_cond = False
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                batch_coord = curr_block_coord[2][1]
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )

                if not continue_cond:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                    seqlen_kv_loop_start, seqlen_kv_loop_steps = (
                        FusedMask.get_trip_start_count_via_block_info(
                            mma_block_coord,
                            self.qk_mma_tiler,
                            seqlen_q,
                            seqlen_k,
                            self.is_causal,
                            self.is_local,
                            window_size_left,
                            window_size_right,
                        )
                    )
                    seqlen_kv_loop_end = seqlen_kv_loop_start + seqlen_kv_loop_steps

                    load_q_releaser = load_q_consumer.clone()
                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    if seqlen_kv_loop_steps > 1:
                        # QK0
                        if is_leader_cta:
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                load_q_consumer.wait_and_advance()
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_kv_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                            s_handle.commit()
                        for i in cutlass.range(1, seqlen_kv_loop_steps - 1, 1, unroll=1):
                            # QKi
                            if is_leader_cta:
                                s_handle = mma_s_producer.acquire_and_advance()
                                tStS_slice = tStS[None, None, None, s_handle.index]
                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                for iter in cutlass.range(self.iterations_qk, unroll=1):
                                    tSrQ_slice = tSrQ[None, None, None, iter]
                                    k_handle = load_kv_consumer.wait_and_advance()
                                    tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                    num_kphases = cute.size(tSrQ_slice, mode=[2])
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    k_handle.release()
                                s_handle.commit()

                                # PVi-1
                                p_handle = p_mma_consumer.wait_and_advance()
                                o_handle = mma_corr_producer.acquire_and_advance()
                                pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                                for iter in cutlass.range(self.iterations_pv, unroll=1):
                                    v_handle = load_kv_consumer.wait_and_advance()
                                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                                    tOtO_slice = tOtO_staged[None, None, None, iter]
                                    tStS_slice = tStS[None, None, None, p_handle.index]
                                    tP = cute.make_tensor(
                                        tStS_slice.iterator, p_tmem_layout_staged.outer
                                    )
                                    tOrP = pv_thr_mma.make_fragment_A(tP)
                                    tOrP_slice = cute.make_tensor(
                                        cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                                        tOrP.layout,
                                    )
                                    tOrV_slice = tOrV[None, None, None, v_handle.index]
                                    num_kphases = cute.size(tOrV_slice, mode=[2])
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            pv_tiled_mma,
                                            tOtO_slice,
                                            tOrP_slice[kphase_coord],
                                            tOrV_slice[kphase_coord],
                                            tOtO_slice,
                                        )
                                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    v_handle.release()
                                o_handle.commit()
                                p_handle.release()
                        if is_leader_cta:
                            # QKend
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_kv_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                                load_q_releaser.release()
                                load_q_releaser.advance()
                            s_handle.commit()

                            # PVend-1
                            p_handle = p_mma_consumer.wait_and_advance()
                            o_handle = mma_corr_producer.acquire_and_advance()
                            pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                            for iter in cutlass.range(self.iterations_pv, unroll=1):
                                v_handle = load_kv_consumer.wait_and_advance()
                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                                tOtO_slice = tOtO_staged[None, None, None, iter]
                                tStS_slice = tStS[None, None, None, p_handle.index]
                                tP = cute.make_tensor(
                                    tStS_slice.iterator, p_tmem_layout_staged.outer
                                )
                                tOrP = pv_thr_mma.make_fragment_A(tP)
                                tOrP_slice = cute.make_tensor(
                                    cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                                    tOrP.layout,
                                )
                                tOrV_slice = tOrV[None, None, None, v_handle.index]
                                num_kphases = cute.size(tOrV_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        pv_tiled_mma,
                                        tOtO_slice,
                                        tOrP_slice[kphase_coord],
                                        tOrV_slice[kphase_coord],
                                        tOtO_slice,
                                    )
                                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                v_handle.release()
                            o_handle.commit()
                            p_handle.release()
                    else:
                        if is_leader_cta:
                            # QK0
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                load_q_consumer.wait_and_advance()
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_kv_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                                load_q_releaser.release()
                                load_q_releaser.advance()
                            s_handle.commit()

                    if is_leader_cta:
                        # PVend
                        p_handle = p_mma_consumer.wait_and_advance()
                        o_handle = mma_corr_producer.acquire_and_advance()
                        pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                        for iter in cutlass.range(self.iterations_pv, unroll=1):
                            v_handle = load_kv_consumer.wait_and_advance()
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                            tOtO_slice = tOtO_staged[None, None, None, iter]
                            tStS_slice = tStS[None, None, None, p_handle.index]
                            tP = cute.make_tensor(tStS_slice.iterator, p_tmem_layout_staged.outer)
                            tOrP = pv_thr_mma.make_fragment_A(tP)
                            tOrP_slice = cute.make_tensor(
                                cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                                tOrP.layout,
                            )
                            tOrV_slice = tOrV[None, None, None, v_handle.index]
                            num_kphases = cute.size(tOrV_slice, mode=[2])
                            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                kphase_coord = (None, None, kphase_idx)
                                cute.gemm(
                                    pv_tiled_mma,
                                    tOtO_slice,
                                    tOrP_slice[kphase_coord],
                                    tOrV_slice[kphase_coord],
                                    tOtO_slice,
                                )
                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            v_handle.release()
                        o_handle.commit()
                        p_handle.release()
                work_tile = tile_sched.advance_to_next_work()
            # End of persistent scheduler loop
            mma_s_producer.tail()
            mma_corr_producer.tail()

        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax_warp_ids[0]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                cuseqlen_q = Int32(0)
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )
                if not continue_cond:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                    row_max = -Float32.inf
                    row_max_prev = -Float32.inf
                    row_sum = 0.0

                    start_count, trip_count = FusedMask.get_trip_start_count_via_block_info(
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        self.is_causal,
                        self.is_local,
                        window_size_left,
                        window_size_right,
                    )
                    end_count = start_count + trip_count
                    # require at least one softmax iteration for zero trip_count case;
                    # rely on masking this iteration for correctness
                    if end_count <= start_count:
                        start_count = 0
                        end_count = 1
                    if cutlass.const_expr(self.use_semantic_trip_range):
                        n_block_min_causal_local_mask, n_block_min_before_local_mask = (
                            FusedMask.get_trip_mask_bounds_via_block_info(
                                mma_block_coord,
                                self.qk_mma_tiler,
                                seqlen_q,
                                seqlen_k,
                                self.is_causal,
                                self.is_local,
                                window_size_left,
                                window_size_right,
                            )
                        )
                    cS_base = cute.make_identity_tensor(
                        (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
                    )
                    cS = cute.domain_offset((mma_block_coord[0] * self.qk_mma_tiler[0], 0), cS_base)
                    tScS = qk_thr_mma.partition_C(cS)

                    for step in cutlass.range(start_count, end_count, 1, unroll=1):
                        cS_iter = cute.domain_offset((0, step * self.qk_mma_tiler[1]), cS)
                        tScS_iter = qk_thr_mma.partition_C(cS_iter)
                        if cutlass.const_expr(self.use_semantic_trip_range):
                            need_apply_mask = (
                                step >= n_block_min_causal_local_mask
                                or step < n_block_min_before_local_mask
                                or step == end_count - 1
                            )
                        else:
                            # Residual path only needs seqlen masking on the last K tile.
                            need_apply_mask = step == end_count - 1
                        # Si -> Pi
                        (
                            row_max,
                            row_sum,
                            mma_s_consumer,
                            p_mma_producer,
                            s_corr_producer,
                        ) = self.softmax_step(
                            (need_apply_mask, window_size_left, window_size_right),
                            (
                                row_max_prev,
                                row_sum,
                                seqlen_q,
                                seqlen_k,
                                scale_softmax_log2,
                            ),
                            (tStS, tScS_iter),
                            (mma_s_consumer, p_mma_producer, s_corr_producer),
                        )
                        row_max_prev = row_max
                    sum_producer = self.store_sum_max(
                        row_max,
                        mLSE,
                        row_sum,
                        sSum,
                        sum_producer,
                        curr_block_coord,
                        seqlen_q,
                        cum_seqlen_q,
                        cuseqlen_q,
                        scale_softmax,
                    )
                work_tile = tile_sched.advance_to_next_work()
            p_mma_producer.tail()
            s_corr_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                continue_cond = False
                cuseqlen_q = Int32(0)
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )

                if not continue_cond:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                    mO_qdl_eff = mO_qdl
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        block_offset_o = (
                            cuseqlen_q,
                            Int32(0),
                            Int32(0),
                            ((Int32(0), Int32(0)), Int32(0)),
                        )
                        mO_qdl_eff = cute.domain_offset(
                            cute.select(block_offset_o, mode=[0, 2, 3]), mO_qdl
                        )

                    # (bM, bN, loopM, loopN, loopL)
                    gO_qdl = cute.flat_divide(
                        mO_qdl_eff, cute.select(self.pv_block_tiler, mode=[0, 1])
                    )
                    cO_qdl = cute.flat_divide(
                        cute.make_identity_tensor(mO_qdl_eff.shape),
                        cute.select(self.pv_block_tiler, mode=[0, 1]),
                    )

                    _, seqlen_kv_loop_steps = FusedMask.get_trip_start_count_via_block_info(
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        self.is_causal,
                        self.is_local,
                        window_size_left,
                        window_size_right,
                    )
                    gO_staged = gO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                    cO_staged = cO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                    cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
                    tScS = qk_thr_mma.partition_C(cS)

                    # Empty step as the first step is no need for correction
                    stats_handle = s_corr_consumer.wait_and_advance()
                    stats_handle.release()
                    for step in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # Oi-1 -> Oi
                        mma_corr_consumer, s_corr_consumer = self.correction_rescale(
                            scale_softmax_log2,
                            (s_corr_consumer, tStS, tScS),
                            (mma_corr_consumer, tOtO_staged, cO_staged),
                            self.epi_tile,
                        )
                    # O_partial -> O_final
                    mma_corr_consumer, sum_consumer = self.correction_epilog(
                        (seqlen_q, scale_output),
                        (sum_consumer, sSum),
                        (mma_corr_consumer, gO_staged, cO_staged, tOtO_staged),
                        self.epi_tile,
                    )
                work_tile = tile_sched.advance_to_next_work()
            # NOTE: tmem.free() moved to kernel end to enable cluster-wide sync

        # ///////////////////////////////////////////////////////////////////////////////
        #  Scheduler Warp (only for CLC dynamic scheduler)
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.use_clc_scheduler):
            is_first_cta_in_cluster = cta_rank_in_cluster == 0

            if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
                while work_tile.is_valid_tile:
                    tile_sched.prefetch_next_work()
                    work_tile = tile_sched.advance_to_next_work()
                tile_sched.producer_tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Empty warps reg dealloc
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.use_clc_scheduler):
            if warp_idx > self.load_warp_id:
                if not (warp_idx == self.sched_warp_id and is_first_cta_in_cluster):
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
        else:
            if warp_idx > self.load_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Cooperative TMEM Deallocation (2CTA)
        # ///////////////////////////////////////////////////////////////////////////////
        # All warps (including scheduler) have finished by this point.
        # Cluster-wide sync ensures both CTAs reach here before dealloc.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

        return

    @cute.jit
    def softmax_step(
        self,
        mask_args: Tuple,
        value_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
    ) -> Tuple[Float32, Float32, pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        need_apply_mask, window_size_left, window_size_right = mask_args
        row_max, row_sum, seqlen_q, seqlen_k, scale_softmax_log2 = value_args
        tStS, tScS = tensor_args
        mma_s_consumer, p_mma_producer, s_corr_producer = pipeline_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        s_handle = mma_s_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
        )
        tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
        tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)

        cute.arch.fence_view_async_tmem_load()
        s_handle.release()
        if need_apply_mask:
            FusedMask.apply_mask_via_causal_local(
                tTMEM_LOADrS,
                tTMEM_LOADcS,
                seqlen_q,
                seqlen_k,
                self.use_semantic_trip_range,
                self.is_causal,
                self.is_local,
                window_size_left,
                window_size_right,
            )
        old_row_max = row_max
        row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        row_max_safe = row_max
        if row_max == -cutlass.Float32.inf:
            row_max_safe = 0.0

        stats_handle = s_corr_producer.acquire_and_advance()
        stats_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 2))
        )
        stats_c_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 2))
        )
        tOtStats = cute.make_tensor(tStS_slice.iterator + self.tilePlikeFP32, stats_layout)
        tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
        tmem_store_stats_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store_stats = tcgen05.make_tmem_copy(tmem_store_stats_atom, tOtStats)
        thr_tmem_store_stats = tiled_tmem_store_stats.get_slice(thread_idx)
        tTMEM_STOREcStats = thr_tmem_store_stats.partition_S(tOcStats)
        tTMEM_STORErStats = cute.make_rmem_tensor(tTMEM_STOREcStats.shape, self.qk_acc_dtype)
        tTMEM_STORErStats[0] = old_row_max
        tTMEM_STORErStats[1] = row_max_safe
        tTMEM_STOREtStats = thr_tmem_store_stats.partition_D(tOtStats)
        cute.copy(tiled_tmem_store_stats, tTMEM_STORErStats, tTMEM_STOREtStats)
        cute.arch.fence_view_async_tmem_store()
        stats_handle.commit()

        scale = scale_softmax_log2
        minus_row_max_scale = (0.0 - row_max_safe) * scale
        # Acquire P write slot early — overlaps any pipeline stall with exp2 compute
        p_handle = p_mma_producer.acquire_and_advance()
        # Fragment-based FMA + exp2 + bf16 conversion
        # Trades SFU for FMA via polynomial emulation on a fraction of elements
        ex2_frg_tile = 32
        ex2_frg_cnt = cute.size(tTMEM_LOADrS) // ex2_frg_tile
        tTMEM_LOADrS_ex2 = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(ex2_frg_tile))
        tTMEM_STORErP = cute.make_rmem_tensor(tTMEM_LOADrS.shape, self.q_dtype)
        tTMEM_STORErP_ex2 = cute.logical_divide(tTMEM_STORErP, cute.make_layout(ex2_frg_tile))
        for j in cutlass.range_constexpr(ex2_frg_cnt):
            for k in cutlass.range_constexpr(0, ex2_frg_tile, 2):
                tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j] = cute.arch.fma_packed_f32x2(
                    (tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j]),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                )
                if cutlass.const_expr(self.ex2_emu_freq == 0):
                    tTMEM_LOADrS_ex2[k, j] = cute.math.exp2(tTMEM_LOADrS_ex2[k, j], fastmath=True)
                    tTMEM_LOADrS_ex2[k + 1, j] = cute.math.exp2(
                        tTMEM_LOADrS_ex2[k + 1, j], fastmath=True
                    )
                else:
                    if cutlass.const_expr(
                        k % self.ex2_emu_freq < self.ex2_emu_freq - self.ex2_emu_res
                        or j >= ex2_frg_cnt - 1
                        or j < self.ex2_emu_start_frg
                    ):
                        tTMEM_LOADrS_ex2[k, j] = cute.math.exp2(
                            tTMEM_LOADrS_ex2[k, j], fastmath=True
                        )
                        tTMEM_LOADrS_ex2[k + 1, j] = cute.math.exp2(
                            tTMEM_LOADrS_ex2[k + 1, j], fastmath=True
                        )
                    else:
                        tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j] = ex2_emulation_2(
                            tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j]
                        )
            tTMEM_STORErP_ex2[None, j].store(tTMEM_LOADrS_ex2[None, j].load().to(self.q_dtype))
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
        )
        tilePlikeFP32 = tStS_slice.shape[1] // Float32.width * self.q_dtype.width
        tStS_P_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], tilePlikeFP32))
        )
        tStS_P = cute.make_tensor(tStS_slice.iterator, tStS_P_layout)
        tScS_P_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], tilePlikeFP32))
        )
        tScS_P = cute.make_tensor(tScS_slice.iterator, tScS_P_layout)
        tmem_tiled_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_store = tmem_tiled_store.get_slice(thread_idx)
        tTMEM_STOREtP = thr_store.partition_D(tStS_P)
        tTMEM_STOREcS = thr_store.partition_S(tScS_P)
        tTMEM_STORErP_ = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErP.iterator, dtype=self.qk_acc_dtype),
            tTMEM_STOREcS.shape,
        )
        cute.copy(tmem_tiled_store, tTMEM_STORErP_, tTMEM_STOREtP)
        cute.arch.fence_view_async_tmem_store()

        p_handle.commit()
        acc_scale_ = scale * (old_row_max - row_max_safe)
        acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
        # TODO: calc row sum with TensorSSA
        row_sum *= acc_scale
        local_row_sum_0 = (row_sum, row_sum)
        local_row_sum_1 = (0.0, 0.0)
        local_row_sum_2 = (0.0, 0.0)
        local_row_sum_3 = (0.0, 0.0)
        reduction_unroll = 4
        frg_tile = cute.size(tTMEM_LOADrS) // reduction_unroll
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))
        for j in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS_frg, mode=[0]), 2):
            local_row_sum_0 = cute.arch.add_packed_f32x2(
                local_row_sum_0, (tTMEM_LOADrS_frg[j, 0], tTMEM_LOADrS_frg[j + 1, 0])
            )
            local_row_sum_1 = cute.arch.add_packed_f32x2(
                local_row_sum_1, (tTMEM_LOADrS_frg[j, 1], tTMEM_LOADrS_frg[j + 1, 1])
            )
            local_row_sum_2 = cute.arch.add_packed_f32x2(
                local_row_sum_2, (tTMEM_LOADrS_frg[j, 2], tTMEM_LOADrS_frg[j + 1, 2])
            )
            local_row_sum_3 = cute.arch.add_packed_f32x2(
                local_row_sum_3, (tTMEM_LOADrS_frg[j, 3], tTMEM_LOADrS_frg[j + 1, 3])
            )
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_1)
        local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, local_row_sum_3)
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_2)
        row_sum = local_row_sum_0[0] + local_row_sum_0[1]
        return row_max, row_sum, mma_s_consumer, p_mma_producer, s_corr_producer

    @cute.jit
    def correction_rescale(
        self,
        scale_softmax_log2: Float32,
        stats_args: tuple,
        o_args: tuple,
        epi_tile: cute.Tile,
    ) -> pipeline.PipelineConsumer:
        (s_corr_consumer, tStS, tScS) = stats_args
        (mma_o_consumer, tOtO_staged, cO_staged) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))

        stats_handle = s_corr_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, stats_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        stats_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 2))
        )
        stats_c_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 2))
        )
        tOtStats = cute.make_tensor(tStS_slice.iterator + self.tilePlikeFP32, stats_layout)
        tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
        tmem_load_stats_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_load_stats = tcgen05.make_tmem_copy(tmem_load_stats_atom, tOtStats)
        thr_tmem_load_stats = tiled_tmem_load_stats.get_slice(thread_idx)
        tTMEM_LOADtStats = thr_tmem_load_stats.partition_S(tOtStats)
        tTMEM_LOADcStats = thr_tmem_load_stats.partition_D(tOcStats)
        tTMEM_LOADrStats = cute.make_rmem_tensor(tTMEM_LOADcStats.shape, self.qk_acc_dtype)
        cute.copy(tiled_tmem_load_stats, tTMEM_LOADtStats, tTMEM_LOADrStats)

        scale = scale_softmax_log2 * (tTMEM_LOADrStats[0] - tTMEM_LOADrStats[1])
        scale = cute.math.exp2(scale, fastmath=True)
        stats_handle.release()
        o_handle = mma_o_consumer.wait_and_advance()
        for iter in cutlass.range(self.iterations_pv, unroll_full=True):
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            cO = cO_staged[None, None, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(16)),
                self.pv_acc_dtype,
            )
            tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_epi)
            thr_load = tmem_tiled_load.get_slice(thread_idx)
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(16)),
                self.pv_acc_dtype,
            )
            tmem_store_atom = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_epi)
            thr_store = tmem_store_atom.get_slice(thread_idx)
            tTMEM_LOADtO = thr_load.partition_S(tOtO_epi)
            tTMEM_LOADcO = thr_load.partition_D(cO_epi)
            tTMEM_STOREtO = thr_store.partition_D(tOtO_epi)
            tTMrO = cute.make_rmem_tensor_like(
                cute.append(
                    cute.make_layout(tTMEM_LOADcO[None, 0, 0].shape),
                    cute.make_layout(2, stride=cute.size(tTMEM_LOADcO[None, 0, 0].shape)),
                ),
                self.pv_acc_dtype,
            )
            tTMEM_LOADtO_0 = tTMEM_LOADtO[None, 0, 0]
            cute.copy(tmem_tiled_load, tTMEM_LOADtO_0, tTMrO[None, 0])
            iter_num = cute.size(tTMEM_LOADtO, mode=[1])
            for i in cutlass.range(1, iter_num, unroll_full=True):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                cute.copy(tmem_tiled_load, tTMEM_LOADtO_i, tTMrO[None, i % 2])
                for j in cutlass.range(0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True):
                    tTMrO[j, (i - 1) % 2], tTMrO[j + 1, (i - 1) % 2] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j, (i - 1) % 2], tTMrO[j + 1, (i - 1) % 2]),
                        (scale, scale),
                    )
                tTMEM_STOREtO_prev_i = tTMEM_STOREtO[None, i - 1, 0]
                cute.copy(tmem_store_atom, tTMrO[None, (i - 1) % 2], tTMEM_STOREtO_prev_i)

            for j in cutlass.range(0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True):
                tTMrO[j, (iter_num - 1) % 2], tTMrO[j + 1, (iter_num - 1) % 2] = (
                    cute.arch.mul_packed_f32x2(
                        (
                            tTMrO[j, (iter_num - 1) % 2],
                            tTMrO[j + 1, (iter_num - 1) % 2],
                        ),
                        (scale, scale),
                    )
                )
            cute.copy(
                tmem_store_atom,
                tTMrO[None, (iter_num - 1) % 2],
                tTMEM_STOREtO[None, iter_num - 1, 0],
            )
        cute.arch.fence_view_async_tmem_store()
        o_handle.release()
        return mma_o_consumer, s_corr_consumer

    @cute.jit
    def correction_epilog(
        self,
        value_args: Tuple,
        sum_args: Tuple,
        o_args: Tuple,
        epi_tile: cute.Tile,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        (seqlen_q, scale_output) = value_args
        (sum_consumer, sSum) = sum_args
        (mma_o_consumer, gO_staged, cO_staged, tOtO_staged) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_consumer.wait_and_advance()
        row_sum = sSum[thread_idx]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        row_sum_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
        scale = scale_output / row_sum if not row_sum_is_zero_or_nan else 0.0
        o_handle = mma_o_consumer.wait_and_advance()
        for iter in cutlass.range(self.iterations_pv):
            gO = gO_staged[None, None, iter]
            cO = cO_staged[None, None, iter]
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            gO_epi = cute.zipped_divide(gO, epi_tile)
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
            tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.pv_acc_dtype
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_epi)
            tTMEM_LOADgO = thr_tmem_load.partition_D(gO_epi)
            tTMEM_LOADcO = thr_tmem_load.partition_D(cO_epi)
            for i in cutlass.range(cute.size(tTMEM_LOADtO, mode=[1]), unroll_full=True):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                tTMEM_LOADgO_i = tTMEM_LOADgO[None, i, 0]
                tTMEM_LOADcO_i = tTMEM_LOADcO[None, i, 0]
                tTMrO = cute.make_rmem_tensor(tTMEM_LOADcO[None, 0, i].shape, self.pv_acc_dtype)
                cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
                for j in cutlass.range(0, cute.size(tTMrO), 2, unroll_full=True):
                    tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j], tTMrO[j + 1]),
                        (scale, scale),
                    )
                tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                o_vec = tTMrO.load()
                tSMrO.store(o_vec.to(self.o_dtype))
                if cute.elem_less(tTMEM_LOADcO_i[0][0], seqlen_q):
                    cute.autovec_copy(tSMrO, tTMEM_LOADgO_i)
        o_handle.release()
        return mma_o_consumer, sum_consumer

    @cute.jit
    def store_sum_max(
        self,
        row_max,
        mLSE,
        row_sum,
        sSum,
        sum_producer,
        current_block_coord,
        seqlen_q,
        cum_seqlen_q,
        cuseqlen_q,
        scale_softmax,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_producer.acquire_and_advance()
        sSum[thread_idx] = row_sum
        cute.arch.fence_view_async_shared()
        sum_handle.commit()
        row_sum_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum

        if cutlass.const_expr(mLSE is not None):
            q_idx = current_block_coord[0] * self.cta_tiler[0] + tidx
            hb_idx = (
                (current_block_coord[2][0], Int32(0))
                if cutlass.const_expr(cum_seqlen_q is not None)
                else current_block_coord[2]
            )
            lse_value = (
                scale_softmax * row_max + cute.math.log(row_sum, fastmath=True)
                if not row_sum_is_zero_or_nan
                else -Float32.inf
            )
            if cute.elem_less(q_idx, seqlen_q):
                global_q_idx = (
                    q_idx + cuseqlen_q if cutlass.const_expr(cum_seqlen_q is not None) else q_idx
                )
                mLSE[global_q_idx, hb_idx] = lse_value
        return sum_producer
