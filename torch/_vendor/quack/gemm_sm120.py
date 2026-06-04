# Copyright (c) 2025-2026, QuACK team.
# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py
# SM120-style GEMM using warp-level MMA (MmaF16BF16Op) + ldmatrix.
# Unlike SM90 WGMMA (which reads A/B from SMEM directly), warp-level MMA
# requires explicit SMEM→RMEM copies via ldmatrix before each MMA instruction.

# This is a work in progress and not very optimized.

import math
from typing import Tuple, Type, Callable, Optional
from functools import partial

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warp
from cutlass import Int32, Boolean, const_expr

from .varlen_utils import VarlenManager
from .pipeline import make_pipeline_state
from . import copy_utils
from .gemm_sm90 import GemmSm90, NamedBarrierGemm
from . import sm80_utils


class GemmSm120(GemmSm90):
    """SM120-style GEMM using warp-level MMA instead of WGMMA.

    Key differences from SM90:
    - Uses MmaF16BF16Op (warp-level, 32 threads) instead of WGMMA (warp-group, 128 threads)
    - Requires explicit SMEM→RMEM copy via ldmatrix before MMA
    - Thread config: num_mma_warps regular warps + 1 DMA warp
    - Pingpong: 2 warp groups of (2,2,1), each processing alternating tiles
    - No fp8 support (warp-level MMA only supports fp16/bf16)
    """

    arch = 120

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: Tuple[int, int] | Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool = False,
        is_persistent: bool = True,
        gather_A: bool = False,
        concat_layout: tuple | None = None,
        use_pdl: bool = True,
    ):
        # Don't call super().__init__ — we set up our own config
        self.acc_dtype = acc_dtype
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        self.use_clc_persistence = False
        self.use_pdl = use_pdl
        self.fp8_slow_accum = False
        self.gather_A = gather_A
        self.concat_layout = concat_layout or ()
        if self.pingpong:
            assert self.is_persistent, "Pingpong gemm requires persistent scheduler"
        if gather_A:
            assert cluster_shape_mnk[1] == 1

        self.cluster_shape_mnk = cluster_shape_mnk
        assert len(tile_shape_mnk) in [2, 3], "CTA tile shape must be (M, N) or (M, N, K)"
        # K dimension: if user provides 3 values, use their K; otherwise default in _setup_tiled_mma.
        self.cta_tile_shape_mnk = (
            tuple(tile_shape_mnk) if len(tile_shape_mnk) == 3 else (*tile_shape_mnk, 0)
        )
        tile_M, tile_N = self.cta_tile_shape_mnk[:2]

        # Pingpong: 2 warp groups each with (2,2,1) atom layout
        # Non-pingpong: 1 group of 8 warps with (4,2,1) atom layout
        self.mma_inst_mnk = (16, 8, 16)
        self.atom_layout_mnk = (4, 2, 1) if not self.pingpong else (2, 2, 1)
        # num_mma_warps = total warps doing MMA (both warp groups in pingpong)
        self.num_mma_warps = math.prod(self.atom_layout_mnk) * (1 if not self.pingpong else 2)
        # For compatibility with SM90 code that uses warp groups
        self.num_threads_per_warp_group = 128
        assert self.num_mma_warps % 4 == 0
        self.mma_warp_groups = self.num_mma_warps // 4
        if self.pingpong:
            assert self.mma_warp_groups == 2
        # threads_per_cta must be a multiple of 128 (warp group size) so that
        # the DMA warp's setmaxnreg.dec.sync has a complete warp group to sync with.
        self.threads_per_cta = (self.mma_warp_groups + 1) * self.num_threads_per_warp_group

        self.num_mcast_ctas_a = cluster_shape_mnk[1]
        if gather_A:
            assert self.num_mcast_ctas_a == 1
        self.num_mcast_ctas_b = cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.occupancy = 1
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}")

        # In pingpong, only 1 warp group (4 warps) participates in epilogue at a time
        self.num_epi_warps = (self.mma_warp_groups if not self.pingpong else 1) * 4
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.Epilogue),
            num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
        )
        self.num_ab_load_warps = 1 if not self.gather_A else 4
        self.ab_load_warp_id = self.num_mma_warps

        if not self.gather_A:
            self.num_regs_load = 40
            self.num_regs_mma = 232
        else:
            self.num_regs_load = 56
            self.num_regs_mma = 224

        self.ab_stage = None
        self.epi_stage = None
        self.epi_m_major = True
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def epi_smem_warp_shape_mnk(self):
        return self.atom_layout_mnk

    def _setup_tiled_mma(self):
        """Set up warp-level MMA (MmaF16BF16Op) and tile K dimension."""
        op = warp.MmaF16BF16Op(self.a_dtype, self.acc_dtype, self.mma_inst_mnk)
        tC = cute.make_layout(self.atom_layout_mnk)
        atom_m, atom_n, atom_k = self.atom_layout_mnk
        # We want each warp to have 16 consecutive elements in the N direction, for STSM
        # and for gated epilogue.
        permutation_n = cute.make_ordered_layout((self.mma_inst_mnk[1], atom_n, 2), order=(0, 2, 1))
        permutation_mnk = (
            atom_m * self.mma_inst_mnk[0],
            permutation_n,
            atom_k * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)
        tile_k = (
            self.cta_tile_shape_mnk[2]
            if self.cta_tile_shape_mnk[2] > 0
            else self.mma_inst_mnk[2] * 4
        )
        assert tile_k > 0, "CTA tile K must be positive"
        assert tile_k % self.mma_inst_mnk[2] == 0, (
            f"CTA tile K ({tile_k}) must be divisible by MMA instruction K ({self.mma_inst_mnk[2]})"
        )
        self.cta_tile_shape_mnk = (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], tile_k)

    # __call__, _setup_attributes, make_ab_pipeline, make_epi_store_pipeline,
    # make_sched_pipeline, epilogue are all inherited from GemmSm90.

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: Optional[cute.CopyAtom],
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl: Optional[cute.Tensor],
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        epilogue_params,
        varlen_params: VarlenManager.Params,
        cluster_layout_mnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_c_smem_layout: cute.ComposedLayout,
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        trace_ptr: Optional[cutlass.Int64] = None,
    ):
        from .trace import TraceContext

        tctx = TraceContext.create(trace_ptr)

        varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
        varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
        if const_expr(self.gather_A):
            assert varlen_m or varlen_k
        has_D = const_expr(mD_mnl is not None)
        has_C = const_expr(mC_mnl is not None)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch TMA descriptors
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
            ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
        )
        epi_pipeline = None
        has_epi_load = const_expr(self.epi_c_stage > 0)
        if const_expr(has_epi_load):
            epi_pipeline = self.make_epi_pipeline(
                epi_pipeline_mbar_ptr=storage.epi_pipeline_array_ptr.data_ptr(),
                tx_count=self.epi_load_bytes_per_stage,
            )
        sched_pipeline = None
        sched_data = None
        if const_expr(self.is_persistent):
            sched_pipeline = self.make_sched_pipeline(
                cluster_layout_mnk,
                sched_pipeline_mbar_ptr=storage.sched_pipeline_array_ptr.data_ptr(),
                varlen_k=varlen_k,
            )
            sched_data = storage.sched_data.get_tensor((4, self.sched_stage))

        # Cluster sync
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)

        # SMEM tensors
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sD = None
        if const_expr(has_D):
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        sC = None
        if const_expr(has_C):
            sC = storage.sC.get_tensor(epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner)
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)

        varlen_manager = VarlenManager.create(
            varlen_params,
            len_m_static=Int32(
                cute.size(mA_mkl, mode=[0])
                if varlen_k or varlen_params.mAIdx is None
                else varlen_params.mAIdx.shape[0]
            ),
            len_k_static=Int32(cute.size(mA_mkl, mode=[1])),
            len_n_static=Int32(cute.size(mB_nkl, mode=[0])),
        )

        TileSchedulerCls = partial(
            TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline
        )

        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        if warp_idx >= self.ab_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if (
                warp_idx >= self.ab_load_warp_id
                and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
            ):
                # Get mcast mask
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=1
                )
                b_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=0
                )
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

                # Persistent tile scheduling loop
                is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                if const_expr(cute.size(cluster_layout_mnk) > 1):
                    is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                while work_tile.is_valid_tile:
                    tctx.b("tma_load")
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    # Local_tile partition global tensors
                    copy_A, prefetch_A = None, None
                    if const_expr(not self.gather_A):
                        mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                        # (bM, bK, RestK)
                        gA_mk = cute.local_tile(
                            mA_mk,
                            cute.select(self.cta_tile_shape_mnk, [0, 2]),
                            (tile_coord_mnkl[0], None),
                        )
                        #  TMA load A partition_S/D
                        copy_A, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_a,
                            cta_coord=block_in_cluster_coord_mnk[1],
                            cta_layout=cute.make_layout(
                                cute.slice_(cluster_layout_mnk, (0, None, 0)).shape
                            ),
                            src_tensor=gA_mk,
                            dst_tensor=sA,
                            mcast_mask=a_mcast_mask,
                        )
                    else:
                        copy_A, prefetch_A = self._make_gather_A_copy(
                            mA_mkl, sA, varlen_manager, tile_coord_mnkl, batch_idx
                        )
                    # (bN, bK, RestK)
                    gB_nk = cute.local_tile(
                        varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                        cute.select(self.cta_tile_shape_mnk, [1, 2]),
                        (tile_coord_mnkl[1], None),
                    )
                    # TMA load B partition_S/D
                    copy_B, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_b,
                        cta_coord=block_in_cluster_coord_mnk[0],
                        cta_layout=cute.make_layout(
                            cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape
                        ),
                        src_tensor=gB_nk,
                        dst_tensor=sB,
                        mcast_mask=b_mcast_mask,
                    )
                    len_k = varlen_manager.len_k(batch_idx)
                    k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    if const_expr(not self.gather_A):
                        ab_producer_state = self.load_tma(
                            ab_pipeline, ab_producer_state, [copy_A, copy_B], k_tile_cnt
                        )
                    else:
                        ab_producer_state = self.load_AB_gather_A(
                            ab_pipeline,
                            ab_producer_state,
                            copy_A,
                            prefetch_A,
                            copy_B,
                            k_tile_cnt,
                            varlen_m=varlen_m,
                        )
                    tctx.e("tma_load")
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                if const_expr(self.pingpong and not varlen_k):
                    # Need to write the tile_idx to smem for the next WG in the pingpong mode
                    if is_scheduler_warp:
                        tile_scheduler.write_work_tile_to_smem(work_tile)
                    work_tile = tile_scheduler.get_current_work()
                ab_pipeline.producer_tail(ab_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()

        # =====================================================================
        # MMA warps
        # =====================================================================
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            is_tma_warp = Boolean(
                (not self.pingpong and warp_idx == 0)
                or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
            )
            tidx, _, _ = cute.arch.thread_idx()
            # For pingpong, adjust tidx to within-warp-group index
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            if const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group

            # ldmatrix copy atoms for SMEM → RMEM
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)
            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)

            # Make fragments
            thr_mma = tiled_mma.get_slice(tidx)
            acc, tCsA, tCsB, tCrA, tCrB = sm80_utils.partition_fragment_ABC(
                thr_mma, self.cta_tile_shape_mnk, sA, sB
            )

            if const_expr(self.pingpong):
                if warp_group_idx == 0:
                    # WG0 needs a start signal at the very beginning
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

            k_tile_cnt_static = cute.ceil_div(
                cute.size(mA_mkl, mode=[1]), self.cta_tile_shape_mnk[2]
            )
            c_tile_cnt = cute.size(cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile))

            ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            epi_store_pipeline = self.make_epi_store_pipeline()
            epi_read_state = make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.epi_c_stage
            )
            epi_producer_state = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.epi_c_stage
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()

            if const_expr(self.pingpong):
                if warp_idx >= 4:
                    # Advance 2nd Math WG pipeline states to the end of 1st Math WG
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    if const_expr(not varlen_k):
                        ab_read_state.advance_iters(k_tile_cnt_static)
                    else:
                        len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                        k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                        ab_read_state.advance_iters(k_tile_cnt)
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                len_k = varlen_manager.len_k(batch_idx)
                k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                acc.fill(0.0)
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                tctx.b("mma")
                ab_read_state = self.mma(
                    ab_pipeline,
                    ab_read_state,
                    tiled_mma,
                    acc,
                    k_tile_cnt,
                    smem_tiled_copy_A,
                    smem_tiled_copy_B,
                    tCsA_copy_view,
                    tCsB_copy_view,
                    tCrA,
                    tCrB,
                )
                if const_expr(self.pingpong):
                    # Cue for next WG's MMA to start
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
                tctx.e("mma")

                # ============================================================
                # EPILOGUE — reuse SM90's epilogue flow
                # ============================================================
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, "epi")
                tctx.b("epilogue")

                copy_D = None
                if const_expr(has_D):
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d,
                        varlen_manager.offset_batch_epi(mD_mnl, tile_coord_mnkl[3]),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sD,
                        tile_coord_mnkl,
                    )
                copy_C = None
                if const_expr(has_C):
                    copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_c,
                        varlen_manager.offset_batch_epi(mC_mnl, tile_coord_mnkl[3]),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sC,
                        tile_coord_mnkl,
                    )
                    copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)
                if const_expr(has_epi_load):
                    tile_load_copy_fns = self.epi_tile_load_g2s_copy_fns(
                        epilogue_params,
                        epi_smem_tensors,
                        tile_coord_mnkl,
                        varlen_manager,
                        epi_pipeline,
                    )
                    copy_C = copy_utils.chain_tma_producer_copy_fns((copy_C, *tile_load_copy_fns))

                d_dtype_for_layout = self.d_dtype if self.d_dtype is not None else cutlass.BFloat16
                tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                    tiled_mma, self.d_layout, d_dtype_for_layout, sD, tidx
                )
                # (R2S, R2S_M, R2S_N, (epi_M, epi_N))
                tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
                load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
                if const_expr(has_C):
                    tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                        tiled_mma, self.c_layout, self.c_dtype, sC, tRS_rD.layout, tidx
                    )
                else:
                    tiled_copy_s2r, tSR_sC, tRS_rC, tSR_rC = None, None, None, None

                self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)

                epi_read_state, epi_producer_state = self.epilogue(
                    epilogue_params,
                    epi_smem_tensors,
                    epi_pipeline,
                    epi_store_pipeline,
                    epi_read_state,
                    epi_producer_state,
                    self.epi_tile,
                    load_acc_subtile,
                    tRS_rD,
                    tRS_rC,
                    None,  # tiled_copy_t2r, for Sm100 only
                    tiled_copy_r2s,
                    tRS_sD,
                    tiled_copy_s2r,
                    tSR_rC,
                    tSR_sC,
                    copy_D,
                    copy_C,
                    tile_coord_mnkl,
                    varlen_manager,
                    self.epilogue_barrier,
                    tile_scheduler,
                    tidx,
                    is_tma_warp,
                )

                if const_expr(self.pingpong):
                    # With pingpong, 2 WGs write two different output tiles to the same smem,
                    # so we have to make sure the smem content is done reading before signaling
                    # the next WG's epilogue.
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
                tctx.e("epilogue")

                if const_expr(not self.pingpong):
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                else:  # Skip a tile for pingpong
                    # Update starting load/store pipeline states for the next tile
                    epi_read_state.advance_iters(c_tile_cnt)
                    epi_producer_state.advance_iters(c_tile_cnt)
                    # Update starting mainloop pipeline state for the next tile
                    if const_expr(not varlen_k):
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                        if work_tile.is_valid_tile:
                            len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                            k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                            ab_read_state.advance_iters(k_tile_cnt)
                            tile_scheduler.advance_to_next_work()
                            work_tile = tile_scheduler.get_current_work()

            # Wait for D store complete
            if const_expr(not self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()

        tctx.flush()

    @cute.jit
    def mma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        k_tile_cnt: Int32,
        smem_tiled_copy_A: cute.TiledCopy,
        smem_tiled_copy_B: cute.TiledCopy,
        tCsA_copy_view: cute.Tensor,
        tCsB_copy_view: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        """Warp-level MMA mainloop: ldmatrix SMEM→RMEM + warp MMA."""
        tCrA_copy_view = smem_tiled_copy_A.retile(tCrA)
        tCrB_copy_view = smem_tiled_copy_B.retile(tCrB)
        load_sA = partial(cute.copy, smem_tiled_copy_A)
        load_sB = partial(cute.copy, smem_tiled_copy_B)

        num_k_blocks = cute.size(tCrA, mode=[2])
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)

        # Load first k-block
        tCsA_p = tCsA_copy_view[None, None, None, ab_read_state.index]
        tCsB_p = tCsB_copy_view[None, None, None, ab_read_state.index]
        load_sA(tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
        load_sB(tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])

        for k_tile in cutlass.range(k_tile_cnt - 1, unroll=1):
            for k in cutlass.range_constexpr(num_k_blocks):
                k_next = 0 if k + 1 == num_k_blocks else k + 1
                if const_expr(k == num_k_blocks - 1):
                    # TMA writes this smem stage through the async proxy, while ldmatrix
                    # reads it through the generic proxy. Fence before release so the
                    # producer's next async-proxy write cannot race those reads; sync the
                    # warp because only one lane signals the empty mbarrier.
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    ab_pipeline.consumer_release(ab_read_state)
                    ab_read_state.advance()
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
                    tCsA_p = tCsA_copy_view[None, None, None, ab_read_state.index]
                    tCsB_p = tCsB_copy_view[None, None, None, ab_read_state.index]
                    ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
                load_sA(tCsA_p[None, None, k_next], tCrA_copy_view[None, None, k_next])
                load_sB(tCsB_p[None, None, k_next], tCrB_copy_view[None, None, k_next])
                cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)

        # Last k-tile (hoisted)
        if 0 < k_tile_cnt:
            for k in cutlass.range_constexpr(num_k_blocks):
                k_next = 0 if k + 1 == num_k_blocks else k + 1
                if const_expr(k == num_k_blocks - 1):
                    # TMA writes this smem stage through the async proxy, while ldmatrix
                    # reads it through the generic proxy. Fence before release so the
                    # producer's next async-proxy write cannot race those reads; sync the
                    # warp because only one lane signals the empty mbarrier.
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    ab_pipeline.consumer_release(ab_read_state)
                    ab_read_state.advance()
                if const_expr(k_next > 0):
                    load_sA(tCsA_p[None, None, k_next], tCrA_copy_view[None, None, k_next])
                    load_sB(tCsB_p[None, None, k_next], tCrB_copy_view[None, None, k_next])
                cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)

        return ab_read_state

    @staticmethod
    def _compute_tile_shape_or_override(
        cta_tile_shape_mnk: Tuple[int, int, int],
        atom_layout_mnk: Tuple[int, int, int],
        element_type: Optional[Type[cutlass.Numeric]] = None,
        epi_tile_override: Tuple[int, int] | None = None,
    ) -> Tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param cta_tile_shape_mnk: CTA tile shape (M,N,K)
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        n_perf = 64 if element_type is not None and element_type.width == 8 else 32
        tile_m = math.gcd(64, cute.size(cta_tile_shape_mnk, mode=[0]))
        tile_n = math.gcd(n_perf, cute.size(cta_tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)
