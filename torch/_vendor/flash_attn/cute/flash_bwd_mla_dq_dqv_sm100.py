# Copyright (c) 2026, Colfax International.

"""
CuTe DSL implementation of dQ+dQv gemm for DSA backward.
Performs both dQ = dS @ K and dQv = dS @ V, where K and V are
gathered according to index tensor mIdxTopK.

This uses MQA with 128 heads.

Inputs:
    - dS:      [batch, seqlen_q, nheads, top_k] or [total_q, nheads, top_k]
    - K:       [batch, seqlen_k, hdim]          or [total_k, hdim]
    - V:       [batch, seqlen_k, hdim_v]        or [total_k, hdim_v]
    - IdxTopK: [batch, seqlen_q, top_k]         or [total_q, top_k]

Outputs:
    - dQ:  [batch, seqlen_q, nheads, hdim]   or [total_q, nheads, hdim]
    - dQv: [batch, seqlen_q, nheads, hdim_v] or [total_q, nheads, hdim_v]

All sizes are known at compile time except seqlen_q, which is the batch dimension.
Representative numbers are:
    - nheads = 128
    - hdim   = 64
    - hdim_v = 512
    - top_k  = 2048

We launch a cluster of shape (1, 2) with mma tile 128x256, so that one cluster
covers the full dQv mma. dS is loaded via TMA and multicast across the CTAs.
Cluster 0 also performs the dQ mma with tile size 128x64.

K and V are loaded via CpAsync, with logic according to CpasyncGatherKVManager.
"""

from functools import partial
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .topk_gather_kv import CpasyncGatherKVManager
from .utils import get_batch_from_cu_tensor


class dQdQvGemmKernel:
    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        nheads: int,
        head_dim_k: Optional[int],
        head_dim_v: int,
        top_k: int,
    ):
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.nheads = nheads
        assert self.nheads == 128, (
            "only 128 heads supported; will expand to include 64 heads in a future PR."
        )
        self.head_dim_k = head_dim_k or 0  # when head_dim_k not provided, dQ is not computed
        self.head_dim_v = head_dim_v
        self.top_k = top_k
        self.tile_k = 128

        self.cluster_shape_mn = (1, 2)
        self.mma_tiler_dQ = (self.nheads, self.head_dim_k, self.tile_k)
        self.mma_tiler_dQv = (self.nheads, self.head_dim_v // 2, self.tile_k)
        self.num_mainloop_iters = self.top_k // self.tile_k
        self.arch = "sm_100"

        self.cta_group = tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.threads_per_warp = cute.arch.WARP_SIZE

        # ---- Set specialized warp ids ----
        self.epilogue_warp_ids = (0, 1, 2, 3)
        self.kv_load_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.sched_warp_id = 10
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                self.sched_warp_id,
                *self.epilogue_warp_ids,
                *self.kv_load_warp_ids,
            )
        )
        # ---- Set barrier id for cta sync, epilogue sync and tmem ptr sync ----
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.kv_load_sync_bar_id = 4

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=self.threads_per_warp * len(self.epilogue_warp_ids),
        )
        self.kv_load_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.kv_load_sync_bar_id,
            num_threads=self.threads_per_warp * len(self.kv_load_warp_ids),
        )

        self.is_persistent = False

        # ---- pipeline stages ---- TODO: tune these
        self.num_stages_dS = 2
        self.num_stages_KV = 2
        self.num_stages_acc = 1
        self.num_stages_clc = 1

        # ---- register allocation ----
        self.num_regs_KV = 224
        self.num_regs_epi = 128
        self.num_regs_other = 112

    @cute.jit
    def __call__(
        self,
        mdS: cute.Tensor,
        mK: Optional[cute.Tensor],
        mV: cute.Tensor,
        mdQ: cute.Tensor,
        mdQv: cute.Tensor,
        mIdxTopK: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        self.compute_dQ = const_expr(mK is not None)

        # ---- dtype info ----
        self.ds_dtype: Type[cutlass.Numeric] = mdS.element_type
        self.kv_dtype: Type[cutlass.Numeric] = mV.element_type
        self.dq_dtype: Type[cutlass.Numeric] = mdQv.element_type
        if const_expr(self.compute_dQ):
            assert self.kv_dtype == mV.element_type
            assert self.dq_dtype == mdQ.element_type

        varlen_q = const_expr(mCuSeqlensQ is not None)
        varlen_k = const_expr(mCuSeqlensK is not None)

        # ------------------------------------------------------------------ #
        # Reshape GMEM layouts for static strides                            #
        # ------------------------------------------------------------------ #
        seqlen_q = Int32(0) if const_expr(varlen_q) else mdS.shape[1]
        seqlen_q_divmod = FastDivmodDivisor(seqlen_q)
        seqlen_k = Int32(0) if const_expr(varlen_k) else mV.shape[1]

        # ---- group batch and seqlen modes in nonvarlen case ----
        def group_batch_seqlen(t: cute.Tensor, varlen: bool) -> cute.Tensor:
            if const_expr(not varlen):
                t = cute.make_tensor(
                    t.iterator,
                    cute.make_layout(
                        (t.shape[1], t.shape[0], *t.shape[2:]),
                        stride=(t.stride[1], t.stride[0], *t.stride[2:]),
                    ),
                )
                t = cute.group_modes(t, 0, 2)
            return t

        mdS = group_batch_seqlen(mdS, varlen_q)
        mdQv = group_batch_seqlen(mdQv, varlen_q)
        mV = group_batch_seqlen(mV, varlen_k)
        mIdxTopK = group_batch_seqlen(mIdxTopK, varlen_q)
        if const_expr(self.compute_dQ):
            mdQ = group_batch_seqlen(mdQ, varlen_q)
            mK = group_batch_seqlen(mK, varlen_k)

        # ---- transpose and make static modes static ----
        def static_reshape(t: cute.Tensor, *static_shapes) -> cute.Tensor:
            static_modes = range(1, len(t.shape))
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (*static_shapes, t.shape[0]),
                    stride=(*(t.stride[i] for i in static_modes), t.stride[0]),
                ),
            )

        mdS = static_reshape(mdS, self.nheads, self.top_k)
        mdQv = static_reshape(mdQv, self.nheads, self.head_dim_v)
        mV = static_reshape(mV, self.head_dim_v)
        mIdxTopK = static_reshape(mIdxTopK, self.top_k)
        if const_expr(self.compute_dQ):
            mdQ = static_reshape(mdQ, self.nheads, self.head_dim_k)
            mK = static_reshape(mK, self.head_dim_k)

        # ---- layout info ----
        self.ds_major_mode = utils.LayoutEnum.from_tensor(mdS).mma_major_mode()
        self.kv_major_mode = utils.LayoutEnum.from_tensor(mV).mma_major_mode()
        self.dq_layout = utils.LayoutEnum.from_tensor(mdQv)
        if const_expr(self.compute_dQ):
            assert self.dq_layout == utils.LayoutEnum.from_tensor(mdQ)

        # ------------------------------------------------------------------ #
        # Setup attributes that depend on kernel inputs                      #
        # ------------------------------------------------------------------ #
        tiled_mma_v = utils.sm100.make_trivial_tiled_mma(
            self.ds_dtype,
            self.ds_major_mode,
            self.kv_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dQv[:2],
        )
        if const_expr(self.compute_dQ):
            tiled_mma_k = utils.sm100.make_trivial_tiled_mma(
                self.ds_dtype,
                self.ds_major_mode,
                self.kv_major_mode,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler_dQ[:2],
            )

        self.cta_tile_shape_dQv = (
            self.mma_tiler_dQv[0],
            self.mma_tiler_dQv[1],
            self.mma_tiler_dQv[2],
        )

        # ---- Compute cluster layout ----
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_v.thr_id.shape,),
        )

        # ---- Compute number of multicast CTAs for A/B ----
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])

        # ---- Compute epi tiles for dQ/dQv ----
        self.epi_tile_dQv = utils.sm100.compute_epilogue_tile_shape(
            self.cta_tile_shape_dQv,
            False,  # use_2cta_instrs
            self.dq_layout,
            self.dq_dtype,
        )
        self.epi_tile_dQ = None
        if const_expr(self.compute_dQ):
            self.epi_tile_dQ = utils.sm100.compute_epilogue_tile_shape(
                self.mma_tiler_dQ,
                False,
                self.dq_layout,
                self.dq_dtype,
            )

        # ---- Device-specific attributes ----
        self.smem_capacity = utils.get_smem_capacity_in_bytes()

        self.num_tmem_alloc_cols = 512

        # ------------------------------------------------------------------ #
        # Make SMEM layouts                                                  #
        # ------------------------------------------------------------------ #
        sdS_layout = utils.sm100.make_smem_layout_a(
            tiled_mma_v,
            self.mma_tiler_dQ,
            self.ds_dtype,
            self.num_stages_dS,
        )
        sV_layout = utils.sm100.make_smem_layout_b(
            tiled_mma_v,
            self.mma_tiler_dQv,
            self.kv_dtype,
            self.num_stages_KV,
        )
        sdQv_layout = utils.sm100.make_smem_layout_epi(
            self.dq_dtype,
            self.dq_layout,
            self.epi_tile_dQv,
            self.num_stages_acc,
        )
        sK_layout, sdQ_layout = None, None
        if const_expr(self.compute_dQ):
            sK_layout = utils.sm100.make_smem_layout_b(
                tiled_mma_k,
                self.mma_tiler_dQ,
                self.kv_dtype,
                self.num_stages_KV,
            )
            sdQ_layout = utils.sm100.make_smem_layout_epi(
                self.dq_dtype,
                self.dq_layout,
                self.epi_tile_dQ,
                self.num_stages_acc,
            )

        # ------------------------------------------------------------------ #
        # Set up TMA load/stores                                             #
        # ------------------------------------------------------------------ #
        atom_thr_size = cute.size(tiled_mma_v.thr_id.shape)

        # ---- Setup TMA load for dS ----
        dS_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma_v.thr_id)
        dS_smem_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        tma_atom_dS, tma_tensor_dS = cute.nvgpu.make_tiled_tma_atom_A(
            dS_op,
            mdS,
            dS_smem_layout,
            self.mma_tiler_dQv,
            tiled_mma_v,
            self.cluster_layout_vmnk.shape,
        )

        dS_copy_size = cute.size_in_bytes(self.ds_dtype, dS_smem_layout)
        self.num_tma_load_bytes = dS_copy_size * atom_thr_size

        # ---- Setup TMA store for dQ and dQV ----
        dQv_epi_smem_layout = cute.select(sdQv_layout, mode=[0, 1])
        tma_atom_dQv, tma_tensor_dQv = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mdQv, dQv_epi_smem_layout, self.epi_tile_dQv
        )
        tma_atom_dQ, tma_tensor_dQ = None, None
        if const_expr(self.compute_dQ):
            dQ_epi_smem_layout = cute.select(sdQ_layout, mode=[0, 1])
            tma_atom_dQ, tma_tensor_dQ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), mdQ, dQ_epi_smem_layout, self.epi_tile_dQ
            )

        # ------------------------------------------------------------------ #
        # Set up shared storage for SMEM                                     #
        # ------------------------------------------------------------------ #

        self.buffer_align_bytes = 1024

        sdS_size = cute.cosize(sdS_layout)
        sK_size = cute.cosize(sK_layout) if const_expr(self.compute_dQ) else 0
        sV_size = cute.cosize(sV_layout)
        sdQ_size = cute.cosize(sdQ_layout) if const_expr(self.compute_dQ) else 0
        sdQv_size = cute.cosize(sdQv_layout)
        assert sdQ_size <= sK_size, f"require {sdQ_size=} <= {sK_size=}"
        assert sdQv_size <= sV_size, f"require {sdQv_size=} <= {sV_size=}"

        self.overlap_kv_epi = self.compute_dQ
        if const_expr(self.overlap_kv_epi):
            sdQ_size = 0
            sdQv_size = 0

        @cute.struct
        class SharedStorage:
            mbar_ptr_dS: cute.struct.MemRange[cutlass.Int64, self.num_stages_dS * 2]
            mbar_ptr_KV: cute.struct.MemRange[cutlass.Int64, self.num_stages_KV * 2]
            mbar_ptr_dQ_dQv: cute.struct.MemRange[cutlass.Int64, self.num_stages_acc * 2]
            mbar_ptr_KV_cpasync: cute.struct.MemRange[cutlass.Int64, self.num_stages_KV * 2]
            mbar_ptr_load_kv_epi: cute.struct.MemRange[cutlass.Int64, 2]
            # Tmem holding buffer
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # Clc pointers
            clc_ptr: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, self.num_stages_clc * 2], 16
            ]
            clc_response_ptr: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
            # Smem tensors
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, sdS_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.kv_dtype, sK_size],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.kv_dtype, sV_size],
                self.buffer_align_bytes,
            ]
            sdQ: cute.struct.Align[
                cute.struct.MemRange[self.dq_dtype, sdQ_size],
                self.buffer_align_bytes,
            ]
            sdQv: cute.struct.Align[
                cute.struct.MemRange[self.dq_dtype, sdQv_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # ---- Compute grid size ----
        self.tile_sched_params, grid = self._compute_grid(
            mdQv,
            self.cta_tile_shape_dQv,
            self.cluster_shape_mn,
        )
        self.num_clc_response_bytes = 16
        # permute grid to conform to grid_dim_z <= 65536 constraint;
        # this is undone in the kernel
        grid = (grid[2], grid[1], grid[0])

        # cute.printf("dQ/dQv grid: {}", grid)
        # print("dQ/dQv SMEM: ", self.shared_storage.size_in_bytes())
        # ---- Launch the kernel synchronously ----
        self.kernel(
            tiled_mma_k if const_expr(self.compute_dQ) else None,
            tiled_mma_v,
            tma_atom_dS,
            tma_tensor_dS,
            mK,
            mV,
            tma_atom_dQ,
            tma_tensor_dQ,
            tma_atom_dQv,
            tma_tensor_dQv,
            mIdxTopK,
            mCuSeqlensQ,
            mCuSeqlensK,
            seqlen_q_divmod,
            self.cluster_layout_vmnk,
            sdS_layout,
            sK_layout,
            sV_layout,
            sdQ_layout,
            sdQv_layout,
            self.epi_tile_dQ,
            self.epi_tile_dQv,
            self.tile_sched_params,
            seqlen_k,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma_k: Optional[cute.TiledMma],
        tiled_mma_v: cute.TiledMma,
        tma_atom_dS: cute.CopyAtom,
        mdS: cute.Tensor,
        mK: Optional[cute.Tensor],
        mV: cute.Tensor,
        tma_atom_dQ: Optional[cute.CopyAtom],
        mdQ: Optional[cute.Tensor],
        tma_atom_dQv: cute.CopyAtom,
        mdQv: cute.Tensor,
        mIdxTopK: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        seqlen_q_divmod: FastDivmodDivisor,
        cluster_layout_vmnk: cute.Layout,
        sdS_layout: cute.ComposedLayout,
        sK_layout: Optional[cute.ComposedLayout],
        sV_layout: cute.ComposedLayout,
        sdQ_layout: Optional[cute.ComposedLayout],
        sdQv_layout: cute.ComposedLayout,
        epi_tile_dQ: Optional[cute.Tile],
        epi_tile_dQv: cute.Tile,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        seqlen_k_static: Int32,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # ------------------------------------------------------------------ #
        # Prefetch TMA descriptors                                           #
        # ------------------------------------------------------------------ #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_dS)
            cpasync.prefetch_descriptor(tma_atom_dQv)
            if const_expr(self.compute_dQ):
                cpasync.prefetch_descriptor(tma_atom_dQ)

        # ------------------------------------------------------------------ #
        # Cluster coordinates                                                #
        # ------------------------------------------------------------------ #
        bidx, bidy, bidz = cute.arch.block_idx()
        gridx, gridy, gridz = cute.arch.grid_dim()
        mma_v_tile_coord_v = bidx % cute.size(tiled_mma_v.thr_id.shape)
        if const_expr(self.compute_dQ):
            mma_k_tile_coord_v = bidx % cute.size(tiled_mma_k.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        is_first_cta = cta_rank_in_cluster == 0
        tidx, _, _ = cute.arch.thread_idx()

        # ------------------------------------------------------------------ #
        # Shared storage allocation                                          #
        # ------------------------------------------------------------------ #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # ------------------------------------------------------------------ #
        # Initialize pipelines                                               #
        # ------------------------------------------------------------------ #
        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        kv_commit_group = ThreadCooperativeGroup(1)
        clc_producer_group = ThreadCooperativeGroup(1)
        num_clc_consumer_threads = 32 * (
            1  # sched warp on CTA0 only
            + cute.size(self.cluster_shape_mn)
            * (1 + len(self.epilogue_warp_ids) + len(self.kv_load_warp_ids) + 1)
            # tma + epi + kv_load + mma, on BOTH CTAs
        )
        clc_consumer_group = ThreadCooperativeGroup(num_clc_consumer_threads)
        mma_warp = ThreadCooperativeGroup(1)
        tma_warp = ThreadCooperativeGroup(cute.size(self.cluster_shape_mn))
        tma_warp_local = ThreadCooperativeGroup(1)
        epilogue_warps = ThreadCooperativeGroup(len(self.epilogue_warp_ids))
        load_warps = ThreadCooperativeGroup(len(self.kv_load_warp_ids))

        pipeline_dS = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_ptr_dS.data_ptr(),
            num_stages=self.num_stages_dS,
            producer_group=mma_warp,
            consumer_group=tma_warp,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_KV = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_ptr_KV.data_ptr(),
            num_stages=self.num_stages_KV,
            producer_group=kv_commit_group,
            consumer_group=mma_warp,
            defer_sync=True,
        )

        pipeline_dQ_dQv = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_ptr_dQ_dQv.data_ptr(),
            num_stages=self.num_stages_acc,
            producer_group=mma_warp,
            consumer_group=epilogue_warps,
            defer_sync=True,
        )

        pipeline_clc = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_ptr.data_ptr(),
            num_stages=self.num_stages_clc,
            producer_group=clc_producer_group,
            consumer_group=clc_consumer_group,
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_load_kv_epi = None
        if const_expr(self.overlap_kv_epi):
            pipeline_load_kv_epi = pipeline.PipelineAsync.create(
                barrier_storage=storage.mbar_ptr_load_kv_epi.data_ptr(),
                num_stages=1,
                producer_group=epilogue_warps,
                consumer_group=load_warps,
                defer_sync=True,
            )

        # ------------------------------------------------------------------ #
        # TMEM Allocation                                                    #
        # ------------------------------------------------------------------ #
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_ids)),
        )
        # ---- Tensor memory dealloc barrier init ----
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_ids[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # ---- Cluster arrive after barrier init ----
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # ---- Initial clc response pointer ----
        clc_response_ptr = storage.clc_response_ptr.data_ptr()

        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages_clc
        )

        # ------------------------------------------------------------------ #
        # SMEM tensors                                                       #
        # ------------------------------------------------------------------ #
        # (MMA, MMA_M, MMA_K, STAGE)
        sdS = storage.sdS.get_tensor(sdS_layout.outer, swizzle=sdS_layout.inner)
        if const_expr(self.compute_dQ):
            sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        # sdQ = storage.sdQ.get_tensor(sdQ_layout.outer, swizzle=sdQ_layout.inner)
        if const_expr(self.compute_dQ):
            sdQ = cute.make_tensor(
                cute.recast_ptr(sK.iterator, sdQ_layout.inner, self.dq_dtype), sdQ_layout.outer
            )
        if const_expr(not self.compute_dQ):
            sdQv = storage.sdQv.get_tensor(sdQv_layout.outer, swizzle=sdQv_layout.inner)
        else:
            sdQv = cute.make_tensor(
                cute.recast_ptr(sV.iterator, sdQv_layout.inner, self.dq_dtype), sdQv_layout.outer
            )

        dS_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )

        # ------------------------------------------------------------------ #
        # Global tile partitioning                                           #
        # ------------------------------------------------------------------ #
        # (bM, bK, RestM, RestK, RestL)
        gdS = cute.local_tile(
            mdS, cute.slice_(self.mma_tiler_dQv, (None, 0, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        if const_expr(self.compute_dQ):
            gdQ = cute.local_tile(
                mdQ, cute.slice_(self.mma_tiler_dQ, (None, None, 0)), (None, None, None)
            )
        gdQv = cute.local_tile(
            mdQv, cute.slice_(self.mma_tiler_dQv, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gdS, mode=[3])

        # ------------------------------------------------------------------ #
        # TiledMMA partitioning                                              #
        # ------------------------------------------------------------------ #
        thr_mma_v = tiled_mma_v.get_slice(mma_v_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tdQvgdS = thr_mma_v.partition_A(gdS)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tdQvgdQv = thr_mma_v.partition_C(gdQv)
        if const_expr(self.compute_dQ):
            thr_mma_k = tiled_mma_k.get_slice(mma_k_tile_coord_v)
            tdQgdQ = thr_mma_k.partition_C(gdQ)

        # ------------------------------------------------------------------ #
        # TMA partition for dS                                               #
        # ------------------------------------------------------------------ #
        dS_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tdSsdS, tdSgdS = cpasync.tma_partition(
            tma_atom_dS,
            block_in_cluster_coord_vmnk[2],
            dS_cta_layout,
            cute.group_modes(sdS, 0, 3),
            cute.group_modes(tdQvgdS, 0, 3),
        )

        # ------------------------------------------------------------------ #
        # MMA fragments                                                      #
        # ------------------------------------------------------------------ #
        # (MMA, MMA_M, MMA_K, STAGE)
        tdQvrdS = tiled_mma_v.make_fragment_A(sdS)
        # (MMA, MMA_N, MMA_K, STAGE)
        tdQvrV = tiled_mma_v.make_fragment_B(sV)
        # (MMA, MMA_M, MMA_N)
        acc_v_shape = tiled_mma_v.partition_shape_C(self.mma_tiler_dQv[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tdQvtAcc_fake = tiled_mma_v.make_fragment_C(cute.append(acc_v_shape, self.num_stages_acc))
        if const_expr(self.compute_dQ):
            # (MMA, MMA_N, MMA_K, STAGE)
            tdQrK = tiled_mma_k.make_fragment_B(sK)
            # (MMA, MMA_M, MMA_N)
            acc_k_shape = tiled_mma_k.partition_shape_C(self.mma_tiler_dQ[:2])
            # (MMA, MMA_M, MMA_N, STAGE)
            tdQtAcc_fake = tiled_mma_k.make_fragment_C(
                cute.append(acc_k_shape, self.num_stages_acc)
            )

        # ------------------------------------------------------------------ #
        # Cluster wait before tensor memory alloc                            #
        # ------------------------------------------------------------------ #
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ------------------------------------------------------------------ #
        # Tile Scheduler                                                     #
        # ------------------------------------------------------------------ #
        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
        )
        work_tile = tile_sched.initial_work_tile_info()

        # ------------------------------------------------------------------ #
        # TMA load warp                                                      #
        # ------------------------------------------------------------------ #
        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

            producer_state_dS = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages=self.num_stages_dS
            )

            while work_tile.is_valid_tile:
                # ---- Get tile coord from tile scheduler ----
                token, cta, _ = work_tile.tile_idx

                # ((atom_v, rest_v), RestK)
                tdSgdS_slice = tdSgdS[(None, 0, None, token)]

                # ---- mainloop ----
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    pipeline_dS.producer_acquire(producer_state_dS)
                    index_dS = producer_state_dS.index

                    # ---- TMA load dS ----
                    cute.copy(
                        tma_atom_dS,
                        tdSgdS_slice[(None, k_tile)],
                        tdSsdS[(None, index_dS)],
                        tma_bar_ptr=pipeline_dS.producer_get_barrier(producer_state_dS),
                        mcast_mask=dS_full_mcast_mask,
                    )

                    producer_state_dS.advance()

                # ---- Advance to next tile ----
                pipeline_clc.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                pipeline_clc.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            # ---- Wait dS buffer empty ----
            pipeline_dS.producer_tail(producer_state_dS)

        # ------------------------------------------------------------------ #
        # Clc Scheduler warp                                                 #
        # ------------------------------------------------------------------ #
        if warp_idx == self.sched_warp_id and is_first_cta:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, self.num_stages_clc
            )

            while work_tile.is_valid_tile:
                pipeline_clc.producer_acquire(clc_producer_state)
                mbar_addr = pipeline_clc.producer_get_barrier(clc_producer_state)
                tile_sched.advance_to_next_work(mbar_addr)
                clc_producer_state.advance()

                pipeline_clc.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                pipeline_clc.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            pipeline_clc.producer_tail(clc_producer_state)

        # ------------------------------------------------------------------ #
        # CpAsync KV load warps                                              #
        # ------------------------------------------------------------------ #
        if warp_idx >= self.kv_load_warp_ids[0] and warp_idx <= self.kv_load_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_KV)
            find_batch = partial(
                self.find_batch_from_q, seqlen_q_divmod=seqlen_q_divmod, mCuSeqlensQ=mCuSeqlensQ
            )

            load_epi_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            producer_state_KV = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages=self.num_stages_KV
            )

            kv_tidx = tidx % (len(self.kv_load_warp_ids) * self.threads_per_warp)
            kv_warp_idx = warp_idx % len(self.kv_load_warp_ids)

            mV_cta = cute.domain_offset((cta_rank_in_cluster * (self.head_dim_v // 2), 0), mV)

            while work_tile.is_valid_tile:
                # ---- Get tile coord from tile scheduler ----
                token, cta, _ = work_tile.tile_idx

                batch_idx = find_batch(token)
                k_batch_offset = (
                    mCuSeqlensK[batch_idx] if const_expr(mCuSeqlensK is not None) else Int32(0)
                )
                seqlen_k = (
                    mCuSeqlensK[batch_idx + 1] - k_batch_offset
                    if const_expr(mCuSeqlensK is not None)
                    else seqlen_k_static
                )
                if const_expr(mCuSeqlensK is not None):
                    if const_expr(self.compute_dQ):
                        mK_cur = cute.domain_offset((0, k_batch_offset), mK)[None, None]
                    mV_cur = cute.domain_offset((0, k_batch_offset), mV_cta)[None, None]
                else:
                    if const_expr(self.compute_dQ):
                        mK_cur = cute.domain_offset((0, (0, batch_idx)), mK)[None, None]
                    mV_cur = cute.domain_offset((0, (0, batch_idx)), mV_cta)[None, None]
                mIdxTopK_cur = mIdxTopK[None, token]

                cpasync_gather_kv_manager = CpasyncGatherKVManager.create(
                    mIdxTopK_cur,
                    0,
                    kv_tidx,
                    kv_warp_idx,
                    self.top_k,
                    seqlen_k,
                    self.mma_tiler_dQv[2],
                    self.head_dim_k,
                    self.mma_tiler_dQv[1],
                    1,
                    len(self.kv_load_warp_ids) * self.threads_per_warp,
                    mV.element_type,
                    1,
                )

                # ---- K/V load mainloop ----
                for k_tile in cutlass.range_constexpr(k_tile_cnt):
                    # ---- Load top-k index tensor ----
                    cpasync_gather_kv_manager.load_index_topk(k_tile, transpose=True)

                    stage = producer_state_KV.index
                    pipeline_KV.producer_acquire(producer_state_KV)

                    # ---- Load V (and optionally load K) ----
                    cpasync_gather_kv_manager.load_X(mV_cur, sV[None, None, None, stage], True, "V")
                    if const_expr(self.compute_dQ):
                        if is_first_cta:
                            cpasync_gather_kv_manager.load_X(
                                mK_cur, sK[None, None, None, stage], True, "K"
                            )

                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    self.kv_load_sync_barrier.arrive_and_wait()

                    cute.arch.fence_proxy("async.shared", space="cta")
                    if kv_warp_idx == 0:
                        with cute.arch.elect_one():
                            pipeline_KV.producer_commit(producer_state_KV)
                    producer_state_KV.advance()

                if const_expr(self.overlap_kv_epi):
                    pipeline_load_kv_epi.consumer_wait(load_epi_consumer_state)
                    with cute.arch.elect_one():
                        pipeline_load_kv_epi.consumer_release(load_epi_consumer_state)
                    load_epi_consumer_state.advance()

                # ---- Advance to next tile ----
                pipeline_clc.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                pipeline_clc.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            pipeline_KV.producer_tail(producer_state_KV)

        # ------------------------------------------------------------------ #
        # MMA warp                                                           #
        # ------------------------------------------------------------------ #
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # --- Retrieve TMEM ptr and make accumulator tensors
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            if const_expr(self.compute_dQ):
                tdQtAcc_base = cute.make_tensor(tmem_ptr, tdQtAcc_fake.layout)
            tdQvtAcc_ptr = tmem_ptr + (
                tcgen05.find_tmem_tensor_col_offset(tdQtAcc_base)
                if const_expr(self.compute_dQ)
                else 0
            )
            tdQvtAcc_base = cute.make_tensor(tdQvtAcc_ptr, tdQvtAcc_fake.layout)

            consumer_state_dS = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages=self.num_stages_dS
            )
            consumer_state_KV = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages=self.num_stages_KV
            )
            producer_state_dQ_dQv = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages_acc
            )

            while work_tile.is_valid_tile:
                # ---- Get tile coord from tile scheduler ----
                token, cta, _ = work_tile.tile_idx

                # ---- Set tensor memory buffer for current tile ----
                # (MMA, MMA_M, MMA_N)
                tdQvtAcc = tdQvtAcc_base[(None, None, None, producer_state_dQ_dQv.index)]
                if const_expr(self.compute_dQ):
                    tdQtAcc = tdQtAcc_base[(None, None, None, producer_state_dQ_dQv.index)]

                # ---- Wait for accumulator buffer empty ----
                pipeline_dQ_dQv.producer_acquire(producer_state_dQ_dQv)

                # ---- Reset the ACCUMULATE field for each tile ----
                tiled_mma_v.set(tcgen05.Field.ACCUMULATE, False)
                if const_expr(self.compute_dQ):
                    tiled_mma_k.set(tcgen05.Field.ACCUMULATE, False)

                # ---- Mma mainloop ----
                for k_tile in cutlass.range_constexpr(k_tile_cnt):
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    pipeline_KV.consumer_wait(consumer_state_KV)
                    dS_stage = consumer_state_dS.index
                    KV_stage = consumer_state_KV.index

                    num_kblocks = cute.size(tdQvrdS, mode=[2])
                    for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                        # dQv += dS @ V
                        cute.gemm(
                            tiled_mma_v,
                            tdQvtAcc,
                            tdQvrdS[(None, None, kblk_idx, dS_stage)],
                            tdQvrV[(None, None, kblk_idx, KV_stage)],
                            tdQvtAcc,
                        )
                        # Enable accumulate on tdQvtAcc after first kblock
                        tiled_mma_v.set(tcgen05.Field.ACCUMULATE, True)

                        if const_expr(self.compute_dQ):
                            if is_first_cta:
                                # dQ += dS @ K
                                cute.gemm(
                                    tiled_mma_k,
                                    tdQtAcc,
                                    tdQvrdS[(None, None, kblk_idx, dS_stage)],
                                    tdQrK[(None, None, kblk_idx, KV_stage)],
                                    tdQtAcc,
                                )
                                # Enable accumulate on tdQtAcc after first kblock
                                tiled_mma_k.set(tcgen05.Field.ACCUMULATE, True)

                    pipeline_dS.consumer_release(consumer_state_dS)
                    pipeline_KV.consumer_release(consumer_state_KV)
                    consumer_state_dS.advance()
                    consumer_state_KV.advance()

                pipeline_dQ_dQv.producer_commit(producer_state_dQ_dQv)
                producer_state_dQ_dQv.advance()

                # ---- Advance to next tile ----
                pipeline_clc.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                pipeline_clc.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            pipeline_dQ_dQv.producer_tail(producer_state_dQ_dQv)

        # ------------------------------------------------------------------ #
        # Epilogue warps                                                     #
        # ------------------------------------------------------------------ #
        if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_epi)
            # ---- Alloc tensor memory buffer ----
            tmem.allocate(self.num_tmem_alloc_cols)

            # ---- Retrieving tensor memory ptr and make accumulator tensor ----
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            if const_expr(self.compute_dQ):
                tdQtAcc_base = cute.make_tensor(tmem_ptr, tdQtAcc_fake.layout)
            tdQvtAcc_ptr = tmem_ptr + (
                tcgen05.find_tmem_tensor_col_offset(tdQtAcc_base)
                if const_expr(self.compute_dQ)
                else 0
            )
            tdQvtAcc_base = cute.make_tensor(tdQvtAcc_ptr, tdQvtAcc_fake.layout)

            epi_idx = tidx
            # print(f"tdQvtAcc_base.layout = {tdQvtAcc_base.layout}")
            # ---- TMEM -> RMEM -> SMEM -> GMEM copies + partitions ----
            tiled_copy_dQv_t2r, tTR_dQvtAcc_base, tTR_dQvrAcc = (
                self.epilogue_tmem_copy_and_partition(
                    epi_idx,
                    tdQvtAcc_base,
                    tdQvgdQv,
                    epi_tile_dQv,
                    self.mma_tiler_dQv,
                    self.dq_layout,
                    self.dq_dtype,
                )
            )
            # print(f"tTR_tdQvtAcc_base.layout = {tTR_dQvtAcc_base.layout}")
            tTR_rdQv = cute.make_rmem_tensor(tTR_dQvrAcc.shape, self.dq_dtype)
            (
                tiled_copy_dQv_r2s,
                tRS_rdQv,
                tRS_sdQv,
            ) = self.epilogue_smem_copy_and_partition(
                self.dq_layout,
                self.dq_dtype,
                tiled_copy_dQv_t2r,
                tTR_rdQv,
                epi_idx,
                sdQv,
            )
            bSG_sdQv, bSG_gdQv_partitioned = self.epilogue_gmem_copy_and_partition(
                tma_atom_dQv,
                tdQvgdQv,
                epi_tile_dQv,
                sdQv,
            )
            if const_expr(self.compute_dQ):
                (tiled_copy_dQ_t2r, tTR_dQtAcc_base, tTR_dQrAcc) = (
                    self.epilogue_tmem_copy_and_partition(
                        epi_idx,
                        tdQtAcc_base,
                        tdQgdQ,
                        epi_tile_dQ,
                        self.mma_tiler_dQ,
                        self.dq_layout,
                        self.dq_dtype,
                    )
                )

                tTR_rdQ = cute.make_rmem_tensor(tTR_dQrAcc.shape, self.dq_dtype)
                (
                    tiled_copy_dQ_r2s,
                    tRS_rdQ,
                    tRS_sdQ,
                ) = self.epilogue_smem_copy_and_partition(
                    self.dq_layout,
                    self.dq_dtype,
                    tiled_copy_dQ_t2r,
                    tTR_rdQ,
                    epi_idx,
                    sdQ,
                )
                bSG_sdQ, bSG_gdQ_partitioned = self.epilogue_gmem_copy_and_partition(
                    tma_atom_dQ,
                    tdQgdQ,
                    epi_tile_dQ,
                    sdQ,
                )

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages_acc
            )
            load_epi_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )

            epi_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                cute.arch.WARP_SIZE,
            )
            pipeline_epi = pipeline.PipelineTmaStore.create(
                num_stages=self.num_stages_acc,
                producer_group=epi_producer_group,
            )

            # ---- Persistent tile scheduling loop for epilogue ----
            while work_tile.is_valid_tile:
                # ---- Get current work tile ----
                token, cta, bid_x = work_tile.tile_idx

                bSG_gdQv = bSG_gdQv_partitioned[
                    (None, None, None, bid_x, cta_rank_in_cluster, token)
                ]
                tTR_dQvtAcc = tTR_dQvtAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]
                tTR_dQvtAcc = cute.group_modes(tTR_dQvtAcc, 3, cute.rank(tTR_dQvtAcc))
                bSG_gdQv = cute.group_modes(bSG_gdQv, 1, cute.rank(bSG_gdQv))

                subtile_cnt_v = cute.size(tTR_dQvtAcc.shape, mode=[3])
                epi_subtile_counter_v = 0

                if const_expr(self.compute_dQ):
                    bSG_gdQ = bSG_gdQ_partitioned[(None, None, None, bid_x, cta, token)]
                    tTR_dQtAcc = tTR_dQtAcc_base[
                        (None, None, None, None, None, acc_consumer_state.index)
                    ]
                    tTR_dQtAcc = cute.group_modes(tTR_dQtAcc, 3, cute.rank(tTR_dQtAcc))
                    bSG_gdQ = cute.group_modes(bSG_gdQ, 1, cute.rank(bSG_gdQ))

                    subtile_cnt_k = cute.size(tTR_dQtAcc.shape, mode=[3])
                    epi_subtile_counter_k = 0

                pipeline_dQ_dQv.consumer_wait(acc_consumer_state)

                for subtile_idx in cutlass.range(subtile_cnt_v, unroll_full=True):
                    store_dQ = (
                        const_expr(self.compute_dQ)
                        and is_first_cta
                        and (subtile_idx < subtile_cnt_k)
                    )

                    if not store_dQ:
                        tTR_dQvtAcc_mn = tTR_dQvtAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_dQv_t2r, tTR_dQvtAcc_mn, tTR_dQvrAcc)
                        cute.arch.fence_view_async_tmem_load()

                        # convert to output dtype
                        tRS_rdQv.store(
                            tiled_copy_dQv_r2s.retile(tTR_dQvrAcc).load().to(self.dq_dtype)
                        )

                        epi_buffer = epi_subtile_counter_v % self.num_stages_acc
                        cute.copy(
                            tiled_copy_dQv_r2s, tRS_rdQv, tRS_sdQv[(None, None, None, epi_buffer)]
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                        if warp_idx == self.epilogue_warp_ids[0]:
                            cute.copy(
                                tma_atom_dQv,
                                bSG_sdQv[(None, epi_buffer)],
                                bSG_gdQv[(None, subtile_idx)],
                            )
                            pipeline_epi.producer_commit()
                            pipeline_epi.producer_acquire()

                        self.epilog_sync_barrier.arrive_and_wait()
                        epi_subtile_counter_v += 1
                    elif const_expr(self.compute_dQ):
                        tTR_dQtAcc_mn = tTR_dQtAcc[(None, None, None, subtile_idx)]
                        tTR_dQvtAcc_mn = tTR_dQvtAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_dQ_t2r, tTR_dQtAcc_mn, tTR_dQrAcc)
                        cute.copy(tiled_copy_dQv_t2r, tTR_dQvtAcc_mn, tTR_dQvrAcc)

                        # convert to output dtype
                        tRS_rdQ.store(tiled_copy_dQ_r2s.retile(tTR_dQrAcc).load().to(self.dq_dtype))
                        tRS_rdQv.store(
                            tiled_copy_dQv_r2s.retile(tTR_dQvrAcc).load().to(self.dq_dtype)
                        )

                        epi_buffer = epi_subtile_counter_v % self.num_stages_acc
                        epi_buffer_k = epi_subtile_counter_k % self.num_stages_acc
                        cute.copy(
                            tiled_copy_dQv_r2s, tRS_rdQv, tRS_sdQv[(None, None, None, epi_buffer)]
                        )
                        cute.copy(
                            tiled_copy_dQ_r2s, tRS_rdQ, tRS_sdQ[(None, None, None, epi_buffer_k)]
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                        if warp_idx == self.epilogue_warp_ids[0]:
                            cute.copy(
                                tma_atom_dQv,
                                bSG_sdQv[(None, epi_buffer)],
                                bSG_gdQv[(None, subtile_idx)],
                            )
                            cute.copy(
                                tma_atom_dQ,
                                bSG_sdQ[(None, epi_buffer_k)],
                                bSG_gdQ[(None, subtile_idx)],
                            )
                            pipeline_epi.producer_commit()
                            pipeline_epi.producer_acquire()

                        self.epilog_sync_barrier.arrive_and_wait()
                        epi_subtile_counter_v += 1
                        epi_subtile_counter_k += 1

                if const_expr(self.overlap_kv_epi):
                    pipeline_load_kv_epi.producer_acquire(load_epi_producer_state)
                    with cute.arch.elect_one():
                        pipeline_load_kv_epi.producer_commit(load_epi_producer_state)
                    load_epi_producer_state.advance()

                with cute.arch.elect_one():
                    pipeline_dQ_dQv.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                # ---- Advance to next tile ----
                pipeline_clc.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                pipeline_clc.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            # ---- Dealloc the tensor memory buffer ----
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            pipeline_epi.producer_tail()

    def epilogue_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        mma_tiler_mnk,
        c_layout,
        c_dtype,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_t2r = utils.sm100.get_tmem_load_op(
            mma_tiler_mnk,
            c_layout,
            c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs=False,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)

        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)

        # (T2R, T2R_M, T2R_N)
        rAcc_shape = tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape
        tTR_rAcc = cute.make_rmem_tensor(rAcc_shape, self.acc_dtype)

        return (tiled_copy_t2r, tTR_tAcc, tTR_rAcc)

    def epilogue_smem_copy_and_partition(
        self,
        c_layout,
        c_dtype: Type[cutlass.Numeric],
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = utils.sm100.get_smem_store_op(
            c_layout, c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilogue_gmem_copy_and_partition(
        self,
        tma_atom,
        gC,
        epi_tile,
        sC,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        gC_epi = cute.flat_divide(gC[((None, None), 0, 0, None, None, None)], epi_tile)

        sC_for_tma_partition = cute.group_modes(sC, 0, 2)

        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)

        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )

        return bSG_sC, bSG_gC

    @cute.jit
    def find_batch_from_q(
        self,
        token: Int32,
        seqlen_q_divmod: FastDivmodDivisor,
        mCuSeqlensQ: Optional[cute.Tensor],
    ) -> Int32:
        """Find batch index from q token (binary search for varlen, divmod otherwise)"""
        if const_expr(mCuSeqlensQ is not None):
            return get_batch_from_cu_tensor(token, mCuSeqlensQ)
        else:
            batch, _ = divmod(token, seqlen_q_divmod)
            return batch

    @staticmethod
    def _compute_grid(c, cta_tile_shape_mnk, cluster_shape_mn):
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        c_logical_shape = gc[(0, (None, None, None))].shape

        num_ctas_mnl = (
            cute.size(c_logical_shape[0]),
            cute.size(c_logical_shape[1]),
            cute.size(c_logical_shape[2]),
        )

        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            num_ctas_mnl, (*cluster_shape_mn, 1)
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
        return tile_sched_params, grid
