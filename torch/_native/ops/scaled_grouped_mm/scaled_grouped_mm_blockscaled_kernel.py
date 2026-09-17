from inspect import isclass

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ._clc_scheduler import ClcState, create_clc_pipeline, make_clc_problem_shape


class ClcGroupedGemmTileSchedulerHelper(utils.StaticPersistentGroupTileScheduler):
    @classmethod
    def create_for_clc(
        cls,
        group_count,
        tile_sched_params,
        cluster_tile_shape_mnk,
        problem_shape_mnkl,
    ):
        return cls(
            tile_sched_params,
            cutlass.Int32(1),  # num_persistent_clusters (unused)
            cutlass.Int32(0),  # _current_work_linear_idx (unused)
            (
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
            ),  # cta_id_in_cluster (unused)
            cutlass.Int32(0),  # _num_tiles_executed (unused)
            cluster_tile_shape_mnk,
            utils.create_initial_search_state(),
            group_count,
            problem_shape_mnkl,
        )

    def __new_from_mlir_values__(self, values):
        if len(values) < 11:
            raise ValueError("Length of mlir values extracted is incorrect.")
        new_num_persistent_clusters = cutlass.new_from_mlir_values(
            self.num_persistent_clusters, [values[0]]
        )
        new_current_work_linear_idx = cutlass.new_from_mlir_values(
            self._current_work_linear_idx, [values[1]]
        )
        new_cta_id_in_cluster = cutlass.new_from_mlir_values(
            self.cta_id_in_cluster, values[2:5]
        )
        new_num_tiles_executed = cutlass.new_from_mlir_values(
            self._num_tiles_executed, [values[5]]
        )
        search_state = cutlass.new_from_mlir_values(self.search_state, values[6:10])
        problem_shape_mnkl = cutlass.new_from_mlir_values(
            self.problem_shape_mnkl, [values[10]]
        )
        params = cutlass.new_from_mlir_values(self.params, values[11:])

        return ClcGroupedGemmTileSchedulerHelper(
            params,
            new_num_persistent_clusters,
            new_current_work_linear_idx,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
            self.cluster_tile_shape_mnk,
            search_state,
            self.group_count,
            problem_shape_mnkl,
        )

    def delinearize_z(self, cta_tile_coord, problem_shape_mnkl):
        linear_idx = cta_tile_coord[2]
        _found, group_idx, problem_mnkl = self._group_search_and_load_problem_shape(
            linear_idx,
            problem_shape_mnkl,
            self.search_state.start_group_idx,
            self.search_state.tile_count_prev_group,
        )
        cluster_tile_idx_in_current_group = (
            linear_idx - self.search_state.tile_count_prev_group
        )
        cluster_count_m, cluster_count_n, cluster_count_k = cute.ceil_div(
            (problem_mnkl[0], problem_mnkl[1], problem_mnkl[2]),
            (
                self.cluster_tile_shape_mnk[0],
                self.cluster_tile_shape_mnk[1],
                self.cluster_tile_shape_mnk[2],
            ),
        )
        cta_tile_idx_m, cta_tile_idx_n = self._compute_cta_tile_coord(
            cluster_tile_idx_in_current_group,
            cta_tile_coord,
            cluster_count_m,
            cluster_count_n,
        )
        return utils.GroupSearchResult(
            group_idx,
            cta_tile_idx_m,
            cta_tile_idx_n,
            problem_mnkl[0],
            problem_mnkl[1],
            problem_mnkl[2],
            cluster_count_k,
        )

    def delinearize_uniform_mn(self, cta_tile_coord, problem_shape_mnkl):
        group_idx = cta_tile_coord[2]
        problem_mnkl = self._get_problem_for_group(problem_shape_mnkl, group_idx)
        cluster_count_k = cute.ceil_div(problem_mnkl[2], self.cluster_tile_shape_mnk[2])
        return utils.GroupSearchResult(
            group_idx,
            cta_tile_coord[0],
            cta_tile_coord[1],
            problem_mnkl[0],
            problem_mnkl[1],
            problem_mnkl[2],
            cluster_count_k,
        )


class Sm100GroupedBlockScaledGemmKernel:
    AB_TMA_LOAD_UNROLL = 4
    NUM_WARPS_PER_CTA = 8
    GENERIC_REG_REQUIREMENT = 136
    ACCUM_REG_REQUIREMENT = 168
    # CLC dispatch swizzle; currently mostly inert because work is along L.
    CLC_SWIZZLE_SIZE = 1

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        transpose_ab: bool = False,
        uniform_mn_groups: bool = False,
    ):
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        self.transpose_ab = transpose_ab
        self.uniform_mn_groups = uniform_mn_groups
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.tensormap_update_mode = utils.TensorMapUpdateMode.SMEM

        self.occupancy = 1
        # Warp roles; epi_load is reserved but inactive.
        #   - mma warp:           0
        #   - scheduler warp:     1  (CLC producer)
        #   - mainloop load warp: 2  (A/B/SF tensormap update/fence)
        #   - epilogue load warp: 3  (reserved, inactive)
        #   - epilogue warps:     4..7  (epilog[0] updates/fences C)
        self.mma_warp_id = 0
        self.scheduler_warp_id = 1
        self.mainloop_load_warp_id = 2
        self.epilogue_load_warp_id = 3
        self.epilog_warp_id = (
            4,
            5,
            6,
            7,
        )
        self.tensormap_worker_warp_id = (
            self.scheduler_warp_id,
            self.mainloop_load_warp_id,
        )
        _highest_specialized_warp_id = max(*self.epilog_warp_id)
        if self.NUM_WARPS_PER_CTA <= _highest_specialized_warp_id:
            raise ValueError(
                "NUM_WARPS_PER_CTA must be greater than the highest specialized "
                f"warp id ({_highest_specialized_warp_id}), got "
                f"{self.NUM_WARPS_PER_CTA}"
            )
        self.threads_per_cta = 32 * self.NUM_WARPS_PER_CTA
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        # A/B descriptor init handshake.
        self.tensormap_ab_init_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * 2,  # MMA + mainload
        )
        # Non-uniform metadata handoff.
        self.tile_metadata_ready_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=32 * (2 + len(self.epilog_warp_id)),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

    def _setup_attributes(self):
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_tile_shape_mnk = tuple(
            x * y for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn, 1))
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Single-stage CLC: issue, wait, consume.
        self.num_clc_stage = 1

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        mbar_smem_bytes = self._get_mbar_smem_bytes(
            num_acc_stage=self.num_acc_stage,
            num_ab_stage=self.num_ab_stage,
            num_c_stage=self.num_c_stage,
        )

        tensormap_smem_bytes = (
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap
            * Sm100GroupedBlockScaledGemmKernel.num_tensormaps
        )
        if (
            mbar_smem_bytes
            + tensormap_smem_bytes
            + Sm100GroupedBlockScaledGemmKernel.tensor_memory_management_bytes
            > self.reserved_smem_bytes
        ):
            raise ValueError(
                f"smem consumption for mbar and tensormap {mbar_smem_bytes + tensormap_smem_bytes} exceeds the "
                f"reserved smem bytes {self.reserved_smem_bytes}"
            )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_c: cute.Tensor,
        initial_sfa: cute.Tensor,
        initial_sfb: cute.Tensor,
        tensor_addr_global_scale: cute.Tensor,
        group_count: cutlass.Int32,
        problem_shape_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_address_sfasfb: cute.Tensor,
        estimate_total_num_clusters: cutlass.Int32,
        total_num_clusters: cute.Tensor,
        tensormap_cute_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        tensor_a = initial_a
        tensor_b = initial_b
        tensor_sfa = initial_sfa
        tensor_sfb = initial_sfb
        if cutlass.const_expr(self.transpose_ab):
            tensor_a = initial_b
            tensor_b = initial_a
            tensor_sfa = initial_sfb
            tensor_sfb = initial_sfa

        self.a_dtype = tensor_a.element_type
        self.b_dtype = tensor_b.element_type
        self.sf_dtype = tensor_sfa.element_type
        self.c_dtype = initial_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(tensor_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(tensor_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(initial_c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        # Keep blocked MKL scale pointers while reusing A/B shapes.
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            tensor_a.shape, self.sf_vec_size
        )
        tensor_sfa = cute.make_tensor(tensor_sfa.iterator, sfa_layout)

        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            tensor_b.shape, self.sf_vec_size
        )
        tensor_sfb = cute.make_tensor(tensor_sfb.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            tensor_a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            tensor_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            tensor_sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            tensor_sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            initial_c,
            epi_smem_layout,
            self.epi_tile,
        )

        direct_problem_shape_mnl = (0, 0, 0)
        if cutlass.const_expr(self.uniform_mn_groups):
            cluster_count_m, cluster_count_n = cute.ceil_div(
                (tensor_a.shape[0], tensor_b.shape[0]),
                self.cluster_tile_shape_mnk[:2],
            )
            direct_problem_shape_mnl = (
                cluster_count_m * self.cluster_shape_mn[0],
                cluster_count_n * self.cluster_shape_mn[1],
                group_count,
            )
        if cutlass.const_expr(self.uniform_mn_groups):
            estimate_tile_sched_params = make_clc_problem_shape(
                self.cluster_shape_mn,
                estimate_total_num_clusters,
                problem_shape_ntile_mnl=direct_problem_shape_mnl,
                swizzle_size=self.CLC_SWIZZLE_SIZE,
            )
        else:
            estimate_tile_sched_params = make_clc_problem_shape(
                self.cluster_shape_mn,
                estimate_total_num_clusters,
                swizzle_size=self.CLC_SWIZZLE_SIZE,
            )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(
            estimate_tile_sched_params
        )

        self.buffer_align_bytes = 1024
        self.size_tensormap_in_i64 = (
            Sm100GroupedBlockScaledGemmKernel.num_tensormaps
            * Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap
            // 8
        )

        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.MemRange[
                cutlass.Int64, self.size_tensormap_in_i64
            ]
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cute.struct.MemRange[cutlass.Int32, 1]
            # CLC response: (m_idx, n_idx, l_idx, valid).
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_clc_stage * 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, self.num_clc_stage * 4]
            # Non-uniform group metadata:
            #   [0] group_idx
            #   [1] cta_tile_idx_m
            #   [2] cta_tile_idx_n
            #   [3] cta_tile_count_k  (k-tile count for the current group)
            #   [4] problem_shape_m
            #   [5] problem_shape_n
            #   [6] problem_shape_k
            tile_meta: cute.struct.MemRange[cutlass.Int32, 7]
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            total_num_clusters,
            tensor_addr_global_scale,
            group_count,
            problem_shape_mnkl,
            strides_abc,
            tensor_address_abc,
            tensor_address_sfasfb,
            tensormap_cute_tensor,
            direct_problem_shape_mnl,
        ).launch(
            grid=grid,
            block=[32 * Sm100GroupedBlockScaledGemmKernel.NUM_WARPS_PER_CTA, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # pyrefly: ignore [missing-attribute]
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.Layout | cute.ComposedLayout,
        epi_tile: cute.Tile,
        total_num_clusters: cute.Tensor,
        tensor_addr_global_scale: cute.Tensor,
        group_count: cutlass.Int32,
        problem_sizes_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        ptrs_abc: cute.Tensor,
        ptrs_sfasfb: cute.Tensor,
        tensormaps: cute.Tensor,
        direct_problem_shape_mnl,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == self.scheduler_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfb)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()
        tile_sched_params = self._compute_tile_sched(
            total_num_clusters[0], self.cluster_shape_mn
        )

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
        tensormap_a_smem_ptr = tensormap_smem_ptr
        tensormap_b_smem_ptr = (
            tensormap_a_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )
        tensormap_sfa_smem_ptr = (
            tensormap_b_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )
        tensormap_sfb_smem_ptr = (
            tensormap_sfa_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )
        tensormap_c_smem_ptr = (
            tensormap_sfb_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf.data_ptr()

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = 2 if use_2cta_instrs else 1
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        if use_2cta_instrs:
            if warp_idx == self.scheduler_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads
                    )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        grid_dim = cute.arch.grid_dim()
        tensormap_workspace_idx = (
            bidz * grid_dim[1] * grid_dim[0] + bidy * grid_dim[0] + bidx
        )

        tensormap_manager = utils.TensorMapManager(
            utils.TensorMapUpdateMode.SMEM,
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap,
        )
        tensormap_a_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 0, None)].iterator
        )
        tensormap_b_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 1, None)].iterator
        )
        tensormap_sfa_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 2, None)].iterator
        )
        tensormap_sfb_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 3, None)].iterator
        )
        tensormap_c_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 4, None)].iterator
        )

        # Scheduler produces CLC work; consumers read it.
        clc_response_ptr = storage.clc_response.data_ptr()
        clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()
        # Non-uniform path metadata handoff.
        tile_meta_smem = cute.make_tensor(
            storage.tile_meta.data_ptr(), cute.make_layout(7)
        )
        num_clc_consumer_warps_per_cta = len(
            (
                self.mma_warp_id,
                self.scheduler_warp_id,
                self.mainloop_load_warp_id,
                *self.epilog_warp_id,
            )
        )
        cluster_size = cute.size(self.cluster_shape_mn)
        clc_pipeline = create_clc_pipeline(
            barrier_storage=clc_mbar_ptr,
            num_stages=self.num_clc_stage,
            num_consumer_warps=num_clc_consumer_warps_per_cta,
            cluster_size=cluster_size,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        if cutlass.const_expr(self.uniform_mn_groups):
            clc_problem_shape = make_clc_problem_shape(
                self.cluster_shape_mn,
                total_num_clusters[0],
                problem_shape_ntile_mnl=direct_problem_shape_mnl,
                swizzle_size=self.CLC_SWIZZLE_SIZE,
            )
        else:
            clc_problem_shape = make_clc_problem_shape(
                self.cluster_shape_mn,
                total_num_clusters[0],
                swizzle_size=self.CLC_SWIZZLE_SIZE,
            )
        clc_state = ClcState.create(
            hw_scheduler=utils.ClcDynamicPersistentTileScheduler.create(
                clc_problem_shape,
                cute.arch.block_idx(),
                grid_dim,
                clc_response_ptr,
            ),
            pipeline=clc_pipeline,
            consumer_state=pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_clc_stage
            ),
            producer_state=pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_clc_stage
            ),
        )
        is_leader_cluster_cta = cta_rank_in_cluster == 0

        if warp_idx == self.scheduler_warp_id:
            cute.arch.setmaxregister_decrease(self.GENERIC_REG_REQUIREMENT)
            work_tile = clc_state.initial_work_tile_info(total_num_clusters[0])

            while work_tile.is_valid_tile:
                if is_leader_cluster_cta:
                    clc_state.prefetch_next_work()
                clc_state.consumer_wait()
                work_tile = clc_state.get_current_work(total_num_clusters[0])
                clc_state.consumer_release()

            if is_leader_cluster_cta:
                clc_state.producer_tail()

        if warp_idx == self.mainloop_load_warp_id:
            cute.arch.setmaxregister_decrease(self.GENERIC_REG_REQUIREMENT)
            group_gemm_ts_helper = ClcGroupedGemmTileSchedulerHelper.create_for_clc(
                group_count,
                tile_sched_params,
                self.cluster_tile_shape_mnk,
                problem_sizes_mnkl,
            )
            tensormap_init_done = cutlass.Boolean(False)
            last_group_idx = cutlass.Int32(-1)

            work_tile = clc_state.initial_work_tile_info(total_num_clusters[0])

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            # Kernel-wide descriptor pointers.
            tma_desc_a = tensormap_manager.get_tensormap_ptr(
                tensormap_a_gmem_ptr, cute.AddressSpace.generic
            )
            tma_desc_b = tensormap_manager.get_tensormap_ptr(
                tensormap_b_gmem_ptr, cute.AddressSpace.generic
            )
            tma_desc_sfa = tensormap_manager.get_tensormap_ptr(
                tensormap_sfa_gmem_ptr, cute.AddressSpace.generic
            )
            tma_desc_sfb = tensormap_manager.get_tensormap_ptr(
                tensormap_sfb_gmem_ptr, cute.AddressSpace.generic
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                if cutlass.const_expr(self.uniform_mn_groups):
                    grouped_gemm_cta_tile_info = (
                        group_gemm_ts_helper.delinearize_uniform_mn(
                            cur_tile_coord, problem_sizes_mnkl
                        )
                    )
                else:
                    grouped_gemm_cta_tile_info = group_gemm_ts_helper.delinearize_z(
                        cur_tile_coord, problem_sizes_mnkl
                    )
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                # Keep barrier order aligned with MMA on the first tile.
                if not tensormap_init_done:
                    self.tensormap_ab_init_barrier.arrive_and_wait()
                    tensormap_init_done = True
                if cutlass.const_expr(not self.uniform_mn_groups):
                    with cute.arch.elect_one():
                        tile_meta_smem[0] = cur_group_idx
                        tile_meta_smem[1] = grouped_gemm_cta_tile_info.cta_tile_idx_m
                        tile_meta_smem[2] = grouped_gemm_cta_tile_info.cta_tile_idx_n
                        tile_meta_smem[3] = cur_k_tile_cnt
                        tile_meta_smem[4] = grouped_gemm_cta_tile_info.problem_shape_m
                        tile_meta_smem[5] = grouped_gemm_cta_tile_info.problem_shape_n
                        tile_meta_smem[6] = grouped_gemm_cta_tile_info.problem_shape_k
                    self.tile_metadata_ready_barrier.arrive_and_wait()
                is_group_changed = cur_group_idx != last_group_idx
                if is_group_changed:
                    problem_shape_mnk = (
                        grouped_gemm_cta_tile_info.problem_shape_m,
                        grouped_gemm_cta_tile_info.problem_shape_n,
                        grouped_gemm_cta_tile_info.problem_shape_k,
                    )
                    real_tensor_a = self.make_tensor_abc_for_tensormap_update(
                        cur_group_idx,
                        self.a_dtype,
                        problem_shape_mnk,
                        strides_abc,
                        ptrs_abc,
                        0,
                    )
                    real_tensor_b = self.make_tensor_abc_for_tensormap_update(
                        cur_group_idx,
                        self.b_dtype,
                        problem_shape_mnk,
                        strides_abc,
                        ptrs_abc,
                        1,
                    )
                    real_tensor_sfa = self.make_tensor_sfasfb_for_tensormap_update(
                        cur_group_idx,
                        self.sf_dtype,
                        problem_shape_mnk,
                        ptrs_sfasfb,
                        0,
                    )
                    real_tensor_sfb = self.make_tensor_sfasfb_for_tensormap_update(
                        cur_group_idx,
                        self.sf_dtype,
                        problem_shape_mnk,
                        ptrs_sfasfb,
                        1,
                    )
                    tensormap_manager.update_tensormap(
                        (
                            real_tensor_a,
                            real_tensor_b,
                            real_tensor_sfa,
                            real_tensor_sfb,
                        ),
                        (tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb),
                        (
                            tensormap_a_gmem_ptr,
                            tensormap_b_gmem_ptr,
                            tensormap_sfa_gmem_ptr,
                            tensormap_sfb_gmem_ptr,
                        ),
                        self.mainloop_load_warp_id,
                        (
                            tensormap_a_smem_ptr,
                            tensormap_b_smem_ptr,
                            tensormap_sfa_smem_ptr,
                            tensormap_sfb_smem_ptr,
                        ),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_a_gmem_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_b_gmem_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_sfa_gmem_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_sfb_gmem_ptr)

                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m
                    // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )

                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < cur_k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )

                for k_tile in cutlass.range(
                    0, cur_k_tile_cnt, 1, unroll=self.AB_TMA_LOAD_UNROLL
                ):
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # Share one warp-uniform broadcast.
                    ab_state_count = cute.arch.make_warp_uniform(
                        ab_producer_state.count
                    )
                    ab_state_index = cute.arch.make_warp_uniform(
                        ab_producer_state.index
                    )
                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)

                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_state_count)],
                        tAsA[(None, ab_state_index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                        tma_desc_ptr=tma_desc_a,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_state_count)],
                        tBsB[(None, ab_state_index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                        tma_desc_ptr=tma_desc_b,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_state_count)],
                        tAsSFA[(None, ab_state_index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfa_full_mcast_mask,
                        tma_desc_ptr=tma_desc_sfa,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_state_count)],
                        tBsSFB[(None, ab_state_index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                        tma_desc_ptr=tma_desc_sfb,
                    )

                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < cur_k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )
                last_group_idx = cur_group_idx

                clc_state.consumer_wait()
                work_tile = clc_state.get_current_work(total_num_clusters[0])
                clc_state.consumer_release()

            # Avoid deadlock when this CTA has no valid work.
            if not tensormap_init_done:
                self.tensormap_ab_init_barrier.arrive_and_wait()

            ab_pipeline.producer_tail(ab_producer_state)

        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.GENERIC_REG_REQUIREMENT)
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_a, tensormap_a_smem_ptr, self.mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_b, tensormap_b_smem_ptr, self.mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_sfa, tensormap_sfa_smem_ptr, self.mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_sfb, tensormap_sfb_smem_ptr, self.mma_warp_id
            )
            self.tensormap_ab_init_barrier.arrive_and_wait()

            self.tmem_alloc_barrier.arrive_and_wait()

            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            group_gemm_ts_helper = ClcGroupedGemmTileSchedulerHelper.create_for_clc(
                group_count,
                tile_sched_params,
                self.cluster_tile_shape_mnk,
                problem_sizes_mnkl,
            )
            work_tile = clc_state.initial_work_tile_info(total_num_clusters[0])
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                if cutlass.const_expr(self.uniform_mn_groups):
                    grouped_gemm_cta_tile_info = (
                        group_gemm_ts_helper.delinearize_uniform_mn(
                            work_tile.tile_idx, problem_sizes_mnkl
                        )
                    )
                    cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                else:
                    self.tile_metadata_ready_barrier.arrive_and_wait()
                    cur_k_tile_cnt = tile_meta_smem[3]

                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < cur_k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in range(cur_k_tile_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)

                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < cur_k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                clc_state.consumer_wait()
                work_tile = clc_state.get_current_work(total_num_clusters[0])
                clc_state.consumer_release()

            acc_pipeline.producer_tail(acc_producer_state)

        if warp_idx >= self.epilog_warp_id[0] and warp_idx <= self.epilog_warp_id[-1]:
            cute.arch.setmaxregister_increase(self.ACCUM_REG_REQUIREMENT)
            # First C update drains the async descriptor init.
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_c,
                tensormap_c_smem_ptr,
                self.epilog_warp_id[0],
            )
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            self.tmem_alloc_barrier.arrive_and_wait()

            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx % (32 * len(self.epilog_warp_id))
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
                )
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            tma_atom_c, bSG_sC, bSG_gC_partitioned = (
                self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                )
            )

            group_gemm_ts_helper = ClcGroupedGemmTileSchedulerHelper.create_for_clc(
                group_count,
                tile_sched_params,
                self.cluster_tile_shape_mnk,
                problem_sizes_mnkl,
            )
            work_tile = clc_state.initial_work_tile_info(total_num_clusters[0])
            # Used for c_buffer rotation.
            epilog_tile_count = cutlass.Int32(0)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32,
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            # Kernel-wide descriptor pointer.
            tma_desc_c = tensormap_manager.get_tensormap_ptr(
                tensormap_c_gmem_ptr, cute.AddressSpace.generic
            )
            # epilog[0] updates the C descriptor.
            last_group_idx = cutlass.Int32(-1)

            while work_tile.is_valid_tile:
                if cutlass.const_expr(self.uniform_mn_groups):
                    grouped_gemm_cta_tile_info = (
                        group_gemm_ts_helper.delinearize_uniform_mn(
                            work_tile.tile_idx, problem_sizes_mnkl
                        )
                    )
                    cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                    cta_tile_idx_m = grouped_gemm_cta_tile_info.cta_tile_idx_m
                    cta_tile_idx_n = grouped_gemm_cta_tile_info.cta_tile_idx_n
                    problem_shape_m = grouped_gemm_cta_tile_info.problem_shape_m
                    problem_shape_n = grouped_gemm_cta_tile_info.problem_shape_n
                    problem_shape_k = grouped_gemm_cta_tile_info.problem_shape_k
                else:
                    # tile_meta is single-buffered.
                    self.tile_metadata_ready_barrier.arrive_and_wait()
                    cur_group_idx = tile_meta_smem[0]
                    cta_tile_idx_m = tile_meta_smem[1]
                    cta_tile_idx_n = tile_meta_smem[2]
                    problem_shape_m = tile_meta_smem[4]
                    problem_shape_n = tile_meta_smem[5]
                    problem_shape_k = tile_meta_smem[6]
                is_group_changed = cur_group_idx != last_group_idx
                if is_group_changed and warp_idx == self.epilog_warp_id[0]:
                    # Inline C tensormap update.
                    real_tensor_c = self.make_tensor_abc_for_tensormap_update(
                        cur_group_idx,
                        self.c_dtype,
                        (problem_shape_m, problem_shape_n, problem_shape_k),
                        strides_abc,
                        ptrs_abc,
                        2,
                    )
                    tensormap_manager.update_tensormap(
                        (real_tensor_c,),
                        (tma_atom_c,),
                        (tensormap_c_gmem_ptr,),
                        self.epilog_warp_id[0],
                        (tensormap_c_smem_ptr,),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_c_gmem_ptr)
                # Ensure C descriptor update is visible before TMA
                # store.
                self.epilog_sync_barrier.arrive_and_wait()

                mma_tile_coord_mnl = (
                    cta_tile_idx_m // cute.size(tiled_mma.thr_id.shape),
                    cta_tile_idx_n,
                    0,
                )
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]

                if warp_idx == self.epilog_warp_id[0]:
                    acc_pipeline.consumer_wait(acc_consumer_state)
                self.epilog_sync_barrier.arrive_and_wait()

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = epilog_tile_count * subtile_cnt
                global_scale = self.load_global_scale_for_group(
                    cur_group_idx, tensor_addr_global_scale
                )
                for subtile_idx in cutlass.range(subtile_cnt, unroll_full=True):
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    acc_vec = acc_vec * global_scale
                    tRS_rC.store(acc_vec.to(self.c_dtype))

                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # Make shared-memory store visible to TMA store.
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, subtile_idx)],
                            tma_desc_ptr=tma_desc_c,
                        )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()
                if warp_idx == self.epilog_warp_id[0]:
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                self.epilog_sync_barrier.arrive_and_wait()
                acc_consumer_state.advance()

                epilog_tile_count += 1
                last_group_idx = cur_group_idx
                clc_state.consumer_wait()
                work_tile = clc_state.get_current_work(total_num_clusters[0])
                clc_state.consumer_release()

            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            self.epilog_sync_barrier.arrive_and_wait()
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )
            c_pipeline.producer_tail()

    @cute.jit
    def make_tensor_abc_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_index: int,
    ):
        ptr_i64 = tensor_address_abc[(group_idx, tensor_index)]
        if cutlass.const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16
        )

        strides_tensor_gmem = strides_abc[(group_idx, tensor_index, None)]
        strides_tensor_reg = cute.make_rmem_tensor(
            cute.make_layout(2),
            strides_abc.element_type,
        )
        cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
        stride_mn = strides_tensor_reg[0]
        stride_k = strides_tensor_reg[1]
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int64(0)

        if cutlass.const_expr(tensor_index == 0):
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        elif cutlass.const_expr(tensor_index == 1):
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((n, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        else:
            m = problem_shape_mnk[0]
            n = problem_shape_mnk[1]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, n, c1), stride=(stride_mn, stride_k, c0)),
            )

    @cute.jit
    def make_tensor_sfasfb_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        tensor_address_sfasfb: cute.Tensor,
        tensor_index: int,
    ):
        ptr_i64 = tensor_address_sfasfb[(group_idx, tensor_index)]
        if cutlass.const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16
        )

        c1 = cutlass.Int32(1)
        if cutlass.const_expr(tensor_index == 0):
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (m, k, c1), self.sf_vec_size
            )
            return cute.make_tensor(
                tensor_gmem_ptr,
                sfa_layout,
            )
        else:
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (n, k, c1), self.sf_vec_size
            )
            return cute.make_tensor(
                tensor_gmem_ptr,
                sfb_layout,
            )

    @cute.jit
    def load_global_scale_for_group(
        self,
        group_idx: cutlass.Int32,
        tensor_addr_global_scale: cute.Tensor,
    ):
        ptr_i64 = tensor_addr_global_scale[group_idx]
        scale_gmem_ptr = cute.make_ptr(
            cutlass.Float32,
            ptr_i64,
            cute.AddressSpace.gmem,
            assumed_align=4,
        )
        scale_tensor = cute.make_tensor(
            scale_gmem_ptr,
            cute.make_layout((cutlass.Int32(1),), stride=(cutlass.Int64(1),)),
        )
        return scale_tensor[0]

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: cutlass.Boolean | bool,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: cute.CopyAtom | cute.TiledCopy,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int, int]:
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        num_c_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_tile_sched(
        total_num_clusters: int,
        cluster_shape_mn: tuple[int, int],
    ) -> utils.PersistentTileSchedulerParams:
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )

        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, (*cluster_shape_mn, 1)
        )
        return tile_sched_params

    @staticmethod
    def _get_mbar_smem_bytes(**kwargs_stages: int) -> int:
        num_barriers_per_stage = 2
        num_bytes_per_barrier = 8
        mbar_smem_consumption = sum(
            [
                num_barriers_per_stage * num_bytes_per_barrier * stage
                for stage in kwargs_stages.values()
            ]
        )
        return mbar_smem_consumption

    reserved_smem_bytes = 1024
    bytes_per_tensormap = 128
    num_tensormaps = 5
    tensor_memory_management_bytes = 12


__all__ = ["Sm100GroupedBlockScaledGemmKernel"]
