# Modified from CUTLASS example file, original copyright:
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Callable, Optional, Tuple, Union
import cuda.bindings.driver as cuda

import cutlass
from cutlass import Int32, const_expr
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
from cutlass.utils.gemm.sm100 import (
    transform_partitioned_tensor_layout,
    epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition,
)
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05

from .utils import get_batch_from_cu_tensor


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    """
    Get a pointer to an element at the specified coordinate in a tensor.

    Args:
        x: The tensor (typically a shared memory tensor)
        coord: The coordinate tuple, can be hierarchical like (row, (col, cluster_idx))

    Returns:
        Pointer to the element at the specified coordinate
    """
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


class dKGemmKernel:
    def __init__(
        self,
        topk: int,
        heads: int,
        dim: int,
        varlen: bool,
    ):
        self.varlen = varlen
        self.topk = topk
        self.heads = heads
        self.dim = dim
        # A operand dS'^T: (total_q, heads, topk), topk-major
        self.ab_dtype = cutlass.BFloat16
        self.a_major_mode = cute.nvgpu.OperandMajorMode.MN
        # B operand Q: (total_q, heads, dim), dim-major
        self.b_major_mode = cute.nvgpu.OperandMajorMode.MN
        # Index operand I: (total_q, seqlen_k)
        self.idx_dtype = cutlass.Int32

        # Output dKaccum: (total_q, seqlen_k, dim), dim-major
        self.c_dtype = cutlass.Float32
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        self.acc_dtype = cutlass.Float32

        if self.topk % 256 == 1:
            # The kernel schedule requires 2CTA instructions with this tile shape
            self.cluster_shape_mn = (2, 1)
            self.use_2cta_instrs = True
        else:
            self.cluster_shape_mn = (1, 1)
            self.use_2cta_instrs = False
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.atom_thr_size = 2 if self.use_2cta_instrs else 1
        self.cta_tile_shape_mnk = (128, 64, 1)
        self.mma_tiler_dK = (
            self.cta_tile_shape_mnk[0] * self.atom_thr_size,
            self.cta_tile_shape_mnk[1],
            1,
        )
        self.arch = "sm_100"

        self.occupancy = 1
        # Set specialized warp ids
        self.epilogue_warp_id = [0, 1, 2, 3]
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        self.threads_per_cta = 32 * len(
            [
                self.mma_warp_id,
                self.tma_warp_id,
                self.sched_warp_id,
                *self.epilogue_warp_id,
            ]
        )
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.I_load_bar_id = 3

    @cute.jit
    def __call__(
        self,
        dS: cute.Tensor,  # (batch, seqlen_q, heads, topk) or (total_q, heads, topk)
        I: cute.Tensor,  # (batch, seqlen_q, heads, topk) or (total_q, heads, topk)
        Q: cute.Tensor,  # (batch, seqlen_q, heads, dim) or (total_q, heads, dim)
        dKaccum: cute.Tensor,  # (batch, seqlen_k, dim) or (total_k, dim)
        cuSeqlensQ: Optional[cute.Tensor],  # (batch + 1,)
        cuSeqlensK: Optional[cute.Tensor],  # (batch + 1,)
        stream: cuda.CUstream,
    ):
        if const_expr(self.ab_dtype != dS.element_type):
            raise TypeError(f"Type must match: {self.ab_dtype} != {dS.element_type}")
        if const_expr(self.ab_dtype != Q.element_type):
            raise TypeError(f"Type must match: {self.ab_dtype} != {Q.element_type}")
        if const_expr(self.c_dtype != dKaccum.element_type):
            raise TypeError(f"Type must match: {self.c_dtype} != {dKaccum.element_type}")

        if const_expr(self.varlen):
            assert cuSeqlensQ is not None
            assert cuSeqlensK is not None

        # For non-varlen, group batch and seqlen modes into token mode
        if const_expr(not self.varlen):
            batch_divmod = cute.FastDivmodDivisor(dS.shape[0])
            dS = cute.group_modes(dS, 0, 2)
            I = cute.group_modes(I, 0, 2)
            Q = cute.group_modes(Q, 0, 2)
            dKaccum = cute.group_modes(dKaccum, 0, 2)
        else:
            batch_divmod = cute.FastDivmodDivisor(0)

        # Permute everything to (_, heads, tokens) for MMA
        dS_mkl = cute.make_tensor(dS.iterator, cute.select(dS.layout, [2, 1, 0]))
        I_ml = cute.make_tensor(I.iterator, cute.select(I.layout, [1, 0]))
        Q_nkl = cute.make_tensor(Q.iterator, cute.select(Q.layout, [2, 1, 0]))
        dKaccum_nl = cute.make_tensor(dKaccum.iterator, cute.select(dKaccum.layout, [1, 0]))

        # Configure tiled mma
        self.tiled_mma_dK = utils.sm100.make_trivial_tiled_mma(
            self.ab_dtype,
            self.ab_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dK[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(self.tiled_mma_dK.shape_mnk, mode=[2])
        mma_inst_tile_k = 4 if self.heads == 64 else 8
        self.mma_tiler_dK = (
            self.mma_tiler_dK[0],
            self.mma_tiler_dK[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk_dK = (
            self.mma_tiler_dK[0] // self.atom_thr_size,
            self.mma_tiler_dK[1],
            self.mma_tiler_dK[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self.atom_thr_size,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile shape for TMA store
        # self.epi_tile_dK = utils.sm100.compute_epilogue_tile_shape(
        #     self.cta_tile_shape_mnk_dK,
        #     self.use_2cta_instrs,
        #     self.c_layout,
        #     self.c_dtype,
        # )
        self.epi_tile_dK = (cute.make_layout(128), cute.make_layout(32))
        self.epi_tile_dK_width = cute.size(self.epi_tile_dK[1].shape)

        self.dK_smem_layout = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile_dK, 1
        )

        smem_capacity = utils.get_smem_capacity_in_bytes()

        # TMEM pipeline stages
        # 64 TMEM columns per stage, 8 stages can fit in TMEM
        self.num_acc_stage = 8
        self.num_tmem_alloc_cols = 512

        # SMEM pipeline stages
        # TMA load Q (persists for the whole token)
        self.num_Q_stage = 1
        assert const_expr(self.mma_tiler_dK[1] * self.num_Q_stage == self.dim)
        Q_smem_layout_stage_one = utils.sm100.make_smem_layout_b(
            self.tiled_mma_dK, self.mma_tiler_dK, self.ab_dtype, 1
        )
        Q_bytes = cute.size_in_bytes(self.ab_dtype, Q_smem_layout_stage_one) * self.num_Q_stage

        # TMA store-reduce dK
        self.num_c_stage = 2
        dK_bytes_per_stage = cute.size_in_bytes(self.c_dtype, self.dK_smem_layout)
        dK_bytes = dK_bytes_per_stage * self.num_c_stage

        # cp.async load I
        self.num_I_stage = 2
        self.I_smem_layout_staged = cute.make_layout(
            (self.mma_tiler_dK[0] // self.atom_thr_size, self.num_I_stage)
        )
        I_bytes = cute.size_in_bytes(self.idx_dtype, self.I_smem_layout_staged)

        # TMA load dS
        dS_smem_layout_stage_one = utils.sm100.make_smem_layout_a(
            self.tiled_mma_dK, self.mma_tiler_dK, self.ab_dtype, 1
        )
        dS_bytes_per_stage = cute.size_in_bytes(self.ab_dtype, dS_smem_layout_stage_one)

        mbar_helpers_bytes = 1024

        # Increase dS stages to fill SMEM
        self.num_dS_stage = (
            smem_capacity // self.occupancy - (mbar_helpers_bytes + dK_bytes + Q_bytes + I_bytes)
        ) // dS_bytes_per_stage
        dS_bytes = self.num_dS_stage * dS_bytes_per_stage

        # Increase dK stages to fill remainder
        self.num_c_stage += (
            smem_capacity // self.occupancy
            - (mbar_helpers_bytes + dK_bytes + dS_bytes + Q_bytes + I_bytes)
        ) // dK_bytes_per_stage

        # Increase I stages to fill remainder

        # Compute shared memory layout
        self.dS_smem_layout_staged = utils.sm100.make_smem_layout_a(
            self.tiled_mma_dK, self.mma_tiler_dK, self.ab_dtype, self.num_dS_stage
        )
        self.Q_smem_layout_staged = utils.sm100.make_smem_layout_b(
            self.tiled_mma_dK, self.mma_tiler_dK, self.ab_dtype, self.num_Q_stage
        )
        self.dK_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile_dK, self.num_c_stage
        )

        # TMA load for dS
        dS_op = utils.sm100.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.tiled_mma_dK.thr_id
        )
        self.dS_smem_layout = cute.slice_(self.dS_smem_layout_staged, (None, None, None, 0))
        tma_atom_dS, tma_tensor_dS = cute.nvgpu.make_tiled_tma_atom_A(
            dS_op,
            dS_mkl,
            self.dS_smem_layout,
            self.mma_tiler_dK,
            self.tiled_mma_dK,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for Q
        Q_op = utils.sm100.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, self.tiled_mma_dK.thr_id
        )
        self.Q_smem_layout = cute.slice_(self.Q_smem_layout_staged, (None, None, None, 0))
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            Q_op,
            Q_nkl,
            self.Q_smem_layout,
            self.mma_tiler_dK,
            self.tiled_mma_dK,
            self.cluster_layout_vmnk.shape,
        )
        self.Q_load_bytes = (
            cute.size_in_bytes(self.ab_dtype, self.Q_smem_layout) * self.atom_thr_size
        )

        self.dS_load_bytes = (
            cute.size_in_bytes(self.ab_dtype, self.dS_smem_layout) * self.atom_thr_size
        )

        # coalesced store for dKaccum: 1 warp copies 1 epi tile row
        vector_width_dK = self.epi_tile_dK_width * self.c_dtype.width // cute.arch.WARP_SIZE
        copy_atom_dK = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.c_dtype,
            num_bits_per_copy=vector_width_dK,
        )
        vector_elts_dK = vector_width_dK // self.c_dtype.width
        thr_layout_dK = cute.make_layout((cute.arch.WARP_SIZE,))
        val_layout_dK = cute.make_layout((vector_elts_dK,))
        tiled_copy_dK = cute.make_tiled_copy_tv(copy_atom_dK, thr_layout_dK, val_layout_dK)

        # cp.async load I: 1 warp copies 32 values, 4 warps copy 128 values
        copy_atom_I = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            self.idx_dtype,
            num_bits_per_copy=32,
        )
        vI = cute.make_layout((1,))
        tI = cute.make_layout((128,))
        tiled_copy_I = cute.make_tiled_copy_tv(copy_atom_I, tI, vI)

        # Setup clc stage by default
        self.num_clc_stage = 1
        assert self.num_clc_stage == 1, "Only single-stage CLC pipeline is supported"

        # Response size is 4B * 4 elements
        self.num_clc_response_bytes = 16

        # Compute grid size and set up tile scheduler
        total_q = cute.size(dS_mkl.shape[2])
        cluster_shape_mnl = (*self.cluster_shape_mn, 1)
        num_ctas_mnl = (self.cluster_shape_mn[0] * total_q, self.cluster_shape_mn[1], 1)
        self.tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(self.tile_sched_params)

        # Define shared storage for kernel
        buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            dS_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_dS_stage * 2]
            Q_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_Q_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, 4]
            sI: cute.struct.Align[
                cute.struct.MemRange[self.idx_dtype, cute.cosize(self.I_smem_layout_staged)],
                buffer_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(self.dS_smem_layout_staged)],
                buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(self.Q_smem_layout_staged)],
                buffer_align_bytes,
            ]
            sdK: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.dK_smem_layout_staged)],
                buffer_align_bytes,
            ]

        # Launch the kernel synchronously
        self.kernel(
            self.tiled_mma_dK,
            tma_atom_dS,
            tma_tensor_dS,
            tma_atom_Q,
            tma_tensor_Q,
            tiled_copy_dK,
            dKaccum_nl,
            tiled_copy_I,
            I_ml,
            batch_divmod,
            cuSeqlensQ,
            cuSeqlensK,
            self.cluster_layout_vmnk,
            self.dS_smem_layout_staged,
            self.Q_smem_layout_staged,
            self.dK_smem_layout_staged,
            self.I_smem_layout_staged,
            self.epi_tile_dK,
            self.tile_sched_params,
            SharedStorage,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma_dK: cute.TiledMma,
        tma_atom_dS: cute.CopyAtom,
        mdS_mkl: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        mQ_nkl: Optional[cute.Tensor],
        tiled_copy_dK: cute.TiledCopy,
        mdKaccum_nl: Optional[cute.Tensor],
        tiled_copy_I: cute.TiledCopy,
        mI_ml: cute.Tensor,
        batch_divmod: cute.FastDivmodDivisor,
        cuSeqlensQ: Optional[cute.Tensor],
        cuSeqlensK: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        dS_smem_layout_staged: cute.ComposedLayout,
        Q_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        dK_smem_layout_staged: Union[cute.ComposedLayout, cute.Layout],
        I_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile_dK: cute.Tile,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_dS)
            cpasync.prefetch_descriptor(tma_atom_Q)

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % self.atom_thr_size
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_first_cta_in_cluster = cta_rank_in_cluster == 0
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop ab_pipeline (barrier) and states
        dS_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        dS_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        dS_producer, dS_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dS_full_mbar_ptr.data_ptr(),
            num_stages=self.num_dS_stage,
            producer_group=dS_pipeline_producer_group,
            consumer_group=dS_pipeline_consumer_group,
            tx_count=self.dS_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        Q_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        Q_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        Q_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_full_mbar_ptr.data_ptr(),
            num_stages=self.num_Q_stage,
            producer_group=Q_pipeline_producer_group,
            consumer_group=Q_pipeline_consumer_group,
            tx_count=self.Q_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (2 if self.use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize clc_pipeline (barrier) and states
        clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(self.cluster_shape_mn)
        num_clc_consumer_threads = 32 * (1 + cluster_size * (1 + len(self.epilogue_warp_id) + 1))
        clc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_clc_consumer_threads
        )
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar_ptr.data_ptr(),
            num_stages=self.num_clc_stage,
            producer_group=clc_pipeline_producer_group,
            consumer_group=clc_pipeline_consumer_group,
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # Initial clc response pointer
        clc_response_ptr = storage.clc_response.data_ptr()

        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_clc_stage
        )

        #
        # Setup smem tensor A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        sdS = storage.sdS.get_tensor(
            dS_smem_layout_staged.outer,
            swizzle=dS_smem_layout_staged.inner,
        )
        # (MMA_M, STAGE)
        sI = storage.sI.get_tensor(I_smem_layout_staged)

        sQ = storage.sQ.get_tensor(
            Q_smem_layout_staged.outer,
            swizzle=Q_smem_layout_staged.inner,
        )
        sdK = storage.sdK.get_tensor(
            dK_smem_layout_staged.outer,
            swizzle=dK_smem_layout_staged.inner,
        )

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if const_expr(self.is_a_mcast or self.is_b_mcast or self.use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gdS_mkl = cute.local_tile(
            mdS_mkl, cute.slice_(self.mma_tiler_dK, (None, 0, None)), (None, None, None)
        )
        # (bM, RestM, RestL)
        gI_ml = cute.local_tile(mI_ml, cute.slice_(self.mma_tiler_dK, (None, 0, 0)), (None, None))
        # (bN, bK, RestN, RestK, RestL)
        gQ_nkl = cute.local_tile(
            mQ_nkl,
            cute.slice_(self.mma_tiler_dK, (0, None, None)),
            (None, None, None),
        )
        n_tile_cnt = cute.size(gQ_nkl, mode=[2])
        # (bN, RestN, RestL)
        gdKaccum_nl = cute.local_tile(
            mdKaccum_nl,
            cute.slice_(self.mma_tiler_dK, (0, None, 0)),
            (None, None),
        )

        m_tile_cnt = cute.size(gdS_mkl, mode=[2])
        k_tile_cnt = cute.size(gdS_mkl, mode=[3])
        assert const_expr(k_tile_cnt == 1)

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgQ = thr_mma_dK.partition_B(gQ_nkl)

        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgdKaccum_fake = thr_mma_dK.partition_C(
            cute.make_identity_tensor(
                cute.append(cute.slice_(self.mma_tiler_dK, (None, None, 0)), 1, up_to_rank=5)
            )
        )

        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgdS = thr_mma_dK.partition_A(gdS_mkl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsdS, tAgdS = cpasync.tma_partition(
            tma_atom_dS,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sdS, 0, 3),
            cute.group_modes(tCgdS, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBsQ, tBgQ = cpasync.tma_partition(
            tma_atom_Q,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgQ, 0, 3),
        )

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrdS = tiled_mma_dK.make_fragment_A(sdS)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrQ = tiled_mma_dK.make_fragment_B(sQ)
        # (MMA, MMA_M, MMA_N)
        acc_shape_dK = tiled_mma_dK.partition_shape_C(self.mma_tiler_dK[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtdK_fake = tiled_mma_dK.make_fragment_C(cute.append(acc_shape_dK, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        #
        # Construct the scheduler
        #
        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
        )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #

        if warp_idx == self.tma_warp_id:
            Q_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_Q_stage
            )
            #
            # Persistent tile scheduling loop
            #

            dS_producer.reset()
            peek_dS_empty_status = dS_producer.try_acquire()

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                total_q_coord = cur_tile_coord[0] // self.cluster_shape_mn[0]

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestM, RestK)
                tAgdS_slice = tAgdS[(None, None, None, total_q_coord)]
                # ((atom_v, rest_v), RestN, RestK)
                tBgQ_slice = tBgQ[(None, None, None, total_q_coord)]

                #
                # Tma load loop -- fully unrolled as all sizes are static
                #
                for m_tile in cutlass.range_constexpr(m_tile_cnt):
                    # Conditionally wait for AB buffer empty
                    dS_handle = dS_producer.acquire_and_advance(peek_dS_empty_status)

                    # TMA load dS
                    cute.copy(
                        tma_atom_dS,
                        tAgdS_slice[(None, m_tile, 0)],
                        tAsdS[(None, dS_handle.index)],
                        tma_bar_ptr=dS_handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )

                    peek_dS_empty_status = dS_producer.try_acquire()

                    if m_tile == 0:
                        # TMA load Q only on first m-tile
                        for n_tile in cutlass.range_constexpr(n_tile_cnt):
                            Q_pipeline.producer_acquire(Q_producer_state)
                            Q_load_barrier = Q_pipeline.producer_get_barrier(Q_producer_state)
                            cute.copy(
                                tma_atom_Q,
                                tBgQ_slice[(None, n_tile, 0)],
                                tBsQ[(None, n_tile)],
                                tma_bar_ptr=Q_load_barrier,
                                mcast_mask=b_full_mcast_mask,
                            )
                            Q_pipeline.producer_commit(Q_producer_state)
                            Q_producer_state.advance()

                #
                # Advance to next tile
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            #
            # Wait A/B buffer empty
            #
            dS_producer.tail()
            Q_pipeline.producer_tail(Q_producer_state)

        #
        # Sched warp
        #
        elif warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
            #
            # Persistent tile scheduling loop
            #
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, self.num_clc_stage
            )

            while work_tile.is_valid_tile:
                #
                # Advance to next tile
                #
                clc_pipeline.producer_acquire(clc_producer_state)
                mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                tile_sched.advance_to_next_work(mbarrier_addr)
                clc_producer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

        #
        # Specialized MMA warp
        #
        elif warp_idx == self.mma_warp_id:
            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtdK_base = cute.make_tensor(tmem_ptr, tCtdK_fake.layout)

            #
            # Persistent tile scheduling loop
            #
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            Q_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_Q_stage
            )

            # Peek (try_wait) AB buffer full for k_tile = 0
            dS_consumer.reset()
            peek_dS_full_status = cutlass.Boolean(1)
            if is_leader_cta:
                peek_dS_full_status = dS_consumer.try_wait()

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                total_q_coord = cur_tile_coord[0] // self.cluster_shape_mn[0]

                #
                # Mma mainloop
                #
                Q_consumer_state_wait = Q_consumer_state.clone()
                for m_tile in cutlass.range(m_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        dS_handle = dS_consumer.wait_and_advance(peek_dS_full_status)
                        for n_tile in cutlass.range_constexpr(n_tile_cnt):
                            tCtdK = tCtdK_base[(None, None, None, acc_producer_state.index)]
                            tiled_mma_dK.set(tcgen05.Field.ACCUMULATE, False)
                            acc_pipeline.producer_acquire(acc_producer_state)

                            if m_tile == 0:
                                Q_pipeline.consumer_wait(Q_consumer_state_wait)
                                Q_consumer_state_wait.advance()

                            num_kblocks = cute.size(tCrdS, mode=[2])
                            for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                                cute.gemm(
                                    tiled_mma_dK,
                                    tCtdK,
                                    tCrdS[(None, None, kblk_idx, dS_handle.index)],
                                    tCrQ[(None, None, kblk_idx, n_tile)],
                                    tCtdK,
                                )
                                # Enable accumulate on tCtdK after first kblock
                                tiled_mma_dK.set(tcgen05.Field.ACCUMULATE, True)

                            if m_tile == m_tile_cnt - 1:
                                Q_pipeline.consumer_release(Q_consumer_state)
                                Q_consumer_state.advance()

                            acc_pipeline.producer_commit(acc_producer_state)
                            acc_producer_state.advance()
                        # Async arrive AB buffer empty
                        dS_handle.release()

                        peek_dS_full_status = dS_consumer.try_wait()
                    else:
                        for n_tile in cutlass.range_constexpr(n_tile_cnt):
                            acc_producer_state.advance()

                #
                # Advance to next tile
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        elif warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtdK_base = cute.make_tensor(tmem_ptr, tCtdK_fake.layout)

            # Both gemms share accumulator and TMA store pipelines
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
            )

            I_load_barrier = pipeline.NamedBarrier(
                barrier_id=self.I_load_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )

            # (EPI_TOPK, REST_TOPK, TOKENS)
            gI_tile = cute.local_tile(
                gI_ml,
                (epi_tile_dK[0],),
                (0 if is_leader_cta else 1, None, None),
            )

            thr_copy_I = tiled_copy_I.get_slice(tidx)
            # (COPY_ATOM, EPI_TOPK, REST_TOPK, TOKENS)
            tIgI = thr_copy_I.partition_S(gI_tile)
            # (COPY_ATOM, EPI_TOPK, STAGE)
            tIsI = thr_copy_I.partition_D(sI)

            num_subtiles_executed = 0
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                token = cutlass.Int64(cur_tile_coord[0]) // self.cluster_shape_mn[0]

                # Get batch index from token_coord and cuSeqlensQ
                # Get seqlen_k offset from cuSeqlensK
                if const_expr(self.varlen):
                    batch = get_batch_from_cu_tensor(token, cuSeqlensQ)
                    seqlen_k_offset = cuSeqlensK[batch]
                else:
                    _, batch = divmod(token, batch_divmod)
                    seqlen_k_offset = Int32(0)  # unused

                sI_read_stage = 0
                sI_write_stage = 0

                # Prefetch load I
                I_load_barrier.arrive_and_wait()
                for m_tile in cutlass.range_constexpr(min(m_tile_cnt, self.num_I_stage - 1)):
                    cute.copy(
                        tiled_copy_I,
                        tIgI[(None, None, m_tile, token)],
                        tIsI[(None, None, sI_write_stage)],
                    )
                    cute.arch.cp_async_commit_group()
                    sI_write_stage = (sI_write_stage + 1) % self.num_I_stage

                for m_tile in cutlass.range(m_tile_cnt):
                    I_load_barrier.arrive_and_wait()
                    # cp.async load I
                    if m_tile < m_tile_cnt - 1:
                        cute.copy(
                            tiled_copy_I,
                            tIgI[(None, None, m_tile + 1, token)],
                            tIsI[(None, None, sI_write_stage)],
                        )
                    cute.arch.cp_async_commit_group()
                    sI_write_stage = (sI_write_stage + 1) % self.num_I_stage
                    cute.arch.cp_async_wait_group(self.num_I_stage - 1)
                    I_load_barrier.arrive_and_wait()

                    sI_tile = sI[(None, sI_read_stage)]

                    for n_tile in cutlass.range_constexpr(n_tile_cnt):
                        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
                        mma_tile_coord_mnl = (
                            m_tile,
                            n_tile,
                            token,
                        )
                        acc_consumer_state = self.epilogue_scatter_reduce(
                            tidx,
                            0 if is_leader_cta else 1,
                            tiled_copy_dK,
                            tCtdK_base,
                            sdK,
                            sI_tile,
                            gdKaccum_nl,
                            tCgdKaccum_fake,
                            batch,
                            seqlen_k_offset,
                            epi_tile_dK,
                            num_subtiles_executed,
                            mma_tile_coord_mnl,
                            acc_consumer_state,
                            acc_pipeline,
                        )
                        num_subtiles_executed += self.mma_tiler_dK[1] // cute.size(epi_tile_dK[1])
                    # Advance I consumer pipeline
                    sI_read_stage = (sI_read_stage + 1) % self.num_I_stage

                cute.arch.cp_async_wait_group(0)
                #
                # Advance to next tile
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            # # Wait for C store complete
            # dK_pipeline.producer_tail()
            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.jit
    def epilogue_scatter_reduce(
        self,
        epi_tidx: Int32,
        cta_idx: Int32,
        tiled_copy_c: cute.CopyAtom,
        tCtAcc_base: cute.Tensor,
        sC: cute.Tensor,
        sI_tile: cute.Tensor,
        gC_base: cute.Tensor,
        tCgC_fake: cute.Tensor,
        batch: Int32,
        seqlen_k_offset: Int32,
        epi_tile: cute.Tile,
        num_subtiles_executed: Int32,
        mma_tile_coord_mnl: Tuple[Int32, Int32, cutlass.Int64],
        acc_consumer_state: pipeline.PipelineState,
        acc_pipeline: pipeline.PipelineAsync,
    ) -> pipeline.PipelineState:
        warp_idx = cute.arch.make_warp_uniform(epi_tidx // 32)

        # Layout transformation for tCgC_base
        # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, REST_M, REST_N, REST_L)
        # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), REST_M, REST_N, REST_L)
        tCgC_fake = transform_partitioned_tensor_layout(tCgC_fake)

        # Layout transformation for tCtAcc_base
        # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, STAGE)
        # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), STAGE)
        tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
            self,
            epi_tidx,
            tCtAcc,
            tCgC_fake,
            epi_tile,
            self.use_2cta_instrs,
        )

        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
            self, tiled_copy_t2r, tTR_rC, epi_tidx, sC
        )

        # (EPI_TILE_N, EPI_N, RestN, RestL)
        gC_epi = cute.flat_divide(gC_base, (epi_tile[1],))
        # (EPI_TILE_N, EPI_N, SEQLEN_K)
        gC_epi = gC_epi[None, None, mma_tile_coord_mnl[1], None]
        # (EPI_TILE_N, MMA_M, STAGE)
        sC_epi = cute.make_tensor(
            sC.iterator,
            cute.select(sC.layout, [1, 0, 2]),
            # swizzle=sC.layout.inner,
        )

        thr_copy_c = tiled_copy_c.get_slice(epi_tidx % 32)
        # (COPY_ATOM_N, COPY_N, MMA_M, STAGE)
        tCsC = thr_copy_c.partition_S(sC_epi)
        # (COPY_ATOM_N, COPY_N)
        tCrC = cute.make_fragment_like(tCsC[None, None, 0, 0])
        # (COPY_ATOM_N, COPY_N, EPI_N, SEQLEN_K)
        tCgC = thr_copy_c.partition_D(gC_epi)

        # Set tensor memory buffer for current tile
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

        #
        # Wait for accumulator buffer full
        #
        acc_pipeline.consumer_wait(acc_consumer_state)

        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

        epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )

        #
        # Store accumulator to global memory in subtiles
        #
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in range(subtile_cnt):
            #
            # Load accumulator from tensor memory buffer to register
            #
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

            #
            # Convert to C type
            #
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            acc_vec = acc_vec.to(self.c_dtype)
            tRS_rC.store(acc_vec)

            #
            # Store C to shared memory
            #
            c_buffer = (num_subtiles_executed + subtile_idx) % self.num_c_stage
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])

            # Fence and barrier to make sure shared memory store is visible to TMA store
            cute.arch.fence_proxy("async.shared", space="cta")
            epilog_sync_barrier.arrive_and_wait()

            #
            # TMA store C to global memory (issued by lane 0 from all epi warps)
            #
            for topk_idx_in_warp in cutlass.range(32):
                topk_idx = topk_idx_in_warp + warp_idx * 32
                seqlen_k_idx_in_batch = sI_tile[topk_idx]
                if const_expr(self.varlen):
                    seqlen_k_idx = seqlen_k_idx_in_batch + seqlen_k_offset
                else:
                    seqlen_k_idx = (batch, seqlen_k_idx_in_batch)
                cute.copy(tiled_copy_c, tCsC[(None, None, topk_idx, c_buffer)], tCrC)
                for j in cutlass.range_constexpr(cute.size(tCrC, mode=[1])):
                    for i in cutlass.range_constexpr(cute.size(tCrC, mode=[0])):
                        ptr = elem_pointer(tCgC, (i, j, subtile_idx, seqlen_k_idx))
                        cute.arch.atomic_add(
                            ptr=ptr,
                            val=tCrC[i, j],
                        )
            epilog_sync_barrier.arrive_and_wait()

        epilog_sync_barrier.arrive_and_wait()

        #
        # Async arrive accumulator buffer empty
        #
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        return acc_consumer_state

    def check_can_implement(self):
        """Check if parameters are valid.

        :raises testing.CantImplementError: If the mma tiler, cluster shape, or alignments are invalid
        """
        if self.dim != 64:
            raise testing.CantImplementError(f"Only dim = 64 supported for dK gemm, got {self.dim}")
        # Check valid MMA tile shape and cluster shape
        if not (
            (not self.use_2cta_instrs and self.mma_tiler_dK[0] in [64, 128])
            or (self.use_2cta_instrs and self.mma_tiler_dK[0] in [128, 256])
        ):
            raise testing.CantImplementError(
                f"Invalid mma tiler & use_2cta_instrs: {self.mma_tiler_dK}, {self.use_2cta_instrs}"
            )
        if self.mma_tiler_dK[1] not in range(32, 257, 32):
            raise testing.CantImplementError(f"Invalid mma tiler N: {self.mma_tiler_dK[1]}")
        # Skip illegal cluster shape
        if self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0:
            raise testing.CantImplementError(f"Invalid cluster shape M: {self.cluster_shape_mn[0]}")
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 16
            or self.cluster_shape_mn[0] <= 0
            or self.cluster_shape_mn[1] <= 0
            or not is_power_of_2(self.cluster_shape_mn[0])
            or not is_power_of_2(self.cluster_shape_mn[1])
        ):
            raise testing.CantImplementError(f"Invalid cluster shape: {self.cluster_shape_mn}")

        # Check that all tensors are 16B aligned for TMA

        def check_contiguous_16B_alignment(dtype, num_major_elements):
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contiguous_16B_alignment(self.ab_dtype, self.topk)
            or not check_contiguous_16B_alignment(self.ab_dtype, self.dim)
            or not check_contiguous_16B_alignment(self.c_dtype, self.dim)
        ):
            raise testing.CantImplementError(
                f"Invalid tensor alignment: {self.ab_dtype=}, {self.c_dtype=}, {self.topk=}, {self.dim=}"
            )
