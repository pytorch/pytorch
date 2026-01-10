# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import functools
from typing import List, Type, Union
from inspect import isclass

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.torch as cutlass_torch

"""
A grouped GEMM example for the NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of grouped GEMM using a TMA plus Blackwell SM100 TensorCore
warp-specialized persistent kernel.
The grouped GEMM workload computes a batch of GEMM operations with distinct problem sizes. Pointers to matrices
in global memory are passed to the kernel in an array (also held in global memory). Similarly, problem shapes and
strides are also stored in arrays in GMEM.

This differs from "Batched Array" GEMM since the size of each GEMM problem in the grouped GEMM concept may be distinct.

To run this example:

.. code-block:: bash

    python examples/blackwell/grouped_gemm.py                                                 \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                                \
      --mma_tiler_mn 128,64 --cluster_shape_mn 1,1                                            \
      --problem_sizes_mnkl "(8192,1280,32,1),(16,384,1536,1),(640,1280,16,1),(640,160,16,1)"  \
      --num_groups 4  --tensormap_update_mode SMEM

The above example command makes 4 groups of different m, n, k sizes. The Blackwell tcgen05 MMA tile shape
is specified as (128, 64) and the cluster shape is (1,1). The input, mma accumulator and output data type
are set as fp16, fp32 and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/grouped_gemm.py                                             \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32                                \
      --mma_tiler_mn 128,64 --cluster_shape_mn 1,1                                            \
      --problem_sizes_mnkl "(8192,1280,32,1),(16,384,1536,1),(640,1280,16,1),(640,160,16,1)"  \
      --num_groups 4  --tensormap_update_mode SMEM                                            \
      --warmup_iterations 1 --iterations 10 --skip_ref_check

There are some constrains for this example. Besides the constrains from the Balckwell dense GEMM persistent example,
there are also the following constrains:
* Only fp16 and bf16 data types are supported as inputs.
* Output data types could be fp16, bf16 or fp32.
* The contiguous dimension of each tensor must be at least 16 bytes aligned.
* The l mode(aka, batch size) for each group must be 1.
* The majorness for A, B and C must be the same across all groups.
"""


class GroupedGemmKernel:
    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        tensormap_update_mode: utils.TensorMapUpdateMode = utils.TensorMapUpdateMode.SMEM,
    ):
        """Initializes the configuration for a Blackwell grouped GEMM kernel.

        Besides configurations for dense persistent GEMM, there is an extra config specific to grouped GEMM:

        Tensormap Update Mode:
        - tensormap_update_mode: Specifies whether the tensormap is
            updated in global memory(GMEM) or shared memory(SMEM).
           The 2 modes are functionally equivalent and the difference are:
            - We buffer 3 tensormaps in SMEM for A, B, and C tensors (each TMA descriptor takes 128B) when TMA updates performed on SMEM.
            - Performance varies between modes depending on problem size; optimal choice differs across workloads.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: tuple[int, int]
        :param cluster_shape_mn: tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: tuple[int, int]
        :param tensormap_update_mode: Mode for updating the tensormap (GMEM or SMEM), defaults to SMEM.
        :type tensormap_update_mode: utils.TensorMapUpdateMode, optional
        """
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.tensormap_update_mode = tensormap_update_mode
        # Delegate tensormap ab initialization to MMA warp when SMEM mode is used for better latency hiding
        self.delegate_tensormap_ab_init = (
            tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        )

        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        # Set barrier id for cta sync, epilog sync, tmem ptr sync and tensormap update sync
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        # Barrier ID used by MMA/TMA warps to signal A/B tensormap initialization completion
        self.tensormap_ab_init_bar_id = 4
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tma_load_bytes = 0

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        Most of the implementation follows standard dense GEMM patterns,
        with the key difference being additional consideration for SMEM
        buffer needed for tensormap updates.
        """
        # Configure tiled mma
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
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

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_epi_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
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
        self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_epi_stage,
        )

        tensor_smem_bytes = self._get_tensor_smem_bytes(
            self.a_smem_layout_staged,
            self.a_dtype,
            self.b_smem_layout_staged,
            self.b_dtype,
            self.epi_smem_layout_staged,
            self.c_dtype,
        )
        mbar_smem_bytes = self._get_mbar_smem_bytes(
            num_acc_stage=self.num_acc_stage,
            num_ab_stage=self.num_ab_stage,
            num_epi_stage=self.num_epi_stage,
        )
        tensormap_smem_bytes = self._get_tensormap_smem_bytes(
            self.tensormap_update_mode
        )
        if (
            mbar_smem_bytes
            + tensormap_smem_bytes
            + GroupedGemmKernel.tensor_memory_management_bytes
            > self.reserved_smem_bytes
        ):
            raise ValueError(
                f"smem consumption for mbar and tensormap {mbar_smem_bytes + tensormap_smem_bytes} exceeds the "
                f"reserved smem bytes {self.reserved_smem_bytes}"
            )

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage
        )

    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_c: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        total_num_clusters: cutlass.Constexpr[int],
        tensormap_cute_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        For grouped GEMM, tensor shapes, tensor strides, and tensor address are all provided
        by different tensors in global memory. The "initial" tensors only carry data type and
        majorness information.

        :param initial_a: Initial tensor A, used for data type and majorness information.
        :type initial_a: cute.Tensor
        :param initial_b: Initial tensor B, used for data type and majorness information.
        :type initial_b: cute.Tensor
        :param initial_c: Initial tensor C, used for data type and majorness information.
        :type initial_c: cute.Tensor
        :param group_count: The number of GEMM groups.
        :type group_count: cutlass.Constexpr[int]
        :param problem_shape_mnkl: Tensor containing the (M, N, K, L) shape for each group.
        :type problem_shape_mnkl: cute.Tensor
        :param strides_abc: Tensor containing the strides for A, B, and C for each group.
        :type strides_abc: cute.Tensor
        :param tensor_address_abc: Tensor containing the base addresses for A, B, and C for each group.
        :type tensor_address_abc: cute.Tensor
        :param total_num_clusters: Total number of clusters needed for all groups.
        :type total_num_clusters: cutlass.Constexpr[int]
        :param tensormap_cute_tensor: Tensor for storing tensormaps.
        :type tensormap_cute_tensor: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr[int]
        :param stream: CUDA stream for asynchronous execution.
        :type stream: cuda.CUstream
        :raises TypeError: If A and B data types do not match.
        """
        self.a_dtype = initial_a.element_type
        self.b_dtype = initial_b.element_type
        self.c_dtype = initial_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(initial_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(initial_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(initial_c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            initial_a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            initial_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(initial_c.shape), self.epi_tile
        )
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            initial_c,
            epi_smem_layout,
            c_cta_v_layout,
        )

        self.tile_sched_params, grid = self._compute_grid(
            total_num_clusters, self.cluster_shape_mn, max_active_clusters
        )

        self.buffer_align_bytes = 1024
        self.size_tensormap_in_i64 = (
            0
            if self.tensormap_update_mode == utils.TensorMapUpdateMode.GMEM
            else GroupedGemmKernel.num_tensormaps
            * GroupedGemmKernel.bytes_per_tensormap
            // 8
        )

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.MemRange[
                cutlass.Int64, self.size_tensormap_in_i64
            ]
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged.outer),
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

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            group_count,
            problem_shape_mnkl,
            strides_abc,
            tensor_address_abc,
            tensormap_cute_tensor,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        group_count: cutlass.Constexpr[int],
        problem_sizes_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        ptrs_abc: cute.Tensor,
        tensormaps: cute.Tensor,
    ):
        """
        GPU device kernel performing the grouped GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coord inside cluster
        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: tensormap buffer, a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tensormap_a_smem_ptr = None
        tensormap_b_smem_ptr = None
        tensormap_c_smem_ptr = None
        if cutlass.const_expr(
            self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        ):
            tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
            tensormap_a_smem_ptr = tensormap_smem_ptr
            tensormap_b_smem_ptr = (
                tensormap_a_smem_ptr + GroupedGemmKernel.bytes_per_tensormap // 8
            )
            tensormap_c_smem_ptr = (
                tensormap_b_smem_ptr + GroupedGemmKernel.bytes_per_tensormap // 8
            )
        ab_full_mbar_ptr = storage.ab_full_mbar_ptr.data_ptr()
        ab_empty_mbar_ptr = storage.ab_empty_mbar_ptr.data_ptr()
        acc_full_mbar_ptr = storage.acc_full_mbar_ptr.data_ptr()
        acc_empty_mbar_ptr = storage.acc_empty_mbar_ptr.data_ptr()
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        #  init barrier for loading A, B with TMA
        if warp_idx == self.epilog_warp_id[0]:
            for k_stage in range(self.num_ab_stage):
                num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(ab_full_mbar_ptr + k_stage, 1)
                    cute.arch.mbarrier_init(
                        ab_empty_mbar_ptr + k_stage, num_tma_producer
                    )
        # Accumulator barrier init
        if warp_idx == self.mma_warp_id:
            for acc_stage in range(self.num_acc_stage):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(acc_full_mbar_ptr + acc_stage, 1)
                    cute.arch.mbarrier_init(
                        acc_empty_mbar_ptr + acc_stage, 8 if use_2cta_instrs else 4
                    )
        # Tensor memory dealloc barrier init
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads
                    )
        cute.arch.mbarrier_init_fence()

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # Setup smem tensor A/B/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        #
        # Compute multicast mask for A/B buffer full and empty
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        ab_empty_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            ab_empty_mcast_mask = a_full_mcast_mask | b_full_mcast_mask
        acc_full_mcast_mask = None
        if cutlass.const_expr(use_2cta_instrs):
            acc_full_mcast_mask = cute.make_layout_image_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mode=0
            )
            block_in_cluster_coord_vmnk_peer = (
                block_in_cluster_coord_vmnk[0] ^ 1,
                *block_in_cluster_coord_vmnk[1:],
            )
            a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=2
            )
            b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=1
            )
            ab_empty_mcast_mask = (
                a_full_mcast_mask_peer
                | b_full_mcast_mask_peer
                | cutlass.Int16(
                    0 if ab_empty_mcast_mask is None else ab_empty_mcast_mask
                )
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for load A, B with TMA
        #
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
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
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

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
            )

        #
        # Get tensormap buffer address
        #
        grid_dim = cute.arch.grid_dim()
        tensormap_workspace_idx = (
            bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]
        )

        tensormap_manager = utils.TensorMapManager(
            self.tensormap_update_mode, GroupedGemmKernel.bytes_per_tensormap
        )
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 0, None)].iterator
        )
        tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 1, None)].iterator
        )
        tensormap_c_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 2, None)].iterator
        )
        # Setup tensormap initialization pointer based on the mode
        if cutlass.const_expr(
            self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        ):
            tensormap_a_init_ptr = tensormap_a_smem_ptr
            tensormap_b_init_ptr = tensormap_b_smem_ptr
            tensormap_c_init_ptr = tensormap_c_smem_ptr
        else:
            tensormap_a_init_ptr = tensormap_a_ptr
            tensormap_b_init_ptr = tensormap_b_ptr
            tensormap_c_init_ptr = tensormap_c_ptr

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            # Initialize tensormaps for A, B
            if cutlass.const_expr(self.delegate_tensormap_ab_init == False):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_init_ptr, self.tma_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_b, tensormap_b_init_ptr, self.tma_warp_id
                )
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, bid, grid_dim
            )
            # grouped gemm tile scheduler helper will compute the group index for the tile we're working on
            group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
                group_count,
                tile_sched_params,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
            )
            tensormap_init_done = cutlass.Boolean(False)
            # tile count we have searched
            total_k_tile_cnt = cutlass.Int32(0)
            # group index of last tile
            last_group_idx = cutlass.Int32(-1)
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                grouped_gemm_cta_tile_info = group_gemm_ts_helper.delinearize_z(
                    cur_tile_coord,
                    problem_sizes_mnkl,
                )
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                is_group_changed = cur_group_idx != last_group_idx
                # skip tensormap update if we're working on the same group
                if is_group_changed:
                    real_tensor_a = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.a_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        0,  # 0 for tensor A
                    )
                    real_tensor_b = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.b_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        1,  # 1 for tensor B
                    )
                    # wait tensormap initialization complete before update
                    if tensormap_init_done == False:
                        if cutlass.const_expr(self.delegate_tensormap_ab_init):
                            cute.arch.barrier(
                                barrier_id=self.tensormap_ab_init_bar_id,
                                number_of_threads=64,
                            )
                        tensormap_manager.fence_tensormap_initialization()
                        tensormap_init_done = True

                    tensormap_manager.update_tensormap(
                        (real_tensor_a, real_tensor_b),
                        (tma_atom_a, tma_atom_b),
                        (tensormap_a_ptr, tensormap_b_ptr),
                        self.tma_warp_id,
                        (tensormap_a_smem_ptr, tensormap_b_smem_ptr),
                    )

                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m
                    // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                num_prev_k_blk = total_k_tile_cnt
                total_k_tile_cnt += cur_k_tile_cnt

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                tma_wr_k_tile = cutlass.Int32(0)
                smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % self.num_ab_stage
                tma_wr_ab_empty_phase = (
                    num_prev_k_blk + tma_wr_k_tile
                ) // self.num_ab_stage % 2 ^ 1
                peek_ab_empty_status = cute.arch.mbarrier_conditional_try_wait(
                    tma_wr_k_tile < cur_k_tile_cnt,
                    ab_empty_mbar_ptr + smem_wr_buffer,
                    tma_wr_ab_empty_phase,
                )
                # ensure the update to tensormap has completed before using it
                if is_group_changed:
                    tensormap_manager.fence_tensormap_update(tensormap_a_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_b_ptr)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                    tma_wr_k_tile_next = tma_wr_k_tile + 1
                    smem_wr_buffer_next = (
                        num_prev_k_blk + tma_wr_k_tile_next
                    ) % self.num_ab_stage
                    tma_wr_ab_empty_phase_next = (
                        tma_wr_ab_empty_phase ^ 1
                        if smem_wr_buffer_next == 0
                        else tma_wr_ab_empty_phase
                    )

                    smem_full_mbar_ptr = ab_full_mbar_ptr + smem_wr_buffer

                    # Wait for AB buffer empty
                    if peek_ab_empty_status == 0:
                        cute.arch.mbarrier_wait(
                            ab_empty_mbar_ptr + smem_wr_buffer, tma_wr_ab_empty_phase
                        )

                    # Arrive AB buffer and expect full transaction bytes
                    if is_leader_cta:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                smem_full_mbar_ptr, self.num_tma_load_bytes
                            )

                    # Load A/B with TMA
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, tma_wr_k_tile)],
                        tAsA[(None, smem_wr_buffer)],
                        tma_bar_ptr=smem_full_mbar_ptr,
                        mcast_mask=a_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_a_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, tma_wr_k_tile)],
                        tBsB[(None, smem_wr_buffer)],
                        tma_bar_ptr=smem_full_mbar_ptr,
                        mcast_mask=b_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_b_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    peek_ab_empty_status = cute.arch.mbarrier_conditional_try_wait(
                        tma_wr_k_tile_next < cur_k_tile_cnt,
                        ab_empty_mbar_ptr + smem_wr_buffer_next,
                        tma_wr_ab_empty_phase_next,
                    )

                    tma_wr_k_tile = tma_wr_k_tile_next
                    smem_wr_buffer = smem_wr_buffer_next
                    tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            # initialize tensormap A, B for TMA warp
            if cutlass.const_expr(self.delegate_tensormap_ab_init):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_init_ptr, self.mma_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_b, tensormap_b_init_ptr, self.mma_warp_id
                )
                # signal tensormap initialization has finished
                cute.arch.barrier(
                    barrier_id=self.tensormap_ab_init_bar_id, number_of_threads=64
                )
            #  Bar sync for retrieve tmem ptr from shared mem
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, bid, grid_dim
            )
            # grouped gemm tile scheduler helper will compute the group index for the tile we're working on
            group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
                group_count,
                tile_sched_params,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
            )

            work_tile = tile_sched.initial_work_tile_info()
            # tile count we have searched
            total_k_tile_cnt = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                # MMA warp is only interested in number of tiles along K dimension
                (
                    cur_k_tile_cnt,
                    cur_group_idx,
                ) = group_gemm_ts_helper.search_cluster_tile_count_k(
                    cur_tile_coord,
                    problem_sizes_mnkl,
                )
                # Set tensor memory buffer for current tile
                acc_buf_idx = tile_sched.num_tiles_executed % self.num_acc_stage
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_buf_idx)]

                num_prev_k_blk = total_k_tile_cnt
                total_k_tile_cnt += cur_k_tile_cnt

                # Peek (try_wait) AB buffer full for k_tile = 0
                mma_rd_k_tile = cutlass.Int32(0)
                smem_rd_buffer = (num_prev_k_blk + mma_rd_k_tile) % self.num_ab_stage
                need_check_rd_buffer_full = (
                    mma_rd_k_tile < cur_k_tile_cnt and is_leader_cta
                )
                mma_rd_ab_full_phase = (
                    (num_prev_k_blk + mma_rd_k_tile) // self.num_ab_stage % 2
                )
                peek_ab_full_status = cute.arch.mbarrier_conditional_try_wait(
                    need_check_rd_buffer_full,
                    ab_full_mbar_ptr + smem_rd_buffer,
                    mma_rd_ab_full_phase,
                )

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_empty_phase = (
                        tile_sched.num_tiles_executed // self.num_acc_stage % 2 ^ 1
                    )
                    cute.arch.mbarrier_wait(
                        acc_empty_mbar_ptr + acc_buf_idx, acc_empty_phase
                    )

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in range(cur_k_tile_cnt):
                    mma_rd_k_tile_next = cutlass.Int32(k_tile + 1)
                    smem_rd_buffer_next = (
                        num_prev_k_blk + mma_rd_k_tile_next
                    ) % self.num_ab_stage
                    mma_rd_ab_full_phase_next = (
                        mma_rd_ab_full_phase ^ 1
                        if smem_rd_buffer_next == 0
                        else mma_rd_ab_full_phase
                    )
                    if is_leader_cta:
                        # Wait for AB buffer full
                        if peek_ab_full_status == 0:
                            cute.arch.mbarrier_wait(
                                ab_full_mbar_ptr + smem_rd_buffer, mma_rd_ab_full_phase
                            )

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, smem_rd_buffer)

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        with cute.arch.elect_one():
                            tcgen05.commit(
                                ab_empty_mbar_ptr + smem_rd_buffer,
                                ab_empty_mcast_mask,
                                self.cta_group,
                            )

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    need_check_rd_buffer_full = (
                        mma_rd_k_tile_next < cur_k_tile_cnt and is_leader_cta
                    )

                    peek_ab_full_status = cute.arch.mbarrier_conditional_try_wait(
                        need_check_rd_buffer_full,
                        ab_full_mbar_ptr + smem_rd_buffer_next,
                        mma_rd_ab_full_phase_next,
                    )

                    mma_rd_k_tile = mma_rd_k_tile_next
                    smem_rd_buffer = smem_rd_buffer_next
                    mma_rd_ab_full_phase = mma_rd_ab_full_phase_next

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    with cute.arch.elect_one():
                        tcgen05.commit(
                            acc_full_mbar_ptr + acc_buf_idx,
                            acc_full_mcast_mask,
                            self.cta_group,
                        )

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            # initialize tensorap for C
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_c,
                tensormap_c_init_ptr,
                self.epilog_warp_id[0],
            )
            # Alloc tensor memory buffer
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx
            #
            # Partition for epilogue
            #
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(tma_atom_c, tCgC, epi_tile, sC)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, bid, grid_dim
            )
            # grouped gemm tile scheduler helper will compute the group index for the tile we're working on
            group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
                group_count,
                tile_sched_params,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
            )

            work_tile = tile_sched.initial_work_tile_info()
            # wait tensormap initialization complete before update
            tensormap_manager.fence_tensormap_initialization()
            # tile count we have searched
            total_k_tile_cnt = cutlass.Int32(0)
            # group index of last tile
            last_group_idx = cutlass.Int32(-1)
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                grouped_gemm_cta_tile_info = group_gemm_ts_helper.delinearize_z(
                    cur_tile_coord,
                    problem_sizes_mnkl,
                )
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                is_group_changed = cur_group_idx != last_group_idx
                if is_group_changed:
                    # construct tensor C based on real address, shape and stride information
                    real_tensor_c = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.c_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        2,  # 2 for tensor C
                    )
                    tensormap_manager.update_tensormap(
                        ((real_tensor_c),),
                        ((tma_atom_c),),
                        ((tensormap_c_ptr),),
                        self.epilog_warp_id[0],
                        (tensormap_c_smem_ptr,),
                    )

                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m
                    // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                total_k_tile_cnt += cur_k_tile_cnt

                #
                # Slice to per mma tile index
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                # Set tensor memory buffer for current tile
                acc_buf_idx = tile_sched.num_tiles_executed % self.num_acc_stage
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_buf_idx)]

                #
                # Wait for accumulator buffer full
                #
                acc_full_phase = tile_sched.num_tiles_executed // self.num_acc_stage % 2
                cute.arch.mbarrier_wait(acc_full_mbar_ptr + acc_buf_idx, acc_full_phase)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                # ensure the update to tensormap has completed before using it
                if is_group_changed:
                    if warp_idx == self.epilog_warp_id[0]:
                        tensormap_manager.fence_tensormap_update(tensormap_c_ptr)
                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in range(subtile_cnt):
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # Convert to output type
                    #
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    tRS_rC.store(acc_vec.to(self.c_dtype))
                    #
                    # Store C to shared memory
                    #
                    epi_buffer = (num_prev_subtiles + subtile_idx) % self.num_epi_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, epi_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    epilog_threads = 32 * len(self.epilog_warp_id)
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )
                    #
                    # store C to global memory with TMA
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, epi_buffer)],
                            bSG_gC[(None, subtile_idx)],
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_c_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(
                            self.num_epi_stage - 1, read=True
                        )
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )
                #
                # Async arrive accumulator buffer empty
                #
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(
                        acc_empty_mbar_ptr + acc_buf_idx,
                        cta_rank_in_cluster // 2 * 2 if use_2cta_instrs else None,
                    )

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            #
            # Dealloc the tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            epilog_threads = 32 * len(self.epilog_warp_id)
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
            )
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )

            #
            # Wait a/b buffer empty
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.mbarrier_wait(
                    (ab_empty_mbar_ptr + ((total_k_tile_cnt - 1) % self.num_ab_stage)),
                    (((total_k_tile_cnt - 1) // self.num_ab_stage) % 2),
                )

    @cute.jit
    def make_tensor_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_index: int,
    ):
        """Extract stride and tensor address for a given group and construct a global tensor.

        This function is used within the kernel to dynamically create a CUTE tensor
        representing A, B, or C for the current group being processed, using the
        group-specific address, shape, and stride information.

        :param group_idx: The index of the current group within the grouped GEMM.
        :type group_idx: cutlass.Int32
        :param dtype: The data type of the tensor elements (e.g., cutlass.Float16).
        :type dtype: Type[cutlass.Numeric]
        :param problem_shape_mnk: The (M, N, K) problem shape for the current group.
        :type problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        :param strides_abc: Tensor containing strides for A, B, C for all groups. Layout: (group_count, 3, 2).
        :type strides_abc: cute.Tensor
        :param tensor_address_abc: Tensor containing global memory addresses for A, B, C for all groups. Layout: (group_count, 3).
        :type tensor_address_abc: cute.Tensor
        :param tensor_index: Specifies which tensor to create: 0 for A, 1 for B, 2 for C.
        :type tensor_index: int
        :return: A CUTE tensor representing the requested global memory tensor (A, B, or C) for the specified group.
        :rtype: cute.Tensor
        :raises TypeError: If the provided dtype is not a subclass of cutlass.Numeric.
        """
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
        strides_tensor_reg = cute.make_fragment(
            cute.make_layout(2),
            strides_abc.element_type,
        )
        cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
        stride_mn = strides_tensor_reg[0]
        stride_k = strides_tensor_reg[1]
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        if cutlass.const_expr(tensor_index == 0):  # tensor A
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        elif cutlass.const_expr(tensor_index == 1):  # tensor B
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((n, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        else:  # tensor C
            m = problem_shape_mnk[0]
            n = problem_shape_mnk[1]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, n, c1), stride=(stride_mn, stride_k, c0)),
            )

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load(t2r)
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
        tTR_rAcc = cute.make_fragment(
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
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
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
        tma_atom_c: cute.CopyAtom,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to partition
        shared memory (source) and global memory (destination) for TMA store version.

        :param tma_atom_c: The TMA copy atom configured for storing tensor C.
        :type tma_atom_c: cute.CopyAtom
        :param gC_mnl: The global memory tensor C.
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler defining the granularity of the operation.
        :type epi_tile: cute.Tile
        :param sC: The shared memory epilogue buffer tensor.
        :type sC: cute.Tensor
        :return: A tuple containing:
                 - tma_atom_c: The input TMA copy atom (passed through).
                 - bSG_sC: The source shared memory tensor partitioned for the TMA operation.
                 - tCgC: The destination global memory tensor partitioned for the TMA operation.
        :rtype: tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
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
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int, int]:
        """Computes the number of stages for accumulator, A/B operands, and epilogue based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C in global memory.
        :type c_layout: utils.LayoutEnum
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (accumulator stages, A/B operand stages, epilogue stages)
        :rtype: tuple[int, int, int]
        """
        # Default accumulator and epilogue stages
        num_acc_stage = 2
        num_epi_stage = 2

        # Calculate smem layout and size for one stage of A, B, and Epilogue
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # stage=1
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # stage=1
        )
        epi_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,  # stage=1
        )
        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)

        epi_bytes_per_stage = cute.size_in_bytes(c_dtype, epi_smem_layout_staged_one)
        epi_bytes = epi_bytes_per_stage * num_epi_stage

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial epilogue bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            smem_capacity // occupancy
            - GroupedGemmKernel.reserved_smem_bytes
            - epi_bytes
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        remaining_smem = (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (GroupedGemmKernel.reserved_smem_bytes + epi_bytes)
        )
        num_epi_stage += remaining_smem // (occupancy * epi_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_epi_stage

    @staticmethod
    def _compute_grid(
        total_num_clusters: int,
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr[int],
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """Compute tile scheduler parameters and grid shape for grouped GEMM operations.

        :param total_num_clusters: Total number of clusters to process across all groups.
        :type total_num_clusters: int
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr[int]

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: tuple[utils.PersistentTileSchedulerParams, tuple[int, ...]]
        """
        # Create problem shape with M, N dimensions from cluster shape
        # and L dimension representing the total number of clusters.
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )

        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, (*cluster_shape_mn, 1)
        )

        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _get_mbar_smem_bytes(**kwargs_stages: int) -> int:
        """Calculate shared memory consumption for memory barriers based on provided stages.

        Each stage requires 2 barriers, and each barrier consumes 8 bytes of shared memory.
        The total consumption is the sum across all provided stages. This function calculates the total
        shared memory needed for these barriers.

        :param kwargs_stages: Variable keyword arguments where each key is a stage name
                              (e.g., num_acc_stage, num_ab_stage) and each value is the
                              number of stages of that type.
        :type kwargs_stages: int
        :return: Total shared memory bytes required for all memory barriers.
        :rtype: int
        """
        num_barriers_per_stage = 2
        num_bytes_per_barrier = 8
        mbar_smem_consumption = sum(
            [
                num_barriers_per_stage * num_bytes_per_barrier * stage
                for stage in kwargs_stages.values()
            ]
        )
        return mbar_smem_consumption

    @staticmethod
    def _get_tensormap_smem_bytes(
        tensormap_update_mode: utils.TensorMapUpdateMode,
    ) -> int:
        """Get the SMEM consumption for the tensormap buffer based on the update mode.

        :param tensormap_update_mode: Specifies whether tensormaps are updated in GMEM or SMEM.
        :type tensormap_update_mode: utils.TensorMapUpdateMode
        :return: The shared memory bytes required for the tensormap buffer. Returns 0 if mode is GMEM.
        :rtype: int
        :raises ValueError: If an invalid tensormap update mode is provided.
        """
        if tensormap_update_mode == utils.TensorMapUpdateMode.GMEM:
            return 0
        elif tensormap_update_mode == utils.TensorMapUpdateMode.SMEM:
            return (
                GroupedGemmKernel.bytes_per_tensormap * GroupedGemmKernel.num_tensormaps
            )
        else:
            raise ValueError(f"Invalid tensormap update mode: {tensormap_update_mode}")

    @staticmethod
    def _get_tensor_smem_bytes(
        a_smem_layout_staged: cute.Layout,
        a_dtype: Type[cutlass.Numeric],
        b_smem_layout_staged: cute.Layout,
        b_dtype: Type[cutlass.Numeric],
        epi_smem_layout_staged: cute.Layout,
        c_dtype: Type[cutlass.Numeric],
    ) -> int:
        """Compute the total SMEM consumption for tensor A, B and C."""
        ab_bytes = cute.size_in_bytes(
            a_dtype, a_smem_layout_staged
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged)

        epi_bytes = cute.size_in_bytes(c_dtype, epi_smem_layout_staged)
        return ab_bytes + epi_bytes

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        """
        Compute the number of tensor memory allocation columns.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tile.
        :type mma_tiler: tuple[int, int, int]
        :param acc_stage: The stage of the accumulator tensor.
        :type acc_stage: int

        :return: The number of tensor memory allocation columns.
        :rtype: int
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        return num_tmem_alloc_cols

    # Size of smem we reserved for mbarrier, tensor memory management and tensormap update
    reserved_smem_bytes = 1024
    bytes_per_tensormap = 128
    num_tensormaps = 3
    # size of smem used for tensor memory management
    tensor_memory_management_bytes = 12


# Create tensor and return the pointer, tensor, and stride
def create_tensor_and_stride(
    l: int,
    mode0: int,
    mode1: int,
    is_mode0_major: bool,
    dtype: type[cutlass.Numeric],
    is_dynamic_layout: bool = True,
    torch_tensor_cpu: torch.Tensor = None,
) -> tuple[int, torch.Tensor, cute.Tensor, torch.Tensor, tuple[int, int]]:
    """Create a GPU tensor from scratch or based on an existing CPU tensor.

    :param torch_tensor_cpu: Optional existing CPU tensor to reuse. If None, creates a new one.
    :type torch_tensor_cpu: torch.Tensor, optional
    """
    if torch_tensor_cpu is None:
        # Create new CPU tensor
        torch_tensor_cpu = cutlass_torch.matrix(l, mode0, mode1, is_mode0_major, dtype)

    # Create GPU tensor from CPU tensor (new or existing)
    cute_tensor, torch_tensor = cutlass_torch.cute_tensor_like(
        torch_tensor_cpu, dtype, is_dynamic_layout, assumed_align=16
    )
    return (
        torch_tensor.data_ptr(),
        torch_tensor,
        cute_tensor,
        torch_tensor_cpu,
        torch_tensor.stride()[:-1],
    )


def create_tensors_for_all_groups(
    problem_sizes_mnkl: List[tuple[int, int, int, int]],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    torch_fp32_tensors_abc: List[List[torch.Tensor]] = None,
) -> tuple[
    List[List[int]],
    List[List[torch.Tensor]],
    List[tuple],
    List[List[tuple]],
    List[List[torch.Tensor]],
]:
    if torch_fp32_tensors_abc is not None and len(torch_fp32_tensors_abc) != len(
        problem_sizes_mnkl
    ):
        raise ValueError("torch_fp32_tensors_abc must have one entry per group")

    # Initialize lists to store tensors for all groups
    new_torch_fp32_tensors_abc = (
        [] if torch_fp32_tensors_abc is None else torch_fp32_tensors_abc
    )
    torch_tensors_abc = []
    cute_tensors_abc = []
    strides_abc = []
    ptrs_abc = []

    # Iterate through all groups and create tensors for each group
    for group_idx, (m, n, k, l) in enumerate(problem_sizes_mnkl):
        # Get existing CPU tensors if available, otherwise None
        existing_cpu_a = (
            torch_fp32_tensors_abc[group_idx][0] if torch_fp32_tensors_abc else None
        )
        existing_cpu_b = (
            torch_fp32_tensors_abc[group_idx][1] if torch_fp32_tensors_abc else None
        )
        existing_cpu_c = (
            torch_fp32_tensors_abc[group_idx][2] if torch_fp32_tensors_abc else None
        )

        # Create tensors (reusing CPU tensors if provided)
        (
            ptr_a,
            torch_tensor_a,
            cute_tensor_a,
            tensor_fp32_a,
            stride_mk_a,
        ) = create_tensor_and_stride(
            l, m, k, a_major == "m", ab_dtype, torch_tensor_cpu=existing_cpu_a
        )
        (
            ptr_b,
            torch_tensor_b,
            cute_tensor_b,
            tensor_fp32_b,
            stride_nk_b,
        ) = create_tensor_and_stride(
            l, n, k, b_major == "n", ab_dtype, torch_tensor_cpu=existing_cpu_b
        )
        (
            ptr_c,
            torch_tensor_c,
            cute_tensor_c,
            tensor_fp32_c,
            stride_mn_c,
        ) = create_tensor_and_stride(
            l, m, n, c_major == "m", c_dtype, torch_tensor_cpu=existing_cpu_c
        )

        # Only append to new_torch_fp32_tensors_abc if we created new CPU tensors
        if torch_fp32_tensors_abc is None:
            new_torch_fp32_tensors_abc.append(
                [tensor_fp32_a, tensor_fp32_b, tensor_fp32_c]
            )

        ptrs_abc.append([ptr_a, ptr_b, ptr_c])
        torch_tensors_abc.append([torch_tensor_a, torch_tensor_b, torch_tensor_c])
        strides_abc.append([stride_mk_a, stride_nk_b, stride_mn_c])
        cute_tensors_abc.append(
            (
                cute_tensor_a,
                cute_tensor_b,
                cute_tensor_c,
            )
        )

    return (
        ptrs_abc,
        torch_tensors_abc,
        cute_tensors_abc,
        strides_abc,
        new_torch_fp32_tensors_abc,
    )


def run(
    num_groups: int,
    problem_sizes_mnkl: tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    use_2cta_instrs: bool,
    tensormap_update_mode: utils.TensorMapUpdateMode,
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    **kwargs,
):
    """Run grouped GEMM example with specified configurations.

    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache, defaults to False
    :type use_cold_l2: bool, optional
    :return: Execution time of the GEMM kernel in microseconds
    :rtype: float
    """
    print(f"Running Blackwell Grouped GEMM test with:")
    print(f"{num_groups} groups")
    for i, (m, n, k, l) in enumerate(problem_sizes_mnkl):
        print(f"Group {i}: {m}x{n}x{k}x{l}")
    print(f"AB dtype: {ab_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"2CTA MMA instructions: {'True' if use_2cta_instrs else 'False'}")
    print(f"Tensor map update mode: {tensormap_update_mode}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    # Skip unsupported types
    if ab_dtype not in {
        cutlass.Float16,
        cutlass.BFloat16,
    }:
        raise ValueError(f"Skip unsupported ab_dtype {ab_dtype}")
    if c_dtype not in {cutlass.Float16, cutlass.BFloat16, cutlass.Float32}:
        raise ValueError(f"Skip unsupported c_dtype {c_dtype}")
    # Skip unsupported acc dtype
    if acc_dtype not in {cutlass.Float32, cutlass.Float16}:
        raise ValueError(f"Skip unsupported acc_dtype {acc_dtype}")
    # Skip invalid ab_dtype and acc_dtype combination
    if ab_dtype == cutlass.BFloat16 and acc_dtype == cutlass.Float16:
        raise ValueError("Skip invalid ab_dtype and acc_dtype combination")
    # Skip invalid mma tile shape
    if not (
        (not use_2cta_instrs and mma_tiler_mn[0] in [64, 128])
        or (use_2cta_instrs and mma_tiler_mn[0] in [128, 256])
    ):
        raise ValueError(f"Skip invalid mma tiler M {mma_tiler_mn[0]}")
    if mma_tiler_mn[1] not in range(32, 257, 32):
        raise ValueError(f"Skip invalid mma tiler N {mma_tiler_mn[1]}")
    # Skip illegal cluster shape
    if cluster_shape_mn[0] % (2 if use_2cta_instrs else 1) != 0:
        raise ValueError(
            f"cluster_shape_m need align with use_2cta_instrs config {cluster_shape_mn}"
        )
    # Skip invalid cluster shape
    is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
    if (
        cluster_shape_mn[0] * cluster_shape_mn[1] > 16
        or cluster_shape_mn[0] <= 0
        or cluster_shape_mn[1] <= 0
        or not is_power_of_2(cluster_shape_mn[0])
        or not is_power_of_2(cluster_shape_mn[1])
    ):
        raise ValueError(f"Skip invalid cluster shape {cluster_shape_mn}")

    # Skip illegal problem shape for load/store alignment
    def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
        major_mode_idx = 0 if is_mode0_major else 1
        num_major_elements = tensor_shape[major_mode_idx]
        num_contiguous_elements = 16 * 8 // dtype.width
        return num_major_elements % num_contiguous_elements == 0

    if (
        not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
        or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
        or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
    ):
        raise ValueError("Skip invalid problem alignment")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    # Create tensors for all groups using the new function
    (
        ptrs_abc,
        torch_tensors_abc,
        cute_tensors_abc,
        strides_abc,
        torch_fp32_tensors_abc,
    ) = create_tensors_for_all_groups(
        problem_sizes_mnkl,
        ab_dtype,
        c_dtype,
        a_major,
        b_major,
        c_major,
    )

    # Choose A, B, C with the smallest size to create initial tensormaps
    key_size_a = lambda item: item[1][0] * item[1][2]
    key_size_b = lambda item: item[1][1] * item[1][2]
    key_size_c = lambda item: item[1][0] * item[1][1]
    # Find the indices of the groups with the smallest tensor sizes
    min_a_idx, _ = min(enumerate(problem_sizes_mnkl), key=key_size_a)
    min_b_idx, _ = min(enumerate(problem_sizes_mnkl), key=key_size_b)
    min_c_idx, _ = min(enumerate(problem_sizes_mnkl), key=key_size_c)
    initial_cute_tensors_abc = [
        cute_tensors_abc[min_a_idx][0],  # A with smallest (m, k)
        cute_tensors_abc[min_b_idx][1],  # B with smallest (n, k)
        cute_tensors_abc[min_c_idx][2],  # C with smallest (m, n)
    ]

    hardware_info = utils.HardwareInfo()
    sm_count = hardware_info.get_max_active_clusters(1)
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    # Prepare tensormap buffer for each SM
    num_tensormap_buffers = sm_count
    tensormap_shape = (
        num_tensormap_buffers,
        GroupedGemmKernel.num_tensormaps,
        GroupedGemmKernel.bytes_per_tensormap // 8,
    )
    tensor_of_tensormap, tensor_of_tensormap_torch = cutlass_torch.cute_tensor_like(
        torch.empty(tensormap_shape, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
    )

    grouped_gemm = GroupedGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        tensormap_update_mode,
    )

    # layout (num_groups, 4):(4, 1)
    (
        tensor_of_dim_size_mnkl,
        tensor_of_dim_size_mnkl_torch,
    ) = cutlass_torch.cute_tensor_like(
        torch.tensor(problem_sizes_mnkl, dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    # layout (num_groups, 3, 2):(6, 2, 1)
    tensor_of_strides_abc, tensor_of_strides_abc_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(strides_abc, dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )

    # layout (num_groups,3):(3, 1)
    tensor_of_ptrs_abc, tensor_of_ptrs_abc_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(ptrs_abc, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
        assumed_align=16,
    )

    # Compute total number of cluster tiles we need to compute for given grouped GEMM problem
    def compute_total_num_clusters(
        problem_sizes_mnkl: List[tuple[int, int, int, int]],
        cluster_tile_shape_mn: tuple[int, int],
    ) -> int:
        total_num_clusters = 0
        for m, n, _, _ in problem_sizes_mnkl:
            num_clusters_mn = tuple(
                (x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)
            )
            total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)
        return total_num_clusters

    # Compute cluster tile shape
    def compute_cluster_tile_shape(
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        use_2cta_instrs: bool,
    ) -> tuple[int, int]:
        cta_tile_shape_mn = list(mma_tiler_mn)
        if use_2cta_instrs:
            cta_tile_shape_mn[0] = cta_tile_shape_mn[0] // 2
        return tuple(x * y for x, y in zip(cta_tile_shape_mn, cluster_shape_mn))

    cluster_tile_shape_mn = compute_cluster_tile_shape(
        mma_tiler_mn, cluster_shape_mn, use_2cta_instrs
    )
    total_num_clusters = compute_total_num_clusters(
        problem_sizes_mnkl, cluster_tile_shape_mn
    )

    # Initialize Stream
    current_stream = cutlass_torch.default_stream()

    # Compile grouped GEMM kernel
    compiled_grouped_gemm = cute.compile(
        grouped_gemm,
        initial_cute_tensors_abc[0],
        initial_cute_tensors_abc[1],
        initial_cute_tensors_abc[2],
        num_groups,
        tensor_of_dim_size_mnkl,
        tensor_of_strides_abc,
        tensor_of_ptrs_abc,
        total_num_clusters,
        tensor_of_tensormap,
        max_active_clusters,
        current_stream,
    )

    if not skip_ref_check:
        compiled_grouped_gemm(
            initial_cute_tensors_abc[0],
            initial_cute_tensors_abc[1],
            initial_cute_tensors_abc[2],
            tensor_of_dim_size_mnkl,
            tensor_of_strides_abc,
            tensor_of_ptrs_abc,
            tensor_of_tensormap,
            current_stream,
        )

        # Compute reference result
        for i, (a, b, c) in enumerate(torch_tensors_abc):
            ref = torch.einsum(
                "mkl,nkl->mnl",
                a.cpu().to(dtype=torch.float32),
                b.cpu().to(dtype=torch.float32),
            )
            print(f"checking group {i}")
            torch.testing.assert_close(
                c.cpu(),
                ref.to(cutlass_torch.dtype(c_dtype)),
                atol=tolerance,
                rtol=1e-05,
            )

    def generate_tensors():
        # Reuse existing CPU tensors and create new GPU tensors from them
        (
            ptrs_abc_workspace,
            torch_tensors_abc_workspace,
            cute_tensors_abc_workspace,
            strides_abc_workspace,
            _,
        ) = create_tensors_for_all_groups(
            problem_sizes_mnkl,
            ab_dtype,
            c_dtype,
            a_major,
            b_major,
            c_major,
            torch_fp32_tensors_abc,
        )

        initial_cute_tensors_abc_workspace = [
            cute_tensors_abc_workspace[min_a_idx][0],  # A with smallest (m, k)
            cute_tensors_abc_workspace[min_b_idx][1],  # B with smallest (n, k)
            cute_tensors_abc_workspace[min_c_idx][2],  # C with smallest (m, n)
        ]

        # Create new tensors for this workspace
        tensor_of_strides_abc_workspace, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(strides_abc_workspace, dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )

        tensor_of_ptrs_abc_workspace, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(ptrs_abc_workspace, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
            assumed_align=16,
        )

        tensormap_workspace, _ = cutlass_torch.cute_tensor_like(
            torch.empty(tensormap_shape, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
        )

        return testing.JitArguments(
            initial_cute_tensors_abc_workspace[0],
            initial_cute_tensors_abc_workspace[1],
            initial_cute_tensors_abc_workspace[2],
            tensor_of_dim_size_mnkl,
            tensor_of_strides_abc_workspace,
            tensor_of_ptrs_abc_workspace,
            tensormap_workspace,
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            sum(
                [
                    sum(
                        [
                            torch_tensor.numel() * torch_tensor.element_size()
                            for torch_tensor in group_tensors
                        ]
                    )
                    for group_tensors in torch_tensors_abc
                ]
            )
            +
            # Add size of strides tensor
            tensor_of_strides_abc_torch.numel()
            * tensor_of_strides_abc_torch.element_size()
            +
            # Add size of ptrs tensor
            tensor_of_ptrs_abc_torch.numel() * tensor_of_ptrs_abc_torch.element_size()
            +
            # Add size of tensormap tensor
            tensor_of_tensormap_torch.numel() * tensor_of_tensormap_torch.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_grouped_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    return exec_time  # Return execution time in microseconds


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    def parse_comma_separated_tuples(s: str) -> List[tuple[int, ...]]:
        if s.strip().startswith("("):
            # Split on ),( to separate tuples
            tuples = s.strip("()").split("),(")
            result = []
            tuple_len = None

            for t in tuples:
                # Parse individual tuple
                nums = [int(x.strip()) for x in t.split(",")]

                # Validate tuple length consistency
                if tuple_len is None:
                    tuple_len = len(nums)
                elif len(nums) != tuple_len:
                    raise argparse.ArgumentTypeError(
                        "All tuples must have the same length"
                    )

                result.append(tuple(nums))
            return result

        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers or list of tuples"
        )

    parser = argparse.ArgumentParser(
        description="Example of Grouped GEMM on Blackwell."
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        default=2,
        help="Number of groups",
    )
    parser.add_argument(
        "--problem_sizes_mnkl",
        type=parse_comma_separated_tuples,
        default=((128, 128, 128, 1), (128, 128, 128, 1)),
        help="a tuple of problem sizes for each group (comma-separated tuples)",
    )
    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="Mma tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument(
        "--tensormap_update_mode",
        type=str,
        default="SMEM",
        help="Tensor map update mode",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument(
        "--use_2cta_instrs",
        action="store_true",
        help="Enable 2CTA MMA instructions feature",
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if (
        len(args.problem_sizes_mnkl) != 0
        and len(args.problem_sizes_mnkl) != args.num_groups
    ):
        parser.error("--problem_sizes_mnkl must contain exactly num_groups tuples")

    # l mode must be 1 for all groups
    for _, _, _, l in args.problem_sizes_mnkl:
        if l != 1:
            parser.error("l must be 1 for all groups")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    if args.tensormap_update_mode not in ["GMEM", "SMEM"]:
        parser.error("--tensormap_update_mode must be GMEM or SMEM")

    if args.tensormap_update_mode == "GMEM":
        tensormap_update_mode = utils.TensorMapUpdateMode.GMEM
    else:
        tensormap_update_mode = utils.TensorMapUpdateMode.SMEM

    torch.manual_seed(2025)

    run(
        args.num_groups,
        args.problem_sizes_mnkl,
        args.ab_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.use_2cta_instrs,
        tensormap_update_mode,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    print("PASS")
