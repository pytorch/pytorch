# Copyright (c) 2025-2026, QuACK team.

# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py

from typing import Tuple, Type, Callable, Optional
from functools import partial
import math


import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Int32, Float32, Float16, Boolean, const_expr
from cutlass.utils import LayoutEnum


from . import layout_utils
from .gemm_base import GemmTmaBase, NamedBarrierGemm
from .tile_scheduler import TileSchedulerOptions
from .varlen_utils import VarlenArguments, VarlenManager

# return PipelineStateWAdvance instead of PipelineState
from .pipeline import make_pipeline_state
from . import copy_utils
from . import sm90_utils as quack_sm90_utils

"""
A high-performance batched dense GEMM (C = A * B) example for the NVIDIA Hopper architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports multi-stage pipeline to overlap computation and memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. Perform matrix multiply-accumulate (MMA) operations using WGMMA instruction.
3. Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM) with TMA operations.

Hopper WGMMA instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Perform MMA operation and store the result in Accumulator(register)

Constraints:
* Supported input data types: fp16, fp8 (e4m3fn, e5m2)
* For fp16 types, A and B must have the same data type
* For fp8 types, A and B can have different types (e4m3fn or e5m2) but both must be 8-bit
* Fp8 types only support k-major layout
* Only fp32 accumulation is supported in this example
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* CTA tile shape K must be 64
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.
"""


class GemmSm90(GemmTmaBase):
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs with persistent tile scheduling and warp specialization.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mnk: Shape of the CTA tile. Pass (M, N) to default K to
        4 MMA instructions, or (M, N, K) to set K explicitly.
    :type tile_shape_mnk: Tuple[int, int] | Tuple[int, int, int]
    :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
    :type cluster_shape_mnk: Tuple[int, int, int]

    :note: Data type requirements:
        - For 16-bit types: A and B must have the same data type
        - For 8-bit types: A and B can have different types (Float8E4M3FN/Float8E5M2) as long as both are 8-bit
        - Float8 types only support k-major layout

    :note: Supported data types:
        - Float16
        - BFloat16
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulation types:
        - Float32 (for all floating point inputs)

    :note: Constraints:
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = GemmSm90(
        ...     acc_dtype=Float32,
        ...     tile_shape_mnk=(128, 256),
        ...     cluster_shape_mnk=(1, 1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    arch = 90
    EpilogueArguments = GemmTmaBase.EpilogueArguments
    EpilogueParams = GemmTmaBase.EpilogueParams

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: Tuple[int, int] | Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool = False,
        is_persistent: bool = True,
        fp8_fast_accum: bool = False,
        gather_A: bool = False,
        use_clc_persistence: bool = False,
        concat_layout: tuple | None = None,
        use_pdl: bool = True,
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mnk: Shape of the CTA tile (M,N) or (M,N,K)
        :type tile_shape_mnk: Tuple[int, int] | Tuple[int, int, int]
        :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
        :type cluster_shape_mnk: Tuple[int, int, int]
        """

        self.acc_dtype = acc_dtype
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        self.use_clc_persistence = use_clc_persistence
        if self.use_clc_persistence:
            assert self.arch == 100
        self.use_pdl = use_pdl
        if self.pingpong:
            assert self.is_persistent, "Pingpong gemm requires persistent scheduler"
        self.fp8_slow_accum = not fp8_fast_accum and a_dtype.width == 8
        self.gather_A = gather_A
        self.concat_layout = concat_layout or ()
        if gather_A:
            assert cluster_shape_mnk[1] == 1, "Cluster shape N must be 1 for gather A "

        self.cluster_shape_mnk = cluster_shape_mnk
        assert len(tile_shape_mnk) in [2, 3], "CTA tile shape must be (M, N) or (M, N, K)"
        # K dimension: if user provides 3 values, use their K; otherwise default in _setup_tiled_mma.
        self.cta_tile_shape_mnk = (
            tuple(tile_shape_mnk) if len(tile_shape_mnk) == 3 else (*tile_shape_mnk, 0)
        )
        tile_M, tile_N = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]
        # check the cta tile shape
        if not self.pingpong:
            if tile_M not in [64, 128, 192, 256, 320]:
                raise ValueError("CTA tile shape M must be 64/128/192/256/320")
            if tile_M in [192, 320]:  # special case
                tile_N_max = 256 if tile_M == 192 else 160
                if not (tile_N % 32 == 0 and tile_N <= tile_N_max):
                    raise ValueError(
                        f"If tile_m == {tile_M}, CTA tile shape N must be divisible by 32 and <= {tile_N_max}"
                    )
            else:
                if not (
                    (tile_N % 16 == 0 and tile_N <= 256) or (tile_N % 32 == 0 and tile_N <= 512)
                ):
                    raise ValueError(
                        "CTA tile shape N must be divisible by 16 and <= 256, or divisible by 32 and <= 512"
                    )
        else:
            if tile_M not in [64, 128, 192]:
                raise ValueError("CTA tile shape M must be 64/128/192 if pingpong")
            tile_N_max = 256 if tile_M == 64 else (208 if tile_M == 128 else 128)
            if not (tile_N % 16 == 0 and tile_N <= tile_N_max):
                raise ValueError(f"CTA tile shape N must be divisible by 16 and <= {tile_N_max}")

        if not self.pingpong:
            if tile_M == 320:  # tile_M / 64 is not even so we have to split along N
                atom_layout_m, atom_layout_n = 1, 2
            elif tile_M == 192:
                if tile_N <= 128:
                    atom_layout_m, atom_layout_n = 3, 1
                else:
                    atom_layout_m, atom_layout_n = 1, 2
            else:
                atom_layout_m = (
                    self.cta_tile_shape_mnk[0] // 64 if self.cta_tile_shape_mnk[0] < 256 else 2
                )
                atom_layout_n = 1
            assert atom_layout_m in [1, 2, 3] and atom_layout_n in [1, 2]
        else:
            atom_layout_m, atom_layout_n = 1, 1
        self.atom_layout_mnk = (atom_layout_m, atom_layout_n, 1)

        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        if self.gather_A:
            assert self.num_mcast_ctas_a == 1
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk) * (1 if not self.pingpong else 2)
        if self.pingpong:
            assert self.mma_warp_groups == 2
        assert self.mma_warp_groups in [1, 2, 3]
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (self.mma_warp_groups + 1) * self.num_threads_per_warp_group
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes("sm_90")
        self.num_epi_warps = (self.mma_warp_groups if not self.pingpong else 1) * 4
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.Epilogue),
            num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
        )
        self.num_ab_load_warps = 1 if not self.gather_A else 4
        self.ab_load_warp_id = self.mma_warp_groups * 4

        regs_per_thread = math.prod(self.cta_tile_shape_mnk[:2]) // (
            math.prod(self.atom_layout_mnk) * self.num_threads_per_warp_group
        )
        if self.fp8_slow_accum:
            regs_per_thread *= 2
        if not self.gather_A:
            if self.mma_warp_groups == 3:
                self.num_regs_load, self.num_regs_mma = 32, 160
            else:
                heavy_register_pressure = regs_per_thread >= 208
                self.num_regs_load, self.num_regs_mma = (
                    (40, 232) if not heavy_register_pressure else (24, 240)
                )
        else:
            if self.mma_warp_groups == 3:
                self.num_regs_load, self.num_regs_mma = 56, 152
            else:
                self.num_regs_load, self.num_regs_mma = (56, 224)

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
        atom_m, atom_n, atom_k = self.atom_layout_mnk
        return (atom_m * 4, atom_n, atom_k)

    def _setup_tiled_mma(self):
        """Set up tiled MMA and tile K dimension. Override for different MMA types."""
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.cta_tile_shape_mnk[1] // self.atom_layout_mnk[1]),
        )
        if const_expr(self.atom_layout_mnk[1] > 1):
            # If N dimension is split among 2 WGs, we need to permute the N dimension so
            # that in the epilogue, WG0 and WG1 can write to epi smem of size e.g. (64, 32)
            # containing accumulators that are next to each other in the N dimension.
            # Without permutation WG0 would write to epi smem of size (64, 16) and
            # WG1 would write to a separate epi smem of size (64, 16) that's far away.
            atom_n = self.atom_layout_mnk[1]
            permutation_n = cute.make_ordered_layout(
                (8, self.cta_tile_shape_mnk[1] // atom_n // 8, atom_n), order=(0, 2, 1)
            )
            self.tiled_mma = cute.make_tiled_mma(
                cute.make_mma_atom(self.tiled_mma.op),
                self.atom_layout_mnk,
                permutation_mnk=(None, permutation_n, None),
            )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        tile_k = (
            self.cta_tile_shape_mnk[2] if self.cta_tile_shape_mnk[2] > 0 else mma_inst_shape_k * 4
        )
        assert tile_k > 0, "CTA tile K must be positive"
        assert tile_k % mma_inst_shape_k == 0, (
            f"CTA tile K ({tile_k}) must be divisible by MMA instruction K ({mma_inst_shape_k})"
        )
        self.cta_tile_shape_mnk = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1],
            tile_k,
        )

    def _setup_attributes(self, epilogue_args: EpilogueArguments):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        """
        self._setup_tiled_mma()
        self.epi_m_major = self.resolve_epi_m_major(epilogue_args)

        self.cluster_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.epi_tile = self._compute_tile_shape_or_override(
            self.cta_tile_shape_mnk,
            self.atom_layout_mnk,
            self.d_dtype,
        )
        self.epi_tile_shape = cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile)

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage, self.epi_c_stage = self._compute_stages(
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.d_dtype,
            self.c_dtype,
            epilogue_args,
            cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}"),  # smem_capacity
            self.occupancy,
            self.epi_smem_warp_shape_mnk(),
        )
        self.sched_stage = 2 if self.pingpong else 1

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.d_dtype,
            self.d_layout,
            self.epi_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_c_stage,
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args: tuple,
        scheduler_args: TileSchedulerOptions,
        varlen_args: Optional[VarlenArguments],
        stream: cuda.CUstream,
        trace_ptr: Optional[cutlass.Int64] = None,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param mA: Input tensor A
        :type mA: cute.Tensor
        :param mB: Input tensor B
        :type mB: cute.Tensor
        :param mD: Output tensor D
        :type mD: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # Concat layout: interleave the non-contiguous dim (detected via leading_dim).
        mA, mB, mD, mC = [
            layout_utils.concat_to_interleave(mT, 1 - mT.leading_dim)
            if const_expr(name in self.concat_layout and mT is not None)
            else mT
            for name, mT in [("A", mA), ("B", mB), ("out", mD), ("C", mC)]
        ]

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type if mD is not None else None
        self.c_dtype = mC.element_type if mC is not None else None
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD) if mD is not None else None
        self.c_layout = LayoutEnum.from_tensor(mC) if mC is not None else None

        if const_expr(self.a_dtype.width == 16 and self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}")
        if const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16 or float8")

        if const_expr(varlen_args is None):
            varlen_args = VarlenArguments()
        assert (varlen_args.mAIdx is not None) == self.gather_A
        varlen_m = varlen_args.mCuSeqlensM is not None
        varlen_k = varlen_args.mCuSeqlensK is not None

        self._setup_attributes(epilogue_args)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = self.make_tma_load_atoms_and_tensors(
            mA, mB, a_smem_layout, b_smem_layout, varlen_k
        )

        self.num_tma_load_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        if const_expr(not self.gather_A):
            self.num_tma_load_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)

        tma_atom_d, tma_tensor_d, tma_atom_c, tma_tensor_c = (
            self.make_tma_epilogue_atoms_and_tensors(mD, mC, epilogue_args, varlen_m)
        )

        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)
        varlen_params = VarlenManager.to_underlying_arguments(varlen_args)

        self.epi_load_bytes_per_stage = self.epi_smem_bytes(
            epilogue_args,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.epi_smem_warp_shape_mnk(),
        ).c_stage
        if const_expr(mC is not None):
            c_smem_layout = cute.slice_(self.epi_c_smem_layout_staged, (None, None, 0))
            self.epi_load_bytes_per_stage += cute.size_in_bytes(self.c_dtype, c_smem_layout)

        TileSchedulerCls = self.get_scheduler_class(varlen_m=varlen_m)
        tile_sched_args = self.get_scheduler_arguments(
            mA, mB, mD, scheduler_args, varlen_args, epilogue_args
        )
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)
        grid = TileSchedulerCls.get_grid_shape(
            tile_sched_params, scheduler_args.max_active_clusters
        )

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged) if mD is not None else 0
        epi_c_smem_size = cute.cosize(self.epi_c_smem_layout_staged) if mC is not None else 0

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            epi_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_c_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            sched_data: cute.struct.MemRange[Int32, self.sched_stage * 4]
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype if self.d_dtype is not None else Int32, epi_smem_size
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype if self.c_dtype is not None else Int32, epi_c_smem_size
                ],
                self.buffer_align_bytes,
            ]
            epi: self.epi_get_smem_struct(epilogue_params)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a if const_expr(not self.gather_A) else mA,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            epilogue_params,
            varlen_params,
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            tile_sched_params,
            TileSchedulerCls,
            trace_ptr,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )
        return

    #  GPU device kernel
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
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_d: TMA copy atom for D tensor
        :type tma_atom_d: cute.CopyAtom
        :param mD_mnl: Output tensor D
        :type mD_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cluster_layout_mnk: CTA layout
        :type cluster_layout_mnk: cute.Layout
        :param a_smem_layout: Shared memory layout for A
        :type a_smem_layout: cute.ComposedLayout
        :param b_smem_layout: Shared memory layout for B
        :type b_smem_layout: cute.ComposedLayout
        :param epi_smem_layout: Shared memory layout for epilogue
        :type epi_smem_layout: cute.ComposedLayout
        """

        from .trace import TraceContext

        tctx = TraceContext.create(trace_ptr)

        varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
        varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
        assert not (varlen_m and varlen_k)
        if const_expr(self.gather_A):
            assert varlen_m or varlen_k
        has_D = const_expr(mD_mnl is not None)
        has_C = const_expr(mC_mnl is not None)
        has_epi_load = const_expr(self.epi_c_stage > 0)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch Tma desc
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # Alloc and init AB full/empty + ACC full mbar (pipeline)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
            ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
        )
        epi_pipeline = None
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

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)

        # Generate smem tensor A/B
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
            # Only used if not varlen_m
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

        # Cluster wait for barrier init
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        if warp_idx >= self.ab_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if (
                warp_idx >= self.ab_load_warp_id
                and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
            ):
                # PDL: wait for prior kernel before any TMA loads (matches cutlass C++ sm90 mainloop producer)
                if const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()
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
                if warp_idx == self.ab_load_warp_id:
                    ab_pipeline.producer_tail(ab_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()

        if warp_idx < self.ab_load_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            is_tma_warp = Boolean(
                (not self.pingpong and warp_idx == 0)
                or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
            )
            # Partition global tensor for TiledMMA_A/B/C
            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            if const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group
            warp_group_thread_layout = cute.make_layout(
                self.mma_warp_groups if const_expr(not self.pingpong) else 1,
                stride=self.num_threads_per_warp_group,
            )
            thr_mma = tiled_mma.get_slice(
                warp_group_thread_layout(warp_group_idx if not self.pingpong else 0)
            )

            # Make fragments
            acc, tCrA, tCrB = quack_sm90_utils.partition_fragment_ABC(
                thr_mma, self.cta_tile_shape_mnk, sA, sB
            )
            acc_slow = None
            if const_expr(self.fp8_slow_accum):
                acc_slow = cute.make_rmem_tensor(acc.shape, self.acc_dtype)
            mma_fn = partial(quack_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)

            if const_expr(self.pingpong):
                if warp_group_idx == 0:
                    # WG0 needs a start signal at the very beginning
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

            k_tile_cnt_static = cute.ceil_div(
                cute.size(mA_mkl, mode=[1]), self.cta_tile_shape_mnk[2]
            )
            c_tile_cnt = cute.size(self.epi_tile_shape)

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
                    # TODO: do we need to check if work_tile is valid?
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                len_k = varlen_manager.len_k(batch_idx)
                k_tile_cnt = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                tctx.b("mma")
                ab_read_state = self.mma(
                    ab_pipeline, ab_read_state, mma_fn, acc, acc_slow, k_tile_cnt, warp_group_idx
                )
                if const_expr(varlen_k):
                    if k_tile_cnt == 0:
                        acc.fill(0.0)
                tctx.e("mma")

                # EPILOGUE
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, "epi")
                tctx.b("epilogue")

                copy_D = None
                if const_expr(has_D):
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d,
                        varlen_manager.offset_batch_epi(mD_mnl, batch_idx),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sD,
                        tile_coord_mnkl,
                    )
                copy_C = None
                if const_expr(has_C):
                    copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_c,
                        varlen_manager.offset_batch_epi(mC_mnl, batch_idx),
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
                # End of persistent scheduler loop

            # PDL: hint next kernel to launch (matches cutlass C++ sm90 consumer)
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

            # Wait for D store complete
            if const_expr(not self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()

        tctx.flush()

    @cute.jit
    def load_AB_gather_A(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        copy_A: Callable,
        prefetch_A: Optional[Callable],
        copy_B: Callable,
        k_tile_cnt: Int32,
        varlen_m: bool = True,
    ) -> cutlass.pipeline.PipelineState:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt
        peek_ab_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # TMA load on B and cp.async on A
        for k_tile in cutlass.range(k_tile_cnt - 1, unroll=1):
            prefetch_out = ()
            if const_expr(prefetch_A is not None):  # Prefetch early, even before smem is free
                prefetch_out = (prefetch_A(k_tile),)
            # Wait for A/B buffers to be empty before loading into them
            # Also sets the transaction barrier for the A/B buffers
            # A tiny bit faster to rotate the warp that does TMA
            is_tma_warp = warp_idx == self.ab_load_warp_id + (k_tile % self.num_ab_load_warps)
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status, is_tma_warp)
            smem_idx = ab_producer_state.index
            # A bit faster to load B first while we calculate the indices for A
            if is_tma_warp:
                tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
                copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_A(k_tile, smem_idx, *prefetch_out)
            # This tells mbarrier to track the completion of cp.async
            ab_pipeline.producer_cpasync_commit(ab_producer_state)
            ab_producer_state.advance()
            peek_ab_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # bound checking in the K dimension on the last k_tile
        if 0 < k_tile_cnt:
            k_tile = k_tile_cnt - 1
            prefetch_out = ()
            if const_expr(prefetch_A is not None):  # Prefetch early, even before smem is free
                prefetch_out = (prefetch_A(k_tile, pred=True),)
            is_tma_warp = warp_idx == self.ab_load_warp_id + k_tile % self.num_ab_load_warps
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status, is_tma_warp)
            smem_idx = ab_producer_state.index
            if is_tma_warp:
                tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
                copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_A(k_tile, smem_idx, *prefetch_out, pred=True)
            ab_pipeline.producer_cpasync_commit(ab_producer_state)
            ab_producer_state.advance()
        return ab_producer_state

    @cute.jit
    def _make_gather_A_copy(
        self,
        mA_mkl: cute.Tensor,
        sA: cute.Tensor,
        varlen_manager: VarlenManager,
        tile_coord_mnkl,
        batch_idx: Int32,
    ):
        """Create copy_A and prefetch_A for gather_A (shared by SM90/SM120 DMA)."""
        varlen_m = varlen_manager.varlen_m
        mAIdx_mk = varlen_manager.offset_batch_AIdx(batch_idx)
        if const_expr(varlen_m):
            gAIdx = cute.local_tile(mAIdx_mk, (self.cta_tile_shape_mnk[0],), (tile_coord_mnkl[0],))
            mA_mk = mA_mkl
        else:
            gAIdx = cute.flat_divide(mAIdx_mk, (self.cta_tile_shape_mnk[2],))
            mA_mk = cute.local_tile(
                mA_mkl, (self.cta_tile_shape_mnk[0],), (tile_coord_mnkl[0], None)
            )
        len_m = varlen_manager.len_m(batch_idx)
        len_k = varlen_manager.len_k(batch_idx)
        tiled_copy_A = self._make_gmem_tiled_copy_A(
            mA_mkl.element_type, self.a_layout, self.num_ab_load_warps * 32
        )
        dma_tidx = cute.arch.thread_idx()[0] - cute.arch.WARP_SIZE * self.ab_load_warp_id
        thr_copy_A = tiled_copy_A.get_slice(dma_tidx)
        copy_A, prefetch_A = None, None
        if const_expr(varlen_m):
            copy_A = copy_utils.gather_m_get_copy_fn(
                thr_copy_A,
                mA_mk,
                sA,
                gAIdx,
                limit_m=len_m - tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0],
                limit_k=len_k,
            )
        else:
            copy_A, prefetch_A = copy_utils.gather_k_get_copy_fn(
                thr_copy_A,
                mA_mk,
                sA,
                gAIdx,
                limit_m=len_m - tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0],
                limit_k=len_k,
            )
        return copy_A, prefetch_A

    @cute.jit
    def mma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        mma_fn: Callable,
        acc: cute.Tensor,
        acc_slow: Optional[cute.Tensor],
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
    ) -> cutlass.pipeline.PipelineState:
        # Prologue MMAs
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        zero_init = Boolean(True)
        for k_tile in cutlass.range(num_prologue_mma):
            # Wait for A/B buffer to be ready
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=zero_init)
            zero_init = Boolean(False)
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        # If k_tile_cnt == 0, this is not correct. But we will set acc to 0 in the mainloop
        # in that case.
        if const_expr(self.fp8_slow_accum):
            warpgroup.wait_group(0)
            acc_slow.store(acc.load())

        # MAINLOOP
        for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
            # Wait for TMA copies to complete
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            if const_expr(self.fp8_slow_accum):
                zero_init = Boolean(True)
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=zero_init)
            zero_init = Boolean(False)
            # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
            if const_expr(not self.fp8_slow_accum):
                warpgroup.wait_group(k_pipe_mmas)
            else:
                warpgroup.wait_group(0)
                acc_slow.store(acc_slow.load() + acc.load())
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        if const_expr(self.pingpong):
            # Cue for next WG's MMA to start
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        if const_expr(not self.fp8_slow_accum):
            # fp8_slow_accum would already called wait_group(0) inside the loop
            warpgroup.wait_group(0)
        for k_tile in cutlass.range(num_prologue_mma, unroll=1):
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        if const_expr(self.fp8_slow_accum):
            acc.store(acc_slow.load())
        return ab_read_state

    def epi_retile_acc(self, acc, tRS_rD, tiled_copy_r2s):
        """Retile accumulator for epilogue subtile access."""
        acc_reshaped = layout_utils.reshape_acc_to_frgA(acc)  # ((2, 2, 2), MMA_M, MMA_N)
        # ((2, 2, 2), MMA_M / epi_M, MMA_N / epi_N)
        epi_acc_shape = (
            acc_reshaped.shape[0],
            *cute.ceil_div(acc_reshaped.shape[1:], self.epi_tile_shape),
        )
        # ((2, 2, 2), MMA_M / epi_M, MMA_N / epi_N, (1, 1, 1), epi_M, epi_N)
        acc_divide = cute.flat_divide(acc_reshaped, epi_acc_shape)
        assert cute.size(acc_divide, mode=[3]) == 1
        # ((2, 2, 2), MMA_M / epi_M, MMA_N / epi_N, (epi_M, epi_N))
        tRS_rAcc = cute.group_modes(acc_divide[None, None, None, 0, None, None], 3, 5)
        # (((2,2,2),1), MMA_M / epi_M, MMA_N / epi_N, (epi_M, epi_N))
        return tiled_copy_r2s.retile(tRS_rAcc)

    def epilog_smem_copy_atom(self, tiled_mma: cute.TiledMma) -> cute.TiledCopy:
        copy_atom_C = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(
                self.d_layout.is_m_major_c() if self.d_layout is not None else False,
                num_matrices=4 if self.epi_tile[1] % 16 == 0 else 2,
            ),
            Float16,  # this is just to get the right source layout
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        return tiled_copy_C_atom

    def epilog_smem_store_and_partition(
        self,
        tiled_mma: cute.TiledMma,
        d_layout: Optional[LayoutEnum],
        dtype: Type[cutlass.Numeric],
        sD: Optional[cute.Tensor],
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        if d_layout is None:
            d_layout = LayoutEnum.ROW_MAJOR
        tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_mma)
        # Doesn't work with tile_N % 8 == 0 but tile_n % 16 != since this always
        # get st.matrix with num_matrices=4
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            d_layout, elem_ty_d=dtype, elem_ty_acc=self.acc_dtype
        )
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD) if sD is not None else None
        sD_shape = sD.shape[:2] if sD is not None else self.epi_tile
        tRS_rD_shape = thr_copy_r2s.partition_S(cute.make_identity_tensor(sD_shape)).shape
        tRS_rD = cute.make_rmem_tensor(tRS_rD_shape, self.acc_dtype)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    def epilog_smem_load_and_partition(
        self,
        tiled_mma: cute.TiledMma,
        c_layout: LayoutEnum,
        dtype: Type[cutlass.Numeric],
        sC: cute.Tensor,
        tRS_rD_layout: cutlass.Layout,
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_mma)
        copy_atom_s2r = copy_utils.sm90_get_smem_load_op(c_layout, dtype)
        tiled_copy_s2r = cute.make_tiled_copy_S(copy_atom_s2r, tiled_copy_C_atom)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        tSR_sC = thr_copy_s2r.partition_S(sC)
        tRS_rC = cute.make_rmem_tensor(tRS_rD_layout, dtype)
        tSR_rC = thr_copy_s2r.retile(tRS_rC)
        return tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC

    def pingpong_barrier_sync(self, warp_group_idx: Int32, stage: str):
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
        cute.arch.barrier(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    def pingpong_barrier_arrive(self, warp_group_idx: Int32, stage: str):
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
        cute.arch.barrier_arrive(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    def make_sched_pipeline(
        self, cluster_layout_mnk: cute.Layout, sched_pipeline_mbar_ptr: cute.Pointer, varlen_k: bool
    ):
        sched_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_layout_mnk)
        # Each warp will contribute 1 to the arrive count
        # If pingpong and varlen_k, then all 8 mma warps will participate in the scheduler barrier
        # at each round. If pingpong and not varlen_k, then only 4 mma warp will participate.
        consumer_arrive_cnt = (
            (self.mma_warp_groups if not (self.pingpong and not varlen_k) else 1) * 4
            + self.num_ab_load_warps
        ) * cluster_size
        sched_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return pipeline.PipelineAsync.create(
            barrier_storage=sched_pipeline_mbar_ptr,
            num_stages=self.sched_stage,
            producer_group=sched_pipeline_producer_group,
            consumer_group=sched_pipeline_consumer_group,
            # If there's cluster, the consumers must arrive at the mbar of CTA 0 in the cluster.
            consumer_mask=None if const_expr(cluster_size == 1) else 0,
            defer_sync=True,
        )

    @classmethod
    def _compute_stages(
        cls,
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        d_dtype: Optional[Type[cutlass.Numeric]],
        c_dtype: Optional[Type[cutlass.Numeric]],
        epilogue_args: EpilogueArguments,
        smem_capacity: int,
        occupancy: int,
        warp_shape_mnk: Tuple[int, int, int] | None = None,
    ) -> Tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: Tuple[int, int]
        """

        epi_stage = 4 if epi_tile[1] <= 16 else 2
        epi_smem_bytes = cls.epi_smem_bytes(
            epilogue_args, cta_tile_shape_mnk, epi_tile, warp_shape_mnk
        )
        has_tile_load = epi_smem_bytes.c_stage > 0
        epi_tile_elems = cute.size(cute.shape(epi_tile))
        d_bytes_per_stage = epi_tile_elems * d_dtype.width // 8 if d_dtype is not None else 0
        epi_bytes_per_stage = d_bytes_per_stage + epi_smem_bytes.d_stage
        epi_bytes = epi_smem_bytes.unstaged + epi_bytes_per_stage * epi_stage
        epi_c_stage = (
            0 if c_dtype is None and not has_tile_load else (4 if epi_tile[1] <= 16 else 2)
        )
        if c_dtype is not None:
            epi_bytes += epi_tile_elems * c_dtype.width // 8 * epi_c_stage
        if has_tile_load:
            epi_bytes += epi_smem_bytes.c_stage * epi_c_stage

        a_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8 + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        remaining_bytes = smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ab_stage = remaining_bytes // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        if epi_bytes_per_stage > 0:
            epi_stage += (remaining_bytes - ab_bytes_per_stage * ab_stage) // epi_bytes_per_stage
        return ab_stage, epi_stage, epi_c_stage

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
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if cta_tile_shape_mnk[0] % 128 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(128, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(32, cute.size(cta_tile_shape_mnk, mode=[1]))
        elif cta_tile_shape_mnk[0] % 192 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(192, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(32, cute.size(cta_tile_shape_mnk, mode=[1]))
        else:
            # In the case of tile shape 128 x N but atom_layout 1 x 2, we need to set
            # epi_tile_m = 64. If epi_tile_m = 128, the epilogue would iterate along the
            # M dimension first, then move to the N dimension. But the accumulator in registers
            # iterate along the N dimension first, then move to the M dimension.
            # We could change the epilogue to accommodate this,
            # but it's easier to just set epi_tile_m = 64.
            n_perf = 64 if element_type is not None and element_type.width == 8 else 32
            tile_m = math.gcd(64, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(n_perf, cute.size(cta_tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        a_layout: LayoutEnum,
        b_dtype: Type[cutlass.Numeric],
        b_layout: LayoutEnum,
        ab_stage: int,
        d_dtype: Optional[Type[cutlass.Numeric]],
        d_layout: LayoutEnum,
        epi_stage: int,
        c_dtype: Optional[Type[cutlass.Numeric]],
        c_layout: Optional[LayoutEnum],
        epi_c_stage: int,
    ) -> Tuple[
        cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout, Optional[cute.ComposedLayout]
    ]:
        """Create shared memory layouts for A, B, and C tensors.

        :param cta_tile_shape_mnk: CTA tile shape (M,N,K)
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param d_dtype: Data type for output matrix D
        :type d_dtype: type[cutlass.Numeric]
        :param d_layout: Layout enum for the output matrix C
        :type d_layout: LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K
        b_is_k_major = b_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K
        a_major_mode_size = cta_tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, a_dtype, a_major_mode_size),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))

        b_major_mode_size = cta_tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, b_dtype, b_major_mode_size),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        epi_smem_layout_staged = None
        if d_dtype is not None:
            epi_smem_layout_staged = quack_sm90_utils.make_smem_layout_epi(
                d_dtype, d_layout, epi_tile, epi_stage
            )

        epi_c_smem_layout_staged = None
        if c_dtype is not None:
            assert c_layout is not None
            epi_c_smem_layout_staged = quack_sm90_utils.make_smem_layout_epi(
                c_dtype, c_layout, epi_tile, epi_c_stage
            )

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
            epi_c_smem_layout_staged,
        )

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Optional[Type[cutlass.Numeric]],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if a_dtype not in {Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2}:
            is_valid = False
        # tested b_dtype
        if b_dtype not in {Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2}:
            is_valid = False
        if acc_dtype not in {Float32, Float16}:
            is_valid = False
        # tested d_dtype
        if d_dtype not in {
            None,
            Float32,
            Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # make sure a_dtype == b_dtype for Float16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        # make sure a_dtype.width == b_dtype.width (i.e, Float8E4M3FN or Float8E5M2)
        if a_dtype.width != b_dtype.width:
            is_valid = False

        # for Float8 types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (b_dtype.width == 8 and b_major != "k"):
            is_valid = False
        return is_valid
