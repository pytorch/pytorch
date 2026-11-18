# Copyright (c) 2025-2026, QuACK team.

# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py

from typing import Tuple, Type, Callable, Optional, Union
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
from cutlass.utils import LayoutEnum, SmemPartition


from torch._vendor.quack import layout_utils
from torch._vendor.quack import pipeline_checks
from torch._vendor.quack.gemm_base import GemmTmaBase, NamedBarrierGemm, reinterpret_packed_fp6
from torch._vendor.quack.gemm_config import SplitKMode
from torch._vendor.quack.operand_transform.transform import TransformAOperand
from torch._vendor.quack.tile_scheduler import TileSchedulerOptions, ag_wait_m_tile
from torch._vendor.quack.varlen_utils import VarlenArguments, VarlenManager

# return PipelineStateWAdvance instead of PipelineState
from torch._vendor.quack.pipeline import PipelineAsync as QuackPipelineAsync, make_pipeline_state
import torch._vendor.quack.copy_utils as copy_utils
import torch._vendor.quack.sm90_utils as quack_sm90_utils

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

    Warp roles and pipeline schedule (arrive counts validated by quack.pipeline_checks
    at construction):

    Roles per CTA: mma_warp_groups warpgroups (128 threads each) running WGMMA + epilogue
    (cooperative: all on one tile; pingpong: two warpgroups alternate tiles), plus one
    producer warpgroup containing the AB-load warp (or 4 cp.async warps when gather_A),
    the C-load warp, and the scheduler warp.

    Pipelines (producer role -> consumer role):
      ab       TmaAsync: AB-load -> MMA warpgroups.  full: TMA tx bytes (+cp.async lane
               arrives when gather_A); empty: 1 arrive per mma warp, delivered to every
               CTA in this CTA's A/B-multicast peer set.
      sched    Async:    scheduler -> mma + load warps. full: armed as tx barrier by the
               producer; empty: 1 arrive per consumer warp, every CTA routed to CTA 0
               (pingpong halves the participating mma warps unless varlen_k).
      epi (C)  TmaAsync: C-load -> epilogue warps.   full: TMA tx; empty: 1 arrive per
               epi warp (elected lane).
      epi_store TmaStore: epilogue TMA-store completion pacing (bulk-group, no mbarrier).

    Per-role timeline (steady state):
      scheduler: [per tile]  sched.acquire -> produce next tile slot (arms tx)
      AB-load:   [per tile]  sched slot read -> sched.release
                 [per k]     ab.acquire (wait empty + arm tx) -> TMA A,B (+cp.async A if
                 gather_A, committed via cp.async.mbarrier.arrive)
      MMA wg:    [per k]     ab full wait -> wgmma -> ab.release (per warp, to mcast peers)
                 [tile end]  epilogue: per subtile: epi wait (if C) -> regs -> smem ->
                 epi_store.acquire -> TMA store -> epi.release

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
    # Base (pre-refinement) C-load stage depth in _compute_stages: SM90's TMA
    # epilogue dispatch policy StagesC = min(EpiTiles, 4). SM120 overrides to
    # 2 (its CUTLASS builder uses StagesC = StagesD = min(EpiTiles, 2) — the
    # 100 KB smem budget can't afford 4 upfront C stages without dropping an
    # AB stage); the leftover refinement below still deepens C when smem is
    # free.
    epi_c_stage_base = 4
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
        split_k: int = 1,
        split_k_mode: int = SplitKMode.SERIAL,
        mma_is_rs: bool = False,
        transform_a: Optional[Callable] = None,
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
        # The MMA compute dtype for A. Without a transform, mA must arrive
        # typed exactly this; a layout-owning transform decouples storage
        # (self.a_dtype, from the tensor) from compute and must produce
        # mma_a_dtype fragments.
        self.mma_a_dtype = a_dtype
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        self.use_clc_persistence = use_clc_persistence
        if self.use_clc_persistence:
            assert self.arch == 100
        self.use_pdl = use_pdl
        if self.pingpong:
            assert self.is_persistent, "Pingpong gemm requires persistent scheduler"
        # A-operand transform (quack/operand_transform/): a factory gemm ->
        # TransformA, instantiated below after the default register budgets so
        # it can override them. The kernel is agnostic to what the transform
        # computes — it consumes only the declarative contract (A layout
        # ownership, tile_k, the in-kernel copy_block hook). A transform
        # implies the RS mainloop; for layout-owning transforms mA crosses the
        # boundary in its own storage format.
        self._transform_a_factory = transform_a
        if transform_a is not None:
            mma_is_rs = True
        # Transforms own their accumulation policy: an 8-bit-MMA transform
        # (w4a8) does its own scaled per-k-tile promotion (transform.promote),
        # not the plain fp8 slow-accum path.
        self.fp8_slow_accum = not fp8_fast_accum and a_dtype.width == 8 and transform_a is None
        # RS mainloop: A comes from registers (canonical ldmatrix s2r load from smem) instead
        # of the SS descriptor, CUTLASS rs_warpspecialized style — one tile-wide fragment, s2r
        # load of k16 block b+1 interleaved between WGMMA(b) and WGMMA(b+1), one commit group per
        # block, wait_group(mma_k - 2).
        self.mma_is_rs = mma_is_rs
        if mma_is_rs:
            assert not self.fp8_slow_accum, "mma_is_rs requires 16-bit A for now"
        self.gather_A = gather_A
        self.concat_layout = concat_layout or ()
        if gather_A:
            assert cluster_shape_mnk[1] == 1, "Cluster shape N must be 1 for gather A "
        self._init_split_k(split_k, split_k_mode)

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
        if self.mma_is_rs:
            # A fragment registers: per-warpgroup M extent x tile_K 16-bit
            # elements across 128 threads. tile_K may still be 0 here
            # (defaulted in _setup_tiled_mma); estimate with 64.
            tile_k_est = self.cta_tile_shape_mnk[2] or 64
            regs_per_thread += (
                (self.cta_tile_shape_mnk[0] // self.atom_layout_mnk[0])
                * tile_k_est
                * 2
                // (self.num_threads_per_warp_group * 4)
            )
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

        # TransformA: created after the default register budgets above so it
        # can override them (and occupancy) per its config. The transform may
        # install an aux A-side operand (per-stage strip riding the AB
        # pipeline); the aux facility itself is transform-agnostic.
        self.transform_a = None
        self.aux_a = None
        if transform_a is not None:
            self.transform_a = transform_a(self)
            self.aux_a = self.transform_a.aux

        # Blockscaled (real SFA/SFB operands) is only supported by the SM120
        # subclass; the __call__/kernel seams below are gated on this flag.
        self.blockscaled = False
        self.sf_vec_size = None
        self.sfa_smem_layout_staged = None
        self.sfb_smem_layout_staged = None
        # B's MMA element type when it differs from the storage dtype (packed
        # fp6 crosses the FFI boundary as raw bytes); SM120-blockscaled only.
        self.b_mma_dtype_cfg = None

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

    def _sf_smem_bytes_per_stage(self) -> int:
        """SFA+SFB smem bytes per AB pipeline stage; nonzero only for blockscaled."""
        return 0

    def _setup_tiled_mma(self):
        """Set up tiled MMA and tile K dimension. Override for different MMA types."""
        # The MMA computes A in the constructor-declared a_dtype (self.
        # mma_a_dtype) — a transform never changes that; a layout-owning
        # transform only decouples it from A's STORAGE dtype (self.a_dtype,
        # from the tensor), producing mma_a_dtype fragments from whatever mA
        # holds. The fragment major likewise comes from the tensor layout,
        # except for layout-owning transforms (a blob has no natural major;
        # the transform declares the fragment's).
        a_major_mode = self.a_layout.sm90_mma_major_mode()
        if self.transform_a is not None and self.transform_a.owns_a_layout:
            a_major_mode = self.transform_a.a_major_mode
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.mma_a_dtype,
            self.b_dtype,
            a_major_mode,
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.cta_tile_shape_mnk[1] // self.atom_layout_mnk[1]),
            a_source=(
                warpgroup.OperandSource.RMEM if self.mma_is_rs else warpgroup.OperandSource.SMEM
            ),
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
        if self.transform_a is not None and self.transform_a.tile_k is not None:
            assert tile_k == self.transform_a.tile_k, (
                f"transform_a requires tile_K == {self.transform_a.tile_k}, got {tile_k}"
            )
        if self.mma_is_rs:
            # Slot 0 is reloaded while later blocks' WGMMAs are in flight;
            # wait_group(mma_k - 2) guarantees its reader retired only with
            # >= 4 blocks per tile (same floor as CUTLASS rs mixed-input).
            assert tile_k // mma_inst_shape_k >= 4, (
                f"mma_is_rs needs >= 4 k16 blocks per tile, got tile_k={tile_k}"
            )
        self.cta_tile_shape_mnk = (*self.cta_tile_shape_mnk[:2], tile_k)

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

        # Compute stage before compute smem layout. Smem accounting and layout
        # atoms use the smem STORAGE dtypes: identical to the element dtypes
        # except SM120 mixed-blockscaled fp4 operands, whose 16U4_ALIGN8B smem
        # footprint is one byte per element (Int8, set by _setup_tiled_mma).
        self.ab_stage, self.epi_stage, self.epi_c_stage = self._compute_stages(
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_smem_dtype,
            self.b_smem_dtype,
            self.d_dtype,
            self.c_dtype,
            epilogue_args,
            cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}"),  # smem_capacity
            self.occupancy,
            self.epi_smem_warp_shape_mnk(),
            # layout-owning transform_a: A's smem bytes come from the
            # transform, not the (tile_M, tile_K) shape
            a_bytes_per_stage_override=(
                self.transform_a.a_bytes_per_stage()
                if self.transform_a is not None and self.transform_a.owns_a_layout
                else None
            ),
            # aux A-side operand (e.g. a scale strip) and blockscaled SFA/SFB
            # both ride the AB stages
            ab_extra_bytes_per_stage=(
                (self.aux_a.bytes_per_stage() if self.aux_a is not None else 0)
                + self._sf_smem_bytes_per_stage()
            ),
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
            self.a_smem_dtype,
            self.a_layout,
            self.b_smem_dtype,
            self.b_layout,
            self.ab_stage,
            self.d_dtype,
            self.d_layout,
            self.epi_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_c_stage,
        )
        if const_expr(self.transform_a is not None and self.transform_a.owns_a_layout):
            # the transform owns A's smem layout (its storage format, not
            # the (tile_M, tile_K) shape)
            self.a_smem_layout_staged = self.transform_a.make_a_smem_layout_staged(self.ab_stage)
        self.aux_a_smem_layout_staged = (
            self.aux_a.make_smem_layout_staged(self.ab_stage) if self.aux_a is not None else None
        )

    @cute.jit
    def __call__(
        self,
        # a plain (M, K) tensor, or the TransformAOperand bundle (blob +
        # optional aux strip) of a layout-owning transform
        mA: Union[cute.Tensor, TransformAOperand],
        mB: cute.Tensor,
        mD: Optional[cute.Tensor],
        mC: Optional[cute.Tensor],
        epilogue_args: tuple,
        scheduler_args: TileSchedulerOptions,
        varlen_args: Optional[VarlenArguments],
        stream: cuda.CUstream,
        # Unified SM90/SM100/SM120 signature: the trailing SF slots exist on
        # every TMA arch (the compiled TVM-FFI arg spec bakes the full arity,
        # defaults included, so hosts must always pass them — see
        # launch_gemm). Real scale factors are SM120-blockscaled only.
        mSFA: Optional[cute.Tensor] = None,
        mSFB: Optional[cute.Tensor] = None,
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
        a_owned = const_expr(self.transform_a is not None and self.transform_a.owns_a_layout)
        # Transforms with runtime operands bundle A and the optional aux strip
        # into ONE mA argument (TransformAOperand) — the host layer never
        # learns the bundle's anatomy, and the signature arity stays fixed for
        # plain GEMMs. For layout-owning transforms blob is the repacked
        # storage; for value transforms it is the plain (M, K) operand and
        # continues down the standard path below.
        if const_expr(self.blockscaled):
            assert mSFA is not None and mSFB is not None
            # Dense unbatched (rank-5) SFs: prepend the trivial batch mode so the
            # rest of the kernel sees the usual (l, rm/rn, rk, 32, 4, 4) shape.
            if const_expr(cute.rank(mSFA) == 5):
                mSFA = layout_utils.expand(mSFA, 0, 1)
            if const_expr(cute.rank(mSFB) == 5):
                mSFB = layout_utils.expand(mSFB, 0, 1)
        else:
            # the slots are part of the unified signature; only blockscaled
            # (SM120) kernels consume them
            assert mSFA is None and mSFB is None, "mSFA/mSFB require a blockscaled GEMM"
        mAuxA = None
        if const_expr(isinstance(mA, TransformAOperand)):
            assert self.transform_a is not None, "TransformAOperand requires a transform_a"
            mA, mAuxA = mA.blob, mA.sf
        if const_expr(self.aux_a is not None):
            assert mAuxA is not None, "the transform's aux operand needs mA.sf"
        elif const_expr(self.transform_a is not None and self.transform_a.aux_raw):
            assert mAuxA is not None, "the transform's raw aux operand (e.g. seed) needs mA.sf"
        if const_expr(not a_owned):
            # Tensors arrive batch-first: rotate (l, x, y) -> (x, y, l) at trace time.
            # Dense rank-2 operands get a trivial batch mode appended instead.
            mA, mB, mD, mC, epilogue_args = self.rotate_batch_last(
                mA, mB, mD, mC, epilogue_args, append_batch_if_2d=const_expr(varlen_args is None)
            )

            # Concat layout: interleave the non-contiguous dim (detected via leading_dim).
            mA, mB, mD, mC = [
                layout_utils.concat_to_interleave(mT, 1 - mT.leading_dim)
                if const_expr(name in self.concat_layout and mT is not None)
                else mT
                for name, mT in [("A", mA), ("B", mB), ("out", mD), ("C", mC)]
            ]
        else:
            # Layout-owning transform: mA (the storage blob) and mAuxA cross
            # kernel-native and untouched; B/D/C are ordinary operands and
            # rotate as usual (2-D callers get the trivial batch appended).
            assert mD is not None, "a layout-owning transform_a requires an output tensor D"
            _, mB, mD, mC, epilogue_args = self.rotate_batch_last(
                None, mB, mD, mC, epilogue_args, append_batch_if_2d=const_expr(varlen_args is None)
            )

        if const_expr(self.blockscaled):
            # Packed 6-bit operands cross the FFI boundary as raw bytes (torch
            # has no fp6 dtype): reinterpret (mn, 3k/4[, l]) Uint8 as
            # (mn, k[, l]) fp6, same as the SM100 path.
            if const_expr(self.mma_a_dtype.width == 6):
                mA = reinterpret_packed_fp6(mA, self.mma_a_dtype)
            if const_expr(self.b_mma_dtype_cfg is not None and self.b_mma_dtype_cfg.width == 6):
                mB = reinterpret_packed_fp6(mB, self.b_mma_dtype_cfg)

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type if mD is not None else None
        self.c_dtype = mC.element_type if mC is not None else None
        self.sf_dtype = mSFA.element_type if const_expr(mSFA is not None) else None
        # smem storage / TMA-internal dtypes default to the element dtypes; the
        # SM120 mixed-blockscaled path overrides sub-byte (fp4/fp6) sides to
        # Int8 in _setup_tiled_mma (16U4_ALIGN8B / 16U6_ALIGN16B footprint,
        # see gemm_sm120.py).
        self.a_smem_dtype = self.a_dtype
        self.b_smem_dtype = self.b_dtype
        self.a_tma_internal_dtype = None
        self.b_tma_internal_dtype = None
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD) if mD is not None else None
        self.c_layout = LayoutEnum.from_tensor(mC) if mC is not None else None

        if const_expr(not a_owned):
            # (For layout-owning transforms, self.a_dtype is the storage
            # format's dtype — e.g. a uint8 blob — and only the transform
            # relates it to the mma_a_dtype fragments it produces.)
            if const_expr(self.a_dtype != self.mma_a_dtype):
                raise TypeError(
                    f"A arrived as {self.a_dtype} but the GEMM was built for {self.mma_a_dtype}"
                )
            if const_expr(self.a_dtype.width == 16 and self.a_dtype != self.b_dtype):
                raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
            # Blockscaled (SM120) admits mixed-width fp4 x fp8 pairs; its dtype
            # legality is enforced in GemmSm120._setup_tiled_mma.
            if const_expr(not self.blockscaled):
                if const_expr(self.a_dtype.width != self.b_dtype.width):
                    raise TypeError(
                        f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}"
                    )
                if const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
                    raise TypeError("a_dtype should be float16 or float8")
            if const_expr(self.mma_is_rs and self.a_dtype.width != 16):
                # The canonical RS s2r load is ldmatrix (trans for m-major A),
                # which is 16-bit only; other-width RS variants bring their own
                # produce (TransformA).
                raise TypeError("mma_is_rs requires a 16-bit A dtype")

        if const_expr(varlen_args is None):
            varlen_args = VarlenArguments()
        assert (varlen_args.mAIdx is not None) == self.gather_A
        varlen_m = varlen_args.mCuSeqlensM is not None
        varlen_k = varlen_args.mCuSeqlensK is not None
        # Stash for epilogue ops (epi_to_underlying_arguments runs later and
        # SFD needs the varlen mode to shape its logical scale layout).
        self.varlen_m = varlen_m
        self.varlen_k = varlen_k

        self._setup_attributes(epilogue_args)

        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        if const_expr(not a_owned):
            a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
            tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = (
                self.make_tma_load_atoms_and_tensors(
                    mA,
                    mB,
                    a_smem_layout,
                    b_smem_layout,
                    varlen_k,
                    a_internal_type=self.a_tma_internal_dtype,
                    b_internal_type=self.b_tma_internal_dtype,
                    varlen_m_zero_fill=varlen_m and self.epilogue_zero_fill_varlen_m(epilogue_args),
                )
            )
        else:
            # packed-weight A: the transform owns A's TMA (blob boxes); its
            # scale-factor strip rides the SFA slots below
            a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
            tma_atom_a, tma_tensor_a = self.transform_a.make_a_tma(mA)
            tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
                mB,
                b_smem_layout,
                (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]),
                self.cluster_shape_mnk[0],
            )

        self.num_tma_load_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        if const_expr(not self.gather_A):
            self.num_tma_load_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)

        tma_atom_aux_a, tma_tensor_aux_a = None, None
        if const_expr(self.aux_a is not None):
            tma_atom_aux_a, tma_tensor_aux_a = self.aux_a.make_tma(mAuxA)
            self.num_tma_load_bytes += self.aux_a.bytes_per_stage()
        elif const_expr(self.transform_a is not None and self.transform_a.aux_raw):
            # raw aux operand (e.g. a dropout seed): crosses to the kernel in
            # the mAuxA slot untouched — no TMA atom, no smem, no pipeline
            tma_tensor_aux_a = mAuxA

        tma_atom_sfa, tma_tensor_sfa, tma_atom_sfb, tma_tensor_sfb = None, None, None, None
        if const_expr(self.blockscaled):
            # Rebuild the SF logical (M/N, K, L) layouts from the blocked scale
            # tensors' actual strides so non-packed buffers (slices of larger
            # scale tensors) work; only the inner 512-B atom must be contiguous.
            # For varlen the SF buffer is padded (tile-aligned per-batch padding
            # along M for varlen_m / along K for varlen_k), so its extent comes
            # from the SF tensor itself, not the packed operand.
            if const_expr(cute.rank(mA) == 3):
                sfa_shape = mA.shape
            elif const_expr(varlen_m):
                sfa_shape = (mSFA.shape[1] * 128, mA.shape[1])
            else:  # varlen_k
                sfa_shape = (mA.shape[0], mSFA.shape[2] * 128)
            sfa_layout = layout_utils.tile_atom_to_shape_SF_strided(
                sfa_shape, self.sf_vec_size, mSFA.stride
            )
            mSFA = cute.make_tensor(mSFA.iterator, sfa_layout)
            if const_expr(cute.rank(mB) == 3):
                sfb_shape = mB.shape
            else:  # varlen_k
                sfb_shape = (mB.shape[0], mSFB.shape[2] * 128)
            sfb_layout = layout_utils.tile_atom_to_shape_SF_strided(
                sfb_shape, self.sf_vec_size, mSFB.stride
            )
            mSFB = cute.make_tensor(mSFB.iterator, sfb_layout)
            sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, 0))
            sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, 0))
            # The SF layouts have stride-0 broadcast modes (sf_vec_size elements
            # share one scale), which TMA can't express in E8M0/E4M3 element
            # units; Int16 internal type views each 512-B atom as 256 x Int16
            # boxes (same trick as the SM100 path and the CUTLASS SM120 example).
            tma_atom_sfa, tma_tensor_sfa = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mSFA,
                sfa_smem_layout,
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]),
                internal_type=cutlass.Int16,
            )
            tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mSFB,
                sfb_smem_layout,
                (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]),
                internal_type=cutlass.Int16,
            )
            self.num_tma_load_bytes += cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            self.num_tma_load_bytes += cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)

        if const_expr(self.split_k > 1):
            assert mD is not None, "split_k requires an output tensor D"
        (
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
        ) = self.make_tma_epilogue_atoms_and_tensors(mD, mC, epilogue_args, varlen_m)

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
        aux_a_smem_size = 0
        if const_expr(self.aux_a is not None):
            aux_a_smem_size = cute.cosize(self.aux_a_smem_layout_staged)
        sf_dtype_storage = self.sf_dtype if self.blockscaled else Int32
        sfa_smem_size = cute.cosize(self.sfa_smem_layout_staged) if self.blockscaled else 0
        sfb_smem_size = cute.cosize(self.sfb_smem_layout_staged) if self.blockscaled else 0

        @cute.struct
        class SharedStorage:
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
                cute.struct.MemRange[self.a_smem_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_smem_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sAuxA: cute.struct.Align[
                cute.struct.MemRange[
                    self.aux_a.dtype if self.aux_a is not None else Int32, aux_a_smem_size
                ],
                128,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[sf_dtype_storage, sfa_smem_size],
                128,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[sf_dtype_storage, sfb_smem_size],
                128,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a if const_expr(not self.gather_A) else mA,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            epilogue_params,
            varlen_params,
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            tma_atom_aux_a,
            tma_tensor_aux_a,
            self.aux_a_smem_layout_staged,
            tile_sched_params,
            TileSchedulerCls,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            # occupancy > 1 (e.g. W4 decode shapes) needs the launch bound so
            # ptxas caps registers for 2 resident CTAs
            min_blocks_per_mp=self.occupancy,
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
        # blockscaled SFA/SFB slots (real scale factors are SM120-only; always
        # None here)
        tma_atom_sfa: Optional[cute.CopyAtom],
        mSFA_mkl: Optional[cute.Tensor],
        tma_atom_sfb: Optional[cute.CopyAtom],
        mSFB_nkl: Optional[cute.Tensor],
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl: Optional[cute.Tensor],
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        epilogue_params,
        varlen_params: VarlenManager.Params,
        cluster_layout_mnk: cute.Layout,
        # plain Layout for layout-owning transforms (unswizzled blob smem)
        a_smem_layout: Union[cute.ComposedLayout, cute.Layout],
        b_smem_layout: cute.ComposedLayout,
        sfa_smem_layout: Optional[cute.Layout],
        sfb_smem_layout: Optional[cute.Layout],
        epi_smem_layout: cute.ComposedLayout,
        epi_c_smem_layout: cute.ComposedLayout,
        tma_atom_aux_a: Optional[cute.CopyAtom],
        mAuxA_mkl: Optional[cute.Tensor],
        aux_a_smem_layout: Optional[cute.Layout],
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
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

        from cutlass.cute.experimental import iket

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
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c, tma_atom_aux_a):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # Alloc and init AB full/empty + ACC full mbar (pipeline)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
        )
        epi_pipeline = None
        if const_expr(has_epi_load):
            epi_pipeline = self.make_epi_pipeline(tx_count=self.epi_load_bytes_per_stage)
        sched_pipeline = None
        sched_data = None
        if const_expr(self.is_persistent):
            sched_pipeline = self.make_sched_pipeline(
                cluster_layout_mnk,
                # split_k > 1 makes per-tile k-tile counts dynamic, so pingpong consumes
                # work tiles one at a time, exactly like varlen_k.
                varlen_k=varlen_k or self.split_k > 1,
            )
            # Keep scheduler scratch out of SharedStorage. A small buffer before
            # the 1024-byte aligned epilogue tensors can add a 1 KiB pad; CLC
            # responses also use i128 copies, so this stays 16-byte aligned.
            # No drain-mailbox tail (+6 Int32, cf. gemm_sm100): this kernel never
            # calls cancel_pending_tail — add the tail if that ever changes.
            sched_data = smem.allocate_tensor(
                Int32,
                cute.make_layout((4, self.sched_stage)),
                byte_alignment=16,
                partition=SmemPartition.RESERVED,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)

        # Generate smem tensor A/B
        a_owned = const_expr(self.transform_a is not None and self.transform_a.owns_a_layout)
        if const_expr(not a_owned):
            sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        else:
            # TMA-facing staged blob view (plain layout, no swizzle); the
            # transform's per-thread math view recasts the same bytes inside
            # make_copy_block
            sA = storage.sA.get_tensor(a_smem_layout)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sAuxA = None
        if const_expr(self.aux_a is not None):
            sAuxA = storage.sAuxA.get_tensor(aux_a_smem_layout)
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
                (
                    cute.size(mA_mkl, mode=[0])
                    if const_expr(not a_owned)
                    else cute.size(mD_mnl, mode=[0])
                )
                if varlen_k or varlen_params.mAIdx is None
                else varlen_params.mAIdx.shape[0]
            ),
            len_k_static=Int32(cute.size(mB_nkl, mode=[1])),
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
                # block_copy's lowering wants the coordinate held fixed by the
                # multicast mask: A is same-M across N peers, while B is
                # same-N across M peers. Degenerate cluster dimensions are
                # left for the compiler lowering to simplify.
                a_tma_multicast = {
                    "cluster_shape": self.cluster_shape_mnk[:2],
                    "multicast_dim": "M",
                }
                b_tma_multicast = {
                    "cluster_shape": self.cluster_shape_mnk[:2],
                    "multicast_dim": "N",
                }

                # Persistent tile scheduling loop
                is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                if const_expr(cute.size(cluster_layout_mnk) > 1):
                    is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                ag_last_gate = Int32(-1)  # 1-entry satisfied-gate cache (see ag_wait_m_tile)
                ab_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                while work_tile.is_valid_tile:
                    # (pid_m, pid_n, split_idx | None, batch_idx), decoded by the scheduler
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
                    # AllGather+GEMM: block until this tile's M-shard of A has
                    # been pushed into local HBM by the owner rank. Only the
                    # load warp gates — the MMA/epilogue warps are downstream
                    # of the AB pipeline. With the ring-rotated shard-major
                    # schedule the flag is normally already set and this is a
                    # single L2 load. getattr: the varlen scheduler Params
                    # classes have no ag_* fields at all.
                    if const_expr(getattr(tile_sched_params, "ag", None) is not None):
                        iket.range_push("ag_wait")
                        ag_last_gate = ag_wait_m_tile(
                            tile_sched_params,
                            tile_coord_mnkl[0],
                            self.cluster_shape_mnk[0],
                            ag_last_gate,
                        )
                        iket.range_pop()
                    iket.range_push("tma_load")
                    # Local_tile partition global tensors
                    copy_A, prefetch_A = None, None
                    if const_expr(a_owned):
                        # the transform owns A's gmem interpretation
                        gA_owned = self.transform_a.a_gmem_slice(mA_mkl, tile_coord_mnkl, batch_idx)
                        copy_A = copy_utils.tma_get_block_copy_fn(
                            tma_atom_a,
                            src_tensor=gA_owned,
                            dst_tensor=sA,
                            tma_multicast=a_tma_multicast,
                        )
                    elif const_expr(not self.gather_A):
                        mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                        # (bM, bK, RestK)
                        gA_mk = cute.local_tile(
                            mA_mk,
                            cute.select(self.cta_tile_shape_mnk, [0, 2]),
                            (tile_coord_mnkl[0], None),
                        )
                        #  TMA load A partition_S/D
                        copy_A = copy_utils.tma_get_block_copy_fn(
                            tma_atom_a,
                            src_tensor=gA_mk,
                            dst_tensor=sA,
                            tma_multicast=a_tma_multicast,
                        )
                    else:
                        copy_A, prefetch_A = self._make_gather_A_copy(
                            mA_mkl, sA, varlen_manager, tile_coord_mnkl, batch_idx
                        )
                    copy_AuxA = None
                    if const_expr(self.aux_a is not None):
                        # aux A-side operand: one box per k-tile alongside A/B
                        gAux = self.aux_a.gmem_slice(mAuxA_mkl, tile_coord_mnkl, batch_idx)
                        copy_AuxA = copy_utils.tma_get_block_copy_fn(
                            tma_atom_aux_a,
                            src_tensor=gAux,
                            dst_tensor=sAuxA,
                            # small-box aux operands (e.g. 128 B scale strips)
                            # may opt out of the A-side multicast: each CTA
                            # loads its own copy instead of splitting the box
                            tma_multicast=a_tma_multicast
                            if const_expr(getattr(self.aux_a, "multicast", True))
                            else None,
                        )
                    # (bN, bK, RestK)
                    gB_nk = cute.local_tile(
                        varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                        cute.select(self.cta_tile_shape_mnk, [1, 2]),
                        (tile_coord_mnkl[1], None),
                    )
                    # TMA load B partition_S/D
                    copy_B = copy_utils.tma_get_block_copy_fn(
                        tma_atom_b,
                        src_tensor=gB_nk,
                        dst_tensor=sB,
                        tma_multicast=b_tma_multicast,
                    )
                    len_k = varlen_manager.len_k(batch_idx)
                    k_tile_total = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    k_tile_start, k_tile_cnt = tile_scheduler.get_split_k_tile_range(
                        k_tile_total, split_idx
                    )
                    if const_expr(not self.gather_A):
                        ab_producer_state = self.load_tma(
                            ab_pipeline,
                            ab_producer_state,
                            [copy_A, copy_B, copy_AuxA],
                            k_tile_cnt,
                            k_tile_start=k_tile_start,
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
                    iket.range_pop()
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                if const_expr(self.pingpong and not varlen_k and self.split_k == 1):
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
            transform_promote = const_expr(
                self.transform_a is not None and self.transform_a.promote
            )
            acc_slow = None
            if const_expr(self.fp8_slow_accum or transform_promote):
                acc_slow = cute.make_rmem_tensor(acc.shape, self.acc_dtype)
            promote_fn = None
            if const_expr(not self.mma_is_rs):
                mma_fn = partial(quack_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)
            else:
                # tCrA is the (unstaged, tile-wide) RS fragment; copy_block is
                # the mainloop's produce seam — canonical ldmatrix s2r load, or
                # a transform's own produce (e.g. blob LDS + dequant)
                if const_expr(self.transform_a is not None):
                    copy_block = self.transform_a.make_copy_block(
                        tiled_mma,
                        sA,
                        tCrA,
                        tidx,
                        warp_group_idx,
                        sAux=sAuxA,
                        mAux=mAuxA_mkl if const_expr(self.transform_a.aux_raw) else None,
                    )
                    if const_expr(transform_promote):
                        # slow-accum transform: WGMMAs write acc as the
                        # per-k-tile wave; the transform folds it into
                        # acc_slow at each tile's drain
                        promote_fn = partial(self.transform_a.promote_acc, acc_slow, acc)
                else:
                    copy_block = self.canonical_a_load(tiled_mma, sA, tidx, tCrA)

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
                    if const_expr(not varlen_k and self.split_k == 1):
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        ab_read_state.advance_iters(k_tile_cnt_static)
                    else:
                        # varlen_k and split_k > 1 both make the per-tile k-tile count dynamic
                        batch_idx_pp, split_idx_pp = (
                            work_tile.tile_idx[3],
                            work_tile.tile_idx[2],
                        )
                        len_k = varlen_manager.len_k(batch_idx=batch_idx_pp)
                        k_tile_total = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                        _, k_tile_cnt = tile_scheduler.get_split_k_tile_range(
                            k_tile_total, split_idx_pp
                        )
                        ab_read_state.advance_iters(k_tile_cnt)
                        # Under split-K, only finalizer tiles run the epilogue (and thus
                        # produce/consume C stages); the peer advance must match.
                        c_cnt = Int32(c_tile_cnt)
                        if const_expr(
                            self.split_k > 1 and self.split_k_mode != SplitKMode.SEPARATE
                        ):
                            if split_idx_pp != self.split_k - 1:
                                c_cnt = Int32(0)
                        epi_read_state.advance_iters(c_cnt)
                        epi_producer_state.advance_iters(c_cnt)
                    # TODO: do we need to check if work_tile is valid?
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
            while work_tile.is_valid_tile:
                # (pid_m, pid_n, split_idx | None, batch_idx), decoded by the scheduler
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
                len_k = varlen_manager.len_k(batch_idx)
                k_tile_total = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                k_tile_start_mma, k_tile_cnt = tile_scheduler.get_split_k_tile_range(
                    k_tile_total, split_idx
                )
                if const_expr(self.mma_is_rs and self.transform_a is not None):
                    if const_expr(self.transform_a.uses_work_tile):
                        # per-work-tile register state (e.g. dropout's per-row
                        # RNG coordinates); every copy_block until the next
                        # hook — incl. the slot-0 preloads — is this tile's
                        self.transform_a.on_work_tile(tile_coord_mnkl)
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                iket.range_push("mma")
                if const_expr(not self.mma_is_rs):
                    ab_read_state = self.mma(
                        ab_pipeline,
                        ab_read_state,
                        mma_fn,
                        acc,
                        acc_slow,
                        k_tile_cnt,
                        warp_group_idx,
                    )
                else:
                    ab_read_state = self.mma_rs_interleaved(
                        ab_pipeline,
                        ab_read_state,
                        tiled_mma,
                        acc,
                        k_tile_cnt,
                        warp_group_idx,
                        copy_block,
                        tCrA,
                        tCrB,
                        k_tile_start=k_tile_start_mma,
                        promote_fn=promote_fn,
                    )
                    if const_expr(transform_promote):
                        # the epilogue reads acc (mirrors fp8_slow_accum)
                        if k_tile_cnt > 0:
                            acc.store(acc_slow.load())
                if const_expr(varlen_k or self.split_k > 1):
                    if k_tile_cnt == 0:
                        acc.fill(0.0)
                iket.range_pop()

                # EPILOGUE
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, "epi")
                iket.range_push("epilogue")

                copy_D = None
                if const_expr(has_D):
                    # Staged split-K: D is the f32 partials workspace, whose batch mode is the
                    # combined (l * split_k + split) index from the scheduler.
                    d_batch_idx = batch_idx
                    if const_expr(self.split_k > 1 and self.split_k_mode == SplitKMode.SEPARATE):
                        d_batch_idx = tile_scheduler.get_combined_batch_idx(batch_idx, split_idx)
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d,
                        varlen_manager.offset_batch_epi(mD_mnl, d_batch_idx),
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

                # Split-K (serial/parallel): non-finalizing splits commit raw f32 partials
                # to the tile's workspace and skip the epilogue; the last split waits for
                # the tile's completion flag and runs the full epilogue on the summed
                # accumulator (CUTLASS-3.x stream-K fixup semantics).
                epi_fn = partial(
                    self.epilogue,
                    epilogue_params,
                    epi_smem_tensors,
                    epi_pipeline,
                    epi_store_pipeline,
                    epi_read_state,
                    epi_producer_state,
                    self.epi_tile,
                    # load_acc_subtile is the one argument left unbound
                    tRS_rD=tRS_rD,
                    tRS_rC=tRS_rC,
                    tiled_copy_t2r=None,  # Sm100 only
                    tiled_copy_r2s=tiled_copy_r2s,
                    tRS_sD=tRS_sD,
                    tiled_copy_s2r=tiled_copy_s2r,
                    tSR_rC=tSR_rC,
                    tSR_sC=tSR_sC,
                    copy_D=copy_D,
                    copy_C=copy_C,
                    tile_coord_mnkl=tile_coord_mnkl,
                    varlen_manager=varlen_manager,
                    epilogue_barrier=self.epilogue_barrier,
                    tile_scheduler=tile_scheduler,
                    tidx=tidx,
                    is_tma_warp=is_tma_warp,
                )
                epi_read_state, epi_producer_state = self.epilogue_split_k(
                    epilogue_params,
                    epi_fn,
                    load_acc_subtile,
                    tRS_rD,
                    self.epi_tile,
                    epi_read_state,
                    epi_producer_state,
                    epi_store_pipeline,
                    tile_coord_mnkl,
                    self.epilogue_barrier,
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
                iket.range_pop()

                if const_expr(not self.pingpong):
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                else:  # Skip a tile for pingpong
                    # Update starting load/store/mainloop pipeline states for the next tile
                    if const_expr(not varlen_k and self.split_k == 1):
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                        if work_tile.is_valid_tile:
                            batch_idx_pp, split_idx_pp = (
                                work_tile.tile_idx[3],
                                work_tile.tile_idx[2],
                            )
                            len_k = varlen_manager.len_k(batch_idx=batch_idx_pp)
                            k_tile_total = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                            _, k_tile_cnt = tile_scheduler.get_split_k_tile_range(
                                k_tile_total, split_idx_pp
                            )
                            ab_read_state.advance_iters(k_tile_cnt)
                            # Under split-K, only finalizer tiles run the epilogue (and
                            # thus produce/consume C stages); the peer advance must match.
                            c_cnt = Int32(c_tile_cnt)
                            if const_expr(
                                self.split_k > 1 and self.split_k_mode != SplitKMode.SEPARATE
                            ):
                                if split_idx_pp != self.split_k - 1:
                                    c_cnt = Int32(0)
                            epi_read_state.advance_iters(c_cnt)
                            epi_producer_state.advance_iters(c_cnt)
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

    def canonical_a_load(self, tiled_mma, sA, tidx, tCrA):
        """The arch's canonical A-fragment produce (the seam transforms wrap
        or replace): ldmatrix s2r, major mode from the WGMMA op. SM120
        overrides to pass the smem major explicitly (its warp-MMA op carries
        none)."""
        return quack_sm90_utils.canonical_a_load_s2r(
            tiled_mma, sA, tidx, tCrA, position_independent=True
        )

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

    @cute.jit
    def _rs_wgmma_block(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        stage_idx: Int32,
        b: cutlass.Constexpr[int],
        zero_init: cutlass.Constexpr[bool] = False,
    ):
        """One k16 block's WGMMAs as their own commit group (b is a static
        Python int: register indexing)."""
        warpgroup.fence()
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
        cute.gemm(mma_atom, acc, tCrA[None, None, b], tCrB[None, None, b, stage_idx], acc)
        warpgroup.commit_group()

    @cute.jit
    def mma_rs_interleaved(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        copy_block: Callable,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        k_tile_start: Int32 = 0,
        promote_fn: Optional[Callable] = None,
    ) -> cutlass.pipeline.PipelineState:
        """RS mainloop, CUTLASS sm90 rs_warpspecialized scheme: one tile-wide
        fragment, produce of block k+1 (``copy_block(stage, k, k_tile)`` — the
        canonical ldmatrix s2r load, or a transform's decode; ``k_tile`` the
        GLOBAL k-tile index of the produced block, split-k correct via
        ``k_tile_start``) issued between
        WGMMA(k) and WGMMA(k+1), one commit group per k16 block,
        wait_group(mma_k - 2) after each — the deepest safe wait: producing a
        slot overwrites registers whose previous reader is mma_k - 2 commit
        groups back. The produce is the caller's; the WGMMA issue and
        commit-group discipline stay here — the wait/reload safety argument
        counts THESE groups. Slot 0 is reloaded from the NEXT stage during
        the current tile's last block under the same bound. A stage is
        released at block mma_k - 3 of the following tile, the first point
        where the wait guarantees all of its WGMMAs retired (mma_k >= 4
        asserted at setup; for the canonical 4-block tile this is CUTLASS's
        wait<2> / release-at-block-1).

        ``promote_fn`` (slow-accum transforms, e.g. w4a8): acc becomes the
        per-k-tile WAVE accumulator — zero-init at every tile's block 0,
        wait_group(0) after its last block, then ``promote_fn(zero_init=
        first_tile)`` folds it into the transform's persistent accumulator.
        The drain is issued AFTER the next tile's slot-0 preload, so the
        preload's LDS + decode run under the draining WGMMAs; the caller
        copies the persistent accumulator back into acc afterwards. Per-tile
        drains only strengthen the wait/release bounds above, so the block
        schedule is unchanged."""
        mma_k = const_expr(cute.size(tCrA.shape[2]))
        promote = const_expr(promote_fn is not None)
        wgmma_block = partial(self._rs_wgmma_block, tiled_mma, acc, tCrA, tCrB)
        ab_release_state = ab_read_state.clone()
        peek = Boolean(True)
        kt = Int32(k_tile_start)  # global k-tile index of the tile being produced
        # ---- first k-tile: the produces run ahead of the WGMMAs, no reload
        # hazard, so no waits inside the block loop ----
        if 0 < k_tile_cnt:
            ab_pipeline.consumer_wait(ab_read_state, ab_pipeline.consumer_try_wait(ab_read_state))
            stage = ab_read_state.index
            ab_read_state.advance()
            if 1 < k_tile_cnt:
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
            copy_block(stage, 0, kt)
            for k in cutlass.range_constexpr(mma_k - 1):
                copy_block(stage, k + 1, kt)
                wgmma_block(stage, k, zero_init=k == 0)
            wgmma_block(stage, mma_k - 1, zero_init=False)
        # Preload slot 0 of the second tile: wait(mma_k - 1) retires exactly its reader (this
        # tile's block-0 group); the steady loop's first produce additionally needs block 1's
        # group retired, hence the post-copy wait(mma_k - 2)
        if 1 < k_tile_cnt:
            ab_pipeline.consumer_wait(ab_read_state, peek)
            warpgroup.wait_group(mma_k - 1)
            copy_block(ab_read_state.index, 0, kt + 1)
            if const_expr(not promote):
                warpgroup.wait_group(mma_k - 2)
        if const_expr(promote):
            if 0 < k_tile_cnt:
                warpgroup.wait_group(0)
                promote_fn(zero_init=True)
        # ---- steady tiles (all but the first and last) ----
        for _ in cutlass.range(max(k_tile_cnt - 2, 0), unroll=1):
            kt += 1
            stage = ab_read_state.index
            ab_read_state.advance()
            for k in cutlass.range_constexpr(mma_k):
                if const_expr(k == 0):
                    peek = ab_pipeline.consumer_try_wait(ab_read_state)
                if const_expr(k == mma_k - 1):
                    # ab_read_state advanced at tile start: its index is the
                    # NEXT tile's stage, used only for this slot-0 preload
                    ab_pipeline.consumer_wait(ab_read_state, peek)
                    copy_block(ab_read_state.index, 0, kt + 1)
                else:
                    copy_block(stage, k + 1, kt)
                wgmma_block(stage, k, zero_init=const_expr(promote) and k == 0)
                # In promote mode the previous tile is fully drained, so
                # produces of slots 1.. have no pending reader; the only
                # required intra-tile wait is the one before the slot-0
                # preload (retire this tile's block-0 group) — the rest are
                # pure DEPBAR overhead.
                if const_expr(not promote or k == mma_k - 2):
                    warpgroup.wait_group(mma_k - 2)
                if const_expr(k == mma_k - 3):
                    # earliest block whose wait leaves only THIS tile's groups pending. Eg for
                    # mma_k = 4, at k == 1, we have called wait_group(2), so the only
                    # outstanding MMAs are the current k_tile with k=0, 1, which means that the
                    # previous k_tile has retired and its stage can be released.
                    # (Promote mode: the previous tile retired at its drain.)
                    ab_pipeline.consumer_release(ab_release_state)
                    ab_release_state.advance()
            if const_expr(promote):
                warpgroup.wait_group(0)
                promote_fn()
        # ---- last tile (slot 0 already loaded; nothing to prefetch) ----
        if 1 < k_tile_cnt:
            kt += 1
            stage = ab_read_state.index
            ab_read_state.advance()
            for k in cutlass.range_constexpr(mma_k - 1):
                copy_block(stage, k + 1, kt)
                wgmma_block(stage, k, zero_init=const_expr(promote) and k == 0)
                if const_expr(not promote):
                    warpgroup.wait_group(mma_k - 2)
                if const_expr(k == mma_k - 3):
                    ab_pipeline.consumer_release(ab_release_state)
                    ab_release_state.advance()
            wgmma_block(stage, mma_k - 1, zero_init=False)
        if const_expr(self.pingpong):
            # Cue for next WG's MMA to start
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        # Drain all WGMMAs and release the final stage
        warpgroup.wait_group(0)
        if const_expr(promote):
            if 1 < k_tile_cnt:
                promote_fn()
        if 0 < k_tile_cnt:
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
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

    def epi_r2s_pair_xor(self) -> bool:
        # 32-bit D with n-major layout: the wgmma acc pairs are contiguous in
        # smem and both the SW128 (epi_tile_n % 32 == 0) and SW64
        # (epi_tile_n == 16, i.e. tile_n % 32 == 16 like 112/176) swizzle
        # atoms make STS.64 uniformly 2-way bank conflicted; the pair-XOR
        # STS.32 split is conflict-free under both (ncu-verified, no spills).
        # epi_tile_n == 8 (SW32) is not verified, keep the default store.
        # Gated off when C is present: the fp32-C epilogue sits exactly at the
        # setmaxnreg cap and the split's few extra registers spill to local
        # (measured ~150MB STL, a net loss; same for a mirrored LDS split).
        return (
            self.d_dtype is not None
            and self.d_dtype.width == 32
            and (self.d_layout is None or self.d_layout.is_n_major_c())
            and self.epi_tile[1] % 16 == 0
            and self.c_dtype is None
        )

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

    def make_sched_pipeline(self, cluster_layout_mnk: cute.Layout, varlen_k: bool):
        sched_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_layout_mnk)
        # Each warp will contribute 1 to the arrive count
        # If pingpong and varlen_k, then all 8 mma warps will participate in the scheduler barrier
        # at each round. If pingpong and not varlen_k, then only 4 mma warp will participate.
        sched_consumer_warps_per_cta = (
            self.mma_warp_groups if not (self.pingpong and not varlen_k) else 1
        ) * 4 + self.num_ab_load_warps
        consumer_arrive_cnt = sched_consumer_warps_per_cta * cluster_size
        # One arrive per consumer warp (elected lane); consumer_mask=0 routes every CTA
        # in the cluster to CTA 0's barrier. per_warp is bound to elect_one_release below.
        elect_one_release = True
        pipeline_checks.check_arrive_count(
            "sm90 sched_pipeline.consumer",
            consumer_arrive_cnt,
            pipeline_checks.async_thread_arrives(
                sched_consumer_warps_per_cta, per_warp=elect_one_release, ctas_routed=cluster_size
            ),
            warps_per_cta=sched_consumer_warps_per_cta,
            cluster_size=cluster_size,
        )
        sched_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return QuackPipelineAsync.create(
            num_stages=self.sched_stage,
            producer_group=sched_pipeline_producer_group,
            consumer_group=sched_pipeline_consumer_group,
            # If there's cluster, the consumers must arrive at the mbar of CTA 0 in the cluster.
            consumer_mask=None if const_expr(cluster_size == 1) else 0,
            defer_sync=True,
            # One arrive per consumer warp (consumer_arrive_cnt counts warps): syncwarp
            # so every lane's slot read is complete, then one elected lane signals.
            elect_one_release=elect_one_release,
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
        a_bytes_per_stage_override: Optional[int] = None,
        ab_extra_bytes_per_stage: int = 0,
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

        # Stage split mirrors CUTLASS's sm90 TMA epilogue dispatch policy
        # (sm90_get_tma_dispatch_policy): StagesD = min(EpiTiles, 2),
        # StagesC = min(EpiTiles, 4). C loads are latency-critical
        # (consumer_wait sits on the epilogue's serial path; at epi_c_stage=2
        # it measured ~15% of warp time on C-heavy epilogues like dgated at
        # 65536x2048x768), while D stores are fire-and-forget through the TMA
        # store pipeline and L2 absorbs them (producer_acquire waits measured
        # near-zero even at epi_stage=2). Narrow epi tiles keep the deeper
        # 4-stage D (per-stage bytes are small).
        epi_tiles = (cta_tile_shape_mnk[0] * cta_tile_shape_mnk[1]) // cute.size(
            cute.shape(epi_tile)
        )
        epi_stage = min(epi_tiles, 4 if epi_tile[1] <= 16 else 2)
        epi_smem_bytes = cls.epi_smem_bytes(
            epilogue_args, cta_tile_shape_mnk, epi_tile, warp_shape_mnk
        )
        has_tile_load = epi_smem_bytes.c_stage > 0
        epi_tile_elems = cute.size(cute.shape(epi_tile))
        d_bytes_per_stage = epi_tile_elems * d_dtype.width // 8 if d_dtype is not None else 0
        epi_bytes_per_stage = d_bytes_per_stage + epi_smem_bytes.d_stage
        epi_bytes = epi_smem_bytes.unstaged + epi_bytes_per_stage * epi_stage
        epi_c_stage = (
            0 if c_dtype is None and not has_tile_load else min(epi_tiles, cls.epi_c_stage_base)
        )
        if c_dtype is not None:
            epi_bytes += epi_tile_elems * c_dtype.width // 8 * epi_c_stage
        if has_tile_load:
            epi_bytes += epi_smem_bytes.c_stage * epi_c_stage

        a_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))
        a_bytes = (
            cute.size(a_shape) * a_dtype.width // 8
            if a_bytes_per_stage_override is None
            else a_bytes_per_stage_override
        )
        ab_bytes_per_stage = (
            a_bytes + cute.size(b_shape) * b_dtype.width // 8 + ab_extra_bytes_per_stage
        )
        mbar_helpers_bytes = 1024
        # SharedStorage packs [sD|sC|epi op smem][sA|sB][sAuxA] with sA/sB
        # Align[1024] and the struct size rounded up to its 1024 alignment.
        # The byte-packed sums here can't see two alignment pads (≤1KB each):
        # the epi op-smem member pads up to sA's alignment, and a non-empty
        # sAuxA (128B-granular strip/SF boxes) leaves the struct end
        # unaligned so the total rounds up. Reserve a quantum for each
        # exactly when it can be nonzero — plain kernels (no op smem, no
        # aux) keep identical stage picks. Otherwise the epi-stage
        # refinement below can fill smem to the exact byte and the real
        # struct overflows at launch.
        # (op-smem member total varies with the refined stage counts, so any
        # op smem at all reserves; the aux total is per_stage * ab_stage, so
        # a 1024-multiple per_stage provably never pads.)
        op_smem_bytes = epi_smem_bytes.unstaged + epi_smem_bytes.d_stage + epi_smem_bytes.c_stage
        align_pad_bytes = (1024 if op_smem_bytes else 0) + (
            1024 if ab_extra_bytes_per_stage % 1024 else 0
        )

        remaining_bytes = (
            smem_capacity // occupancy - mbar_helpers_bytes - align_pad_bytes - epi_bytes
        )
        ab_stage = remaining_bytes // ab_bytes_per_stage

        # Refine epilogue stages with the smem left below one more A/B stage,
        # C first (one extra C stage measured ~2.6% on dgated at
        # 65536x2048x768, leftover-to-D never won an interleaved A/B), capped
        # at min(5, epi_tiles) — the epilogue only prefetches C within the
        # current tile, so deeper than epi_tiles is dead smem. The rest goes
        # to D stages.
        leftover = remaining_bytes - ab_bytes_per_stage * ab_stage
        c_bytes_per_stage = (
            epi_tile_elems * c_dtype.width // 8 if c_dtype is not None else 0
        ) + epi_smem_bytes.c_stage
        if epi_c_stage > 0 and c_bytes_per_stage > 0:
            add = min(leftover // c_bytes_per_stage, min(5, epi_tiles) - epi_c_stage)
            if add > 0:
                epi_c_stage += add
                leftover -= add * c_bytes_per_stage
        if epi_bytes_per_stage > 0:
            epi_stage += leftover // epi_bytes_per_stage
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
