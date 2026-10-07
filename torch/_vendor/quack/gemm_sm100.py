# Copyright (c) 2025-2026, QuACK team.
# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py

from typing import Optional, Type, Tuple, Union, Callable, Literal
from functools import partial
import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu.warp import (
    LdMatrix8x8x16bOp,
    LdMatrix16x16x8bOp,
    StMatrix8x8x16bOp,
    StMatrix16x8x8bOp,
)
from cutlass import Int32, Float32, Boolean, const_expr
from cutlass.utils import LayoutEnum
from cutlass.cute.experimental import iket


from torch._vendor.quack.pipeline import (
    PipelineAsync as QuackPipelineAsync,
    PipelineCpAsync,
    PipelineTmaUmma,
    PipelineTmaCpAsyncUmma,
    PipelineUmmaAsync,
    mbarrier_arrive_release_cluster,
    mbarrier_acquire_cluster,
)
from torch._vendor.quack.dsl.smem_struct import Reserved, partitioned_struct
from torch._vendor.quack.tile_scheduler import (
    TileSchedulerOptions,
    ag_wait_m_tile,
)
from torch._vendor.quack.varlen_utils import VarlenArguments, VarlenManager
from torch._vendor.quack.gemm_base import GemmTmaBase, NamedBarrierGemm
from torch._vendor.quack.gemm_config import SplitKMode
from torch._vendor.quack import layout_utils
from torch._vendor.quack import pipeline_checks
import torch._vendor.quack.copy_utils as copy_utils
import torch._vendor.quack.sm100_utils as quack_sm100_utils
from torch._vendor.quack.epilogue.quantize_out import active_row_sfd_reqs
from torch._vendor.quack.layout_utils import tile_atom_to_shape_SF_strided

# TOMBSTONE — AllGather+GEMM arrival-gate placement experiment (2026-07-15,
# MEASURED NEGATIVE, code removed; do not retry). Tried: the SCHEDULER warp
# spins on the shard flag after decoding each CLC work tile (before its next
# query) INSTEAD of the AB-load warp gating before a tile's first TMA (the
# shipped design). It is structurally NOT a gate: under CLC persistence the
# multicast response lands in every CTA's smem directly from hardware (no
# STAS publish step by the sched warp), so there is no pre-commit window and
# the sched warp's spin cannot order any load warp's TMA behind flag
# arrival — the load warps free-run on already-multicast (and initial
# blockIdx-derived) tiles. Measured on B300 TP2 (tests) + TP4 ce_push
# (bench: 100 warmup + 500 timed, one event pair, cross-rank max, randn/8
# bf16, tile 128x256 cluster (2,1)):
#   - correctness: EVERY AG case fails (err ~5-12 vs tol ~0.016, delayed AND
#     non-delayed copies; gathered_err=0 — the transport delivered, the GEMM
#     consumed early).
#   - (8192,2048,8192) gating-bound corner: 258.6us vs 284.8/285.4us
#     (roof ~200) — the racy free-run recovers ~26us of the gate cost, an
#     UPPER BOUND for any scheme that moves gating off the load warp's
#     critical path (a correct one must add sync this version doesn't pay).
#   - (16384,4096,8192): 962.0 vs 961.1/964.7us; (32768,2048,8192): 993.7 vs
#     997.0/995.3us — parity within A/B/A drift, so the load-warp gate costs
#     ~nothing where NVLink delivery fits the GEMM window (matches the
#     overhead model in quack/distributed/all_gather_gemm.py's docstring).

# return PipelineStateWAdvance instead of PipelineState

"""
A high-performance persistent batched dense GEMM example for the NVIDIA Blackwell SM100 architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp: Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp:
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Type convert C matrix to output type.
    - Optionally store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations,
      or directly store C matrix from registers (RMEM) to global memory (GMEM) without TMA operations.

SM100 tcgen05.mma instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Constraints:
* Supported input data types: fp16, bf16, tf32, int8, uint8, fp8 (e4m3fn, e5m2),
  see detailed valid dtype combinations in below GemmSm100 class documentation
* A/B tensor must have the same data type
* Mma tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
* Mma tiler N must be 32-256, step 32
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 4, 8, and 16 for TFloat32,
  Float16/BFloat16, and Int8/Uint8/Float8, respectively.
"""


# canonical home moved to gemm_base (shared with the SM120 blockscaled path)
from torch._vendor.quack.gemm_base import reinterpret_packed_fp6 as _reinterpret_packed_fp6  # noqa: E402


class GemmSm100(GemmTmaBase):
    """This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    Warp roles and pipeline schedule (arrive counts validated by quack.pipeline_checks
    at construction; per-role acquire/work/commit timelines, one line per pipeline stage):

    Roles per CTA: epilogue warps (num_epi_warps, ids 0..n-1) | MMA warp | AB-load warp(s)
    (1, or 4 cp.async warps when gather_A) | C-load warp (when C) | scheduler warp |
    A-prefetch warp (gather_A only).

    Pipelines (producer role -> consumer role):
      ab       TmaUmma:  AB-load -> MMA.       full: TMA tx bytes (+cp.async lane arrives
               when gather_A); empty: 1 hw tcgen05.commit per A/B-mcast peer CTA.
      acc      UmmaAsync: MMA -> epilogue.     full: 1 hw commit; empty: 1 arrive per epi
               warp (elected lane), x2 CTAs routed to leader when 2-CTA MMA.
      sched    Async:    scheduler -> all warps. full: armed as tx barrier by the producer
               (CLC multicast try_cancel or STAS); empty: 1 arrive per consumer warp,
               every CTA routed to CTA 0.
      epi (C)  TmaAsync: C-load -> epilogue.   full: TMA tx; empty: 1 arrive per epi warp.
      a_pref   CpAsync:  A-prefetch -> AB-load (gather_A only). full: 32 lane arrives;
               empty: 1 arrive per AB-load warp.
      epi_store TmaStore: epilogue TMA-store completion pacing (bulk-group, no mbarrier).

    Per-role timeline (steady state, per scheduled tile / k-stage):
      scheduler: [per tile]  sched.acquire -> CLC try_cancel (arms tx) -> (hw completes)
      AB-load:   [per tile]  sched slot read -> sched.release
                 [per k]     ab.acquire (wait empty + arm tx) -> TMA A,B[,SFA,SFB]
      MMA:       [per tile]  acc.acquire (once, before k loop)
                 [per k]     ab full wait -> tcgen05.mma (k-blocks) -> commit -> ab.release
                 [tile end]  acc.commit (via tcgen05.commit)
      epilogue:  [per tile]  acc.wait -> per epi subtile: tmem load -> epi_store.acquire
                 -> smem -> TMA store -> acc.release (elected, after last read)
      C-load:    [per tile]  per epi subtile: epi.acquire (arm tx) -> TMA load C

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param mma_tiler_mnk: Shape of the MMA tile. Pass (M, N) to default K to
        4 MMA instructions, or (M, N, K) to set the K tile size explicitly.
    :type mma_tiler_mnk: Union[Tuple[int, int], Tuple[int, int, int]]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported A/B data types:
        - TFloat32
        - Float16/BFloat16
        - Int8/Uint8
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulator data types:
        - Float32 (for all floating point A/B data types)
        - Float16 (only for fp16 and fp8 A/B data types)
        - Int32 (only for uint8/int8 A/B data types)

    :note: Supported C data types:
        - Float32 (for float32 and int32 accumulator data types)
        - Int32 (for float32 and int32 accumulator data types)
        - Float16/BFloat16 (for fp16 and fp8 accumulator data types)
        - Int8/Uint8 (for uint8/int8 accumulator data types)
        - Float8E4M3FN/Float8E5M2 (for float32 accumulator data types)

    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16

    Example:
        >>> gemm = GemmSm100(
        ...     acc_dtype=Float32,
        ...     mma_tiler_mnk=(128, 128),
        ...     cluster_shape_mn=(2, 2)
        ... )
        >>> gemm(mA, mB, mD, max_active_clusters, stream)
    """

    arch = 100

    EpilogueArguments = GemmTmaBase.EpilogueArguments
    EpilogueParams = GemmTmaBase.EpilogueParams

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],  # ignored for now
        mma_tiler_mnk: Union[Tuple[int, int], Tuple[int, int, int]],
        cluster_shape_mnk: Tuple[int, int, int],
        sf_vec_size: Optional[int] = None,
        gather_A: bool = False,
        use_tma_gather: bool = False,
        use_clc_persistence: bool = True,
        concat_layout: tuple | None = None,
        use_pdl: bool = True,
        split_k: int = 1,
        split_k_mode: int = SplitKMode.SERIAL,
        # MMA element types when they differ from the tensor (storage/copy) dtypes:
        # packed fp6 crosses the FFI boundary as raw uint8 bytes (torch has no fp6
        # dtype) and is reinterpreted in-kernel to the fp6 MMA type. None: same as
        # the tensor dtype.
        a_mma_dtype: Optional[Type[cutlass.Numeric]] = None,
        b_mma_dtype: Optional[Type[cutlass.Numeric]] = None,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mnk: The (M, N) or (M, N, K) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mnk: The (ClusterM, ClusterN) shape of the CTA cluster.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param mma_tiler_mnk: (M, N) or (M, N, K) shape of the MMA tile.
            If only (M, N) is given, K defaults to 4 * instruction K.
        :type mma_tiler_mnk: Union[Tuple[int, int], Tuple[int, int, int]]
        :param cluster_shape_mnk: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mnk: Tuple[int, int]
        """

        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.a_mma_dtype = a_mma_dtype
        self.b_mma_dtype = b_mma_dtype
        self.sf_vec_size = sf_vec_size
        self.blockscaled = sf_vec_size is not None
        assert len(mma_tiler_mnk) in [2, 3], "MMA tiler must be (M, N) or (M, N, K)"
        valid_2cta_m = (128, 256) if not self.blockscaled else (256,)
        self.use_2cta_instrs = cluster_shape_mnk[0] % 2 == 0 and mma_tiler_mnk[0] in valid_2cta_m
        self.cluster_shape_mnk = cluster_shape_mnk
        assert cluster_shape_mnk[2] == 1, "Cluster shape K must be 1"
        # K dimension: if user provides 3 values, use their K; otherwise default in _setup_attributes
        if len(mma_tiler_mnk) == 3:
            self.mma_tiler = tuple(mma_tiler_mnk)
        else:
            self.mma_tiler = (*mma_tiler_mnk, 0)
        self.is_persistent = True
        self.use_clc_persistence = use_clc_persistence
        self.epi_m_major = True
        self.gather_A = gather_A
        self.concat_layout = concat_layout or ()
        self.use_tma_gather = use_tma_gather
        self.use_pdl = use_pdl
        if gather_A:
            assert cluster_shape_mnk[1] == 1, "Cluster shape N must be 1 for gather A "
        if use_tma_gather:
            assert gather_A, "TMA gather requires gather_A=True"
        self._init_split_k(split_k, split_k_mode)
        if split_k > 1 and self.blockscaled:
            # Block-scaled composes with the finalizer-only split-K device path as-is:
            # the SF (scale-factor) TMA loads ride the same k_tile_start-offset copy_fn
            # list as A/B, and the accumulator is already descaled f32 before the
            # epilogue, so summing raw f32 partials over disjoint K-ranges is exact.
            # SEPARATE additionally needs a block-scaled-reachable reduction kernel, which
            # the block-scaled host does not yet wire.
            assert self.split_k_mode != SplitKMode.SEPARATE, (
                "block-scaled split_k does not support SEPARATE yet; use SERIAL or PARALLEL"
            )

        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.num_ab_load_warps = 1 if not self.gather_A else 4
        self.occupancy = 1
        # Set specialized warp ids
        self.epi_warps_per_accumulator = 4
        num_epi_warps = self.epi_warps_per_accumulator
        self.epilog_warp_id = tuple(range(num_epi_warps))
        self.mma_warp_id = len(self.epilog_warp_id)
        self.ab_load_warp_id = self.mma_warp_id + 1
        self.epi_load_warp_id = self.ab_load_warp_id + self.num_ab_load_warps
        self.scheduler_warp_id = self.epi_load_warp_id + 1
        # For gather_A: separate A-index prefetch warp (was the empty warp)
        self.a_prefetch_warp_id = self.scheduler_warp_id + 1 if self.gather_A else None
        self.num_epi_warps = len(self.epilog_warp_id)
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.Epilogue),
            num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
        )
        # CLC throttle: paces query issue to tile consumption so the multi-stage
        # lookahead can't over-cancel the pending pool. Producer = CTA0 load warp
        # (arrive per tile started), consumer = CTA0 scheduler warp (sync per
        # query); 2 warps => 64 threads. Lives here (not the scheduler) because a
        # NamedBarrier id is a whole-CTA resource coordinated by NamedBarrierGemm,
        # and the participating-thread count is arch-specific (warp layout). A
        # single barrier suffices: the dependency chain commit(k+1) <- fetch(k+1)
        # <- query(k+1) <- sync(k) forces strict producer/consumer alternation, so
        # <= 1 credit is ever outstanding; bar.sync also gives a hardware wakeup vs
        # an mbarrier pipeline's PHASECHK + NANOSLEEP polling.
        self.clc_throttle_barrier = (
            pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierGemm.ClcThrottle), num_threads=2 * cute.arch.WARP_SIZE
            )
            if self.use_clc_persistence
            else None
        )
        # Register reallocation for gather_A (3 warp groups, 504 regs total, 168 per WG default).
        # Heavy epilogues (e.g. colvec_reduce in DGated) override these to avoid register spilling.
        # Without gather_A there are only 2 WGs (512 total, 256 per WG = max), no reallocation needed.
        self.num_regs_other = 120
        self.num_regs_epi = 256
        extra_warp_ids = (self.a_prefetch_warp_id,) if self.gather_A else ()
        self.threads_per_cta = cute.arch.WARP_SIZE * (
            self.num_ab_load_warps
            + len(
                (
                    self.mma_warp_id,
                    self.epi_load_warp_id,
                    self.scheduler_warp_id,
                    *self.epilog_warp_id,
                    *extra_warp_ids,
                )
            )
        )
        # Multiple of 4 warps to increase/decrease number of registers
        assert self.threads_per_cta % 128 == 0

    def epi_smem_warp_shape_mnk(self):
        # Mirrors cutlass.utils.blackwell_helpers.compute_epilogue_tile_shape:
        # the epilogue tmem layout uses two M warps and two N warps when the
        # per-CTA M tile is 64 and the kernel uses 2-CTA instructions.
        warp_m, warp_n = (
            (2, 2) if self.cta_tile_shape_mnk[0] == 64 and self.use_2cta_instrs else (4, 1)
        )
        return (warp_m, warp_n, 1)

    def _setup_attributes(self, epilogue_args: EpilogueArguments, varlen_args: VarlenArguments):
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
        - Computing tensor memory allocation columns
        """
        self.epi_m_major = self.resolve_epi_m_major(epilogue_args)

        # Compute mma instruction shapes
        mma_inst_bits_k = 256
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        mma_inst_shape_n = self.mma_tiler[1] if self.mma_tiler[1] <= 256 else self.mma_tiler[1] // 2
        if const_expr(self.blockscaled):
            mma_inst_k = GemmSm100._blockscaled_mma_inst_k(self.a_dtype, self.b_dtype)
        else:
            mma_inst_k = mma_inst_bits_k // self.a_dtype.width
        self.mma_inst_shape_mnk = (
            self.mma_tiler[0],
            mma_inst_shape_n,
            mma_inst_k,
        )
        # Configure tiled mma
        if const_expr(not self.blockscaled):
            self.tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.acc_dtype,
                self.cta_group,
                self.mma_inst_shape_mnk[:2],
            )
        else:
            # Per-architecture support gate: each GemmSmXXX owns the asserts for
            # the (A, B, SF, D) dtype combinations it implements.
            assert GemmSm100.is_valid_dtypes_and_scale_factor_vec_size(
                self.a_mma_dtype,
                self.b_mma_dtype,
                self.sf_dtype,
                self.sf_vec_size,
                self.d_dtype,
                a_copy_dtype=self.a_dtype,
                b_copy_dtype=self.b_dtype,
            ), (
                f"GemmSm100 blockscaled does not support A={self.a_mma_dtype} "
                f"(storage {self.a_dtype}), B={self.b_mma_dtype} (storage {self.b_dtype}), "
                f"SF={self.sf_dtype}, sf_vec_size={self.sf_vec_size}, D={self.d_dtype}"
            )
            # quack's wrapper, not the DSL helper: both-fp4 pairs always run
            # the single kind::mxf4nvf4 atom with the format's scale config.
            self.tiled_mma = quack_sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_mma_dtype,
                self.b_mma_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape_mnk[:2],
            )

        # Compute mma/cluster/tile shapes
        if self.mma_tiler[2] > 0:
            if const_expr(self.blockscaled):
                sf_chunk_k = self.sf_vec_size * 4
                assert self.mma_tiler[2] % sf_chunk_k == 0, (
                    f"Blockscaled MMA tiler K ({self.mma_tiler[2]}) must be divisible by "
                    f"the scale-factor chunk K ({sf_chunk_k})"
                )
            assert self.mma_tiler[2] % self.mma_inst_shape_mnk[2] == 0, (
                f"MMA tiler K ({self.mma_tiler[2]}) must be divisible by "
                f"MMA instruction K ({self.mma_inst_shape_mnk[2]})"
            )
            mma_inst_tile_k = self.mma_tiler[2] // self.mma_inst_shape_mnk[2]
        else:
            mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(self.tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        if const_expr(self.blockscaled):
            # The SF atom fixed by the tcgen05 MMA (BlockScaledBasicChunk) is 128
            # wide in N and one atom is 512 contiguous gmem bytes (128 N x 4
            # sf-k blocks), but cta_tile_n need not be a multiple of 128. SFB is
            # loaded per atom with free-form atom coordinates (see the SFB TMA
            # setup in __call__); tile_n=64/192 tiles that start 64 into an atom
            # are handled by a per-tile SFB tmem column offset in the mma warp
            # (sfb_n_atom_misaligned). tile_n must be a multiple of 64: the SF
            # tmem datapath (both the tcgen05.cp write and the MMA read) is
            # 2-column / 64-N granular, so odd multiples of 32 (96, 160, 224)
            # are hardware-unreachable without rotating the SF content by 32 N.
            assert (
                self.cta_tile_shape_mnk[1] % 64 == 0 and 64 <= self.cta_tile_shape_mnk[1] <= 256
            ), (
                f"blockscaled tile_n must be a multiple of 64 in [64, 256], "
                f"got {self.cta_tile_shape_mnk[1]}"
            )
            self.sfb_n_atom_misaligned = (self.cta_tile_shape_mnk[1] // 64) % 2 == 1
            # SFB smem/tmem window per tile, in 128-wide SF atoms.
            self.sfb_window_atoms = cute.round_up(self.cta_tile_shape_mnk[1], 128) // 128
            # 512-byte SF chunks (one atom-N x 4 sf-k blocks) per k-tile.
            self.sfb_chunks_per_ktile = self.mma_tiler[2] // (self.sf_vec_size * 4)

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        if self.gather_A:
            assert self.num_mcast_ctas_a == 1
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        tile_load_layout = None
        tile_load_dtype = None
        # If TileLoad exists without C, use the first non-None tile-load tensor as
        # the C-like input for SM100's epilogue tile shape. Multiple TileLoads
        # share the same epi_tile shape.
        for op in getattr(self, "_epi_ops", ()):
            if op.is_tile_load():
                tile_load_tensor = getattr(epilogue_args, op.name, None)
                if tile_load_tensor is not None:
                    tile_load_layout = LayoutEnum.from_tensor(tile_load_tensor)
                    tile_load_dtype = tile_load_tensor.element_type
                    break
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.d_layout if self.d_layout is not None else LayoutEnum.ROW_MAJOR,
            self.d_dtype if self.d_dtype is not None else cutlass.BFloat16,
            layout_c=self.c_layout if self.c_layout is not None else tile_load_layout,
            elem_ty_c=self.c_dtype if self.c_dtype is not None else tile_load_dtype,
        )
        # TMA store tile starts must stay aligned when advancing across CTA-N tiles.
        # There's a bug w compute_epilogue_tile_shape (as of cutlass-dsl 4.4.2) where if
        # tile_n = 224 and there's C, it will set epi_tile to (128, 64).
        if const_expr(self.cta_tile_shape_mnk[1] % cute.size(self.epi_tile[1]) != 0):
            warp_n = 2 if (self.cta_tile_shape_mnk[0] == 64 and self.use_2cta_instrs) else 1
            epi_tile_n = math.gcd(self.cta_tile_shape_mnk[1], cute.size(self.epi_tile[1]))
            epi_tile_n_layout = cute.make_layout(
                (epi_tile_n // warp_n, warp_n), stride=(1, self.cta_tile_shape_mnk[1] // warp_n)
            )
            self.epi_tile = (self.epi_tile[0], cute.coalesce(epi_tile_n_layout))
        # Quantized-output SF vectors and contracted narrow stores may require
        # a wider N run than the default epi subtile. The stage computation
        # below re-budgets smem after any widening.
        _, sfd_min_n = active_row_sfd_reqs(getattr(self, "_epi_ops", ()), epilogue_args)
        store_min_n = 0
        for op in getattr(self, "_epi_ops", ()):
            if not op.is_tile_store():
                continue
            required_n = op.min_epi_tile_n(getattr(epilogue_args, op.name, None))
            if required_n is not None:
                store_min_n = max(store_min_n, required_n)
        min_epi_n = max(sfd_min_n or 0, store_min_n)
        epi_tile_n = self.epi_tile[1]
        if cute.size(epi_tile_n) < min_epi_n:
            # Only widen a contiguous N tile (an int or a trivial n:1 layout);
            # the strided non-pow2 fixup layout can't be widened.
            contiguous = not isinstance(epi_tile_n, cute.Layout) or (
                cute.coalesce(epi_tile_n).stride == 1
            )
            if contiguous and self.cta_tile_shape_mnk[1] % min_epi_n == 0:
                self.epi_tile = (self.epi_tile[0], cute.make_layout(min_epi_n))
            elif cute.size(epi_tile_n) < store_min_n:
                if not contiguous or self.cta_tile_shape_mnk[1] % store_min_n:
                    raise ValueError(f"epilogue N tile cannot satisfy required width {store_min_n}")
                self.epi_tile = (self.epi_tile[0], cute.make_layout(store_min_n))

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        prefetch_A_idx = (
            None
            if not self.gather_A
            else ("varlen_m" if varlen_args.mCuSeqlensM is not None else "varlen_k")
        )
        (
            self.num_acc_stage,
            self.ab_stage,
            self.epi_stage,
            self.epi_c_stage,
        ) = self._compute_stages(
            self.tiled_mma,
            self.mma_tiler,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_smem_dtype,
            self.b_smem_dtype,
            self.sf_dtype,
            self.sf_vec_size,
            self.d_dtype,
            self.c_dtype,
            self.d_layout,
            self.c_layout,
            epilogue_args,
            prefetch_A_idx,
            cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}"),  # smem_capacity
            self.occupancy,
            self.epi_smem_warp_shape_mnk(),
        )
        # With CLC the try_cancel response lands directly in the consumer slot, so
        # the next query can only be issued once all consumers (cluster-wide)
        # release that slot. >=2 stages keep a query in flight while the previous
        # tile's info is still being read (cutlass's SchedulerPipelineStageCount
        # >= 2); the 3rd stage buys response slack for epilogue-bound tiles (e.g.
        # symmetric's double store, ~3% at M=8192 K=512) and costs only 12 smem
        # ints + one mbarrier pair.
        self.sched_stage = 3 if self.use_clc_persistence else 1
        self.a_prefetch_stage = (
            0
            if not self.gather_A
            else (2 if varlen_args.mCuSeqlensM is not None else self.ab_stage)
        )

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, self.a_smem_dtype, self.ab_stage
        )
        self.a_smem_load_layout_staged = self.a_smem_layout_staged
        if const_expr(self.gather_A):
            if const_expr(self.use_tma_gather):
                self.a_smem_load_layout_staged = quack_sm100_utils.make_smem_layout_tma_gather_a(
                    self.tiled_mma, self.mma_tiler, self.a_dtype, self.ab_stage
                )
            else:
                self.a_smem_load_layout_staged = quack_sm100_utils.make_smem_layout_cpasync_a(
                    self.tiled_mma, self.mma_tiler, self.a_dtype, self.ab_stage
                )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, self.b_smem_dtype, self.ab_stage
        )
        self.epi_smem_layout_staged = None
        if const_expr(self.d_dtype is not None):
            self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.d_dtype, self.d_layout, self.epi_tile, self.epi_stage
            )
        self.epi_c_smem_layout_staged = None
        if const_expr(self.c_dtype is not None):
            self.epi_c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.epi_c_stage
            )
        if const_expr(self.blockscaled):
            self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
                self.tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                self.ab_stage,
            )
            self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                self.ab_stage,
            )
        else:
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged = None, None

        # Compute the number of tensor memory allocation columns
        if const_expr(not self.blockscaled):
            self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
                self.tiled_mma, self.mma_tiler, self.num_acc_stage
            )
        else:
            self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        # Overlapping accumulator and scaling factor in tmem, targetting the case tile_n == 256
        # For iter 0, 2, ..., accum is in col 0...255 and SF are in col 256...256+SF_size.
        # For iter 1, 3, ..., accum is in col 256...511 and SF are in col 0...0+SF_size.
        # During the epilogue, we release acc_pipeline after being done with @SF_size columns.
        # In the cute-dsl example,
        # https://github.com/NVIDIA/cutlass/blob/08185b9c3e90510ee2b656662ed0d53b06d28157/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py#L369
        # instead the 2 stages of accum are in col 0...255 and 256-SF_size...512-SF_size, and
        # the SF are in 512-SF_size...511. The 2 accum stages overlap, so in the epilogue,
        # they alternate the direction of epi tiles (from right to left, then from left to right)
        # to release acc_pipeline early.
        # The two approaches perform about the same.
        self.overlap_accum_sf = self.blockscaled and self.num_acc_stage == 1
        if const_expr(self.overlap_accum_sf):
            num_sf_tmem_cols = (
                (
                    cute.ceil_div(self.cta_tile_shape_mnk[0], 128)
                    + cute.ceil_div(self.cta_tile_shape_mnk[1], 128)
                )
                * 4  # 4 cols per stage
                * (self.mma_inst_shape_mnk[2] // self.sf_vec_size)
            )
            self.iter_acc_early_release = num_sf_tmem_cols // cute.size(self.epi_tile[1])
        else:
            self.iter_acc_early_release = -1

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
        mSFA: Optional[cute.Tensor] = None,
        mSFB: Optional[cute.Tensor] = None,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param mA: Input tensor A
        :type mA: cute.Tensor
        :param mB: Input tensor B
        :type mB: cute.Tensor
        :param mD: Output tensor D
        :type mD: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        :raises AssertionError: If OOB (Out-Of-Bounds) tiles are present when TMA store is disabled.
        """
        if const_expr(self.blockscaled):
            assert mSFA is not None and mSFB is not None
            # Dense unbatched (rank-5) SFs: prepend the trivial batch mode so the
            # rest of the kernel sees the usual (l, rm/rn, rk, 32, 4, 4) shape.
            if const_expr(cute.rank(mSFA) == 5):
                mSFA = layout_utils.expand(mSFA, 0, 1)
            if const_expr(cute.rank(mSFB) == 5):
                mSFB = layout_utils.expand(mSFB, 0, 1)
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
        # Packed 6-bit operands cross the FFI boundary as raw bytes (torch has
        # no fp6 dtype): reinterpret (mn, 3k/4[, l]) Uint8 as (mn, k[, l]) fp6.
        # The fp6-typed gmem tensor is what selects the U6_UNPACK_U8 unpack
        # tensormap in the TMA atom builder.
        if const_expr(self.a_mma_dtype is not None and self.a_mma_dtype.width == 6):
            mA = _reinterpret_packed_fp6(mA, self.a_mma_dtype)
        if const_expr(self.b_mma_dtype is not None and self.b_mma_dtype.width == 6):
            mB = _reinterpret_packed_fp6(mB, self.b_mma_dtype)
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype = mA.element_type  # storage/copy dtype (smem layouts, TMA, sizes)
        self.b_dtype = mB.element_type
        # MMA element types default to the storage dtypes (identical except for
        # packed fp6, whose FFI-boundary dtype was Uint8 before the reinterpret
        # above - post-reinterpret they coincide too).
        self.a_mma_dtype = self.a_mma_dtype if self.a_mma_dtype is not None else self.a_dtype
        self.b_mma_dtype = self.b_mma_dtype if self.b_mma_dtype is not None else self.b_dtype
        # TMA-unpack operands: a packed sub-byte gmem operand under tcgen05
        # kind::mxf8f6f4 (any blockscaled pair other than both-packed-fp4,
        # which runs the denser kind::mxf4nvf4). TMA expands the packed
        # stream to one byte of smem footprint per element (TmaDataFormat
        # U4_UNPACK_U8/U6_UNPACK_U8 via internal_type=Uint8), so ALL smem-side
        # accounting - layout atom selection, storage, capacity - is byte-domain
        # (Uint8), mirroring CUTLASS C++ SmemAllocType. The sub-byte identity
        # survives only in (1) the gmem tensor's element type (which selects the
        # tensormap format) and (2) the MMA instruction descriptor via
        # a/b_mma_dtype. The Uint8-typed smem tensor feeds make_fragment_A/B
        # directly: the smem descriptor is built from width x stride products,
        # and the byte layout is the convention kind::mxf8f6f4 expects.
        if const_expr(self.blockscaled):
            both_packed_fp4 = (
                self.a_dtype is cutlass.Float4E2M1FN and self.b_dtype is cutlass.Float4E2M1FN
            )
            self.a_unpack = self.a_dtype.width < 8 and not both_packed_fp4
            self.b_unpack = self.b_dtype.width < 8 and not both_packed_fp4
        else:
            self.a_unpack, self.b_unpack = False, False
        self.a_smem_dtype = cutlass.Uint8 if self.a_unpack else self.a_dtype
        self.b_smem_dtype = cutlass.Uint8 if self.b_unpack else self.b_dtype
        self.d_dtype = mD.element_type if mD is not None else None
        self.c_dtype = mC.element_type if mC is not None else None
        self.sf_dtype: Optional[Type[cutlass.Numeric]] = (
            mSFA.element_type if mSFA is not None else None
        )
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD) if mD is not None else None
        self.c_layout = LayoutEnum.from_tensor(mC) if mC is not None else None
        self.a_major_mode = LayoutEnum.from_tensor(mA).mma_major_mode()
        self.b_major_mode = LayoutEnum.from_tensor(mB).mma_major_mode()

        # Check if input data types are compatible with MMA instruction.
        # Blockscaled tcgen05 kind::mxf8f6f4 supports mixed A/B element types;
        # the supported combinations are asserted in _setup_attributes.
        if const_expr(not self.blockscaled and self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        if const_expr(varlen_args is None):
            varlen_args = VarlenArguments()
        assert (varlen_args.mAIdx is not None) == self.gather_A
        varlen_m = varlen_args.mCuSeqlensM is not None
        varlen_k = varlen_args.mCuSeqlensK is not None
        # Stash for epilogue ops (epi_to_underlying_arguments runs later and
        # SFD needs the varlen mode to shape its logical scale layout).
        self.varlen_m = varlen_m
        self.varlen_k = varlen_k

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes(epilogue_args, varlen_args)

        if const_expr(self.blockscaled):
            # Rebuild the SFA layout from mSFA's actual strides so non-packed
            # buffers work (e.g. a slice of a larger scale tensor).
            # Only the innermost 512-B tile must be contiguous.
            # For varlen_m, mSFA is sized for per-expert 128-row-padded storage
            # (tile-aligned per-batch padding), so use its own M dim (= total_padded_rm * 128)
            # instead of mA.shape[0] (= total_m, unpadded).
            if const_expr(cute.rank(mA) == 3):
                sfa_shape = mA.shape
            elif const_expr(varlen_m):
                sfa_shape = (mSFA.shape[1] * 128, mA.shape[1])
            else:  # varlen_k
                sfa_shape = (mA.shape[0], mSFA.shape[2] * 128)
            sfa_layout = tile_atom_to_shape_SF_strided(sfa_shape, self.sf_vec_size, mSFA.stride)
            mSFA = cute.make_tensor(mSFA.iterator, sfa_layout)
            # SFB needs no (N, K, L) logical layout: it is loaded per 512-B SF
            # atom (chunk) with free-form atom coordinates. View the raw
            # (L, RN, RK, 32, 4, 4) scale tensor as (chunk, RK, RN, L) in Int16
            # (TMA box inner dim is capped at 256 elements, so a 512-B chunk is
            # 256 x Int16). The (256):(1) chunk mode is the blocked SF format's
            # contract (hardware-fixed packed atom) and must be imposed
            # statically — dlpack strides are dynamic and TMA needs a static
            # V-map — so only the outer modes come from the tensor.
            mSFB_i16 = layout_utils.select(cute.recast_tensor(mSFB, cutlass.Int16), [2, 1, 0])
            mSFB = cute.make_tensor(
                mSFB_i16.iterator, cute.prepend(mSFB_i16.layout, cute.make_layout(256))
            )

        atom_thr_size = cute.size(self.tiled_mma.thr_id.shape)

        # Setup TMA load for A & B
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = None, None
        a_op = (
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
            if const_expr(not self.gather_A)
            else sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mnk, self.tiled_mma.thr_id
            )
        )
        if const_expr(not self.gather_A):
            # varlen_m + an active M-fold reduce sink: rag A (2-extra-dim
            # wraparound; loads can't ptr_shift) so rows past the sequence end
            # zero-fill instead of reading the next sequence — restores the
            # "OOB accumulator lanes are zero" invariant that the unpredicated
            # M-fold relies on. Without such a sink, garbage rows are masked at
            # every store, so plain domain_offset keeps the descriptor 2-D.
            if const_expr(varlen_m and self.epilogue_zero_fill_varlen_m(epilogue_args)):
                mA_tma = copy_utils.create_ragged_tensor_for_tma(mA, ragged_dim=0, ptr_shift=False)
            elif const_expr(varlen_k):
                mA_tma = copy_utils.create_ragged_tensor_for_tma(mA, ragged_dim=1, ptr_shift=False)
            else:
                mA_tma = mA
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                mA_tma,
                a_smem_layout,
                self.mma_tiler,
                self.tiled_mma,
                self.cluster_layout_vmnk.shape,
                # Uint8 internal type + sub-byte gmem element selects the
                # U4/U6_UNPACK_U8 tensormap (packed gmem -> byte-container smem).
                internal_type=(
                    cutlass.Uint8
                    if const_expr(self.a_unpack)
                    else (cutlass.TFloat32 if mA.element_type is Float32 else None)
                ),
            )
        elif const_expr(self.use_tma_gather):
            # gather4 descriptor: box has 1 in the gathered dim, tile size in the contiguous dim.
            # varlen_m (K-major): box (1, tile_K), gather M rows at K offset
            # varlen_k (M-major): box (64, 1), gather K cols at M offset
            tma_smem_layout = quack_sm100_utils.make_smem_layout_atom_tma_gather_a(
                self.tiled_mma, self.mma_tiler, self.a_dtype, gather_size=1
            )
            tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
                a_op,
                mA,
                tma_smem_layout,
                tma_smem_layout.shape,
                internal_type=(cutlass.TFloat32 if mA.element_type is Float32 else None),
            )
        # block_copy takes compiler-driven multicast metadata at the copy site,
        # so the TMA atom itself must stay the non-multicast variant here.
        b_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            copy_utils.create_ragged_tensor_for_tma(mB, ragged_dim=1) if varlen_k else mB,
            b_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.Uint8
                if const_expr(self.b_unpack)
                else (cutlass.TFloat32 if mB.element_type is Float32 else None)
            ),
        )

        tma_atom_sfa, tma_tensor_sfa = None, None
        tma_atom_sfb, tma_tensor_sfb = None, None
        if const_expr(self.blockscaled):
            # Setup TMA load for SFA
            sfa_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
            sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
            tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op,
                mSFA,
                sfa_smem_layout,
                self.mma_tiler,
                self.tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            # Setup TMA load for SFB.
            # One box per stage covering the tile's SF window at chunk
            # granularity: (256 Int16 = one 512-B atom-chunk, chunks-per-k-tile,
            # window atoms). Because the box coordinates are free-form (not
            # tiler multiples), a tile whose window starts mid-way into the
            # atom sequence (tile_n=192 advances 1.5 atoms per tile) needs no
            # layout tricks: the kernel computes first_atom = j*tile_n//128 and
            # slices there. Tiles start (j*tile_n) % 128 into their first atom;
            # the mma warp corrects for that via an SFB tmem column offset.
            # The atom-n dim's extent is the allocated atom count, so a last
            # tile whose window straddles past it (tile_n=192 with e.g. N=448:
            # window atoms {3,4}, only 4 allocated) gets the out-of-range atom
            # hardware-zero-filled — its columns are beyond N, where B is
            # zero-filled too. This invariant is load-bearing: the previous
            # overlapped-window remap presented atoms in groups of 4 and
            # zero-filled past the *presented* extent instead, silently zeroing
            # the last valid columns (see the N=448 regression test).
            # SFB is multicast across all cluster-M CTAs (compiler-driven at the
            # copy site, like B); the op carries cta_group so 2-CTA kernels use
            # cta_group::2 multicast, whose transaction bytes aggregate at the
            # pair leader's barrier as num_tma_load_bytes expects. The atom
            # stays the non-multicast variant with num_multicast=1 (same as
            # A/B): block_copy's lowering derives the mask and slicing from the
            # tma_multicast dict.
            sfb_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
            sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
            # Compact column-major: (chunk, k-chunks, window atoms).
            sfb_window_layout = cute.make_layout(
                (256, self.sfb_chunks_per_ktile, self.sfb_window_atoms)
            )
            # The chunk view must tile the actual SFB smem stage bytes exactly
            # (atom-major, k-chunks inner — the order make_smem_layout_sfb
            # produces).
            assert cute.cosize(sfb_smem_layout) == 2 * cute.cosize(sfb_window_layout)
            assert cute.cosize(self.sfb_smem_layout_staged) == (
                2 * cute.cosize(sfb_window_layout) * self.ab_stage
            )
            tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(
                sfb_op,
                mSFB,
                sfb_window_layout,
                sfb_window_layout.shape,
            )

        # Transaction bytes are counted with the GMEM dtype: for unpack operands
        # the layout is byte-domain (cosize == smem footprint bytes) but TMA
        # reports only the packed data bytes it copies (elements x 4 or 6 bits;
        # the intra-16B gaps are not written), matching CUTLASS C++
        # (sizeof_bits<ElementA> x cosize of the uint8-alloc smem layout).
        self.num_tma_load_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        if const_expr(not self.gather_A or self.use_tma_gather):
            self.num_tma_load_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)
        if const_expr(self.blockscaled):
            sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            self.num_tma_load_bytes += sfa_copy_size + sfb_copy_size
        self.num_tma_load_bytes *= atom_thr_size

        # Setup TMA store for D and TMA load for C.
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

        self.buffer_align_bytes = 1024

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged) if mD is not None else 0
        epi_c_smem_size = cute.cosize(self.epi_c_smem_layout_staged) if mC is not None else 0
        sf_dtype = self.sf_dtype if const_expr(self.blockscaled) else cutlass.Float8E8M0FNU
        sfa_smem_size = (
            cute.cosize(self.sfa_smem_layout_staged) if const_expr(self.blockscaled) else 0
        )
        sfb_smem_size = (
            cute.cosize(self.sfb_smem_layout_staged) if const_expr(self.blockscaled) else 0
        )
        a_idx_smem_size = 0
        if const_expr(self.gather_A):
            a_idx_smem_size = self.a_prefetch_stage * (
                self.cta_tile_shape_mnk[0] if varlen_m else self.cta_tile_shape_mnk[2]
            )

        # Define shared storage for kernel. sched_data lives in the RESERVED
        # smem partition (with the pipeline mbarriers / TMEM holding buf): a
        # small buffer before the 1024-byte aligned epilogue tensors would add
        # a 1 KiB pad; CLC responses use i128 copies, so it stays 16-byte
        # aligned.
        # 4 Int32 per stage, shared by the two (mode-exclusive) users: STATIC/DYNAMIC
        # store the STAS-broadcast (pid_m, pid_n, batch_idx, is_valid); CLC stores the
        # 16-byte try_cancel response (16B-aligned since each stage slot is 16 bytes).
        # +6 Int32 after the ring: the retirement drain's private response
        # slot (16 B) + mbarrier (8 B); see TileScheduler.cancel_pending_tail.
        sched_smem_size = 4 * self.sched_stage + 6 if self.is_persistent else 0

        @partitioned_struct
        class SharedStorage:
            sched_data: Reserved[
                cute.struct.Align[cute.struct.MemRange[Int32, sched_smem_size], 16]
            ]
            sAIdx: cute.struct.Align[cute.struct.MemRange[Int32, a_idx_smem_size], 16]
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
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
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_smem_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_smem_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[sf_dtype, sfa_smem_size],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[sf_dtype, sfb_smem_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a if const_expr(not self.gather_A or self.use_tma_gather) else mA,
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
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.a_smem_load_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
            TileSchedulerCls,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: Optional[cute.CopyAtom],
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: Optional[cute.CopyAtom],
        mSFA_mkl: Optional[cute.Tensor],
        tma_atom_sfb: Optional[cute.CopyAtom],
        mSFB_chunks: Optional[cute.Tensor],
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl: Optional[cute.Tensor],
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        epilogue_params,
        varlen_params: VarlenManager.Params,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        a_smem_load_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        sfa_smem_layout: Optional[cute.Layout],
        sfb_smem_layout: Optional[cute.Layout],
        epi_smem_layout: Union[cute.Layout, cute.ComposedLayout, None],
        epi_c_smem_layout: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """

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
            for tma_atom in (
                tma_atom_a,
                tma_atom_b,
                tma_atom_sfa,
                tma_atom_sfb,
                tma_atom_d,
                tma_atom_c,
            ):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # Setup cta/thread coordinates
        # Coords inside cluster
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        smem = cutlass.utils.SmemAllocator()
        storage = self.shared_storage.allocate(smem)

        # Initialize pipelines and states
        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cluster_layout_vmnk,
            is_leader_cta=is_leader_cta,
        )
        epi_pipeline = None
        if const_expr(has_epi_load):
            epi_pipeline = self.make_epi_pipeline(tx_count=self.epi_load_bytes_per_stage)
        acc_pipeline = self.make_acc_pipeline(cluster_layout_vmnk=cluster_layout_vmnk)
        sched_pipeline = None
        sched_data = None
        if const_expr(self.is_persistent):
            sched_pipeline = self.make_sched_pipeline(self.cluster_shape_mnk, has_C=has_epi_load)
            sched_data = storage.sched_data.get_tensor(cute.make_layout((4, self.sched_stage)))
        a_prefetch_pipeline = None
        if const_expr(self.gather_A):
            a_prefetch_pipeline = self.make_a_prefetch_pipeline()

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        # Tensor memory dealloc barrier init
        tmem = cutlass.utils.TmemAllocator(
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # Setup smem tensor A/B/D
        # (MMA, MMA_M, MMA_K, STAGE)
        sA_mma = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sA = storage.sA.get_tensor(a_smem_load_layout.outer, swizzle=a_smem_load_layout.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sAIdx = None
        if const_expr(self.gather_A):
            a_idx_smem_dim = self.cta_tile_shape_mnk[0] if varlen_m else self.cta_tile_shape_mnk[2]
            a_idx_smem_layout = cute.make_layout((a_idx_smem_dim, self.a_prefetch_stage))
            sAIdx = storage.sAIdx.get_tensor(a_idx_smem_layout)
        sSFA, sSFB, sSFB_chunks = None, None, None
        if const_expr(self.blockscaled):
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout)
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB = storage.sSFB.get_tensor(sfb_smem_layout)
            # Chunk view of the same bytes for the SFB TMA load: one 512-B SF
            # atom-chunk = 256 x Int16, k-chunks inner, atoms outer (the order
            # make_smem_layout_sfb produces; asserted against sfb_smem_layout
            # at TMA setup). Compact column-major.
            sSFB_chunks = cute.make_tensor(
                cute.recast_ptr(sSFB.iterator, dtype=cutlass.Int16),
                cute.make_layout(
                    (256, self.sfb_chunks_per_ktile, self.sfb_window_atoms, self.ab_stage)
                ),
            )
        sD = None
        if const_expr(has_D):
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        sC = None
        if const_expr(has_C):
            sC = storage.sC.get_tensor(epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner)
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage if not self.overlap_accum_sf else 2)
        )

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
            TileSchedulerCls.create,
            tile_sched_params,
            sched_data,
            sched_pipeline,
            throttle_barrier=self.clc_throttle_barrier,
        )

        epi_load_barrier = None
        if const_expr(has_epi_load):
            epi_load_barrier = pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierGemm.EpilogueLoad),
                num_threads=(self.num_ab_load_warps + 1) * cute.arch.WARP_SIZE,
            )

        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # Specialized AB load warps
        if (
            warp_idx >= self.ab_load_warp_id
            and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
        ):
            # PDL: wait for prior kernel before any TMA loads (matches cutlass C++ main_load)
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            if const_expr(self.gather_A):
                cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Persistent tile scheduling loop
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            ag_last_gate = Int32(-1)  # 1-entry satisfied-gate cache (see ag_wait_m_tile)
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            a_prefetch_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.a_prefetch_stage
            )
            do_epi_load_barrier_arrive = Boolean(True)
            # CLC throttle producer: only the first load warp of CTA 0 in the
            # cluster signals; commit once per work tile started, or the scheduler
            # warp starves of credits.
            is_throttle_producer = Boolean(warp_idx == self.ab_load_warp_id)
            if const_expr(cute.size(cluster_layout_vmnk) > 1):
                is_throttle_producer = is_throttle_producer & Boolean(
                    cute.arch.block_idx_in_cluster() == 0
                )
            while work_tile.is_valid_tile:
                tile_scheduler.throttle_producer_commit(is_throttle_producer)
                # (pid_m, pid_n, split_idx | None, batch_idx), decoded by the scheduler
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
                # AllGather+GEMM: block until this tile's M-shard of A has been
                # pulled into local HBM by the copy stream. Only the load warp
                # gates — the MMA/epilogue warps are downstream of the AB
                # pipeline. With the ring-rotated shard-major schedule the flag
                # is normally already set and this is a single L2 load.
                # getattr: the varlen/triangular scheduler Params classes have
                # no ag_* fields at all (pre-existing trace error since the AG
                # gate landed; surfaced on any cold-cache varlen compile).
                if const_expr(getattr(tile_sched_params, "ag", None) is not None):
                    iket.range_push("ag_wait")
                    ag_last_gate = ag_wait_m_tile(
                        tile_sched_params,
                        tile_coord_mnkl[0],
                        self.cluster_shape_mnk[0],
                        ag_last_gate,
                    )
                    iket.range_pop()
                # Local_tile partition global tensors
                mma_tile_coord_mnl = (
                    tile_coord_mnkl[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_coord_mnkl[1],
                    tile_coord_mnkl[3],
                )
                gA_mk = None
                if const_expr(not self.gather_A):
                    mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                    # (bM, bK, RestK)
                    gA_mk = cute.local_tile(
                        mA_mk,
                        cute.select(self.mma_tiler, [0, 2]),
                        (mma_tile_coord_mnl[0], None),
                    )
                # (bN, bK, RestK)
                gB_nk = cute.local_tile(
                    varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                    cute.select(self.mma_tiler, [1, 2]),
                    (mma_tile_coord_mnl[1], None),
                )
                if const_expr(self.blockscaled):
                    # (bM, bK)
                    # SFA uses the tile-aligned per-batch offset (padded SF layout), not
                    # the A-data offset — allows varlen_m seqlens that aren't
                    # multiples of 128.
                    gSFA_mkl = cute.local_tile(
                        varlen_manager.offset_batch_SFA(mSFA_mkl, batch_idx),
                        cute.select(self.mma_tiler, [0, 2]),
                        (mma_tile_coord_mnl[0], None),
                    )
                    # (chunk, chunks-per-k-tile, window-atoms, RestK)
                    # SFB is chunk-granular: place the tile's SF window at atom
                    # coordinate j*tile_n/128, as a domain_offset because it is
                    # free-form, not a tiler multiple (tile_n=64 shares an atom
                    # between adjacent tiles, tile_n=192 advances 1.5 atoms per
                    # tile — both are just this one formula).
                    sfb_first_atom = (mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1]) // 128
                    gSFB_chunks = cute.local_tile(
                        cute.domain_offset(
                            (None, None, sfb_first_atom),
                            varlen_manager.offset_batch_SFB(mSFB_chunks, batch_idx),
                        ),
                        (256, self.sfb_chunks_per_ktile, self.sfb_window_atoms),
                        (0, None, 0),
                    )

                # Partition global tensor for TiledMMA_A/B/D
                # Then partition global/shared tensor for TMA load A/B
                len_k = varlen_manager.len_k(batch_idx)
                # block_copy's lowering wants the coordinate held fixed by the
                # multicast mask: A/SFA are same-M across N peers, while B/SFB
                # are same-N across M peers. Degenerate cluster dimensions are
                # left for the compiler lowering to simplify.
                a_tma_multicast = {
                    "cluster_shape": self.cluster_shape_mnk[:2],
                    "multicast_dim": "M",
                }
                b_tma_multicast = {
                    "cluster_shape": self.cluster_shape_mnk[:2],
                    "multicast_dim": "N",
                }
                # SFB is duplicated (not V-split like B) across the 2-CTA MMA
                # pair, so unlike B its multicast group spans every cluster-M
                # CTA including the pair peer: use_2cta_mma_inst=False makes
                # the lowering slice/multicast across all of them (halving SFB
                # gmem traffic within a pair), while the op's cta_group still
                # aggregates transaction bytes at the pair leader's barrier.
                sfb_tma_multicast = {
                    "cluster_shape": self.cluster_shape_mnk[:2],
                    "multicast_dim": "N",
                    "use_2cta_mma_inst": False,
                }
                copy_A, prefetch_A = None, None
                if const_expr(not self.gather_A):
                    # (MMA, MMA_M, MMA_K, RestK)
                    tCgA = thr_mma.partition_A(gA_mk)
                    copy_A = copy_utils.tma_get_block_copy_fn(
                        tma_atom_a, src_tensor=tCgA, dst_tensor=sA, tma_multicast=a_tma_multicast
                    )
                else:
                    # For varlen_m paths (TMA or cp.async): consume indices from
                    # a_prefetch_pipeline once per work tile.
                    sAIdx_stage = sAIdx
                    if const_expr(varlen_m):
                        a_prefetch_pipeline.consumer_wait(a_prefetch_consumer_state)
                        sAIdx_stage = sAIdx[None, a_prefetch_consumer_state.index]
                    copy_A, prefetch_A = self._make_gather_A_copy(
                        mA_mkl,
                        sA,
                        sAIdx_stage,
                        tma_atom_a,
                        varlen_manager,
                        tile_coord_mnkl,
                        batch_idx,
                        warp_idx,
                    )
                    if const_expr(varlen_m):
                        a_prefetch_pipeline.consumer_release(a_prefetch_consumer_state)
                        a_prefetch_consumer_state.advance()
                    if const_expr(prefetch_A is not None):
                        prefetch_A = partial(prefetch_A, a_prefetch_pipeline)
                # (MMA, MMA_N, MMA_K, RestK)
                tCgB = thr_mma.partition_B(gB_nk)
                if const_expr(self.blockscaled):
                    # (MMA, MMA_M, MMA_K)
                    tCgSFA = thr_mma.partition_A(gSFA_mkl)
                # TMA load B partition_S/D
                copy_B = copy_utils.tma_get_block_copy_fn(
                    tma_atom_b, src_tensor=tCgB, dst_tensor=sB, tma_multicast=b_tma_multicast
                )
                copy_SFA, copy_SFB = None, None
                if const_expr(self.blockscaled):
                    #  TMA load SFA partition_S/D
                    copy_SFA = copy_utils.tma_get_block_copy_fn(
                        tma_atom_sfa,
                        src_tensor=tCgSFA,
                        dst_tensor=sSFA,
                        tma_multicast=a_tma_multicast,
                    )
                    # SFB multicast: same-N across all cluster-M CTAs (see
                    # sfb_tma_multicast above).
                    copy_SFB = copy_utils.tma_get_block_copy_fn(
                        tma_atom_sfb,
                        src_tensor=gSFB_chunks,
                        dst_tensor=sSFB_chunks,
                        tma_multicast=sfb_tma_multicast,
                    )
                k_tile_total = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                k_tile_start, k_tile_cnt = tile_scheduler.get_split_k_tile_range(
                    k_tile_total, split_idx
                )
                iket.range_push("tma_load")
                if const_expr(not self.gather_A):
                    ab_producer_state = self.load_tma(
                        ab_pipeline,
                        ab_producer_state,
                        [copy_A, copy_B, copy_SFA, copy_SFB],
                        k_tile_cnt,
                        k_tile_start=k_tile_start,
                    )
                elif const_expr(self.use_tma_gather):
                    ab_producer_state, a_prefetch_consumer_state = self.load_AB_tma_gather(
                        ab_pipeline,
                        ab_producer_state,
                        a_prefetch_consumer_state,
                        copy_A,
                        prefetch_A,
                        copy_B,
                        k_tile_cnt,
                    )
                else:
                    ab_producer_state, a_prefetch_consumer_state = self.load_AB_gather_A(
                        ab_pipeline,
                        ab_producer_state,
                        a_prefetch_consumer_state,
                        copy_A,
                        prefetch_A,
                        copy_B,
                        k_tile_cnt,
                    )
                iket.range_pop()
                if const_expr(epi_load_barrier is not None):
                    # In the first work tile, the epi load warp will wait for the signal
                    # from the mainloop load warp to start loading C, to avoid interfering
                    # with loading A and B.
                    if do_epi_load_barrier_arrive:
                        epi_load_barrier.arrive()
                        do_epi_load_barrier_arrive = Boolean(False)
                # Advance to next tile
                iket.range_push("sched_fetch")
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                iket.range_pop()
            # Wait A/B buffer empty
            if warp_idx == self.ab_load_warp_id:
                ab_pipeline.producer_tail(ab_producer_state)

        # Specialized scheduler warp
        if const_expr(self.is_persistent or self.gather_A):
            if warp_idx == self.scheduler_warp_id:
                # PDL: wait for prior kernel before reading CLC state (matches cutlass C++ sched)
                if const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()
                if const_expr(self.gather_A):
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                is_scheduler_warp = True
                if const_expr(cute.size(cluster_layout_vmnk) > 1):
                    is_scheduler_warp = cute.arch.block_idx_in_cluster() == 0
                # Persistent tile scheduling loop
                tile_scheduler = TileSchedulerCls(is_scheduler_warp=is_scheduler_warp)
                work_tile = tile_scheduler.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    # Advance to next tile
                    iket.range_push("clc_produce")
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    iket.range_pop()
                    iket.range_push("clc_consume")
                    work_tile = tile_scheduler.get_current_work()
                    iket.range_pop()
                    # End of persistent scheduler loop
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()
                    # Drain the pending padding tail with SERIAL-OBSERVED
                    # cancels (issue -> wait -> decode, one at a time), gated
                    # on a decoded-phantom retirement. The old fire-and-forget
                    # burst spray caused the July 2026 varlen corruption; see
                    # cancel_pending_tail and
                    # AI/clc_spurious_invalid_investigation.md.
                    tile_scheduler.cancel_pending_tail()

        # Specialized A-index prefetch warp (gather_A only)
        if const_expr(self.gather_A):
            if warp_idx == self.a_prefetch_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                tile_M = self.cta_tile_shape_mnk[0]
                tile_K = self.cta_tile_shape_mnk[2]
                tiled_copy_AIdx = copy_utils.tiled_copy_1d(Int32, num_threads=32, is_async=True)
                thr_copy_AIdx = tiled_copy_AIdx.get_slice(cute.arch.lane_idx())
                tAsAIdx = thr_copy_AIdx.partition_D(sAIdx)
                tAcAIdx = thr_copy_AIdx.partition_S(
                    cute.make_identity_tensor(tile_M if varlen_m else tile_K)
                )
                # Persistent tile scheduling loop
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                a_prefetch_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.a_prefetch_stage
                )
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    mAIdx_mk = varlen_manager.offset_batch_AIdx(batch_idx)
                    if const_expr(varlen_m):
                        # (tile_M,)
                        gAIdx = cute.local_tile(mAIdx_mk, (tile_M,), (tile_coord_mnkl[0],))
                        tAgAIdx = thr_copy_AIdx.partition_S(gAIdx)
                        len_m = varlen_manager.len_m(batch_idx)
                        m_limit = len_m - tile_coord_mnkl[0] * tile_M
                        tApAIdx_m = cute.make_rmem_tensor((1, tAsAIdx.shape[1]), Boolean)
                        for m in cutlass.range(tAsAIdx.shape[1], unroll_full=True):
                            tApAIdx_m[0, m] = tAcAIdx[0, m] < m_limit
                        a_prefetch_pipeline.producer_acquire(a_prefetch_producer_state)
                        cute.copy(
                            thr_copy_AIdx,
                            tAgAIdx,
                            tAsAIdx[None, None, a_prefetch_producer_state.index],
                            pred=tApAIdx_m,
                        )
                        a_prefetch_pipeline.producer_commit(a_prefetch_producer_state)
                        a_prefetch_producer_state.advance()
                    else:
                        # (tile_K, RestK)
                        gAIdx = cute.flat_divide(mAIdx_mk, (tile_K,))
                        tAgAIdx = thr_copy_AIdx.partition_S(gAIdx)
                        len_k = varlen_manager.len_k(batch_idx)
                        k_tile_cnt = cute.ceil_div(len_k, tile_K)
                        for k_tile in cutlass.range(k_tile_cnt - 1, unroll=1):
                            a_prefetch_pipeline.producer_acquire(a_prefetch_producer_state)
                            cute.copy(
                                thr_copy_AIdx,
                                tAgAIdx[None, None, k_tile],
                                tAsAIdx[None, None, a_prefetch_producer_state.index],
                            )
                            a_prefetch_pipeline.producer_commit(a_prefetch_producer_state)
                            a_prefetch_producer_state.advance()
                        if 0 < k_tile_cnt:
                            k_tile = k_tile_cnt - 1
                            k_limit = len_k - k_tile * tile_K
                            tApAIdx_k = cute.make_rmem_tensor((1, tAsAIdx.shape[1]), Boolean)
                            for m in cutlass.range(tAsAIdx.shape[1], unroll_full=True):
                                tApAIdx_k[0, m] = tAcAIdx[0, m] < k_limit
                            a_prefetch_pipeline.producer_acquire(a_prefetch_producer_state)
                            cute.copy(
                                tiled_copy_AIdx,
                                tAgAIdx[None, None, k_tile],
                                tAsAIdx[None, None, a_prefetch_producer_state.index],
                                pred=tApAIdx_k,
                            )
                            a_prefetch_pipeline.producer_commit(a_prefetch_producer_state)
                            a_prefetch_producer_state.advance()
                    # Advance to next tile
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()

        # Specialized TMA epi load warp
        if warp_idx == self.epi_load_warp_id:
            if const_expr(self.gather_A):
                cute.arch.setmaxregister_decrease(self.num_regs_other)
            # PDL: wait for prior kernel before any C TMA loads (matches cutlass C++ epi_load)
            if const_expr(self.use_pdl and has_epi_load):
                cute.arch.griddepcontrol_wait()
            if const_expr(has_epi_load):
                epi_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.epi_c_stage
                )
                do_epi_load_barrier_wait = Boolean(True)
                # Persistent tile scheduling loop
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    # Get tile coord from tile scheduler
                    # (pid_m, pid_n, split_idx | None, batch_idx), decoded by the scheduler
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
                    copy_C = None
                    if const_expr(has_C):
                        copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                            tma_atom_c,
                            varlen_manager.offset_batch_epi(mC_mnl, batch_idx),
                            self.cta_tile_shape_mnk[:2],
                            epi_tile,
                            sC,
                            tile_coord_mnkl,
                        )
                        copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)
                    tile_load_copy_fns = self.epi_tile_load_g2s_copy_fns(
                        epilogue_params,
                        epi_smem_tensors,
                        tile_coord_mnkl,
                        varlen_manager,
                        epi_pipeline,
                    )
                    copy_epi_load = copy_utils.chain_tma_producer_copy_fns(
                        (copy_C, *tile_load_copy_fns)
                    )
                    if do_epi_load_barrier_wait:
                        epi_load_barrier.arrive_and_wait()
                        do_epi_load_barrier_wait = Boolean(False)
                    epi_tile_shape = cute.zipped_divide(
                        cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
                    ).shape[1]
                    epi_tile_num = const_expr(cute.size(epi_tile_shape))
                    # Hier subtile coords, ordered exactly as the epilogue store
                    # loop consumes stages (gemm_base.epilogue): every copy fn
                    # receives a subscriptable (epi_m, epi_n) coordinate — flat
                    # indices only match consumption order for m-major epilogues.
                    epi_load_layout = cute.make_ordered_layout(
                        epi_tile_shape, order=(0, 1) if const_expr(self.epi_m_major) else (1, 0)
                    )
                    # Split-K (serial/parallel): only the finalizing split runs the
                    # epilogue, so only its tiles consume C — skip the loads (and the
                    # pipeline slots) for non-finalizing splits, symmetric with the
                    # epilogue warps' skip in epilogue_split_k. The const_expr prefix
                    # folds at trace time (quack.dsl.mixed_constexpr_if), so
                    # split_k == 1 codegen has no dynamic if at all.
                    if (
                        const_expr(self.split_k == 1 or self.split_k_mode == SplitKMode.SEPARATE)
                        or split_idx == self.split_k - 1
                    ):
                        for epi_idx in cutlass.range(epi_tile_num, unroll=1):
                            epi_pipeline.producer_acquire(epi_producer_state)
                            copy_epi_load(
                                src_idx=epi_load_layout.get_hier_coord(epi_idx),
                                producer_state=epi_producer_state,
                            )
                            # Epi pipeline's producer commit is a NOP
                            epi_pipeline.producer_commit(epi_producer_state)
                            epi_producer_state.advance()
                    # Advance to next tile
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                epi_pipeline.producer_tail(epi_producer_state)

        # Specialized MMA warp
        if warp_idx == self.mma_warp_id:
            if const_expr(self.gather_A):
                cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Retrieving tensor memory ptr and make accumulator tensor
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # Partition shared/tensor memory tensor for TiledMMA_A/B/D
            # (MMA, MMA_M, MMA_K, STAGE)
            tCrA = tiled_mma.make_fragment_A(sA_mma)
            # (MMA, MMA_N, MMA_K, STAGE)
            tCrB = tiled_mma.make_fragment_B(sB)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            if const_expr(self.blockscaled):
                # Make SFA tmem tensor
                acc_tmem_col_offset = const_expr(
                    tcgen05.find_tmem_tensor_col_offset(
                        tCtAcc_base
                        if const_expr(not self.overlap_accum_sf)
                        else tCtAcc_base[None, None, None, 0]
                    )
                )
                sfa_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + acc_tmem_col_offset, dtype=self.sf_dtype
                )
                # (MMA, MMA_M, MMA_K)
                tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfa_smem_layout, (None, None, None, 0)),
                )
                tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
                # Make SFB tmem tensor
                sfa_tmem_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                sfb_tmem_col_offset = acc_tmem_col_offset + sfa_tmem_col_offset
                sfb_tmem_base_ptr = acc_tmem_ptr + sfb_tmem_col_offset
                sfb_tmem_ptr = cute.recast_ptr(sfb_tmem_base_ptr, dtype=self.sf_dtype)
                # (MMA, MMA_N, MMA_K)
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfb_smem_layout, (None, None, None, 0)),
                )
                tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            else:
                tCtSFA, tCtSFB = None, None

            # Persistent tile scheduling loop
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                # (pid_m, pid_n, split_idx | None, batch_idx), decoded by the scheduler
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
                k_len = varlen_manager.len_k(batch_idx)
                k_tile_total = cute.ceil_div(k_len, self.mma_tiler[2])
                _, k_tile_cnt = tile_scheduler.get_split_k_tile_range(k_tile_total, split_idx)
                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                acc_stage_idx = (
                    acc_producer_state.phase ^ 1
                    if const_expr(self.overlap_accum_sf)
                    else acc_producer_state.index
                )
                tCtAcc = tCtAcc_base[None, None, None, acc_stage_idx]
                tCtSFB_mma = tCtSFB
                if const_expr(self.blockscaled and self.sfb_n_atom_misaligned):
                    # Odd N-tiles start 64 into a 128-wide SF atom: shift the SFB
                    # tmem base by 2 columns (in the atom layout (32,4):(16,4),
                    # N+64 is element offset 8 = 2 tmem columns).
                    tCtSFB_mma = cute.make_tensor(
                        cute.recast_ptr(
                            sfb_tmem_base_ptr + Int32((tile_coord_mnkl[1] % 2) * 2),
                            dtype=self.sf_dtype,
                        ),
                        tCtSFB.layout,
                    )
                copy_s2t_sfa, copy_s2t_sfb = None, None
                sf_valid_insts = None
                if const_expr(self.blockscaled):
                    copy_s2t_sfa = copy_utils.s2t_get_copy_fn(sSFA, tCtSFA, self.cta_group)
                    copy_s2t_sfb = copy_utils.s2t_get_copy_fn(sSFB, tCtSFB, self.cta_group)
                    # Exploits the fact that for mxfp8 the MMA instruction K size
                    # equals the SF vec size (== 32), so one instruction consumes
                    # exactly one SF block and the mma loop can skip the
                    # instructions for SF pad blocks on a ragged-K last tile (see
                    # the comment in self.mma). fp4 has inst_k 64 spanning
                    # multiple SF blocks, but we don't do varlen_k for
                    # mxfp4/nvfp4. Valid instructions in that tile; % maps
                    # "aligned or full tile" to 0 = nothing to skip.
                    if const_expr(self.mma_inst_shape_mnk[2] == self.sf_vec_size):
                        num_insts = self.mma_tiler[2] // self.mma_inst_shape_mnk[2]
                        sf_valid_insts = (
                            cute.ceil_div(k_len % self.mma_tiler[2], self.sf_vec_size) % num_insts
                        )
                iket.range_push("mma")
                ab_consumer_state, acc_producer_state, tiled_mma = self.mma(
                    ab_pipeline,
                    acc_pipeline,
                    ab_consumer_state,
                    acc_producer_state,
                    tiled_mma,
                    tCrA,
                    tCrB,
                    tCtAcc,
                    k_tile_cnt,
                    is_leader_cta,
                    cta_rank_in_cluster,
                    tCtSFA,
                    tCtSFB_mma,
                    copy_s2t_sfa,
                    copy_s2t_sfb,
                    sf_valid_insts,
                )
                if const_expr(self.overlap_accum_sf):
                    # After iter 0, 2, ..., shift tmem ptr by -256.
                    # After iter 1, 3, ..., shift tmem ptr by 256.
                    tCtSFA, tCtSFB = [
                        cute.make_tensor(
                            cute.recast_ptr(
                                # Doing tmem ptr arithmetic requires 32-bit type, wrong otherwise
                                cute.recast_ptr(mT.iterator, dtype=Float32)
                                + cute.assume(
                                    acc_tmem_col_offset * (acc_producer_state.phase * 2 - 1),
                                    divby=acc_tmem_col_offset,
                                ),
                                dtype=self.sf_dtype,
                            ),
                            mT.layout,
                        )
                        for mT in [tCtSFA, tCtSFB]
                    ]
                iket.range_pop()
                # Advance to next tile
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

            # PDL: hint the next kernel to launch early now that all MMAs are issued
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

            tmem_alloc_barrier.arrive()
            # Wait for accumulator buffer empty
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            if const_expr(self.gather_A):
                cute.arch.setmaxregister_increase(self.num_regs_epi)
            # Alloc tensor memory buffer
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()

            is_tma_warp = Boolean(warp_idx == self.epilog_warp_id[0])

            # Retrieving tensor memory ptr and make accumulator tensor
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Partition for epilogue
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, epi_tile, use_2cta_instrs
            )

            tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                tiled_copy_t2r, self.d_layout, self.d_dtype, tTR_rD, sD, epi_tidx
            )
            tRS_rC, tSR_rC, tSR_sC = None, None, None
            tiled_copy_s2r = None
            if const_expr(mC_mnl is not None):
                tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                    tiled_copy_t2r, self.c_layout, self.c_dtype, sC, tRS_rD.layout, epi_tidx
                )

            # Persistent tile scheduling loop
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            epi_store_pipeline = self.make_epi_store_pipeline()
            epi_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.epi_c_stage
            )
            while work_tile.is_valid_tile:
                # Prefetch the next work tile before the epilogue: the response is
                # already in smem (3-stage sched pipeline), and consuming it here
                # hides the ~300ns decode (swizzle + async fence) behind this tile's
                # epilogue — the pacing chain for small-K / double-store epilogues.
                # advance_to_next_work stays after the body: num_tiles_executed must
                # count completed tiles during the body (sD stage cycling).
                next_work_tile = tile_scheduler.get_current_work()
                # Get tile coord from tile scheduler
                # (pid_m, pid_n, split_idx | None, batch_idx), decoded by the scheduler
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                epi_acc_stage = (
                    acc_consumer_state.index
                    if const_expr(not self.overlap_accum_sf)
                    else acc_consumer_state.phase
                )
                tTR_tAcc = tTR_tAcc_base[None, None, None, None, None, epi_acc_stage]
                # Wait for accumulator buffer full
                acc_pipeline.consumer_wait(acc_consumer_state)

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
                        epi_tile,
                        sD,
                        tile_coord_mnkl,
                    )

                copy_C = None  # We're using a separate warp to load C

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                k_len = varlen_manager.len_k(batch_idx)
                epi_tile_num = cute.size(
                    cute.zipped_divide(cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile),
                    mode=[1],
                )
                clear_acc = varlen_k and k_len == 0
                if const_expr(self.split_k > 1):
                    # An empty split (split_k > total k-tiles) still runs the epilogue with a
                    # zero contribution so the serial-semaphore turnstile advances.
                    k_tile_total = cute.ceil_div(k_len, self.cta_tile_shape_mnk[2])
                    _, k_tile_cnt_split = tile_scheduler.get_split_k_tile_range(
                        k_tile_total, split_idx
                    )
                    clear_acc = k_tile_cnt_split == 0
                load_acc_subtile = partial(
                    self.epi_load_acc_subtile,
                    tiled_copy_t2r,
                    tiled_copy_r2s,
                    tTR_tAcc,
                    tTR_rAcc,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    acc_release_idx=self.iter_acc_early_release
                    if const_expr(self.overlap_accum_sf)
                    else epi_tile_num - 1,
                    clear_acc=clear_acc,
                )

                # Split-K (serial/parallel): non-finalizing splits commit raw f32 partials
                # to the tile's workspace and skip the epilogue; the last split waits for
                # the tile's completion flag and runs the full epilogue on the summed
                # accumulator (CUTLASS-3.x stream-K fixup semantics).
                iket.range_push("epilogue")
                epi_fn = partial(
                    self.epilogue,
                    epilogue_params,
                    epi_smem_tensors,
                    epi_pipeline,
                    epi_store_pipeline,
                    epi_read_state,
                    None,  # epi_producer_state
                    epi_tile,
                    # load_acc_subtile is the one argument left unbound
                    tRS_rD=tRS_rD,
                    tRS_rC=tRS_rC,
                    tiled_copy_t2r=tiled_copy_t2r,
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
                    tidx=epi_tidx,
                    is_tma_warp=is_tma_warp,
                )
                epi_read_state, _ = self.epilogue_split_k(
                    epilogue_params,
                    epi_fn,
                    load_acc_subtile,
                    tRS_rD,
                    epi_tile,
                    epi_read_state,
                    None,  # epi_producer_state
                    epi_store_pipeline,
                    tile_coord_mnkl,
                    self.epilogue_barrier,
                    epi_tidx,
                    is_tma_warp,
                )
                # acc_pipeline.consumer_release was already called in self.epi_load_acc_subtile
                acc_consumer_state.advance()
                iket.range_pop()

                # Advance to next tile
                tile_scheduler.advance_to_next_work()
                work_tile = next_work_tile

            # Wait for D store complete
            if is_tma_warp:
                epi_store_pipeline.producer_tail()

            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)

    @cute.jit
    def _make_gather_A_copy(
        self,
        mA_mkl: cute.Tensor,
        sA: cute.Tensor,
        sAIdx: cute.Tensor,  # if varlen, this is already sliced into the current prefetch stage
        tma_atom_a: Optional[cute.CopyAtom],
        varlen_manager: VarlenManager,
        tile_coord_mnkl,
        batch_idx: Int32,
        warp_idx: Int32,
    ):
        """Create copy_A and prefetch_A for gather_A (cp.async and TMA gather paths).
        sAIdx: sAIdx sliced to the current prefetch stage (for varlen_m paths).
        For varlen_k TMA gather, sAIdx (full) is used instead.
        """
        varlen_m = varlen_manager.varlen_m
        varlen_k = varlen_manager.varlen_k
        if const_expr(varlen_m):
            mA_mk = mA_mkl
        else:
            mA_mk = cute.local_tile(
                mA_mkl, (self.cta_tile_shape_mnk[0],), (tile_coord_mnkl[0], None)
            )
        len_m = varlen_manager.len_m(batch_idx)
        len_k = varlen_manager.len_k(batch_idx)
        num_cta = 2 if self.use_2cta_instrs else 1
        dma_warp_idx = warp_idx - self.ab_load_warp_id
        dma_tidx = cute.arch.thread_idx()[0] - self.ab_load_warp_id * 32
        copy_A, prefetch_A = None, None
        if const_expr(self.use_tma_gather):
            if const_expr(varlen_m):
                copy_A = copy_utils.gather_m_get_tma_copy_fn(
                    tma_atom_a,
                    mA_mk,
                    sA,
                    sAIdx,
                    dma_warp_idx,
                    num_warps=self.num_ab_load_warps,
                    num_cta=num_cta,
                )
            elif const_expr(varlen_k):
                col_idx = Int32(tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0])
                copy_A, prefetch_A = copy_utils.gather_k_get_tma_copy_fn(
                    tma_atom_a,
                    sA,
                    sAIdx,
                    col_idx,
                    dma_warp_idx,
                    num_warps=self.num_ab_load_warps,
                    num_cta=num_cta,
                )
        else:
            # cp.async path
            tiled_copy_A = self._make_gmem_tiled_copy_A(
                self.a_dtype, self.a_layout, self.num_ab_load_warps * 32
            )
            thr_copy_A = tiled_copy_A.get_slice(dma_tidx)
            if const_expr(varlen_m):
                copy_A = copy_utils.gather_m_get_copy_fn(
                    thr_copy_A,
                    mA_mk,
                    sA,
                    sAIdx,
                    limit_m=len_m - tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0],
                    limit_k=len_k,
                )
            else:
                copy_A, prefetch_A = copy_utils.gather_k_get_copy_fn(
                    thr_copy_A,
                    mA_mk,
                    sA,
                    sAIdx,
                    limit_m=len_m - tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0],
                    limit_k=len_k,
                )
        return copy_A, prefetch_A

    @cute.jit
    def load_AB_gather_A(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        a_prefetch_consumer_state: Optional[cutlass.pipeline.PipelineState],
        copy_A: Callable,
        prefetch_A: Optional[Callable],
        copy_B: Callable,
        k_tile_cnt: Int32,
        varlen_m: bool = True,
    ) -> Tuple[cutlass.pipeline.PipelineState, Optional[cutlass.pipeline.PipelineState]]:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt
        peek_ab_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # TMA load on B and cp.async on A
        for k_tile in cutlass.range(k_tile_cnt - 1, unroll=2 if const_expr(varlen_m) else 1):
            smem_idx = ab_producer_state.index
            prefetch_out = ()
            if const_expr(prefetch_A is not None):  # Prefetch early, even before smem is free
                prefetch_out = (prefetch_A(k_tile, smem_idx, a_prefetch_consumer_state),)
                a_prefetch_consumer_state.advance()
            # Wait for A/B buffers to be empty before loading into them
            # Also sets the transaction barrier for the A/B buffers
            # A tiny bit faster to rotate the warp that does TMA
            is_tma_warp = warp_idx == self.ab_load_warp_id + (k_tile % self.num_ab_load_warps)
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status, is_tma_warp)
            # A bit faster to load B first while we calculate the indices for A
            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            if is_tma_warp:
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
            smem_idx = ab_producer_state.index
            prefetch_out = ()
            if const_expr(prefetch_A is not None):  # Prefetch early, even before smem is free
                prefetch_out = (prefetch_A(k_tile, smem_idx, a_prefetch_consumer_state, pred=True),)
                a_prefetch_consumer_state.advance()
            is_tma_warp = warp_idx == self.ab_load_warp_id + k_tile % self.num_ab_load_warps
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status, is_tma_warp)
            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            if is_tma_warp:
                copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_A(k_tile, smem_idx, *prefetch_out, pred=True)
            ab_pipeline.producer_cpasync_commit(ab_producer_state)
            ab_producer_state.advance()
        return ab_producer_state, a_prefetch_consumer_state

    @cute.jit
    def load_AB_tma_gather(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        a_prefetch_consumer_state: Optional[cutlass.pipeline.PipelineState],
        copy_A: Callable,
        prefetch_A: Optional[Callable],
        copy_B: Callable,
        k_tile_cnt: Int32,
    ) -> Tuple[cutlass.pipeline.PipelineState, Optional[cutlass.pipeline.PipelineState]]:
        """Unified TMA gather loading loop for both varlen_m and varlen_k.

        For varlen_m: a_prefetch_pipeline is None, copy_A receives k_tile as src_idx.
        For varlen_k: a_prefetch_pipeline is set, copy_A receives the prefetch stage index,
          and indices are consumed/released per K-tile.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        peek_ab_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            smem_idx = ab_producer_state.index
            prefetch_out = ()
            if const_expr(prefetch_A is not None):  # Prefetch early, even before smem is free
                prefetch_out = (prefetch_A(k_tile, smem_idx, a_prefetch_consumer_state),)
                a_prefetch_consumer_state.advance()
            is_tma_warp = warp_idx == self.ab_load_warp_id + (k_tile % self.num_ab_load_warps)
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status, is_tma_warp)
            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            if is_tma_warp:
                copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_A(k_tile, smem_idx, *prefetch_out, tma_bar_ptr=tma_bar_ptr)
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
            peek_ab_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        return ab_producer_state, a_prefetch_consumer_state

    @cute.jit
    def mma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        acc_pipeline: cutlass.pipeline.PipelineAsync,
        ab_consumer_state: cutlass.pipeline.PipelineState,
        acc_producer_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        acc: cute.Tensor,
        k_tile_cnt: Int32,
        is_leader_cta: Boolean,
        cta_rank_in_cluster: Int32,
        tCtSFA: Optional[cute.Tensor] = None,
        tCtSFB: Optional[cute.Tensor] = None,
        copy_s2t_sfa: Optional[Callable] = None,
        copy_s2t_sfb: Optional[Callable] = None,
        sf_valid_insts_last_tile: Optional[Int32] = None,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState, cute.TiledMma]:
        blockscaled = const_expr(copy_s2t_sfa is not None)
        if const_expr(blockscaled):
            assert all(x is not None for x in (tCtSFA, tCtSFB))
            assert copy_s2t_sfb is not None
        skip_sf_pad_insts = const_expr(sf_valid_insts_last_tile is not None)
        # If gather_A and use_2cta_instrs, the cp.async for the non-leader CTA will
        # arrive at an mbarrier on the non-leader CTA side, then the mma warp of the non-leader
        # CTA will wait for that then arrive at the mbarrier on the leader CTA.
        need_nonleader_cta = const_expr(
            self.gather_A and self.use_2cta_instrs and not self.use_tma_gather
        )
        # Peek (try_wait) AB buffer full for k_tile = 0
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt and (is_leader_cta or need_nonleader_cta):
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        # Wait for accumulator buffer empty
        if is_leader_cta:
            acc_pipeline.producer_acquire(acc_producer_state)
        # Reset the ACCUMULATE field for each tile
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        # Mma mainloop
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            if const_expr(need_nonleader_cta):
                if not is_leader_cta:
                    ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                    with cute.arch.elect_one():
                        # The odd CTA signals the even CTA. The arrive must release this
                        # CTA's cp.async smem writes at cluster scope so that the leader's
                        # 2-CTA MMA, which reads our smem over DSMEM, is guaranteed to
                        # observe them; a plain mbarrier.arrive is only release.cta
                        # (https://github.com/Dao-AILab/quack/issues/63).
                        mbarrier_arrive_release_cluster(
                            ab_pipeline.sync_object_full.get_barrier(ab_consumer_state.index),
                            cta_rank_in_cluster & 0xFE,
                        )
            if is_leader_cta:
                # Conditionally wait for AB buffer full
                ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                if const_expr(need_nonleader_cta):
                    # consumer_wait acquires at cta scope only; pair the non-leader's
                    # cluster-scope release with a cluster-scope acquire of the (already
                    # completed) phase before the MMA reads the peer CTA's smem.
                    mbarrier_acquire_cluster(
                        ab_pipeline.sync_object_full.get_barrier(ab_consumer_state.index),
                        ab_consumer_state.phase,
                    )
                #  Copy SFA/SFB from smem to tmem
                if const_expr(blockscaled):
                    copy_s2t_sfa(ab_consumer_state.index)
                    copy_s2t_sfb(ab_consumer_state.index)
                # Ragged K: the last k-tile's SF atom holds pad bytes beyond the
                # valid scale blocks. We exploit the fact that for mxfp8 the MMA
                # instruction K size equals the SF vec size (both 32), i.e. each
                # instruction consumes exactly one SF block: skipping the
                # instructions for pad blocks — whose A/B values are
                # TMA-zero-filled and contribute nothing — means the pad scales
                # are never consumed and the gmem pad may be arbitrary (e8m0 0xFF
                # = NaN would otherwise poison the accumulator via 0-value x
                # NaN-scale products). Instruction issue is a leader-only
                # decision, so this covers 2-CTA MMA too. fp4 has inst_k 64 (2
                # SF blocks for mxfp4, 4 for nvfp4), but we don't do varlen_k
                # for those formats.
                # (The set/gemm sequence is duplicated below because the DSL
                # rejects closures capturing staged values inside a dynamic if.)
                if const_expr(skip_sf_pad_insts):
                    num_mma_insts = Int32(num_k_blocks)
                    if sf_valid_insts_last_tile > 0 and k_tile == k_tile_cnt - 1:
                        num_mma_insts = sf_valid_insts_last_tile
                for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    k_blk_coord = (None, None, k_blk_idx, ab_consumer_state.index)
                    if const_expr(blockscaled):
                        # Set SFA/SFB tensor to tiled_mma
                        sf_kblock_coord = (None, None, k_blk_idx)
                        if const_expr(skip_sf_pad_insts):
                            if k_blk_idx < num_mma_insts:
                                tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                                tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator)
                                cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        else:
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator)
                            cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    else:
                        cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                # Async arrive AB buffer empty
                ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()
            # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt and (is_leader_cta or need_nonleader_cta):
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        # Async arrive accumulator buffer full
        if is_leader_cta:
            acc_pipeline.producer_commit(acc_producer_state)
        acc_producer_state.advance()
        # If we don't return the tiled_mma, we get compiler error
        # "operand #0 does not dominate this use"
        return ab_consumer_state, acc_producer_state, tiled_mma

    @cute.jit
    def epi_load_acc_subtile(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tiled_copy_r2s: cute.TiledCopy,
        tTR_tAcc: cute.Tensor,
        tTR_rAcc: cute.Tensor,
        tRS_rD: cute.Tensor,
        epi_coord: [int, int],
        acc_pipeline: pipeline.PipelineAsync,
        acc_consumer_state: pipeline.PipelineState,
        acc_release_idx: int,
        clear_acc: Boolean = False,
        no_release: cutlass.Constexpr[bool] = False,
    ):
        if not clear_acc:
            # Load accumulator from tensor memory buffer to register
            cute.copy(tiled_copy_t2r, tTR_tAcc[None, None, None, epi_coord], tTR_rAcc)
            tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
            tRS_rD.store(tRS_rAcc.load())
        else:
            tRS_rD.fill(0.0)
        assert epi_coord[0] == 0  # For Sm100, we assume epi_M = 1
        # no_release: the epilogue prepass (epi_needs_acc_prepass) re-reads the
        # accumulator in the store pass, so it must stay valid.
        if const_expr(not no_release):
            if epi_coord[1] == acc_release_idx:
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)

    def epilog_tmem_copy_and_partition(
        self,
        tidx: Int32,
        tAcc: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
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
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout if self.d_layout is not None else LayoutEnum.ROW_MAJOR,
            self.d_dtype if self.d_dtype is not None else cutlass.BFloat16,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        cAcc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        cAcc_epi = cute.flat_divide(cAcc, epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_cAcc = thr_copy_t2r.partition_D(cAcc_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(tTR_cAcc[None, None, None, 0, 0].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_store_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        d_layout: Optional[LayoutEnum],
        dtype: Optional[Type[cutlass.Numeric]],
        tTR_rD: cute.Tensor,
        sD: cute.Tensor,
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rD: The partitioned accumulator tensor
        :type tTR_rD: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: Int32
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rD, tRS_sD) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rD: The partitioned tensor C (register source)
            - tRS_sD: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            d_layout if d_layout is not None else LayoutEnum.ROW_MAJOR,
            dtype if dtype is not None else cutlass.BFloat16,
            self.acc_dtype,
            tiled_copy_t2r,
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD) if sD is not None else None
        # (R2S, R2S_M, R2S_N)
        tRS_rD = tiled_copy_r2s.retile(tTR_rD)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    def epilog_smem_load_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        c_layout: LayoutEnum,
        dtype: Type[cutlass.Numeric],
        # tTR_rC: cute.Tensor,
        sC: cute.Tensor,
        tRS_rD_layout: cutlass.Layout,
        tidx: Int32,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            c_layout, dtype, self.acc_dtype, tiled_copy_t2r
        )
        store_op = copy_atom_r2s.op
        # m8n8 16-bit path
        if isinstance(store_op, StMatrix8x8x16bOp):
            op = LdMatrix8x8x16bOp(num_matrices=store_op.num_matrices, transpose=store_op.transpose)
        # m16n8 8-bit store -> m16n16 8-bit load
        elif isinstance(store_op, StMatrix16x8x8bOp) and store_op.num_matrices in [2, 4]:
            # transpose=True is enforced by the class
            op = LdMatrix16x16x8bOp(num_matrices=store_op.num_matrices // 2)
        else:
            op = cute.nvgpu.CopyUniversalOp()
        copy_atom_s2r = cute.make_copy_atom(op, dtype)
        tiled_copy_s2r = cute.make_tiled_copy_D(copy_atom_s2r, tiled_copy_t2r)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        tSR_sC = thr_copy_s2r.partition_S(sC)
        tRS_rC = cute.make_rmem_tensor(tRS_rD_layout, dtype)
        # (R2S, R2S_M, R2S_N)
        tSR_rC = tiled_copy_s2r.retile(tRS_rC)
        return tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC

    @cute.jit
    def make_ab_pipeline(
        self,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        is_leader_cta: Boolean,
    ) -> pipeline.PipelineAsync:
        # If gather_A and use_2cta_instrs, the cp.async for the non-leader CTA will
        # arrive at an mbarrier on the non-leader CTA side, then the mma warp of the non-leader
        # CTA will wait for that then arrive at the mbarrier on the leader CTA.
        # The producer count for the leader CTA is 1 (TMA) + num_cpasync_threads
        # + 1 (from non-leader CTA).
        # The producer count for the non-leader CTA is num_cpasync_threads
        # (TMA doesn't arrive there).
        if const_expr(not self.gather_A or self.use_tma_gather):
            producer_cnt = 1
        else:
            producer_cnt = self.num_ab_load_warps * cute.arch.WARP_SIZE
            if const_expr(not self.use_2cta_instrs):
                producer_cnt += 1
            else:
                producer_cnt += Int32(2) if is_leader_cta else Int32(0)
        # gather_A + 2cta leaves producer_cnt leader-dependent (Int32) — check skips there.
        pipeline_checks.check_arrive_count(
            "sm100 ab_pipeline.producer",
            producer_cnt,
            pipeline_checks.tma_producer_arrives(
                num_tma_warps=1,
                cpasync_warps=0
                if const_expr(not self.gather_A or self.use_tma_gather)
                else self.num_ab_load_warps,
            ),
            gather_A=self.gather_A,
            num_ab_load_warps=self.num_ab_load_warps,
        )
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, producer_cnt)
        # Each warp will contribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size
        pipeline_checks.check_arrive_count(
            "sm100 ab_pipeline.consumer",
            consumer_arrive_cnt,
            pipeline_checks.mcast_peer_ctas(
                num_mcast_ctas_a=self.num_mcast_ctas_a,
                num_mcast_ctas_b=self.num_mcast_ctas_b,
            ),
            num_mcast_ctas_a=self.num_mcast_ctas_a,
            num_mcast_ctas_b=self.num_mcast_ctas_b,
        )
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        if const_expr(not self.gather_A or self.use_tma_gather):
            pipeline_ab = PipelineTmaUmma.create(
                num_stages=self.ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_ab = PipelineTmaCpAsyncUmma.create(
                num_stages=self.ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        return pipeline_ab

    def make_acc_pipeline(self, cluster_layout_vmnk: cute.Layout) -> pipeline.PipelineAsync:
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = self.num_epi_warps * (2 if self.use_2cta_instrs else 1)
        # One arrive per epi warp (elected lane); with 2-CTA MMA both CTAs' epi warps
        # route their arrives to the leader CTA's barrier. per_warp is bound to the
        # elect_one_release flag passed to create below so flipping one trips the check.
        elect_one_release = True
        pipeline_checks.check_arrive_count(
            "sm100 acc_pipeline.consumer",
            num_acc_consumer_threads,
            pipeline_checks.async_thread_arrives(
                self.num_epi_warps,
                per_warp=elect_one_release,
                ctas_routed=2 if self.use_2cta_instrs else 1,
            ),
            num_epi_warps=self.num_epi_warps,
            use_2cta_instrs=self.use_2cta_instrs,
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        return PipelineUmmaAsync.create(
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
            elect_one_release=elect_one_release,
            # TMEM load consumers are already ordered by fence_view_async_tmem_load()
            syncwarp_before_release=False,
        )

    def make_sched_pipeline(
        self,
        cluster_layout_mnk: cute.Layout,
        has_C: bool = False,
    ) -> pipeline.PipelineAsync:
        # Threads/warps participating in this pipeline
        sched_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_layout_mnk)
        # Each warp will contribute 1 to the arrive count
        extra_warp_ids = (self.a_prefetch_warp_id,) if self.gather_A else ()
        warps_per_cta = self.num_ab_load_warps + len(
            (self.mma_warp_id, *self.epilog_warp_id, self.scheduler_warp_id, *extra_warp_ids)
        )
        if has_C:
            warps_per_cta += 1
        consumer_arrive_cnt = warps_per_cta * cluster_size
        # One arrive per consumer warp (elected lane); consumer_mask=0 routes every CTA
        # in the cluster to CTA 0's barrier. per_warp is bound to elect_one_release below.
        elect_one_release = True
        pipeline_checks.check_arrive_count(
            "sm100 sched_pipeline.consumer",
            consumer_arrive_cnt,
            pipeline_checks.async_thread_arrives(
                warps_per_cta, per_warp=elect_one_release, ctas_routed=cluster_size
            ),
            warps_per_cta=warps_per_cta,
            cluster_size=cluster_size,
        )
        sched_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        # Plain PipelineAsync on purpose (vs the DSL example's PipelineClcFetchAsync):
        # expect_tx is per-phase mbarrier state, so each mode's producer arms the full
        # barrier as a transaction barrier itself — CLC's multicast try_cancel or
        # STATIC/DYNAMIC's STAS st.async, both arrive_and_expect_tx(16) per CTA — and
        # only the consumer protocol (wait full, elect-one arrive at CTA 0's empty
        # barrier) is shared across modes. A CLC-specific pipeline would hardwire the
        # producer and still need this one for STATIC/DYNAMIC.
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

    @cute.jit
    def make_a_prefetch_pipeline(self) -> pipeline.PipelineAsync:
        producer_cnt = 32
        # One cp.async producer warp, all-thread arrives (cp.async.mbarrier.arrive per lane).
        pipeline_checks.check_arrive_count(
            "sm100 a_prefetch_pipeline.producer",
            producer_cnt,
            pipeline_checks.async_thread_arrives(1, per_warp=False),
            num_prefetch_warps=1,
        )
        a_prefetch_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, producer_cnt)
        consumer_arrive_cnt = self.num_ab_load_warps
        # One arrive per AB-load consumer warp; per_warp bound to elect_one_release below.
        elect_one_release = True
        pipeline_checks.check_arrive_count(
            "sm100 a_prefetch_pipeline.consumer",
            consumer_arrive_cnt,
            pipeline_checks.async_thread_arrives(
                self.num_ab_load_warps, per_warp=elect_one_release
            ),
            num_ab_load_warps=self.num_ab_load_warps,
        )
        a_prefetch_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return PipelineCpAsync.create(
            num_stages=self.a_prefetch_stage,
            producer_group=a_prefetch_producer_group,
            consumer_group=a_prefetch_consumer_group,
            defer_sync=True,
            elect_one_release=elect_one_release,
            syncwarp_before_release=True,
        )

    @classmethod
    def _compute_stages(
        cls,
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: cute.Tile,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Optional[Type[cutlass.Numeric]],
        sf_vec_size: Optional[int],
        d_dtype: Optional[Type[cutlass.Numeric]],
        c_dtype: Optional[Type[cutlass.Numeric]],
        d_layout: Optional[LayoutEnum],
        c_layout: Optional[LayoutEnum],
        epilogue_args: EpilogueArguments,
        prefetch_A_idx: Literal[None, "varlen_m", "varlen_k"],
        smem_capacity: int,
        occupancy: int,
        warp_shape_mnk: Tuple[int, int, int] | None = None,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

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
        :param d_dtype: Data type of operand C (output).
        :type d_dtype: type[cutlass.Numeric]
        :param d_layout: Layout enum of operand D.
        :type d_layout: LayoutEnum
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        blockscaled = sf_dtype is not None
        # Default ACC stages
        if const_expr(not blockscaled):
            num_acc_stage = 1 if mma_tiler_mnk[1] > 256 else 2
        else:
            num_acc_stage = 1 if mma_tiler_mnk[1] >= 256 else 2

        # Default D stages
        epi_stage = 4 if cute.size(epi_tile[1]) <= 16 else 2
        epi_smem_bytes = cls.epi_smem_bytes(
            epilogue_args, cta_tile_shape_mnk, epi_tile, warp_shape_mnk
        )
        has_tile_load = epi_smem_bytes.c_stage > 0
        epi_c_stage = (
            0
            if c_dtype is None and not has_tile_load
            else (4 if cute.size(epi_tile[1]) <= 16 else 2)
        )

        # Calculate smem layout and size for one stage of A, B, and C
        a_smem_layout_staged_one = sm100_utils.make_smem_layout_a(
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
        d_smem_layout_staged_one = (
            sm100_utils.make_smem_layout_epi(d_dtype, d_layout, epi_tile, 1)
            if d_dtype is not None
            else None
        )
        c_smem_layout_staged_one = (
            sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
            if c_dtype is not None
            else None
        )
        if const_expr(blockscaled):
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

        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_staged_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        if const_expr(prefetch_A_idx == "varlen_k"):  # Need smem to prefetch A indices
            ab_bytes_per_stage += Int32.width // 8 * cta_tile_shape_mnk[2]
        if const_expr(blockscaled):
            ab_bytes_per_stage += cute.size_in_bytes(
                sf_dtype, sfa_smem_layout_staged_one
            ) + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        if const_expr(prefetch_A_idx == "varlen_m"):
            mbar_helpers_bytes += Int32.width // 8 * cta_tile_shape_mnk[0] * 2
        d_bytes_per_stage = (
            cute.size_in_bytes(d_dtype, d_smem_layout_staged_one) if d_dtype is not None else 0
        )
        epi_bytes_per_stage = d_bytes_per_stage + epi_smem_bytes.d_stage
        epi_bytes = epi_smem_bytes.unstaged + epi_bytes_per_stage * epi_stage
        if const_expr(c_dtype is not None):
            c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
            epi_bytes += c_bytes_per_stage * epi_c_stage
        if const_expr(has_tile_load):
            epi_bytes += epi_smem_bytes.c_stage * epi_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        remaining_bytes = smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ab_stage = remaining_bytes // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        if epi_bytes_per_stage > 0:
            epi_stage += (remaining_bytes - ab_bytes_per_stage * ab_stage) // epi_bytes_per_stage
        return num_acc_stage, ab_stage, epi_stage, epi_c_stage

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        """
        Compute the number of tensor memory allocation columns.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tile.
        :type mma_tiler: tuple[int, int, int]
        :param num_acc_stage: The stage of the accumulator tensor.
        :type num_acc_stage: int

        :return: The number of tensor memory allocation columns.
        :rtype: int
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = cutlass.utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        return num_tmem_alloc_cols

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

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if b_dtype != a_dtype:
            is_valid = False
        ab_dtype = a_dtype
        if ab_dtype not in {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.TFloat32,
            cutlass.Uint8,
            cutlass.Int8,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        if (
            acc_dtype not in {Float32, cutlass.Float16, Int32}
            or acc_dtype == cutlass.Float16
            and ab_dtype not in {cutlass.Float16, cutlass.Float8E4M3FN, cutlass.Float8E5M2}
            or acc_dtype == Int32
            and ab_dtype not in {cutlass.Uint8, cutlass.Int8}
        ):
            is_valid = False
        if d_dtype is not None and (
            acc_dtype == Float32
            and d_dtype
            not in {
                Float32,
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
                cutlass.Float4E2M1FN,
                Int32,
                cutlass.Int8,
                cutlass.Uint8,
            }
            or acc_dtype == cutlass.Float16
            and d_dtype
            not in {
                cutlass.BFloat16,
                cutlass.Float16,
            }
            or acc_dtype == Int32
            and d_dtype
            not in {
                cutlass.BFloat16,
                cutlass.Float16,
                Float32,
                Int32,
                cutlass.Int8,
                cutlass.Uint8,
            }
        ):
            is_valid = False
        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid

    @staticmethod
    def _blockscaled_mma_inst_k(a_copy_dtype, b_copy_dtype) -> int:
        """Instruction K is a property of the MMA kind: kind::mxf4nvf4 (both
        operands packed fp4 - ONE atom parameterized by the format's scale
        config: vec 32 e8m0 = mxfp4, vec 16 = nvfp4) consumes 64 elements per
        instruction; kind::mxf8f6f4 (everything else - mixed A/B pairs and fp6,
        with sub-byte operands TMA-unpacked into byte-container smem) consumes
        32. Mirrors quack.blockscaled.operand.mma_kind_for_pair; keep the two in
        sync. test_mma_kind_mirrors_kernel_inst_k pins the correspondence.
        """
        both_packed_fp4 = (
            a_copy_dtype is cutlass.Float4E2M1FN and b_copy_dtype is cutlass.Float4E2M1FN
        )
        return 64 if both_packed_fp4 else 32

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        d_dtype: Optional[Type[cutlass.Numeric]],
        a_copy_dtype: Optional[Type[cutlass.Numeric]] = None,
        b_copy_dtype: Optional[Type[cutlass.Numeric]] = None,
    ) -> bool:
        """
        The per-architecture support gate: which (A, B, SF, D) dtype combinations
        THIS kernel implements. ``a_dtype``/``b_dtype`` are the MMA element types;
        ``a/b_copy_dtype`` the storage dtypes as seen at the FFI boundary (None:
        same; packed fp6 arrives as raw Uint8 bytes). (Hardware pair legality is
        a separate, wider set - see quack.blockscaled.operand.mma_kind_for_pair;
        this kernel's subset must stay consistent with those kind rules.)

        :param a_dtype: The data type of the A operand
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of the B operand
        :type b_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factors
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param d_dtype: The data type of the output tensor (None: no D store)
        :type d_dtype: Optional[Type[cutlass.Numeric]]

        :return: True if the dtypes and sf_vec_size are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        a_copy_dtype = a_dtype if a_copy_dtype is None else a_copy_dtype
        b_copy_dtype = b_dtype if b_copy_dtype is None else b_copy_dtype

        # Per-operand MMA element types this kernel implements. Mixed A/B pairs
        # (and any sub-byte operand outside both-packed-fp4) run tcgen05
        # kind::mxf8f6f4: the packed sub-byte gmem operand is expanded by TMA
        # into 8-bit smem containers (CU_TENSOR_MAP_DATA_TYPE_16U4/16U6_ALIGN16B,
        # selected via internal_type=Uint8). Both-packed-fp4 runs the denser
        # kind::mxf4nvf4 (one atom, scale config from the format). Storage must
        # be the packed sub-byte dtype itself;
        # byte-container (Uint8-per-element) storage has no kernel path.
        fp8 = {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
        sub_byte = {cutlass.Float4E2M1FN, cutlass.Float6E2M3FN, cutlass.Float6E3M2FN}
        supported = fp8 | sub_byte
        if a_dtype not in supported or b_dtype not in supported:
            is_valid = False
        # Copy dtype: the packed sub-byte dtype itself, except packed fp6 which
        # crosses the FFI boundary as raw bytes and is reinterpreted in-kernel.
        for mma_dt, copy_dt in ((a_dtype, a_copy_dtype), (b_dtype, b_copy_dtype)):
            if copy_dt is not mma_dt and not (mma_dt.width == 6 and copy_dt is cutlass.Uint8):
                is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        # vec 16 (nvfp4) requires fp4 on BOTH operands (kind::mxf4nvf4)
        if sf_vec_size == 16 and (
            a_dtype is not cutlass.Float4E2M1FN or b_dtype is not cutlass.Float4E2M1FN
        ):
            is_valid = False

        # Check valid d_dtype (None: gemm_act with no preact store)
        if d_dtype is not None and d_dtype not in {
            Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            cutlass.Float4E2M1FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mnk: Union[Tuple[int, int], Tuple[int, int, int]],
        cluster_shape_mn: Tuple[int, int],
        blockscaled: bool,
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param mma_tiler_mnk: The (M, N) or (M, N, K) shape of the MMA instruction tiler
        :type mma_tiler_mnk: Union[Tuple[int, int], Tuple[int, int, int]]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if not blockscaled:
            if mma_tiler_mnk[0] not in [64, 128, 256]:
                is_valid = False
        else:
            if mma_tiler_mnk[0] not in [128, 256]:
                is_valid = False
        mma_inst_n = mma_tiler_mnk[1] if mma_tiler_mnk[1] <= 256 else mma_tiler_mnk[1] // 2
        if not blockscaled:
            if mma_inst_n not in range(32, 257, 32):
                is_valid = False
        else:
            # Blockscaled supports tile_n multiples of 64: the SF tmem datapath
            # (tcgen05.cp write and MMA read) is 64-N granular, so odd
            # multiples of 32 (96, 160, 224) are unreachable.
            if mma_tiler_mnk[1] % 64 != 0 or not (64 <= mma_tiler_mnk[1] <= 256):
                is_valid = False
        if cluster_shape_mn[0] % (2 if mma_tiler_mnk[0] == 256 else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        if blockscaled:
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            if cluster_shape_mn[0] > 4 or cluster_shape_mn[1] > 4:
                is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        d_major: str,
        b_dtype: Optional[Type[cutlass.Numeric]] = None,  # defaults to ab_dtype (equal A/B)
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param d_major: The major axis of the C tensor
        :type d_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            # fp6 has no 16-byte packing quantum (16 * 8 / 6 is fractional); its
            # ALIGN16B unpack tensormap instead requires 128-element granularity.
            num_contiguous_elements = 128 if dtype.width == 6 else 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        b_dtype = ab_dtype if b_dtype is None else b_dtype
        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(b_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(d_dtype, d_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        a_dtype,
        b_dtype,
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        mma_tiler_mnk: Union[Tuple[int, int], Tuple[int, int, int]],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        d_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        Polymorphic over the operand kind: ``a_dtype`` / ``b_dtype`` are either
        both plain cutlass dtypes, selecting the dense kernel checks, or both
        :class:`quack.blockscaled.operand.BlockScaledFormat` descriptors,
        selecting the blockscaled checks with all format properties (scale
        config, storage dtype, packing) read from the descriptors. One of each
        is rejected.

        :param a_dtype: The data type of the A operand, or its blockscaled format
        :type a_dtype: Union[Type[cutlass.Numeric], BlockScaledFormat]
        :param b_dtype: The data type of the B operand, or its blockscaled format
        :type b_dtype: Union[Type[cutlass.Numeric], BlockScaledFormat]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mnk: The (M, N) or (M, N, K) shape of the MMA instruction tiler
        :type mma_tiler_mnk: Union[Tuple[int, int], Tuple[int, int, int]]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param d_major: The major axis of the C tensor
        :type d_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        # Lazy import: blockscaled/__init__ pulls blockscaled/utils, which
        # imports the kernel layer - a top-level import here would be circular.
        from torch._vendor.quack.blockscaled.operand import BlockScaledFormat
        from torch._vendor.quack.cute_dsl_utils import torch2cute_dtype_map

        a_is_blockscaled = isinstance(a_dtype, BlockScaledFormat)
        b_is_blockscaled = isinstance(b_dtype, BlockScaledFormat)
        if a_is_blockscaled != b_is_blockscaled:
            return False
        if not a_is_blockscaled:
            # Dense kernel: A and B must share one dtype.
            if a_dtype is not b_dtype:
                return False
            ab_dtype = a_dtype
            can_implement = True
            # Skip unsupported types
            if not GemmSm100.is_valid_dtypes(
                ab_dtype, ab_dtype, acc_dtype, d_dtype, a_major, b_major
            ):
                can_implement = False
            # Skip invalid mma tile shape and cluster shape
            if not GemmSm100.is_valid_mma_tiler_and_cluster_shape(
                mma_tiler_mnk, cluster_shape_mn, blockscaled=False
            ):
                can_implement = False
            # Skip illegal problem shape for load/store alignment
            if not GemmSm100.is_valid_tensor_alignment(
                m, n, k, l, ab_dtype, d_dtype, a_major, b_major, d_major
            ):
                can_implement = False
            return can_implement
        # Blockscaled: everything derives from the format descriptors.
        fmt_a, fmt_b = a_dtype, b_dtype
        # Deprecated byte-container sub-byte formats are host-side migration
        # inputs only; actual dispatch rejects them before kernel selection.
        if fmt_a.is_byte_container or fmt_b.is_byte_container:
            return False
        # tcgen05 blockscaled MMA accumulates in f32 only.
        if acc_dtype is not Float32:
            return False
        # The scale config is instruction-wide: both operands must share the
        # scale dtype and vec size (this also rejects nvfp4 mixed with anything
        # else).
        if fmt_a.scale_dtype != fmt_b.scale_dtype or fmt_a.sf_vec_size != fmt_b.sf_vec_size:
            return False
        a_mma_dtype = fmt_a.to_cutlass_dtype()
        b_mma_dtype = fmt_b.to_cutlass_dtype()
        a_copy_dtype = torch2cute_dtype_map[fmt_a.qdata_dtype]
        b_copy_dtype = torch2cute_dtype_map[fmt_b.qdata_dtype]
        sf_dtype = torch2cute_dtype_map[fmt_a.scale_dtype]
        sf_vec_size = fmt_a.sf_vec_size
        can_implement = True
        if (
            len(mma_tiler_mnk) == 3
            and mma_tiler_mnk[2] > 0
            and mma_tiler_mnk[2] % (sf_vec_size * 4) != 0
        ):
            can_implement = False
        if not GemmSm100.is_valid_dtypes_and_scale_factor_vec_size(
            a_mma_dtype,
            b_mma_dtype,
            sf_dtype,
            sf_vec_size,
            d_dtype,
            a_copy_dtype=a_copy_dtype,
            b_copy_dtype=b_copy_dtype,
        ):
            can_implement = False
        sub_byte = {cutlass.Float4E2M1FN, cutlass.Float6E2M3FN, cutlass.Float6E3M2FN}
        if (a_mma_dtype in sub_byte and a_major != "k") or (
            b_mma_dtype in sub_byte and b_major != "k"
        ):
            can_implement = False
        # TMA-unpack operands (sub-byte under kind::mxf8f6f4, i.e. any pair other
        # than both-packed-fp4) use the ALIGN16B tensormap formats, which require
        # the contiguous gmem extent to be a multiple of 128 elements.
        both_fp4 = a_mma_dtype is cutlass.Float4E2M1FN and b_mma_dtype is cutlass.Float4E2M1FN
        if (a_mma_dtype in sub_byte or b_mma_dtype in sub_byte) and not both_fp4 and k % 128 != 0:
            can_implement = False
        if not GemmSm100.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mnk, cluster_shape_mn, blockscaled=True
        ):
            can_implement = False
        if not GemmSm100.is_valid_tensor_alignment(
            m,
            n,
            k,
            l,
            a_copy_dtype,
            d_dtype,
            a_major,
            b_major,
            d_major,
            b_dtype=b_copy_dtype,
        ):
            can_implement = False
        return can_implement
