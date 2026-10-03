# Copyright (c) 2025-2026, QuACK team.
# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py
# SM120-style GEMM using warp-level MMA (MmaF16BF16Op) + ldmatrix.
# Unlike SM90 WGMMA (which reads A/B from SMEM directly), warp-level MMA
# requires explicit SMEM→RMEM copies via ldmatrix before each MMA instruction.

# Measured facts (RTX 5090, 2026-07-29 — see AI/sm120_transform_fp8_tuning.md
# for the full session data; verify before assuming they changed):
# - Warp-mma dense rates (boosted clocks): bf16/f32acc 255 TF, fp8/f32acc
#   507 TF via BOTH the Ada-era sm_89 mma.sync (MmaFP8Op) AND the plain
#   kind::f8f6f4 instruction — measured head-to-head via inline PTX in one
#   harness (507.9 vs 506.4): the 2x datapath is EXCLUSIVE to the
#   .block_scale variant (kind::mxf8f6f4 unit-scale 918 in that harness,
#   1005 in the real kernel; fp8/f16acc 1017). That is why fp8 rides
#   MmaMXF8Op with constant unit ue8m0 scales below, and why a plain
#   kind::f8f6f4 atom (absent from the DSL anyway) would buy nothing.
# - fp8 mma.sync f32 accumulate keeps ~21-22 mantissa bits, TRUNCATING (RZ):
#   +1 onto 2^n survives to n=20 (bf16 datapath: n=21); identical boundary
#   for the mxf8f6f4 instruction. No Hopper-style slow accum needed; drift
#   is ~(K/32)*2^-21 relative.
# - W4 decode shapes (m <= 64) want tile_m=128 with split-k pushed until
#   k-tiles/split < 32 (relax to 16 under ~96 CTAs); the 170-SM part rewards
#   grids well past the H100 112-CTA target. Short-K prefill wants tile_n
#   128 (pick_w4_cfg's sm120 branch has the measured numbers).
# - PTX 9.2 direct fp4/fp8->bf16x2 cvts decode 2.0x/1.6x faster than the
#   sm90 prmt-LUT/f16-route sequences (2.45x with the fused ue8m0 scale) —
#   see quack/blockscaled/nvfp4_utils.py `_arch_has_bf16_narrow_cvt`.

import math
from dataclasses import dataclass
from typing import Tuple, Type, Callable, Optional, Union
from functools import partial

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.nvgpu.warp import mma as _warp_mma
from cutlass import Int32, Float32, Boolean, const_expr
from cutlass.utils import SmemPartition

import cutlass.utils.blackwell_helpers as blackwell_helpers
from cutlass.utils import blockscaled_layout

from torch._vendor.quack.varlen_utils import VarlenManager
from torch._vendor.quack.pipeline import make_pipeline_state
from torch._vendor.quack import copy_utils
from torch._vendor.quack.gemm_sm90 import GemmSm90, NamedBarrierGemm
from torch._vendor.quack.gemm_config import SplitKMode
from torch._vendor.quack.tile_scheduler import ag_wait_m_tile
from torch._vendor.quack import sm80_utils
import torch._vendor.quack.sm90_utils as quack_sm90_utils


_MXF8F6F4_DTYPES = (
    cutlass.Float8E4M3FN,
    cutlass.Float8E5M2,
    cutlass.Float6E2M3FN,
    cutlass.Float6E3M2FN,
    cutlass.Float4E2M1FN,
)


@dataclass(frozen=True)
class MmaFP8MixedOp(warp.MmaFP8Op):
    """``MmaFP8Op`` with an independent B dtype.

    The sm_89 fp8 instruction takes ``.e4m3``/``.e5m2`` qualifiers
    independently (PTX emits e.g. ``mma.sync.aligned.m16n8k32.row.col.f32
    .e4m3.e5m2.f32``), and the ``MmaAtomSM89Type`` IR type already accepts
    separate (aType, bType) — only the upstream Python op class narrows to
    same-dtype. This is the mixed-fp8 FALLBACK for targets without
    kind::mxf8f6f4 (H100 CI proxy legs); fp6/fp4 pairs have no fallback (the
    SM89 atom's verifier admits e4m3/e5m2 only)."""

    b_dtype: Type[cutlass.Numeric] = cutlass.Float8E5M2

    def __post_init__(self) -> None:
        fp8_dtypes = (cutlass.Float8E4M3FN, cutlass.Float8E5M2)
        assert self.ab_dtype in fp8_dtypes and self.b_dtype in fp8_dtypes, (
            f"MmaFP8MixedOp operands must be fp8, got {self.ab_dtype} x {self.b_dtype}"
        )
        assert self.acc_dtype in (Float32, cutlass.Float16)
        assert self.shape_mnk in [(16, 8, 32), (16, 8, 16)]

    def _make_trait(self, *, loc=None, ip=None, **kwargs):
        shape_mnk = _warp_mma._pack_shape(self.shape_mnk, loc=loc, ip=ip)
        ty = _warp_mma._cute_nvgpu_ir.MmaAtomSM89Type.get(
            shape_mnk.type.attribute,
            self.ab_dtype.mlir_type,
            self.b_dtype.mlir_type,
            self.acc_dtype.mlir_type,
        )
        return _warp_mma.MmaFP8Trait(_warp_mma.make_atom(ty, loc=loc, ip=ip))


@dataclass(frozen=True)
class MmaMXF8F6F4OpFull(warp.MmaMXF8F6F4Op):
    """``MmaMXF8F6F4Op`` minus the upstream pair allow-list.

    PTX ``mma.sync.kind::mxf8f6f4`` admits ANY (a, b) pair from
    {e4m3, e5m2, e3m2, e2m3, e2m1} — the CUTLASS C++ ``SM120_16x8x32_TN``
    matrix instantiates all 25 combinations — but the DSL op class only
    whitelists fp4 x fp8. The trait construction (`_make_trait`) is fully
    dtype-generic, so this subclass keeps the arch / f32-acc / ue8m0-SF checks
    and drops the pair restriction. Same-dtype fp8 and fp4 still route to the
    dedicated ``MmaMXF8Op`` / ``MmaMXF4Op`` at dispatch (this op is only
    minted for genuinely mixed pairs and same-dtype fp6)."""

    def __post_init__(self) -> None:
        arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        assert arch in self.admissible_archs, (
            f"MmaMXF8F6F4OpFull expects arch in {self.admissible_archs}, got {arch}"
        )
        assert self.acc_dtype == Float32, "kind::mxf8f6f4 requires f32 accumulation"
        assert self.sf_type == cutlass.Float8E8M0FNU, "kind::mxf8f6f4 requires ue8m0 scales"
        assert self.a_dtype in _MXF8F6F4_DTYPES and self.b_dtype in _MXF8F6F4_DTYPES, (
            f"kind::mxf8f6f4 operand dtypes must be one of {_MXF8F6F4_DTYPES}, "
            f"got {self.a_dtype} x {self.b_dtype}"
        )


# Bits to shift an FP4 register byte left by before mma.sync.kind::mxf8f6f4:
# ldsm.b4x16_p64 places the FP4 nibble in the LOW half of its 8-bit container,
# while the MMA reads it from the MIDDLE of the byte (cute::fp4_shift_A/B in
# cutlass/include/cute/atom/mma_traits_sm120.hpp). FP6 and FP8 need no shift.
FP4_SHIFT_BITS = 2


def _subbyte_ldmatrix_atom(dtype):
    """s2r atom for a sub-byte (fp4/fp6) operand of a kind::mxf8f6f4 pair:
    TMA left padded ALIGN8B/ALIGN16B groups in smem, and ldsm.b4x16_p64 /
    b6x16_p32 expands them into one byte lane per element."""
    return cute.make_copy_atom(
        warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4, unpack_bits=dtype.width),
        cutlass.Int8,
    )


def _fp4_shift_block(frag, k_block):
    """Shift one k-block of an FP4 mixed-mode A/B fragment into MMA position
    (see FP4_SHIFT_BITS). Runs after each ldmatrix produce; every block is
    reloaded before its next use, so the shift applies exactly once."""
    v = cute.recast_tensor(frag[None, None, k_block], cutlass.Int8)
    for i in range(cute.size(v)):
        v[i] = cutlass.Int8(v[i] << FP4_SHIFT_BITS)


def _make_sf_copy_block(tiled_mma, sf_dtype, sSFA, sSFB, tCrSFA, tCrSFB, tidx):
    """s2r copies for REAL blockscaled SFA/SFB fragments (universal copy; the
    SF TV layouts are fixed by the block-scaled MMA atom, see
    mma_traits_sm120.hpp). Returns ``copy_sf_block(stage_idx, k_block)`` which
    copies both operands' scale fragments for one k-block of one pipeline
    stage — same produce seam shape as ``canonical_a_load_s2r``."""
    atom_copy_SF = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), sf_dtype)
    perm_mnk = tiled_mma.permutation_mnk
    smem_tiled_copy_SFA = cute.make_tiled_copy(
        atom_copy_SF,
        blackwell_helpers.get_layoutSFA_TV(tiled_mma),
        (cute.size(perm_mnk[0]), cute.size(perm_mnk[2])),
    )
    smem_tiled_copy_SFB = cute.make_tiled_copy(
        atom_copy_SF,
        blackwell_helpers.get_layoutSFB_TV(tiled_mma),
        (cute.size(perm_mnk[1]), cute.size(perm_mnk[2])),
    )
    thr_copy_SFA = smem_tiled_copy_SFA.get_slice(tidx)
    thr_copy_SFB = smem_tiled_copy_SFB.get_slice(tidx)
    # (CPY, CPY_MN, CPY_K, STAGE)
    tCsSFA_copy_view = thr_copy_SFA.partition_S(sSFA)
    tCsSFB_copy_view = thr_copy_SFB.partition_S(sSFB)
    tCrSFA_copy_view = thr_copy_SFA.retile(tCrSFA)
    tCrSFB_copy_view = thr_copy_SFB.retile(tCrSFB)

    def copy_sf_block(stage_idx, k_block):
        # The SF views carry stride-0 broadcast modes (one scale per
        # sf_vec_size elements); filter AFTER slicing the k-block/stage so
        # mode coalescing cannot disturb the indexing — each scale is then
        # copied once.
        cute.copy(
            smem_tiled_copy_SFA,
            cute.filter_zeros(tCsSFA_copy_view[None, None, k_block, stage_idx]),
            cute.filter_zeros(tCrSFA_copy_view[None, None, k_block]),
        )
        cute.copy(
            smem_tiled_copy_SFB,
            cute.filter_zeros(tCsSFB_copy_view[None, None, k_block, stage_idx]),
            cute.filter_zeros(tCrSFB_copy_view[None, None, k_block]),
        )

    return copy_sf_block


def _sf_group_vmk(t, k_atoms):
    """Group a raw SM120 SF fragment (V, rest-MN modes..., K modes...) to
    rank-3 (V, MN, K), walking the K modes off the right until their sizes
    multiply to ``k_atoms`` (= tile_K / 32). Plain-python trace-time helper:
    shapes are static, and it must NOT run under the DSL preprocessor (an
    in-kernel ``while`` is rewritten to dynamic control flow, turning the
    mode index into an Int32 that cute.size rejects)."""
    r = cute.rank(t)
    i, prod = r, 1
    while prod < k_atoms and i > 1:
        i -= 1
        prod *= cute.size(t, mode=[i])
    t = cute.group_modes(t, i, r)
    return cute.group_modes(t, 1, i)


class GemmSm120(GemmSm90):
    """SM120-style GEMM using warp-level MMA instead of WGMMA.

    Key differences from SM90:
    - Uses warp-level MMA (MmaF16BF16Op m16n8k16, or MmaFP8Op m16n8k32 for
      8-bit operands) instead of WGMMA (warp-group, 128 threads)
    - Requires explicit SMEM→RMEM copy via ldmatrix before MMA
    - Thread config: num_mma_warps regular warps + 1 DMA warp
    - Pingpong: 2 warp groups of (2,2,1), each processing alternating tiles
    - fp8 (e4m3/e5m2): k-major A and B only (ldmatrix has no 8-bit
      transpose that matches the fp8 fragment). No slow-accum path: unlike
      Hopper's ~fp13 QGMMA accumulator, SM120's fp8 mma.sync f32 accumulate
      keeps ~21-22 mantissa bits (measured on RTX 5090: +1 onto 2^n survives
      through n=20, one bit short of the bf16 datapath; truncating add), so
      the per-k-tile promotion buys nothing.

    A-operand transforms (quack/operand_transform/) are supported through the
    same ``copy_block(stage_idx, b, k_tile)`` produce seam as GemmSm90's RS
    mainloop — A is always register-sourced here, so value fns / dropout wrap
    the canonical ldmatrix load, and layout-owning W4 decodes replace it.
    W4A8 fast-accum (int4smf) rides the fp8 warp MMA — the block-scaled 2x
    instruction when the tile qualifies; W4A8 promote (int4sm) stays
    SM90-only until this mainloop grows the per-k-tile promote seam.

    Warp roles and pipeline schedule: inherited from GemmSm90 (see its docstring),
    including the quack.pipeline_checks arrive-count validation at construction.
    SM120 deltas to the facts: no thread-block clusters, so the A/B multicast peer
    set is always 1 (ab empty barrier gets one arrive per mma warp, CTA-local) and
    the scheduler barrier routes within a single CTA; the ab-pipeline consumers are
    mma.sync warps fed by ldmatrix rather than WGMMA warpgroups (same per-warp
    release counts via tiled_mma.size).
    """

    arch = 120
    # CUTLASS sm120_builder StagesC policy (sm120_get_tma_dispatch_policy):
    # StagesC = StagesD = min(EpiTiles, 2) — "smaller stage counts in order to
    # fit within the limited shared memory capacity". At 101376 B the SM90
    # base of 4 upfront C stages costs a whole AB stage wherever the C/D
    # footprint is large relative to the smem left over an AB-stage boundary:
    # on the autotune grid that's f32 C or D at 128x128, bf16 C at
    # 128x192/128x64/64x128, and fp4 with f32 C (the ubiquitous
    # bf16-C-into-bf16-D 128x128 case lands on identical picks either way —
    # the (64, 32) epi tile is small enough that the leftover refinement
    # converges to the same fixed point). Measured on RTX 5090 at 8192x8192
    # (settled interleaved medians, 2026-08-01): bf16 128x192coop+C
    # (1,9,5)->(2,2,2) 196->234 TF, bf16 128x128pp+f32 C (1,6,5)->(2,2,3)
    # 201->239 TF, fp8 128x128pp+f32 C 525->637 TF; no-flip controls flat.
    # The leftover refinement still deepens C when smem is actually free.
    # (CUTLASS's ReuseSmemC branch — StagesC = StagesD+1 sharing D's smem —
    # is not implemented here.)
    epi_c_stage_base = 2

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
        split_k: int = 1,
        split_k_mode: int = SplitKMode.SERIAL,
        transform_a: Optional[Callable] = None,
        sf_vec_size: Optional[int] = None,
        # Blackwell cluster-launch-control dynamic persistence (CLC is
        # supported on sm_120a/121a, same as sm_100 — CUTLASS_ARCH_CLC_ENABLED
        # covers the GeForce parts). The scheduler warp doubles as the load
        # warp here, so no throttle barrier is needed (it self-paces).
        use_clc_persistence: bool = False,
        # blockscaled MMA element types when they differ from the storage
        # dtypes (packed fp6 crosses the FFI boundary as raw bytes)
        a_mma_dtype: Optional[Type[cutlass.Numeric]] = None,
        b_mma_dtype: Optional[Type[cutlass.Numeric]] = None,
    ):
        # Don't call super().__init__ — we set up our own config
        self.acc_dtype = acc_dtype
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        self.use_clc_persistence = use_clc_persistence
        if use_clc_persistence:
            # pingpong is fine: it consumes CLC responses one-at-a-time (both
            # WGs read every sched slot — see pingpong_sched_skip in kernel())
            assert is_persistent, "CLC persistence requires the persistent scheduler"
        self.use_pdl = use_pdl
        self.fp8_slow_accum = False
        # Blockscaled (real SFA/SFB operands loaded from gmem): see
        # _setup_tiled_mma for the supported dtype-pair / kind matrix.
        self.sf_vec_size = sf_vec_size
        self.blockscaled = sf_vec_size is not None
        self.sfa_smem_layout_staged = None
        self.sfb_smem_layout_staged = None
        # Mixed-dtype pairs (kind::mxf8f6f4 with independent a/b dtype
        # qualifiers) and same-dtype fp4 (kind::mxf4/mxf4nvf4): resolved in
        # _setup_tiled_mma once b_dtype is known.
        self.use_mxf8f6f4_op = False
        self.use_mxf4_op = False
        self.a_fp4_in_mixed = False
        self.b_fp4_in_mixed = False
        if self.blockscaled:
            assert sf_vec_size in (16, 32), (
                "SM120 blockscaled requires sf_vec_size 32 (MX formats) or 16 (NVFP4)"
            )
            assert not gather_A, "Blockscaled SM120 GEMM does not support gather_A"
            assert transform_a is None, "Blockscaled SM120 GEMM does not support transform_a"
        # The warp-MMA mainloop always consumes A from registers (ldmatrix
        # s2r), so there is no SS/RS mode split; mma_is_rs stays False for the
        # inherited __call__/_setup_attributes checks. A-operand transforms
        # (quack/operand_transform/) plug into the same copy_block seam as
        # SM90's RS mainloop — instantiated below after the register budgets.
        self.mma_is_rs = False
        self._transform_a_factory = transform_a
        if a_mma_dtype is not None or b_mma_dtype is not None:
            assert sf_vec_size is not None, "a/b_mma_dtype are blockscaled-only (packed fp6)"
        self.mma_a_dtype = a_mma_dtype if a_mma_dtype is not None else a_dtype
        self.b_mma_dtype_cfg = b_mma_dtype
        self.gather_A = gather_A
        self.concat_layout = concat_layout or ()
        if self.pingpong:
            assert self.is_persistent, "Pingpong gemm requires persistent scheduler"
        if gather_A:
            assert cluster_shape_mnk[1] == 1
        self._init_split_k(split_k, split_k_mode)

        self.cluster_shape_mnk = cluster_shape_mnk
        assert len(tile_shape_mnk) in [2, 3], "CTA tile shape must be (M, N) or (M, N, K)"
        # K dimension: if user provides 3 values, use their K; otherwise default in _setup_tiled_mma.
        self.cta_tile_shape_mnk = (
            tuple(tile_shape_mnk) if len(tile_shape_mnk) == 3 else (*tile_shape_mnk, 0)
        )
        tile_M, tile_N = self.cta_tile_shape_mnk[:2]
        if self.blockscaled:
            # The SF smem layouts cover whole 128-row/col SF atoms, and the
            # upstream partition_fragment_SFA/SFB helpers only handle whole
            # 128-wide tiles (the CUTLASS SM120 example has the same
            # restriction).
            assert tile_M in (128, 256) and tile_N in (128, 256), (
                f"Blockscaled SM120 GEMM requires tile_M/tile_N in (128, 256), "
                f"got ({tile_M}, {tile_N})"
            )

        # Pingpong: 2 warp groups each with (2,2,1) atom layout
        # Non-pingpong: 1 group of 8 warps with (4,2,1) atom layout.
        # Layout-owning transforms (W4 decodes) get atom_n = 1 instead: with
        # atom_n = 2 the A fragment — and therefore the whole in-register
        # dequant — is duplicated across the N warp pair, and the 32-wide N
        # span forces tile_N >= 32 (2x padded-B traffic at decode shapes).
        # atom_m = 8 when tile_M has whole 128-row steps (8 warps, prefill),
        # else 4 (one 4-warp MMA group, 256-thread CTA, decode tiles).
        self.mma_inst_mnk = (16, 8, 16) if self.mma_a_dtype.width == 16 else (16, 8, 32)
        w4_owned = transform_a is not None and getattr(transform_a, "owned_fmt", None) is not None
        if self.pingpong:
            self.atom_layout_mnk = (2, 2, 1)
        elif w4_owned:
            self.atom_layout_mnk = (8, 1, 1) if tile_M % 128 == 0 else (4, 1, 1)
        else:
            self.atom_layout_mnk = (4, 2, 1)
        if tile_N % (16 * self.atom_layout_mnk[1]) != 0:
            # the N permutation gives each warp 16 consecutive columns: the
            # tiled MMA spans 16 * atom_n N
            raise ValueError(
                f"SM120 CTA tile N must be divisible by {16 * self.atom_layout_mnk[1]}"
            )
        # Consecutive N columns each warp owns in the tiled-MMA permutation.
        # 16 is the default (STSM / gated-epilogue layout). _setup_attributes
        # widens it to 32 when a vec-32 row SFD is active so a whole SF vector
        # sits inside one warp and the cross-warp amax exchange compiles away.
        self.mma_n_warp_run = 16
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

        # TransformA: created after the default register budgets above so it
        # can override them (and occupancy) per its config. The transform may
        # install an aux A-side operand (per-stage strip riding the AB
        # pipeline) — same contract as GemmSm90.
        self.transform_a = None
        self.aux_a = None
        if transform_a is not None:
            self.transform_a = transform_a(self)
            self.aux_a = self.transform_a.aux

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

    def _sf_smem_bytes_per_stage(self) -> int:
        if not self.blockscaled:
            return 0
        tile_m, tile_n, tile_k = self.cta_tile_shape_mnk
        # One 8-bit scale per sf_vec_size K elements, for both SFA and SFB.
        return (tile_m + tile_n) * tile_k // self.sf_vec_size

    def _sfd_row_reqs(self, epilogue_args):
        """(vec_acc, epi_n_min) of the active row-direction SFD codecs —
        the shared scan (quack.epilogue.quantize_out.active_row_sfd_reqs)
        also used by GemmSm100's epi-tile widening."""
        from torch._vendor.quack.epilogue.quantize_out import active_row_sfd_reqs

        return active_row_sfd_reqs(type(self)._epi_ops, epilogue_args)

    def _setup_attributes(self, epilogue_args):
        # Row-direction SFD wider than the 16-column warp run (vec-32
        # mxfp8/mxfp4 quantized D, or a gated postact whose SF vector covers
        # up to 64 acc columns): widen the per-warp N run to 32 so as much of
        # the vector as possible is warp-local (the quantize amax stays a
        # lane butterfly; wider vectors fold their warp_N neighbors through
        # the sExch exchange). EXCEPT when the quantized target is a gated
        # aux output: the halved postact store rides the dummy-MMA STSM
        # retile contract (TileStore._make_tiled_copy_r2s), which encodes the
        # 16-column run — under a widened run its warp N offsets misplace
        # whole 8-column groups (verified empirically: a [0,2,1,3] block
        # permutation of the stored postact). The run stays 16 there and the
        # exchange covers the quantize instead. Decided BEFORE super() so
        # _setup_tiled_mma sees it; _compute_tile_shape_or_override widens
        # the epi tile N to cover both the 32 * atom_n permutation span and
        # whole SF vectors. Every B path follows the permutation:
        # make_tiled_copy_B / the SF fragment TV helpers derive from the
        # tiled MMA (blockscaled verified bitwise vs the 16-run kernel), the
        # hand-built n-major fp8 B ldmatrix is parametrized by the run (see
        # _nmajor_b_tiled_copy), and the plain fp8 path's unit ue8m0
        # fragments are permutation-insensitive constants. tile_n that the
        # 32 * atom_n permutation period does not divide falls back to the
        # sExch exchange, correct for any layout.
        from torch._vendor.quack.epilogue.quantize_out import BlockScaleFactorStore

        ops = getattr(type(self), "_epi_ops", ())
        sfd_vec_acc, sfd_epi_n_min = self._sfd_row_reqs(epilogue_args)
        gated_aux_quant = any(
            isinstance(op, BlockScaleFactorStore)
            and op.direction == "row"
            and op.quant_output != "D"
            and getattr(epilogue_args, op.name, None) is not None
            and next(o for o in ops if o.is_tile_store() and o.name == op.quant_output).gated
            for op in ops
        )
        if (
            sfd_vec_acc > 16
            and not gated_aux_quant
            and self.atom_layout_mnk[1] > 1
            and self.cta_tile_shape_mnk[1] % (32 * self.atom_layout_mnk[1]) == 0
        ):
            self.mma_n_warp_run = 32
        # Epi tile N must cover whole SF vectors (asserted in to_params);
        # consumed by _compute_tile_shape_or_override, only when it divides
        # the CTA tile (otherwise leave the default and let the trace assert
        # report the unsupported tile).
        self._sfd_epi_n_min = (
            sfd_epi_n_min
            if sfd_epi_n_min and self.cta_tile_shape_mnk[1] % sfd_epi_n_min == 0
            else 0
        )
        super()._setup_attributes(epilogue_args)
        if self.blockscaled:
            self.sfa_smem_layout_staged = blockscaled_layout.sm120_make_smem_layout_sfa(
                self.tiled_mma, self.cta_tile_shape_mnk, self.sf_vec_size, self.ab_stage
            )
            self.sfb_smem_layout_staged = blockscaled_layout.sm120_make_smem_layout_sfb(
                self.tiled_mma, self.cta_tile_shape_mnk, self.sf_vec_size, self.ab_stage
            )

    def _setup_tiled_mma(self):
        """Set up warp-level MMA (MmaF16BF16Op / MmaMXF8Op / MmaMXF8F6F4OpFull
        / MmaFP8Op) and tile K.

        Every fp8/fp6/fp4 pair rides the BLOCK-SCALED kind::mxf8f6f4 mma
        whenever it can — with real SF operands when blockscaled, and with
        constant unit (2^0) scale fragments otherwise: the DSL exposes no
        plain kind::f8f6f4 atom (MmaFP8Op is the Ada-era sm_89 opcode, which
        runs at HALF the block-scaled instruction's rate — measured RTX 5090:
        507 vs 1005 TFLOPS dense e4m3, with the identical ~21-22-bit
        truncating f32 accumulator — probed bit-for-bit, same +1-onto-2^n
        keep/lost boundary at n=20/21 and the same RZ signature). The unit-SF
        operand costs one constant byte fragment per (m-atom, k-atom) and no
        loads. A/B dtypes are INDEPENDENT (the CUTLASS C++ SM120_16x8x32_TN
        matrix instantiates all 25 combinations of e4m3/e5m2/e2m3/e3m2/e2m1;
        the upstream DSL op only whitelists fp4 x fp8, relaxed by
        MmaMXF8F6F4OpFull); same-dtype fp8 keeps the dedicated MmaMXF8Op and
        same-dtype fp4 the packed kind::mxf4 / mxf4nvf4 atoms (blockscaled
        only, inst K 64). Constraints for the unit-scale path: f32 accumulator,
        tile_M % 128 == 0 (the SF fragment partition helpers assume whole
        128-row SF blocks), tile_K % 128, and a sm_120/121 COMPILE TARGET —
        kind::mxf8f6f4 has no Hopper equivalent (MmaMXF8Op admits only
        sm_120a/f, sm_121a/f), so the H100 CI proxy legs (QUACK_ARCH=120
        compiled for sm_90a) take the MmaFP8Op fallback, which is sm_89+,
        same-dtype-fp8-only, and numerically stricter there (full fp32 RNE
        accumulate vs SM120's ~21-22-bit RZ). Mixed pairs have NO fallback —
        off-target they raise."""
        # mma_a_dtype, not a_dtype: a layout-owning transform's mA is a
        # storage blob (e.g. uint8) decoded to the MMA compute dtype
        tile_k_resolved = (
            self.cta_tile_shape_mnk[2]
            if self.cta_tile_shape_mnk[2] > 0
            else self.mma_inst_mnk[2] * 4
        )
        mma_arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        if const_expr(self.blockscaled):
            # Real block-scaled operands: unlike the unit-scale fast path
            # below, there is no MmaFP8Op fallback — the SF operands are real
            # data, so every condition is a hard requirement. Same-dtype fp8
            # rides the dedicated MmaMXF8Op; same-dtype fp4 rides the packed
            # kind::mxf4 (e8m0 vec32) / kind::mxf4nvf4 (e4m3 vec16) atoms with
            # inst K 64; every other legal pair (any mix of
            # e4m3/e5m2/e2m3/e3m2/e2m1, plus same-dtype fp6) rides
            # kind::mxf8f6f4 with independent a/b dtypes (MmaMXF8F6F4OpFull).
            assert self.mma_a_dtype in _MXF8F6F4_DTYPES and self.b_dtype in _MXF8F6F4_DTYPES, (
                f"SM120 blockscaled GEMM operand dtypes must be one of "
                f"{_MXF8F6F4_DTYPES}, got {self.mma_a_dtype} x {self.b_dtype}"
            )
            self.use_mxf4_op = self.mma_a_dtype.width == 4 and self.b_dtype.width == 4
            if const_expr(self.use_mxf4_op):
                # Packed nibbles all the way (regular fp4 smem/TMA, no
                # ALIGN8B unpack, no register shift); one instruction covers
                # K 64.
                self.mma_inst_mnk = (16, 8, 64)
                if const_expr(self.sf_vec_size == 16):
                    assert self.sf_dtype == cutlass.Float8E4M3FN, (
                        f"NVFP4 (sf_vec_size=16) requires e4m3 scales, got {self.sf_dtype}"
                    )
                else:
                    assert self.sf_dtype == cutlass.Float8E8M0FNU, (
                        f"MXFP4 (sf_vec_size=32) requires e8m0 scales, got {self.sf_dtype}"
                    )
            else:
                assert self.sf_vec_size == 32 and self.sf_dtype == cutlass.Float8E8M0FNU, (
                    f"SM120 blockscaled (kind::mxf8f6f4) requires e8m0 scales at "
                    f"sf_vec_size 32, got {self.sf_dtype} / {self.sf_vec_size}"
                )
            assert self.acc_dtype == Float32, "SM120 blockscaled GEMM requires f32 accumulation"
            # MN-major operands are supported for fp8 only: both sides ride
            # the byte-granularity transposing ldmatrix (m16n16.trans.b8) —
            # A via make_tiled_copy_A, B via a hand-built TV layout (see
            # _nmajor_b_tiled_copy). Sub-byte operands pack two/more
            # MN-adjacent elements per byte when MN-major, which no ldmatrix
            # variant can transpose.
            assert not self.a_layout.is_m_major_a() or self.mma_a_dtype.width == 8, (
                "SM120 blockscaled GEMM supports M-major A for fp8 only"
            )
            assert not self.b_layout.is_n_major_b() or self.b_dtype.width == 8, (
                "SM120 blockscaled GEMM supports N-major B for fp8 only"
            )
            # one SF atom column covers 4 * sf_vec_size K elements
            assert tile_k_resolved % (4 * self.sf_vec_size) == 0, (
                f"Blockscaled CTA tile K ({tile_k_resolved}) must be divisible by one SF atom "
                f"column (4 * sf_vec_size = {4 * self.sf_vec_size})"
            )
            assert mma_arch in warp.MmaMXF8Op.admissible_archs, (
                "SM120 blockscaled GEMM needs an sm_120a/f or sm_121a/f compile target "
                "(the block_scale mma kinds have no Hopper equivalent)"
            )
            self.use_mxf8_mma = True
        else:
            # Plain (unit-scale) path: the same kind::mxf8f6f4 instruction
            # with constant 2^0 scales — including mixed pairs, which the
            # Ada-era MmaFP8Op cannot express at all. The conditions are the
            # unit-scale SF fragment machinery's requirements.
            pair_in_matrix = (
                self.mma_a_dtype in _MXF8F6F4_DTYPES
                and self.b_dtype in _MXF8F6F4_DTYPES
                and not (self.mma_a_dtype.width == 4 and self.b_dtype.width == 4)
            )
            self.use_mxf8_mma = (
                pair_in_matrix
                and self.acc_dtype == Float32
                and self.cta_tile_shape_mnk[0] % 128 == 0
                # the SF layout is 4-SF (128-k at vec 32) granular
                and tile_k_resolved % 128 == 0
                # ptxas-target gate, not a dispatch gate (see docstring): mirror
                # the op's own __post_init__ admissibility check
                and mma_arch in warp.MmaMXF8Op.admissible_archs
            )
            if const_expr(not self.use_mxf8_mma and self.mma_a_dtype.width < 16):
                # The Ada-instruction fallback (m16n8k32, sm_89+: H100 CI
                # proxy legs, or tiles the SF fragment helpers can't cover)
                # exists for fp8 pairs only — including MIXED e4m3 x e5m2
                # (independent PTX qualifiers, MmaFP8MixedOp); fp6/fp4 pairs
                # have none (the SM89 atom's verifier admits e4m3/e5m2 only).
                assert self.mma_a_dtype.width == 8 and self.b_dtype.width == 8, (
                    f"SM120 GEMM with sub-byte operands ({self.mma_a_dtype} x {self.b_dtype}) "
                    f"requires the kind::mxf8f6f4 path: an sm_120a/f or sm_121a/f compile "
                    f"target, tile_M % 128 == 0, and tile_K % 128 == 0 (no Ada fallback)"
                )
        # Pairs served by kind::mxf8f6f4 with independent a/b dtype qualifiers
        # (genuinely mixed, or same-dtype fp6 which has no dedicated op);
        # same-dtype fp8 stays on MmaMXF8Op, same-dtype fp4 on MXF4/NVF4.
        self.use_mxf8f6f4_op = self.use_mxf8_mma and (
            self.mma_a_dtype != self.b_dtype or self.mma_a_dtype.width == 6
        )
        # Sub-byte sides of a kind::mxf8f6f4 pair are TMA-loaded via the
        # padded tensormaps — 16U4_ALIGN8B for fp4 (8 packed-nibble bytes + 8
        # pad bytes per 16 elements), 16U6_ALIGN16B for fp6 (12 data + 4 pad
        # bytes) — so smem storage and TMA-internal dtype are byte-domain
        # (Int8); ldsm.b4x16_p64 / b6x16_p32 expands into byte lanes at s2r
        # time. Only fp4 additionally needs the << 2 register shift (see
        # FP4_SHIFT_BITS). Same-dtype fp4 (kind::mxf4/mxf4nvf4) stays PACKED
        # throughout — regular fp4 smem, plain 16-bit ldmatrix, no shift.
        self.a_fp4_in_mixed = self.use_mxf8f6f4_op and self.mma_a_dtype.width == 4
        self.b_fp4_in_mixed = self.use_mxf8f6f4_op and self.b_dtype.width == 4
        if const_expr(self.use_mxf8f6f4_op):
            if const_expr(self.mma_a_dtype.width < 8):
                self.a_smem_dtype = cutlass.Int8
                self.a_tma_internal_dtype = cutlass.Int8
            if const_expr(self.b_dtype.width < 8):
                self.b_smem_dtype = cutlass.Int8
                self.b_tma_internal_dtype = cutlass.Int8
        if const_expr(self.mma_a_dtype.width == 16):
            op = warp.MmaF16BF16Op(self.mma_a_dtype, self.acc_dtype, self.mma_inst_mnk)
        elif const_expr(self.use_mxf4_op):
            if const_expr(self.sf_vec_size == 16):
                op = warp.MmaMXF4NVF4Op(self.mma_a_dtype, self.acc_dtype, cutlass.Float8E4M3FN)
            else:
                op = warp.MmaMXF4Op(self.mma_a_dtype, self.acc_dtype, cutlass.Float8E8M0FNU)
        elif const_expr(self.use_mxf8f6f4_op):
            op = MmaMXF8F6F4OpFull(
                self.mma_a_dtype, self.b_dtype, self.acc_dtype, cutlass.Float8E8M0FNU
            )
        elif const_expr(self.use_mxf8_mma):
            op = warp.MmaMXF8Op(self.mma_a_dtype, self.acc_dtype, cutlass.Float8E8M0FNU)
        elif const_expr(self.mma_a_dtype != self.b_dtype):
            # mixed-fp8 Ada fallback (independent .e4m3/.e5m2 qualifiers)
            op = MmaFP8MixedOp(
                self.mma_a_dtype, self.acc_dtype, self.mma_inst_mnk, b_dtype=self.b_dtype
            )
        else:
            op = warp.MmaFP8Op(self.mma_a_dtype, self.acc_dtype, self.mma_inst_mnk)
        tC = cute.make_layout(self.atom_layout_mnk)
        atom_m, atom_n, atom_k = self.atom_layout_mnk
        # Each warp owns mma_n_warp_run consecutive N columns: 16 by default
        # (for STSM and for gated epilogue), 32 when a vec-32 row SFD wants
        # whole SF vectors warp-local (see _setup_attributes).
        permutation_n = cute.make_ordered_layout(
            (self.mma_inst_mnk[1], atom_n, self.mma_n_warp_run // self.mma_inst_mnk[1]),
            order=(0, 2, 1),
        )
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
        if self.transform_a is not None and self.transform_a.tile_k is not None:
            assert tile_k == self.transform_a.tile_k, (
                f"transform_a requires tile_K == {self.transform_a.tile_k}, got {tile_k}"
            )
        self.cta_tile_shape_mnk = (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], tile_k)

    def canonical_a_load(self, tiled_mma, sA, tidx, tCrA):
        """The canonical A produce for the warp-MMA mainloop: the same
        ldmatrix seam as SM90 RS (identical fragment atoms), with the smem
        major passed explicitly — the warp MMA ops carry no major mode
        (operand layout is fixed K-major; the smem major only picks LDSM vs
        LDSM.T). fp8 A rides the same 16-bit LDSM atom typed at the fp8
        element (a k-major byte pair is one 16-bit unit; the m16n8k32 fp8
        fragment is the m16n8k16 16-bit fragment at twice the k density), so
        it is k-major only — enforced by the inherited __call__ checks."""
        atom = None
        position_independent = True
        if const_expr(self.use_mxf8f6f4_op and self.mma_a_dtype.width < 8):
            # Sub-byte side of a kind::mxf8f6f4 pair: smem holds padded
            # ALIGN8B/ALIGN16B groups; ldsm.b4x16_p64 / b6x16_p32 expands into
            # byte lanes (k-major only, no transposing variant). Plain
            # partition — the padded source addressing lives in the atom's
            # value layout.
            atom = _subbyte_ldmatrix_atom(self.mma_a_dtype)
            position_independent = False
        elif const_expr(self.mma_a_dtype.width == 8 and self.a_layout.is_m_major_a()):
            # M-major fp8: the byte-granularity transposing ldmatrix
            # (m16n16.trans.b8, sm_100a/sm_120a)
            atom = cute.make_copy_atom(
                warp.LdMatrix16x16x8bOp(transpose=True, num_matrices=2), self.mma_a_dtype
            )
            position_independent = False
        elif const_expr(self.mma_a_dtype.width < 16):
            # K-major fp8, and the packed same-dtype fp4 kinds (a k-major
            # nibble quartet is one 16-bit unit)
            atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.mma_a_dtype
            )
        return quack_sm90_utils.canonical_a_load_s2r(
            tiled_mma,
            sA,
            tidx,
            tCrA,
            position_independent=position_independent,
            transpose=self.a_layout.is_m_major_a(),
            atom=atom,
        )

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
        # blockscaled SFA/SFB operands (real scale factors riding the AB
        # pipeline; None unless self.blockscaled)
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
        # aux A-side operand slots (e.g. a transform's scale-factor strip
        # riding the AB pipeline, or a raw dropout seed tensor)
        tma_atom_aux_a: Optional[cute.CopyAtom],
        mAuxA_mkl: Optional[cute.Tensor],
        aux_a_smem_layout: Optional[cute.Layout],
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
    ):
        from cutlass.cute.experimental import iket

        varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
        varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
        if const_expr(self.gather_A):
            assert varlen_m or varlen_k
        has_D = const_expr(mD_mnl is not None)
        has_C = const_expr(mC_mnl is not None)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch TMA descriptors
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (
                tma_atom_a,
                tma_atom_b,
                tma_atom_sfa,
                tma_atom_sfb,
                tma_atom_d,
                tma_atom_c,
                tma_atom_aux_a,
            ):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
        )
        epi_pipeline = None
        has_epi_load = const_expr(self.epi_c_stage > 0)
        if const_expr(has_epi_load):
            epi_pipeline = self.make_epi_pipeline(tx_count=self.epi_load_bytes_per_stage)
        sched_pipeline = None
        sched_data = None
        # Pingpong sched-slot SKIP mode: each math WG consumes only its own
        # alternate sched slots (advance_count=2), and the producer hand-writes
        # one extra invalid record after its loop for the trailing WG. That
        # hand-off is STATIC-scheduler-only: under CLC the slots hold hardware
        # CLC responses (a hand-written 4-int record would be misdecoded, and
        # the trailing WG's tail slot has no producer), so CLC pingpong
        # consumes work tiles one at a time instead — both WGs read every
        # response, exactly like the varlen_k / split-k pingpong modes.
        pingpong_sched_skip = const_expr(
            self.pingpong and not varlen_k and self.split_k == 1 and not self.use_clc_persistence
        )
        if const_expr(self.is_persistent):
            sched_pipeline = self.make_sched_pipeline(
                cluster_layout_mnk,
                # one-at-a-time consumption whenever pingpong is NOT in skip
                # mode (the flag name is historical: varlen_k was the first
                # such mode)
                varlen_k=varlen_k
                or self.split_k > 1
                or (self.pingpong and not pingpong_sched_skip),
            )
            # Keep scheduler scratch out of SharedStorage. A small buffer before
            # the 1024-byte aligned epilogue tensors can add a 1 KiB pad; CLC
            # responses also use i128 copies, so this stays 16-byte aligned.
            # CLC needs a +6 Int32 tail after the response ring: the retirement
            # drain's private response slot (16 B) + mbarrier (8 B) — see
            # TileScheduler.cancel_pending_tail (same layout as gemm_sm100).
            sched_smem_flat = smem.allocate_tensor(
                Int32,
                cute.make_layout(
                    4 * self.sched_stage + (6 if const_expr(self.use_clc_persistence) else 0)
                ),
                byte_alignment=16,
                partition=SmemPartition.RESERVED,
            )
            sched_data = cute.make_tensor(
                sched_smem_flat.iterator, cute.make_layout((4, self.sched_stage))
            )

        # Cluster sync
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)

        # SMEM tensors
        a_owned = const_expr(self.transform_a is not None and self.transform_a.owns_a_layout)
        if const_expr(not a_owned):
            sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        else:
            # TMA-facing staged blob view (plain layout, no swizzle); the
            # transform's per-thread math view recasts the same bytes inside
            # make_copy_block
            sA = storage.sA.get_tensor(a_smem_layout)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sSFA, sSFB = None, None
        if const_expr(self.blockscaled):
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout)
            sSFB = storage.sSFB.get_tensor(sfb_smem_layout)
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
            # Only used if not varlen_m; a layout-owning transform's mA is a
            # storage blob, so kernel-M comes from D instead.
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

        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        if warp_idx >= self.ab_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if (
                warp_idx >= self.ab_load_warp_id
                and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
            ):
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
                    # been pushed into local HBM by the owner rank (see
                    # gemm_sm90.py — same shared-code gate).
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
                    copy_SFA, copy_SFB = None, None
                    if const_expr(self.blockscaled):
                        # (bM, bK, RestK). offset_batch_SFA lands on this batch's
                        # tile-aligned region of the padded SF buffer for
                        # varlen_m, and is a plain batch slice otherwise.
                        gSFA_mk = cute.local_tile(
                            varlen_manager.offset_batch_SFA(mSFA_mkl, batch_idx),
                            cute.select(self.cta_tile_shape_mnk, [0, 2]),
                            (tile_coord_mnkl[0], None),
                        )
                        copy_SFA = copy_utils.tma_get_block_copy_fn(
                            tma_atom_sfa,
                            src_tensor=gSFA_mk,
                            dst_tensor=sSFA,
                            tma_multicast=a_tma_multicast,
                        )
                        # (bN, bK, RestK). SFB is K-padded for varlen_k (same
                        # tile-offset formula as SFA); per-batch otherwise.
                        if const_expr(varlen_k):
                            mSFB_nk = varlen_manager.offset_batch_SFA(mSFB_nkl, batch_idx)
                        else:
                            mSFB_nk = mSFB_nkl[None, None, batch_idx]
                        gSFB_nk = cute.local_tile(
                            mSFB_nk,
                            cute.select(self.cta_tile_shape_mnk, [1, 2]),
                            (tile_coord_mnkl[1], None),
                        )
                        copy_SFB = copy_utils.tma_get_block_copy_fn(
                            tma_atom_sfb,
                            src_tensor=gSFB_nk,
                            dst_tensor=sSFB,
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
                            [copy_A, copy_B, copy_AuxA, copy_SFA, copy_SFB],
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
                if const_expr(pingpong_sched_skip):
                    # Need to write the tile_idx to smem for the next WG in the pingpong mode
                    if is_scheduler_warp:
                        tile_scheduler.write_work_tile_to_smem(work_tile)
                    work_tile = tile_scheduler.get_current_work()
                ab_pipeline.producer_tail(ab_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()
                    if const_expr(self.use_clc_persistence):
                        # Serial-observed drain of the pending padding tail
                        # (see TileScheduler.cancel_pending_tail).
                        tile_scheduler.cancel_pending_tail()

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

            # ldmatrix copy atom for SMEM → RMEM (B side; A goes through the
            # copy_block seam below)
            if const_expr(
                self.blockscaled and self.b_dtype.width == 8 and self.b_layout.is_n_major_b()
            ):
                # n-major fp8 B: transposing b8 ldmatrix with a hand-built TV
                # layout (make_tiled_copy_B's auto-derived pairing fetches the
                # wrong k — see _nmajor_b_tiled_copy)
                smem_tiled_copy_B = self._nmajor_b_tiled_copy()
            else:
                if const_expr(self.use_mxf8f6f4_op and self.b_dtype.width < 8):
                    # sub-byte B of a kind::mxf8f6f4 pair: padded ALIGN smem +
                    # unpacking ldmatrix (see canonical_a_load; k-major only).
                    # (Same-dtype fp4 kinds keep packed smem and take the plain
                    # 16-bit ldmatrix below.)
                    atom_copy_ldmatrix_B = _subbyte_ldmatrix_atom(self.b_dtype)
                else:
                    atom_copy_ldmatrix_B = cute.make_copy_atom(
                        warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                        self.b_dtype,
                    )
                smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)

            # Make fragments
            thr_mma = tiled_mma.get_slice(tidx)
            if const_expr(not a_owned):
                acc, tCsA, tCsB, tCrA, tCrB = sm80_utils.partition_fragment_ABC(
                    thr_mma, self.cta_tile_shape_mnk, sA, sB
                )
            else:
                # mA is a storage blob: the A fragment can't be partitioned
                # from sA — build it from the tile shape (the transform's
                # copy_block fills it in fragment order)
                acc = cute.make_rmem_tensor(
                    thr_mma.partition_shape_C(self.cta_tile_shape_mnk[:2]), Float32
                )
                if const_expr(not self.use_mxf8_mma):
                    tCrA = thr_mma.make_fragment_A(
                        thr_mma.partition_shape_A(cute.select(self.cta_tile_shape_mnk, [0, 2]))
                    )
                else:
                    # the block-scaled atom's fragment verifier rejects
                    # shape-built fragments; partition a dummy k-major
                    # (tile_M, tile_K) view instead (only shapes are used —
                    # the pointer is never dereferenced) and fragment that.
                    cA_fake = cute.make_tensor(
                        cute.recast_ptr(sB.iterator, dtype=self.mma_a_dtype),
                        cute.make_ordered_layout(
                            cute.select(self.cta_tile_shape_mnk, [0, 2]), order=(1, 0)
                        ),
                    )
                    tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(cA_fake))
                tCsB = thr_mma.partition_B(sB)
                tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])

            # Block-scaled fp8 (MmaMXF8Op). Two flavors:
            # - self.blockscaled: REAL scale factors — SF fragments partitioned
            #   from the staged smem SF tensors and copied s2r per k-block
            #   alongside A/B (universal copy; the SF TV layouts are fixed by
            #   the block-scaled MMA atom, see mma_traits_sm120.hpp).
            # - plain fp8 fast path: constant unit-scale fragments (see below).
            tCrSFA, tCrSFB = None, None
            copy_sf_block = None
            if const_expr(self.blockscaled):
                tCrSFA = blackwell_helpers.partition_fragment_SFA(
                    sSFA[None, None, 0], thr_mma, tidx
                )
                tCrSFB = blackwell_helpers.partition_fragment_SFB(
                    sSFB[None, None, 0], thr_mma, tidx
                )
                # Normalize to (V, MN, K) — same rest-mode walk as the
                # unit-scale path below. K atoms = k-blocks per tile (one SF
                # fragment slice per mma issue: inst K 32 for fp8/mixed, 64
                # for the packed fp4 kinds).
                k_atoms = self.cta_tile_shape_mnk[2] // self.mma_inst_mnk[2]
                tCrSFA = _sf_group_vmk(tCrSFA, k_atoms)
                tCrSFB = _sf_group_vmk(tCrSFB, k_atoms)
                copy_sf_block = _make_sf_copy_block(
                    tiled_mma, self.sf_dtype, sSFA, sSFB, tCrSFA, tCrSFB, tidx
                )
            elif const_expr(self.use_mxf8_mma):
                # Constant unit-scale fragments: the partition helpers only
                # consume layout shapes, so a dummy tensor over any pointer
                # serves; the fragments are filled with ue8m0 1.0 (0x7F) once
                # and never reloaded — the block-scaled instruction is purely a
                # 2x-rate fp8 mma here.
                sfa_layout = blockscaled_layout.sm120_make_smem_layout_sfa(
                    tiled_mma, self.cta_tile_shape_mnk, 32, 1
                )
                # the SF blob is 128-N granular (SFB layouts assert it); for
                # tile_N < 128 CUTLASS bumps the SFB tile and broadcast-
                # slices — with unit scales any N-slice is valid, so bump
                # here and restrict the fragment's N mode to MMA_N below
                sfb_tile = (
                    self.cta_tile_shape_mnk[0],
                    max(self.cta_tile_shape_mnk[1], 128),
                    self.cta_tile_shape_mnk[2],
                )
                sfb_layout = blockscaled_layout.sm120_make_smem_layout_sfb(
                    tiled_mma, sfb_tile, 32, 1
                )
                sf_ptr = cute.recast_ptr(sB.iterator, dtype=cutlass.Float8E8M0FNU)
                sSFA_like = cute.make_tensor(sf_ptr, sfa_layout)
                sSFB_like = cute.make_tensor(sf_ptr, sfb_layout)
                tCrSFA = blackwell_helpers.partition_fragment_SFA(
                    sSFA_like[None, None, 0], thr_mma, tidx
                )
                tCrSFB = blackwell_helpers.partition_fragment_SFB(
                    sSFB_like[None, None, 0], thr_mma, tidx
                )

                # Normalize to (V, MN, K): the raw fragments grow extra
                # rest-M/N modes past 128 rows (e.g. tile_N=256 -> rank 4
                # with a size-2 block mode BEFORE K) — grouping blindly from
                # mode 2 folds that block mode into K and scrambles the
                # per-k-block slices (read-out-of-fragment ue8m0 bytes decode
                # to NaN scales). Walk K off the right by size instead.
                tCrSFA = _sf_group_vmk(tCrSFA, self.cta_tile_shape_mnk[2] // 32)
                tCrSFB = _sf_group_vmk(tCrSFB, self.cta_tile_shape_mnk[2] // 32)
                if const_expr(cute.size(tCrSFB, mode=[1]) != cute.size(tCrB, mode=[1])):
                    # tile_N < 128: the bumped SFB fragment has more N atoms
                    # than the tile; restrict to MMA_N (any slice is valid —
                    # every SF byte is the same unit scale)
                    tCrSFB = cute.composition(
                        tCrSFB,
                        (None, cute.make_layout(cute.size(tCrB, mode=[1])), None),
                    )
                cute.recast_tensor(tCrSFA, cutlass.Int8).fill(127)
                cute.recast_tensor(tCrSFB, cutlass.Int8).fill(127)

            # A produce seam: the canonical ldmatrix s2r load, or a
            # transform's own produce (e.g. blob LDS + dequant) — same
            # copy_block(stage_idx, b, k_tile) contract as GemmSm90's RS
            # mainloop.
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
                    if const_expr(pingpong_sched_skip):
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        # varlen_k and split_k > 1 both make the per-tile k-tile count
                        # dynamic (CLC pingpong also lands here: one-at-a-time slot
                        # consumption)
                        batch_idx_pp, split_idx_pp = (
                            work_tile.tile_idx[3],
                            work_tile.tile_idx[2],
                        )
                        if not work_tile.is_valid_tile:
                            # padding tile: the counts below are unused (the
                            # validity guard skips the advances), but the
                            # cu_seqlens read must stay in bounds
                            batch_idx_pp = Int32(0)
                        len_k = varlen_manager.len_k(batch_idx=batch_idx_pp)
                        k_tile_total = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                        _, k_tile_cnt = tile_scheduler.get_split_k_tile_range(
                            k_tile_total, split_idx_pp
                        )
                        # Under split-K, only finalizer tiles run the epilogue (and thus
                        # produce/consume C stages); the peer advance must match.
                        c_cnt = Int32(c_tile_cnt)
                        if const_expr(
                            self.split_k > 1 and self.split_k_mode != SplitKMode.SEPARATE
                        ):
                            if split_idx_pp != self.split_k - 1:
                                c_cnt = Int32(0)
                        # PADDING clusters must skip the bootstrap entirely: a
                        # CLC grid may exceed the tile count (varlen_m sizes it
                        # before cu_seqlens is known), and a padding cluster's
                        # producer never issues a CLC query — a ring read here
                        # would wait forever. (Static grids never exceed the
                        # tile count, so this only bites under CLC.)
                        if work_tile.is_valid_tile:
                            ab_read_state.advance_iters(k_tile_cnt)
                            epi_read_state.advance_iters(c_cnt)
                            epi_producer_state.advance_iters(c_cnt)
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
                if const_expr(self.transform_a is not None):
                    if const_expr(self.transform_a.uses_work_tile):
                        # per-work-tile register state (e.g. dropout's per-row
                        # RNG coordinates); every copy_block until the next
                        # hook — incl. the slot-0 preloads — is this tile's
                        self.transform_a.on_work_tile(tile_coord_mnkl)
                acc.fill(0.0)
                sf_valid_insts_last_tile = None
                if const_expr(self.blockscaled and varlen_k):
                    # MMA instructions covering valid K in the globally LAST
                    # k-tile (the mma loop skips the rest — arbitrary SF pad,
                    # see mma(); same scheme and name as GemmSm100.mma).
                    # Splits not covering the last k-tile see a full tile.
                    k_valid = len_k - (k_tile_total - 1) * self.cta_tile_shape_mnk[2]
                    sf_valid_insts_last_tile = cute.ceil_div(k_valid, self.mma_inst_mnk[2])
                    if (
                        const_expr(self.split_k > 1)
                        and k_tile_start_mma + k_tile_cnt != k_tile_total
                    ):
                        sf_valid_insts_last_tile = Int32(
                            self.cta_tile_shape_mnk[2] // self.mma_inst_mnk[2]
                        )
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                iket.range_push("mma")
                ab_read_state = self.mma(
                    ab_pipeline,
                    ab_read_state,
                    tiled_mma,
                    acc,
                    k_tile_cnt,
                    copy_block,
                    smem_tiled_copy_B,
                    tCsB_copy_view,
                    tCrA,
                    tCrB,
                    k_tile_start=k_tile_start_mma,
                    tCrSFA=tCrSFA,
                    tCrSFB=tCrSFB,
                    copy_sf_block=copy_sf_block,
                    sf_valid_insts_last_tile=sf_valid_insts_last_tile,
                )
                if const_expr(self.pingpong):
                    # Cue for next WG's MMA to start
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
                iket.range_pop()

                # ============================================================
                # EPILOGUE — reuse SM90's epilogue flow
                # ============================================================
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
                    if const_expr(pingpong_sched_skip):
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        # one-at-a-time (varlen_k / split-k / CLC): read and
                        # discard the peer WG's slot, then take the next
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

            # Wait for D store complete
            if const_expr(not self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()

    def _nmajor_b_tiled_copy(self):
        """Tiled copy for n-major fp8 B: ldmatrix.m16n16.x2.trans.b8 with a
        hand-built TV layout, so partition_S / cute.copy work exactly like
        every other B copy (only retile is replaced — see _retile_b).

        The DSL atom's trait describes the SOURCE correctly (lane l provides
        the 16-byte n-row at k = l) but its DST value layout is provably wrong
        vs hardware — 12 of 16 slots per lane; byte-level repro in
        AI/cute_dsl_ldmatrix16x16x8b_trans_bug_report.md. Compositions that
        consult the broken Dst therefore mis-place bytes unless the error
        happens to cancel: ``make_tiled_copy_A`` for m-major A does cancel
        (verified), ``make_tiled_copy_B`` does not (it fetches the wrong k).
        This construction never consults the Dst: partition_S composes only
        the (correct) Src side, and _retile_b's fragment regroup encodes the
        MEASURED delivery (per the C++ SM100_U8x16_LDSM_T trait, cute/atom/
        copy_traits_sm100.hpp): lane (a, b) = (l%4, l//4) receives bytes
        (kb, np, h) at n = wn*n_run + b + 8*np, k = 4a + kb + 16h (h = the
        x2-matrix mode = the k16 half). M-warps duplicate (stride 0). A
        widened warp run (mma_n_warp_run 32, vec-32 SFD) issues the atom
        n_run/16 times per warp, each repetition 16 columns further along N.
        If a DSL upgrade fixes the trait, the bit-exact n-major-vs-k-major
        tests (test_sm120_b_n_major*) will catch any placement change."""
        atom = cute.make_copy_atom(
            warp.LdMatrix16x16x8bOp(transpose=True, num_matrices=2), self.b_dtype
        )
        atom_m, atom_n, _ = self.atom_layout_mnk
        n_run = self.mma_n_warp_run  # consecutive N per warp (16, or 32 for SFD)
        n_span = n_run * atom_n  # the tiled-mma N span
        if n_run == 16:
            layout_tv = cute.make_layout(
                ((4, 8, atom_m, atom_n), (4, 2, 2)),
                stride=((4 * n_span, 1, 0, 16), (n_span, 8, 16 * n_span)),
            )
        else:
            # Two atom invocations per warp along N, 16 columns apart; the
            # value modes stay atom-major so each invocation's 16 bytes are
            # contiguous in mode order.
            layout_tv = cute.make_layout(
                ((4, 8, atom_m, atom_n), ((4, 2, 2), n_run // 16)),
                stride=((4 * n_span, 1, 0, n_run), ((n_span, 8, 16 * n_span), 16)),
            )
        return cute.make_tiled_copy(atom, layout_tv, (n_span, 32))

    def _retile_b(self, smem_tiled_copy_B, tCrB):
        """``smem_tiled_copy_B.retile(tCrB)``, except for the hand-built
        n-major fp8 B copy: the DSL retile consults the atom trait's Dst value
        layout, which is wrong vs hardware (see _nmajor_b_tiled_copy) — with
        the custom TV it pairs the copy's 16-value mode with (fragment-V x
        next-K-BLOCK) instead of (fragment-V x n-pair). Regroup the fragment
        by its own modes instead, in the MEASURED delivery order: (V=(kb,h),
        N=(np,g...), K) -> ((kb, np, h), (g...), K). Strides come from the
        fragment layout itself, so the tile_n 256 N-rest split around K
        carries through untouched."""
        if const_expr(
            not (self.blockscaled and self.b_dtype.width == 8 and self.b_layout.is_n_major_b())
        ):
            return smem_tiled_copy_B.retile(tCrB)
        (kb_s, h_s), n_shp, k_shp = tCrB.layout.shape
        (kb_d, h_d), n_std, k_std = tCrB.layout.stride
        np_size, np_d = n_shp[0], n_std[0]
        assert np_size == self.mma_n_warp_run // 8, (
            f"n-major fp8 B fragment np mode ({np_size}) != warp run {self.mma_n_warp_run} / 8"
        )
        if const_expr(self.mma_n_warp_run == 16):
            # profile ((V, rest_v), N, K), congruent with partition_S's
            # ((16,1), N, K)
            layout = cute.make_layout(
                (((kb_s, np_size, h_s), 1), n_shp[1:] if len(n_shp) > 1 else (1,), k_shp),
                stride=(((kb_d, np_d, h_d), 0), n_std[1:] if len(n_std) > 1 else (0,), k_std),
            )
        else:
            # Widened run: congruent with partition_S's ((16, n_run/16), N, K).
            # V = one atom invocation's (kb, np-pair, h) bytes; rest_v = the
            # per-warp atom repetitions along N — np splits (in-atom pair,
            # repetition) with strides (np_d, 2 * np_d).
            layout = cute.make_layout(
                (((kb_s, 2, h_s), np_size // 2), n_shp[1:] if len(n_shp) > 1 else (1,), k_shp),
                stride=(
                    ((kb_d, np_d, h_d), 2 * np_d),
                    n_std[1:] if len(n_std) > 1 else (0,),
                    k_std,
                ),
            )
        return cute.make_tensor(tCrB.iterator, layout)

    @cute.jit
    def mma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        k_tile_cnt: Int32,
        copy_block: Callable,
        smem_tiled_copy_B: cute.TiledCopy,
        tCsB_copy_view: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        k_tile_start: Int32 = 0,
        tCrSFA: Optional[cute.Tensor] = None,
        tCrSFB: Optional[cute.Tensor] = None,
        copy_sf_block: Optional[Callable] = None,
        sf_valid_insts_last_tile: Optional[Int32] = None,
    ) -> cutlass.pipeline.PipelineState:
        """Warp-level MMA mainloop: A produced per k16 block through the
        ``copy_block(stage_idx, b, k_tile)`` seam (canonical ldmatrix s2r, or
        a transform's decode; ``k_tile`` is the GLOBAL k-tile index of the
        produced block, split-k correct via ``k_tile_start``), B via
        ldmatrix, then warp MMA. Same produce rhythm as CUTLASS's SM120
        collective and GemmSm90.mma_rs_interleaved: produce block k+1 (slot 0
        of the next stage at the tile's last block), then MMA block k — the
        warp-synchronous mma.sync needs none of the WGMMA commit-group/wait
        discipline, so the seam contract is the schedule alone.

        For real blockscaled operands, ``copy_sf_block(stage_idx, k_block)``
        copies the SFA/SFB scale fragments smem->rmem alongside each A/B
        k-block (same stage/slot rhythm).

        ``sf_valid_insts_last_tile`` (blockscaled varlen_k only) is the number
        of MMA instructions covering valid K in the LAST k-tile — the rest are
        skipped so the arbitrary SF pad bytes there (0xFF is e8m0 NaN) never
        poison the accumulator via NaN * 0 against the TMA-zero-filled value
        tail (same scheme as GemmSm100.mma)."""
        tCrB_copy_view = self._retile_b(smem_tiled_copy_B, tCrB)
        load_sB = partial(cute.copy, smem_tiled_copy_B)

        num_k_blocks = cute.size(tCrA, mode=[2])
        kt = Int32(k_tile_start)  # global k-tile index of the tile being consumed
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)

        # Load first k-block
        stage = ab_read_state.index
        tCsB_p = tCsB_copy_view[None, None, None, stage]
        copy_block(stage, 0, kt)
        load_sB(tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])
        if const_expr(self.a_fp4_in_mixed):
            _fp4_shift_block(tCrA, 0)
        if const_expr(self.b_fp4_in_mixed):
            _fp4_shift_block(tCrB, 0)
        if const_expr(copy_sf_block is not None):
            copy_sf_block(stage, 0)

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
                    stage = ab_read_state.index
                    tCsB_p = tCsB_copy_view[None, None, None, stage]
                    ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
                # the wrap load is the NEXT tile's slot-0 preload
                copy_block(stage, k_next, kt + 1 if k == num_k_blocks - 1 else kt)
                load_sB(tCsB_p[None, None, k_next], tCrB_copy_view[None, None, k_next])
                if const_expr(self.a_fp4_in_mixed):
                    _fp4_shift_block(tCrA, k_next)
                if const_expr(self.b_fp4_in_mixed):
                    _fp4_shift_block(tCrB, k_next)
                if const_expr(copy_sf_block is not None):
                    copy_sf_block(stage, k_next)
                if const_expr(tCrSFA is not None):
                    # block-scaled mma: the constant unit-scale SF fragments
                    # ride as list operands (see kernel())
                    cute.gemm(
                        tiled_mma,
                        acc,
                        [tCrA[None, None, k], tCrSFA[None, None, k]],
                        [tCrB[None, None, k], tCrSFB[None, None, k]],
                        acc,
                    )
                else:
                    cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            kt += 1

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
                    copy_block(stage, k_next, kt)
                    load_sB(tCsB_p[None, None, k_next], tCrB_copy_view[None, None, k_next])
                    if const_expr(self.a_fp4_in_mixed):
                        _fp4_shift_block(tCrA, k_next)
                    if const_expr(self.b_fp4_in_mixed):
                        _fp4_shift_block(tCrB, k_next)
                    if const_expr(copy_sf_block is not None):
                        copy_sf_block(stage, k_next)
                if const_expr(tCrSFA is not None):
                    # ragged K (varlen_k): skip the instructions covering the
                    # zero-filled value tail — its SF pad bytes may be
                    # arbitrary (0xFF is e8m0 NaN, and NaN * 0 would poison
                    # the accumulator)
                    if const_expr(sf_valid_insts_last_tile is None) or k < sf_valid_insts_last_tile:
                        cute.gemm(
                            tiled_mma,
                            acc,
                            [tCrA[None, None, k], tCrSFA[None, None, k]],
                            [tCrB[None, None, k], tCrSFB[None, None, k]],
                            acc,
                        )
                else:
                    cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)

        return ab_read_state

    def _compute_tile_shape_or_override(
        self,
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
        # The epilogue tile must cover the tiled MMA's warp spans, or the r2s
        # partition wraps across warps (permuted/duplicated corruption): M is
        # atom_m * 16 rows (hit by the (8,1,1) W4 layout whose span is 128 >
        # the default 64), N is mma_n_warp_run * atom_n columns (64 when the
        # run is widened to 32 for vec-32 SFD; a 32-column subtile then holds
        # a single warp's columns while the copy's N period is 64 — caught as
        # a cute.copy size mismatch in epi_load_acc_subtile). An active row
        # SFD additionally needs the subtile to cover whole SF vectors
        # (_sfd_epi_n_min, up to 64 acc columns for a gated postact vector).
        m_span = atom_layout_mnk[0] * 16
        n_span = max(self.mma_n_warp_run * atom_layout_mnk[1], getattr(self, "_sfd_epi_n_min", 0))
        tile_m = max(math.gcd(64, cute.size(cta_tile_shape_mnk, mode=[0])), m_span)
        tile_n = max(math.gcd(n_perf, cute.size(cta_tile_shape_mnk, mode=[1])), n_span)
        return (tile_m, tile_n)
