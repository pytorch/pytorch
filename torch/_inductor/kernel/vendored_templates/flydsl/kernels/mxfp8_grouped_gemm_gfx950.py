# SPDX-License-Identifier: MIT
#
# RAGGED MXFP8 FORWARD grouped GEMM, 4-wave AGPR body.
#
#     out[group_g] = X[group_g] @ W[g]^T        X(M,K), W(E,N,K) -> (M,N)
#
# The ragged axis is NOT the contraction: K is uniform across experts, so the
# K-walk never learns the groups are ragged. Raggedness is therefore confined
# to the four points below -- the 8-buffer LDS ping-pong, the AGPR-pinned
# scaled MFMA, the cooperative scale load, the interleave plan and the
# hand-counted vmcnt barriers are all independent of it.
#
#   1. `OFFS` is a new kernel argument, and the block -> (group, row tile,
#      col tile) mapping is resolved ON DEVICE from it, rather than from a
#      constexpr M_G.
#   2. `row0 = m_start + r_base` instead of `g * M_G + r_base`.
#   3. The epilogue store masks on `active & (r_ < m_end)`: a row tile may
#      overhang its group, and those rows hold this expert's weights applied to
#      the next group's tokens.
#   4. The compile key drops M_G and M_TOTAL, keeping only (K, N, E, BLOCK_C).
#      Both are routing-dependent, so keying on them recompiles per token
#      count -- invisible under balanced routing, a per-step JIT under real
#      routing.
#
# Host side: the grid is over-provisioned to ceildiv(M, BLOCK_R) + E row tiles
# and the output is zero-initialised, because a masked store no longer covers
# every row.
#
# Two structural consequences of the short contraction (K/128 = 11-16 steps):
#   * The K-loop is fully unrolled at compile time; the loop-carried a0/b0
#     fragments and scale chunk disappear.
#   * Pipeline fill and drain are a much larger fraction of the kernel (2 of 16
#     steps are prologue, 2 are tails), so prologue cost matters here.
#
# The vmcnt accounting is derived, not transcribed: with the loop unrolled the
# issue pattern is not perfectly periodic, so `_VmCounter` tracks issued
# vector-memory ops and each barrier waits for exactly "everything up to the op
# I depend on".

# NOTE: no `from __future__ import annotations` -- fx.struct needs real types.

import dataclasses
import functools
import logging
import os

import torch
from torch._inductor.runtime.flydsl_cache import run_cached_flydsl


# NOTE: no flydsl_compat. `_buffer_ops` implements the descriptor loads/stores
# on rocdl ops that exist in BOTH flydsl 0.2.4 and 0.3.0, so this kernel runs on
# either without the shim that grafted 0.3.0's deleted modules back.


# OCP MX: 32 elements share one E8M0 scale.
SCALE_BLOCK = 32

BLOCK_R = 256  # DEFAULT output rows (tokens) per tile; see pick_block_r
BLOCK_C_DEFAULT = 256  # output cols (N) per tile; 128 also supported, see pick_tile
BLOCK_M = 128  # contraction elements per pipeline step (= BLOCK_K)


def pick_block_r(m_total: int, e: int) -> int:
    """Row tile height, from the AVERAGE group size.

    A row tile is charged for every row it covers: a group of m_avg rows under a
    BLOCK_R tile is charged ceildiv(m_avg, BLOCK_R) * BLOCK_R rows and discards
    the rest at the masked store. A narrower tile removes that overhang but
    costs per-block efficiency -- the MFMA-per-ds_read ratio is NA*NB/(NA+NB),
    2.0 at 4x4 (BLOCK_R=256), 1.33 at 2x4 (128), 0.8 at 1x4 (64) -- so the two
    are scored against each other below.

    AVERAGE, not per-group, and that is forced: the group sizes live on the
    device and this kernel's whole design is not to sync on them. m_total and E
    are host-side shape metadata, so m_total // E costs nothing. Under real MoE
    routing the groups are within ~2x of the mean; a skewed batch gets a tile
    sized for its mean, which is no worse than the fixed 256.

    BLOCK_R=64 computed wrong answers until the `wait_barrier` lgkmcnt fix in
    `mxfp8_gemm_utils`; verified clean since.
    """
    if e <= 0:
        return 256
    m_avg = max(m_total // e, 1)

    # Density against actual overhang, not a threshold on m_avg: at m_avg=256
    # every legal tile has overhang 1.0, so narrowing there gives up the ratio
    # for nothing. Shapes with m_avg >= 384 pick 256 either way.
    best_block_r, best_score = 256, -1.0
    for block_r in (64, 128, 256):
        n_tiles_a = block_r // 64
        # MFMA-per-ds_read ratio at this tile height, against a 4-wide tile.
        density = n_tiles_a * 4 / (n_tiles_a + 4)
        overhang = ceildiv(m_avg, block_r) * block_r / m_avg
        score = density / overhang
        if score > best_score:
            best_block_r, best_score = block_r, score
    return best_block_r


def pick_tile(m_total: int, e: int, n: int) -> tuple:
    """(BLOCK_R, BLOCK_C) for this shape, shrinking both when the grid starves.

    `pick_block_r` answers a local question and does not look at how many
    blocks the resulting tile produces. That is fine everywhere except
    small N, where BLOCK_C=256 leaves ONE column tile and the row dim is the
    only parallelism left: K4096 x N256 tiles to 16 blocks on a 256-CU part.

    Halving a tile dimension doubles the block count, and at a starved grid that
    trade is strongly positive even though the narrower tile is worse per block.
    THE THRESHOLD IS ONE FULL WAVE, and the shrink stops at (128, 128): going
    on to BLOCK_R=64 gains too little to pay for another branch here.
    """
    br = pick_block_r(m_total, e)
    # BLOCK_C is always 256 to start: a tile that overhangs N is not free, but
    # the narrow tile is still slower where it removes the whole overhang,
    # because per K-step a wave issues 4*NA*NB MFMAs against 4*(NA+NB) LDS reads
    # -- ratio 2.0 at 4x4 against 1.33 at 4x2. 128 is taken below only when the
    # grid is starved, where occupancy outweighs that ratio.
    bc = BLOCK_C_DEFAULT
    if e <= 0 or n <= 0:
        return br, bc
    # Row tiles are per-group, so the average group is what the grid sees.
    tiles = max(1, e * ceildiv(max(m_total // e, 1), br)) * ceildiv(n, bc)
    if tiles >= _STARVED_TILES:
        return br, bc
    # Only a tile that is 256 in BOTH dims can be halved: halving one that is
    # already 128 lands on (128, 128). SHRINK ONE STEP AT A TIME, re-checking;
    # the partial-wave shapes gain from the first step. (128,128) and
    # BLOCK_R=64 were banned until the `wait_barrier` lgkmcnt fix, which is what
    # made the N_ACCUMS=4 tiles compute wrong answers.
    tiles_at = lambda r, c: (
        max(1, e * ceildiv(max(m_total // e, 1), r)) * ceildiv(n, c)
    )
    if bc == 256 and n % 128 == 0:
        bc = 128
        if tiles_at(br, bc) >= _STARVED_TILES:
            return br, bc
    if br == 256:
        br = 128
        if tiles_at(br, bc) >= _STARVED_TILES:
            return br, bc
    return br, bc


log = logging.getLogger(__name__)

_STARVED_TILES = 256  # one full wave on MI350X's 256 CUs; see pick_tile

_guard_fallback_warned = set()

# (K, N, E, BLOCK_C, BLOCK_R) whose guarded kernel failed to TRACE. FlyDSL
# traces on first launch, not at compile, so this is discovered at the launch
# site and remembered here -- see `launch_for` / `launch_guarded`.
_guard_broken = set()


def _warn_guard_fallback(K, N, E, BLOCK_C, BLOCK_R, err) -> None:
    """Say once, per shape, that the surplus-slot guard did not compile.

    Loud enough to notice (this shape is now paying for its surplus row-tile
    slots, up to 1.74x) but not per call, and never fatal.
    """
    key = (K, N, E, BLOCK_C, BLOCK_R)
    if key in _guard_fallback_warned:
        return
    _guard_fallback_warned.add(key)
    log.warning(
        "FlyDSL MXFP8 grouped GEMM: surplus-slot guard failed to compile at "
        "K=%d N=%d E=%d BLOCK_C=%d BLOCK_R=%d (%s: %s); falling back to the "
        "unguarded kernel, which is correct but pays for its surplus slots.",
        K,
        N,
        E,
        BLOCK_C,
        BLOCK_R,
        type(err).__name__,
        str(err).splitlines()[0][:120],
    )




import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Vector as Vec

from . import mxfp8_buffer_ops as buffer_ops
from .mxfp8_gemm_utils import (
    compute_global_swizzle,
    G2SLoader,
    make_fp8_buffer_tensor,
    pack_i32x4_i32x8,
    S2RLoader,
    swizzle_128,
    wait_barrier,
)


# Each quadrant's 16 MFMAs are interleaved with the g2s/s2r loads of the NEXT
# fragment so they co-issue in the MFMA execute shadow.


def _mfma_scale_agpr(a, b, sa, sb, acc):
    """fp8 16x16x128 scaled MFMA with the accumulator pinned in AGPR (=a,...,0)
    so the f32x4 accumulates in place and the compiler does not shuffle it
    between AGPR slots. a/b are i32x8 (K=128 fp8); sa/sb are broadcast-i32
    E8M0 scales (one e8m0 replicated to 4 bytes). cbsz:0 blgp:0, no op_sel."""
    asm = "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 cbsz:0 blgp:0"
    return _llvm.inline_asm(
        Vec.make_type(4, fx.Float32),
        [
            arith._to_raw(a),
            arith._to_raw(b),
            arith._to_raw(sa),
            arith._to_raw(sb),
            arith._to_raw(acc),
        ],
        asm,
        "=a,v,v,v,v,0",
        has_side_effects=True,
    )


class _VmCounter:
    """Issue-order bookkeeping for `s_waitcnt vmcnt(n)`.

    vmcnt retires IN ORDER, so "wait until op X has landed" is spelled
    "allow at most (ops issued after X) to be outstanding". Every vector
    memory op the kernel issues is counted here; `mark()` records a
    watermark right after the ops a later barrier will depend on, and
    `wait(mark)` emits the barrier with the exact count.

    Getting this wrong in the unsafe direction is silent: too LARGE a count
    under-waits and reads LDS the copy has not filled yet. Deriving it from
    the issue order removes the chance to mis-transcribe a constant.
    """

    def __init__(self):
        self.n = 0

    def issue(self, k):
        self.n += k
        return self.n

    def mark(self):
        return self.n

    def wait(self, mark):
        wait_barrier(self.n - mark)


def _compile(
    K: int, N: int, E: int, BLOCK_C: int, BLOCK_R: int = BLOCK_R, GUARD: bool = True
):
    """Compile a RAGGED forward grouped GEMM for one (K, N, E) shape.

    Deliberately absent from the key: the token count and every group size.
    Both are runtime. Under real MoE routing they change every step, and a
    shape-keyed compile leaks a JIT compile plus a retained GPU module per
    step per layer; only balanced routing would hide it.
    """
    if K % BLOCK_M != 0:
        raise ValueError(f"K ({K}) must be a multiple of {BLOCK_M}")
    if BLOCK_C not in (128, 256):
        raise ValueError(f"BLOCK_C {BLOCK_C} not supported")
    # 64/128/256 give N_TILES_A of 1/2/4. Below 64 a half is under one MFMA
    # tile (16 rows x 2 wave-rows) and the row structure stops holding.
    if BLOCK_R not in (64, 128, 256):
        raise ValueError(f"BLOCK_R {BLOCK_R} not supported")

    K_ITERS = K // BLOCK_M
    if K_ITERS < 4:
        raise ValueError(
            f"K {K} gives {K_ITERS} steps; need >= 4 (2 prologue + 2 tails)"
        )

    SCALE_I32_ROW = K // 128  # i32 of e8m0 per operand row

    N_TILES_A = BLOCK_R // 4 // 16
    N_TILES_B = BLOCK_C // 4 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    N_LDS_ROUNDS = max(N_TILES_A, N_TILES_B)

    # A dwordx2 scale load needs an even element index. The index is
    # lane*SCALE_I32_ROW + (row_base*SCALE_I32_ROW) + k with k always even
    # (a chunk starts on an even K-step) and no per-group K offset here, so
    # the row stride alone decides. K=2048 -> 16 (paired); K=1408 -> 11 (two
    # dword loads instead; same cache lines, one extra instruction).
    SC_PAIR = SCALE_I32_ROW % 2 == 0
    N_SC_OPS = 4 if SC_PAIR else 8  # scale vm ops per chunk (2 K-steps)
    N_CHUNKS = (K_ITERS + 1) // 2

    LDS_BLOCK_R = BLOCK_R // 2  # each wave-dim half owns BLOCK_R/2 rows
    LDS_BLOCK_C = BLOCK_C // 2

    def _interleave_plan(n_mfma, n_g2s, n_tiles):
        """Where to slot each load into the cluster's MFMA stream.

        Returns three lists indexed by MFMA position: the g2s steps, the
        first-half s2r tiles and the second-half s2r tiles to issue just
        before that MFMA. Generated rather than written out because the
        counts move with the tile -- BLOCK_C=256 is 16 MFMAs against 4 g2s
        steps and 4x2 s2r reads, BLOCK_C=128 is 8 against 4 and 2x2.

        Computed HERE, outside the kernel body, on purpose: FlyDSL's AST
        rewriter turns an `if` inside a nested kernel-body function into a
        separate `__then_N` function that cannot see the enclosing closure,
        so schedule logic has to be plain Python that runs before tracing.
        The kernel body then walks the plan with no branches at all.
        """
        acts = []
        for r in range(max(n_g2s, n_tiles)):
            if r < n_g2s:
                acts.append(("g", r))
            if r < n_tiles:
                acts.append(("s0", r))
                acts.append(("s1", r))
        g_at = [[] for _ in range(n_mfma)]
        s0_at = [[] for _ in range(n_mfma)]
        s1_at = [[] for _ in range(n_mfma)]
        for n_, act in enumerate(acts):
            # Spread the loads evenly across the MFMA stream. The clamp is
            # load-bearing at NA=1 (BLOCK_R=64), where 9 acts share only 4
            # MFMA slots: the last act lands at round(9*4/10) = round(3.6) =
            # 4, one past the end. `round` can exceed the floor whenever the
            # ratio's fraction is >= .5, so the clamp -- not the ratio -- is
            # what guarantees every load keeps a slot. It is a no-op at the
            # 4x4/2x4/4x2 tiles, where the ratio never reaches n_mfma - 0.5.
            pos = min(n_mfma - 1, int(round((n_ + 1) * n_mfma / (len(acts) + 1))))
            {"g": g_at, "s0": s0_at, "s1": s1_at}[act[0]][pos].append(act[1])
        return g_at, s0_at, s1_at

    MFMA_SEQ = [(i, j) for i in range(N_TILES_A) for j in range(N_TILES_B)]
    # Keyed by (g2s steps, s2r tiles): clusters 1 and 4 push A and pull B,
    # clusters 2 and 3 the other way round. Keys collide harmlessly when the
    # tile is square.
    PLANS = {
        (N_TILES_A, N_TILES_B): _interleave_plan(len(MFMA_SEQ), N_TILES_A, N_TILES_B),
        (N_TILES_B, N_TILES_A): _interleave_plan(len(MFMA_SEQ), N_TILES_B, N_TILES_A),
    }
    a_lds_size = LDS_BLOCK_R * BLOCK_M
    b_lds_size = LDS_BLOCK_C * BLOCK_M

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_fwd(
        A: fx.Tensor,
        B: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        OFFS: fx.Tensor,
        n_c_tiles: fx.Int32,
        out_m: fx.Int32,
        out_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

        # ── Block -> (group, row tile, col tile), resolved ON DEVICE ──
        # Groups partition the OUTPUT ROWS here, not the contraction, so
        # unlike the wgrad kernel nothing about the K-walk changes: K is
        # uniform across experts. Only this mapping and the epilogue mask
        # care that the groups are ragged.
        bid = fx.block_idx.x
        slot = ArithValue(bid) // n_c_tiles  # which row tile overall
        c_base = (ArithValue(bid) % n_c_tiles) * fx.Int32(BLOCK_C)

        offs_rsrc = buffer_ops.create_buffer_resource(
            OFFS, max_size=False, num_records_bytes=E * 4
        )
        # Uniform (SGPR) loads: every lane in the block wants the same
        # boundaries, so this is E scalar loads, not E per-lane loads.
        ends = [
            ArithValue(
                buffer_ops.buffer_load(
                    offs_rsrc, fx.Int32(i), vec_width=1, dtype=T.i32, is_scalar=True
                )
            )
            for i in range(E)
        ]
        starts = [ArithValue(fx.Int32(0))] + ends[:-1]

        # Walk the groups accumulating row tiles until the slot lands in one.
        # E is constexpr, so this unrolls to E selects on the SALU against a
        # K-walk of many iterations -- the same O(E) prologue the wgrad
        # kernel pays. `active` is false for slots past the last group's
        # tiles, which predicates those blocks off in the epilogue.
        g = ArithValue(fx.Int32(0))
        r_base = ArithValue(fx.Int32(0))
        m_start = ArithValue(fx.Int32(0))
        m_end = ArithValue(fx.Int32(0))
        active = fx.Int32(0) > fx.Int32(0)  # false
        cum = ArithValue(fx.Int32(0))
        # range_constexpr, not range: the AST rewriter turns a plain `range`
        # into a device-side scf.for, whose induction variable is an
        # ArithValue and cannot index the `ends`/`starts` Python lists. This
        # loop must unroll at trace time.
        for i in range_constexpr(E):
            m_i = ends[i] - starts[i]
            t_i = (m_i + fx.Int32(BLOCK_R - 1)) // fx.Int32(BLOCK_R)
            hit = (slot >= cum) & (slot < cum + t_i)
            g = arith.select(hit, fx.Int32(i), g)
            r_base = arith.select(hit, (slot - cum) * fx.Int32(BLOCK_R), r_base)
            m_start = arith.select(hit, starts[i], m_start)
            m_end = arith.select(hit, ends[i], m_end)
            active = active | hit
            cum = cum + t_i

        # A is offset by the group's first token row, B by the expert's
        # weight plane. Both are plain row offsets at the same stride K --
        # the "dynamic per-group stride" problem the MXFP8-MoE post
        # describes does not arise on this axis.
        row0 = m_start + r_base
        wrow0 = ArithValue(g) * fx.Int32(N) + c_base

        # ── SURPLUS SLOTS EXIT HERE, BEFORE ANY GLOBAL TRAFFIC ──
        # The host cannot know how many row tiles the groups actually need
        # -- that is sum_g ceildiv(m_g, BLOCK_R), and the m_g live on the
        # device -- so it launches the upper bound, ceildiv(M, BLOCK_R) + E.
        # The slack is real: at g8x512x4096x4096 it is 24 slots launched for
        # 16 needed, and since n_blocks = n_slots * n_c that is 384 blocks
        # where 256 would do. On 256 CUs that is the difference between one
        # wave and two, which is why the surplus cost 1.74x there and
        # 1.12-1.68x across the shapes -- far more than its 33% share of the
        # blocks, because it is wave quantization and not just wasted bytes.
        #
        # `active` was already computed for the store mask; consulting it
        # here instead skips the K-walk entirely. It is BLOCK-UNIFORM (it
        # depends only on block_idx and on scalar loads of OFFS), so every
        # thread takes the same branch and the barriers inside stay legal.
        # `GUARD` is a plain Python bool from the compile key, so the tracer
        # resolves this at TRACE time: True leaves `proceed` dynamic and the
        # rewriter builds the scf.if; False makes it a Python True and the
        # body is traced unconditionally, reproducing the pre-guard kernel
        # byte for byte. One copy of the body, two kernels. The fallback
        # exists because the guard does not compile everywhere -- see
        # `cached_launch`.
        proceed = active if GUARD else True
        if proceed:
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
            a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
            b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
            b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1

            lane_id = fx.thread_idx.x % 64
            wave_id = fx.thread_idx.x // 64
            wave_i = wave_id // 2
            wave_j = wave_id % 2

            m_lane = lane_id % fx.Int32(16)
            k_grp = lane_id // fx.Int32(16)
            kshift = k_grp * fx.Int32(8)
            mask_ff = fx.Int32(0xFF)
            bcast = fx.Int32(0x01010101)

            A0_gl = row0 * fx.Int32(K)
            A1_gl = (row0 + fx.Int32(LDS_BLOCK_R)) * fx.Int32(K)
            B0_gl = wrow0 * fx.Int32(K)
            B1_gl = (wrow0 + fx.Int32(LDS_BLOCK_C)) * fx.Int32(K)

            gA = make_fp8_buffer_tensor(A, F8_IR_t)
            gB = make_fp8_buffer_tensor(B, F8_IR_t)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            # BOUND THE SCALE DESCRIPTORS. `max_size=True` sets num_records to
            # 0xFFFFFFFF, so an out-of-range e8m0 read is NOT clamped by the
            # hardware -- it touches unmapped memory and faults. This kernel
            # over-provisions row tiles (`n_slots = ceildiv(M, BLOCK_R) + E`) and
            # relies on `active` to predicate the surplus blocks off in the
            # epilogue, so a scale read issued before that predicate goes past
            # the plane whenever the group boundaries leave a slot mapping past
            # the last group.
            #
            # This is observed, not theoretical: group boundaries that leave a
            # slot mapping past the last group fault with hipErrorIllegalAddress,
            # reliably when every boundary sits on a row-tile boundary and
            # intermittently otherwise.
            #
            # A bounded descriptor returns 0 for an out-of-range read, and 0 as
            # an e8m0 is 2**-127, which underflows the product exactly as the
            # epilogue mask intends.
            #
            # A is (M, K/32) with M a runtime value; B is (E, N, K/32), all
            # constexpr, so its bound folds to a constant.
            sa_bytes = arith.index_cast(
                T.i64, arith.index_cast(T.index, out_m * fx.Int32(K // SCALE_BLOCK))
            )
            sa_rsrc = buffer_ops.create_buffer_resource(
                A_scale, max_size=False, num_records_bytes=sa_bytes
            )
            sb_rsrc = buffer_ops.create_buffer_resource(
                B_scale, max_size=False, num_records_bytes=E * N * (K // SCALE_BLOCK)
            )

            gl_off_a = compute_global_swizzle(
                lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False
            )
            gl_off_b = compute_global_swizzle(
                lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False
            )

            a_g2s = G2SLoader(a_div, gl_off_a, N_TILES_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_TILES_B, F8_IR_t, wave_id)
            a_s2r = S2RLoader(wave_i, N_TILES_A)
            b_s2r = S2RLoader(wave_j, N_TILES_B)

            vm = _VmCounter()

            # ── MXFP8 scales: cooperative wide load + ds_bpermute ──
            # Each operand-half needs the e8m0 byte of 64 consecutive rows (4 MFMA
            # tiles x 16 rows) at this K-step; one i32 per row covers 128 K, of
            # which lane group k_grp takes byte k_grp and broadcasts it to 4 bytes.
            #
            # The naive form -- one buffer_load per (half, tile) -- issues 16 dword
            # loads per K-step that touch the SAME 16 cache lines four times over,
            # because lanes m_lane..m_lane+48 share an address. That is the same L1
            # request count as all of the step's data traffic, and it, not the
            # matrix core, sets the pace: stubbing the scale loads out nearly
            # doubles measured throughput.
            #
            # Instead: ONE cooperative load per half, lane L taking row
            # (half_base + L), so 64 lanes cover 64 rows with no duplicate address,
            # widened to a dwordx2 spanning two K-steps. 8x fewer scale L1
            # requests. Each lane then pulls the tile it actually needs with
            # ds_bpermute (lane L from lane t*16 + L%16) -- LDS crossbar, not HBM.
            #
            # A half spans N_TILES*16 rows, so 64 lanes cover it exactly at
            # N_TILES=4 and cover it twice over at N_TILES=2 (BLOCK_C=128). The
            # duplicate half of that load is wasted address bandwidth but stays
            # ONE instruction, and the rows it touches are the next 32 of the same
            # weight plane -- already-warm cache lines, or an in-bounds read the
            # bpermute never selects.

            sa0_base = row0 + wave_i * fx.Int32(N_TILES_A * 16)
            sa1_base = row0 + fx.Int32(LDS_BLOCK_R) + wave_i * fx.Int32(N_TILES_A * 16)
            sb0_base = wrow0 + wave_j * fx.Int32(N_TILES_B * 16)
            sb1_base = wrow0 + fx.Int32(LDS_BLOCK_C) + wave_j * fx.Int32(N_TILES_B * 16)

            # Only the row term is per-lane; the half base and the K index are
            # wave-uniform and ride the buffer instruction's SGPR soffset, so the
            # address costs no per-step VALU and exactly one VGPR.
            sc_voff = lane_id * fx.Int32(SCALE_I32_ROW)
            sc_sbase = [
                (_rs, _base * fx.Int32(SCALE_I32_ROW * 4))
                for _rs, _base in (
                    (sa_rsrc, sa0_base),
                    (sa_rsrc, sa1_base),
                    (sb_rsrc, sb0_base),
                    (sb_rsrc, sb1_base),
                )
            ]
            sc_perm = [
                fx.Int32(t_ * 64) + m_lane * fx.Int32(4)
                for t_ in range_constexpr(max(N_TILES_A, N_TILES_B))
            ]
            # Tiles per half, in the order the chunk stores them: A0, A1, B0, B1.
            sc_tiles = (N_TILES_A, N_TILES_A, N_TILES_B, N_TILES_B)

            def issue_chunk(k_i32):
                """Cooperative scale loads for K-steps (k_i32, k_i32+1).

                Returns 2 raw i32 per operand-half, flat: [a0_lo, a0_hi, a1_lo, ...].
                A chunk for a K-step past the end reads the next row's scales (or
                out of bounds, which the buffer resource returns as 0) and is never
                consumed -- cheaper than predicating the last prefetch.
                """
                out = []
                for _rs, _sb in sc_sbase:
                    soff = _sb + fx.Int32(k_i32 * 4)
                    if SC_PAIR:
                        v = buffer_ops.buffer_load(
                            _rs, sc_voff, vec_width=2, dtype=T.i32, soffset_bytes=soff
                        )
                        out.append(
                            buffer_ops.vec_extract(
                                v, static_position=[0], dynamic_position=[]
                            )
                        )
                        out.append(
                            buffer_ops.vec_extract(
                                v, static_position=[1], dynamic_position=[]
                            )
                        )
                    else:
                        out.append(
                            buffer_ops.buffer_load(
                                _rs,
                                sc_voff,
                                vec_width=1,
                                dtype=T.i32,
                                soffset_bytes=soff,
                            )
                        )
                        out.append(
                            buffer_ops.buffer_load(
                                _rs,
                                sc_voff,
                                vec_width=1,
                                dtype=T.i32,
                                soffset_bytes=soff + fx.Int32(4),
                            )
                        )
                vm.issue(N_SC_OPS)
                return out

            def use_chunk(chunk, p):
                """Redistribute + consume the chunk's K-step `p` -> (sa0, sa1, sb0, sb1)."""
                groups = []
                for h_ in range_constexpr(4):
                    src = chunk[2 * h_ + p]
                    grp = []
                    for t_ in range_constexpr(sc_tiles[h_]):
                        v = rocdl.ds_bpermute(res=T.i32, index=sc_perm[t_], src=src)
                        grp.append(((ArithValue(v) >> kshift) & mask_ff) * bcast)
                    groups.append(grp)
                return groups

            def mma(a, b, c, sa, sb):
                for i in range_constexpr(N_TILES_A):
                    for j in range_constexpr(N_TILES_B):
                        idx = i * N_TILES_B + j
                        c[idx] = _mfma_scale_agpr(a[i], b[j], sa[i], sb[j], c[idx])
                return c

            zero = Vec.filled(4, 0.0, fx.Float32)
            c00 = [zero] * N_ACCUMS
            c01 = [zero] * N_ACCUMS
            c10 = [zero] * N_ACCUMS
            c11 = [zero] * N_ACCUMS

            def _lds_swizzle(s2r):
                out = []
                for row_off in range_constexpr(s2r.n_tiles):
                    row = (
                        s2r.wave_idx * fx.Int32(s2r.n_tiles * 16)
                        + fx.Int32(row_off * 16)
                        + (lane_id % fx.Int32(16))
                    )
                    swz = []
                    for ii in range_constexpr(2):
                        col = (lane_id // fx.Int32(16)) * fx.Int32(16) + fx.Int32(
                            ii * 64
                        )
                        r_, c_ = swizzle_128(row, col)
                        swz.append(r_ * fx.Int32(BLOCK_M) + c_)
                    out.append(swz)
                return out

            def _cluster(lds_dst, g2s, k_off, s2r, lds_src, a, b, c, sa, sb):
                # Interleave this quadrant's MFMAs with the g2s and s2r loads of
                # the NEXT fragment, so the loads co-issue in the MFMA execute
                # shadow. Mirrors fp8_gemm_4wave's _interleaved_cluster; the MFMA
                # is scaled + AGPR-pinned here. The schedule comes from
                # `_interleave_plan` (computed before tracing), so this walks it
                # with no branches -- see that function on why.
                swz = _lds_swizzle(s2r)
                rt = [None] * s2r.n_tiles
                half = [None] * s2r.n_tiles
                g_at, s0_at, s1_at = PLANS[(g2s.n_load_steps, s2r.n_tiles)]

                for idx in range_constexpr(len(MFMA_SEQ)):
                    for gi in range_constexpr(len(g_at[idx])):
                        g2s.load_one(lds_dst, k_off, g_at[idx][gi])
                    for si in range_constexpr(len(s0_at[idx])):
                        t_ = s0_at[idx][si]
                        half[t_] = s2r.load_one(lds_src, swz[t_][0])
                    for si in range_constexpr(len(s1_at[idx])):
                        t_ = s1_at[idx][si]
                        rt[t_] = pack_i32x4_i32x8(
                            half[t_], s2r.load_one(lds_src, swz[t_][1])
                        )
                    i_, j_ = MFMA_SEQ[idx]
                    c[i_ * N_TILES_B + j_] = _mfma_scale_agpr(
                        a[i_], b[j_], sa[i_], sb[j_], c[i_ * N_TILES_B + j_]
                    )

                vm.issue(g2s.n_load_steps)
                return c, rt

            # ── Prologue: pre-fill the 8-buffer LDS pipeline (2 K-steps) ──
            # Chunk 0 goes first so it is the oldest thing in flight and the
            # prologue barriers retire it for free.
            chunks = [None] * N_CHUNKS
            chunks[0] = issue_chunk(0)

            a_g2s.load(a_cur0, A0_gl + 0 * BLOCK_M)
            mk_ac0 = vm.issue(N_TILES_A)
            b_g2s.load(b_cur0, B0_gl + 0 * BLOCK_M)
            mk_bc0 = vm.issue(N_TILES_B)
            b_g2s.load(b_cur1, B1_gl + 0 * BLOCK_M)
            vm.issue(N_TILES_B)
            a_g2s.load(a_cur1, A1_gl + 0 * BLOCK_M)
            mk_a = vm.issue(N_TILES_A)

            a_g2s.load(a_next0, A0_gl + 1 * BLOCK_M)
            vm.issue(N_TILES_A)
            b_g2s.load(b_next0, B0_gl + 1 * BLOCK_M)
            mk_b = vm.issue(N_TILES_B)
            b_g2s.load(b_next1, B1_gl + 1 * BLOCK_M)
            vm.issue(N_TILES_B)
            a_g2s.load(a_next1, A1_gl + 1 * BLOCK_M)
            mk_c = vm.issue(N_TILES_A)

            vm.wait(mk_ac0)
            a0 = a_s2r.load(a_cur0)
            vm.wait(mk_bc0)
            b0 = b_s2r.load(b_cur0)

            # ── Main K-steps 0 .. K_ITERS-3, fully unrolled ──
            # Each step consumes the LDS buffers holding step k, loads step k+2
            # into them, and reads step k+1's leading fragments. Which watermark
            # each barrier waits on follows from the ping-pong depth:
            #   * clusters 1-2 read the halves written by step k-2's SECOND half,
            #   * clusters 3-4 read the halves written by step k-1's FIRST half.
            # The two prologue groups stand in for steps -2 and -1: `mk_a` closes
            # the k=0 group (step 0's second-half operands), `mk_b` sits right
            # after b_next0 (step 1's leading fragments) and `mk_c` closes the
            # k=1 group (step 1's second-half operands).
            bufs = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)
            sec = {-2: mk_a, -1: mk_c}  # second-half watermark, by step index
            fst = {-1: mk_b}  # first-half watermark, by step index

            for kk in range_constexpr(K_ITERS - 2):
                ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
                k2 = (kk + 2) * BLOCK_M
                p = kk % 2

                # The scales for this step came off HBM a full step-pair ago; the
                # only HBM touch is the prefetch for the pair two K-steps out --
                # the same horizon the data ping-pong runs at.
                vm.wait(sec[kk - 2])
                if p == 0 and (kk // 2) + 1 < N_CHUNKS:
                    chunks[(kk // 2) + 1] = issue_chunk(kk + 2)
                sa0, sa1, sb0, sb1 = use_chunk(chunks[kk // 2], p)

                c00, b1 = _cluster(
                    ac0, a_g2s, A0_gl + k2, b_s2r, bc1, a0, b0, c00, sa0, sb0
                )
                c01, a1 = _cluster(
                    bc0, b_g2s, B0_gl + k2, a_s2r, ac1, a0, b1, c01, sa0, sb1
                )
                fst[kk] = vm.mark()

                vm.wait(fst[kk - 1])
                c10, a0 = _cluster(
                    bc1, b_g2s, B1_gl + k2, a_s2r, an0, a1, b0, c10, sa1, sb0
                )
                c11, b0 = _cluster(
                    ac1, a_g2s, A1_gl + k2, b_s2r, bn0, a1, b1, c11, sa1, sb1
                )
                sec[kk] = vm.mark()

                bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)

            a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1 = bufs

            # ── Tail step K_ITERS-2: drain, no more g2s ──
            k_tail0 = K_ITERS - 2
            vm.wait(sec[k_tail0 - 2])
            sa0, sa1, sb0, sb1 = use_chunk(chunks[k_tail0 // 2], k_tail0 % 2)
            b1 = b_s2r.load(b_cur1)
            c00 = mma(a0, b0, c00, sa0, sb0)
            a1 = a_s2r.load(a_cur1)
            c01 = mma(a0, b1, c01, sa0, sb1)
            vm.wait(fst[k_tail0 - 1])
            a0 = a_s2r.load(a_next0)
            c10 = mma(a1, b0, c10, sa1, sb0)
            b0 = b_s2r.load(b_next0)
            c11 = mma(a1, b1, c11, sa1, sb1)

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

            # ── Tail step K_ITERS-1 ──
            k_tail1 = K_ITERS - 1
            vm.wait(sec[k_tail1 - 2])
            sa0, sa1, sb0, sb1 = use_chunk(chunks[k_tail1 // 2], k_tail1 % 2)
            b1 = b_s2r.load(b_cur1)
            a1 = a_s2r.load(a_cur1)
            c00 = mma(a0, b0, c00, sa0, sb0)
            c01 = mma(a0, b1, c01, sa0, sb1)
            c10 = mma(a1, b0, c10, sa1, sb0)
            c11 = mma(a1, b1, c11, sa1, sb1)

            # ── Epilogue: bf16 into out(M, N); the MFMA already applied the scales ──
            # Storing f32 and converting outside would cost a full extra pass over
            # M*N (553 MB at the e2e forward shape) -- more than half the kernel.
            out_m_i = arith.index_cast(T.index, out_m)
            out_n_i = arith.index_cast(T.index, out_n)
            nbytes = arith.index_cast(T.i64, out_m_i * out_n_i * fx.Index(2))
            o_rsrc = buffer_ops.create_buffer_resource(
                OUT, max_size=False, num_records_bytes=nbytes
            )
            base_row = row0 + wave_i * fx.Int32(N_TILES_A * 16)
            base_col = c_base + wave_j * fx.Int32(N_TILES_B * 16)
            oob = out_m * out_n

            def store_group(frag, br, bc):
                for ti in range_constexpr(N_TILES_A):
                    row = br + fx.Int32(ti * 16) + k_grp * fx.Int32(4)
                    for tj in range_constexpr(N_TILES_B):
                        col = bc + fx.Int32(tj * 16) + m_lane
                        col_ok = col < out_n
                        vec = Vec(frag[ti * N_TILES_B + tj])
                        for e in range_constexpr(4):
                            r_ = row + fx.Int32(e)
                            # `r_ < m_end` is the ragged part. A row tile may
                            # overhang its group, in which case the overhanging
                            # rows hold this expert's weights applied to the
                            # NEXT group's tokens -- correct arithmetic, wrong
                            # expert -- so they must not be stored. `active`
                            # kills whole slots past the last group's tiles.
                            # Reading those A rows is harmless: each output
                            # element is one A row dotted with one B column, so
                            # nothing leaks across rows, and any fp8 NaN in the
                            # pad region past m_total lands only in rows we drop.
                            ok = active & (r_ < m_end) & (r_ < out_m) & col_ok
                            off = arith.select(ok, r_ * out_n + col, oob)
                            buffer_ops.buffer_store(vec[e].to(fx.BFloat16), o_rsrc, off)

            store_group(c00, base_row, base_col)
            store_group(c01, base_row, base_col + fx.Int32(LDS_BLOCK_C))
            store_group(c10, base_row + fx.Int32(LDS_BLOCK_R), base_col)
            store_group(
                c11, base_row + fx.Int32(LDS_BLOCK_R), base_col + fx.Int32(LDS_BLOCK_C)
            )

    @flyc.jit
    def launch_fwd(
        A,
        B,
        OUT,
        A_scale,
        B_scale,
        OFFS,
        n_blocks: fx.Int32,
        n_c_tiles: fx.Int32,
        out_m: fx.Int32,
        out_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_fwd(
            A,
            B,
            OUT,
            A_scale,
            B_scale,
            OFFS,
            n_c_tiles,
            out_m,
            out_n,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(n_blocks, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_fwd


@functools.lru_cache(maxsize=None)
def cached_launch(
    K: int, N: int, E: int, BLOCK_C: int, BLOCK_R: int = 256, GUARD: bool = True
):
    """Build the launcher. GUARD is part of the key -- see `launch_for`.

    NOTE: this does NOT trace. FlyDSL traces lazily, on the first call
    through `fast_launch`, so a tracer error surfaces at LAUNCH time and
    cannot be caught here. That is what `launch_for` is for.
    """
    return _compile(K, N, E, BLOCK_C, BLOCK_R, GUARD=GUARD)


def launch_for(K: int, N: int, E: int, BLOCK_C: int, BLOCK_R: int):
    """The guarded launcher, unless this shape already failed to trace.

    The surplus-slot guard is worth 1.2-1.74x but does not trace on every
    shape: as a bare `if active:` it can raise "cannot evaluate dynamic
    'Boolean' as Python bool", and only once enough other kernels have been
    traced in the same process, so it is not reliably reproducible from a
    single shape.

    The fallback therefore has to live at the LAUNCH site: an untraceable
    shape would otherwise raise RuntimeError out of the op rather than
    degrading to an unguarded launch. GUARD=False is verified bitwise-identical
    to GUARD=True; it costs throughput, nothing else.
    """
    return cached_launch(
        K,
        N,
        E,
        BLOCK_C,
        BLOCK_R,
        GUARD=(K, N, E, BLOCK_C, BLOCK_R) not in _guard_broken,
    )


def launch_guarded(launch, key, compile_args_factory, dispatch_args, param):
    """Run `launch`, and on a tracer error retry once unguarded, forever.

    Compilation goes through Inductor's FlyDSL cache so the compiled dispatcher
    is shared with the rest of the Inductor-generated FlyDSL kernels and is
    keyed the same way (pid, device, constexpr param).
    """
    try:
        return run_cached_flydsl(
            launch,
            constexpr_param=param,
            compiler=flyc.compile,
            dispatch_args=dispatch_args,
            compile_args_factory=compile_args_factory,
        )
    except Exception as e:
        # The guard fallback exists for ONE failure mode: the surplus-slot `if`
        # not tracing on some shapes. An out-of-memory is not that, and treating
        # it as such both misreports the cause and permanently pins this shape
        # to the unguarded kernel (up to 1.74x slower). Re-raise it instead.
        if isinstance(e, torch.OutOfMemoryError) or key in _guard_broken:
            raise
        _guard_broken.add(key)
        _warn_guard_fallback(*key, e)
        return run_cached_flydsl(
            cached_launch(*key, GUARD=False),
            constexpr_param=param.unguarded(),
            compiler=flyc.compile,
            dispatch_args=dispatch_args,
            compile_args_factory=compile_args_factory,
        )


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclasses.dataclass(frozen=True)
class MXFP8GroupedGemmParam:
    """The compile key of one MXFP8 grouped GEMM specialisation.

    Deliberately absent: the token count and every group size. Both are
    runtime values under real MoE routing, and a shape-keyed compile would
    leak a JIT compile plus a retained GPU module per step per layer.

    This is a plain dataclass rather than an ``fx.struct``: the kernel body
    closes over these values at trace time (see ``_compile``) instead of
    reading them from a device-side struct, so FlyDSL never sees the param.
    It carries ``__cache_signature__`` only so ``run_cached_flydsl`` can key
    on it exactly as it keys on the dense ``GemmGfx950Param``.
    """

    k: int
    n: int
    group_count: int
    block_c: int
    block_r: int
    guarded: bool = True

    def key(self) -> tuple[int, int, int, int, int]:
        """The tuple ``cached_launch`` / ``_guard_broken`` are keyed on."""
        return (self.k, self.n, self.group_count, self.block_c, self.block_r)

    def unguarded(self) -> "MXFP8GroupedGemmParam":
        return dataclasses.replace(self, guarded=False)

    def __cache_signature__(self) -> tuple[int, int, int, int, int, bool]:
        return (*self.key(), self.guarded)


def make_mxfp8_grouped_gemm_kernel_name(param: MXFP8GroupedGemmParam) -> str:
    return (
        f"mxfp8_grouped_gemm_k{param.k}_n{param.n}_e{param.group_count}"
        f"_t{param.block_r}x{param.block_c}"
        f"_guard{int(param.guarded)}"
    )


def make_mxfp8_grouped_gemm_param(
    k: int,
    n: int,
    group_count: int,
    block_r: int = BLOCK_R,
    block_c: int = BLOCK_C_DEFAULT,
) -> MXFP8GroupedGemmParam:
    """Validate one (K, N, E, BLOCK_R, BLOCK_C) against what `_compile` accepts.

    Raises ``ValueError`` rather than tripping the asserts inside ``_compile``,
    so the Inductor heuristics can prune a config without catching
    ``AssertionError``.
    """
    if k <= 0 or n <= 0 or group_count <= 0:
        raise ValueError(f"degenerate shape K={k}, N={n}, E={group_count}")
    if k % BLOCK_M != 0:
        raise ValueError(f"K ({k}) must be a multiple of {BLOCK_M}")
    if k // BLOCK_M < 4:
        raise ValueError(
            f"K {k} gives {k // BLOCK_M} pipeline steps; need >= 4 "
            "(2 prologue + 2 tails)"
        )
    if block_c not in (128, 256):
        raise ValueError(f"BLOCK_C {block_c} not supported")
    if block_r not in (64, 128, 256):
        raise ValueError(f"BLOCK_R {block_r} not supported")
    if block_c == 128 and n % 128 != 0:
        # A 128-wide tile is only ever chosen to remove overhang; taking it on
        # an N it does not divide adds overhang instead of removing it.
        raise ValueError(f"BLOCK_C 128 needs N ({n}) to be a multiple of 128")
    return MXFP8GroupedGemmParam(
        k=int(k),
        n=int(n),
        group_count=int(group_count),
        block_c=int(block_c),
        block_r=int(block_r),
    )


def make_mxfp8_grouped_gemm_param_and_validate(
    k: int, n: int, group_count: int, block_r: int, block_c: int
) -> MXFP8GroupedGemmParam | None:
    """`make_mxfp8_grouped_gemm_param`, returning None instead of raising."""
    try:
        return make_mxfp8_grouped_gemm_param(k, n, group_count, block_r, block_c)
    except ValueError:
        return None


def get_mxfp8_grouped_gemm_grid_size(
    param: MXFP8GroupedGemmParam, rows: int
) -> tuple[int, int]:
    """(n_blocks, n_c_tiles) for a launch covering `rows` tokens.

    The row-tile count is over-provisioned to ``ceildiv(rows, BLOCK_R) + E``:
    a group boundary that falls inside a tile costs the next group a slot, and
    E boundaries can each do that. Blocks whose slot lands past the last group
    resolve to ``active == False`` and are predicated off by the surplus-slot
    guard, so the surplus is a launch-descriptor cost, not work.
    """
    n_c_tiles = ceildiv(param.n, param.block_c)
    n_slots = ceildiv(rows, param.block_r) + param.group_count
    return n_slots * n_c_tiles, n_c_tiles


def launch_mxfp8_grouped_gemm_gfx950(
    out,
    mat_a,
    mat_b,
    scale_a,
    scale_b,
    offs,
    param: MXFP8GroupedGemmParam,
    stream,
    tensor_arg=None,
    compile_only=False,
):
    """MXFP8 forward grouped GEMM over RAGGED token groups, into `out`.

        out[group_g] = mat_a[group_g] @ mat_b[g]^T

    Groups partition the token (output row) dim with device-resident, unequal
    sizes. Because the ragged axis is NOT the contraction, the whole pipelined
    K-walk carries over from the even-groups body untouched -- only the
    block -> (group, tile) mapping and the epilogue mask change.

        out       (M, N)      bf16, contiguous, allocated by the caller
        mat_a     (M, K)      fp8 e4m3, row-major
        mat_b     (E, N, K)   fp8 e4m3, row-major (i.e. a [E, K, N] operand
                              whose K dim is the contiguous one)
        scale_a   (M, K//32)  e8m0, contiguous
        scale_b   (E, N, K//32) e8m0, contiguous
        offs      (E,) int32  cumulative group ends along M, ON DEVICE.
                              Read by the block itself -- never synced here.

    ROWS PAST ``offs[-1]`` ARE LEFT ALONE. They are covered by no block, which
    matches what ATen's own 2d-3d grouped GEMMs do with the same tail; the
    alternative is a full M*N memset, measured at 9.5% of the kernel.

    `tensor_arg` wraps each tensor for the *compile* call; Inductor passes
    ``flyc.from_torch_tensor(t).mark_layout_dynamic()`` so one compiled
    dispatcher serves every M.

    `compile_only` warms the disk cache and returns without dispatching. It is
    what Inductor's precompile entry point uses, where the tensors are fake and
    the only thing wanted is the compiled artifact. Only the first window is
    compiled: every window traces the same kernel, and the layout-dynamic
    compile args make the window's row count a runtime slot.
    """
    if tensor_arg is None:

        def tensor_arg(tensor):
            return tensor

    total_m = int(out.shape[0])
    if total_m == 0:
        return out

    a = mat_a.view(torch.int8)
    a_sc = scale_a.view(torch.uint8)
    b = mat_b.view(torch.int8).view(-1)
    b_sc = scale_b.view(torch.uint8).view(-1)
    offs_i32 = offs.view(torch.int32)

    key = param.key()
    for row_start, rows, offs_window in _row_windows(
        total_m, param.k, param.n, offs_i32, param.block_r
    ):
        n_blocks, n_c_tiles = get_mxfp8_grouped_gemm_grid_size(param, rows)
        dispatch_args = (
            a.narrow(0, row_start, rows).view(-1),
            b,
            out.narrow(0, row_start, rows).view(-1),
            a_sc.narrow(0, row_start, rows).view(-1),
            b_sc,
            offs_window,
            n_blocks,
            n_c_tiles,
            rows,
            param.n,
            stream,
        )

        # Deferred: the descriptors are only needed if a compile happens, and
        # after the first call for a given param it never does. Building them
        # eagerly cost ~8us of the launcher's ~26us per call, which is dead
        # weight on shapes whose kernel time is of that order.
        def compile_args_factory(dispatch_args=dispatch_args):
            return tuple(
                tensor_arg(arg) if idx < 6 else arg
                for idx, arg in enumerate(dispatch_args)
            )

        if compile_only:
            flyc.compile(launch_for(*key), *compile_args_factory())
            return out
        # A window that fell back re-resolves, so later windows skip the retry.
        launch_guarded(
            launch_for(*key), key, compile_args_factory, dispatch_args, param
        )
    return out


def _row_windows(M, K, N, offs, block_r=BLOCK_R):
    """Split the token dim into windows no single launch can overflow int32 on.

    FlyDSL's CABI packs a tensor's shape entries as int32
    (``jit_argument._LayoutPlan``: ``"i" * len(shape)``), and every operand goes
    over as a 1-D view, so ``shape[0]`` IS the element count. Past 2**31 the pack
    raises ``struct.error`` from inside the dispatch, which surfaces as a hard
    failure rather than a fallback. This is reachable in practice: at 65k
    tokens/rank, M=1,108,768 x K=2048 = 2,270,756,864.

    Splitting is sound HERE, and only here, because the groups partition the
    OUTPUT ROWS rather than the contraction: each window owns a disjoint row
    range, so the windows never share a partial sum and need no accumulation.

    Yields ``(row_start, n_rows, window_offsets)``. The offsets are rebased and
    clamped ON DEVICE -- a group that straddles a boundary ends at ``n_rows`` in
    one window and starts at 0 in the next, and groups wholly outside collapse to
    size 0. No host sync, which is the property this whole kernel is built around.
    """
    # Bound the widest M-dependent operand: A is (M, K), out is (M, N), and A's
    # scales are (M, K//32).
    rows_max = (2**31 - 1) // max(K, N)
    # Align down to the row-tile size so a window boundary never splits a tile;
    # the tiling inside each window is then identical to the unsplit case.
    rows_max = (rows_max // block_r) * block_r
    if rows_max < block_r:
        raise NotImplementedError(
            f"a single row tile of {block_r} rows already exceeds the int32 "
            f"operand limit at K={K} N={N}"
        )

    if M <= rows_max:
        yield 0, M, offs  # unsplit: the single-launch path
        return
    for r0 in range(0, M, rows_max):
        rows = min(rows_max, M - r0)
        yield r0, rows, (offs - r0).clamp_(0, rows)


# Name the lazy-export table in `kernels/__init__.py` binds to.
pick_mxfp8_grouped_gemm_tile = pick_tile
