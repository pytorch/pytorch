# SPDX-License-Identifier: MIT
# Vendored from ROCm/AMD-TorchTitan-Ops amd_titan/_ops/mxwgrad/_kernel_v1.py.
"""Ragged MXFP8 weight-gradient grouped GEMM for gfx950.

The inputs are row-major ``A[N, M]`` and ``B[K, M]`` views produced by the
32x1 MXFP8 cast. Device-resident offsets partition the contracting dimension
``M`` and the kernel writes ``out[G, N, K]``.

Each workgroup owns one output tile and walks the full contraction interval for
one group. Group sizes and the operand row stride stay runtime values, so routed
token counts do not become FlyDSL compile keys. The contraction window is
rounded outward to 256 elements for cooperative scale loads; scale operands are
masked on both sides outside the group, including empty groups.

The body uses four waves, an eight-buffer LDS pipeline, AGPR accumulators, and
device-side group scheduling. Large grids use constant-index offset loads and
XCD-aware rectangular tile rasterization.
"""

import dataclasses
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Vector as Vec

import torch
from torch._inductor.runtime.flydsl_cache import run_cached_flydsl

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


BLOCK_R = 256
BLOCK_C = 256
BLOCK_M = 128
SCALE_BLOCK = 32
WINDOW_ALIGN = 256
LPT_MAX_E = 32
ORDER_MODES = ("id", "idsel", "lpt")
_INTERLEAVE = True
_IDSEL_MIN_BLOCKS = 1024
_XCD_COUNT = 8
_SUPERTILE_TARGET = 4


def _divisors(value: int) -> list[int]:
    return [divisor for divisor in range(1, value + 1) if value % divisor == 0]


def _pick_supertile(n_r: int, n_c: int) -> tuple[int, int]:
    """Pick the largest rectangular tile no bigger than the measured target."""
    best_score = (0, 0)
    best_r, best_c = 1, 1
    for group_r in _divisors(n_r):
        for group_c in _divisors(n_c):
            area = group_r * group_c
            if area > _SUPERTILE_TARGET:
                continue
            score = (area, -(group_r + group_c))
            if score > best_score:
                best_score = score
                best_r, best_c = group_r, group_c
    return best_r, best_c


def _swizzle_params(n: int, k: int) -> tuple[int, int, int]:
    """Return the XCD gather count and rectangular output-tile dimensions."""
    n_r = ceildiv(n, BLOCK_R)
    n_c = ceildiv(k, BLOCK_C)
    group_r, group_c = _pick_supertile(n_r, n_c)
    xcd_count = _XCD_COUNT if (n_r * n_c) % _XCD_COUNT == 0 else 1
    return xcd_count, group_r, group_c


def _mfma_scale_agpr(a, b, sa, sb, acc):
    """fp8 16x16x128 scaled MFMA, accumulator pinned in AGPR (=a,...,0) so the
    f32x4 acc accumulates in place and the compiler does not shuffle it
    between AGPR slots. a/b are i32x8 (K=128 fp8); sa/sb are broadcast-i32
    E8M0 scales. cbsz:0 blgp:0, no op_sel (fp8). Identical to v3.3's."""
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


def _uniform(value):
    """Place a value known to be wave-uniform in an SGPR."""
    return fx.Int32(rocdl.readfirstlane(res=T.i32, src=arith._to_raw(value)))


def _compile(
    R: int,
    C: int,
    E: int,
    sc_pair: bool,
    order: str,
    xcd_count: int,
    group_r: int,
    group_c: int,
):
    """Build a kernel keyed by output geometry, group count, and scale loading."""
    if order not in ORDER_MODES:
        raise AssertionError(f"unknown order {order!r}")
    ORDER = order
    N_R_TILES = ceildiv(R, BLOCK_R)
    N_C_TILES = ceildiv(C, BLOCK_C)
    TILES_PER_GROUP = N_R_TILES * N_C_TILES
    SUPERTILES_PER_ROW = N_C_TILES // group_c

    def tile_coordinates(tile):
        if xcd_count > 1:
            tile = (tile % fx.Int32(xcd_count)) * fx.Int32(
                TILES_PER_GROUP // xcd_count
            ) + tile // fx.Int32(xcd_count)
        if group_r * group_c > 1:
            supertile = tile // fx.Int32(group_r * group_c)
            within = tile % fx.Int32(group_r * group_c)
            row = (supertile // fx.Int32(SUPERTILES_PER_ROW)) * fx.Int32(
                group_r
            ) + within // fx.Int32(group_c)
            col = (supertile % fx.Int32(SUPERTILES_PER_ROW)) * fx.Int32(
                group_c
            ) + within % fx.Int32(group_c)
            return row, col
        return tile // fx.Int32(N_C_TILES), tile % fx.Int32(N_C_TILES)

    N_TILES_A = BLOCK_R // 4 // 16
    N_TILES_B = BLOCK_C // 4 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    N_LDS_ROUNDS = max(N_TILES_A, N_TILES_B)

    # Pair loads require M_ROW % 256 == 0. Other 128-aligned rows use two
    # dword loads; the choice is a compile key because it changes wait counts.
    SC_PAIR = sc_pair
    N_SC_OPS = 4 if SC_PAIR else 8  # scale vm ops per unrolled body
    N_CHUNK = 8  # raw i32 carried per chunk (4 halves x 2 K-steps)

    # These counts track the vector loads trailing each LDS fill. A count that
    # is too large can expose an incomplete fill before the barrier.
    W1A = 2 * (N_TILES_A + N_TILES_B)
    W2A = 2 * (N_TILES_A + N_TILES_B) + N_SC_OPS
    W1B = 2 * (N_TILES_A + N_TILES_B) + N_SC_OPS
    W2B = 2 * (N_TILES_A + N_TILES_B)

    LDS_BLOCK_R = BLOCK_R // 2  # 128: each wave-dim half owns 128 rows
    LDS_BLOCK_C = BLOCK_C // 2

    a_lds_size = LDS_BLOCK_R * BLOCK_M
    b_lds_size = LDS_BLOCK_C * BLOCK_M

    def slot_to_group(offs_rsrc, slot):
        """Map a dispatch slot to a group and its device-resident bounds.

        This helper remains outside the traced kernel so the order branch and
        fixed-length offset loops are resolved and unrolled at compile time.
        """
        if ORDER == "id":
            # Clamp the discarded g=-1 read to the bounded offsets buffer.
            g = slot
            g_prev = arith.select(g > fx.Int32(0), g - fx.Int32(1), fx.Int32(0))
            prev_raw = buffer_ops.buffer_load(
                offs_rsrc, g_prev, vec_width=1, dtype=T.i32, is_scalar=True
            )
            end_raw = buffer_ops.buffer_load(
                offs_rsrc, g, vec_width=1, dtype=T.i32, is_scalar=True
            )
            m_start = arith.select(g > fx.Int32(0), ArithValue(prev_raw), fx.Int32(0))
            return g, ArithValue(m_start), ArithValue(end_raw)

        # Constant indices let all group boundaries issue together.
        ends = [
            ArithValue(
                buffer_ops.buffer_load(
                    offs_rsrc, fx.Int32(i), vec_width=1, dtype=T.i32, is_scalar=True
                )
            )
            for i in range(E)
        ]
        starts = [ArithValue(fx.Int32(0))] + ends[:-1]

        if ORDER == "idsel":
            g = slot
            m_start = ArithValue(fx.Int32(0))
            m_end = ends[0]
            for i in range(E):
                mine = g == fx.Int32(i)
                m_start = arith.select(mine, starts[i], m_start)
                m_end = arith.select(mine, ends[i], m_end)
            return g, ArithValue(m_start), ArithValue(m_end)

        cost = [
            ends[i] - (starts[i] // fx.Int32(WINDOW_ALIGN)) * fx.Int32(WINDOW_ALIGN)
            for i in range(E)
        ]

        g = ArithValue(fx.Int32(0))
        m_start = ArithValue(fx.Int32(0))
        m_end = ends[0]
        for i in range(E):
            rank = ArithValue(fx.Int32(0))
            for j in range(E):
                # Index order breaks ties and makes the ranking a permutation.
                ahead = (cost[j] >= cost[i]) if j < i else (cost[j] > cost[i])
                rank = rank + arith.select(ahead, fx.Int32(1), fx.Int32(0))
            mine = rank == slot
            g = arith.select(mine, fx.Int32(i), g)
            m_start = arith.select(mine, starts[i], m_start)
            m_end = arith.select(mine, ends[i], m_end)
        return ArithValue(g), ArithValue(m_start), ArithValue(m_end)

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
    def kernel_wgrad(
        A: fx.Tensor,
        B: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        OFFS: fx.Tensor,
        out_r: fx.Int32,
        out_c: fx.Int32,
        m_row: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

        # Runtime row stride. One e8m0 per 32 contracting elements, 4 packed
        # per i32, so a row is m_row/128 i32 and m_row/32 bytes.
        sc_i32_row = ArithValue(m_row) // fx.Int32(BLOCK_M)
        sc_e8_row = ArithValue(m_row) // fx.Int32(SCALE_BLOCK)

        # -- Block -> (group, output tile). One block per (g, tile). --
        bid = ArithValue(fx.block_idx.x)
        slot = bid // fx.Int32(TILES_PER_GROUP)
        tile = bid % fx.Int32(TILES_PER_GROUP)
        row_tile, col_tile = tile_coordinates(tile)
        r_base = row_tile * fx.Int32(BLOCK_R)
        c_base = col_tile * fx.Int32(BLOCK_C)

        # -- Group bounds, read ON DEVICE (uniform s.buffer.load, no sync) --
        offs_rsrc = buffer_ops.create_buffer_resource(
            OFFS, max_size=False, num_records_bytes=E * 4
        )

        g, m_start, m_end = slot_to_group(offs_rsrc, slot)

        # -- The window: base rounded DOWN to 256, length rounded UP to an
        # even number of K-steps >= 4. See notes 1-3 in the file header. --
        lo0 = (m_start // fx.Int32(WINDOW_ALIGN)) * fx.Int32(WINDOW_ALIGN)
        head = m_start - lo0  # 0..224, multiple of 32
        span = m_end - lo0  # real elements from the window base
        k_need = (span + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M)
        k_even = ((k_need + fx.Int32(1)) // fx.Int32(2)) * fx.Int32(2)
        K_eff = arith.select(k_even < fx.Int32(4), fx.Int32(4), k_even)
        N_ROLLED = K_eff - fx.Int32(2)  # even, >= 2: no constexpr peel needed
        m_start_i32 = lo0 // fx.Int32(BLOCK_M)  # even (lo0 is 256-aligned)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
        a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
        b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
        b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = _uniform(fx.thread_idx.x // 64)
        wave_i = wave_id // 2
        wave_j = wave_id % 2

        m_lane = lane_id % fx.Int32(16)
        k_grp = lane_id // fx.Int32(16)
        kshift = k_grp * fx.Int32(8)
        kgrp32 = k_grp * fx.Int32(SCALE_BLOCK)  # this lane's sub-block offset
        mask_ff = fx.Int32(0xFF)
        bcast = fx.Int32(0x01010101)
        zero_i32 = fx.Int32(0)

        # Contraction (K-walk) global offsets; row stride = m_row, base = lo0.
        A0_gl = r_base * m_row + lo0
        A1_gl = (r_base + fx.Int32(LDS_BLOCK_R)) * m_row + lo0
        B0_gl = c_base * m_row + lo0
        B1_gl = (c_base + fx.Int32(LDS_BLOCK_C)) * m_row + lo0

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        # Scale loads can overhang a row tile and the rounded contraction
        # window. Bounded descriptors make both cases return zero instead of
        # reading a neighboring row or an e8m0 NaN.
        sa_bytes = arith.index_cast(
            T.i64, arith.index_cast(T.index, fx.Int32(R) * sc_e8_row)
        )
        sb_bytes = arith.index_cast(
            T.i64, arith.index_cast(T.index, fx.Int32(C) * sc_e8_row)
        )
        sa_rsrc = buffer_ops.create_buffer_resource(
            A_scale, max_size=False, num_records_bytes=sa_bytes
        )
        sb_rsrc = buffer_ops.create_buffer_resource(
            B_scale, max_size=False, num_records_bytes=sb_bytes
        )

        gl_off_a = compute_global_swizzle(
            lane_id, wave_id, m_row, N_LDS_ROUNDS, preshuffled=False
        )
        gl_off_b = compute_global_swizzle(
            lane_id, wave_id, m_row, N_LDS_ROUNDS, preshuffled=False
        )

        a_g2s = G2SLoader(a_div, gl_off_a, N_TILES_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_TILES_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_i, N_TILES_A)
        b_s2r = S2RLoader(wave_j, N_TILES_B)

        # Each lane loads one scale row for both K-steps. ds_bpermute then
        # redistributes the four tile scales within the wave.
        if N_TILES_A != 4 or N_TILES_B != 4:
            raise AssertionError("cooperative scale load assumes 4x16=64 rows/half")

        sa0_base = r_base + wave_i * fx.Int32(N_TILES_A * 16)
        sa1_base = r_base + fx.Int32(LDS_BLOCK_R) + wave_i * fx.Int32(N_TILES_A * 16)
        sb0_base = c_base + wave_j * fx.Int32(N_TILES_B * 16)
        sb1_base = c_base + fx.Int32(LDS_BLOCK_C) + wave_j * fx.Int32(N_TILES_B * 16)

        # Half bases and K indices are wave-uniform and use the SGPR soffset.
        sc_voff = lane_id * sc_i32_row
        sc_sbase = [
            (_rs, (_base * sc_i32_row + m_start_i32) * fx.Int32(4))
            for _rs, _base in (
                (sa_rsrc, sa0_base),
                (sa_rsrc, sa1_base),
                (sb_rsrc, sb0_base),
                (sb_rsrc, sb1_base),
            )
        ]
        sc_perm = [
            fx.Int32(t_ * 64) + m_lane * fx.Int32(4)
            for t_ in range_constexpr(N_TILES_A)
        ]

        def issue_chunk(k_i32):
            """Cooperative scale loads for K-steps (k_i32, k_i32+1).

            Returns 2 raw i32 per operand-half, flat: [a0_lo, a0_hi, a1_lo, ...].
            """
            out = []
            for _rs, _sb in sc_sbase:
                soff = _uniform(_sb + k_i32 * fx.Int32(4))
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
                    # Two dwords handle an odd scale-row stride.
                    out.append(
                        buffer_ops.buffer_load(
                            _rs, sc_voff, vec_width=1, dtype=T.i32, soffset_bytes=soff
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
            return out

        def window_ok(k_idx):
            """Return whether this lane's scale block belongs to the group."""
            e = k_idx * fx.Int32(BLOCK_M) + kgrp32
            return (e >= head) & (e < span)

        def use_chunk(chunk, p, k_idx):
            """Redistribute a scale chunk and mask both operands to the group."""
            ok = window_ok(k_idx)
            groups = []
            for h_ in range_constexpr(4):
                src = chunk[2 * h_ + p]
                grp = []
                for t_ in range_constexpr(N_TILES_A):
                    v = rocdl.ds_bpermute(res=T.i32, index=sc_perm[t_], src=src)
                    sc = ((ArithValue(v) >> kshift) & mask_ff) * bcast
                    grp.append(arith.select(ok, sc, zero_i32))
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

        # Pre-fill both sides of the eight-buffer LDS pipeline.
        chunk = issue_chunk(fx.Int32(0))
        a_g2s.load(a_cur0, A0_gl + 0 * BLOCK_M)
        b_g2s.load(b_cur0, B0_gl + 0 * BLOCK_M)
        b_g2s.load(b_cur1, B1_gl + 0 * BLOCK_M)
        a_g2s.load(a_cur1, A1_gl + 0 * BLOCK_M)

        a_g2s.load(a_next0, A0_gl + 1 * BLOCK_M)
        b_g2s.load(b_next0, B0_gl + 1 * BLOCK_M)
        b_g2s.load(b_next1, B1_gl + 1 * BLOCK_M)
        a_g2s.load(a_next1, A1_gl + 1 * BLOCK_M)

        wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))
        a0 = a_s2r.load(a_cur0)
        wait_barrier((3 * N_TILES_A) + (3 * N_TILES_B))
        b0 = b_s2r.load(b_cur0)

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
                    col = (lane_id // fx.Int32(16)) * fx.Int32(16) + fx.Int32(ii * 64)
                    r_, c_ = swizzle_128(row, col)
                    swz.append(r_ * fx.Int32(BLOCK_M) + c_)
                out.append(swz)
            return out

        def _cluster_plain(lds_dst, g2s, k_off, s2r, lds_src, a, b, c, sa, sb):
            g2s.load(lds_dst, k_off)
            rt = s2r.load(lds_src)
            c = mma(a, b, c, sa, sb)
            return c, rt

        def _cluster_il(lds_dst, g2s, k_off, s2r, lds_src, a, b, c, sa, sb):
            # Interleave the 16 scaled MFMAs (4x4) with the g2s (4 steps) and
            # s2r (4 tiles x 2 halves) loads of the NEXT fragment, so the loads
            # co-issue in the MFMA execute shadow.
            swz = _lds_swizzle(s2r)
            rt = []

            def M(i, j):
                c[i * N_TILES_B + j] = _mfma_scale_agpr(
                    a[i], b[j], sa[i], sb[j], c[i * N_TILES_B + j]
                )

            M(0, 0)
            M(0, 1)
            g2s.load_one(lds_dst, k_off, 0)
            d0 = s2r.load_one(lds_src, swz[0][0])
            M(0, 2)
            d1 = s2r.load_one(lds_src, swz[0][1])
            rt.append(pack_i32x4_i32x8(d0, d1))
            M(0, 3)
            g2s.load_one(lds_dst, k_off, 1)
            d0 = s2r.load_one(lds_src, swz[1][0])
            M(1, 0)
            M(1, 1)
            d1 = s2r.load_one(lds_src, swz[1][1])
            rt.append(pack_i32x4_i32x8(d0, d1))
            M(1, 2)
            M(1, 3)
            g2s.load_one(lds_dst, k_off, 2)
            d0 = s2r.load_one(lds_src, swz[2][0])
            M(2, 0)
            M(2, 1)
            d1 = s2r.load_one(lds_src, swz[2][1])
            rt.append(pack_i32x4_i32x8(d0, d1))
            M(2, 2)
            M(2, 3)
            g2s.load_one(lds_dst, k_off, 3)
            d0 = s2r.load_one(lds_src, swz[3][0])
            M(3, 0)
            M(3, 1)
            d1 = s2r.load_one(lds_src, swz[3][1])
            rt.append(pack_i32x4_i32x8(d0, d1))
            M(3, 2)
            M(3, 3)
            return c, rt

        _cluster = _cluster_il if _INTERLEAVE else _cluster_plain

        # The runtime K-loop advances two steps per body, matching the LDS
        # ping-pong period and leaving two steps for the drain below.
        _R = arith._to_raw
        NA, NB = N_TILES_A, N_TILES_B

        def _one_step(kk_i, a0, b0, accs, bufs, chunk, p, issue_k):
            """One K-step. `p` is the step's slot in the unrolled body (0 or 1);
            slot 0 also issues the chunk for K-steps (issue_k, issue_k+1)."""
            c00, c01, c10, c11 = accs
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            k2 = (kk_i + fx.Int32(2)) * fx.Int32(BLOCK_M)
            wait_barrier(W1A if p == 0 else W1B)
            # Prefetch scales for the body two K-steps ahead.
            ch_new = issue_chunk(issue_k) if p == 0 else None
            sa0, sa1, sb0, sb1 = use_chunk(chunk, p, kk_i)
            c00, b1 = _cluster(
                ac0, a_g2s, A0_gl + k2, b_s2r, bc1, a0, b0, c00, sa0, sb0
            )
            c01, a1 = _cluster(
                bc0, b_g2s, B0_gl + k2, a_s2r, ac1, a0, b1, c01, sa0, sb1
            )
            wait_barrier(W2A if p == 0 else W2B)
            c10, a0n = _cluster(
                bc1, b_g2s, B1_gl + k2, a_s2r, an0, a1, b0, c10, sa1, sb0
            )
            c11, b0n = _cluster(
                ac1, a_g2s, A1_gl + k2, b_s2r, bn0, a1, b1, c11, sa1, sb1
            )
            new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)
            return a0n, b0n, (c00, c01, c10, c11), new_bufs, ch_new

        bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)

        def _pack(a0, b0, accs, chunk):
            c00, c01, c10, c11 = accs
            return (
                [_R(x) for x in a0]
                + [_R(x) for x in b0]
                + [_R(x) for x in c00]
                + [_R(x) for x in c01]
                + [_R(x) for x in c10]
                + [_R(x) for x in c11]
                + [_R(x) for x in chunk]
            )

        def _unpack(state):
            o = 0
            a0 = list(state[o : o + NA])
            o += NA
            b0 = list(state[o : o + NB])
            o += NB
            c00 = list(state[o : o + N_ACCUMS])
            o += N_ACCUMS
            c01 = list(state[o : o + N_ACCUMS])
            o += N_ACCUMS
            c10 = list(state[o : o + N_ACCUMS])
            o += N_ACCUMS
            c11 = list(state[o : o + N_ACCUMS])
            o += N_ACCUMS
            chunk = list(state[o : o + N_CHUNK])
            o += N_CHUNK
            return a0, b0, (c00, c01, c10, c11), chunk

        init_state = _pack(a0, b0, (c00, c01, c10, c11), chunk)
        state = init_state
        for kk, state in range(0, N_ROLLED, 2, init=init_state):
            a0, b0, accs, chunk = _unpack(state)
            kk_i = fx.Int32(kk)
            a0, b0, accs, bufs, ch_new = _one_step(
                kk_i, a0, b0, accs, bufs0, chunk, 0, kk_i + fx.Int32(2)
            )
            a0, b0, accs, bufs, _ = _one_step(
                kk_i + fx.Int32(1), a0, b0, accs, bufs, chunk, 1, None
            )
            state = yield _pack(a0, b0, accs, ch_new)

        a0, b0, accs, chunk = _unpack(state)
        c00, c01, c10, c11 = accs
        a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1 = bufs0

        # Drain the final two prefetched K-steps.
        t0 = K_eff - fx.Int32(2)
        t1 = K_eff - fx.Int32(1)

        wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
        sa0, sa1, sb0, sb1 = use_chunk(chunk, 0, t0)
        b1 = b_s2r.load(b_cur1)
        c00 = mma(a0, b0, c00, sa0, sb0)
        a1 = a_s2r.load(a_cur1)
        c01 = mma(a0, b1, c01, sa0, sb1)
        wait_barrier((1 * N_TILES_A) + (1 * N_TILES_B))
        a0 = a_s2r.load(a_next0)
        c10 = mma(a1, b0, c10, sa1, sb0)
        b0 = b_s2r.load(b_next0)
        c11 = mma(a1, b1, c11, sa1, sb1)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        wait_barrier(0)
        sa0, sa1, sb0, sb1 = use_chunk(chunk, 1, t1)
        b1 = b_s2r.load(b_cur1)
        a1 = a_s2r.load(a_cur1)
        c00 = mma(a0, b0, c00, sa0, sb0)
        c01 = mma(a0, b1, c01, sa0, sb1)
        c10 = mma(a1, b0, c10, sa1, sb0)
        c11 = mma(a1, b1, c11, sa1, sb1)

        # One block owns each output tile. Empty groups are selected to exact
        # zero because e8m0 0x00 represents 2**-127 rather than mathematical 0.
        nonempty = m_end > m_start
        f32_zero = fx.Float32(0.0)
        out_r_i = arith.index_cast(T.index, out_r)
        out_c_i = arith.index_cast(T.index, out_c)
        n_experts = fx.Int32(E)
        nbytes = arith.index_cast(
            T.i64,
            out_r_i * out_c_i * arith.index_cast(T.index, n_experts) * fx.Index(2),
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            OUT, max_size=False, num_records_bytes=nbytes
        )
        base_row = r_base + wave_i * fx.Int32(N_TILES_A * 16)
        base_col = c_base + wave_j * fx.Int32(N_TILES_B * 16)
        g_off = g * out_r * out_c
        oob = out_r * out_c * n_experts

        def store_group(frag, br, bc):
            for ti in range_constexpr(N_TILES_A):
                row = br + fx.Int32(ti * 16) + k_grp * fx.Int32(4)
                for tj in range_constexpr(N_TILES_B):
                    col = bc + fx.Int32(tj * 16) + m_lane
                    col_ok = col < out_c
                    vec = frag[ti * N_TILES_B + tj]
                    for e in range_constexpr(4):
                        r_ = row + fx.Int32(e)
                        ok = (r_ < out_r) & col_ok
                        off = arith.select(ok, g_off + r_ * out_c + col, oob)
                        val = buffer_ops.vec_extract(
                            vec, static_position=[e], dynamic_position=[]
                        )
                        val = arith.select(nonempty, val, f32_zero)
                        val = arith.truncf(fx.BFloat16.ir_type, val)
                        buffer_ops.buffer_store(val, o_rsrc, off)

        store_group(c00, base_row, base_col)
        store_group(c01, base_row, base_col + fx.Int32(LDS_BLOCK_C))
        store_group(c10, base_row + fx.Int32(LDS_BLOCK_R), base_col)
        store_group(
            c11,
            base_row + fx.Int32(LDS_BLOCK_R),
            base_col + fx.Int32(LDS_BLOCK_C),
        )

    @flyc.jit
    def launch_wgrad(
        A,
        B,
        OUT,
        A_scale,
        B_scale,
        OFFS,
        n_blocks: fx.Int32,
        out_r: fx.Int32,
        out_c: fx.Int32,
        m_row: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_wgrad(
            A,
            B,
            OUT,
            A_scale,
            B_scale,
            OFFS,
            out_r,
            out_c,
            m_row,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(n_blocks, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_wgrad


@functools.lru_cache(maxsize=None)
def cached_launch(
    R: int,
    C: int,
    E: int,
    sc_pair: bool,
    order: str,
    xcd_count: int,
    group_r: int,
    group_c: int,
):
    return _compile(R, C, E, sc_pair, order, xcd_count, group_r, group_c)


@dataclasses.dataclass(frozen=True)
class MXFP8WgradParam:
    n: int
    k: int
    group_count: int
    sc_pair: bool
    order: str
    xcd_count: int
    group_r: int
    group_c: int

    def key(self) -> tuple[int, int, int, bool, str, int, int, int]:
        return (
            self.n,
            self.k,
            self.group_count,
            self.sc_pair,
            self.order,
            self.xcd_count,
            self.group_r,
            self.group_c,
        )

    def __cache_signature__(self) -> tuple[int, int, int, bool, str, int, int, int]:
        return self.key()


def make_mxfp8_wgrad_param(
    n: int, k: int, group_count: int, *, sc_pair: bool
) -> MXFP8WgradParam:
    if n <= 0 or k <= 0 or group_count <= 0:
        raise NotImplementedError(
            "MXFP8 wgrad needs positive N, K, and group count; "
            f"got {n}, {k}, {group_count}"
        )
    if n % 16 != 0 or k % 16 != 0:
        raise NotImplementedError(
            f"MXFP8 wgrad needs N and K divisible by 16; got N={n}, K={k}"
        )
    blocks = group_count * ceildiv(n, BLOCK_R) * ceildiv(k, BLOCK_C)
    if blocks >= _IDSEL_MIN_BLOCKS:
        order = "idsel"
    else:
        order = "lpt" if group_count <= LPT_MAX_E else "id"
    xcd_count, group_r, group_c = _swizzle_params(n, k)
    return MXFP8WgradParam(
        n, k, group_count, sc_pair, order, xcd_count, group_r, group_c
    )


def launch_mxfp8_wgrad_gfx950(
    out,
    mat_a,
    mat_b,
    scale_a,
    scale_b,
    offs,
    param: MXFP8WgradParam,
    stream,
    tensor_arg=None,
    compile_only=False,
):
    """Run the 2D x 2D MXFP8 grouped GEMM with groups along contraction."""
    if tensor_arg is None:

        def tensor_arg(tensor):
            return tensor

    n, m = mat_a.shape
    k, m_b = mat_b.shape
    if m != m_b:
        raise AssertionError(f"contracting dimensions differ: {m} and {m_b}")
    if m % BLOCK_M != 0:
        raise NotImplementedError(
            f"MXFP8 wgrad row stride must be divisible by {BLOCK_M}; got {m}"
        )
    if tuple(out.shape) != (param.group_count, n, k):
        raise AssertionError(
            f"output shape {tuple(out.shape)} != {(param.group_count, n, k)}"
        )

    n_r = ceildiv(n, BLOCK_R)
    n_c = ceildiv(k, BLOCK_C)
    n_blocks = param.group_count * n_r * n_c
    dispatch_args = (
        mat_a.view(torch.int8).view(-1),
        mat_b.view(torch.int8).view(-1),
        out.view(-1),
        scale_a.view(torch.uint8).view(-1),
        scale_b.view(torch.uint8).view(-1),
        offs.view(torch.int32),
        n_blocks,
        n,
        k,
        m,
        stream,
    )

    def compile_args_factory():
        return tuple(
            tensor_arg(arg) if index < 6 else arg
            for index, arg in enumerate(dispatch_args)
        )

    launch = cached_launch(*param.key())
    if compile_only:
        flyc.compile(launch, *compile_args_factory())
        return out
    run_cached_flydsl(
        launch,
        constexpr_param=param,
        compiler=flyc.compile,
        dispatch_args=dispatch_args,
        compile_args_factory=compile_args_factory,
    )
    return out


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)
