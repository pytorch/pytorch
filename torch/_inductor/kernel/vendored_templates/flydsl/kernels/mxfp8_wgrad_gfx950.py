# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ragged MXFP8 weight-gradient grouped GEMM for gfx950.

``out[g] = A[:, offs[g-1]:offs[g]] @ B[:, offs[g-1]:offs[g]].T`` for row-major
``A[N, M]`` and ``B[K, M]`` from the 32x1 MXFP8 cast, with device-resident
``offs``. Each workgroup owns one output tile of one group and walks the group's
tokens rounded out to a 256-token window; window tokens outside the group are
zero-filled at load, so they contribute nothing, NaN or Inf included.
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
from .mxfp8_grouped_gemm_gfx950 import _compile_gfx950, _INT32_MAX


BLOCK_R = 256
BLOCK_C = 256
BLOCK_M = 128
SCALE_BLOCK = 32
WINDOW_ALIGN = 256
LPT_MAX_E = 32
ORDER_MODES = ("id", "idsel", "lpt")
_IDSEL_MIN_BLOCKS = 1024
_XCD_COUNT = 8  # full MI355X; partitioned parts pass their count as num_xcd
_SUPERTILE_TARGET = 4
_ATTRS = {"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"}


def _divisors(value: int) -> list[int]:
    return [divisor for divisor in range(1, value + 1) if value % divisor == 0]


def _pick_supertile(n_r: int, n_c: int) -> tuple[int, int]:
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


def _swizzle_params(n: int, k: int, num_xcd: int) -> tuple[int, int, int]:
    n_r = ceildiv(n, BLOCK_R)
    n_c = ceildiv(k, BLOCK_C)
    group_r, group_c = _pick_supertile(n_r, n_c)
    xcd_count = num_xcd if num_xcd > 1 and (n_r * n_c) % num_xcd == 0 else 1
    return xcd_count, group_r, group_c


def _mfma_scale_agpr(a, b, sa, sb, acc):
    # The accumulator is pinned in AGPRs and tied to the output, so it
    # accumulates in place instead of being shuffled between slots.
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


# Past every operand's num_records (operands are int32-bounded): reads as zero.
_OOB_VOFFSET = 0x7FFFFFFF


class _WindowedG2S(G2SLoader):
    # Window-edge tokens belong to neighbouring groups or the padding past
    # offs[-1]. Zeroing their e8m0 scale gives 2**-127, not 0, so NaN/Inf would
    # survive. Group bounds are multiples of 32, so each lane's 16-token chunk
    # is wholly inside or outside the group; an outside chunk is loaded from an
    # out-of-range offset, which the buffer range check turns into zeros.
    def __init__(self, gl_src, gl_offsets, gl_cols, head, span, *args):
        super().__init__(gl_src, gl_offsets, *args)
        self.gl_cols = gl_cols
        self.head = head
        self.span = span

    def _src(self, step, k_col):
        e = k_col + self.gl_cols[step]
        inside = (e >= self.head) & (e < self.span)
        off = arith.select(
            inside, fx.Int32(self.gl_offsets[step]), fx.Int32(_OOB_VOFFSET)
        )
        return fx.slice(self.gl_src, (None, off))

    def load(self, lds_dst, k_offset, k_col):
        for step in range_constexpr(self.n_load_steps):
            self.load_one(lds_dst, k_offset, step, k_col)

    def load_one(self, lds_dst, k_offset, step, k_col):
        dst = self._lds_dst_at(lds_dst, step)
        fx.copy(
            self.g2lds_atom, self._src(step, k_col), dst, soffset=fx.Int32(k_offset)
        )


def _global_swizzle_cols(lane_id, wave_id, n_rounds):
    # The column part of compute_global_swizzle(..., preshuffled=False).
    n_waves = fx.block_dim.x // 64
    cols = []
    for round in range_constexpr(n_rounds):
        row = lane_id // 8 + wave_id * 8 + round * (n_waves * 8)
        _, c = swizzle_128(row, (lane_id % 8) * 16)
        cols.append(c)
    return cols


def _trap_if(bad):
    # s_trap 2 (as __builtin_trap) when the wave-uniform `bad` is nonzero.
    _llvm.inline_asm(
        res=None,
        operands_=[arith._to_raw(bad)],
        asm_string="s_cmp_eq_u32 $0, 0\ns_cbranch_scc1 1f\ns_trap 2\n1:",
        constraints="s",
        has_side_effects=True,
    )


def _uniform(value):
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

    # Paired scale loads need M_ROW % 256 == 0, else two dword loads; a compile
    # key because it changes the wait counts below.
    SC_PAIR = sc_pair
    N_SC_OPS = 4 if SC_PAIR else 8  # scale vm ops per unrolled body
    N_CHUNK = 8  # raw i32 carried per chunk (4 halves x 2 K-steps)

    # Vector loads trailing each LDS fill. Too large a count can expose an
    # incomplete fill before the barrier.
    W1A = 2 * (N_TILES_A + N_TILES_B)
    W2A = 2 * (N_TILES_A + N_TILES_B) + N_SC_OPS
    W1B = 2 * (N_TILES_A + N_TILES_B) + N_SC_OPS
    W2B = 2 * (N_TILES_A + N_TILES_B)
    LDS_BLOCK_R = BLOCK_R // 2  # each wave-dim half owns 128 rows
    LDS_BLOCK_C = BLOCK_C // 2
    a_lds_size = LDS_BLOCK_R * BLOCK_M
    b_lds_size = LDS_BLOCK_C * BLOCK_M

    # Outside the traced kernel, so the order branch and offset loops unroll.
    def slot_to_group(offs_rsrc, slot):
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
        # One e8m0 per 32 tokens, 4 per i32: a scale row is m_row/128 i32.
        sc_i32_row = ArithValue(m_row) // fx.Int32(BLOCK_M)
        sc_e8_row = ArithValue(m_row) // fx.Int32(SCALE_BLOCK)
        bid = ArithValue(fx.block_idx.x)
        slot = bid // fx.Int32(TILES_PER_GROUP)
        tile = bid % fx.Int32(TILES_PER_GROUP)
        row_tile, col_tile = tile_coordinates(tile)
        r_base = row_tile * fx.Int32(BLOCK_R)
        c_base = col_tile * fx.Int32(BLOCK_C)
        offs_rsrc = buffer_ops.create_buffer_resource(
            OFFS, max_size=False, num_records_bytes=E * 4
        )
        g, m_start, m_end = slot_to_group(offs_rsrc, slot)
        # Trap unless bounds are multiples of the 32-token scale block (a block
        # cannot straddle two groups), non-decreasing, and within M, like MSLK.
        one_i32, zero_c = fx.Int32(1), fx.Int32(0)
        misaligned = ((m_start | m_end) & fx.Int32(SCALE_BLOCK - 1)) != zero_c
        out_of_range = (m_end > ArithValue(m_row)) | (m_end < m_start)
        _trap_if(arith.select(misaligned | out_of_range, one_i32, zero_c))

        # Window base rounded down to 256 (an even scale dword for the paired
        # loads); length rounded up to an even number of K-steps (two per loop
        # body), at least 4 (two prologue steps plus two drained at the end).
        lo0 = (m_start // fx.Int32(WINDOW_ALIGN)) * fx.Int32(WINDOW_ALIGN)
        head = m_start - lo0  # 0..224, multiple of 32
        span = m_end - lo0
        k_need = (span + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M)
        k_even = ((k_need + fx.Int32(1)) // fx.Int32(2)) * fx.Int32(2)
        K_eff = arith.select(k_even < fx.Int32(4), fx.Int32(4), k_even)
        N_ROLLED = K_eff - fx.Int32(2)
        m_start_i32 = lo0 // fx.Int32(BLOCK_M)

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
        kgrp32 = k_grp * fx.Int32(SCALE_BLOCK)
        mask_ff = fx.Int32(0xFF)
        bcast = fx.Int32(0x01010101)
        zero_i32 = fx.Int32(0)
        A0_gl = r_base * m_row + lo0
        A1_gl = (r_base + fx.Int32(LDS_BLOCK_R)) * m_row + lo0
        B0_gl = c_base * m_row + lo0
        B1_gl = (c_base + fx.Int32(LDS_BLOCK_C)) * m_row + lo0
        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        # Scale loads past the last row return zero via the range check, which
        # may not include soffset, so each lane's whole row offset is in the
        # VGPR. Window overhang stays inside the plane and is masked by window_ok.
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
        gl_off = compute_global_swizzle(
            lane_id, wave_id, m_row, N_LDS_ROUNDS, preshuffled=False
        )
        gl_cols = _global_swizzle_cols(lane_id, wave_id, N_LDS_ROUNDS)
        a_g2s = _WindowedG2S(
            a_div, gl_off, gl_cols, head, span, N_TILES_A, F8_IR_t, wave_id
        )
        b_g2s = _WindowedG2S(
            b_div, gl_off, gl_cols, head, span, N_TILES_B, F8_IR_t, wave_id
        )
        a_s2r = S2RLoader(wave_i, N_TILES_A)
        b_s2r = S2RLoader(wave_j, N_TILES_B)
        # Each lane loads one scale row (4 tiles x 16 rows per half) for both
        # K-steps; ds_bpermute then hands each tile its scales.
        sa0_base = r_base + wave_i * fx.Int32(N_TILES_A * 16)
        sa1_base = r_base + fx.Int32(LDS_BLOCK_R) + wave_i * fx.Int32(N_TILES_A * 16)
        sb0_base = c_base + wave_j * fx.Int32(N_TILES_B * 16)
        sb1_base = c_base + fx.Int32(LDS_BLOCK_C) + wave_j * fx.Int32(N_TILES_B * 16)
        sc_vbase = [
            (_rs, (_base + lane_id) * sc_i32_row + m_start_i32)
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

        # Scales for K-steps (k_i32, k_i32 + 1), flat: [a0_lo, a0_hi, a1_lo, ...].
        def issue_chunk(k_i32):
            out = []
            for _rs, _vb in sc_vbase:
                voff = _vb + k_i32
                if SC_PAIR:
                    v = buffer_ops.buffer_load(_rs, voff, vec_width=2, dtype=T.i32)
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
                        buffer_ops.buffer_load(_rs, voff, vec_width=1, dtype=T.i32)
                    )
                    out.append(
                        buffer_ops.buffer_load(
                            _rs, voff + fx.Int32(1), vec_width=1, dtype=T.i32
                        )
                    )
            return out

        def window_ok(k_idx):
            e = k_idx * fx.Int32(BLOCK_M) + kgrp32
            return (e >= head) & (e < span)

        def use_chunk(chunk, p, k_idx):
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
        k_col0, k_col1 = fx.Int32(0), fx.Int32(BLOCK_M)
        a_g2s.load(a_cur0, A0_gl + 0 * BLOCK_M, k_col0)
        b_g2s.load(b_cur0, B0_gl + 0 * BLOCK_M, k_col0)
        b_g2s.load(b_cur1, B1_gl + 0 * BLOCK_M, k_col0)
        a_g2s.load(a_cur1, A1_gl + 0 * BLOCK_M, k_col0)
        a_g2s.load(a_next0, A0_gl + 1 * BLOCK_M, k_col1)
        b_g2s.load(b_next0, B0_gl + 1 * BLOCK_M, k_col1)
        b_g2s.load(b_next1, B1_gl + 1 * BLOCK_M, k_col1)
        a_g2s.load(a_next1, A1_gl + 1 * BLOCK_M, k_col1)
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

        def _cluster(lds_dst, g2s, k_off, k_col, s2r, lds_src, a, b, c, sa, sb):
            # The 16 MFMAs interleaved with the next fragment's g2s and s2r
            # loads, which co-issue in the MFMA shadow.
            swz = _lds_swizzle(s2r)
            rt = []
            mfmas = iter(range_constexpr(N_ACCUMS))

            def M(count):
                for _ in range_constexpr(count):
                    idx = next(mfmas)
                    i, j = idx // N_TILES_B, idx % N_TILES_B
                    c[idx] = _mfma_scale_agpr(a[i], b[j], sa[i], sb[j], c[idx])

            # MFMAs 0-1, then per tile: g2s, s2r lo, MFMAs, s2r hi, MFMAs, with
            # one MFMA per gap for tile 0 and two after.
            M(2)
            for t in range_constexpr(N_TILES_A):
                g2s.load_one(lds_dst, k_off, t, k_col)
                d0 = s2r.load_one(lds_src, swz[t][0])
                M((1, 2, 2, 2)[t])
                d1 = s2r.load_one(lds_src, swz[t][1])
                rt.append(pack_i32x4_i32x8(d0, d1))
                M((1, 2, 2, 2)[t])
            return c, rt

        _R = arith._to_raw
        NA, NB = N_TILES_A, N_TILES_B

        # One K-step; slot p == 0 of the two-step body also issues the scale
        # chunk for K-steps (issue_k, issue_k + 1).
        def _one_step(kk_i, a0, b0, accs, bufs, chunk, p, issue_k):
            c00, c01, c10, c11 = accs
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            k2 = (kk_i + fx.Int32(2)) * fx.Int32(BLOCK_M)
            wait_barrier(W1A if p == 0 else W1B)
            ch_new = issue_chunk(issue_k) if p == 0 else None
            sa0, sa1, sb0, sb1 = use_chunk(chunk, p, kk_i)
            c00, b1 = _cluster(
                ac0, a_g2s, A0_gl + k2, k2, b_s2r, bc1, a0, b0, c00, sa0, sb0
            )
            c01, a1 = _cluster(
                bc0, b_g2s, B0_gl + k2, k2, a_s2r, ac1, a0, b1, c01, sa0, sb1
            )
            wait_barrier(W2A if p == 0 else W2B)
            c10, a0n = _cluster(
                bc1, b_g2s, B1_gl + k2, k2, a_s2r, an0, a1, b0, c10, sa1, sb0
            )
            c11, b0n = _cluster(
                ac1, a_g2s, A1_gl + k2, k2, b_s2r, bn0, a1, b1, c11, sa1, sb1
            )
            new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)
            return a0n, b0n, (c00, c01, c10, c11), new_bufs, ch_new

        bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)

        def _pack(a0, b0, accs, chunk):
            return [_R(x) for part in (a0, b0, *accs, chunk) for x in part]

        def _unpack(state):
            parts, o = [], 0
            for size in (NA, NB, N_ACCUMS, N_ACCUMS, N_ACCUMS, N_ACCUMS, N_CHUNK):
                parts.append(list(state[o : o + size]))
                o += size
            a0, b0, c00, c01, c10, c11, chunk = parts
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

        wait_barrier(0)
        sa0, sa1, sb0, sb1 = use_chunk(chunk, 1, t1)
        b1 = b_s2r.load(b_next1)
        a1 = a_s2r.load(a_next1)
        c00 = mma(a0, b0, c00, sa0, sb0)
        c01 = mma(a0, b1, c01, sa0, sb1)
        c10 = mma(a1, b0, c10, sa1, sb0)
        c11 = mma(a1, b1, c11, sa1, sb1)

        # Empty groups are selected to exact zero: their e8m0 0x00 is 2**-127.
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
            c11, base_row + fx.Int32(LDS_BLOCK_R), base_col + fx.Int32(LDS_BLOCK_C)
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
            A, B, OUT, A_scale, B_scale, OFFS, out_r, out_c, m_row, value_attrs=_ATTRS
        ).launch(grid=(n_blocks, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_wgrad


cached_launch = functools.lru_cache(maxsize=None)(_compile)


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
        return dataclasses.astuple(self)

    def __cache_signature__(self) -> tuple[int, int, int, bool, str, int, int, int]:
        return self.key()


def make_mxfp8_wgrad_param(
    n: int, k: int, group_count: int, *, sc_pair: bool, num_xcd: int = _XCD_COUNT
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
    # The output goes over as a 1-D view; FlyDSL packs its element count as int32.
    if group_count * n * k > _INT32_MAX:
        raise NotImplementedError(
            f"MXFP8 wgrad output (G={group_count}, N={n}, K={k}) exceeds the "
            f"int32 operand limit {_INT32_MAX}"
        )
    blocks = group_count * ceildiv(n, BLOCK_R) * ceildiv(k, BLOCK_C)
    if blocks >= _IDSEL_MIN_BLOCKS:
        order = "idsel"
    else:
        order = "lpt" if group_count <= LPT_MAX_E else "id"
    xcd_count, group_r, group_c = _swizzle_params(n, k, num_xcd)
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
    # M is the contraction, so the operands cannot be split into windows the
    # way the forward splits rows: refuse before FlyDSL's int32 packing fails.
    if max(n, k) * m > _INT32_MAX:
        raise NotImplementedError(
            f"MXFP8 wgrad operand ({max(n, k)} x {m}) exceeds the int32 operand "
            f"limit {_INT32_MAX}"
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
        _compile_gfx950(launch, *compile_args_factory())
        return out
    run_cached_flydsl(
        launch,
        constexpr_param=param,
        compiler=_compile_gfx950,
        dispatch_args=dispatch_args,
        compile_args_factory=compile_args_factory,
    )
    return out


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)
