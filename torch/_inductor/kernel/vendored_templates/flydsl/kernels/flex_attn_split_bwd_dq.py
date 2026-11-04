# mypy: allow-untyped-defs
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Split-backward dQ kernel: one CTA per (head, Q block) streams K and V.

Head dims of 256 and up use 16x16x32 MFMAs; narrower ones 32x32x16.
"""

from dataclasses import dataclass, fields, replace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, math as fmath, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from . import flex_attn_split_bwd_common as common, flex_attn_split_bwd_ops as ops
from .flex_attn_split_bwd_common import (
    _ds_read_tr_v4f16_imm,
    _k_read_base,
    _ks_offset,
    _slab_span_elems,
    exp2_wait_state,
    FmhaInputMetadata,
    make_traits,
    mfma_operand_wait_state,
    ParityDualwaveTraits,
    ParityGemmHelper,
    ParityKernelContext,
    ParityKvGmemToLdsLoader,
    ParityKvLdsToVgprLoader,
    ParityQLoader,
    ParitySoftmaxHelper,
    ParityStoreHelper,
    traits_cache_key,
    wire_ptr,
    wire_view,
)
from .flex_attn_split_bwd_ops import MFMA16_M
from .mxfp8_buffer_ops import buffer_load, buffer_store


@dataclass(frozen=True)
class BwdDqTraits(ParityDualwaveTraits):
    MFMA_M: int = 32
    MFMA_N: int = 32
    MFMA_K: int = 16

    @property
    def ACC_ELEMS(self):
        return self.MFMA_M * self.MFMA_N // self.WARP_SIZE


def make_dq_traits(meta):
    width = common.tile_width_for(meta.head_dim)
    m16 = width >= 256
    waves = 4 if m16 or width < 128 else 2
    m, n, k = (16, 16, 32) if m16 else (32, 32, 16)
    base = make_traits(
        num_heads=meta.num_heads,
        num_kv_heads=meta.num_kv_heads,
        head_dim=width,
        num_waves=waves,
        block_m=waves * m,
        block_n=32 if m16 else 64,
        granule=32 if width % 64 else 64,
        causal=meta.causal,
        window=meta.window,
        dtype_str=meta.dtype_str,
        waves_per_eu=1,
    )
    carried = {
        f.name: getattr(base, f.name)
        for f in fields(base)
        if f.name != "LDS_KV_TOTAL_SIZE"
    }
    return BwdDqTraits(
        **carried,
        LDS_KV_TOTAL_SIZE=base.DUALWAVE_SWP_KV_PER_BUFFER,
        MFMA_M=m,
        MFMA_N=n,
        MFMA_K=k,
    )


_s_barrier = rocdl.s_barrier
_s_waitcnt = rocdl.s_waitcnt
_sched_barrier = rocdl.sched_barrier

MFMA16_K = 32
LANE_ELEMS = 8


def acc_elems(traits):
    return MFMA16_M * MFMA16_M // traits.WARP_SIZE


def k_steps16(traits):
    return traits.HEAD_DIM // MFMA16_K


def d_chunks16(traits):
    return traits.HEAD_DIM // MFMA16_M


class M16DqReader:
    def __init__(self, ctx, *, region, line_stride):
        self.ctx = ctx
        self.traits = ctx.traits
        self.line = line_stride
        self.base_elems = (
            ops._v_buf_base(self.traits, 0)
            if region == "v"
            else ops._k_buf_base(self.traits, 0)
        )
        self.scope = ops._dualwave_lds_scope(region, 0)

    def _tok_off(self, t):
        n = self.traits.SMEM_N_RPT
        return (t // n) * self.traits.D_128B_SIZE + (t % n) * self.line

    def _d_off(self, d):
        g = self.traits.D_128B_SIZE
        return (d // g) * self.traits.SMEM_N_RPT * self.line + (d % g)

    def a_token_base(self, lane):
        n = fx.Index(self.traits.SMEM_N_RPT)
        lane16 = lane % fx.Index(MFMA16_M)
        return (lane16 // n) * fx.Index(self.traits.D_128B_SIZE) + (
            lane16 % n
        ) * fx.Index(self.line)

    def load_a_tokens(self, lane, step, ks):
        d0 = LANE_ELEMS * (lane // fx.Index(MFMA16_M)) + fx.Index(ks * MFMA16_K)
        idx = (
            fx.Index(self.base_elems)
            + self.a_token_base(lane)
            + fx.Index(self._tok_off(step * MFMA16_M))
            + self._d_off_dyn(d0)
        )
        return self._lds_pack(idx)

    def _d_off_dyn(self, d):
        g = fx.Index(self.traits.D_128B_SIZE)
        return (d // g) * fx.Index(self.traits.SMEM_N_RPT * self.line) + (d % g)

    def _lds_pack(self, elem_idx):
        traits = self.traits
        ptr = ops._lds_byte_gep(self.ctx.lds_kv_base_ptr, elem_idx * traits.BF16_BYTES)
        return llvm.LoadOp(
            self.ctx.kv_mfma_pack_type,
            ptr,
            alignment=16,
            alias_scopes=ops._dualwave_lds_alias_scopes(self.scope),
            noalias_scopes=ops._dualwave_lds_noalias_scopes(
                self.scope, traits.LDS_SCOPE_NAMES
            ),
        ).result

    def load_a_all(self, lane):
        traits = self.traits
        steps = k_steps16(traits)
        return [
            [self.load_a_tokens(lane, s, ks) for ks in range_constexpr(steps)]
            for s in (0, 1)
        ]

    def _kt_read_base(self, lane):
        traits = self.traits
        g = lane // fx.Index(MFMA16_M)
        c = (lane % fx.Index(MFMA16_M)) // fx.Index(4)
        return (
            g * fx.Index(traits.D_128B_SIZE)
            + c * fx.Index(self.line)
            + (lane % fx.Index(4)) * fx.Index(4)
        )

    def load_kt(self, lane):
        traits = self.traits
        base = fx.Index(self.base_elems) + self._kt_read_base(lane)
        pair_bytes = self._tok_off(MFMA16_M) * traits.BF16_BYTES
        packs = []
        for c in range_constexpr(d_chunks16(traits)):
            imm = self._d_off(c * MFMA16_M) * traits.BF16_BYTES

            def read(off):
                return _ds_read_tr_v4f16_imm(
                    base,
                    off,
                    lds_kv_base_idx=self.ctx.lds_kv_base_idx,
                    v_lds_read_vec4_type=self.ctx.v_lds_read_vec4_type,
                    scope_name=self.scope,
                    scope_names=traits.LDS_SCOPE_NAMES,
                )

            lo, hi = read(imm), read(imm + pair_bytes)
            packs.append(Vec(lo).shuffle(Vec(hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
        return packs


class M16QLoader:
    def __init__(self, ctx, div, stride_seq_v, gmem_elem_offset):
        self.ctx = ctx
        self.traits = ctx.traits
        self.div = div
        self.stride = stride_seq_v
        self.gmem = gmem_elem_offset

    def load_all(self, q_row_in_block, lane):
        traits = self.traits
        packs = []
        for ks in range_constexpr(k_steps16(traits)):
            col = LANE_ELEMS * (lane // fx.Index(MFMA16_M)) + fx.Index(ks * MFMA16_K)
            raw = ops._buffer_load_128(
                self.gmem + q_row_in_block * self.stride + col,
                _load_atom_128=self.ctx.load_atom_128,
                q_div=self.div,
                q_load_i32x4_type=self.ctx.q_load_i32x4_type,
            )
            pack = Vec(raw, (4,), fx.Int32).bitcast(self.ctx.elem_dtype).ir_value()
            packs.append(pack)
        return packs


class M16DqGemmHelper:
    def __init__(self, ctx):
        self.ctx = ctx
        self.traits = ctx.traits
        self.atom = fx.make_mma_atom(
            fx.rocdl.MFMA(MFMA16_M, MFMA16_M, MFMA16_K, ctx.elem_dtype)
        )
        self.acc_ty = Vec.make_type(acc_elems(ctx.traits), fx.Float32)
        self.zero = Vec.filled(acc_elems(ctx.traits), 0.0, fx.Float32)

    def _mfma(self, a, b, c):
        return ops._mfma_acc(a, b, c, self.atom, self.acc_ty)

    def qk(self, a_packs, b_packs):
        out = []
        for step in (0, 1):
            acc = self.zero
            for ks in range_constexpr(k_steps16(self.traits)):
                acc = self._mfma(a_packs[step][ks], b_packs[ks], acc)
            out.append(acc)
        ops._s_nop(1)
        return tuple(out)

    def pv(self, ds_pack, kt_packs, v_dq):
        for c in range_constexpr(d_chunks16(self.traits)):
            v_dq[c] = self._mfma(kt_packs[c], ds_pack, v_dq[c])
        return v_dq


class M16DqSoftmax:
    def __init__(self, ctx):
        self.ctx = ctx
        self.traits = ctx.traits
        self.n = acc_elems(ctx.traits)

    def to_lists(self, v_s):
        return (
            [Vec(v_s[0])[r] for r in range_constexpr(self.n)],
            [Vec(v_s[1])[r] for r in range_constexpr(self.n)],
        )

    def to_vecs(self, lists):
        return tuple(
            Vec.from_elements([as_mlir_value(v) for v in half], fx.Float32).ir_value()
            for half in lists
        )

    def scale_and_sub_lse(self, v_s, qk_scale, lse2):
        fm = self.ctx.fm_fast
        scale_v = Vec.from_elements([fx.Float32(qk_scale)], fx.Float32).broadcast_to(
            self.n
        )
        neg = ops._fsub(self.ctx.c_zero_f, lse2, fm)
        neg_v = Vec.from_elements([fx.Float32(neg)], fx.Float32).broadcast_to(self.n)
        return tuple(
            as_mlir_value(fmath.fma(Vec(h), scale_v, neg_v, fastmath=fm)) for h in v_s
        )

    def kv_col(self, tile_idx, lane, step, i):
        return (
            fx.Int32(tile_idx * self.traits.BLOCK_N)
            + fx.Int32(step * MFMA16_M + i)
            + fx.Int32(4 * (lane // fx.Index(MFMA16_M)))
        )

    def seq_pad_mask(self, v_s, tile_idx, lane):
        traits = self.traits
        seqlen_kv_i32 = self.ctx.seqlen_kv_i32
        neg_inf = self.ctx.c_neg_inf
        to_lists, to_vecs, kv_col, n = self.to_lists, self.to_vecs, self.kv_col, self.n

        @flyc.jit
        def _mask_if_needed(v_s, tile_idx):
            s_lo, s_hi = v_s
            if fx.Int32((tile_idx + fx.Index(1)) * traits.BLOCK_N) > seqlen_kv_i32:
                lists = to_lists(v_s)
                for step in range_constexpr(2):
                    for i in range_constexpr(n):
                        live = kv_col(tile_idx, lane, step, i) < seqlen_kv_i32
                        lists[step][i] = live.select(
                            fx.Float32(lists[step][i]), neg_inf
                        )
                s_lo, s_hi = to_vecs(lists)
            return s_lo, s_hi

        return _mask_if_needed(v_s, tile_idx)

    def causal_mask(self, v_s, tile_idx, lane, q_row):
        traits = self.traits
        ctx = self.ctx
        neg_inf = ctx.c_neg_inf
        delta_i32 = ctx.delta_i32
        to_lists, to_vecs, kv_col, n = self.to_lists, self.to_vecs, self.kv_col, self.n
        windowed = const_expr(traits.WINDOW)
        window_left_i32 = ctx.window_left_i32 if windowed else None
        qs_i32 = fx.Int32(ctx.q_start) + fx.Int32(ctx.wave_id_uni) * fx.Int32(MFMA16_M)
        q_row_i32 = fx.Int32(q_row)

        @flyc.jit
        def _mask_if_needed(v_s, tile_idx):
            s_lo, s_hi = v_s
            kv_end = fx.Int32((tile_idx + fx.Index(1)) * traits.BLOCK_N)
            need = qs_i32 + delta_i32 < kv_end
            if windowed:
                kv_start = fx.Int32(tile_idx * traits.BLOCK_N)
                need = need | (
                    kv_start < qs_i32 + fx.Int32(MFMA16_M - 1) - window_left_i32
                )
            if need:
                lists = to_lists(v_s)
                for step in range_constexpr(2):
                    for i in range_constexpr(n):
                        col = kv_col(tile_idx, lane, step, i)
                        keep = col <= q_row_i32 + delta_i32
                        if windowed:
                            keep = keep & (col >= q_row_i32 - window_left_i32)
                        lists[step][i] = keep.select(
                            fx.Float32(lists[step][i]), neg_inf
                        )
                s_lo, s_hi = to_vecs(lists)
            return s_lo, s_hi

        return _mask_if_needed(v_s, tile_idx)

    def exp2(self, v_s):
        return tuple(
            exp2_wait_state(
                [
                    rocdl.exp2(T.f32, as_mlir_value(Vec(h)[r]))
                    for r in range_constexpr(self.n)
                ]
            )
            for h in v_s
        )

    def dscores(self, p_lists, dp, delta):
        fm = self.ctx.fm_fast
        return tuple(
            [
                ops._fmul(p_lists[h][r], ops._fsub(dp[h][r], delta, fm), fm)
                for r in range_constexpr(self.n)
            ]
            for h in (0, 1)
        )

    def pack_ds(self, ds_lists):
        vals = list(ds_lists[0]) + list(ds_lists[1])
        return mfma_operand_wait_state(
            ops._bf16_trunc_pack_v8(self.traits, vals, elem_dtype=self.ctx.elem_dtype)
        )


class M16DqStore:
    def __init__(self, ctx):
        self.ctx = ctx
        self.traits = ctx.traits
        self.oob = ctx.dq_oob_off
        span = _slab_span_elems(ctx.seqlen_q_v, ctx.stride_dq_seq, ctx.hdim_qk)
        self.rsrc = ops.make_buffer_rsrc(
            ctx.DQ,
            ctx._slab_byte_base(
                ctx.stride_dq_batch,
                ctx.stride_dq_head,
                ctx.stride_dq_seq,
                ctx.q_row_off,
                ctx.q_head_idx,
            ),
            span * fx.Index(self.traits.BF16_BYTES),
        )

    def store(self, v_dq, q_row, lane):
        ctx, traits = self.ctx, self.traits
        n = acc_elems(traits)
        row_base = q_row * ctx.stride_dq_seq_v
        in_row = q_row < ctx.seqlen_q_v
        col_lane = fx.Index(4) * (lane // fx.Index(MFMA16_M))
        for c in range_constexpr(d_chunks16(traits)):
            col = fx.Index(c * MFMA16_M) + col_lane
            vals = [
                fx.Float32(Vec(v_dq[c])[r]).to(ctx.elem_dtype)
                for r in range_constexpr(n)
            ]
            pack = Vec.from_elements(vals, ctx.elem_dtype)
            live = in_row
            off = fx.Index(live.select(row_base + col, self.oob))
            buffer_store(as_mlir_value(pack), self.rsrc, as_mlir_value(fx.Int32(off)))


def make_m16_dq_body(ctx, traits, *, kv_gmem_to_lds):
    lane = ctx.lane
    q_row_in_block = ctx.wave_id * fx.Index(MFMA16_M) + lane % fx.Index(MFMA16_M)
    q_row = ctx.q_start + q_row_in_block

    k_reader = M16DqReader(ctx, region="v", line_stride=traits.SMEM_V_LINE_STRIDE)
    v_reader = M16DqReader(ctx, region="k", line_stride=traits.SMEM_K_LINE_STRIDE)
    q_loader = M16QLoader(ctx, ctx.q_div, ctx.stride_q_seq_v, ctx.q_gmem_elem_offset)
    do_loader = M16QLoader(
        ctx, ctx.do_div, ctx.stride_do_seq_v, ctx.do_gmem_elem_offset
    )
    gemm = M16DqGemmHelper(ctx)
    softmax = M16DqSoftmax(ctx)
    store = M16DqStore(ctx)
    n_chunks = d_chunks16(traits)

    @flyc.jit
    def _m16_body():
        q_packs = q_loader.load_all(q_row_in_block, lane)
        do_packs = do_loader.load_all(q_row_in_block, lane)
        lse2, delta = ctx.load_row_scalars(q_row)

        init_args = [gemm.zero for _ in range_constexpr(n_chunks)]
        loop_results = init_args
        for j, loop_args in range(
            ctx.split_tile(0), ctx.split_t_end, fx.Index(1), init=init_args
        ):
            v_dq = [loop_args[i] for i in range_constexpr(n_chunks)]
            tile_start = ctx.tile_start(j)

            _s_barrier()
            kv_gmem_to_lds.stage_k(tile_start)
            kv_gmem_to_lds.stage_v(tile_start)
            _s_waitcnt(0)
            _sched_barrier(0)
            _s_barrier()

            v_s = gemm.qk(k_reader.load_a_all(lane), q_packs)
            v_s = softmax.scale_and_sub_lse(v_s, ctx.c_sm_scale_log2e, lse2)
            v_s = softmax.seq_pad_mask(v_s, j, lane)
            if const_expr(traits.CAUSAL):
                v_s = softmax.causal_mask(v_s, j, lane, q_row)
            v_dp = gemm.qk(v_reader.load_a_all(lane), do_packs)

            dp_lists = softmax.to_lists(v_dp)
            ds = softmax.dscores(softmax.exp2(v_s), dp_lists, delta)
            ds_pack = softmax.pack_ds(ds)
            v_dq = gemm.pv(ds_pack, k_reader.load_kt(lane), v_dq)

            loop_results = yield v_dq

        v_dq = [loop_results[i] for i in range_constexpr(n_chunks)]
        scale = Vec.from_elements([ctx.c_sm_scale], fx.Float32).broadcast_to(
            acc_elems(traits)
        )
        for c in range_constexpr(n_chunks):
            v_dq[c] = ops._fmul(Vec(v_dq[c]), scale, ctx.fm_fast)
        _s_barrier()
        store.store(v_dq, q_row, lane)

    return _m16_body


def _carried(values):
    if const_expr(isinstance(values, (list, tuple))):
        return list(values)
    return [values]


class BwdDqKernelContext(ParityKernelContext):
    """The shared context plus dQ, dO and the LSE/delta rows; dQ is Q-shaped."""

    def __init__(
        self,
        traits,
        *,
        dq_strides,
        do_strides,
        DQ=None,
        DO=None,
        Delta=None,
        **kwargs,
    ):
        super().__init__(traits, **kwargs)
        self.stride_dq_batch, self.stride_dq_head, self.stride_dq_seq = dq_strides
        self.stride_do_batch, self.stride_do_head, self.stride_do_seq = do_strides
        self.DQ = DQ
        self.DO = DO
        self.Delta = Delta

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        self.stride_dq_seq_v = fx.Index(self.stride_dq_seq)
        self.stride_do_seq_v = fx.Index(self.stride_do_seq)

    def init_descriptors(self, **kwargs):
        super().init_descriptors(**kwargs)
        self.dq_div, self.dq_oob_off = self.q_row_slab(
            self.DQ,
            self.stride_dq_batch,
            self.stride_dq_head,
            self.stride_dq_seq,
            self.hdim_qk,
        )
        self.do_div = self._slab_view(
            self.DO,
            self.stride_do_batch,
            self.stride_do_head,
            self.stride_do_seq,
            self.q_row_off,
            self.q_head_idx,
            self.seqlen_q_v,
            self.hdim_vo,
        )
        self.do_gmem_elem_offset = self.q_start * self.stride_do_seq_v

    def compute_active_guard(self):
        return self.q_start < self.seqlen_q_v

    def init_tile_bounds(self, **kwargs):
        super().init_tile_bounds(**kwargs)
        traits = self.traits
        if const_expr(traits.CAUSAL):
            end_i32 = fx.Int32(
                (self.causal_end_raw_i32 > fx.Int32(0)).select(
                    self.causal_end_raw_i32, fx.Int32(0)
                )
            )
            n = (fx.Index(end_i32) + fx.Index(traits.BLOCK_N - 1)) // fx.Index(
                traits.BLOCK_N
            )
            n = fx.Index((n < self.num_kv_tiles).select(n, self.num_kv_tiles))
        else:
            n = self.num_kv_tiles
        self.max_num_tiles = n
        self.split_t0 = fx.Index(0)
        self.split_t_end = n
        if const_expr(traits.WINDOW):
            self._skip_dead_leading_tiles()

    def _skip_dead_leading_tiles(self):
        traits = self.traits
        first_col_i32 = fx.Int32(self.q_start) - self.window_left_i32
        first_col_i32 = fx.Int32(
            (first_col_i32 > fx.Int32(0)).select(first_col_i32, fx.Int32(0))
        )
        t0 = fx.Index(first_col_i32) // fx.Index(traits.BLOCK_N)
        self.split_t0 = fx.Index((t0 < self.split_t_end).select(t0, self.split_t_end))

    def init_row_inputs(self):
        tokens = fx.Index(self.lse_tokens_i32)
        slab_bytes = tokens * fx.Index(4)
        base_bytes = (
            self.batch_idx * fx.Index(self.num_head_q) + self.q_head_idx
        ) * slab_bytes
        self.row_base = self.q_row_off
        self.row_oob_off = tokens
        self.lse_rsrc = ops.make_buffer_rsrc(self.LSE, base_bytes, slab_bytes)
        self.delta_rsrc = ops.make_buffer_rsrc(self.Delta, base_bytes, slab_bytes)

    def load_row_scalars(self, q_row):
        off = self.row_base + q_row
        off = fx.Index((q_row < self.seqlen_q_v).select(off, self.row_oob_off))
        off_i32 = as_mlir_value(fx.Int32(off))
        lse = buffer_load(self.lse_rsrc, off_i32, vec_width=1, dtype=fx.Float32)
        delta = buffer_load(self.delta_rsrc, off_i32, vec_width=1, dtype=fx.Float32)
        lse2 = ops._fmax(fx.Float32(lse), self.c_neg_floor, self.fm_fast)
        return lse2, fx.Float32(delta)


class BwdDqSoftmaxHelper(ParitySoftmaxHelper):
    def scale_and_sub_lse(self, v_s, qk_scale, lse2):
        s_lo, s_hi = v_s
        acc = self.traits.ACC_ELEMS
        scale_v = Vec.from_elements([fx.Float32(qk_scale)], fx.Float32).broadcast_to(
            acc
        )
        neg_lse2 = ops._fsub(self.c_zero_f, lse2, self.fm_fast)
        neg_v = Vec.from_elements([fx.Float32(neg_lse2)], fx.Float32).broadcast_to(acc)
        lo = fmath.fma(Vec(s_lo), scale_v, neg_v, fastmath=self.fm_fast)
        hi = fmath.fma(Vec(s_hi), scale_v, neg_v, fastmath=self.fm_fast)
        return as_mlir_value(lo), as_mlir_value(hi)

    def cast_p(self, v_p, tile_idx=None):
        p_lo_packs, p_hi_packs = ops.DualwaveSoftmaxHelper.cast_p(self, v_p)
        return (
            [mfma_operand_wait_state(p) for p in p_lo_packs],
            [mfma_operand_wait_state(p) for p in p_hi_packs],
        )

    def exp2(self, v_s, start, length):
        if const_expr(start == 0):
            s_lo = [Vec(v_s[0])[r] for r in range_constexpr(16)]
            lo_partial = exp2_wait_state(
                [rocdl.exp2(T.f32, as_mlir_value(s_lo[r])) for r in range_constexpr(16)]
            )
            return Vec.from_elements(lo_partial, fx.Float32).ir_value(), v_s[1]
        lo_partial = [Vec(v_s[0])[r] for r in range_constexpr(16)]
        hi_full = exp2_wait_state(
            [
                rocdl.exp2(T.f32, as_mlir_value(Vec(v_s[1])[r]))
                for r in range_constexpr(16)
            ]
        )
        return lo_partial, hi_full


class BwdSecondaryQLoader(ParityQLoader):
    def __init__(self, ctx, div, stride_seq_v, gmem_elem_offset, hdim):
        super().__init__(ctx)
        self.q_div = div
        self.stride_q_n_v = stride_seq_v
        self.q_gmem_elem_offset = gmem_elem_offset
        self.hdim_qk = hdim


class BwdDqKvGmemToLdsLoader(ParityKvGmemToLdsLoader):
    def _stage(self, tile_start, src_div, stride_seq_v, dma_m0, buf_id):
        self.stride_kv_n_v = stride_seq_v
        self._async_load_kv_linear(
            dma_m0,
            buf_id,
            src_div,
            self.kv_gmem_elem_offset,
            tile_start * stride_seq_v,
            self.NUM_DMA_K,
        )

    def stage_k(self, tile_start):
        self._stage(
            tile_start, self.k_div, self.stride_k_seq_v, self.ctx_ref.v_dma_m0, 0
        )

    def stage_v(self, tile_start):
        self._stage(
            tile_start, self.v_div, self.stride_v_seq_v, self.ctx_ref.k_dma_m0, 0
        )


class BwdDqKvLdsToVgprLoader(ParityKvLdsToVgprLoader):
    def __init__(self, ctx, *, v_layout):
        super().__init__(ctx)
        self.v_layout = bool(v_layout)

    def _lds_pack(self, elem_idx, scope_name):
        traits = self.traits
        ptr = ops._lds_byte_gep(self.lds_kv_base_ptr, elem_idx * traits.BF16_BYTES)
        return llvm.LoadOp(
            self.kv_mfma_pack_type,
            ptr,
            alignment=16,
            alias_scopes=ops._dualwave_lds_alias_scopes(scope_name),
            noalias_scopes=ops._dualwave_lds_noalias_scopes(
                scope_name, traits.LDS_SCOPE_NAMES
            ),
        ).result

    def _read_k_packs(self, buf_id, urk_base):
        if const_expr(not self.v_layout):
            return super()._read_k_packs(buf_id, urk_base)
        full = self.traits
        self.traits = replace(
            full,
            SMEM_K_LINE_STRIDE=full.SMEM_V_LINE_STRIDE,
            K_LDS_TO_REG_KSTEP_OUTER_STRIDE=full.SMEM_N_RPT * full.SMEM_V_LINE_STRIDE,
        )
        try:
            traits = self.traits
            scope = ops._dualwave_lds_scope("v", buf_id)
            base = ops._v_buf_base(traits, buf_id) + _k_read_base(
                traits, self.lane_mod_32, self.lane_div_32
            )
            k_lo = [None] * traits.K_STEPS_QK
            k_hi = [None] * traits.K_STEPS_QK
            for ks in range_constexpr(traits.K_STEPS_QK):
                idx = base + _ks_offset(traits, ks)
                k_lo[ks] = self._lds_pack(idx, scope)
                k_hi[ks] = self._lds_pack(
                    idx + traits.K_LDS_TO_REG_N_STRIP_STRIDE, scope
                )
            return k_lo, k_hi
        finally:
            self.traits = full


class BwdDqStoreHelper(ParityStoreHelper):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.o_div = ctx.dq_div
        self.stride_o_seq_v = ctx.stride_dq_seq_v
        self.hdim_vo = ctx.hdim_qk


def build_fmha_bwd_dq_module(meta):
    traits = make_dq_traits(meta)
    ACC_ELEMS = traits.ACC_ELEMS
    M16 = traits.MFMA_N == 16
    _cache_tag = traits_cache_key(traits)
    _lds_elem_dtype = ops._dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def fmha_bwd_dq_gfx950_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        DO: fx.Pointer,
        DQ: fx.Pointer,
        LSE: fx.Pointer,
        Delta: fx.Pointer,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
    ):
        ctx = BwdDqKernelContext(
            traits,
            strides=(
                stride_q_batch,
                stride_q_head,
                stride_q_seq,
                stride_k_batch,
                stride_k_head,
                stride_k_seq,
                stride_v_batch,
                stride_v_head,
                stride_v_seq,
            ),
            dq_strides=(stride_dq_batch, stride_dq_head, stride_dq_seq),
            do_strides=(stride_do_batch, stride_do_head, stride_do_seq),
            sm_scale=sm_scale,
            num_head_q=num_head_q,
            num_head_k=num_head_k,
            hdim_qk=hdim_qk,
            hdim_vo=hdim_vo,
            window_left=window_left,
            window_right=window_right,
            Q=wire_view(Q),
            K=wire_view(K),
            V=wire_view(V),
            DQ=wire_view(DQ),
            DO=wire_view(DO),
            Delta=wire_view(Delta),
            seq_len=max_seqlen_q,
            seq_len_kv=max_seqlen_k,
            stride_q_n=stride_q_seq,
            stride_kv_n=stride_k_seq,
            head_dim_runtime=hdim_qk,
            LSE=wire_view(LSE),
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        ctx.init_sequence_lengths()
        ctx.init_descriptors()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_tile_bounds()
        ctx.init_active_guard()
        ctx.init_lds_read_bases()
        ctx.init_dma_m0_tables()
        ctx.init_q_row()
        ctx.init_row_inputs()

        kv_gmem_to_lds = BwdDqKvGmemToLdsLoader(ctx)
        k_reader = BwdDqKvLdsToVgprLoader(ctx, v_layout=True)
        v_reader = BwdDqKvLdsToVgprLoader(ctx, v_layout=False)
        q_loader = ParityQLoader(ctx)
        do_loader = BwdSecondaryQLoader(
            ctx, ctx.do_div, ctx.stride_do_seq_v, ctx.do_gmem_elem_offset, hdim_vo
        )
        gemm_helper = ParityGemmHelper(ctx)
        softmax_helper = BwdDqSoftmaxHelper(ctx)
        output_store = BwdDqStoreHelper(ctx)

        def _body():
            q_all_bf16 = q_loader.load_all()
            do_all_bf16 = do_loader.load_all()
            lse2, delta = ctx.load_row_scalars(ctx.q_row)

            init_args = [ctx.c_zero_v16f32 for _ in range_constexpr(traits.D_CHUNKS)]
            loop_results = init_args
            for j, loop_args in range(
                ctx.split_tile(0), ctx.split_t_end, fx.Index(1), init=init_args
            ):
                v_dq = _carried(loop_args)
                tile_start = ctx.tile_start(j)

                _s_barrier()
                kv_gmem_to_lds.stage_k(tile_start)
                kv_gmem_to_lds.stage_v(tile_start)
                _s_waitcnt(0)
                _sched_barrier(0)
                _s_barrier()

                v_s = gemm_helper.qk(k_reader.load_k(0), q_all_bf16)
                v_s = softmax_helper.scale_and_sub_lse(v_s, ctx.c_sm_scale_log2e, lse2)
                v_s = softmax_helper.seq_pad_mask_if_needed(v_s, j)
                if const_expr(traits.CAUSAL):
                    v_s = softmax_helper.causal_mask_prologue_if_needed(
                        v_s, j, kv_end_tile=j + fx.Index(1)
                    )

                v_dp = gemm_helper.qk(v_reader.load_k(0), do_all_bf16)

                v_p = softmax_helper.exp2(v_s, 0, ACC_ELEMS)
                v_p = softmax_helper.exp2(v_p, ACC_ELEMS, ACC_ELEMS)

                dp_lo, dp_hi = softmax_helper.v_s_vec_to_lists(v_dp)
                ds_lo = [None] * ACC_ELEMS
                ds_hi = [None] * ACC_ELEMS
                for r in range_constexpr(ACC_ELEMS):
                    ds_lo[r] = ops._fmul(
                        v_p[0][r], ops._fsub(dp_lo[r], delta, ctx.fm_fast), ctx.fm_fast
                    )
                    ds_hi[r] = ops._fmul(
                        v_p[1][r], ops._fsub(dp_hi[r], delta, ctx.fm_fast), ctx.fm_fast
                    )

                v_ds = softmax_helper.cast_p((ds_lo, ds_hi))

                v_dq = gemm_helper.pv(v_ds, k_reader.load_v(0), v_dq)

                loop_results = yield v_dq

            v_dq = _carried(loop_results)
            softmax_helper.scale_o(v_dq, ctx.c_sm_scale)
            _s_barrier()
            output_store.store_final_o(v_dq, ctx.q_row)

        if const_expr(M16):
            _body = make_m16_dq_body(ctx, traits, kv_gmem_to_lds=kv_gmem_to_lds)

        active = ctx.active

        def _run_body_if_active():
            if active:
                _body()

        _run_body_if_active()

    @flyc.jit
    def launch_fmha_bwd_dq_gfx950(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        DO: fx.Pointer,
        DQ: fx.Pointer,
        LSE: fx.Pointer,
        Delta: fx.Pointer,
        batch_size: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _cache_tag
        num_q_blocks = (fx.Index(max_seqlen_q) + traits.BLOCK_M - 1) // traits.BLOCK_M
        fmha_bwd_dq_gfx950_kernel(
            Q,
            K,
            V,
            DO,
            DQ,
            LSE,
            Delta,
            max_seqlen_q,
            max_seqlen_k,
            window_left,
            window_right,
            num_head_q,
            num_head_k,
            hdim_qk,
            hdim_vo,
            sm_scale,
            stride_q_batch,
            stride_q_head,
            stride_q_seq,
            stride_k_batch,
            stride_k_head,
            stride_k_seq,
            stride_v_batch,
            stride_v_head,
            stride_v_seq,
            stride_do_batch,
            stride_do_head,
            stride_do_seq,
            stride_dq_batch,
            stride_dq_head,
            stride_dq_seq,
            value_attrs=common.kernel_attrs(traits),
        ).launch(
            grid=(num_head_q, num_q_blocks, fx.Index(batch_size)),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    def _args(
        Q,
        K,
        V,
        DO,
        DQ,
        LSE,
        Delta,
        batch_size,
        seqlen_q,
        *,
        seqlen_k,
        scale,
        window,
        stream=None,
    ):
        return (
            wire_ptr(Q),
            wire_ptr(K),
            wire_ptr(V),
            wire_ptr(DO),
            wire_ptr(DQ),
            wire_ptr(LSE, fx.Float32),
            wire_ptr(Delta, fx.Float32),
            int(batch_size),
            int(seqlen_q),
            int(seqlen_k),
            *common.window_args(traits, window),
            *common.shape_args(Q, K, V),
            float(scale),
            *common.strides_of(Q, K, V, DO, DQ),
        ), stream

    return common.make_launcher(launch_fmha_bwd_dq_gfx950, _args)


def build_split_bwd_dq(**kwargs):
    meta = FmhaInputMetadata(**kwargs)
    return build_fmha_bwd_dq_module(meta)
