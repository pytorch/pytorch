# mypy: allow-untyped-defs
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Split-backward dK/dV kernel: one CTA per (KV head, KV block) holds K and V
in registers and streams Q and dO through LDS for every query head it serves.
"""

from dataclasses import dataclass, fields

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly, llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from . import flex_attn_split_bwd_common as common, flex_attn_split_bwd_ops as ops
from .flex_attn_fwd_dualwave_gfx950 import _waitcnt_vm_n
from .flex_attn_split_bwd_common import (
    _ds_read_tr16_b64_imm,
    _ds_read_tr_v4f16_imm,
    _score_column_runs,
    _slab_span_elems,
    _v_imm_lo,
    exp2_wait_state,
    FmhaInputMetadata,
    make_traits,
    mfma_operand_wait_state,
    ParityDualwaveTraits,
    ParityGemmHelper,
    ParityKernelContext,
    ParityKvGmemToLdsLoader,
    ParitySoftmaxHelper,
    traits_cache_key,
    wire_ptr,
    wire_view,
)
from .flex_attn_split_bwd_ops import (
    a16_chunk_offset,
    a16_read_base,
    lds_elem,
    MFMA16_M,
    tok_off,
    tok_off_dyn,
)
from .mxfp8_buffer_ops import buffer_load, buffer_store


# (num_waves, waves_per_eu, mfma_rows, block_q) per tile width.
_GEOMETRY = {
    32: (4, 2, 32, 64),
    64: (4, 1, 32, 64),
    96: (4, 1, 16, 64),
    128: (4, 1, 16, 64),
    160: (4, 1, 32, 64),
    192: (4, 1, 16, 32),
    224: (4, 1, 32, 64),
    256: (4, 1, 16, 32),
    384: (4, 1, 16, 32),
    512: (4, 1, 16, 32),
}

# Widths that read row packs as needed rather than all at once, to fit in registers.
_TIGHT_REGISTERS = (32, 384, 512)


def _geometry_for(block_dmodel, causal):
    geometry = _GEOMETRY[block_dmodel]
    if block_dmodel == 64:
        geometry = (4, 1, 16, 64) if causal else (4, 2, 32, 64)
    return geometry + (block_dmodel in (*_TIGHT_REGISTERS, 224),)


@dataclass(frozen=True)
class BwdDkDvTraits(ParityDualwaveTraits):
    NUM_STREAM_BUFFERS: int = 2
    MFMA_ROWS: int = 32

    @property
    def BLOCK_KV(self):
        return self.BLOCK_M

    @property
    def BLOCK_Q(self):
        return self.BLOCK_N

    @property
    def STREAM_LINE_STRIDE(self):
        return self.SMEM_V_LINE_STRIDE

    @property
    def STREAM_TILE_ELEMS(self):
        return self.SMEM_V_TILE_ELEMS

    @property
    def LDS_STREAM_TOTAL_SIZE(self):
        return 2 * self.NUM_STREAM_BUFFERS * self.STREAM_TILE_ELEMS


def make_dkdv_traits(meta):
    width = common.tile_width_for(meta.head_dim)
    waves, waves_per_eu, rows, block_q, tight = _geometry_for(width, meta.causal)
    granule = 32 if width % 64 else 64
    # Double-buffer Q/dO when two slots of each fit in the device's LDS.
    slot_bytes = block_q // (512 // granule) * (width // granule) * 544 * 2
    base = make_traits(
        num_heads=meta.num_heads,
        num_kv_heads=meta.num_kv_heads,
        head_dim=width,
        num_waves=waves,
        block_m=rows * waves,
        block_n=block_q,
        granule=granule,
        causal=meta.causal,
        window=meta.window,
        dtype_str=meta.dtype_str,
        waves_per_eu=waves_per_eu,
    )
    traits = BwdDkDvTraits(
        **{f.name: getattr(base, f.name) for f in fields(base)},
        NUM_STREAM_BUFFERS=2 if 4 * slot_bytes <= meta.lds_bytes else 1,
        MFMA_ROWS=rows,
    )
    return traits, tight


MFMA16_K = 32
LANE_K = MFMA16_K * MFMA16_M // 64
ACC16 = MFMA16_M * MFMA16_M // 64


def q_groups(traits):
    return traits.BLOCK_Q // MFMA16_K


def score_subs(traits):
    return traits.BLOCK_Q // MFMA16_M


def k_steps32(traits):
    return traits.HEAD_DIM // MFMA16_K


def d_chunks16(traits):
    return traits.HEAD_DIM // MFMA16_M


def _lds_pack_read(traits, lds_ptr, elem_idx, scope_name, pack_type):
    ptr = ops._lds_byte_gep(lds_ptr, elem_idx * traits.BF16_BYTES)
    return llvm.LoadOp(
        pack_type,
        ptr,
        alignment=16,
        alias_scopes=ops._dualwave_lds_alias_scopes(scope_name),
        noalias_scopes=ops._dualwave_lds_noalias_scopes(
            scope_name, traits.LDS_SCOPE_NAMES
        ),
    ).result


class M16StreamReader(ops.DualwaveKernelContext):
    def init16(self):
        traits = self.traits
        lane = self.lane
        lm = lane % fx.Index(MFMA16_M)
        perm_row = fx.Index(8) * (lm // fx.Index(4)) + (lm % fx.Index(4))
        self.row_base16 = [
            tok_off_dyn(
                traits, fx.Index((sub // 2) * MFMA16_K + (sub % 2) * 4) + perm_row
            )
            + (lane // fx.Index(MFMA16_M)) * fx.Index(LANE_K)
            for sub in range(score_subs(traits))
        ]
        self.col_base16 = a16_read_base(traits, lane, LANE_K)

    def read_row_pack(self, tile_base, scope_name, sub, ks, hdim):
        traits = self.traits
        idx = tile_base + self.row_base16[sub] + lds_elem(traits, 0, ks * MFMA16_K)
        pack = _lds_pack_read(
            traits, self.lds_kv_base_ptr, idx, scope_name, self.kv_mfma_pack_type
        )
        return pack

    def read_column_chunk(self, tile_base, scope_name, c):
        traits = self.traits
        addr = fx.Int32(
            (tile_base + self.col_base16) * traits.BF16_BYTES + self.lds_kv_base_idx
        )
        chunk_off = a16_chunk_offset(traits, c) * traits.BF16_BYTES
        pair = tok_off(traits, 4) * traits.BF16_BYTES
        packs = []
        for g in range_constexpr(q_groups(traits)):
            base = chunk_off + tok_off(traits, g * MFMA16_K) * traits.BF16_BYTES
            a = _ds_read_tr16_b64_imm(
                self.v_lds_read_vec4_type,
                addr,
                base,
                scope_name=scope_name,
                scope_names=traits.LDS_SCOPE_NAMES,
            )
            b = _ds_read_tr16_b64_imm(
                self.v_lds_read_vec4_type,
                addr,
                base + pair,
                scope_name=scope_name,
                scope_names=traits.LDS_SCOPE_NAMES,
            )
            packs.append(Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
        return packs


class M16ResidentLoader(ops.DualwaveKernelContext):
    def load_resident(self, which, hdim):
        traits = self.traits
        rsrc = self.k_res_rsrc if which == "k" else self.v_res_rsrc
        elem_base = self.k_res_elem_base if which == "k" else self.v_res_elem_base
        stride_seq = self.stride_k_seq_v if which == "k" else self.stride_v_seq_v
        row = elem_base + self.kv_row_in_block * stride_seq
        lane_d = (self.lane // fx.Index(MFMA16_M)) * fx.Index(LANE_K)
        packs = []
        for ks in range_constexpr(k_steps32(traits)):
            col = fx.Index(ks * MFMA16_K) + lane_d
            pack = buffer_load(
                rsrc,
                as_mlir_value(fx.Int32(row + col)),
                vec_width=LANE_K,
                dtype=self.elem_dtype,
            )
            packs.append(pack)
        return packs


class M16GemmHelper(ops.DualwaveKernelContext):
    def init16(self):
        self.mma16 = fx.make_mma_atom(
            fx.rocdl.MFMA(MFMA16_M, MFMA16_M, MFMA16_K, self.elem_dtype)
        )
        self.acc16_ty = Vec.make_type(ACC16, fx.Float32)
        self.zero16 = Vec.filled(ACC16, 0.0, fx.Float32).ir_value()

    def _mfma(self, a, b, c):
        return fly.mma_atom_call_ssa([self.acc16_ty], self.mma16, a, b, c)

    def contract_d(self, read_a, b_packs):
        acc = self.zero16
        for ks in range_constexpr(k_steps32(self.traits)):
            acc = self._mfma(read_a(ks), b_packs[ks], acc)
        return acc

    def contract_q(self, a_packs, b_packs, acc):
        for g in range_constexpr(q_groups(self.traits)):
            acc = self._mfma(a_packs[g], b_packs[g], acc)
        return acc


class DkDvSoftmaxMath:
    def _mask_if_clipped(self, p_list, tile_idx, rel0, rows, offsets):
        traits = self.traits
        ctx = self.ctx_ref
        lo_i32 = ctx.causal_lo_i32
        left_i32 = ctx.window_left_i32
        kv_lo = fx.Int32(ctx.kv_start) + fx.Int32(ctx.wave_kv_offset_uni)
        tile_q0 = fx.Int32(tile_idx * fx.Index(traits.BLOCK_Q))
        need = tile_q0 < kv_lo + fx.Int32(rows - 1) + lo_i32
        if const_expr(traits.WINDOW):
            need = need | (tile_q0 + fx.Int32(traits.BLOCK_Q - 1) > kv_lo + left_i32)
        zero_f = self.c_zero_f
        window = traits.WINDOW

        def _apply(vals):
            out = list(vals)
            for r in range_constexpr(self.ACC_ELEMS):
                rel = rel0 + fx.Int32(offsets[r])
                keep = rel >= lo_i32
                if const_expr(window):
                    keep = keep & (rel <= left_i32)
                out[r] = keep.select(fx.Float32(vals[r]), zero_f)
            return out

        @flyc.jit
        def _mask_if_needed(p_vec):
            out = p_vec
            if need:
                out = Vec.from_elements(
                    [
                        as_mlir_value(v)
                        for v in _apply(
                            [Vec(p_vec)[r] for r in range_constexpr(self.ACC_ELEMS)]
                        )
                    ],
                    fx.Float32,
                ).ir_value()
            return out

        packed = Vec.from_elements(
            [as_mlir_value(fx.Float32(v)) for v in p_list], fx.Float32
        ).ir_value()
        return [
            Vec(_mask_if_needed(packed))[r] for r in range_constexpr(self.ACC_ELEMS)
        ]

    ACC_ELEMS = 16

    def probabilities(self, v_s, neg_lse2):
        values = [Vec(v_s)[r] for r in range_constexpr(self.ACC_ELEMS)]
        scale = self.ctx_ref.c_sm_scale_log2e
        fm = self.fm_fast
        scaled = [
            ops._fmul(values[r], scale, fm) for r in range_constexpr(self.ACC_ELEMS)
        ]
        return exp2_wait_state(
            [
                rocdl.exp2(T.f32, as_mlir_value(ops._fadd(scaled[r], neg_lse2[r], fm)))
                for r in range_constexpr(self.ACC_ELEMS)
            ]
        )

    def dscores(self, p_list, v_dp, delta):
        values = [Vec(v_dp)[r] for r in range_constexpr(self.ACC_ELEMS)]
        return [
            ops._fmul(
                p_list[r], ops._fsub(values[r], delta[r], self.fm_fast), self.fm_fast
            )
            for r in range_constexpr(self.ACC_ELEMS)
        ]


class M16SoftmaxHelper(DkDvSoftmaxMath, ops.DualwaveKernelContext):
    ACC_ELEMS = ACC16

    def load_row_values(self, rsrc, tile_base, sub, scale):
        row0 = fx.Int32(
            tile_base + fx.Index((sub // 2) * MFMA16_K + (sub % 2) * 4)
        ) + fx.Int32(self.lane // fx.Index(MFMA16_M)) * fx.Int32(8)
        span = buffer_load(rsrc, as_mlir_value(row0), vec_width=ACC16, dtype=fx.Float32)
        vec = Vec(span, (ACC16,), fx.Float32)
        return [ops._fmul(vec[i], scale, self.fm_fast) for i in range_constexpr(ACC16)]

    def mask_if_clipped(self, p_list, tile_idx, sub):
        traits = self.traits
        ctx = self.ctx_ref
        sub_base = (sub // 2) * MFMA16_K + (sub % 2) * 4
        rel0 = (
            fx.Int32(tile_idx * fx.Index(traits.BLOCK_Q))
            + fx.Int32(sub_base)
            + fx.Int32(self.lane // fx.Index(MFMA16_M)) * fx.Int32(8)
            - fx.Int32(ctx.kv_row)
        )
        return self._mask_if_clipped(
            p_list, tile_idx, rel0, MFMA16_M, tuple(range(ACC16))
        )

    def pack_group(self, lo, hi):
        return mfma_operand_wait_state(
            ops._bf16_trunc_pack_v8(
                self.traits, list(lo) + list(hi), elem_dtype=self.elem_dtype
            )
        )


class M16StoreHelper(ops.DualwaveKernelContext):
    def _pack_out(self, vals):
        if const_expr(self.traits.DTYPE_STR == "bf16"):
            return Vec.from_elements(
                [
                    rocdl.cvt_pk_bf16_f32(vals[2 * j], vals[2 * j + 1])
                    for j in range_constexpr(ACC16 // 2)
                ],
                fx.Int32,
            ).bitcast(self.elem_dtype)
        return Vec.from_elements(
            [fx.Float32(vals[i]).to(self.elem_dtype) for i in range_constexpr(ACC16)],
            self.elem_dtype,
        )

    def store_accs(self, accs, which, hdim):
        rsrc = self.dk_rsrc if which == "dk" else self.dv_rsrc
        stride_seq = self.stride_dk_seq_v if which == "dk" else self.stride_dv_seq_v
        base = self.kv_row * stride_seq
        lane_d = (self.lane // fx.Index(MFMA16_M)) * fx.Index(ACC16)
        for c in range_constexpr(len(accs)):
            col = fx.Index(c * MFMA16_M) + lane_d
            vals = [Vec(accs[c])[i] for i in range_constexpr(ACC16)]
            packed = self._pack_out(vals)
            off = base + col
            buffer_store(packed.ir_value(), rsrc, as_mlir_value(fx.Int32(off)))


class M16TileBody(ops.DualwaveKernelContext):
    def __init__(self, ctx, *, stream, reader, gemm, softmax, hdim_qk, hdim_vo):
        super().__init__(ctx)
        self.stream = stream
        self.reader = reader
        self.gemm = gemm
        self.softmax = softmax
        self.hdim_qk = hdim_qk
        self.hdim_vo = hdim_vo

    def run(self, tile_idx, buf_id, prefetch_idx, v_k, v_v, dv, dk):
        traits = self.traits
        ctx = self.ctx_ref
        q_base = ctx.q_buf_base(buf_id)
        do_base = ctx.do_buf_base(buf_id)
        q_scope = ops._dualwave_lds_scope("k", buf_id)
        do_scope = ops._dualwave_lds_scope("v", buf_id)
        tile_base = tile_idx * fx.Index(traits.BLOCK_Q)
        n_grp = q_groups(traits)
        n_chunks = d_chunks16(traits)

        _waitcnt_vm_n((traits.NUM_STREAM_BUFFERS - 1) * (ctx.NUM_DMA_K + ctx.NUM_DMA_V))
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

        p_packs = [None] * n_grp
        ds_packs = [None] * n_grp
        for g in range_constexpr(n_grp):
            p_half = []
            ds_half = []
            for h in range_constexpr(2):
                sub = 2 * g + h
                s = self.gemm.contract_d(
                    self._reader_for(q_base, q_scope, sub, self.hdim_qk), v_k
                )
                dp = self.gemm.contract_d(
                    self._reader_for(do_base, do_scope, sub, self.hdim_vo), v_v
                )
                neg_lse2 = self.softmax.load_row_values(
                    ctx.lse_rsrc, tile_base, sub, ctx.c_neg_log2e
                )
                p_list = self.softmax.probabilities(s, neg_lse2)
                if const_expr(traits.CAUSAL):
                    p_list = self.softmax.mask_if_clipped(p_list, tile_idx, sub)
                delta = self.softmax.load_row_values(
                    ctx.delta_rsrc, tile_base, sub, ctx.c_one_f
                )
                ds_half.append(self.softmax.dscores(p_list, dp, delta))
                p_half.append(p_list)
            p_packs[g] = self.softmax.pack_group(p_half[0], p_half[1])
            ds_packs[g] = self.softmax.pack_group(ds_half[0], ds_half[1])

        for c in range_constexpr(n_chunks):
            dv[c] = self.gemm.contract_q(
                self.reader.read_column_chunk(do_base, do_scope, c), p_packs, dv[c]
            )
        for c in range_constexpr(n_chunks):
            dk[c] = self.gemm.contract_q(
                self.reader.read_column_chunk(q_base, q_scope, c), ds_packs, dk[c]
            )

        rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        self.stream.stage_q_tile(prefetch_idx, buf_id)
        self.stream.stage_do_tile(prefetch_idx, buf_id)
        return dv, dk

    def _reader_for(self, base, scope, sub, hdim):
        return lambda ks: self.reader.read_row_pack(base, scope, sub, ks, hdim)


_s_barrier = rocdl.s_barrier
_s_waitcnt = rocdl.s_waitcnt
_sched_barrier = rocdl.sched_barrier

_ROW_RUNS = _score_column_runs()

_ROW_THRESHOLDS = [0] * 16
for _e0, _c0, _w in _ROW_RUNS:
    for _j in range(_w):
        _ROW_THRESHOLDS[_e0 + _j] = _c0 + _j


def _stream_row_read_base(traits, lane_mod_32, lane_div_32):
    return (
        (lane_mod_32 % traits.SMEM_N_RPT) * traits.STREAM_LINE_STRIDE
        + (lane_mod_32 // traits.SMEM_N_RPT) * traits.D_128B_SIZE
        + lane_div_32 * traits.VEC_KV
    )


def _stream_ks_offset(traits, ks):
    per_band = traits.K_STEPS_PER_BAND
    return (ks // per_band) * (traits.SMEM_N_RPT * traits.STREAM_LINE_STRIDE) + (
        ks % per_band
    ) * traits.K_STEP_QK


class BwdDkDvKernelContext(ParityKernelContext):
    def __init__(self, traits, *, strides, DO, DK, DV, Delta, **kwargs):
        super().__init__(traits, strides=strides[:9], **kwargs)
        self.DO = DO
        self.DK = DK
        self.DV = DV
        self.Delta = Delta
        (
            self.stride_do_batch,
            self.stride_do_head,
            self.stride_do_seq,
            self.stride_dk_batch,
            self.stride_dk_head,
            self.stride_dk_seq,
            self.stride_dv_batch,
            self.stride_dv_head,
            self.stride_dv_seq,
        ) = strides[9:18]

    def init_lds(self, shared_storage):
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.stream.ptr))
        self.lds_kv_base_ptr = lds.stream.ptr.llvm_ptr

    def q_buf_base(self, buf_id):
        return (2 * buf_id) * self.traits.STREAM_TILE_ELEMS

    def do_buf_base(self, buf_id):
        return (2 * buf_id + 1) * self.traits.STREAM_TILE_ELEMS

    def k_dma_base(self, buf_id, d):
        return self._dma_m0(self.q_buf_base(buf_id), self.traits.STREAM_LINE_STRIDE, d)

    def v_dma_base(self, buf_id, d):
        return self._dma_m0(self.do_buf_base(buf_id), self.traits.STREAM_LINE_STRIDE, d)

    def dma_m0_table(self, base_fn, count):
        return tuple(
            tuple(base_fn(buf, d) for d in range(count))
            for buf in range(self.traits.NUM_STREAM_BUFFERS)
        )

    def init_lds_read_bases(self):
        super().init_lds_read_bases()
        traits = self.traits
        self.stream_row_read_base = _stream_row_read_base(
            traits, self.lane_mod_32, self.lane_div_32
        )
        col_base = self.v_lds_read_base_per_lane
        self.stream_col_read_base = col_base

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        self.stride_do_seq_v = fx.Index(self.stride_do_seq)
        self.stride_dk_seq_v = fx.Index(self.stride_dk_seq)
        self.stride_dv_seq_v = fx.Index(self.stride_dv_seq)

    def init_thread_mapping(self):
        super().init_thread_mapping()
        traits = self.traits
        self.kv_start = self.q_start
        self.gqa_q_head_base = self.q_head_idx
        self.wave_kv_offset = self.wave_q_offset
        rows_uni = traits.ROWS_PER_WAVE
        self.wave_kv_offset_uni = self.wave_id_uni * rows_uni

    def compute_active_guard(self):
        return self.kv_start < self.seqlen_kv_v

    def init_kv_row(self):
        self.init_q_row()
        rows = self.traits.MFMA_ROWS
        if const_expr(rows == 32):
            self.kv_row_in_block = self.q_row_in_block
        else:
            self.kv_row_in_block = self.wave_kv_offset + self.lane % fx.Index(rows)
        self.kv_row = self.kv_start + self.kv_row_in_block

    def init_descriptors(self):
        self._init_descriptor_state()
        self.q_gmem_elem_offset = fx.Index(0)
        self.kv_gmem_elem_offset = fx.Index(0)

        self._init_q_head_invariants()
        self.bind_q_head(fx.Index(0))

        self.k_res_div = self._slab_view(
            self.K,
            self.stride_k_batch,
            self.stride_k_head,
            self.stride_k_seq,
            self.kv_row_off,
            self.kv_head_idx,
            self.seqlen_kv_v,
            self.hdim_qk,
            batch_idx=self.kv_batch_idx,
        )
        self.v_res_div = self._slab_view(
            self.V,
            self.stride_v_batch,
            self.stride_v_head,
            self.stride_v_seq,
            self.kv_row_off,
            self.kv_head_idx,
            self.seqlen_kv_v,
            self.hdim_vo,
            batch_idx=self.kv_batch_idx,
        )
        self.dk_div = self._slab_view(
            self.DK,
            self.stride_dk_batch,
            self.stride_dk_head,
            self.stride_dk_seq,
            self.kv_row_off,
            self.kv_head_idx,
            self.seqlen_kv_v,
            self.hdim_qk,
            batch_idx=self.kv_batch_idx,
        )
        self.dv_div = self._slab_view(
            self.DV,
            self.stride_dv_batch,
            self.stride_dv_head,
            self.stride_dv_seq,
            self.kv_row_off,
            self.kv_head_idx,
            self.seqlen_kv_v,
            self.hdim_vo,
            batch_idx=self.kv_batch_idx,
        )

        self.k_res_rsrc = self._slab_rsrc(
            self.K,
            self.stride_k_batch,
            self.stride_k_head,
            self.stride_k_seq,
            self.hdim_qk,
        )
        self.v_res_rsrc = self._slab_rsrc(
            self.V,
            self.stride_v_batch,
            self.stride_v_head,
            self.stride_v_seq,
            self.hdim_vo,
        )
        self.dk_rsrc = self._slab_rsrc(
            self.DK,
            self.stride_dk_batch,
            self.stride_dk_head,
            self.stride_dk_seq,
            self.hdim_qk,
        )
        self.dv_rsrc = self._slab_rsrc(
            self.DV,
            self.stride_dv_batch,
            self.stride_dv_head,
            self.stride_dv_seq,
            self.hdim_vo,
        )

        self.k_res_elem_base = self.kv_start * self.stride_k_seq_v
        self.v_res_elem_base = self.kv_start * self.stride_v_seq_v

    def _init_q_head_invariants(self):
        elem_bytes = fx.Index(self.traits.BF16_BYTES)
        head0 = self.gqa_q_head_base

        self.q_slab_span_elems = _slab_span_elems(
            self.seqlen_q_v, self.stride_q_seq, self.hdim_qk
        )
        self.q_slab_base0_bytes = self._slab_byte_base(
            self.stride_q_batch,
            self.stride_q_head,
            self.stride_q_seq,
            self.q_row_off,
            head0,
        )
        self.q_head_step_bytes = fx.Index(self.stride_q_head) * elem_bytes

        self.do_slab_span_elems = _slab_span_elems(
            self.seqlen_q_v, self.stride_do_seq, self.hdim_vo
        )
        self.do_slab_base0_bytes = self._slab_byte_base(
            self.stride_do_batch,
            self.stride_do_head,
            self.stride_do_seq,
            self.q_row_off,
            head0,
        )
        self.do_head_step_bytes = fx.Index(self.stride_do_head) * elem_bytes

        tokens = fx.Index(self.lse_tokens_i32)
        lse_base = (
            self.batch_idx * fx.Index(self.num_head_q) + head0
        ) * tokens + self.q_row_off
        self.lse_base0_bytes = lse_base * fx.Index(4)
        self.lse_head_step_bytes = tokens * fx.Index(4)
        self.lse_row_span_bytes = self.seqlen_q_v * fx.Index(4)

    def bind_q_head(self, g):
        self.q_head_idx = self.gqa_q_head_base + g

        self.k_div = self._slab_view_at(
            self.Q,
            self.q_slab_base0_bytes + g * self.q_head_step_bytes,
            self.q_slab_span_elems,
        )
        self.v_div = self._slab_view_at(
            self.DO,
            self.do_slab_base0_bytes + g * self.do_head_step_bytes,
            self.do_slab_span_elems,
        )
        self.q_div = self.k_div

        base_bytes = self.lse_base0_bytes + g * self.lse_head_step_bytes
        self.lse_rsrc = ops.make_buffer_rsrc(
            self.LSE, base_bytes, self.lse_row_span_bytes
        )
        self.delta_rsrc = ops.make_buffer_rsrc(
            self.Delta, base_bytes, self.lse_row_span_bytes
        )

    def _slab_rsrc(self, tensor, s0, s1, s2, hdim):
        span_bytes = _slab_span_elems(self.seqlen_kv_v, s2, hdim) * fx.Index(
            self.traits.BF16_BYTES
        )
        return ops.make_buffer_rsrc(
            tensor,
            self._slab_byte_base(
                s0,
                s1,
                s2,
                self.kv_row_off,
                self.kv_head_idx,
                batch_idx=self.kv_batch_idx,
            ),
            span_bytes,
        )

    def init_tile_bounds(self, **kwargs):
        traits = self.traits
        self.kv_tile_size = traits.BLOCK_Q
        if const_expr(not traits.CAUSAL):
            self.window_left_i32 = None
            self.causal_lo_i32 = None
            tiles = (self.seqlen_q_v + fx.Index(traits.BLOCK_Q - 1)) // fx.Index(
                traits.BLOCK_Q
            )
            tiles = ((tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
            tiles = fx.Index((tiles < fx.Index(2)).select(fx.Index(2), tiles))
            self.split_t0 = fx.Index(0)
            self.split_t_end = tiles
        else:
            left, right = (
                fx.Int32(self.window_left_arg),
                fx.Int32(self.window_right_arg),
            )
            self.window_left_i32 = fx.Int32(left)
            self.causal_lo_i32 = fx.Int32(fx.Int32(0) - fx.Int32(right))
            regions = common.decompose_causal_regions(
                fx.Int32(self.kv_start),
                self.seqlen_kv_i32,
                fx.Int32(self.seqlen_q_v),
                fx.Int32(right),
                fx.Int32(left),
                traits.BLOCK_KV,
                traits.BLOCK_Q,
                fx.Boolean(True),
            )
            n_tiles = regions.n_left + regions.n_full + regions.n_right
            n_tiles = fx.Index((n_tiles > fx.Int32(0)).select(n_tiles, fx.Int32(0)))
            n_tiles = ((n_tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
            self.split_t0 = fx.Index(regions.left_col0) // fx.Index(traits.BLOCK_Q)
            self.split_t_end = self.split_t0 + n_tiles
        self.num_q_tiles = self.split_t_end - self.split_t0
        self.max_num_tiles = self.num_q_tiles
        self.causal_end_raw_i32 = None


class BwdDkDvStreamLoader(ParityKvGmemToLdsLoader):
    def stage_q_tile(self, tile_idx, buf_id):
        self.stride_kv_n_v = self.ctx_ref.stride_q_seq_v
        self.k_div = self.ctx_ref.k_div
        ops.DualwaveKvGmemToLdsLoader.load_k(self, self.tile_start(tile_idx), buf_id)

    def stage_do_tile(self, tile_idx, buf_id):
        self.stride_kv_n_v = self.ctx_ref.stride_do_seq_v
        self.v_div = self.ctx_ref.v_div
        ops.DualwaveKvGmemToLdsLoader.load_v(self, self.tile_start(tile_idx), buf_id)


class BwdDkDvResidentLoader(ops.DualwaveKernelContext):
    def load_resident(self, which, hdim):
        traits = self.traits
        div = self.k_res_div if which == "k" else self.v_res_div
        elem_base = self.k_res_elem_base if which == "k" else self.v_res_elem_base
        stride_seq = self.stride_k_seq_v if which == "k" else self.stride_v_seq_v
        row_in_block = self.kv_row_in_block
        packs = []
        for ks in range_constexpr(traits.K_STEPS_QK):
            col = fx.Index(ks * traits.K_STEP_QK) + self.lane_div_32 * fx.Index(
                traits.MFMA_LANE_K
            )
            raw = ops._buffer_load_128(
                elem_base + row_in_block * stride_seq + col,
                _load_atom_128=self.load_atom_128,
                q_div=div,
                q_load_i32x4_type=self.q_load_i32x4_type,
            )
            pack = Vec(raw, (4,), fx.Int32).bitcast(self.elem_dtype).ir_value()
            packs.append(pack)
        return packs


class BwdDkDvStreamReader(ops.DualwaveKernelContext):
    def read_row_pack(self, tile_base, scope_name, half, ks, hdim):
        traits = self.traits
        idx = (
            tile_base
            + self.stream_row_read_base
            + half * traits.K_LDS_TO_REG_N_STRIP_STRIDE
            + _stream_ks_offset(traits, ks)
        )
        pack = _lds_pack_read(
            traits, self.lds_kv_base_ptr, idx, scope_name, self.kv_mfma_pack_type
        )
        return pack

    def read_column_chunk(self, tile_base, scope_name, dc):
        traits = self.traits
        lds_base = tile_base + self.stream_col_read_base
        pair = traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE * traits.BF16_BYTES
        packs = []
        for k_substep in range_constexpr(4):
            imm_lo = _v_imm_lo(traits, dc, k_substep)
            a = _ds_read_tr_v4f16_imm(
                lds_base,
                imm_lo,
                lds_kv_base_idx=self.lds_kv_base_idx,
                v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                scope_name=scope_name,
                scope_names=traits.LDS_SCOPE_NAMES,
            )
            b = _ds_read_tr_v4f16_imm(
                lds_base,
                imm_lo + pair,
                lds_kv_base_idx=self.lds_kv_base_idx,
                v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                scope_name=scope_name,
                scope_names=traits.LDS_SCOPE_NAMES,
            )
            packs.append(Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
        return packs


class BwdDkDvGemmHelper(ParityGemmHelper):
    def contract_d(self, read_a, b_packs):
        acc = self.c_zero_v16f32
        for ks in range_constexpr(self.traits.K_STEPS_QK):
            acc = ops._mfma_acc(
                read_a(ks), b_packs[ks], acc, self.mma_atom, self.mfma_acc_vec_type
            )
        return acc

    def contract_q(self, a_packs, b_packs, acc):
        for step in range_constexpr(4):
            acc = ops._mfma_acc(
                a_packs[step], b_packs[step], acc, self.mma_atom, self.mfma_acc_vec_type
            )
        return acc


class BwdDkDvSoftmaxHelper(DkDvSoftmaxMath, ParitySoftmaxHelper):
    def load_row_values(self, rsrc, tile_base, half, scale):
        values = [None] * 16
        row_base = fx.Int32(tile_base + fx.Index(half * 32)) + fx.Int32(
            self.lane_div_32
        ) * fx.Int32(4)
        for elem0, col_off, width in _ROW_RUNS:
            span = buffer_load(
                rsrc,
                as_mlir_value(row_base + fx.Int32(col_off)),
                vec_width=width,
                dtype=fx.Float32,
            )
            vec = Vec(span, (width,), fx.Float32)
            for j in range_constexpr(width):
                values[elem0 + j] = ops._fmul(vec[j], scale, self.fm_fast)
        return values

    def mask_if_clipped(self, p_list, tile_idx, half):
        traits = self.traits
        ctx = self.ctx_ref
        rows = traits.MFMA_ROWS
        rel0 = (
            fx.Int32(tile_idx * fx.Index(traits.BLOCK_Q))
            + fx.Int32(half * 32)
            + fx.Int32(self.lane_div_32) * fx.Int32(4)
            - fx.Int32(ctx.kv_row)
        )
        return self._mask_if_clipped(p_list, tile_idx, rel0, rows, _ROW_THRESHOLDS)

    def pack_half(self, values):
        return [
            mfma_operand_wait_state(
                ops._bf16_trunc_pack_v8(
                    self.traits,
                    [values[p * 8 + s] for s in range_constexpr(8)],
                    elem_dtype=self.elem_dtype,
                )
            )
            for p in range_constexpr(self.traits.PV_K_STEPS)
        ]


class BwdDkDvStoreHelper(ops.DualwaveStoreHelper):
    def store_accs(self, accs, which, hdim):
        traits = self.traits
        div = self.dk_div if which == "dk" else self.dv_div
        stride_seq = self.stride_dk_seq_v if which == "dk" else self.stride_dv_seq_v
        base = self.kv_row * stride_seq + self.lane_div_32 * fx.Index(8)
        for dc in range_constexpr(traits.D_CHUNKS):
            for g in range_constexpr(2):
                off = base + fx.Index(dc * traits.D_CHUNK + 2 * g * 8)
                ops._buffer_store_128(
                    ops._packed_o_128_vec(
                        traits, accs, dc, g, self.lane_div_32, self.elem_dtype
                    ),
                    off,
                    _o_store_reg_128=self.o_store_reg_128,
                    _store_atom_128=self.store_atom_128,
                    o_div=div,
                )


class BwdDkDvTileBody(ops.DualwaveKernelContext):
    def __init__(
        self, ctx, *, stream, reader, gemm, softmax, hdim_qk, hdim_vo, tight_registers
    ):
        super().__init__(ctx)
        self.stream = stream
        self.reader = reader
        self.gemm = gemm
        self.softmax = softmax
        self.hdim_qk = hdim_qk
        self.hdim_vo = hdim_vo
        self.tight_registers = tight_registers
        self.half_groups = ((0,), (1,)) if tight_registers else ((0, 1),)

    def _row_reader(self, base, scope, half, hdim):
        if const_expr(self.tight_registers):
            return lambda ks: self.reader.read_row_pack(base, scope, half, ks, hdim)
        packs = [
            self.reader.read_row_pack(base, scope, half, ks, hdim)
            for ks in range_constexpr(self.traits.K_STEPS_QK)
        ]
        return lambda ks: packs[ks]

    def run(self, tile_idx, buf_id, prefetch_idx, v_k, v_v, dv, dk):
        traits = self.traits
        ctx = self.ctx_ref
        q_base = ctx.q_buf_base(buf_id)
        do_base = ctx.do_buf_base(buf_id)
        q_scope = ops._dualwave_lds_scope("k", buf_id)
        do_scope = ops._dualwave_lds_scope("v", buf_id)
        tile_base = tile_idx * fx.Index(traits.BLOCK_Q)

        _waitcnt_vm_n((traits.NUM_STREAM_BUFFERS - 1) * (ctx.NUM_DMA_K + ctx.NUM_DMA_V))
        _sched_barrier(0)
        _s_barrier()

        p_packs = [None] * 4
        ds_packs = [None] * 4
        for group in self.half_groups:
            s = {
                h: self.gemm.contract_d(
                    self._row_reader(q_base, q_scope, h, self.hdim_qk), v_k
                )
                for h in group
            }
            dp = {
                h: self.gemm.contract_d(
                    self._row_reader(do_base, do_scope, h, self.hdim_vo), v_v
                )
                for h in group
            }
            neg_lse2 = {
                h: self.softmax.load_row_values(
                    ctx.lse_rsrc, tile_base, h, ctx.c_neg_log2e
                )
                for h in group
            }
            p_list = {h: self.softmax.probabilities(s[h], neg_lse2[h]) for h in group}
            if const_expr(traits.CAUSAL):
                p_list = {
                    h: self.softmax.mask_if_clipped(p_list[h], tile_idx, h)
                    for h in group
                }
            delta = {
                h: self.softmax.load_row_values(
                    ctx.delta_rsrc, tile_base, h, ctx.c_one_f
                )
                for h in group
            }
            ds_list = {
                h: self.softmax.dscores(p_list[h], dp[h], delta[h]) for h in group
            }
            for h in group:
                p_packs[2 * h : 2 * h + 2] = self.softmax.pack_half(p_list[h])
                ds_packs[2 * h : 2 * h + 2] = self.softmax.pack_half(ds_list[h])

        for dc in range_constexpr(traits.D_CHUNKS):
            dv[dc] = self.gemm.contract_q(
                self.reader.read_column_chunk(do_base, do_scope, dc), p_packs, dv[dc]
            )
        for dc in range_constexpr(traits.D_CHUNKS):
            dk[dc] = self.gemm.contract_q(
                self.reader.read_column_chunk(q_base, q_scope, dc), ds_packs, dk[dc]
            )

        _s_waitcnt(traits.LGKMCNT_0_ONLY)
        _sched_barrier(0)
        _s_barrier()
        self.stream.stage_q_tile(prefetch_idx, buf_id)
        self.stream.stage_do_tile(prefetch_idx, buf_id)
        return dv, dk


def build_fmha_bwd_dkdv_module(meta):
    traits, tight_registers = make_dkdv_traits(meta)
    NBUF = traits.NUM_STREAM_BUFFERS
    M16 = traits.MFMA_ROWS == 16
    TIGHT_REGISTERS = tight_registers
    _cache_tag = (traits_cache_key(traits), TIGHT_REGISTERS)
    _lds_elem_dtype = ops._dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        stream: fx.Array[_lds_elem_dtype, traits.LDS_STREAM_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def fmha_bwd_dkdv_gfx950_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
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
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
    ):
        ctx = BwdDkDvKernelContext(
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
                stride_do_batch,
                stride_do_head,
                stride_do_seq,
                stride_dk_batch,
                stride_dk_head,
                stride_dk_seq,
                stride_dv_batch,
                stride_dv_head,
                stride_dv_seq,
            ),
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
            DO=wire_view(DO),
            DK=wire_view(DK),
            DV=wire_view(DV),
            Delta=wire_view(Delta),
            LSE=wire_view(LSE),
            seq_len=max_seqlen_q,
            seq_len_kv=max_seqlen_k,
            stride_q_n=stride_q_seq,
            stride_kv_n=stride_k_seq,
            head_dim_runtime=hdim_qk,
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
        ctx.init_kv_row()
        # The forward's LSE is already in log2 units.
        ctx.c_neg_log2e = fx.Float32(-1.0)
        ctx.c_one_f = fx.Float32(1.0)

        stream = BwdDkDvStreamLoader(ctx)
        if const_expr(M16):
            reader = M16StreamReader(ctx)
            reader.init16()
            resident = M16ResidentLoader(ctx)
            gemm = M16GemmHelper(ctx)
            gemm.init16()
            store = M16StoreHelper(ctx)
            tile = M16TileBody(
                ctx,
                stream=stream,
                reader=reader,
                gemm=gemm,
                softmax=M16SoftmaxHelper(ctx),
                hdim_qk=hdim_qk,
                hdim_vo=hdim_vo,
            )
            n_acc = d_chunks16(traits)
            zero_acc = Vec.filled(ACC16, 0.0, fx.Float32).ir_value()
            acc_width = ACC16
        else:
            resident = BwdDkDvResidentLoader(ctx)
            store = BwdDkDvStoreHelper(ctx)
            tile = BwdDkDvTileBody(
                ctx,
                stream=stream,
                reader=BwdDkDvStreamReader(ctx),
                gemm=BwdDkDvGemmHelper(ctx),
                softmax=BwdDkDvSoftmaxHelper(ctx),
                hdim_qk=hdim_qk,
                hdim_vo=hdim_vo,
                tight_registers=TIGHT_REGISTERS,
            )
            n_acc = traits.D_CHUNKS
            zero_acc = ctx.c_zero_v16f32
            acc_width = 16

        t0 = ctx.split_t0
        t_end = ctx.split_t_end
        scale_vec = Vec.from_elements([ctx.c_sm_scale], fx.Float32).broadcast_to(
            acc_width
        )

        @flyc.jit
        def _dkdv_body(tile):
            v_k = resident.load_resident("k", hdim_qk)
            v_v = resident.load_resident("v", hdim_vo)

            init_args = []
            for _ in range_constexpr(2 * n_acc):
                init_args.append(zero_acc)
            group_results = init_args

            for g, group_args in range(
                fx.Index(0), ctx.gqa_group, fx.Index(1), init=init_args
            ):
                ctx.bind_q_head(g)

                @flyc.jit
                def _drain_between_heads(g):
                    if g != fx.Index(0):
                        _waitcnt_vm_n(0)
                        rocdl.sched_barrier(0)
                        rocdl.s_barrier()

                _drain_between_heads(g)

                for b in range_constexpr(NBUF):
                    stream.stage_q_tile(t0 + fx.Index(b), b)
                    stream.stage_do_tile(t0 + fx.Index(b), b)

                loop_results = group_args
                for j, loop_args in range(t0, t_end, fx.Index(2), init=group_args):
                    dv = [loop_args[i] for i in range_constexpr(n_acc)]
                    dk = [loop_args[n_acc + i] for i in range_constexpr(n_acc)]
                    for slot in range_constexpr(2):
                        dv, dk = tile.run(
                            j + fx.Index(slot),
                            slot % NBUF,
                            j + fx.Index(slot + NBUF),
                            v_k,
                            v_v,
                            dv,
                            dk,
                        )
                    loop_results = yield dv + dk

                group_results = yield [
                    loop_results[i] for i in range_constexpr(2 * n_acc)
                ]

            dv = [group_results[i] for i in range_constexpr(n_acc)]
            dk = [group_results[n_acc + i] for i in range_constexpr(n_acc)]

            for dc in range_constexpr(n_acc):
                dk[dc] = ops._fmul(Vec(dk[dc]), scale_vec, ctx.fm_fast)
            store.store_accs(dv, "dv", hdim_vo)
            store.store_accs(dk, "dk", hdim_qk)

        active = ctx.active

        def _run_body_if_active():
            if active:
                _dkdv_body(tile)

        _run_body_if_active()

    @flyc.jit
    def launch_fmha_bwd_dkdv_gfx950(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
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
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _cache_tag
        num_kv_blocks = (
            fx.Index(max_seqlen_k) + fx.Index(traits.BLOCK_KV - 1)
        ) // fx.Index(traits.BLOCK_KV)
        fmha_bwd_dkdv_gfx950_kernel(
            Q,
            K,
            V,
            DO,
            DK,
            DV,
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
            stride_dk_batch,
            stride_dk_head,
            stride_dk_seq,
            stride_dv_batch,
            stride_dv_head,
            stride_dv_seq,
            value_attrs=common.kernel_attrs(traits),
        ).launch(
            grid=(fx.Index(num_head_k), num_kv_blocks, fx.Index(batch_size)),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    def _args(
        Q,
        K,
        V,
        DO,
        DK,
        DV,
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
            wire_ptr(DK),
            wire_ptr(DV),
            wire_ptr(LSE, fx.Float32),
            wire_ptr(Delta, fx.Float32),
            int(batch_size),
            int(seqlen_q),
            int(seqlen_k),
            *common.window_args(traits, window),
            *common.shape_args(Q, K, V),
            float(scale),
            *common.strides_of(Q, K, V, DO, DK, DV),
        ), stream

    return common.make_launcher(launch_fmha_bwd_dkdv_gfx950, _args)


def build_split_bwd_dkdv(**kwargs):
    meta = FmhaInputMetadata(**kwargs)
    return build_fmha_bwd_dkdv_module(meta)
