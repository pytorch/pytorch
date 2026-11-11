# mypy: allow-untyped-defs
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Traits, tile geometry and launch plumbing for the split gfx950 backward.

The split backward runs dQ and dK/dV as two kernels over a dense BHSD layout:
no score_mod, no BlockMask beyond causal and a left sliding window, LSE in
log2 units. Derived from AOTriton's gfx950 ``flyc`` backward (38434b8e).
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from . import flex_attn_split_bwd_ops as ops
from .flex_attn_target import GFX950_LIMITS, WAVE_SIZE as WARP_SIZE


def _pow2_shift(value: int) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"expected a power of two, got {value}")
    return value.bit_length() - 1


def ssel(pred, a, b):
    return fx.Int32(fx.Boolean(pred).select(a, b))


def smin(a, b):
    return ssel((a < b), a, b)


def smax(a, b):
    return ssel((a > b), a, b)


def sdiv_rd_pow2(value, divisor: int):
    return fx.Int32(value) >> fx.Int32(_pow2_shift(divisor))


@dataclass(frozen=True, slots=True)
class CausalRegions:
    n_left: fx.Int32
    n_full: fx.Int32
    n_right: fx.Int32
    left_col0: fx.Int32
    full_col0: fx.Int32
    right_col0: fx.Int32
    masked_col0: fx.Int32


def decompose_causal_regions(
    start_q, q_len, k_len, window_left, window_right, block_m, block_n, alive
):
    one = fx.Int32(1)
    zero = fx.Int32(0)
    bn = fx.Int32(block_n)

    q_start = fx.Int32(start_q)
    q_hi = smin(q_start + fx.Int32(block_m), q_len)
    q_last = q_hi - one

    blk_last = sdiv_rd_pow2(k_len - one, block_n)
    blk_last_whole = sdiv_rd_pow2(k_len, block_n) - one

    v_lo = smax(sdiv_rd_pow2(q_start - window_left, block_n), zero)
    v_hi = smin(blk_last, sdiv_rd_pow2(q_last + window_right, block_n))
    v_hi = ssel(alive, v_hi, v_lo - one)

    l_first_full = sdiv_rd_pow2(q_last - window_left + fx.Int32(block_n - 1), block_n)
    r_first_mask = sdiv_rd_pow2(q_start + window_right + one, block_n)

    fb_lo = smax(l_first_full, v_lo)
    fb_hi = smin(smin(r_first_mask - one, blk_last_whole), v_hi)
    fb_empty = fb_lo > fb_hi

    lb_hi = ssel(fb_empty, v_hi, fb_lo - one)
    rb_lo = ssel(fb_empty, v_hi + one, fb_hi + one)

    n_left = smax(lb_hi - v_lo + one, zero)
    n_full = smax(fb_hi - fb_lo + one, zero)
    n_right = smax(v_hi - rb_lo + one, zero)

    left_col0 = v_lo * bn
    right_col0 = rb_lo * bn
    full_col0 = fb_lo * bn
    masked_col0 = smax(ssel(n_left > zero, left_col0, right_col0), zero)
    return CausalRegions(
        n_left, n_full, n_right, left_col0, full_col0, right_col0, masked_col0
    )


LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)

DMA_BYTES = 16
BF16_BYTES = 2


def tile_width_for(head_dim):
    for rung in LADDER:
        if head_dim <= rung:
            return rung
    raise ValueError(f"head_dim {head_dim} exceeds the widest tile ({max(LADDER)})")


@dataclass(frozen=True)
class FmhaInputMetadata:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    causal: bool
    window: bool
    dtype_str: str
    lds_bytes: int = GFX950_LIMITS.lds_bytes


@dataclass(frozen=True)
class ParityDualwaveTraits(ops.DualwaveSwpTraits):
    SMEM_D_BUCKETS: int = 0
    K_STEPS_PER_BAND: int = 0
    D_CHUNKS_PER_BAND: int = 0

    WINDOW: bool = False


VEC_KV = 8
MFMA_LANE_K = 8
K_STEP_QK = 16
MFMA_M = 32
D_CHUNK = 32
PV_K_STEP = 16
K_SUB_N = 32


def make_traits(
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    num_waves,
    block_m,
    block_n,
    granule,
    causal,
    window,
    dtype_str,
    waves_per_eu,
):
    block_size = num_waves * WARP_SIZE
    rows_per_wave = block_m // num_waves

    k_steps_qk = head_dim // K_STEP_QK
    d_chunks = head_dim // D_CHUNK
    pv_k_steps = K_SUB_N // PV_K_STEP

    gqa_group_size = num_heads // num_kv_heads

    smem_linear_wave = WARP_SIZE * DMA_BYTES // BF16_BYTES
    smem_n_per_wave = smem_linear_wave // granule
    smem_n_rpt = block_n // smem_n_per_wave
    smem_d_rpt = head_dim // granule

    smem_k_line_stride = smem_linear_wave + DMA_BYTES // BF16_BYTES
    smem_v_line_stride = smem_linear_wave + 64 // BF16_BYTES
    smem_k_tile_elems = smem_n_rpt * smem_d_rpt * smem_k_line_stride
    smem_v_tile_elems = smem_n_rpt * smem_d_rpt * smem_v_line_stride

    kv_per_buffer = smem_k_tile_elems + smem_v_tile_elems
    lds_kv_total_size = 2 * kv_per_buffer

    k_lds_to_reg_n_strip_stride = (WARP_SIZE // 2) * VEC_KV
    k_lds_to_reg_kstep_inner_stride = K_STEP_QK
    k_lds_to_reg_kstep_outer_stride = smem_n_rpt * smem_k_line_stride

    def _tok_off(t):
        return (t // smem_n_rpt) * granule + (t % smem_n_rpt) * smem_v_line_stride

    v_lds_to_reg_half_wave_stride = _tok_off(4)
    v_lds_to_reg_lane_quad_stride = smem_v_line_stride
    v_lds_to_reg_n_group_stride = 2 * VEC_KV
    v_lds_to_reg_lane_in_quad_stride = 4
    v_lds_to_reg_k_substep_stride = _tok_off(16)
    v_lds_to_reg_dchunk_pair_stride = smem_n_rpt * smem_v_line_stride
    v_lds_to_reg_dchunk_in_pair_stride = D_CHUNK
    v_lds_to_reg_transpose_pair_stride = _tok_off(8)

    return ParityDualwaveTraits(
        SMEM_D_BUCKETS=granule // VEC_KV,
        K_STEPS_PER_BAND=granule // K_STEP_QK,
        D_CHUNKS_PER_BAND=granule // D_CHUNK,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        WARP_SIZE=WARP_SIZE,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=block_size,
        ROWS_PER_WAVE=rows_per_wave,
        HEAD_DIM=head_dim,
        K_STEP_QK=K_STEP_QK,
        K_STEPS_QK=k_steps_qk,
        D_CHUNK=D_CHUNK,
        D_CHUNKS=d_chunks,
        PV_K_STEPS=pv_k_steps,
        MFMA_LANE_K=MFMA_LANE_K,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        CAUSAL=causal,
        WINDOW=window,
        DTYPE_STR=dtype_str,
        WAVES_PER_EU=waves_per_eu,
        DMA_BYTES=DMA_BYTES,
        BF16_BYTES=BF16_BYTES,
        D_128B_SIZE=granule,
        VEC_KV=VEC_KV,
        SMEM_N_RPT=smem_n_rpt,
        SMEM_D_RPT=smem_d_rpt,
        SMEM_K_LINE_STRIDE=smem_k_line_stride,
        SMEM_V_LINE_STRIDE=smem_v_line_stride,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        SMEM_V_TILE_ELEMS=smem_v_tile_elems,
        DUALWAVE_SWP_KV_PER_BUFFER=kv_per_buffer,
        LDS_KV_TOTAL_SIZE=lds_kv_total_size,
        DUALWAVE_SWP_K_BUF_BASE=(0, kv_per_buffer),
        DUALWAVE_SWP_V_BUF_BASE=(smem_k_tile_elems, smem_k_tile_elems + kv_per_buffer),
        K_LDS_TO_REG_N_STRIP_STRIDE=k_lds_to_reg_n_strip_stride,
        K_LDS_TO_REG_KSTEP_INNER_STRIDE=k_lds_to_reg_kstep_inner_stride,
        K_LDS_TO_REG_KSTEP_OUTER_STRIDE=k_lds_to_reg_kstep_outer_stride,
        V_LDS_TO_REG_HALF_WAVE_STRIDE=v_lds_to_reg_half_wave_stride,
        V_LDS_TO_REG_LANE_QUAD_STRIDE=v_lds_to_reg_lane_quad_stride,
        V_LDS_TO_REG_N_GROUP_STRIDE=v_lds_to_reg_n_group_stride,
        V_LDS_TO_REG_LANE_IN_QUAD_STRIDE=v_lds_to_reg_lane_in_quad_stride,
        V_LDS_TO_REG_K_SUBSTEP_STRIDE=v_lds_to_reg_k_substep_stride,
        V_LDS_TO_REG_DCHUNK_PAIR_STRIDE=v_lds_to_reg_dchunk_pair_stride,
        V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE=v_lds_to_reg_dchunk_in_pair_stride,
        V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE=v_lds_to_reg_transpose_pair_stride,
        LDS_SCOPE_NAMES=("lds_k0", "lds_k1", "lds_v0", "lds_v1"),
        NEG_INF_F32_BITS=0xFF800000,
        LGKMCNT_0_ONLY=0xC07F,
    )


_D_ACCESS_ELEMS = DMA_BYTES // BF16_BYTES


def traits_cache_key(traits):
    return tuple((f.name, getattr(traits, f.name)) for f in dataclass_fields(traits))


_COMPILE_HINTS = {
    "fast_fp_math": True,
    "unsafe_fp_math": True,
    "llvm_options": {
        "enable-post-misched": False,
        "lsr-drop-solution": True,
    },
}


def kernel_attrs(traits):
    return {
        "rocdl.waves_per_eu": traits.WAVES_PER_EU,
        "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
        "passthrough": [
            ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
            ["no-nans-fp-math", "true"],
            ["unsafe-fp-math", "true"],
        ],
    }


def window_args(traits, window):
    if not traits.CAUSAL:
        return 0, 0
    left, right = window
    return int(left), int(right)


def shape_args(q, k, v):
    return q.shape[1], k.shape[1], q.shape[3], v.shape[3]


def strides_of(*tensors):
    return [t.stride(i) for t in tensors for i in range(3)]


def make_launcher(jit_fn, pack):
    """Launch ``jit_fn`` on the arguments ``pack`` builds; the first call
    compiles, later calls take the compiled function's fast dispatch."""
    compiled = []

    def launch(*args, **kwargs):
        packed, stream = pack(*args, **kwargs)
        stream = fx.Stream(None) if stream is None else stream
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            if compiled:
                compiled[0](*packed, stream)
            else:
                compiled.append(flyc.compile(jit_fn, *packed, stream))

    def compile(*args, **kwargs):
        packed, stream = pack(*args, **kwargs)
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            return flyc.compile(jit_fn, *packed, fx.Stream(stream))

    launch.compile = compile
    return launch


def wire_ptr(t, elem_type=None):
    if elem_type is None:
        elem_type = _TORCH_DTYPE_TO_FX[str(t.dtype)]
    return flyc.from_c_void_p(elem_type, t.data_ptr())


_TORCH_DTYPE_TO_FX = {
    "torch.bfloat16": fx.BFloat16,
    "torch.float16": fx.Float16,
    "torch.float32": fx.Float32,
    "torch.uint8": fx.Uint8,
    "torch.int32": fx.Int32,
    "torch.int64": fx.Int64,
}


def wire_view(ptr):
    return fx.make_view(ptr, fx.make_layout(fx.Int32(1), fx.Int32(1)))


def mfma_operand_wait_state(pack):
    ir_val = as_mlir_value(pack)
    return llvm.inline_asm(
        ir_val.type, [ir_val], "s_nop 1", "=v,0", has_side_effects=False
    )


def exp2_wait_state(values):
    irs = [as_mlir_value(v) for v in values]
    n = len(irs)
    ret_ty = ir.Type.parse(f"!llvm.struct<({', '.join(['f32'] * n)})>")
    constraints = ",".join(["=v"] * n + [str(i) for i in range(n)])
    ret = llvm.inline_asm(ret_ty, irs, "s_nop 0", constraints, has_side_effects=False)
    return [llvm.extractvalue(irs[i].type, ret, [i]) for i in range(n)]


def _lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def _lds_ptr_with_imm(addr_i32, imm):
    ty = _lds_ptr_ty()
    base = llvm.inttoptr(ty, as_mlir_value(fx.Int32(addr_i32)))
    if imm == 0:
        return base
    return llvm.getelementptr(
        ty,
        base,
        [],
        [imm],
        ir.IntegerType.get_signless(8),
        llvm.GEPNoWrapFlags.inbounds,
    )


def _tag_lds_alias(op, scope_name, scope_names):
    op.operation.attributes["alias_scopes"] = ops._dualwave_lds_alias_scopes(scope_name)
    op.operation.attributes["noalias_scopes"] = ops._dualwave_lds_noalias_scopes(
        scope_name, scope_names
    )
    return op


def _ds_read_tr16_b64_imm(
    result_type, addr_i32, imm_offset=0, scope_name=None, scope_names=()
):
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    ptr = _lds_ptr_with_imm(addr_i32, int(imm_offset))
    op = rocdl.ds_read_tr16_b64(ir.VectorType.get([4], ir.F16Type.get()), ptr)
    raw = _tag_lds_alias(op, scope_name, scope_names).result
    raw = vector.BitCastOp(raw_type, raw).result
    return vector.BitCastOp(result_type, raw).result


def _ds_read_tr_v4f16_imm(
    lds_base_elem_idx,
    imm_bytes,
    lds_kv_base_idx,
    v_lds_read_vec4_type,
    scope_name=None,
    scope_names=(),
):
    byte_offset = lds_base_elem_idx * 2 + lds_kv_base_idx
    addr_i32 = fx.Int32(byte_offset)
    return _ds_read_tr16_b64_imm(
        v_lds_read_vec4_type,
        addr_i32,
        imm_bytes,
        scope_name=scope_name,
        scope_names=scope_names,
    )


def _k_read_base(traits, lane_mod_32, lane_div_32):
    return (
        (lane_mod_32 % traits.SMEM_N_RPT) * traits.SMEM_K_LINE_STRIDE
        + (lane_mod_32 // traits.SMEM_N_RPT) * traits.D_128B_SIZE
        + lane_div_32 * traits.VEC_KV
    )


def _ks_offset(traits, ks):
    per_band = traits.K_STEPS_PER_BAND
    return (ks // per_band) * traits.K_LDS_TO_REG_KSTEP_OUTER_STRIDE + (
        ks % per_band
    ) * traits.K_LDS_TO_REG_KSTEP_INNER_STRIDE


def _v_dc_offset(traits, dc):
    per_band = traits.D_CHUNKS_PER_BAND
    return (dc // per_band) * traits.V_LDS_TO_REG_DCHUNK_PAIR_STRIDE + (
        dc % per_band
    ) * traits.V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE


def _v_imm_lo(traits, dc, k_substep):
    return (
        k_substep * traits.V_LDS_TO_REG_K_SUBSTEP_STRIDE + _v_dc_offset(traits, dc)
    ) * traits.BF16_BYTES


def _slab_span_elems(rows, stride_seq, hdim):
    rows_v = fx.Index(rows)
    width = fx.Index(_D_ACCESS_ELEMS)
    row_elems = ((fx.Index(hdim) + width - fx.Index(1)) // width) * width
    span = rows_v * fx.Index(stride_seq) - fx.Index(stride_seq) + row_elems
    return fx.Index((rows_v > fx.Index(0)).select(span, fx.Index(0)))


def _score_column_runs():
    thresholds = []
    for thr_x, thr_y in ops.CAUSAL_PAIR_THRESHOLDS:
        thresholds.extend((thr_x, thr_y))
    runs, start = [], 0
    for i in range(1, len(thresholds) + 1):
        if i == len(thresholds) or thresholds[i] != thresholds[i - 1] + 1:
            runs.append((start, thresholds[start], i - start))
            start = i
    return runs


class ParityGemmHelper(ops.DualwaveGemmHelper):
    def qk(self, v_k, q_all_bf16):
        k_lo, k_hi = v_k
        v_s_lo, v_s_hi = self.c_zero_v16f32, self.c_zero_v16f32
        for ks in range_constexpr(self.traits.K_STEPS_QK):
            q_pack = mfma_operand_wait_state(
                ops._get_q_pack(self.traits, q_all_bf16, ks)
            )
            v_s_lo = ops._mfma_acc(
                k_lo[ks], q_pack, v_s_lo, self.mma_atom, self.mfma_acc_vec_type
            )
            v_s_hi = ops._mfma_acc(
                k_hi[ks], q_pack, v_s_hi, self.mma_atom, self.mfma_acc_vec_type
            )
        ops._s_nop(1)
        return (v_s_lo, v_s_hi)

    def pv(self, v_p, v_v, v_o):
        for step in range_constexpr(4):
            v_o = self.pv_step_k(step, v_p, v_v, v_o)
        return v_o


class _ParityKvStaging:
    def _issue_split(self, d_flat):
        return d_flat // self.ISSUES_PER_WAVE, d_flat % self.ISSUES_PER_WAVE

    def _dma_line(self, d_flat):
        band, issue = self._issue_split(d_flat)
        return (
            self.wave_id_uni
            + issue * self.traits.NUM_WAVES
            + band * self.traits.SMEM_N_RPT
        )

    def _dma_m0(self, buf_base_elems, line_stride, d_flat):
        addr = (
            self.lds_kv_base_idx
            + (buf_base_elems + self._dma_line(d_flat) * line_stride)
            * self.traits.BF16_BYTES
        )
        return rocdl.readfirstlane(T.i32, as_mlir_value(fx.Int32(addr)))

    def k_dma_base(self, buf_id, d):
        return self._dma_m0(
            ops._k_buf_base(self.traits, buf_id), self.traits.SMEM_K_LINE_STRIDE, d
        )

    def v_dma_base(self, buf_id, d):
        return self._dma_m0(
            ops._v_buf_base(self.traits, buf_id), self.traits.SMEM_V_LINE_STRIDE, d
        )

    def kv_src_elem(self, src_base, d_flat):
        band, issue = self._issue_split(d_flat)
        line_n = self.wave_id + issue * self.traits.NUM_WAVES
        n_in_tile = self.n_in_warp * self.traits.SMEM_N_RPT + line_n
        global_d = self.d_bucket * self.traits.VEC_KV + band * self.traits.D_128B_SIZE
        return src_base + n_in_tile * self.stride_kv_n_v + global_d


class ParityKernelContext(_ParityKvStaging, ops.DualwaveKernelContext):
    def __init__(
        self,
        traits,
        *,
        strides,
        o_strides=(0, 0, 0),
        sm_scale,
        num_head_q,
        num_head_k,
        hdim_qk,
        hdim_vo,
        window_left=None,
        window_right=None,
        **kwargs,
    ):
        super().__init__(traits, **kwargs)
        (
            self.stride_q_batch,
            self.stride_q_head,
            self.stride_q_seq,
            self.stride_k_batch,
            self.stride_k_head,
            self.stride_k_seq,
            self.stride_v_batch,
            self.stride_v_head,
            self.stride_v_seq,
        ) = strides
        self.stride_o_batch, self.stride_o_head, self.stride_o_seq = o_strides
        self.sm_scale_arg = sm_scale
        self.num_head_q = num_head_q
        self.num_head_k = num_head_k
        self.hdim_qk = hdim_qk
        self.hdim_vo = hdim_vo
        self.window_left_arg = window_left
        self.window_right_arg = window_right

    def init_types_and_constants(self, head_dim_runtime=None):
        super().init_types_and_constants(head_dim_runtime=head_dim_runtime)
        self.ISSUES_PER_WAVE = self.traits.SMEM_N_RPT // self.traits.NUM_WAVES
        self.NUM_DMA_K = self.traits.SMEM_D_RPT * self.ISSUES_PER_WAVE
        self.NUM_DMA_V = self.NUM_DMA_K
        self.c_sm_scale = fx.Float32(self.sm_scale_arg)
        self.c_sm_scale_log2e = fx.Float32(
            arith.mulf(
                fx.as_ir_value(fx.Float32(self.sm_scale_arg)),
                fx.as_ir_value(fx.Float32(ops._LOG2E)),
                fastmath=self.fm_fast,
            )
        )

    def init_thread_mapping(self):
        super().init_thread_mapping()
        num_head_k = fx.Index(self.num_head_k)
        gqa_group = fx.Index(self.num_head_q) // num_head_k
        self.gqa_group = gqa_group
        self.h_kv_idx = self.h_idx % num_head_k
        self.group_id = self.h_idx // num_head_k
        self.q_head_idx = self.h_kv_idx * gqa_group + self.group_id
        self.kv_head_idx = self.h_kv_idx

    def init_dma_thread_offsets(self):
        traits = self.traits
        self.lane_in_warp = self.tid % traits.WARP_SIZE
        self.n_in_warp = self.lane_in_warp // traits.SMEM_D_BUCKETS
        self.d_bucket = self.lane_in_warp % traits.SMEM_D_BUCKETS

    def init_lds_read_bases(self):
        super().init_lds_read_bases()
        self.k_lds_read_base_per_lane = _k_read_base(
            self.traits, self.lane_mod_32, self.lane_div_32
        )

    def init_sequence_lengths(self, **kwargs):
        self.kv_batch_idx = self.batch_idx
        self.seqlen_q_v = self.seq_len_v
        self.seqlen_kv_v = self.seq_len_kv_v
        self.seqlen_kv_i32 = fx.Int32(self.seq_len_kv_v)
        self.lse_tokens_i32 = fx.Int32(self.seq_len_v)
        self.q_tok_base = fx.Index(0)
        self.kv_tok_base = fx.Index(0)

    def init_tile_bounds(self, **kwargs):
        if const_expr(self.traits.WINDOW):
            self.window_left_i32 = fx.Int32(self.window_left_arg)
            self.delta_i32 = fx.Int32(self.window_right_arg)
        super().init_tile_bounds(**kwargs)
        if const_expr(self.traits.WINDOW):
            self._skip_dead_leading_tiles()

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        self.stride_q_seq_v = fx.Index(self.stride_q_seq)
        self.stride_k_seq_v = fx.Index(self.stride_k_seq)
        self.stride_v_seq_v = fx.Index(self.stride_v_seq)
        self.stride_o_seq_v = fx.Index(self.stride_o_seq)
        self.stride_q_n_v = self.stride_q_seq_v
        self.stride_kv_n_v = self.stride_k_seq_v

    def _slab_byte_base(self, s0, s1, s2, row_off, head_idx, batch_idx=None):
        if batch_idx is None:
            batch_idx = self.batch_idx
        elems = (
            batch_idx * fx.Index(s0) + head_idx * fx.Index(s1) + row_off * fx.Index(s2)
        )
        return elems * fx.Index(self.traits.BF16_BYTES)

    def _slab_view(
        self, tensor, s0, s1, s2, row_off, head_idx, rows, hdim, batch_idx=None
    ):
        span_elems = _slab_span_elems(rows, s2, hdim)
        base_bytes = self._slab_byte_base(
            s0, s1, s2, row_off, head_idx, batch_idx=batch_idx
        )
        return self._slab_view_at(tensor, base_bytes, span_elems)

    def _slab_view_at(self, tensor, base_bytes, span_elems):
        return ops._make_rebased_view(
            fx.get_iter(tensor),
            base_bytes,
            span_elems * fx.Index(self.traits.BF16_BYTES),
            fx.make_layout(fx.Int32(span_elems), fx.Int32(1)),
            _buf_flags_i32=self.buf_flags_i32,
            _elem_ir=self.elem_ir,
        )

    def q_row_slab(self, tensor, s0, s1, s2, hdim):
        view = self._slab_view(
            tensor, s0, s1, s2, self.q_row_off, self.q_head_idx, self.seqlen_q_v, hdim
        )
        return view, self.seqlen_q_v * fx.Index(s2)

    def _init_descriptor_state(self):
        self.delta_i32 = fx.Int32(self.seqlen_kv_i32 - fx.Int32(self.seqlen_q_v))
        self.buf_flags_i32 = fx.Int32(ops._CDNA_BUFFER_FLAGS)
        self.elem_ir = self.elem_dtype.ir_type
        self.q_row_off = fx.Index(0)
        self.kv_row_off = fx.Index(0)

    def init_descriptors(self):
        self._init_descriptor_state()

        self.q_gmem_elem_offset = self.q_start * self.stride_q_seq_v
        self.kv_gmem_elem_offset = fx.Index(0)

        self.q_div = self._slab_view(
            self.Q,
            self.stride_q_batch,
            self.stride_q_head,
            self.stride_q_seq,
            self.q_row_off,
            self.q_head_idx,
            self.seqlen_q_v,
            self.hdim_qk,
        )
        self.k_div = self._slab_view(
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
        self.v_div = self._slab_view(
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


class ParityQLoader(ops.DualwaveQLoader):
    def load_all(self):
        traits = self.traits
        ctx = self.ctx_ref
        ctx.init_q_row()
        acc = self.load_pack(ctx.q_row_in_block, 0)
        for ks in range_constexpr(traits.K_STEPS_QK - 1):
            acc = ops._concat_vectors(acc, self.load_pack(ctx.q_row_in_block, ks + 1))
        return Vec(acc, (traits.K_STEPS_QK * traits.MFMA_LANE_K,), self.elem_dtype)


class ParityKvLdsToVgprLoader(ops.DualwaveKvLdsToVgprLoader):
    def _read_k_packs(self, buf_id, urk_base):
        traits = self.traits
        k_base = ops._k_buf_base(traits, buf_id)
        k_lo = [None] * traits.K_STEPS_QK
        k_hi = [None] * traits.K_STEPS_QK
        for ks in range_constexpr(traits.K_STEPS_QK):
            k_lo[ks], k_hi[ks] = self._load_k_pair(
                buf_id, k_base + urk_base + _ks_offset(traits, ks)
            )
        return k_lo, k_hi

    def load_k(self, buf_id):
        return self._read_k_packs(buf_id, self.k_lds_read_base_per_lane)

    def read_v_packs(self, buf_id, urv_base):
        traits = self.traits
        lds_base = ops._v_buf_base(traits, buf_id) + urv_base
        v_scope = ops._dualwave_lds_scope("v", buf_id)
        packs = [[None] * traits.D_CHUNKS for _ in range(4)]
        for dc in range_constexpr(traits.D_CHUNKS):
            for k_substep in range_constexpr(4):
                imm_lo = _v_imm_lo(traits, dc, k_substep)
                pair = traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE * traits.BF16_BYTES
                read = lambda off: _ds_read_tr_v4f16_imm(  # noqa: E731
                    lds_base,
                    off,
                    lds_kv_base_idx=self.lds_kv_base_idx,
                    v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                    scope_name=v_scope,
                    scope_names=traits.LDS_SCOPE_NAMES,
                )
                a, b = read(imm_lo), read(imm_lo + pair)
                packs[k_substep][dc] = (
                    Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                )
        return packs

    def load_v(self, buf_id):
        return self.read_v_packs(buf_id, self.v_lds_read_base_per_lane)


class ParityKvGmemToLdsLoader(_ParityKvStaging, ops.DualwaveKvGmemToLdsLoader):
    def _async_load_kv_linear(
        self, dma_m0, buf_id, src_div, src_base, soffset, num_dma
    ):
        for d in range_constexpr(num_dma):
            self._issue_kv_dma(
                src_div, dma_m0[buf_id][d], self.kv_src_elem(src_base, d), soffset
            )


class ParitySoftmaxHelper(ops.DualwaveSoftmaxHelper):
    def _causal_mask_inplace(self, v_s, tile_idx, q_row_i32=None):
        super()._causal_mask_inplace(v_s, tile_idx, q_row_i32=q_row_i32)
        traits = self.traits
        s_lo, s_hi = v_s
        kv_start_i32 = fx.Int32(tile_idx * traits.BLOCK_N)
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(4)
        rel_lo = fx.Int32(
            q_row_i32 - self.ctx_ref.window_left_i32 - kv_start_i32 - lane_off_i32
        )
        rel_hi = fx.Int32(rel_lo - fx.Int32(32))
        for p in range_constexpr(len(ops.CAUSAL_PAIR_THRESHOLDS)):
            thr_x, thr_y = ops.CAUSAL_PAIR_THRESHOLDS[p]
            ix, iy = 2 * p, 2 * p + 1
            s_lo[ix] = (rel_lo <= fx.Int32(thr_x)).select(s_lo[ix], self.c_neg_inf)
            s_lo[iy] = (rel_lo <= fx.Int32(thr_y)).select(s_lo[iy], self.c_neg_inf)
            s_hi[ix] = (rel_hi <= fx.Int32(thr_x)).select(s_hi[ix], self.c_neg_inf)
            s_hi[iy] = (rel_hi <= fx.Int32(thr_y)).select(s_hi[iy], self.c_neg_inf)

    def causal_mask_prologue_if_needed(
        self, v_s, tile_idx=None, kv_end_pos=None, **kwargs
    ):
        traits = self.traits
        kv_end_tile = kwargs.get("kv_end_tile")
        if kv_end_pos is None:
            kv_end_pos = self.tile_start(
                tile_idx + fx.Index(1) if kv_end_tile is None else kv_end_tile
            )
        kv_start_pos = self.tile_start(tile_idx)
        q_start_pos_i32 = kwargs.get("q_start_pos_i32") or self.ctx_ref.q_start_pos_i32
        q_row_i32 = kwargs.get("q_row_i32") or self.ctx_ref.q_row_i32
        window_left_i32 = self.ctx_ref.window_left_i32
        delta_i32 = self.delta_i32
        mask_inplace = self._causal_mask_inplace
        to_lists = self.v_s_vec_to_lists

        @flyc.jit
        def _window_mask_if_needed(
            v_s, tile_idx, kv_end_pos, kv_start_pos, q_start_pos_i32, q_row_i32
        ):
            s_lo, s_hi = v_s
            clipped_right = q_start_pos_i32 + delta_i32 < fx.Int32(kv_end_pos)
            clipped_left = (
                fx.Int32(kv_start_pos)
                < q_start_pos_i32 + fx.Int32(traits.BLOCK_M - 1) - window_left_i32
            )
            if clipped_right | clipped_left:
                lo_list, hi_list = to_lists(v_s)
                mask_inplace((lo_list, hi_list), tile_idx, q_row_i32=q_row_i32)
                s_lo, s_hi = ops._score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _window_mask_if_needed(
            v_s, tile_idx, kv_end_pos, kv_start_pos, q_start_pos_i32, q_row_i32
        )


class ParityStoreHelper(ops.DualwaveStoreHelper):
    def _final_o_base(self, q_row):
        return q_row * self.stride_o_seq_v + self.lane_div_32 * 8
