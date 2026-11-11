# mypy: allow-untyped-defs
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Dual-wave context, loaders and register helpers shared by the split backward.

Derived from FlyDSL v0.3.1's ``kernels/attention/flash_attn_utils.py`` by way
of AOTriton's gfx950 ``flyc`` backward (38434b8e), cut down to the paths the
split backward runs.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, rocdl as rocdl_ir
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace as _TargetAddressSpace
from flydsl.expr import arith, const_expr, gpu, math as fmath, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from .flex_attn_fwd_dualwave_gfx950 import (
    _dtype_to_elem_type,
    _lds_alias_scope_array,
    _lds_byte_gep,
    _LOG2E,
)
from .mxfp8_buffer_ops import _CDNA_BUFFER_FLAGS


def _s_nop(x):
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"), [], f"s_nop {x}", "", has_side_effects=True
    )


def _fadd(a, b, fm_fast):
    return arith.addf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fsub(a, b, fm_fast):
    return arith.subf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fmul(a, b, fm_fast):
    return arith.mulf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fmax(a, b, fm_fast):
    return arith.MaxNumFOp(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast).result


def _mfma_acc(a, b, c, _mma_atom, mfma_acc_vec_type):
    return fly.mma_atom_call_ssa([mfma_acc_vec_type], _mma_atom, a, b, c)


def _concat_vectors(lhs, rhs):
    lhs_vec = Vec(lhs)
    rhs_vec = Vec(rhs)
    return lhs_vec.shuffle(
        rhs_vec,
        list(range(lhs_vec.numel)) + [lhs_vec.numel + i for i in range(rhs_vec.numel)],
    )


def _bitcast_i32(value):
    return as_mlir_value(fx.Float32(value).bitcast(fx.Int32).ir_value())


def _bitcast_f32(value):
    return as_mlir_value(fx.Int32(value).bitcast(fx.Float32).ir_value())


def _attn_mask_vec2_imm(rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32):
    asm_str = (
        f"v_cmp_lt_i32_e64 $0, $6, {int(thr_x)}\n\t"
        f"v_cmp_lt_i32_e64 $1, $6, {int(thr_y)}\n\t"
        "v_cndmask_b32_e64 $2, $4, $7, $0\n\t"
        "v_cndmask_b32_e64 $3, $5, $7, $1"
    )
    ret_struct_ty = ir.Type.parse("!llvm.struct<(i64, i64, i32, i32)>")
    ret = llvm.inline_asm(
        ret_struct_ty,
        [
            as_mlir_value(x_ref_i32),
            as_mlir_value(y_ref_i32),
            as_mlir_value(rel_i32),
            as_mlir_value(neg_inf_i32),
        ],
        asm_str,
        "=s,=s,=v,=v,2,3,v,v,~{vcc}",
        has_side_effects=True,
    )
    return llvm.extractvalue(T.i32, ret, [2]), llvm.extractvalue(T.i32, ret, [3])


def _swap_halves(dw):
    pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(
        pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False
    )
    lo_res = llvm.extractvalue(T.i32, swapped, [0])
    hi_res = llvm.extractvalue(T.i32, swapped, [1])
    return lo_res, hi_res


def _fused_o_128_dwords(lane_div_32, d0_a, d1_a, d0_b, d1_b):
    is_hi_half = lane_div_32 != fx.Index(0)
    y0_a_lo, y0_a_hi = _swap_halves(d0_a)
    y1_a_lo, y1_a_hi = _swap_halves(d1_a)
    y0_b_lo, y0_b_hi = _swap_halves(d0_b)
    y1_b_lo, y1_b_hi = _swap_halves(d1_b)
    y0_a, y1_a = (
        is_hi_half.select(y0_a_lo, y0_a_hi),
        is_hi_half.select(y1_a_lo, y1_a_hi),
    )
    y0_b, y1_b = (
        is_hi_half.select(y0_b_lo, y0_b_hi),
        is_hi_half.select(y1_b_lo, y1_b_hi),
    )
    w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
    w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
    w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
    w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
    return w0, w1, w2, w3


def _o_pack_2dw(traits, v_o, dc, store_group, elem_dtype):
    r_base = store_group * 4
    if const_expr(traits.DTYPE_STR == "bf16"):
        lo = rocdl.cvt_pk_bf16_f32(
            Vec(v_o[dc])[r_base],
            Vec(v_o[dc])[r_base + 1],
        )
        hi = rocdl.cvt_pk_bf16_f32(
            Vec(v_o[dc])[r_base + 2],
            Vec(v_o[dc])[r_base + 3],
        )
        return lo, hi

    o_f16 = []
    for i in range_constexpr(4):
        o_f16.append(fx.Float32(Vec(v_o[dc])[r_base + i]).to(elem_dtype))
    pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32)
    return as_mlir_value(pack[0]), as_mlir_value(pack[1])


def _packed_o_128_dwords(traits, v_o, dc, g, lane_div_32, elem_dtype):
    d0_a, d1_a = _o_pack_2dw(traits, v_o, dc, 2 * g, elem_dtype)
    d0_b, d1_b = _o_pack_2dw(traits, v_o, dc, 2 * g + 1, elem_dtype)
    return _fused_o_128_dwords(lane_div_32, d0_a, d1_a, d0_b, d1_b)


def _packed_o_128_vec(traits, v_o, dc, g, lane_div_32, elem_dtype):
    return Vec.from_elements(
        [
            fx.Int32(w)
            for w in _packed_o_128_dwords(traits, v_o, dc, g, lane_div_32, elem_dtype)
        ],
        fx.Int32,
    )


def _bf16_trunc_pack_v8(traits, f32_vals, elem_dtype):
    if const_expr(traits.DTYPE_STR == "bf16"):
        pairs = []
        for j in range_constexpr(4):
            pairs.append(rocdl.cvt_pk_bf16_f32(f32_vals[j * 2], f32_vals[j * 2 + 1]))
        return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
    f16_vals = []
    for i in range_constexpr(8):
        f16_vals.append(fx.Float32(f32_vals[i]).to(elem_dtype))
    return Vec.from_elements(f16_vals, elem_dtype).ir_value()


def _score_pair_to_lists(v_s):
    s_lo, s_hi = v_s
    return (
        [Vec(s_lo)[r] for r in range_constexpr(16)],
        [Vec(s_hi)[r] for r in range_constexpr(16)],
    )


def _score_lists_to_vecs(v_s_lists):
    s_lo, s_hi = v_s_lists
    return (
        Vec.from_elements([as_mlir_value(v) for v in s_lo], fx.Float32).ir_value(),
        Vec.from_elements([as_mlir_value(v) for v in s_hi], fx.Float32).ir_value(),
    )


def _pack_p_v8_slices(traits, v_p, pack_v8_fn):
    lo_partial_list, hi_full = v_p
    p_lo_packs = []
    p_hi_packs = []
    for pks in range_constexpr(traits.PV_K_STEPS):
        p_base = pks * 8
        lo_slice = [lo_partial_list[p_base + s] for s in range_constexpr(8)]
        hi_slice = hi_full[p_base : p_base + 8]
        p_lo_packs.append(pack_v8_fn(lo_slice))
        p_hi_packs.append(pack_v8_fn(hi_slice))
    return p_lo_packs, p_hi_packs


def _scale_o_accs(v_o, scale_scalar, traits, fm_fast):
    scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
    for dc in range_constexpr(traits.D_CHUNKS):
        v_o[dc] = _fmul(Vec(v_o[dc]), scale_vec, fm_fast)


CAUSAL_PAIR_THRESHOLDS = (
    (0, 1),
    (2, 3),
    (8, 9),
    (10, 11),
    (16, 17),
    (18, 19),
    (24, 25),
    (26, 27),
)


def _apply_dualwave_causal_mask_pair(s_values, rel_i32, neg_inf_i32, pair_thresholds):
    for p in range_constexpr(len(pair_thresholds)):
        thr_x, thr_y = pair_thresholds[p]
        idx_x = p * 2
        idx_y = p * 2 + 1
        x_bits = _bitcast_i32(s_values[idx_x])
        y_bits = _bitcast_i32(s_values[idx_y])
        new_x, new_y = _attn_mask_vec2_imm(
            rel_i32, neg_inf_i32, thr_x, thr_y, x_bits, y_bits
        )
        s_values[idx_x] = _bitcast_f32(new_x)
        s_values[idx_y] = _bitcast_f32(new_y)


def _dualwave_lds_scope(kind, buf_id):
    return f"lds_{kind}{buf_id}"


def _dualwave_lds_alias_scopes(name):
    return _lds_alias_scope_array([name])


def _dualwave_lds_noalias_scopes(name, scope_names):
    return _lds_alias_scope_array(
        [scope_name for scope_name in scope_names if scope_name != name]
    )


def make_buffer_rsrc(tensor, byte_offset, nrec_bytes):
    """A buffer descriptor over ``nrec_bytes`` of ``tensor`` from ``byte_offset``."""
    addr = fx.Int64(fx.ptrtoint(fx.get_iter(tensor))) + fx.Int64(byte_offset)
    return rocdl_ir.MakeBufferRsrcOp(
        ir.Type.parse("!llvm.ptr<8>"),
        llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), as_mlir_value(addr)).result,
        fx.Int16(0).ir_value(),
        fx.Int64(nrec_bytes).ir_value(),
        fx.Int32(_CDNA_BUFFER_FLAGS).ir_value(),
    ).result


def _make_rebased_view(
    base_iter, byte_off, nrec_bytes, layout, _buf_flags_i32, _elem_ir
):
    base_i64 = fx.Int64(fx.ptrtoint(base_iter))
    shifted = fx.inttoptr(base_iter.type, base_i64 + fx.Int64(byte_off))
    buf_ptr_ty = fx.PointerType.get(
        elem_ty=_elem_ir,
        address_space=_TargetAddressSpace.BufferDesc,
        alignment=base_iter.alignment,
    )
    buf_ptr = fx.make_ptr(
        buf_ptr_ty,
        [
            shifted,
            fx.Int16(0).ir_value(),
            fx.Int64(nrec_bytes).ir_value(),
            _buf_flags_i32.ir_value(),
        ],
    )
    return fx.logical_divide(fx.make_view(buf_ptr, layout), fx.make_layout(1, 1))


def _load_k_pack_aligned(traits, lds_kv_base_ptr, elem_idx, buf_id, kv_mfma_pack_type):
    scope_name = _dualwave_lds_scope("k", buf_id)
    byte_offset = elem_idx * traits.BF16_BYTES
    ptr = _lds_byte_gep(lds_kv_base_ptr, byte_offset)
    return llvm.LoadOp(
        kv_mfma_pack_type,
        ptr,
        alignment=16,
        alias_scopes=_dualwave_lds_alias_scopes(scope_name),
        noalias_scopes=_dualwave_lds_noalias_scopes(scope_name, traits.LDS_SCOPE_NAMES),
    ).result


def _buffer_load_128(elem_index, _load_atom_128, q_div, q_load_i32x4_type):
    return fly.copy_atom_call_ssa(
        [q_load_i32x4_type],
        _load_atom_128,
        fx.slice(q_div, (None, fx.Int32(elem_index))),
    )


def _buffer_load_lds_128(
    src_div, lds_byte_addr, src_elem, soffset_elems, _dma_atom, _lds_ptr_ty
):
    lds_ptr = fx.inttoptr(_lds_ptr_ty, fx.Int32(lds_byte_addr))
    dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
    src = fx.slice(src_div, (None, fx.Int32(src_elem)))
    fx.copy(_dma_atom, src, dst, soffset=fx.Int32(soffset_elems))


def _buffer_store_128(
    pack_i32_vec, elem_index, _o_store_reg_128, _store_atom_128, o_div
):
    fx.memref_store_vec(pack_i32_vec, _o_store_reg_128)
    fx.copy(
        _store_atom_128, _o_store_reg_128, fx.slice(o_div, (None, fx.Int32(elem_index)))
    )


def _k_buf_base(traits, buf_id):
    if const_expr(isinstance(buf_id, int)):
        return traits.DUALWAVE_SWP_K_BUF_BASE[buf_id]
    return buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER


def _v_buf_base(traits, buf_id):
    if const_expr(isinstance(buf_id, int)):
        return traits.DUALWAVE_SWP_V_BUF_BASE[buf_id]
    return traits.SMEM_K_TILE_ELEMS + buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER


def _kv_tile_addr(tile_start, kv_gmem_elem_offset, stride_kv_n_v):
    return kv_gmem_elem_offset, tile_start * stride_kv_n_v


def _q_pack_col(traits, ks, lane_div_32):
    return ks * traits.K_STEP_QK + lane_div_32 * traits.MFMA_LANE_K


def _q_pack_global_idx(traits, q_row_in_block, ks, lane_div_32, stride_q_n_v):
    return q_row_in_block * stride_q_n_v + _q_pack_col(
        traits, ks, lane_div_32=lane_div_32
    )


def _get_q_pack(traits, q_all_scaled_bf16, ks):
    q_vec = Vec(q_all_scaled_bf16)
    base = ks * traits.MFMA_LANE_K
    return q_vec.shuffle(
        q_vec, [base + i for i in range(traits.MFMA_LANE_K)]
    ).ir_value()


def _k_lds_read_base_per_lane(traits, lane_mod_32, lane_div_32):
    return (
        (lane_mod_32 % 8) * traits.SMEM_K_LINE_STRIDE
        + (lane_mod_32 // 8) * traits.D_128B_SIZE
        + lane_div_32 * traits.VEC_KV
    )


def _v_lds_read_base_per_lane(traits, lane, lane_div_32):
    return (
        lane_div_32 * traits.V_LDS_TO_REG_HALF_WAVE_STRIDE
        + ((lane % 16) // 4) * traits.V_LDS_TO_REG_LANE_QUAD_STRIDE
        + ((lane // 16) % 2) * traits.V_LDS_TO_REG_N_GROUP_STRIDE
        + (lane % 4) * traits.V_LDS_TO_REG_LANE_IN_QUAD_STRIDE
    )


def _seq_pad_col_base(traits, tile_idx, lane_div_32):
    return fx.Int32(tile_idx * traits.BLOCK_N) + fx.Int32(lane_div_32) * fx.Int32(4)


def _seq_pad_score_threshold(traits, r):
    return (r // 4) * 8 + (r % 4)


def _init_dualwave_thread_mapping(ctx):
    traits = ctx.traits
    ctx.h_idx = fx.Index(gpu.block_idx.x)
    ctx.q_block_idx = fx.Index(gpu.block_idx.y)
    ctx.batch_idx = fx.Index(gpu.block_idx.z)
    ctx.tid = fx.Index(gpu.thread_idx.x)

    ctx.wave_id = ctx.tid // traits.WARP_SIZE
    ctx.lane = ctx.tid % traits.WARP_SIZE
    ctx.lane_mod_32 = ctx.lane % 32
    ctx.lane_div_32 = ctx.lane // 32

    _tid_i32 = as_mlir_value(fx.Int32(ctx.tid))
    _wave_id_uni_i32 = rocdl.readfirstlane(
        T.i32,
        arith.divsi(_tid_i32, as_mlir_value(fx.Int32(traits.WARP_SIZE))),
    )
    ctx.wave_id_uni = fx.Index(_wave_id_uni_i32)

    ctx.wave_q_offset = ctx.wave_id * traits.ROWS_PER_WAVE
    ctx.q_start = ctx.q_block_idx * traits.BLOCK_M

    ctx.h_kv_idx = ctx.h_idx % traits.NUM_HEADS_KV
    ctx.group_id = ctx.h_idx // traits.NUM_HEADS_KV
    ctx.q_head_idx = ctx.h_kv_idx * traits.GQA_GROUP_SIZE + ctx.group_id
    ctx.kv_head_idx = ctx.h_kv_idx


def _init_dualwave_q_row(ctx):
    traits = ctx.traits
    ctx.q_row_in_block = ctx.wave_q_offset + ctx.lane_mod_32
    ctx.q_start_pos_i32 = fx.Int32(ctx.q_start + ctx.wave_id_uni * traits.ROWS_PER_WAVE)
    ctx.q_row = ctx.q_start + ctx.q_row_in_block
    ctx.q_row_i32 = fx.Int32(ctx.q_row)


@dataclass(frozen=True)
class DualwaveSwpTraits:
    BLOCK_M: int
    BLOCK_N: int
    WARP_SIZE: int
    NUM_WAVES: int
    BLOCK_SIZE: int
    ROWS_PER_WAVE: int
    HEAD_DIM: int
    K_STEP_QK: int
    K_STEPS_QK: int
    D_CHUNK: int
    D_CHUNKS: int
    PV_K_STEPS: int
    MFMA_LANE_K: int
    NUM_HEADS_KV: int
    GQA_GROUP_SIZE: int
    CAUSAL: bool
    DTYPE_STR: str
    WAVES_PER_EU: int
    DMA_BYTES: int
    BF16_BYTES: int
    D_128B_SIZE: int
    VEC_KV: int
    SMEM_N_RPT: int
    SMEM_D_RPT: int
    SMEM_K_LINE_STRIDE: int
    SMEM_V_LINE_STRIDE: int
    SMEM_K_TILE_ELEMS: int
    SMEM_V_TILE_ELEMS: int
    DUALWAVE_SWP_KV_PER_BUFFER: int
    LDS_KV_TOTAL_SIZE: int
    DUALWAVE_SWP_K_BUF_BASE: tuple[int, int]
    DUALWAVE_SWP_V_BUF_BASE: tuple[int, int]
    K_LDS_TO_REG_N_STRIP_STRIDE: int
    K_LDS_TO_REG_KSTEP_INNER_STRIDE: int
    K_LDS_TO_REG_KSTEP_OUTER_STRIDE: int
    V_LDS_TO_REG_HALF_WAVE_STRIDE: int
    V_LDS_TO_REG_LANE_QUAD_STRIDE: int
    V_LDS_TO_REG_N_GROUP_STRIDE: int
    V_LDS_TO_REG_LANE_IN_QUAD_STRIDE: int
    V_LDS_TO_REG_K_SUBSTEP_STRIDE: int
    V_LDS_TO_REG_DCHUNK_PAIR_STRIDE: int
    V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE: int
    V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE: int
    LDS_SCOPE_NAMES: tuple[str, str, str, str]
    NEG_INF_F32_BITS: int
    LGKMCNT_0_ONLY: int


class DualwaveKernelContext:
    def __init__(
        self,
        traits_or_ctx,
        Q=None,
        K=None,
        V=None,
        O=None,  # noqa: E741
        seq_len=None,
        seq_len_kv=None,
        stride_q_n=None,
        stride_kv_n=None,
        head_dim_runtime=None,
        LSE=None,
    ):
        if isinstance(traits_or_ctx, DualwaveKernelContext):
            self.__dict__.update(traits_or_ctx.__dict__)
            self.ctx_ref = getattr(traits_or_ctx, "ctx_ref", traits_or_ctx)
            return

        self.ctx_ref = self
        self.traits = traits_or_ctx
        self.Q = Q
        self.K = K
        self.V = V
        self.O = O
        self.LSE = LSE
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.stride_q_n = stride_q_n
        self.stride_kv_n = stride_kv_n
        self.head_dim_runtime = head_dim_runtime

    def init_types_and_constants(self, head_dim_runtime=None):
        if head_dim_runtime is None:
            head_dim_runtime = self.head_dim_runtime
        traits = self.traits
        self.NUM_DMA_K = traits.SMEM_D_RPT
        self.NUM_DMA_V = traits.SMEM_D_RPT

        self.fm_fast = fx.arith.FastMathFlags.fast
        self.elem_dtype = _dtype_to_elem_type(traits.DTYPE_STR)
        self.q_load_i32x4_type = Vec.make_type(4, fx.Int32)
        self.v_lds_read_vec4_type = Vec.make_type(4, self.elem_dtype)
        self.kv_mfma_pack_type = Vec.make_type(8, self.elem_dtype)
        self.mfma_acc_vec_type = Vec.make_type(16, fx.Float32)

        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_neg_floor = fx.Float32(-3.0e38)
        self.c_zero_f = fx.Float32(0.0)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)

    def init_runtime_indices(
        self, seq_len=None, seq_len_kv=None, stride_q_n=None, stride_kv_n=None
    ):
        if seq_len is None:
            seq_len = self.seq_len
        if seq_len_kv is None:
            seq_len_kv = self.seq_len_kv
        if stride_q_n is None:
            stride_q_n = self.stride_q_n
        if stride_kv_n is None:
            stride_kv_n = self.stride_kv_n
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.seq_len_v = fx.Index(seq_len)
        self.seq_len_kv_v = fx.Index(seq_len_kv)
        self.stride_q_n_v = fx.Index(stride_q_n)
        self.stride_kv_n_v = fx.Index(stride_kv_n)

    def init_lds(self, shared_storage):
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.kv.ptr))
        self.lds_kv_base_ptr = lds.kv.ptr.llvm_ptr

    def init_thread_mapping(self):
        _init_dualwave_thread_mapping(self)

    def init_atoms_and_lds_ptrs(self):
        self.load_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.store_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, self.elem_dtype))
        self.o_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        self.lds_ptr_ty = fx.PointerType.get(
            self.elem_dtype.ir_type, 2, self.traits.DMA_BYTES
        )

    def init_q_row(self):
        _init_dualwave_q_row(self)

    def dma_m0_table(self, base_fn, count):
        return tuple(tuple(base_fn(buf, d) for d in range(count)) for buf in range(2))

    def init_dma_m0_tables(self):
        self.k_dma_m0 = self.dma_m0_table(self.k_dma_base, self.NUM_DMA_K)
        self.v_dma_m0 = self.dma_m0_table(self.v_dma_base, self.NUM_DMA_V)

    def init_tile_bounds(self):
        traits = self.traits
        self.kv_tile_size = traits.BLOCK_N
        self.num_kv_tiles = (
            self.seqlen_kv_v + self.kv_tile_size - 1
        ) // self.kv_tile_size
        if const_expr(traits.CAUSAL):
            self.causal_end_raw_i32 = (
                fx.Int32(self.q_start + traits.BLOCK_M) + self.delta_i32
            )
            causal_end_i32 = fx.Int32(
                (self.causal_end_raw_i32 > fx.Int32(0)).select(
                    self.causal_end_raw_i32, fx.Int32(0)
                )
            )
            causal_num_tiles = (
                fx.Index(causal_end_i32) + self.kv_tile_size - 1
            ) // self.kv_tile_size
            self.max_num_tiles = fx.Index(
                (causal_num_tiles < self.num_kv_tiles).select(
                    causal_num_tiles, self.num_kv_tiles
                )
            )
        else:
            self.causal_end_raw_i32 = None
            self.max_num_tiles = self.num_kv_tiles

        self.max_num_tiles = (
            (self.max_num_tiles + fx.Index(1)) // fx.Index(2)
        ) * fx.Index(2)
        self.max_num_tiles = fx.Index(
            (self.max_num_tiles < fx.Index(4)).select(fx.Index(4), self.max_num_tiles)
        )

        self.split_t0 = 0
        self.split_t_end = self.max_num_tiles

    def init_active_guard(self):
        self.active = self.compute_active_guard()

    def init_lds_read_bases(self):
        self.k_lds_read_base_per_lane = _k_lds_read_base_per_lane(
            self.traits, self.lane_mod_32, self.lane_div_32
        )
        self.v_lds_read_base_per_lane = _v_lds_read_base_per_lane(
            self.traits, self.lane, self.lane_div_32
        )

    def split_tile(self, offset_tiles=0):
        return self.split_t0 + fx.Index(offset_tiles)

    def tile_start(self, tile_idx):
        return tile_idx * self.traits.BLOCK_N


class DualwaveQLoader(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_pack(self, q_row_in_block, ks):
        q_i32_pack = _buffer_load_128(
            self.q_gmem_elem_offset
            + _q_pack_global_idx(
                self.traits,
                q_row_in_block,
                ks,
                lane_div_32=self.lane_div_32,
                stride_q_n_v=self.stride_q_n_v,
            ),
            _load_atom_128=self.load_atom_128,
            q_div=self.q_div,
            q_load_i32x4_type=self.q_load_i32x4_type,
        )
        return Vec(q_i32_pack, (4,), fx.Int32).bitcast(self.elem_dtype).ir_value()


class DualwaveGemmHelper(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def pv_step_k(self, step, v_p, v_v, v_o):
        v_p_lo, v_p_hi = v_p
        v_pk = v_v[step]
        if const_expr(step < 2):
            p_pk = v_p_lo[step]
        else:
            p_pk = v_p_hi[step - 2]
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_o[dc] = _mfma_acc(
                v_pk[dc], p_pk, v_o[dc], self.mma_atom, self.mfma_acc_vec_type
            )
        return v_o


class DualwaveSoftmaxHelper(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def cast_p(self, v_p):
        return _pack_p_v8_slices(
            self.traits,
            v_p,
            lambda vals: _bf16_trunc_pack_v8(
                self.traits, vals, elem_dtype=self.elem_dtype
            ),
        )

    def scale_o(self, v_o, scale_scalar):
        _scale_o_accs(v_o, scale_scalar, self.traits, self.fm_fast)

    def v_s_vec_to_lists(self, v_s):
        return _score_pair_to_lists(v_s)

    def _causal_mask_inplace(self, v_s, tile_idx, q_row_i32=None):
        traits = self.traits
        s_lo, s_hi = v_s
        kv_tile_start = tile_idx * traits.BLOCK_N
        kv_start_i32 = fx.Int32(kv_tile_start)
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(4)
        rel_lo_i32 = fx.Int32(q_row_i32 + self.delta_i32 - kv_start_i32 - lane_off_i32)
        rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
        neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)

        pair_thresholds = CAUSAL_PAIR_THRESHOLDS
        _apply_dualwave_causal_mask_pair(s_lo, rel_lo_i32, neg_inf_i32, pair_thresholds)
        _apply_dualwave_causal_mask_pair(s_hi, rel_hi_i32, neg_inf_i32, pair_thresholds)

    def seq_pad_mask_inplace(self, v_s_lists, tile_idx):
        s_lo, s_hi = v_s_lists
        col_base = _seq_pad_col_base(
            self.traits, tile_idx, lane_div_32=self.lane_div_32
        )
        for r in range_constexpr(16):
            col_lo = col_base + fx.Int32(_seq_pad_score_threshold(self.traits, r))
            col_hi = col_lo + fx.Int32(32)
            s_lo[r] = (col_lo < self.seqlen_kv_i32).select(s_lo[r], self.c_neg_inf)
            s_hi[r] = (col_hi < self.seqlen_kv_i32).select(s_hi[r], self.c_neg_inf)

    def seq_pad_mask_if_needed(self, v_s, tile_idx):
        traits = self.traits
        seqlen_kv_i32 = self.seqlen_kv_i32

        @flyc.jit
        def _seq_pad_mask_if_needed(v_s, tile_idx):
            s_lo, s_hi = v_s
            kv_tile_end = (tile_idx + fx.Index(1)) * traits.BLOCK_N
            if fx.Int32(kv_tile_end) > seqlen_kv_i32:
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self.seq_pad_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _seq_pad_mask_if_needed(v_s, tile_idx)


class DualwaveKvGmemToLdsLoader(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.num_dma_k = self.NUM_DMA_K
        self.num_dma_v = self.NUM_DMA_V

    def _issue_kv_dma(self, src_div, lds_addr, src_elem, soffset):
        _buffer_load_lds_128(
            src_div,
            lds_addr,
            src_elem,
            soffset,
            _dma_atom=self.dma_atom,
            _lds_ptr_ty=self.lds_ptr_ty,
        )

    def load_k(self, tile_start, buf_id):
        ctx = self.ctx_ref
        src_base, soffset = _kv_tile_addr(
            tile_start, self.kv_gmem_elem_offset, self.stride_kv_n_v
        )
        self._async_load_kv_linear(
            ctx.k_dma_m0, buf_id, self.k_div, src_base, soffset, self.num_dma_k
        )

    def load_v(self, tile_start, buf_id):
        ctx = self.ctx_ref
        src_base, soffset = _kv_tile_addr(
            tile_start, self.kv_gmem_elem_offset, self.stride_kv_n_v
        )
        self._async_load_kv_linear(
            ctx.v_dma_m0, buf_id, self.v_div, src_base, soffset, self.num_dma_v
        )


class DualwaveKvLdsToVgprLoader(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _load_k_pair(self, buf_id, idx_lo):
        lo = _load_k_pack_aligned(
            self.traits,
            self.lds_kv_base_ptr,
            idx_lo,
            buf_id,
            self.kv_mfma_pack_type,
        )
        hi = _load_k_pack_aligned(
            self.traits,
            self.lds_kv_base_ptr,
            idx_lo + self.traits.K_LDS_TO_REG_N_STRIP_STRIDE,
            buf_id,
            self.kv_mfma_pack_type,
        )
        return lo, hi


class DualwaveStoreHelper(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _final_o_global(self, o_base, dc, g):
        return o_base + (dc * self.traits.D_CHUNK + 2 * g * 8)

    def _store_final_o_128(self, v_o, dc, g, o_base):
        _buffer_store_128(
            _packed_o_128_vec(
                self.traits, v_o, dc, g, self.lane_div_32, self.elem_dtype
            ),
            self._final_o_global(o_base, dc, g),
            _o_store_reg_128=self.o_store_reg_128,
            _store_atom_128=self.store_atom_128,
            o_div=self.o_div,
        )

    def _store_final_o_row(self, v_o, q_row):
        o_base = self._final_o_base(q_row)
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                self._store_final_o_128(v_o, dc, g, o_base)

    def store_final_o(self, v_o, q_row, m_row=None, l_row=None):
        self._store_final_o_row(v_o, q_row)


MFMA16_M = 16


def tok_off(traits, t):
    return (t // traits.SMEM_N_RPT) * traits.D_128B_SIZE + (
        t % traits.SMEM_N_RPT
    ) * traits.SMEM_V_LINE_STRIDE


def tok_off_dyn(traits, t):
    n = fx.Index(traits.SMEM_N_RPT)
    return (t // n) * fx.Index(traits.D_128B_SIZE) + (t % n) * fx.Index(
        traits.SMEM_V_LINE_STRIDE
    )


def lds_elem(traits, t, d):
    g = traits.D_128B_SIZE
    return (
        tok_off(traits, t)
        + (d // g) * traits.SMEM_N_RPT * traits.SMEM_V_LINE_STRIDE
        + (d % g)
    )


def a16_read_base(traits, lane, quad):
    return (
        tok_off_dyn(traits, (lane // fx.Index(MFMA16_M)) * fx.Index(quad))
        + ((lane % fx.Index(MFMA16_M)) // fx.Index(4))
        * fx.Index(traits.SMEM_V_LINE_STRIDE)
        + (lane % fx.Index(4)) * fx.Index(4)
    )


def a16_chunk_offset(traits, c):
    return lds_elem(traits, 0, c * MFMA16_M)
