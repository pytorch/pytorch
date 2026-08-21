# mypy: allow-untyped-defs

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from .flex_attn_mask import (
    evaluate_mask_program,
    is_causal_document_mask_program,
)
from .flex_attn_fwd_gfx950_mfma32 import build_flex_attn_fwd_mfma32_module
from .flex_attn_utils import (
    load_scalar,
    make_global_view,
    make_shared_view,
    store_scalar,
)

_LOG2E = 1.4426950408889634
_LN2 = math.log(2.0)
_NEG_BIG = -1.0e30


def _f32(value):
    return fx.Float32(value)


def _exp2(value):
    return fx.math.exp2(_f32(value))


def _maximum(lhs, rhs):
    return (lhs > rhs).select(lhs, rhs)


def build_flex_attn_fwd_module(
    *,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_q: int,
    seq_kv: int,
    qk_head_dim: int,
    v_head_dim: int,
    block_mask_batch: int,
    block_mask_heads: int,
    num_q_blocks: int,
    max_partial_blocks: int,
    max_full_blocks: int,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    causal_partial_blocks: bool,
    scale: float,
    mask_program=(),
    mask_program_output: int = 0,
    mask_buffer_shapes=(),
    mask_buffer_strides=(),
    q_stride=None,
    k_stride=None,
    v_stride=None,
    o_stride=None,
    output_stats_in_log2: bool = False,
):
    if (
        seq_q not in (4, 8)
        and seq_q % 128 == 0
        and seq_kv % 128 == 0
        and (qk_head_dim, v_head_dim) in ((128, 128), (192, 128))
        and sparse_q_block_size == 128
        and sparse_kv_block_size == 128
    ):
        return build_flex_attn_fwd_mfma32_module(
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            block_mask_batch=block_mask_batch,
            block_mask_heads=block_mask_heads,
            num_q_blocks=num_q_blocks,
            max_partial_blocks=max_partial_blocks,
            max_full_blocks=max_full_blocks,
            sparse_q_block_size=sparse_q_block_size,
            sparse_kv_block_size=sparse_kv_block_size,
            causal_partial_blocks=causal_partial_blocks,
            scale=scale,
            mask_program=mask_program,
            mask_program_output=mask_program_output,
            mask_buffer_shapes=mask_buffer_shapes,
            mask_buffer_strides=mask_buffer_strides,
            q_stride=q_stride,
            k_stride=k_stride,
            v_stride=v_stride,
            o_stride=o_stride,
            output_stats_in_log2=output_stats_in_log2,
        )
    return _build_flex_attn_fwd_module_mfma16(
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        seq_q=seq_q,
        seq_kv=seq_kv,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        block_mask_batch=block_mask_batch,
        block_mask_heads=block_mask_heads,
        num_q_blocks=num_q_blocks,
        max_partial_blocks=max_partial_blocks,
        max_full_blocks=max_full_blocks,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        causal_partial_blocks=causal_partial_blocks,
        scale=scale,
        mask_program=mask_program,
        mask_program_output=mask_program_output,
        mask_buffer_shapes=mask_buffer_shapes,
        mask_buffer_strides=mask_buffer_strides,
        q_stride=q_stride,
        k_stride=k_stride,
        v_stride=v_stride,
        o_stride=o_stride,
        output_stats_in_log2=output_stats_in_log2,
    )


def _build_flex_attn_fwd_module_mfma16(
    *,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_q: int,
    seq_kv: int,
    qk_head_dim: int,
    v_head_dim: int,
    block_mask_batch: int,
    block_mask_heads: int,
    num_q_blocks: int,
    max_partial_blocks: int,
    max_full_blocks: int,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    causal_partial_blocks: bool,
    scale: float,
    mask_program=(),
    mask_program_output: int = 0,
    mask_buffer_shapes=(),
    mask_buffer_strides=(),
    q_stride=None,
    k_stride=None,
    v_stride=None,
    o_stride=None,
    output_stats_in_log2: bool = False,
):
    BM = 64
    BN = 64
    NW = 4
    NT = NW * 64
    VPT = 8

    if (qk_head_dim, v_head_dim) not in ((128, 128), (192, 128)):
        raise ValueError(
            "FlyDSL forward requires (qk_head_dim, v_head_dim) "
            "to be (128, 128) or (192, 128)"
        )
    if num_q_heads % num_kv_heads:
        raise ValueError("FlyDSL forward requires Hq % Hkv == 0")
    if sparse_q_block_size != 128 or sparse_kv_block_size != 128:
        raise ValueError("FlyDSL forward requires sparse Q/KV block size 128")
    if seq_kv % sparse_kv_block_size:
        raise ValueError("FlyDSL forward requires Sk divisible by 128")
    if block_mask_batch not in (1, batch_size):
        raise ValueError("BlockMask batch dimension must be 1 or B")
    if block_mask_heads not in (1, num_kv_heads, num_q_heads):
        raise ValueError("BlockMask head dimension must be 1, Hkv, or Hq")
    if max_partial_blocks <= 0 or max_full_blocks <= 0:
        raise ValueError("FlyDSL forward requires non-empty index storage")
    if len(mask_buffer_shapes) != len(mask_buffer_strides):
        raise ValueError("mask buffer shape/stride descriptors must match")
    if len(mask_buffer_shapes) > 4:
        raise ValueError("FlyDSL forward supports at most four mask buffers")

    B = int(batch_size)
    HQ = int(num_q_heads)
    HKV = int(num_kv_heads)
    SQ = int(seq_q)
    SK = int(seq_kv)
    DQK = int(qk_head_dim)
    DV = int(v_head_dim)
    BMB = int(block_mask_batch)
    BMH = int(block_mask_heads)
    NQB = int(num_q_blocks)
    MAX_PARTIAL = int(max_partial_blocks)
    MAX_FULL = int(max_full_blocks)
    GROUP_SIZE = HQ // HKV
    DECODE = SQ in (4, 8)
    PACKED_Q_ROWS = GROUP_SIZE * SQ if DECODE else SQ
    Q_CHUNKS = (PACKED_Q_ROWS + BM - 1) // BM if DECODE else SQ // BM

    if DECODE:
        if BMH not in (1, HKV):
            raise ValueError(
                "FlyDSL forward decode requires a shared or per-KV-head BlockMask"
            )
        if NQB != 1:
            raise ValueError("FlyDSL forward decode requires one sparse Q block")
        if PACKED_Q_ROWS > 256 or PACKED_Q_ROWS % 32:
            raise ValueError(
                "FlyDSL forward decode requires Hq/Hkv * Sq to be a multiple "
                "of 32 and no greater than 256"
            )
    else:
        if SQ % 256:
            raise ValueError("FlyDSL forward prefill requires Sq divisible by 256")
        if NQB != SQ // sparse_q_block_size:
            raise ValueError("BlockMask Q rows must cover Sq with 128-row blocks")

    def contiguous_stride(heads, sequence, dimension):
        return (heads * sequence * dimension, sequence * dimension, dimension, 1)

    Q_STRIDE = tuple(q_stride or contiguous_stride(HQ, SQ, DQK))
    K_STRIDE = tuple(k_stride or contiguous_stride(HKV, SK, DQK))
    V_STRIDE = tuple(v_stride or contiguous_stride(HKV, SK, DV))
    O_STRIDE = tuple(o_stride or contiguous_stride(HQ, SQ, DV))
    SCALE_LOG2 = float(scale) * _LOG2E
    OUTPUT_STATS_IN_LOG2 = bool(output_stats_in_log2)
    CAUSAL_PARTIAL = bool(causal_partial_blocks)
    MASK_PROGRAM = tuple(mask_program)
    MASK_PROGRAM_OUTPUT = int(mask_program_output)
    MASK_BUFFER_SHAPES = tuple(tuple(shape) for shape in mask_buffer_shapes)
    MASK_BUFFER_STRIDES = tuple(tuple(stride) for stride in mask_buffer_strides)
    MASK_BUFFER_COUNT = len(MASK_BUFFER_SHAPES)
    MASK_BUFFER_SIZES = tuple(
        1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
        for shape, strides in zip(MASK_BUFFER_SHAPES, MASK_BUFFER_STRIDES)
    )
    CAUSAL_DOCUMENT_MASK = is_causal_document_mask_program(
        MASK_PROGRAM,
        MASK_PROGRAM_OUTPUT,
        MASK_BUFFER_STRIDES,
    )

    QP = DQK + 8
    KVP = max(DQK, DV) + 8
    Q_DCH = DQK // VPT
    V_DCH = DV // VPT
    Q_LOAD_IT = (BM * Q_DCH) // NT
    V_LOAD_IT = (BN * V_DCH) // NT
    CPB = sparse_kv_block_size // BN
    CE = [16 * (element // 4) + (element % 4) for element in range(16)]

    @fx.struct
    class FwdSmem:
        query: fx.Array[fx.BFloat16, BM * QP, 16]
        kv: fx.Array[fx.BFloat16, BN * KVP, 16]
        row_max: fx.Array[fx.Float32, BM, 16]
        row_sum: fx.Array[fx.Float32, BM, 16]
        alpha: fx.Array[fx.Float32, BM, 16]

    @flyc.kernel(known_block_size=[NT, 1, 1])
    def kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        LSE: fx.Tensor,
        MaxScores: fx.Tensor,
        KVNumBlocks: fx.Tensor,
        KVIndices: fx.Tensor,
        FullKVNumBlocks: fx.Tensor,
        FullKVIndices: fx.Tensor,
        MaskBuffer0: fx.Tensor,
        MaskBuffer1: fx.Tensor,
        MaskBuffer2: fx.Tensor,
        MaskBuffer3: fx.Tensor,
        O: fx.Tensor,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)
        lane16 = lane % fx.Int32(16)
        lane_group = lane // fx.Int32(16)
        batch = fx.Int32(fx.block_idx.z)
        q_chunk = fx.Int32(fx.block_idx.y)
        q_base = q_chunk * fx.Int32(BM)

        if const_expr(DECODE):
            kv_head = fx.Int32(fx.block_idx.x)
            head = kv_head * fx.Int32(GROUP_SIZE)
        else:
            head = fx.Int32(fx.block_idx.x)
            kv_head = head // fx.Int32(GROUP_SIZE)

        lds = fx.SharedAllocator().allocate(FwdSmem).peek()
        pquery = lds.query.ptr
        pkv = lds.kv.ptr
        pmax = lds.row_max.ptr
        psum = lds.row_sum.ptr
        palpha = lds.alpha.ptr

        gQ = make_global_view(Q, None, (B, HQ, SQ, DQK), Q_STRIDE)
        gK = make_global_view(K, None, (B, HKV, SK, DQK), K_STRIDE)
        gV = make_global_view(V, None, (B, HKV, SK, DV), V_STRIDE)

        q_smem = make_shared_view(pquery, (BM, DQK), (QP, 1))
        k_smem = make_shared_view(pkv, (BN, DQK), (KVP, 1))
        v_transposed = make_shared_view(pkv, (DV, BN), (1, KVP))
        score_anchor = fx.make_view(
            fx.get_iter(q_smem),
            fx.make_layout((BN, BM), (BM, 1)),
        )

        metadata_rows = BMB * BMH * NQB
        gKVN = make_global_view(KVNumBlocks, None, metadata_rows, 1)
        gKVI = make_global_view(
            KVIndices,
            None,
            metadata_rows * MAX_PARTIAL,
            1,
        )
        gFKVN = make_global_view(FullKVNumBlocks, None, metadata_rows, 1)
        gFKVI = make_global_view(
            FullKVIndices,
            None,
            metadata_rows * MAX_FULL,
            1,
        )
        gLSE = make_global_view(LSE, None, B * HQ * SQ, 1)
        gMax = make_global_view(MaxScores, None, B * HQ * SQ, 1)
        mask_buffers = []
        if const_expr(MASK_BUFFER_COUNT >= 1):
            mask_buffers.append(
                make_global_view(MaskBuffer0, None, MASK_BUFFER_SIZES[0], 1)
            )
        if const_expr(MASK_BUFFER_COUNT >= 2):
            mask_buffers.append(
                make_global_view(MaskBuffer1, None, MASK_BUFFER_SIZES[1], 1)
            )
        if const_expr(MASK_BUFFER_COUNT >= 3):
            mask_buffers.append(
                make_global_view(MaskBuffer2, None, MASK_BUFFER_SIZES[2], 1)
            )
        if const_expr(MASK_BUFFER_COUNT >= 4):
            mask_buffers.append(
                make_global_view(MaskBuffer3, None, MASK_BUFFER_SIZES[3], 1)
            )

        i32_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        f32_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        def load_i32(view, index):
            return fx.Int32(load_scalar(i32_atom, view, index, fx.Int32))

        def store_f32(view, index, value):
            store_scalar(f32_atom, view, index, value, fx.Float32)

        def row_coordinates(local_row):
            packed_row = q_base + local_row
            if const_expr(DECODE):
                valid = packed_row < fx.Int32(PACKED_Q_ROWS)
                safe_row = valid.select(packed_row, fx.Int32(0))
                row_head = (
                    kv_head * fx.Int32(GROUP_SIZE)
                    + safe_row // fx.Int32(SQ)
                )
                query_pos = safe_row % fx.Int32(SQ)
            else:
                valid = packed_row < fx.Int32(SQ)
                row_head = head
                query_pos = valid.select(packed_row, fx.Int32(0))
            return valid, row_head, query_pos

        g128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        u64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        tr16 = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans(16, 64),
            fx.BFloat16,
        )
        o16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)

        for load_step in fx.range_constexpr(Q_LOAD_IT):
            linear = fx.Int32(load_step * NT) + tid
            row = linear // fx.Int32(Q_DCH)
            chunk = linear % fx.Int32(Q_DCH)
            valid, row_head, query_pos = row_coordinates(row)
            fragment = fx.make_rmem_tensor(VPT, fx.BFloat16)
            source = fx.logical_divide(
                fx.slice(gQ, (batch, row_head, query_pos, None)),
                fx.make_layout(VPT, 1),
            )
            fx.copy(g128, fx.slice(source, (None, chunk)), fragment)
            loaded = fx.Vector(fragment.load())
            stored = fx.Vector.from_elements(
                [
                    valid.select(loaded[element], fx.BFloat16(0.0))
                    for element in fx.range_constexpr(VPT)
                ],
                fx.BFloat16,
            )
            fx.ptr_store(
                stored,
                pquery + row * fx.Int32(QP) + chunk * fx.Int32(VPT),
            )

        if tid < fx.Int32(BM):
            fx.ptr_store(_f32(_NEG_BIG), pmax + tid)
            fx.ptr_store(_f32(0.0), psum + tid)
            fx.ptr_store(_f32(1.0), palpha + tid)
        fx.gpu.barrier()

        atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        score_mma = fx.make_tiled_mma(
            atom,
            fx.make_layout((1, NW, 1), (0, 1, 0)),
        )
        output_mma = fx.make_tiled_mma(
            atom,
            fx.make_layout((NW, 1, 1), (1, 0, 0)),
        )
        score_thread = score_mma.thr_slice(tid)
        output_thread = output_mma.thr_slice(tid)

        copy_q = fx.make_tiled_copy_B(u64, score_mma).get_slice(tid)
        copy_k = fx.make_tiled_copy_A(u64, score_mma).get_slice(tid)
        copy_v = fx.make_tiled_copy_B(tr16, output_mma).get_slice(tid)
        copy_output = fx.make_tiled_copy_C(o16, output_mma).get_slice(tid)

        query_fragment = score_thread.make_fragment_B(q_smem)
        key_fragment = score_thread.make_fragment_A(k_smem)
        probability_fragment = output_thread.make_fragment_A(score_anchor)
        value_fragment = output_thread.make_fragment_B(v_transposed)

        fx.copy(
            u64,
            copy_q.partition_S(q_smem),
            copy_q.retile(query_fragment),
        )

        if const_expr(DECODE):
            output_offset = (
                batch * fx.Int32(O_STRIDE[0])
                + kv_head * fx.Int32(GROUP_SIZE * O_STRIDE[1])
                + q_base * fx.Int32(DV)
            )
            output_row_stride = DV
        else:
            output_offset = (
                batch * fx.Int32(O_STRIDE[0])
                + head * fx.Int32(O_STRIDE[1])
                + q_base * fx.Int32(O_STRIDE[2])
            )
            output_row_stride = O_STRIDE[2]
        output_tile = make_global_view(
            O,
            output_offset,
            (BM, DV),
            (output_row_stride, 1),
        )
        output_fragment = output_thread.make_fragment_C(output_tile)
        output_fragment.fill(0)

        if const_expr(BMB == 1):
            mask_batch = fx.Int32(0)
        else:
            mask_batch = batch
        if const_expr(BMH == 1):
            mask_head = fx.Int32(0)
        elif const_expr(BMH == HKV):
            mask_head = kv_head
        else:
            mask_head = head
        if const_expr(DECODE):
            mask_q_block = fx.Int32(0)
        else:
            mask_q_block = q_base // fx.Int32(sparse_q_block_size)
        mask_row = (
            (mask_batch * fx.Int32(BMH) + mask_head) * fx.Int32(NQB)
            + mask_q_block
        )

        def reduce_tile_max(scores, output_row):
            tile_max = scores[0]
            for element in fx.range_constexpr(1, 16):
                tile_max = _maximum(tile_max, scores[element])
            for shift in (16, 32):
                tile_max = _maximum(
                    tile_max,
                    _f32(
                        fx.gpu.shuffle_xor(tile_max, shift, 64)
                    ),
                )
            if lane_group == fx.Int32(0):
                old_max = _f32(fx.ptr_load(pmax + output_row))
                new_max = _maximum(old_max, tile_max)
                fx.ptr_store(new_max, pmax + output_row)
                fx.ptr_store(
                    _exp2(old_max - new_max),
                    palpha + output_row,
                )
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            return _f32(fx.ptr_load(pmax + output_row))

        def update_row_sum(probabilities, output_row):
            tile_sum = probabilities[0]
            for element in fx.range_constexpr(1, 16):
                tile_sum = tile_sum + probabilities[element]
            for shift in (16, 32):
                tile_sum = tile_sum + _f32(
                    fx.gpu.shuffle_xor(tile_sum, shift, 64)
                )
            if lane_group == fx.Int32(0):
                old_sum = _f32(fx.ptr_load(psum + output_row))
                alpha = _f32(fx.ptr_load(palpha + output_row))
                fx.ptr_store(
                    old_sum * alpha + tile_sum,
                    psum + output_row,
                )

        def process_tile(kv_chunk, masked):
            kv_base = kv_chunk * fx.Int32(BN)
            for load_step in fx.range_constexpr(Q_LOAD_IT):
                linear = fx.Int32(load_step * NT) + tid
                row = linear // fx.Int32(Q_DCH)
                chunk = linear % fx.Int32(Q_DCH)
                fragment = fx.make_rmem_tensor(VPT, fx.BFloat16)
                source = fx.logical_divide(
                    fx.slice(gK, (batch, kv_head, kv_base + row, None)),
                    fx.make_layout(VPT, 1),
                )
                fx.copy(g128, fx.slice(source, (None, chunk)), fragment)
                fx.ptr_store(
                    fx.Vector(fragment.load()),
                    pkv + row * fx.Int32(KVP) + chunk * fx.Int32(VPT),
                )
            fx.gpu.barrier()

            fx.copy(
                u64,
                copy_k.partition_S(k_smem),
                copy_k.retile(key_fragment),
            )
            fx.gpu.barrier()

            scores_fragment = score_thread.make_fragment_C(score_anchor)
            scores_fragment.fill(0)
            fx.gemm(
                atom,
                scores_fragment,
                key_fragment,
                query_fragment,
                scores_fragment,
            )

            for load_step in fx.range_constexpr(V_LOAD_IT):
                linear = fx.Int32(load_step * NT) + tid
                row = linear // fx.Int32(V_DCH)
                chunk = linear % fx.Int32(V_DCH)
                fragment = fx.make_rmem_tensor(VPT, fx.BFloat16)
                source = fx.logical_divide(
                    fx.slice(gV, (batch, kv_head, kv_base + row, None)),
                    fx.make_layout(VPT, 1),
                )
                fx.copy(g128, fx.slice(source, (None, chunk)), fragment)
                fx.ptr_store(
                    fx.Vector(fragment.load()),
                    pkv + row * fx.Int32(KVP) + chunk * fx.Int32(VPT),
                )
            fx.gpu.barrier()

            raw_scores = fx.Vector(scores_fragment.load())
            scores = []
            keep_values = []
            output_row = fx.Int32(16) * wave + lane16
            row_valid, row_head, query_pos = row_coordinates(output_row)
            for element in fx.range_constexpr(16):
                key_pos = (
                    kv_base
                    + fx.Int32(4) * lane_group
                    + fx.Int32(CE[element])
                )
                keep = row_valid
                if masked and const_expr(CAUSAL_PARTIAL or bool(MASK_PROGRAM)):
                    if const_expr(CAUSAL_DOCUMENT_MASK):
                        document_id = load_i32(
                            mask_buffers[0],
                            query_pos
                            * fx.Int32(MASK_BUFFER_STRIDES[0][0]),
                        )
                        document_start = load_i32(
                            mask_buffers[1],
                            document_id
                            * fx.Int32(MASK_BUFFER_STRIDES[1][0]),
                        )
                        element_keep = (query_pos >= key_pos) & (
                            key_pos >= document_start
                        )
                    elif const_expr(bool(MASK_PROGRAM)):
                        element_keep = evaluate_mask_program(
                            mask_program=MASK_PROGRAM,
                            mask_program_output=MASK_PROGRAM_OUTPUT,
                            mask_buffer_strides=MASK_BUFFER_STRIDES,
                            mask_buffers=mask_buffers,
                            load_i32=load_i32,
                            batch=batch,
                            head=row_head,
                            q_pos=query_pos,
                            kv_pos=key_pos,
                        )
                    else:
                        element_keep = key_pos <= (
                            query_pos + fx.Int32(SK - SQ)
                        )
                    keep = keep & element_keep
                keep_values.append(keep)
                score = _f32(raw_scores[element]) * _f32(SCALE_LOG2)
                scores.append(keep.select(score, _f32(_NEG_BIG)))

            row_max = reduce_tile_max(scores, output_row)
            probabilities = [
                keep_values[element].select(
                    _exp2(scores[element] - row_max),
                    _f32(0.0),
                )
                for element in fx.range_constexpr(16)
            ]

            output_values = fx.Vector(output_fragment.load())
            output_scale = fx.Vector.from_elements(
                [
                    _f32(
                        fx.ptr_load(
                            palpha
                            + fx.Int32(16) * wave
                            + fx.Int32(4) * lane_group
                            + fx.Int32(element % 4)
                        )
                    )
                    for element in fx.range_constexpr(output_values.numel)
                ],
                fx.Float32,
            )
            output_fragment.store((output_values * output_scale).ir_value())

            probability_fragment.store(
                fx.Vector.from_elements(
                    probabilities,
                    fx.Float32,
                )
                .to(fx.BFloat16)
                .ir_value()
            )
            update_row_sum(probabilities, output_row)

            fx.copy(
                tr16,
                copy_v.partition_S(v_transposed),
                copy_v.retile(value_fragment),
            )
            fx.gpu.barrier()
            fx.gemm(
                atom,
                output_fragment,
                probability_fragment,
                value_fragment,
                output_fragment,
            )

        full_count = load_i32(gFKVN, mask_row)
        full_base = mask_row * fx.Int32(MAX_FULL)
        for block_index in range(
            fx.Int32(0),
            full_count,
            fx.Int32(1),
        ):
            sparse_block = load_i32(
                gFKVI,
                full_base + fx.Int32(block_index),
            )
            for sub_block in fx.range_constexpr(CPB):
                process_tile(
                    sparse_block * fx.Int32(CPB) + fx.Int32(sub_block),
                    False,
                )

        partial_count = load_i32(gKVN, mask_row)
        partial_base = mask_row * fx.Int32(MAX_PARTIAL)
        for block_index in range(
            fx.Int32(0),
            partial_count,
            fx.Int32(1),
        ):
            sparse_block = load_i32(
                gKVI,
                partial_base + fx.Int32(block_index),
            )
            for sub_block in fx.range_constexpr(CPB):
                process_tile(
                    sparse_block * fx.Int32(CPB) + fx.Int32(sub_block),
                    True,
                )

        output_values = fx.Vector(output_fragment.load())
        normalizer_values = []
        for element in fx.range_constexpr(output_values.numel):
            value = _f32(
                fx.ptr_load(
                    psum
                    + fx.Int32(16) * wave
                    + fx.Int32(4) * lane_group
                    + fx.Int32(element % 4)
                )
            )
            normalizer_values.append(
                (value > _f32(0.0)).select(
                    _f32(1.0) / value,
                    _f32(0.0),
                )
            )
        normalizer = fx.Vector.from_elements(
            normalizer_values,
            fx.Float32,
        )
        output_fragment.store((output_values * normalizer).ir_value())
        output_bf16 = fx.make_fragment_like(
            output_fragment,
            fx.BFloat16.ir_type,
        )
        output_bf16.store(
            fx.Vector(output_fragment.load()).to(fx.BFloat16).ir_value()
        )

        output_row = fx.Int32(16) * wave + lane16
        row_sum = _f32(fx.ptr_load(psum + output_row))
        has_values = row_sum > _f32(0.0)
        output_valid, output_head, output_q_pos = row_coordinates(output_row)
        output_source = copy_output.retile(output_bf16)
        output_predicate = fx.make_fragment_like(
            output_source,
            dtype=fx.Boolean,
        )
        output_predicate.fill(output_valid)
        fx.copy(
            o16,
            output_source,
            copy_output.partition_S(output_tile),
            pred=output_predicate,
        )

        row_max = _f32(fx.ptr_load(pmax + output_row))
        lse_value = row_max + fx.math.log2(row_sum)
        max_value = row_max
        if const_expr(not OUTPUT_STATS_IN_LOG2):
            lse_value = lse_value * _f32(_LN2)
            max_value = max_value * _f32(_LN2)
        lse_value = has_values.select(
            lse_value,
            _f32(float("-inf")),
        )
        max_value = has_values.select(
            max_value,
            _f32(float("-inf")),
        )
        stats_offset = (
            (batch * fx.Int32(HQ) + output_head) * fx.Int32(SQ)
            + output_q_pos
        )
        stats_mask = output_valid & (lane_group == fx.Int32(0))
        if stats_mask:
            store_f32(gLSE, stats_offset, lse_value)
            store_f32(gMax, stats_offset, max_value)

    def launch_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        LSE: fx.Tensor,
        MaxScores: fx.Tensor,
        KVNumBlocks: fx.Tensor,
        KVIndices: fx.Tensor,
        FullKVNumBlocks: fx.Tensor,
        FullKVIndices: fx.Tensor,
        MaskBuffer0: fx.Tensor,
        MaskBuffer1: fx.Tensor,
        MaskBuffer2: fx.Tensor,
        MaskBuffer3: fx.Tensor,
        O: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(
            Q,
            K,
            V,
            LSE,
            MaxScores,
            KVNumBlocks,
            KVIndices,
            FullKVNumBlocks,
            FullKVIndices,
            MaskBuffer0,
            MaskBuffer1,
            MaskBuffer2,
            MaskBuffer3,
            O,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": "256,256",
                "passthrough": [
                    [
                        "denormal-fp-math-f32",
                        "preserve-sign,preserve-sign",
                    ],
                    ["no-nans-fp-math", "true"],
                    ["unsafe-fp-math", "true"],
                ],
            },
        ).launch(
            grid=(
                HKV if DECODE else HQ,
                Q_CHUNKS,
                B,
            ),
            block=(NT, 1, 1),
            stream=stream,
        )

    if MASK_BUFFER_COUNT == 0:

        @flyc.jit
        def launch(
            Q: fx.Tensor,
            K: fx.Tensor,
            V: fx.Tensor,
            LSE: fx.Tensor,
            MaxScores: fx.Tensor,
            KVNumBlocks: fx.Tensor,
            KVIndices: fx.Tensor,
            FullKVNumBlocks: fx.Tensor,
            FullKVIndices: fx.Tensor,
            O: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                Q,
                K,
                V,
                LSE,
                MaxScores,
                KVNumBlocks,
                KVIndices,
                FullKVNumBlocks,
                FullKVIndices,
                KVNumBlocks,
                KVNumBlocks,
                KVNumBlocks,
                KVNumBlocks,
                O,
                stream,
            )

    elif MASK_BUFFER_COUNT == 1:

        @flyc.jit
        def launch(
            Q: fx.Tensor,
            K: fx.Tensor,
            V: fx.Tensor,
            LSE: fx.Tensor,
            MaxScores: fx.Tensor,
            KVNumBlocks: fx.Tensor,
            KVIndices: fx.Tensor,
            FullKVNumBlocks: fx.Tensor,
            FullKVIndices: fx.Tensor,
            MaskBuffer0: fx.Tensor,
            O: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                Q,
                K,
                V,
                LSE,
                MaxScores,
                KVNumBlocks,
                KVIndices,
                FullKVNumBlocks,
                FullKVIndices,
                MaskBuffer0,
                KVNumBlocks,
                KVNumBlocks,
                KVNumBlocks,
                O,
                stream,
            )

    elif MASK_BUFFER_COUNT == 2:

        @flyc.jit
        def launch(
            Q: fx.Tensor,
            K: fx.Tensor,
            V: fx.Tensor,
            LSE: fx.Tensor,
            MaxScores: fx.Tensor,
            KVNumBlocks: fx.Tensor,
            KVIndices: fx.Tensor,
            FullKVNumBlocks: fx.Tensor,
            FullKVIndices: fx.Tensor,
            MaskBuffer0: fx.Tensor,
            MaskBuffer1: fx.Tensor,
            O: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                Q,
                K,
                V,
                LSE,
                MaxScores,
                KVNumBlocks,
                KVIndices,
                FullKVNumBlocks,
                FullKVIndices,
                MaskBuffer0,
                MaskBuffer1,
                KVNumBlocks,
                KVNumBlocks,
                O,
                stream,
            )

    elif MASK_BUFFER_COUNT == 3:

        @flyc.jit
        def launch(
            Q: fx.Tensor,
            K: fx.Tensor,
            V: fx.Tensor,
            LSE: fx.Tensor,
            MaxScores: fx.Tensor,
            KVNumBlocks: fx.Tensor,
            KVIndices: fx.Tensor,
            FullKVNumBlocks: fx.Tensor,
            FullKVIndices: fx.Tensor,
            MaskBuffer0: fx.Tensor,
            MaskBuffer1: fx.Tensor,
            MaskBuffer2: fx.Tensor,
            O: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                Q,
                K,
                V,
                LSE,
                MaxScores,
                KVNumBlocks,
                KVIndices,
                FullKVNumBlocks,
                FullKVIndices,
                MaskBuffer0,
                MaskBuffer1,
                MaskBuffer2,
                KVNumBlocks,
                O,
                stream,
            )

    else:

        @flyc.jit
        def launch(
            Q: fx.Tensor,
            K: fx.Tensor,
            V: fx.Tensor,
            LSE: fx.Tensor,
            MaxScores: fx.Tensor,
            KVNumBlocks: fx.Tensor,
            KVIndices: fx.Tensor,
            FullKVNumBlocks: fx.Tensor,
            FullKVIndices: fx.Tensor,
            MaskBuffer0: fx.Tensor,
            MaskBuffer1: fx.Tensor,
            MaskBuffer2: fx.Tensor,
            MaskBuffer3: fx.Tensor,
            O: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                Q,
                K,
                V,
                LSE,
                MaxScores,
                KVNumBlocks,
                KVIndices,
                FullKVNumBlocks,
                FullKVIndices,
                MaskBuffer0,
                MaskBuffer1,
                MaskBuffer2,
                MaskBuffer3,
                O,
                stream,
            )

    return launch
