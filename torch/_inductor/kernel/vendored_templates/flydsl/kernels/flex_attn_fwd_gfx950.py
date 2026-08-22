# mypy: allow-untyped-defs

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import Vector as Vec

from .flex_attn_mask import (
    evaluate_mask_program,
    is_causal_document_mask_program,
)
from .flex_attn_utils import load_scalar, make_global_view, store_scalar

_LOG2E = 1.4426950408889634
_LN2 = math.log(2.0)
_NEG_BIG = -1.0e30
_FWD_COMPILE_HINTS = {
    "fast_fp_math": True,
}


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
    """Build the gfx950 prefill kernel.

    A 256-thread CTA owns 128 query rows. Each of its four waves keeps one
    32-row Q tile and the corresponding 32x128 output tile in registers. K/V
    are staged once per CTA and consumed by all four waves.
    """

    BM = 128
    BN = 64
    NW = 4
    NT = NW * 64
    VPT = 8
    MFMA_MN = 32

    if (qk_head_dim, v_head_dim) not in ((128, 128), (192, 128)):
        raise ValueError(
            "FlyDSL forward requires (qk_head_dim, v_head_dim) "
            "to be (128, 128) or (192, 128)"
        )
    if num_q_heads % num_kv_heads:
        raise ValueError("FlyDSL forward requires Hq % Hkv == 0")
    if sparse_q_block_size != BM or sparse_kv_block_size != 128:
        raise ValueError("FlyDSL forward requires sparse block size 128")
    if seq_q % BM or seq_kv % sparse_kv_block_size:
        raise ValueError("FlyDSL forward requires Sq/Sk divisible by 128")
    if num_q_blocks != seq_q // BM:
        raise ValueError("BlockMask Q rows must cover Sq with 128-row blocks")
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
    Q_CHUNKS = SQ // BM
    CPB = sparse_kv_block_size // BN
    K_STEPS = DQK // 16
    D_CHUNKS = DV // MFMA_MN
    K_DCH = DQK // VPT
    V_DCH = DV // VPT
    Q_LOAD_IT = (BM * K_DCH) // NT
    K_LOAD_IT = (BN * K_DCH) // NT
    V_LOAD_IT = (BN * V_DCH) // NT
    DQK_SUBTILES = DQK // MFMA_MN

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

    @fx.struct
    class FwdSmem:
        query: fx.Array[fx.BFloat16, BM * DQK, 16]
        # K needs the largest allocation. V reuses the same storage after every
        # wave has consumed K into registers.
        kv: fx.Array[fx.BFloat16, BN * DQK, 16]

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
        tid = fx.thread_idx.x
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)
        lane_row = lane % fx.Int32(MFMA_MN)
        lane_half = lane // fx.Int32(MFMA_MN)
        batch = fx.block_idx.z
        head = fx.block_idx.x
        kv_head = head // fx.Int32(GROUP_SIZE)
        q_chunk = fx.block_idx.y
        q_base = q_chunk * fx.Int32(BM)
        query_pos = q_base + wave * fx.Int32(MFMA_MN) + lane_row

        lds = fx.SharedAllocator().allocate(FwdSmem).peek()
        pquery = lds.query.ptr
        pkv = lds.kv.ptr

        gQ = make_global_view(Q, None, (B, HQ, SQ, DQK), Q_STRIDE)
        gK = make_global_view(K, None, (B, HKV, SK, DQK), K_STRIDE)
        gV = make_global_view(V, None, (B, HKV, SK, DV), V_STRIDE)
        gO = make_global_view(O, None, (B, HQ, SQ, DV), O_STRIDE)

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
        o64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)

        def load_i32(view, index):
            return load_scalar(i32_atom, view, index, fx.Int32)

        def load_uniform_i32(view, index):
            value = load_i32(view, index)
            return fx.gpu.shuffle_idx(value, 0, 64)

        def store_f32(view, index, value):
            store_scalar(f32_atom, view, index, value, fx.Float32)

        g128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        dma128 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        lds_dma_ptr_type = fx.PointerType.get(
            fx.BFloat16.ir_type,
            2,
            16,
        )
        tr16 = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans(16, 64),
            fx.BFloat16,
        )
        mma_atom = fx.make_mma_atom(
            fx.rocdl.MFMA(MFMA_MN, MFMA_MN, 16, fx.BFloat16)
        )

        def make_fragment(value, size, dtype):
            fragment = fx.make_rmem_tensor(size, dtype)
            fragment.store(Vec(value))
            return fragment

        def mfma(a_v8, b_v8, c_v16):
            a_fragment = make_fragment(a_v8, 8, fx.BFloat16)
            b_fragment = make_fragment(b_v8, 8, fx.BFloat16)
            c_fragment = make_fragment(c_v16, 16, fx.Float32)
            fx.gemm(
                mma_atom,
                c_fragment,
                a_fragment,
                b_fragment,
                c_fragment,
            )
            return c_fragment.load()

        def read_v8bf16_static(pointer_base, element_offset):
            pointer = fx.add_offset(
                pointer_base,
                fx.make_int_tuple(fx.Int32(element_offset)),
            )
            return fx.make_view(pointer, fx.make_layout(8, 1)).load()

        def k_swizzled_offset(row, column):
            # Match the conflict-free 32x32 K subtile layout used by the
            # repository's gfx950 SWA kernel.
            swizzled_column = (
                column
                ^ ((row & fx.Int32(8)) << fx.Int32(1))
                ^ ((row & fx.Int32(16)) >> fx.Int32(1))
            )
            return row * fx.Int32(MFMA_MN) + swizzled_column

        # Pre-scale Q into a swizzled LDS tile. Keeping Q in LDS instead of
        # carrying all 12 MFMA packs across the sparse loop avoids the gfx950
        # VGPR cliff for Dqk=192.
        q_scale = Vec.from_elements([_f32(SCALE_LOG2)], fx.Float32).broadcast_to(
            VPT
        )
        for load_step in fx.range_constexpr(Q_LOAD_IT):
            linear = fx.Int32(load_step * NT) + tid
            row = linear // fx.Int32(K_DCH)
            chunk = linear % fx.Int32(K_DCH)
            column = chunk * fx.Int32(VPT)
            q_fragment = fx.make_rmem_tensor(VPT, fx.BFloat16)
            source = fx.logical_divide(
                fx.slice(gQ, (batch, head, q_base + row, None)),
                fx.make_layout(VPT, 1),
            )
            fx.copy(
                g128,
                fx.slice(source, (None, chunk)),
                q_fragment,
            )
            q_value = Vec(q_fragment.load())
            q_scaled = Vec(q_value.to(fx.Float32)) * q_scale
            q_row_group = row // fx.Int32(MFMA_MN)
            q_row_in_group = row % fx.Int32(MFMA_MN)
            q_d_subtile = column // fx.Int32(MFMA_MN)
            q_column_in_subtile = column % fx.Int32(MFMA_MN)
            q_lds_offset = (
                (q_row_group * fx.Int32(DQK_SUBTILES) + q_d_subtile)
                * fx.Int32(MFMA_MN * MFMA_MN)
                + k_swizzled_offset(q_row_in_group, q_column_in_subtile)
            )
            fx.ptr_store(
                Vec(q_scaled).to(fx.BFloat16),
                pquery + q_lds_offset,
            )
        fx.gpu.barrier()

        def load_q_pack(k_step):
            d_subtile = k_step // 2
            d_half = k_step % 2
            pointer = q_odd_pointer if d_half else q_even_pointer
            return read_v8bf16_static(
                pointer,
                d_subtile * MFMA_MN * MFMA_MN,
            )

        zero16 = Vec.filled(16, 0.0, fx.Float32)
        output = [zero16 for _ in fx.range_constexpr(D_CHUNKS)]
        running_max = _f32(_NEG_BIG)
        running_sum = _f32(0.0)

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
        mask_q_block = q_chunk
        mask_row = (
            (mask_batch * fx.Int32(BMH) + mask_head) * fx.Int32(NQB)
            + mask_q_block
        )

        def stage_k(kv_base):
            for load_step in fx.range_constexpr(K_LOAD_IT):
                linear = fx.Int32(load_step * NT) + tid
                lds_offset = linear * fx.Int32(VPT)
                subtile = lds_offset // fx.Int32(MFMA_MN * MFMA_MN)
                within_subtile = lds_offset % fx.Int32(MFMA_MN * MFMA_MN)
                row_in_half = within_subtile // fx.Int32(MFMA_MN)
                swizzled_column = within_subtile % fx.Int32(MFMA_MN)
                row_half = subtile // fx.Int32(DQK_SUBTILES)
                d_subtile = subtile % fx.Int32(DQK_SUBTILES)
                row = row_half * fx.Int32(MFMA_MN) + row_in_half
                column = (
                    d_subtile * fx.Int32(MFMA_MN)
                    + (
                        swizzled_column
                        ^ ((row_in_half & fx.Int32(8)) << fx.Int32(1))
                        ^ ((row_in_half & fx.Int32(16)) >> fx.Int32(1))
                    )
                )
                chunk = column // fx.Int32(VPT)
                source = fx.logical_divide(
                    fx.slice(gK, (batch, kv_head, kv_base + row, None)),
                    fx.make_layout(VPT, 1),
                )
                destination_pointer = fx.inttoptr(
                    lds_dma_ptr_type,
                    fx.Int32(
                        fx.ptrtoint(
                            fx.add_offset(
                                pkv,
                                fx.make_int_tuple(lds_offset),
                            )
                        )
                    ),
                )
                destination = fx.make_view(
                    destination_pointer,
                    fx.make_layout(1, 1),
                )
                fx.copy(
                    dma128,
                    fx.slice(source, (None, chunk)),
                    destination,
                )

        def load_k_pack(k_step, high_half):
            d_subtile = k_step // 2
            d_half = k_step % 2
            pointer = k_odd_pointer if d_half else k_even_pointer
            row_half_offset = (
                DQK_SUBTILES * MFMA_MN * MFMA_MN
                if high_half
                else 0
            )
            return read_v8bf16_static(
                pointer,
                row_half_offset + d_subtile * MFMA_MN * MFMA_MN,
            )

        def stage_v(kv_base):
            for load_step in fx.range_constexpr(V_LOAD_IT):
                linear = fx.Int32(load_step * NT) + tid
                lds_offset = linear * fx.Int32(VPT)
                subtile = lds_offset // fx.Int32(8 * MFMA_MN)
                within_subtile = lds_offset % fx.Int32(8 * MFMA_MN)
                row_in_group = within_subtile // fx.Int32(MFMA_MN)
                column_in_subtile = within_subtile % fx.Int32(MFMA_MN)
                row_group = subtile // fx.Int32(D_CHUNKS)
                d_subtile = subtile % fx.Int32(D_CHUNKS)
                row = row_group * fx.Int32(8) + row_in_group
                column = d_subtile * fx.Int32(MFMA_MN) + column_in_subtile
                chunk = column // fx.Int32(VPT)
                source = fx.logical_divide(
                    fx.slice(gV, (batch, kv_head, kv_base + row, None)),
                    fx.make_layout(VPT, 1),
                )
                destination_pointer = fx.inttoptr(
                    lds_dma_ptr_type,
                    fx.Int32(
                        fx.ptrtoint(
                            fx.add_offset(
                                pkv,
                                fx.make_int_tuple(lds_offset),
                            )
                        )
                    ),
                )
                destination = fx.make_view(
                    destination_pointer,
                    fx.make_layout(1, 1),
                )
                fx.copy(
                    dma128,
                    fx.slice(source, (None, chunk)),
                    destination,
                )

        def load_v_pack(probability_pack, d_chunk):
            halves = []
            for half in fx.range_constexpr(2):
                subtile = (
                    (probability_pack * 2 + half) * D_CHUNKS + d_chunk
                )
                pointer = fx.add_offset(
                    v_lane_pointer,
                    fx.make_int_tuple(
                        fx.Int32(subtile * 8 * MFMA_MN)
                    ),
                )
                source = fx.make_view(pointer, fx.make_layout(4, 1))
                destination = fx.make_rmem_tensor(4, fx.BFloat16)
                fx.copy(tr16, source, destination)
                halves.append(Vec(destination.load()))
            return halves[0].shuffle(halves[1], list(range(8))).ir_value()

        q_wave_offset = wave * fx.Int32(
            DQK_SUBTILES * MFMA_MN * MFMA_MN
        )
        even_column = lane_half * fx.Int32(VPT)
        odd_column = fx.Int32(16) + even_column
        q_even_pointer = fx.add_offset(
            pquery,
            fx.make_int_tuple(
                q_wave_offset + k_swizzled_offset(lane_row, even_column)
            ),
        )
        q_odd_pointer = fx.add_offset(
            pquery,
            fx.make_int_tuple(
                q_wave_offset + k_swizzled_offset(lane_row, odd_column)
            ),
        )
        k_even_pointer = fx.add_offset(
            pkv,
            fx.make_int_tuple(
                k_swizzled_offset(lane_row, even_column)
            ),
        )
        k_odd_pointer = fx.add_offset(
            pkv,
            fx.make_int_tuple(
                k_swizzled_offset(lane_row, odd_column)
            ),
        )
        v_row_offset = (
            (lane % fx.Int32(16)) // fx.Int32(4)
            + lane_half * fx.Int32(4)
        )
        v_column_offset = (
            (lane % fx.Int32(4)) * fx.Int32(4)
            + ((lane % fx.Int32(MFMA_MN)) // fx.Int32(16))
            * fx.Int32(16)
        )
        v_lane_pointer = fx.add_offset(
            pkv,
            fx.make_int_tuple(
                v_row_offset * fx.Int32(MFMA_MN) + v_column_offset
            ),
        )

        def process_tile(
            kv_chunk,
            masked,
            tile_output,
            tile_running_max,
            tile_running_sum,
        ):
            kv_base = kv_chunk * fx.Int32(BN)
            stage_k(kv_base)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            fx.gpu.barrier()

            scores_lo = zero16
            scores_hi = zero16
            for k_step in fx.range_constexpr(K_STEPS):
                query_pack = load_q_pack(k_step)
                key_lo = load_k_pack(k_step, False)
                key_hi = load_k_pack(k_step, True)
                scores_lo = mfma(key_lo, query_pack, scores_lo)
                scores_hi = mfma(key_hi, query_pack, scores_hi)

            # Every wave must finish its K reads before the shared allocation is
            # reused for V.
            fx.gpu.barrier()
            stage_v(kv_base)

            raw_lo = Vec(scores_lo)
            raw_hi = Vec(scores_hi)
            if const_expr(CAUSAL_DOCUMENT_MASK):
                document_start = fx.Int32(0)
                if masked:
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
            score_values = []
            keep_values = []
            for half in fx.range_constexpr(2):
                raw = raw_lo if half == 0 else raw_hi
                for element in fx.range_constexpr(16):
                    key_pos = (
                        kv_base
                        + fx.Int32(32 * half)
                        + lane_half * fx.Int32(4)
                        + fx.Int32(8 * (element // 4) + element % 4)
                    )
                    keep = fx.Int32(1) == fx.Int32(1)
                    if const_expr(CAUSAL_DOCUMENT_MASK):
                        mask_keep = (query_pos >= key_pos) & (
                            key_pos >= document_start
                        )
                        if isinstance(masked, bool):
                            keep = mask_keep if masked else keep
                        else:
                            keep = (~masked) | mask_keep
                    elif masked and const_expr(
                        CAUSAL_PARTIAL or bool(MASK_PROGRAM)
                    ):
                        if const_expr(bool(MASK_PROGRAM)):
                            keep = evaluate_mask_program(
                                mask_program=MASK_PROGRAM,
                                mask_program_output=MASK_PROGRAM_OUTPUT,
                                mask_buffer_strides=MASK_BUFFER_STRIDES,
                                mask_buffers=mask_buffers,
                                load_i32=load_i32,
                                batch=batch,
                                head=head,
                                q_pos=query_pos,
                                kv_pos=key_pos,
                            )
                        else:
                            keep = key_pos <= (
                                query_pos + fx.Int32(SK - SQ)
                            )
                    keep_values.append(keep)
                    score_values.append(
                        keep.select(_f32(raw[element]), _f32(_NEG_BIG))
                    )

            local_max = score_values[0]
            for element in fx.range_constexpr(1, 32):
                local_max = _maximum(local_max, score_values[element])
            peer_max = _f32(
                fx.gpu.shuffle_xor(local_max, 32, 64)
            )
            tile_max = _maximum(local_max, peer_max)
            new_max = _maximum(tile_running_max, tile_max)
            correction = _exp2(tile_running_max - new_max)

            correction_vec = Vec.from_elements(
                [correction], fx.Float32
            ).broadcast_to(16)
            for d_chunk in fx.range_constexpr(D_CHUNKS):
                tile_output[d_chunk] = (
                    Vec(tile_output[d_chunk]) * correction_vec
                )

            local_sum = _f32(0.0)
            probability_packs = []
            for pack in fx.range_constexpr(4):
                pack_probabilities = []
                for pack_element in fx.range_constexpr(8):
                    element = pack * 8 + pack_element
                    probability = keep_values[element].select(
                        _exp2(score_values[element] - new_max),
                        _f32(0.0),
                    )
                    pack_probabilities.append(probability)
                    local_sum = local_sum + probability
                probability_packs.append(
                    Vec.from_elements(
                        pack_probabilities,
                        fx.Float32,
                    )
                    .to(fx.BFloat16)
                    .ir_value()
                )
            peer_sum = _f32(
                fx.gpu.shuffle_xor(local_sum, 32, 64)
            )
            tile_sum = local_sum + peer_sum
            tile_running_sum = (
                tile_running_sum * correction + tile_sum
            )
            tile_running_max = new_max

            # V writes were issued before the register-only softmax. Synchronize
            # only when the LDS data is actually consumed.
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            fx.gpu.barrier()
            for probability_pack in fx.range_constexpr(4):
                for d_chunk in fx.range_constexpr(D_CHUNKS):
                    value_pack = load_v_pack(probability_pack, d_chunk)
                    tile_output[d_chunk] = mfma(
                        value_pack,
                        probability_packs[probability_pack],
                        tile_output[d_chunk],
                    )

            # Protect V from the next tile's K staging.
            fx.gpu.barrier()
            return tile_output, tile_running_max, tile_running_sum

        full_count = load_uniform_i32(gFKVN, mask_row)
        partial_count = load_uniform_i32(gKVN, mask_row)
        full_base = mask_row * fx.Int32(MAX_FULL)
        partial_base = mask_row * fx.Int32(MAX_PARTIAL)
        initial_state = [running_max, running_sum] + output
        if const_expr(CAUSAL_DOCUMENT_MASK):
            total_count = full_count + partial_count
            final_results = initial_state
            for block_index, iter_args in range(
                fx.Int32(0),
                total_count,
                fx.Int32(1),
                init=initial_state,
            ):
                iter_max = _f32(iter_args[0])
                iter_sum = _f32(iter_args[1])
                iter_output = [
                    iter_args[2 + d_chunk]
                    for d_chunk in fx.range_constexpr(D_CHUNKS)
                ]
                block_index_i32 = fx.Int32(block_index)
                is_partial = block_index_i32 >= full_count
                sparse_block = fx.Int32(0)
                if is_partial:
                    sparse_block = load_uniform_i32(
                        gKVI,
                        partial_base + block_index_i32 - full_count,
                    )
                else:
                    sparse_block = load_uniform_i32(
                        gFKVI,
                        full_base + block_index_i32,
                    )
                for sub_block in fx.range_constexpr(CPB):
                    iter_output, iter_max, iter_sum = process_tile(
                        sparse_block * fx.Int32(CPB) + fx.Int32(sub_block),
                        is_partial,
                        iter_output,
                        iter_max,
                        iter_sum,
                    )
                final_results = yield [iter_max, iter_sum] + iter_output
        else:
            full_results = initial_state
            for block_index, iter_args in range(
                fx.Int32(0),
                full_count,
                fx.Int32(1),
                init=initial_state,
            ):
                iter_max = _f32(iter_args[0])
                iter_sum = _f32(iter_args[1])
                iter_output = [
                    iter_args[2 + d_chunk]
                    for d_chunk in fx.range_constexpr(D_CHUNKS)
                ]
                sparse_block = load_uniform_i32(
                    gFKVI,
                    full_base + fx.Int32(block_index),
                )
                for sub_block in fx.range_constexpr(CPB):
                    iter_output, iter_max, iter_sum = process_tile(
                        sparse_block * fx.Int32(CPB) + fx.Int32(sub_block),
                        False,
                        iter_output,
                        iter_max,
                        iter_sum,
                    )
                full_results = yield [iter_max, iter_sum] + iter_output

            running_max = _f32(full_results[0])
            running_sum = _f32(full_results[1])
            output = [
                full_results[2 + d_chunk]
                for d_chunk in fx.range_constexpr(D_CHUNKS)
            ]
            partial_state = [running_max, running_sum] + output
            final_results = partial_state
            for block_index, iter_args in range(
                fx.Int32(0),
                partial_count,
                fx.Int32(1),
                init=partial_state,
            ):
                iter_max = _f32(iter_args[0])
                iter_sum = _f32(iter_args[1])
                iter_output = [
                    iter_args[2 + d_chunk]
                    for d_chunk in fx.range_constexpr(D_CHUNKS)
                ]
                sparse_block = load_uniform_i32(
                    gKVI,
                    partial_base + fx.Int32(block_index),
                )
                for sub_block in fx.range_constexpr(CPB):
                    iter_output, iter_max, iter_sum = process_tile(
                        sparse_block * fx.Int32(CPB) + fx.Int32(sub_block),
                        True,
                        iter_output,
                        iter_max,
                        iter_sum,
                    )
                final_results = yield [iter_max, iter_sum] + iter_output

        running_max = _f32(final_results[0])
        running_sum = _f32(final_results[1])
        output = [
            final_results[2 + d_chunk]
            for d_chunk in fx.range_constexpr(D_CHUNKS)
        ]

        inverse_sum = (running_sum > _f32(0.0)).select(
            _f32(1.0) / running_sum,
            _f32(0.0),
        )
        inverse_vec = Vec.from_elements(
            [inverse_sum], fx.Float32
        ).broadcast_to(16)

        output_row = fx.logical_divide(
            fx.slice(gO, (batch, head, query_pos, None)),
            fx.make_layout(4, 1),
        )
        for d_chunk in fx.range_constexpr(D_CHUNKS):
            normalized = Vec(output[d_chunk]) * inverse_vec
            for column_group in fx.range_constexpr(4):
                values = Vec.from_elements(
                    [
                        normalized[column_group * 4 + element]
                        for element in fx.range_constexpr(4)
                    ],
                    fx.Float32,
                ).to(fx.BFloat16)
                column = (
                    fx.Int32(d_chunk * MFMA_MN)
                    + lane_half * fx.Int32(4)
                    + fx.Int32(column_group * 8)
                )
                fragment = fx.make_rmem_tensor(4, fx.BFloat16)
                fragment.store(values.ir_value())
                fx.copy(
                    o64,
                    fragment,
                    fx.slice(output_row, (None, column // fx.Int32(4))),
                )

        if lane_half == fx.Int32(0):
            has_values = running_sum > _f32(0.0)
            lse_value = running_max + fx.math.log2(running_sum)
            max_value = running_max
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
                (batch * fx.Int32(HQ) + head) * fx.Int32(SQ)
                + query_pos
            )
            store_f32(gLSE, stats_offset, lse_value)
            store_f32(gMax, stats_offset, max_value)

    def launch_kernel(
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
                "rocdl.waves_per_eu": 1,
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
            grid=(HQ, Q_CHUNKS, B),
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

    launch.compile_hints = dict(_FWD_COMPILE_HINTS)
    return launch
