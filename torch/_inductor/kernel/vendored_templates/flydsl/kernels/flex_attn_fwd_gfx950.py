# mypy: allow-untyped-defs

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import Vector as Vec

from .flex_attn_utils import (
    fast_exp2,
    is_causal_document_mask_program,
    make_global_view,
    make_mask_buffers,
    make_mask_evaluator,
    schedule_fwd_pv_pipeline,
    schedule_fwd_qk_pipeline,
    schedule_fwd_softmax_pipeline,
)

_LOG2E = 1.4426950408889634
_LN2 = math.log(2.0)
_NEG_BIG = -1.0e30
_FWD_COMPILE_HINTS = {
    "fast_fp_math": True,
}
# Four-wave CTAs reduce launch count once the 128-row prefill grid remains
# large enough to keep gfx950 occupied.
_FOUR_WAVE_PREFILL_MIN_CTAS = 512


def _f32(value):
    return fx.Float32(value)


def _exp2(value):
    return fast_exp2(_f32(value))


def _maximum(lhs, rhs):
    return (lhs > rhs).select(lhs, rhs)


def _causal_window_size(mask_program, mask_program_output):
    program = tuple(mask_program)
    if (
        len(program) == 5
        and program[0] == ("ge", 2, 3)
        and program[1] == ("sub", 2, 3)
        and len(program[2]) == 2
        and program[2][0] == "const_i32"
        and program[3] == ("lt", 5, 6)
        and program[4] == ("and", 4, 7)
        and int(mask_program_output) == 8
    ):
        window_size = int(program[2][1])
        return window_size if window_size > 0 else None
    return None


def _select_owner_waves(
    *,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_q: int,
    seq_kv: int,
    qk_head_dim: int,
) -> int:
    """Choose how many independent 32-row query owners share one CTA."""
    if seq_q in (1, 4, 8):
        packed_query_rows = (num_q_heads // num_kv_heads) * seq_q
        for owner_waves in (1, 2, 4, 8):
            if packed_query_rows <= owner_waves * 32:
                return owner_waves
        return 8

    base_ctas = batch_size * num_q_heads * (seq_q // 128)
    if qk_head_dim == 128 and base_ctas >= _FOUR_WAVE_PREFILL_MIN_CTAS:
        return 4
    if qk_head_dim == 128 and (seq_kv <= 1024 or base_ctas < 256):
        return 2
    return 4


def _select_waves_per_eu(
    *,
    owner_waves: int,
    enough_prefill_parallelism: bool,
    seq_kv: int,
    qk_head_dim: int,
) -> int:
    """Select the occupancy hint independently from the owner geometry."""
    if (
        qk_head_dim == 128
        and owner_waves in (4, 8)
        and (seq_kv >= 2048 or enough_prefill_parallelism)
    ):
        return 2
    return 1


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
    """Build the gfx950 prefill or packed-GQA decode kernel.

    Each wave is an independent owner of 32 query rows and its corresponding
    32x128 output tile. The owner count is selected at compile time; K/V are
    staged once per CTA and shared by all owners.
    """

    if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError("FlyDSL forward requires Hq % Hkv == 0")

    decode = seq_q in (1, 4, 8)
    OWNER_WAVES = _select_owner_waves(
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        seq_q=seq_q,
        seq_kv=seq_kv,
        qk_head_dim=qk_head_dim,
    )
    BM = OWNER_WAVES * 32
    BN = 64
    SPLIT_KV = (
        (num_q_heads // num_kv_heads) * seq_q == 1
        and batch_size * num_kv_heads < 256
        and seq_kv >= 2048
    )
    NW = 2 if SPLIT_KV else OWNER_WAVES
    NT = NW * 64
    WAVES_PER_EU = _select_waves_per_eu(
        owner_waves=OWNER_WAVES,
        enough_prefill_parallelism=(
            not decode
            and batch_size * num_q_heads * (seq_q // 128) >= _FOUR_WAVE_PREFILL_MIN_CTAS
        ),
        seq_kv=seq_kv,
        qk_head_dim=qk_head_dim,
    )
    VPT = 8
    MFMA_MN = 32

    if (qk_head_dim, v_head_dim) not in ((128, 128), (192, 128)):
        raise ValueError(
            "FlyDSL forward requires (qk_head_dim, v_head_dim) "
            "to be (128, 128) or (192, 128)"
        )
    if sparse_q_block_size != 128 or sparse_kv_block_size != 128:
        raise ValueError("FlyDSL forward requires sparse block size 128")
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
    DECODE = bool(decode)
    PIPELINED_KV = (
        DQK == 128
        and DECODE
        and OWNER_WAVES in (2, 4)
        and B * HKV <= 256
        and SK >= 2048
    )
    PACKED_Q_ROWS = GROUP_SIZE * SQ if DECODE else SQ
    Q_CHUNKS = (PACKED_Q_ROWS + BM - 1) // BM if DECODE else SQ // BM

    if DECODE:
        if BMH not in (1, HKV):
            raise ValueError(
                "FlyDSL forward decode requires a shared or per-KV-head BlockMask"
            )
        if NQB != 1:
            raise ValueError("FlyDSL forward decode requires one sparse Q block")
        if PACKED_Q_ROWS <= 0 or PACKED_Q_ROWS > 256:
            raise ValueError("FlyDSL forward decode requires 1 <= (Hq/Hkv)*Sq <= 256")
    else:
        if SQ % BM:
            raise ValueError(
                "FlyDSL forward prefill requires Sq divisible by its owner tile"
            )
        if NQB != SQ // sparse_q_block_size:
            raise ValueError("BlockMask Q rows must cover Sq with 128-row blocks")

    CPB = sparse_kv_block_size // BN
    K_STEPS = DQK // 16
    D_CHUNKS = DV // MFMA_MN
    K_DCH = DQK // VPT
    V_DCH = DV // VPT
    Q_LOAD_IT = (BM * K_DCH) // NT
    KV_LOAD_THREADS = 64 if SPLIT_KV else NT
    K_LOAD_IT = (BN * K_DCH) // KV_LOAD_THREADS
    V_LOAD_IT = (BN * V_DCH) // KV_LOAD_THREADS
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
    WINDOW_SIZE = _causal_window_size(MASK_PROGRAM, MASK_PROGRAM_OUTPUT)
    CAUSAL_DOCUMENT_MASK = is_causal_document_mask_program(
        MASK_PROGRAM,
        MASK_PROGRAM_OUTPUT,
        MASK_BUFFER_STRIDES,
    )

    if PIPELINED_KV:

        @fx.struct
        class FwdSmem:
            # The decode pipeline keeps Q in registers and double-buffers K/V
            # so the next tile's DMA can overlap the current tile's math.
            k0: fx.Array[fx.BFloat16, BN * DQK, 16]
            k1: fx.Array[fx.BFloat16, BN * DQK, 16]
            v0: fx.Array[fx.BFloat16, BN * DV, 16]
            v1: fx.Array[fx.BFloat16, BN * DV, 16]

    elif SPLIT_KV:

        @fx.struct
        class FwdSmem:
            query: fx.Array[fx.BFloat16, BM * DQK, 16]
            # One reusable K/V tile per worker wave keeps the CTA below 64 KiB.
            kv: fx.Array[fx.BFloat16, NW * BN * DQK, 16]
            reduction: fx.Array[fx.Float32, 2 * (2 + D_CHUNKS * 16), 16]

    else:

        @fx.struct
        class FwdSmem:
            query: fx.Array[fx.BFloat16, BM * DQK, 16]
            # K needs the largest allocation. V reuses the same storage after
            # every wave has consumed K into registers.
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
        q_chunk = fx.block_idx.y
        q_base = q_chunk * fx.Int32(BM)
        if const_expr(DECODE):
            kv_head = fx.block_idx.x
            head = kv_head * fx.Int32(GROUP_SIZE)
        else:
            head = fx.block_idx.x
            kv_head = head // fx.Int32(GROUP_SIZE)

        def row_coordinates(local_row):
            packed_row = q_base + local_row
            if const_expr(DECODE):
                valid = packed_row < fx.Int32(PACKED_Q_ROWS)
                safe_row = valid.select(packed_row, fx.Int32(0))
                row_head = kv_head * fx.Int32(GROUP_SIZE) + safe_row // fx.Int32(SQ)
                query_position = safe_row % fx.Int32(SQ)
            else:
                valid = fx.Int32(0) == fx.Int32(0)
                row_head = head
                query_position = packed_row
            return valid, row_head, query_position

        query_row = (
            lane_row
            if const_expr(SPLIT_KV)
            else wave * fx.Int32(MFMA_MN) + lane_row
        )
        query_valid, query_head, query_pos = row_coordinates(query_row)

        lds = fx.SharedAllocator().allocate(FwdSmem).peek()
        if const_expr(PIPELINED_KV):
            pk_stages = [lds.k0.ptr, lds.k1.ptr]
            pv_stages = [lds.v0.ptr, lds.v1.ptr]
        else:
            pquery = lds.query.ptr
            pkv = lds.kv.ptr
            if const_expr(SPLIT_KV):
                pkv = fx.add_offset(
                    pkv,
                    fx.make_int_tuple(wave * fx.Int32(BN * DQK)),
                )
            pk_stages = [pkv]
            pv_stages = [pkv]
            if const_expr(SPLIT_KV):
                preduction = lds.reduction.ptr

        batch_i64 = fx.Int64(batch)
        kv_head_i64 = fx.Int64(kv_head)
        kv_offset = batch_i64 * fx.Int64(K_STRIDE[0]) + kv_head_i64 * fx.Int64(
            K_STRIDE[1]
        )
        value_offset = batch_i64 * fx.Int64(V_STRIDE[0]) + kv_head_i64 * fx.Int64(
            V_STRIDE[1]
        )
        gK = make_global_view(
            K,
            kv_offset,
            (SK, DQK),
            (K_STRIDE[2], K_STRIDE[3]),
        )
        gV = make_global_view(
            V,
            value_offset,
            (SK, DV),
            (V_STRIDE[2], V_STRIDE[3]),
        )
        if const_expr(DECODE):
            q_head_base = kv_head_i64 * fx.Int64(GROUP_SIZE)
            q_offset = batch_i64 * fx.Int64(Q_STRIDE[0]) + q_head_base * fx.Int64(
                Q_STRIDE[1]
            )
            output_offset = batch_i64 * fx.Int64(O_STRIDE[0]) + q_head_base * fx.Int64(
                O_STRIDE[1]
            )
            gQ = make_global_view(
                Q,
                q_offset,
                (GROUP_SIZE, SQ, DQK),
                (Q_STRIDE[1], Q_STRIDE[2], Q_STRIDE[3]),
            )
            gO = make_global_view(
                O,
                output_offset,
                (GROUP_SIZE, SQ, DV),
                (O_STRIDE[1], O_STRIDE[2], O_STRIDE[3]),
            )
        else:
            head_i64 = fx.Int64(head)
            q_offset = batch_i64 * fx.Int64(Q_STRIDE[0]) + head_i64 * fx.Int64(
                Q_STRIDE[1]
            )
            output_offset = batch_i64 * fx.Int64(O_STRIDE[0]) + head_i64 * fx.Int64(
                O_STRIDE[1]
            )
            gQ = make_global_view(
                Q,
                q_offset,
                (SQ, DQK),
                (Q_STRIDE[2], Q_STRIDE[3]),
            )
            gO = make_global_view(
                O,
                output_offset,
                (SQ, DV),
                (O_STRIDE[2], O_STRIDE[3]),
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
        mask_buffers = make_mask_buffers(
            make_global_view,
            MASK_BUFFER_COUNT,
            MASK_BUFFER_SIZES,
            MaskBuffer0,
            MaskBuffer1,
            MaskBuffer2,
            MaskBuffer3,
        )

        o64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)

        def load_i32(view, index):
            return fx.Int32(fx.get_iter(view)[index])

        def load_uniform_i32(view, index):
            return fx.gpu.shuffle_idx(load_i32(view, index), 0, 64)

        evaluate_mask = make_mask_evaluator(
            MASK_PROGRAM,
            MASK_PROGRAM_OUTPUT,
            MASK_BUFFER_STRIDES,
            mask_buffers,
            load_i32,
            batch,
            query_head,
        )

        def store_f32(view, index, value):
            fx.get_iter(view)[index] = value

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
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_MN, MFMA_MN, 16, fx.BFloat16))

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
            # Conflict-free 32x32 K subtile layout.
            swizzled_column = (
                column
                ^ ((row & fx.Int32(8)) << fx.Int32(1))
                ^ ((row & fx.Int32(16)) >> fx.Int32(1))
            )
            return row * fx.Int32(MFMA_MN) + swizzled_column

        # Pipelined Dqk=128 decode keeps eight Q packs per lane in registers.
        # Prefill and Dqk=192 use a swizzled LDS tile, avoiding the gfx950 VGPR
        # cliff for Dqk=192.
        q_scale = Vec.from_elements([_f32(SCALE_LOG2)], fx.Float32).broadcast_to(VPT)
        q_register_packs = []
        if const_expr(PIPELINED_KV):
            local_head = query_head - kv_head * fx.Int32(GROUP_SIZE)
            query_source = fx.slice(
                gQ,
                (local_head, query_pos, None),
            )
            query_row = fx.logical_divide(
                query_source,
                fx.make_layout(VPT, 1),
            )
            raw_q_packs = []
            for k_step in fx.range_constexpr(K_STEPS):
                column = fx.Int32(k_step * 16) + lane_half * fx.Int32(VPT)
                q_fragment = fx.make_rmem_tensor(VPT, fx.BFloat16)
                fx.copy(
                    g128,
                    fx.slice(
                        query_row,
                        (None, column // fx.Int32(VPT)),
                    ),
                    q_fragment,
                )
                raw_q_packs.append(Vec(q_fragment.load()))
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            for k_step in fx.range_constexpr(K_STEPS):
                q_register_packs.append(
                    (Vec(raw_q_packs[k_step].to(fx.Float32)) * q_scale)
                    .to(fx.BFloat16)
                    .ir_value()
                )
        else:
            for load_step in fx.range_constexpr(Q_LOAD_IT):
                linear = fx.Int32(load_step * NT) + tid
                row = linear // fx.Int32(K_DCH)
                chunk = linear % fx.Int32(K_DCH)
                column = chunk * fx.Int32(VPT)
                row_valid, row_head, row_query_pos = row_coordinates(row)
                q_fragment = fx.make_rmem_tensor(VPT, fx.BFloat16)
                if const_expr(DECODE):
                    local_head = row_head - kv_head * fx.Int32(GROUP_SIZE)
                    source_row = fx.slice(
                        gQ,
                        (local_head, row_query_pos, None),
                    )
                else:
                    source_row = fx.slice(gQ, (row_query_pos, None))
                source = fx.logical_divide(
                    source_row,
                    fx.make_layout(VPT, 1),
                )
                fx.copy(
                    g128,
                    fx.slice(source, (None, chunk)),
                    q_fragment,
                )
                q_value = Vec(q_fragment.load())
                if const_expr(DECODE):
                    q_value = Vec.from_elements(
                        [
                            row_valid.select(
                                q_value[element],
                                fx.BFloat16(0.0),
                            )
                            for element in fx.range_constexpr(VPT)
                        ],
                        fx.BFloat16,
                    )
                q_scaled = Vec(q_value.to(fx.Float32)) * q_scale
                q_row_group = row // fx.Int32(MFMA_MN)
                q_row_in_group = row % fx.Int32(MFMA_MN)
                q_d_subtile = column // fx.Int32(MFMA_MN)
                q_column_in_subtile = column % fx.Int32(MFMA_MN)
                q_lds_offset = (
                    q_row_group * fx.Int32(DQK_SUBTILES) + q_d_subtile
                ) * fx.Int32(MFMA_MN * MFMA_MN) + k_swizzled_offset(
                    q_row_in_group,
                    q_column_in_subtile,
                )
                fx.ptr_store(
                    Vec(q_scaled).to(fx.BFloat16),
                    pquery + q_lds_offset,
                )
            fx.gpu.barrier()

        def load_q_pack(k_step):
            if const_expr(PIPELINED_KV):
                return q_register_packs[k_step]
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
        if const_expr(DECODE):
            mask_q_block = fx.Int32(0)
        else:
            mask_q_block = q_base // fx.Int32(sparse_q_block_size)
        mask_row = (mask_batch * fx.Int32(BMH) + mask_head) * fx.Int32(
            NQB
        ) + mask_q_block
        if const_expr(CAUSAL_DOCUMENT_MASK):
            document_id = load_i32(
                mask_buffers[0],
                query_pos * fx.Int32(MASK_BUFFER_STRIDES[0][0]),
            )
            document_start = load_i32(
                mask_buffers[1],
                document_id * fx.Int32(MASK_BUFFER_STRIDES[1][0]),
            )

        def stage_k(kv_base, stage=0):
            destination_base = pk_stages[stage]
            for load_step in fx.range_constexpr(K_LOAD_IT):
                load_tid = lane if const_expr(SPLIT_KV) else tid
                linear = fx.Int32(load_step * KV_LOAD_THREADS) + load_tid
                lds_offset = linear * fx.Int32(VPT)
                subtile = lds_offset // fx.Int32(MFMA_MN * MFMA_MN)
                within_subtile = lds_offset % fx.Int32(MFMA_MN * MFMA_MN)
                row_in_half = within_subtile // fx.Int32(MFMA_MN)
                swizzled_column = within_subtile % fx.Int32(MFMA_MN)
                row_half = subtile // fx.Int32(DQK_SUBTILES)
                d_subtile = subtile % fx.Int32(DQK_SUBTILES)
                row = row_half * fx.Int32(MFMA_MN) + row_in_half
                column = d_subtile * fx.Int32(MFMA_MN) + (
                    swizzled_column
                    ^ ((row_in_half & fx.Int32(8)) << fx.Int32(1))
                    ^ ((row_in_half & fx.Int32(16)) >> fx.Int32(1))
                )
                chunk = column // fx.Int32(VPT)
                source = fx.logical_divide(
                    fx.slice(gK, (kv_base + row, None)),
                    fx.make_layout(VPT, 1),
                )
                destination_pointer = fx.inttoptr(
                    lds_dma_ptr_type,
                    fx.Int32(
                        fx.ptrtoint(
                            fx.add_offset(
                                destination_base,
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

        def load_k_pack(k_step, high_half, stage=0):
            d_subtile = k_step // 2
            d_half = k_step % 2
            pointer = k_odd_pointers[stage] if d_half else k_even_pointers[stage]
            row_half_offset = DQK_SUBTILES * MFMA_MN * MFMA_MN if high_half else 0
            return read_v8bf16_static(
                pointer,
                row_half_offset + d_subtile * MFMA_MN * MFMA_MN,
            )

        def stage_v(kv_base, stage=0):
            destination_base = pv_stages[stage]
            for load_step in fx.range_constexpr(V_LOAD_IT):
                load_tid = lane if const_expr(SPLIT_KV) else tid
                linear = fx.Int32(load_step * KV_LOAD_THREADS) + load_tid
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
                    fx.slice(gV, (kv_base + row, None)),
                    fx.make_layout(VPT, 1),
                )
                destination_pointer = fx.inttoptr(
                    lds_dma_ptr_type,
                    fx.Int32(
                        fx.ptrtoint(
                            fx.add_offset(
                                destination_base,
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

        def load_v_pack(probability_pack, d_chunk, stage=0):
            halves = []
            for half in fx.range_constexpr(2):
                subtile = (probability_pack * 2 + half) * D_CHUNKS + d_chunk
                pointer = fx.add_offset(
                    v_lane_pointers[stage],
                    fx.make_int_tuple(fx.Int32(subtile * 8 * MFMA_MN)),
                )
                source = fx.make_view(pointer, fx.make_layout(4, 1))
                destination = fx.make_rmem_tensor(4, fx.BFloat16)
                fx.copy(tr16, source, destination)
                halves.append(Vec(destination.load()))
            return halves[0].shuffle(halves[1], list(range(8))).ir_value()

        q_wave = fx.Int32(0) if const_expr(SPLIT_KV) else wave
        q_wave_offset = q_wave * fx.Int32(DQK_SUBTILES * MFMA_MN * MFMA_MN)
        even_column = lane_half * fx.Int32(VPT)
        odd_column = fx.Int32(16) + even_column
        if const_expr(not PIPELINED_KV):
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
        k_even_pointers = [
            fx.add_offset(
                pointer,
                fx.make_int_tuple(k_swizzled_offset(lane_row, even_column)),
            )
            for pointer in pk_stages
        ]
        k_odd_pointers = [
            fx.add_offset(
                pointer,
                fx.make_int_tuple(k_swizzled_offset(lane_row, odd_column)),
            )
            for pointer in pk_stages
        ]
        v_row_offset = (lane % fx.Int32(16)) // fx.Int32(4) + lane_half * fx.Int32(4)
        v_column_offset = (lane % fx.Int32(4)) * fx.Int32(4) + (
            (lane % fx.Int32(MFMA_MN)) // fx.Int32(16)
        ) * fx.Int32(16)
        v_lane_pointers = [
            fx.add_offset(
                pointer,
                fx.make_int_tuple(v_row_offset * fx.Int32(MFMA_MN) + v_column_offset),
            )
            for pointer in pv_stages
        ]

        def process_tile(
            kv_chunk,
            masked,
            tile_output,
            tile_running_max,
            tile_running_sum,
            stage=0,
            tile_active=None,
        ):
            kv_base = kv_chunk * fx.Int32(BN)
            if const_expr(not PIPELINED_KV):
                stage_k(kv_base)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.gpu.barrier()

            scores_lo = zero16
            scores_hi = zero16
            for k_step in fx.range_constexpr(K_STEPS):
                query_pack = load_q_pack(k_step)
                key_lo = load_k_pack(k_step, False, stage)
                key_hi = load_k_pack(k_step, True, stage)
                scores_lo = mfma(key_lo, query_pack, scores_lo)
                scores_hi = mfma(key_hi, query_pack, scores_hi)
            schedule_fwd_qk_pipeline(
                reduction_steps=K_STEPS,
                vmem_count=(K_LOAD_IT if PIPELINED_KV else 0),
            )

            if const_expr(not PIPELINED_KV):
                # Every wave must finish its K reads before the shared allocation
                # is reused for V.
                fx.gpu.barrier()
                stage_v(kv_base)

            raw_lo = Vec(scores_lo)
            raw_hi = Vec(scores_hi)
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
                    keep = query_valid
                    if tile_active is not None:
                        keep = keep & tile_active
                    if const_expr(CAUSAL_DOCUMENT_MASK):
                        if isinstance(masked, bool):
                            if masked:
                                keep = keep & (query_pos >= key_pos) & (
                                    key_pos >= document_start
                                )
                        else:
                            mask_keep = (query_pos >= key_pos) & (
                                key_pos >= document_start
                            )
                            keep = keep & ((~masked) | mask_keep)
                    elif const_expr(WINDOW_SIZE is not None):
                        if masked:
                            keep = keep & (query_pos >= key_pos) & (
                                query_pos - key_pos < fx.Int32(WINDOW_SIZE)
                            )
                    elif const_expr(CAUSAL_PARTIAL):
                        if masked:
                            keep = keep & (
                                key_pos <= (query_pos + fx.Int32(SK - SQ))
                            )
                    elif const_expr(bool(MASK_PROGRAM)):
                        if masked:
                            keep = keep & evaluate_mask(query_pos, key_pos)
                    keep_values.append(keep)
                    score_values.append(keep.select(_f32(raw[element]), _f32(_NEG_BIG)))

            local_max = score_values[0]
            for element in fx.range_constexpr(1, 32):
                local_max = _maximum(local_max, score_values[element])
            peer_max = _f32(fx.gpu.shuffle_xor(local_max, 32, 64))
            tile_max = _maximum(local_max, peer_max)
            new_max = _maximum(tile_running_max, tile_max)
            correction = _exp2(tile_running_max - new_max)

            correction_vec = Vec.from_elements([correction], fx.Float32).broadcast_to(
                16
            )
            for d_chunk in fx.range_constexpr(D_CHUNKS):
                tile_output[d_chunk] = Vec(tile_output[d_chunk]) * correction_vec

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
            schedule_fwd_softmax_pipeline(vmem_count=V_LOAD_IT)
            peer_sum = _f32(fx.gpu.shuffle_xor(local_sum, 32, 64))
            tile_sum = local_sum + peer_sum
            tile_running_sum = tile_running_sum * correction + tile_sum
            tile_running_max = new_max

            if const_expr(not PIPELINED_KV):
                # V writes were issued before the register-only softmax.
                # Synchronize only when the LDS data is actually consumed.
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.gpu.barrier()
            for probability_pack in fx.range_constexpr(4):
                for d_chunk in fx.range_constexpr(D_CHUNKS):
                    value_pack = load_v_pack(
                        probability_pack,
                        d_chunk,
                        stage,
                    )
                    tile_output[d_chunk] = mfma(
                        value_pack,
                        probability_packs[probability_pack],
                        tile_output[d_chunk],
                    )
            schedule_fwd_pv_pipeline(output_chunks=D_CHUNKS)

            if const_expr(not PIPELINED_KV):
                # Protect V from the next tile's K staging.
                fx.gpu.barrier()
            return tile_output, tile_running_max, tile_running_sum

        def process_pipelined_run(
            block_count,
            block_indices,
            block_base,
            masked,
            run_state,
        ):
            run_results = run_state
            if block_count > fx.Int32(0):
                first_block = load_uniform_i32(block_indices, block_base)
                first_chunk = first_block * fx.Int32(CPB)
                stage_k(first_chunk * fx.Int32(BN), 0)
                stage_v(first_chunk * fx.Int32(BN), 0)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.gpu.barrier()

                pipeline_state = [first_block, *run_state]
                pipeline_results = pipeline_state
                for block_index, iter_args in range(
                    fx.Int32(0),
                    block_count,
                    fx.Int32(1),
                    init=pipeline_state,
                ):
                    current_block = fx.Int32(iter_args[0])
                    iter_max = _f32(iter_args[1])
                    iter_sum = _f32(iter_args[2])
                    iter_output = [
                        iter_args[3 + d_chunk]
                        for d_chunk in fx.range_constexpr(D_CHUNKS)
                    ]

                    first_chunk = current_block * fx.Int32(CPB)
                    second_chunk = first_chunk + fx.Int32(1)
                    stage_k(second_chunk * fx.Int32(BN), 1)
                    stage_v(second_chunk * fx.Int32(BN), 1)
                    iter_output, iter_max, iter_sum = process_tile(
                        first_chunk,
                        masked,
                        iter_output,
                        iter_max,
                        iter_sum,
                        stage=0,
                    )
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    fx.gpu.barrier()

                    next_index = fx.Int32(block_index) + fx.Int32(1)
                    next_block = current_block
                    if next_index < block_count:
                        next_block = load_uniform_i32(
                            block_indices,
                            block_base + next_index,
                        )
                        next_chunk = next_block * fx.Int32(CPB)
                        stage_k(next_chunk * fx.Int32(BN), 0)
                        stage_v(next_chunk * fx.Int32(BN), 0)
                    iter_output, iter_max, iter_sum = process_tile(
                        second_chunk,
                        masked,
                        iter_output,
                        iter_max,
                        iter_sum,
                        stage=1,
                    )
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    fx.gpu.barrier()
                    pipeline_results = yield [
                        next_block,
                        iter_max,
                        iter_sum,
                        *iter_output,
                    ]
                run_results = pipeline_results[1:]
            return run_results

        def process_split_run(
            block_count,
            block_indices,
            block_base,
            masked,
            run_state,
        ):
            run_results = run_state
            split_count = (block_count + fx.Int32(NW - 1)) // fx.Int32(NW)
            for split_index, iter_args in range(
                fx.Int32(0),
                split_count,
                fx.Int32(1),
                init=run_state,
            ):
                iter_max = _f32(iter_args[0])
                iter_sum = _f32(iter_args[1])
                iter_output = [
                    iter_args[2 + d_chunk]
                    for d_chunk in fx.range_constexpr(D_CHUNKS)
                ]
                block_index = fx.Int32(split_index * NW) + wave
                tile_active = block_index < block_count
                safe_index = tile_active.select(block_index, fx.Int32(0))
                sparse_block = load_uniform_i32(
                    block_indices,
                    block_base + safe_index,
                )
                for sub_block in fx.range_constexpr(CPB):
                    iter_output, iter_max, iter_sum = process_tile(
                        sparse_block * fx.Int32(CPB) + fx.Int32(sub_block),
                        masked,
                        iter_output,
                        iter_max,
                        iter_sum,
                        tile_active=tile_active,
                    )
                run_results = yield [iter_max, iter_sum] + iter_output
            return run_results

        def reduce_split_results(split_results):
            split_max = _f32(split_results[0])
            split_sum = _f32(split_results[1])
            split_output = [
                split_results[2 + d_chunk]
                for d_chunk in fx.range_constexpr(D_CHUNKS)
            ]
            reduction_stride = 2 + D_CHUNKS * 16
            reduction_offset = lane_half * fx.Int32(reduction_stride)
            reduction_pointer = fx.add_offset(
                preduction,
                fx.make_int_tuple(reduction_offset),
            )
            if (wave == fx.Int32(1)) & (lane_row == fx.Int32(0)):
                fx.ptr_store(split_max, reduction_pointer)
                fx.ptr_store(
                    split_sum,
                    fx.add_offset(reduction_pointer, fx.make_int_tuple(fx.Int32(1))),
                )
                for d_chunk in fx.range_constexpr(D_CHUNKS):
                    output_pointer = fx.add_offset(
                        reduction_pointer,
                        fx.make_int_tuple(fx.Int32(2 + d_chunk * 16)),
                    )
                    fx.ptr_store(Vec(split_output[d_chunk]), output_pointer)
            fx.gpu.barrier()

            other_max = _f32(fx.ptr_load(reduction_pointer))
            other_sum = _f32(
                fx.ptr_load(
                    fx.add_offset(
                        reduction_pointer,
                        fx.make_int_tuple(fx.Int32(1)),
                    )
                )
            )
            combined_max = _maximum(split_max, other_max)
            split_scale = _exp2(split_max - combined_max)
            other_scale = _exp2(other_max - combined_max)
            split_sum = split_sum * split_scale + other_sum * other_scale
            split_max = combined_max
            split_scale_vec = Vec.from_elements(
                [split_scale], fx.Float32
            ).broadcast_to(16)
            other_scale_vec = Vec.from_elements(
                [other_scale], fx.Float32
            ).broadcast_to(16)
            for d_chunk in fx.range_constexpr(D_CHUNKS):
                output_pointer = fx.add_offset(
                    reduction_pointer,
                    fx.make_int_tuple(fx.Int32(2 + d_chunk * 16)),
                )
                other_output = Vec(
                    fx.make_view(output_pointer, fx.make_layout(16, 1)).load()
                )
                split_output[d_chunk] = (
                    Vec(split_output[d_chunk]) * split_scale_vec
                    + other_output * other_scale_vec
                )
            return [split_max, split_sum] + split_output

        def store_results(final_results):
            final_max = _f32(final_results[0])
            final_sum = _f32(final_results[1])
            final_output = [
                final_results[2 + d_chunk] for d_chunk in fx.range_constexpr(D_CHUNKS)
            ]

            inverse_sum = (final_sum > _f32(0.0)).select(
                _f32(1.0) / final_sum,
                _f32(0.0),
            )
            inverse_vec = Vec.from_elements([inverse_sum], fx.Float32).broadcast_to(16)

            if const_expr(DECODE):
                local_head = query_head - kv_head * fx.Int32(GROUP_SIZE)
                output_source = fx.slice(
                    gO,
                    (local_head, query_pos, None),
                )
            else:
                output_source = fx.slice(gO, (query_pos, None))
            output_row = fx.logical_divide(
                output_source,
                fx.make_layout(4, 1),
            )
            store_valid = query_valid
            if const_expr(SPLIT_KV):
                store_valid = store_valid & (wave == fx.Int32(0))
            if store_valid:
                for d_chunk in fx.range_constexpr(D_CHUNKS):
                    normalized = Vec(final_output[d_chunk]) * inverse_vec
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
                            fx.slice(
                                output_row,
                                (None, column // fx.Int32(4)),
                            ),
                        )

            if store_valid & (lane_half == fx.Int32(0)):
                has_values = final_sum > _f32(0.0)
                lse_value = final_max + fx.math.log2(final_sum)
                max_value = final_max
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
                stats_offset = (batch * fx.Int32(HQ) + query_head) * fx.Int32(
                    SQ
                ) + query_pos
                store_f32(gLSE, stats_offset, lse_value)
                store_f32(gMax, stats_offset, max_value)

        full_count = load_uniform_i32(gFKVN, mask_row)
        partial_count = load_uniform_i32(gKVN, mask_row)
        full_base = mask_row * fx.Int32(MAX_FULL)
        partial_base = mask_row * fx.Int32(MAX_PARTIAL)
        initial_state = [running_max, running_sum] + output

        if const_expr(SPLIT_KV):
            full_results = process_split_run(
                full_count,
                gFKVI,
                full_base,
                False,
                initial_state,
            )
            split_results = process_split_run(
                partial_count,
                gKVI,
                partial_base,
                True,
                full_results,
            )
            final_results = reduce_split_results(split_results)
        elif const_expr(PIPELINED_KV):
            full_results = process_pipelined_run(
                full_count,
                gFKVI,
                full_base,
                False,
                initial_state,
            )
            final_results = process_pipelined_run(
                partial_count,
                gKVI,
                partial_base,
                True,
                full_results,
            )
        elif const_expr(CAUSAL_DOCUMENT_MASK):
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
                    iter_args[2 + d_chunk] for d_chunk in fx.range_constexpr(D_CHUNKS)
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
                    iter_args[2 + d_chunk] for d_chunk in fx.range_constexpr(D_CHUNKS)
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
                full_results[2 + d_chunk] for d_chunk in fx.range_constexpr(D_CHUNKS)
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
                    iter_args[2 + d_chunk] for d_chunk in fx.range_constexpr(D_CHUNKS)
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
        store_results(final_results)

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
                "rocdl.waves_per_eu": WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{NT},{NT}",
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
            grid=(HKV if DECODE else HQ, Q_CHUNKS, B),
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
