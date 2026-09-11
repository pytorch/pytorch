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
    make_qk_shared_layout,
    make_value_shared_layout,
    schedule_fwd_pv_pipeline,
    schedule_fwd_qk_pipeline,
    schedule_fwd_softmax_pipeline,
)

_LOG2E = 1.4426950408889634
_LN2 = math.log(2.0)
# Keep masked exponentials finite under fast math; empty rows store -inf explicitly.
_NEG_BIG = -1.0e30
# Four-wave CTAs reduce launch count once the 128-row prefill grid remains
# large enough to keep gfx950 occupied.
_FOUR_WAVE_PREFILL_MIN_CTAS = 512


def _f32(value):
    return fx.Float32(value)


def _exp2(value):
    return fast_exp2(_f32(value))


def _maximum(lhs, rhs):
    return (lhs > rhs).select(lhs, rhs)


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

    # Keep standalone entry-point validation even though Inductor checks these
    # constraints before registering the vendored kernel.
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError("FlyDSL forward requires Hq % Hkv == 0")

    decode = seq_q in (1, 4, 8)
    owner_waves = _select_owner_waves(
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        seq_q=seq_q,
        seq_kv=seq_kv,
        qk_head_dim=qk_head_dim,
    )
    query_tile_rows = owner_waves * 32
    kv_tile_rows = 64
    split_kv = (
        (num_q_heads // num_kv_heads) * seq_q == 1
        and batch_size * num_kv_heads < 256
        and seq_kv >= 2048
    )
    num_waves = 2 if split_kv else owner_waves
    num_threads = num_waves * 64
    waves_per_eu = _select_waves_per_eu(
        owner_waves=owner_waves,
        enough_prefill_parallelism=(
            not decode
            and batch_size * num_q_heads * (seq_q // 128) >= _FOUR_WAVE_PREFILL_MIN_CTAS
        ),
        seq_kv=seq_kv,
        qk_head_dim=qk_head_dim,
    )
    values_per_thread = 8
    mfma_tile_size = 32

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

    batch_size = int(batch_size)
    num_q_heads = int(num_q_heads)
    num_kv_heads = int(num_kv_heads)
    seq_q = int(seq_q)
    seq_kv = int(seq_kv)
    qk_head_dim = int(qk_head_dim)
    v_head_dim = int(v_head_dim)
    block_mask_batch = int(block_mask_batch)
    block_mask_heads = int(block_mask_heads)
    num_q_blocks = int(num_q_blocks)
    max_partial_blocks = int(max_partial_blocks)
    max_full_blocks = int(max_full_blocks)
    query_heads_per_kv_head = num_q_heads // num_kv_heads
    decode = bool(decode)
    pipelined_kv = (
        qk_head_dim == 128
        and decode
        and owner_waves in (2, 4)
        and batch_size * num_kv_heads <= 256
        and seq_kv >= 2048
    )
    packed_query_rows = query_heads_per_kv_head * seq_q if decode else seq_q
    num_query_chunks = (
        (packed_query_rows + query_tile_rows - 1) // query_tile_rows
        if decode
        else seq_q // query_tile_rows
    )

    if decode:
        if block_mask_heads not in (1, num_kv_heads):
            raise ValueError(
                "FlyDSL forward decode requires a shared or per-KV-head BlockMask"
            )
        if num_q_blocks != 1:
            raise ValueError("FlyDSL forward decode requires one sparse Q block")
        if packed_query_rows <= 0 or packed_query_rows > 256:
            raise ValueError("FlyDSL forward decode requires 1 <= (Hq/Hkv)*Sq <= 256")
    else:
        if seq_q % query_tile_rows:
            raise ValueError(
                "FlyDSL forward prefill requires Sq divisible by its owner tile"
            )
        if num_q_blocks != seq_q // sparse_q_block_size:
            raise ValueError("BlockMask Q rows must cover Sq with 128-row blocks")

    kv_tiles_per_sparse_block = sparse_kv_block_size // kv_tile_rows
    qk_reduction_steps = qk_head_dim // 16
    output_chunks = v_head_dim // mfma_tile_size
    qk_chunks_per_row = qk_head_dim // values_per_thread
    value_chunks_per_row = v_head_dim // values_per_thread
    query_load_iterations = (query_tile_rows * qk_chunks_per_row) // num_threads
    kv_load_threads = 64 if split_kv else num_threads
    key_load_iterations = (kv_tile_rows * qk_chunks_per_row) // kv_load_threads
    value_load_iterations = (kv_tile_rows * value_chunks_per_row) // kv_load_threads
    if (query_tile_rows * qk_chunks_per_row) % num_threads:
        raise ValueError("FlyDSL forward Q staging must evenly cover its tile")
    if (kv_tile_rows * qk_chunks_per_row) % kv_load_threads:
        raise ValueError("FlyDSL forward K staging must evenly cover its tile")
    if (kv_tile_rows * value_chunks_per_row) % kv_load_threads:
        raise ValueError("FlyDSL forward V staging must evenly cover its tile")

    def contiguous_stride(heads, sequence, dimension):
        return (heads * sequence * dimension, sequence * dimension, dimension, 1)

    q_stride = tuple(q_stride or contiguous_stride(num_q_heads, seq_q, qk_head_dim))
    k_stride = tuple(k_stride or contiguous_stride(num_kv_heads, seq_kv, qk_head_dim))
    v_stride = tuple(v_stride or contiguous_stride(num_kv_heads, seq_kv, v_head_dim))
    o_stride = tuple(o_stride or contiguous_stride(num_q_heads, seq_q, v_head_dim))
    scale_log2 = float(scale) * _LOG2E
    output_stats_in_log2 = bool(output_stats_in_log2)
    mask_program = tuple(mask_program)
    mask_program_output = int(mask_program_output)
    mask_buffer_shapes = tuple(tuple(shape) for shape in mask_buffer_shapes)
    mask_buffer_strides = tuple(tuple(stride) for stride in mask_buffer_strides)
    mask_buffer_count = len(mask_buffer_shapes)
    mask_buffer_sizes = tuple(
        1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
        for shape, strides in zip(mask_buffer_shapes, mask_buffer_strides)
    )
    causal_document_mask = is_causal_document_mask_program(
        mask_program,
        mask_program_output,
        mask_buffer_strides,
    )

    if pipelined_kv:

        @fx.struct
        class ForwardSharedMemory:
            # The decode pipeline keeps Q in registers and double-buffers K/V
            # so the next tile's DMA can overlap the current tile's math.
            k0: fx.Array[fx.BFloat16, kv_tile_rows * qk_head_dim, 16]
            k1: fx.Array[fx.BFloat16, kv_tile_rows * qk_head_dim, 16]
            v0: fx.Array[fx.BFloat16, kv_tile_rows * v_head_dim, 16]
            v1: fx.Array[fx.BFloat16, kv_tile_rows * v_head_dim, 16]

    elif split_kv:

        @fx.struct
        class ForwardSharedMemory:
            query: fx.Array[fx.BFloat16, query_tile_rows * qk_head_dim, 16]
            # One reusable K/V tile per worker wave keeps the CTA below 64 KiB.
            kv: fx.Array[
                fx.BFloat16,
                num_waves * kv_tile_rows * qk_head_dim,
                16,
            ]
            reduction_stats: fx.Array[fx.Float32, 2 * 2, 16]
            reduction_output: fx.Array[fx.Float32, 2 * output_chunks * 16, 16]

    else:

        @fx.struct
        class ForwardSharedMemory:
            query: fx.Array[fx.BFloat16, query_tile_rows * qk_head_dim, 16]
            # K needs the largest allocation. V reuses the same storage after
            # every wave has consumed K into registers.
            kv: fx.Array[fx.BFloat16, kv_tile_rows * qk_head_dim, 16]

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        logsumexp: fx.Tensor,
        max_scores: fx.Tensor,
        kv_num_blocks: fx.Tensor,
        kv_indices: fx.Tensor,
        full_kv_num_blocks: fx.Tensor,
        full_kv_indices: fx.Tensor,
        mask_buffer_0: fx.Tensor,
        mask_buffer_1: fx.Tensor,
        mask_buffer_2: fx.Tensor,
        mask_buffer_3: fx.Tensor,
        output: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        # This gfx950-only kernel uses wave64, including on older FlyDSL releases.
        warp_size = 64
        lane = tid % fx.Int32(warp_size)
        wave = tid // fx.Int32(warp_size)
        lane_half = lane // fx.Int32(mfma_tile_size)
        mma_atom = fx.make_mma_atom(
            fx.rocdl.MFMA(
                mfma_tile_size,
                mfma_tile_size,
                16,
                fx.BFloat16,
            )
        )
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.get_slice(lane)
        accumulator_coordinates = thr_mma.partition_C(
            fx.make_view(
                0,
                fx.make_layout((mfma_tile_size, mfma_tile_size), (1, 0)),
            )
        )
        query_coordinates = thr_mma.partition_C(
            fx.make_view(
                0,
                fx.make_layout((mfma_tile_size, mfma_tile_size), (0, 1)),
            )
        )
        query_k_coordinates = thr_mma.partition_B(
            fx.make_view(
                0,
                fx.make_layout((mfma_tile_size, qk_head_dim), (0, 1)),
            )
        )
        batch = fx.block_idx.z
        q_chunk = fx.block_idx.y
        q_base = q_chunk * fx.Int32(query_tile_rows)
        if const_expr(decode):
            kv_head = fx.block_idx.x
            head = kv_head * fx.Int32(query_heads_per_kv_head)
        else:
            head = fx.block_idx.x
            kv_head = head // fx.Int32(query_heads_per_kv_head)

        def row_coordinates(local_row):
            packed_row = q_base + local_row
            if const_expr(decode):
                valid = packed_row < fx.Int32(packed_query_rows)
                safe_row = valid.select(packed_row, fx.Int32(0))
                row_head = kv_head * fx.Int32(
                    query_heads_per_kv_head
                ) + safe_row // fx.Int32(seq_q)
                query_position = safe_row % fx.Int32(seq_q)
            else:
                valid = fx.Int32(0) == fx.Int32(0)
                row_head = head
                query_position = packed_row
            return valid, row_head, query_position

        query_row_in_wave = fx.Int32(fx.get_scalar(query_coordinates[0]))
        query_row = (
            query_row_in_wave
            if const_expr(split_kv)
            else wave * fx.Int32(mfma_tile_size) + query_row_in_wave
        )
        query_valid, query_head, query_pos = row_coordinates(query_row)

        lds = fx.SharedAllocator().allocate(ForwardSharedMemory).peek()
        if const_expr(pipelined_kv):
            shared_key_stages = [lds.k0.ptr, lds.k1.ptr]
            shared_value_stages = [lds.v0.ptr, lds.v1.ptr]
        else:
            shared_query_pointer = lds.query.ptr
            shared_kv_pointer = lds.kv.ptr
            if const_expr(split_kv):
                shared_kv_pointer = fx.get_iter(
                    fx.slice(
                        fx.make_view(
                            shared_kv_pointer,
                            fx.make_layout(
                                (num_waves, kv_tile_rows * qk_head_dim),
                                (kv_tile_rows * qk_head_dim, 1),
                            ),
                        ),
                        (wave, None),
                    )
                )
            shared_key_stages = [shared_kv_pointer]
            shared_value_stages = [shared_kv_pointer]

        batch_i64 = fx.Int64(batch)
        kv_head_i64 = fx.Int64(kv_head)
        key_view = make_global_view(
            key,
            (batch_i64, kv_head_i64, None, None),
            (batch_size, num_kv_heads, seq_kv, qk_head_dim),
            k_stride,
        )
        value_view = make_global_view(
            value,
            (batch_i64, kv_head_i64, None, None),
            (batch_size, num_kv_heads, seq_kv, v_head_dim),
            v_stride,
        )
        if const_expr(decode):
            query_view = make_global_view(
                query,
                (batch_i64, None, kv_head_i64, None, None),
                (
                    batch_size,
                    query_heads_per_kv_head,
                    num_kv_heads,
                    seq_q,
                    qk_head_dim,
                ),
                (
                    q_stride[0],
                    q_stride[1],
                    query_heads_per_kv_head * q_stride[1],
                    q_stride[2],
                    q_stride[3],
                ),
            )
            output_view = make_global_view(
                output,
                (batch_i64, None, kv_head_i64, None, None),
                (
                    batch_size,
                    query_heads_per_kv_head,
                    num_kv_heads,
                    seq_q,
                    v_head_dim,
                ),
                (
                    o_stride[0],
                    o_stride[1],
                    query_heads_per_kv_head * o_stride[1],
                    o_stride[2],
                    o_stride[3],
                ),
            )
        else:
            head_i64 = fx.Int64(head)
            query_view = make_global_view(
                query,
                (batch_i64, head_i64, None, None),
                (batch_size, num_q_heads, seq_q, qk_head_dim),
                q_stride,
            )
            output_view = make_global_view(
                output,
                (batch_i64, head_i64, None, None),
                (batch_size, num_q_heads, seq_q, v_head_dim),
                o_stride,
            )

        metadata_rows = block_mask_batch * block_mask_heads * num_q_blocks
        kv_num_blocks_view = make_global_view(
            kv_num_blocks,
            None,
            metadata_rows,
            1,
        )
        kv_indices_view = make_global_view(
            kv_indices,
            None,
            metadata_rows * max_partial_blocks,
            1,
        )
        full_kv_num_blocks_view = make_global_view(
            full_kv_num_blocks,
            None,
            metadata_rows,
            1,
        )
        full_kv_indices_view = make_global_view(
            full_kv_indices,
            None,
            metadata_rows * max_full_blocks,
            1,
        )
        logsumexp_view = make_global_view(
            logsumexp,
            None,
            batch_size * num_q_heads * seq_q,
            1,
        )
        max_scores_view = make_global_view(
            max_scores,
            None,
            batch_size * num_q_heads * seq_q,
            1,
        )
        mask_buffers = make_mask_buffers(
            make_global_view,
            mask_buffer_count,
            mask_buffer_sizes,
            mask_buffer_0,
            mask_buffer_1,
            mask_buffer_2,
            mask_buffer_3,
        )

        output_copy = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)

        def load_i32(view, index):
            return fx.Int32(view[index])

        def load_uniform_i32(view, index):
            return fx.gpu.shuffle_idx(load_i32(view, index), 0, warp_size)

        evaluate_mask = make_mask_evaluator(
            mask_program,
            mask_program_output,
            mask_buffer_strides,
            mask_buffers,
            load_i32,
            batch,
            query_head,
        )

        global_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        lds_copy = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.BFloat16)
        transposed_lds_copy = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans(16, 64),
            fx.BFloat16,
        )
        key_shared_layout = make_qk_shared_layout(kv_tile_rows, qk_head_dim)
        value_shared_layout = make_value_shared_layout(kv_tile_rows, v_head_dim)
        shared_keys = [
            fx.make_view(pointer, key_shared_layout) for pointer in shared_key_stages
        ]
        key_copy_coordinates = fx.make_composed_layout(
            fx.right_inverse(key_shared_layout.outer),
            fx.make_composed_layout(
                key_shared_layout.inner,
                fx.make_layout(kv_tile_rows * qk_head_dim, 1),
            ),
        )
        value_copy_coordinates = fx.right_inverse(value_shared_layout)
        key_copy_destinations = [
            fx.logical_divide(
                fx.make_view(
                    pointer,
                    fx.make_layout(kv_tile_rows * qk_head_dim, 1),
                ),
                fx.make_layout(values_per_thread, 1),
            )
            for pointer in shared_key_stages
        ]
        value_copy_destinations = [
            fx.logical_divide(
                fx.make_view(
                    pointer,
                    fx.make_layout(kv_tile_rows * v_head_dim, 1),
                ),
                fx.make_layout(values_per_thread, 1),
            )
            for pointer in shared_value_stages
        ]
        if const_expr(not pipelined_kv):
            shared_query = fx.make_view(
                shared_query_pointer,
                make_qk_shared_layout(query_tile_rows, qk_head_dim),
            )
        else:
            # Decode inputs are contiguous; pack the group's query rows.
            shared_query = fx.make_view(
                fx.get_iter(query_view),
                fx.make_layout(
                    (packed_query_rows, qk_head_dim),
                    (qk_head_dim, 1),
                ),
            )
        q_wave = fx.Int32(0) if const_expr(split_kv) else wave
        query_tiles = fx.flat_divide(shared_query, (mfma_tile_size, 16))
        key_tiles = [
            fx.flat_divide(shared_key, (mfma_tile_size, 16))
            for shared_key in shared_keys
        ]
        copy_q = fx.make_tiled_copy_B(global_copy, tiled_mma).get_slice(lane)
        copy_k = fx.make_tiled_copy_A(global_copy, tiled_mma).get_slice(lane)
        copy_v = fx.make_tiled_copy_A(transposed_lds_copy, tiled_mma).get_slice(lane)
        shared_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        probability_coordinates = thr_mma.partition_B(
            fx.make_view(
                0,
                fx.make_layout((mfma_tile_size, 16), (1, mfma_tile_size)),
            )
        )
        # Softmax keeps each lane's C values in register order. Permute V's
        # reduction mode to match that order without a cross-lane P shuffle.
        value_mma_layout = fx.composition(
            fx.select(value_shared_layout, [1, 0]),
            fx.make_tile(
                fx.make_layout(v_head_dim, 1),
                fx.make_layout(
                    (4, 2, 2, kv_tile_rows // 16),
                    (1, 8, 4, 16),
                ),
            ),
        )
        value_tiles = [
            fx.flat_divide(
                fx.make_view(pointer, value_mma_layout),
                (mfma_tile_size, 16),
            )
            for pointer in shared_value_stages
        ]

        def mfma(a_fragment, b_fragment, c_v16):
            c_fragment = fx.make_fragment_like(accumulator_coordinates, fx.Float32)
            c_fragment.store(Vec(c_v16))
            fx.gemm(
                tiled_mma,
                c_fragment,
                a_fragment,
                b_fragment,
                c_fragment,
            )
            return c_fragment.load()

        # Pipelined Dqk=128 decode keeps eight Q packs per lane in registers.
        # Prefill and Dqk=192 use a swizzled LDS tile, avoiding the gfx950 VGPR
        # cliff for Dqk=192.
        query_scale = Vec.from_elements(
            [_f32(scale_log2)],
            fx.Float32,
        ).broadcast_to(values_per_thread)
        query_register_packs = []
        if const_expr(pipelined_kv):
            local_head = query_head - kv_head * fx.Int32(query_heads_per_kv_head)
            query_source = fx.slice(
                query_view,
                (local_head, query_pos, None),
            )
            query_row_packs = fx.logical_divide(
                query_source,
                fx.make_layout(values_per_thread, 1),
            )
            raw_query_packs = []
            for k_step in fx.range_constexpr(qk_reduction_steps):
                column = fx.Int32(fx.get_scalar(query_k_coordinates[0, 0, k_step]))
                q_fragment = fx.make_rmem_tensor(values_per_thread, fx.BFloat16)
                fx.copy(
                    global_copy,
                    fx.slice(
                        query_row_packs,
                        (None, column // fx.Int32(values_per_thread)),
                    ),
                    q_fragment,
                )
                raw_query_packs.append(Vec(q_fragment.load()))
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            for k_step in fx.range_constexpr(qk_reduction_steps):
                query_register_packs.append(
                    (Vec(raw_query_packs[k_step].to(fx.Float32)) * query_scale).to(
                        fx.BFloat16
                    )
                )
        else:
            for load_step in fx.range_constexpr(query_load_iterations):
                linear = fx.Int32(load_step * num_threads) + tid
                row = linear // fx.Int32(qk_chunks_per_row)
                chunk = linear % fx.Int32(qk_chunks_per_row)
                column = chunk * fx.Int32(values_per_thread)
                row_valid, row_head, row_query_pos = row_coordinates(row)
                query_fragment = fx.make_rmem_tensor(
                    values_per_thread,
                    fx.BFloat16,
                )
                if const_expr(decode):
                    local_head = row_head - kv_head * fx.Int32(query_heads_per_kv_head)
                    source_row = fx.slice(
                        query_view,
                        (local_head, row_query_pos, None),
                    )
                else:
                    source_row = fx.slice(query_view, (row_query_pos, None))
                source = fx.logical_divide(
                    source_row,
                    fx.make_layout(values_per_thread, 1),
                )
                fx.copy(
                    global_copy,
                    fx.slice(source, (None, chunk)),
                    query_fragment,
                )
                query_value = Vec(query_fragment.load())
                if const_expr(decode):
                    query_value = Vec.from_elements(
                        [
                            row_valid.select(
                                query_value[element],
                                fx.BFloat16(0.0),
                            )
                            for element in fx.range_constexpr(values_per_thread)
                        ],
                        fx.BFloat16,
                    )
                scaled_query = Vec(query_value.to(fx.Float32)) * query_scale
                query_destination = fx.logical_divide(
                    fx.slice(shared_query, (row, None)),
                    fx.make_layout(values_per_thread, 1),
                )
                query_fragment.store(Vec(scaled_query).to(fx.BFloat16))
                fx.copy(
                    shared_copy,
                    query_fragment,
                    fx.slice(query_destination, (None, chunk)),
                )
            fx.gpu.barrier()

        def load_query_fragment(k_step):
            tile = fx.slice(query_tiles, (None, None, q_wave, k_step))
            fragment = thr_mma.make_fragment_B(tile)
            if const_expr(pipelined_kv):
                fragment.store(Vec(query_register_packs[k_step]))
            else:
                fx.copy(shared_copy, copy_q.partition_S(tile), copy_q.retile(fragment))
            return fragment

        zero16 = Vec.filled(16, 0.0, fx.Float32)
        output_accumulators = [zero16 for _ in fx.range_constexpr(output_chunks)]
        running_max = _f32(_NEG_BIG)
        running_sum = _f32(0.0)

        if const_expr(block_mask_batch == 1):
            mask_batch = fx.Int32(0)
        else:
            mask_batch = batch
        if const_expr(block_mask_heads == 1):
            mask_head = fx.Int32(0)
        elif const_expr(block_mask_heads == num_kv_heads):
            mask_head = kv_head
        else:
            mask_head = head
        if const_expr(decode):
            mask_q_block = fx.Int32(0)
        else:
            mask_q_block = q_base // fx.Int32(sparse_q_block_size)
        mask_row = (mask_batch * fx.Int32(block_mask_heads) + mask_head) * fx.Int32(
            num_q_blocks
        ) + mask_q_block
        if const_expr(causal_document_mask):
            document_id = load_i32(
                mask_buffers[0],
                query_pos * fx.Int32(mask_buffer_strides[0][0]),
            )
            document_start = load_i32(
                mask_buffers[1],
                document_id * fx.Int32(mask_buffer_strides[1][0]),
            )

        def stage_key(kv_base, stage=0):
            for load_step in fx.range_constexpr(key_load_iterations):
                load_tid = lane if const_expr(split_kv) else tid
                linear = fx.Int32(load_step * kv_load_threads) + load_tid
                # LDS DMA assigns consecutive physical packs to consecutive lanes.
                logical = fx.Int32(
                    fx.get_scalar(
                        fx.crd2idx(
                            linear * fx.Int32(values_per_thread),
                            key_copy_coordinates,
                        )
                    )
                )
                row = logical % fx.Int32(kv_tile_rows)
                chunk = logical // fx.Int32(kv_tile_rows * values_per_thread)
                source = fx.logical_divide(
                    fx.slice(key_view, (kv_base + row, None)),
                    fx.make_layout(values_per_thread, 1),
                )
                fx.copy(
                    lds_copy,
                    fx.slice(source, (None, chunk)),
                    fx.slice(key_copy_destinations[stage], (None, linear)),
                )

        def load_key_fragment(k_step, high_half, stage=0):
            tile = fx.slice(key_tiles[stage], (None, None, int(high_half), k_step))
            fragment = thr_mma.make_fragment_A(tile)
            fx.copy(shared_copy, copy_k.partition_S(tile), copy_k.retile(fragment))
            return fragment

        def stage_value(kv_base, stage=0):
            for load_step in fx.range_constexpr(value_load_iterations):
                load_tid = lane if const_expr(split_kv) else tid
                linear = fx.Int32(load_step * kv_load_threads) + load_tid
                logical = fx.Int32(
                    fx.get_scalar(
                        fx.crd2idx(
                            linear * fx.Int32(values_per_thread),
                            value_copy_coordinates,
                        )
                    )
                )
                row = logical % fx.Int32(kv_tile_rows)
                chunk = logical // fx.Int32(kv_tile_rows * values_per_thread)
                source = fx.logical_divide(
                    fx.slice(value_view, (kv_base + row, None)),
                    fx.make_layout(values_per_thread, 1),
                )
                fx.copy(
                    lds_copy,
                    fx.slice(source, (None, chunk)),
                    fx.slice(value_copy_destinations[stage], (None, linear)),
                )

        def load_value_fragment(probability_pack, d_chunk, stage=0):
            tile = fx.slice(value_tiles[stage], (None, None, d_chunk, probability_pack))
            fragment = thr_mma.make_fragment_A(tile)
            fx.copy(
                transposed_lds_copy,
                copy_v.partition_S(tile),
                copy_v.retile(fragment),
            )
            return fragment

        def process_tile(
            kv_chunk,
            masked,
            tile_output,
            tile_running_max,
            tile_running_sum,
            stage=0,
            tile_active=None,
        ):
            kv_base = kv_chunk * fx.Int32(kv_tile_rows)
            if const_expr(not pipelined_kv):
                stage_key(kv_base)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.gpu.barrier()

            scores_lo = zero16
            scores_hi = zero16
            for k_step in fx.range_constexpr(qk_reduction_steps):
                query_pack = load_query_fragment(k_step)
                key_lo = load_key_fragment(k_step, False, stage)
                key_hi = load_key_fragment(k_step, True, stage)
                scores_lo = mfma(key_lo, query_pack, scores_lo)
                scores_hi = mfma(key_hi, query_pack, scores_hi)
            schedule_fwd_qk_pipeline(
                reduction_steps=qk_reduction_steps,
                vmem_count=(key_load_iterations if pipelined_kv else 0),
            )

            if const_expr(not pipelined_kv):
                # Every wave must finish its K reads before the shared allocation
                # is reused for V.
                fx.gpu.barrier()
                stage_value(kv_base)

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
                        + fx.Int32(fx.get_scalar(accumulator_coordinates[element]))
                    )
                    keep = query_valid
                    if tile_active is not None:
                        keep = keep & tile_active
                    if const_expr(causal_document_mask):
                        if isinstance(masked, bool):
                            if masked:
                                keep = (
                                    keep
                                    & (query_pos >= key_pos)
                                    & (key_pos >= document_start)
                                )
                        else:
                            mask_keep = (query_pos >= key_pos) & (
                                key_pos >= document_start
                            )
                            keep = keep & ((~masked) | mask_keep)
                    elif const_expr(bool(mask_program)):
                        if not isinstance(masked, bool):
                            raise AssertionError(
                                "generic mask evaluation requires a static mask flag"
                            )
                        if masked:
                            keep = keep & evaluate_mask(query_pos, key_pos)
                    keep_values.append(keep)
                    score_values.append(keep.select(_f32(raw[element]), _f32(_NEG_BIG)))

            local_max = score_values[0]
            for element in fx.range_constexpr(1, 32):
                local_max = _maximum(local_max, score_values[element])
            peer_max = _f32(fx.gpu.shuffle_xor(local_max, mfma_tile_size, warp_size))
            tile_max = _maximum(local_max, peer_max)
            new_max = _maximum(tile_running_max, tile_max)
            correction = _exp2(tile_running_max - new_max)

            correction_vec = Vec.from_elements([correction], fx.Float32).broadcast_to(
                16
            )
            for d_chunk in fx.range_constexpr(output_chunks):
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
                    ).to(fx.BFloat16)
                )
            schedule_fwd_softmax_pipeline(vmem_count=value_load_iterations)
            peer_sum = _f32(fx.gpu.shuffle_xor(local_sum, mfma_tile_size, warp_size))
            tile_sum = local_sum + peer_sum
            tile_running_sum = tile_running_sum * correction + tile_sum
            tile_running_max = new_max

            if const_expr(not pipelined_kv):
                # V writes were issued before the register-only softmax.
                # Synchronize only when the LDS data is actually consumed.
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.gpu.barrier()
            for probability_pack in fx.range_constexpr(4):
                probability_fragment = fx.make_fragment_like(
                    probability_coordinates, fx.BFloat16
                )
                probability_fragment.store(Vec(probability_packs[probability_pack]))
                for d_chunk in fx.range_constexpr(output_chunks):
                    value_pack = load_value_fragment(
                        probability_pack,
                        d_chunk,
                        stage,
                    )
                    tile_output[d_chunk] = mfma(
                        value_pack,
                        probability_fragment,
                        tile_output[d_chunk],
                    )
            schedule_fwd_pv_pipeline(output_chunks=output_chunks)

            if const_expr(not pipelined_kv):
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
                first_chunk = first_block * fx.Int32(kv_tiles_per_sparse_block)
                stage_key(first_chunk * fx.Int32(kv_tile_rows), 0)
                stage_value(first_chunk * fx.Int32(kv_tile_rows), 0)
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
                        for d_chunk in fx.range_constexpr(output_chunks)
                    ]

                    first_chunk = current_block * fx.Int32(kv_tiles_per_sparse_block)
                    second_chunk = first_chunk + fx.Int32(1)
                    stage_key(second_chunk * fx.Int32(kv_tile_rows), 1)
                    stage_value(second_chunk * fx.Int32(kv_tile_rows), 1)
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
                        next_chunk = next_block * fx.Int32(kv_tiles_per_sparse_block)
                        stage_key(next_chunk * fx.Int32(kv_tile_rows), 0)
                        stage_value(next_chunk * fx.Int32(kv_tile_rows), 0)
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
            split_count = (block_count + fx.Int32(num_waves - 1)) // fx.Int32(num_waves)
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
                    for d_chunk in fx.range_constexpr(output_chunks)
                ]
                block_index = fx.Int32(split_index * num_waves) + wave
                tile_active = block_index < block_count
                safe_index = tile_active.select(block_index, fx.Int32(0))
                sparse_block = load_uniform_i32(
                    block_indices,
                    block_base + safe_index,
                )
                for sub_block in fx.range_constexpr(kv_tiles_per_sparse_block):
                    iter_output, iter_max, iter_sum = process_tile(
                        sparse_block * fx.Int32(kv_tiles_per_sparse_block)
                        + fx.Int32(sub_block),
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
                for d_chunk in fx.range_constexpr(output_chunks)
            ]
            reduction = fx.slice(
                lds.reduction_stats.view(fx.make_layout((2, 2), (2, 1))),
                (lane_half, None),
            )
            reduction_outputs = fx.slice(
                lds.reduction_output.view(
                    fx.make_layout(
                        (2, 16, output_chunks),
                        (output_chunks * 16, 1, 16),
                    )
                ),
                (lane_half, None, None),
            )
            if (wave == fx.Int32(1)) & (query_row_in_wave == fx.Int32(0)):
                reduction[0] = split_max
                reduction[1] = split_sum
                for d_chunk in fx.range_constexpr(output_chunks):
                    fx.slice(reduction_outputs, (None, d_chunk)).store(
                        Vec(split_output[d_chunk])
                    )
            fx.gpu.barrier()

            other_max = _f32(reduction[0])
            other_sum = _f32(reduction[1])
            combined_max = _maximum(split_max, other_max)
            split_scale = _exp2(split_max - combined_max)
            other_scale = _exp2(other_max - combined_max)
            split_sum = split_sum * split_scale + other_sum * other_scale
            split_max = combined_max
            split_scale_vec = Vec.from_elements([split_scale], fx.Float32).broadcast_to(
                16
            )
            other_scale_vec = Vec.from_elements([other_scale], fx.Float32).broadcast_to(
                16
            )
            for d_chunk in fx.range_constexpr(output_chunks):
                other_output = Vec(fx.slice(reduction_outputs, (None, d_chunk)).load())
                split_output[d_chunk] = (
                    Vec(split_output[d_chunk]) * split_scale_vec
                    + other_output * other_scale_vec
                )
            return [split_max, split_sum] + split_output

        def store_results(final_results, lse, max_scores):
            final_max = _f32(final_results[0])
            final_sum = _f32(final_results[1])
            final_output = [
                final_results[2 + d_chunk]
                for d_chunk in fx.range_constexpr(output_chunks)
            ]

            inverse_sum = (final_sum > _f32(0.0)).select(
                _f32(1.0) / final_sum,
                _f32(0.0),
            )
            inverse_vec = Vec.from_elements([inverse_sum], fx.Float32).broadcast_to(16)

            if const_expr(decode):
                local_head = query_head - kv_head * fx.Int32(query_heads_per_kv_head)
                output_source = fx.slice(
                    output_view,
                    (local_head, query_pos, None),
                )
            else:
                output_source = fx.slice(output_view, (query_pos, None))
            output_row = fx.logical_divide(
                output_source,
                fx.make_layout(4, 1),
            )
            store_valid = query_valid
            if const_expr(split_kv):
                store_valid = store_valid & (wave == fx.Int32(0))
            if store_valid:
                for d_chunk in fx.range_constexpr(output_chunks):
                    normalized = Vec(final_output[d_chunk]) * inverse_vec
                    for column_group in fx.range_constexpr(4):
                        values = Vec.from_elements(
                            [
                                normalized[column_group * 4 + element]
                                for element in fx.range_constexpr(4)
                            ],
                            fx.Float32,
                        ).to(fx.BFloat16)
                        column = fx.Int32(d_chunk * mfma_tile_size) + fx.Int32(
                            fx.get_scalar(accumulator_coordinates[column_group * 4])
                        )
                        fragment = fx.make_rmem_tensor(4, fx.BFloat16)
                        fragment.store(values)
                        fx.copy(
                            output_copy,
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
                if const_expr(not output_stats_in_log2):
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
                stats_offset = (batch * fx.Int32(num_q_heads) + query_head) * fx.Int32(
                    seq_q
                ) + query_pos
                lse[stats_offset] = lse_value
                max_scores[stats_offset] = max_value

        full_count = load_uniform_i32(full_kv_num_blocks_view, mask_row)
        partial_count = load_uniform_i32(kv_num_blocks_view, mask_row)
        full_base = mask_row * fx.Int32(max_full_blocks)
        partial_base = mask_row * fx.Int32(max_partial_blocks)
        initial_state = [running_max, running_sum] + output_accumulators

        if const_expr(split_kv):
            full_results = process_split_run(
                full_count,
                full_kv_indices_view,
                full_base,
                False,
                initial_state,
            )
            split_results = process_split_run(
                partial_count,
                kv_indices_view,
                partial_base,
                True,
                full_results,
            )
            final_results = reduce_split_results(split_results)
        elif const_expr(pipelined_kv):
            full_results = process_pipelined_run(
                full_count,
                full_kv_indices_view,
                full_base,
                False,
                initial_state,
            )
            final_results = process_pipelined_run(
                partial_count,
                kv_indices_view,
                partial_base,
                True,
                full_results,
            )
        elif const_expr(causal_document_mask):
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
                    for d_chunk in fx.range_constexpr(output_chunks)
                ]
                block_index_i32 = fx.Int32(block_index)
                is_partial = block_index_i32 >= full_count
                sparse_block = fx.Int32(0)
                if is_partial:
                    sparse_block = load_uniform_i32(
                        kv_indices_view,
                        partial_base + block_index_i32 - full_count,
                    )
                else:
                    sparse_block = load_uniform_i32(
                        full_kv_indices_view,
                        full_base + block_index_i32,
                    )
                for sub_block in fx.range_constexpr(kv_tiles_per_sparse_block):
                    iter_output, iter_max, iter_sum = process_tile(
                        sparse_block * fx.Int32(kv_tiles_per_sparse_block)
                        + fx.Int32(sub_block),
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
                    for d_chunk in fx.range_constexpr(output_chunks)
                ]
                sparse_block = load_uniform_i32(
                    full_kv_indices_view,
                    full_base + fx.Int32(block_index),
                )
                for sub_block in fx.range_constexpr(kv_tiles_per_sparse_block):
                    iter_output, iter_max, iter_sum = process_tile(
                        sparse_block * fx.Int32(kv_tiles_per_sparse_block)
                        + fx.Int32(sub_block),
                        False,
                        iter_output,
                        iter_max,
                        iter_sum,
                    )
                full_results = yield [iter_max, iter_sum] + iter_output

            running_max = _f32(full_results[0])
            running_sum = _f32(full_results[1])
            output_accumulators = [
                full_results[2 + d_chunk]
                for d_chunk in fx.range_constexpr(output_chunks)
            ]
            partial_state = [running_max, running_sum] + output_accumulators
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
                    for d_chunk in fx.range_constexpr(output_chunks)
                ]
                sparse_block = load_uniform_i32(
                    kv_indices_view,
                    partial_base + fx.Int32(block_index),
                )
                for sub_block in fx.range_constexpr(kv_tiles_per_sparse_block):
                    iter_output, iter_max, iter_sum = process_tile(
                        sparse_block * fx.Int32(kv_tiles_per_sparse_block)
                        + fx.Int32(sub_block),
                        True,
                        iter_output,
                        iter_max,
                        iter_sum,
                    )
                final_results = yield [iter_max, iter_sum] + iter_output
        store_results(final_results, logsumexp_view, max_scores_view)

    def launch_kernel(
        query,
        key,
        value,
        logsumexp,
        max_scores,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        mask_buffer_0,
        mask_buffer_1,
        mask_buffer_2,
        mask_buffer_3,
        output,
        stream,
    ):
        kernel(
            query,
            key,
            value,
            logsumexp,
            max_scores,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            mask_buffer_0,
            mask_buffer_1,
            mask_buffer_2,
            mask_buffer_3,
            output,
        ).launch(
            grid=(
                num_kv_heads if decode else num_q_heads,
                num_query_chunks,
                batch_size,
            ),
            block=(num_threads, 1, 1),
            stream=stream,
        )

    # Each JIT entry point exposes only its live mask buffers. The internal
    # placeholder slots alias kv_num_blocks and must remain unused above count.
    if mask_buffer_count == 0:

        @flyc.jit
        def launch(
            query: fx.Tensor,
            key: fx.Tensor,
            value: fx.Tensor,
            logsumexp: fx.Tensor,
            max_scores: fx.Tensor,
            kv_num_blocks: fx.Tensor,
            kv_indices: fx.Tensor,
            full_kv_num_blocks: fx.Tensor,
            full_kv_indices: fx.Tensor,
            output: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                query,
                key,
                value,
                logsumexp,
                max_scores,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                kv_num_blocks,
                kv_num_blocks,
                kv_num_blocks,
                kv_num_blocks,
                output,
                stream,
            )

    elif mask_buffer_count == 1:

        @flyc.jit
        def launch(
            query: fx.Tensor,
            key: fx.Tensor,
            value: fx.Tensor,
            logsumexp: fx.Tensor,
            max_scores: fx.Tensor,
            kv_num_blocks: fx.Tensor,
            kv_indices: fx.Tensor,
            full_kv_num_blocks: fx.Tensor,
            full_kv_indices: fx.Tensor,
            mask_buffer_0: fx.Tensor,
            output: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                query,
                key,
                value,
                logsumexp,
                max_scores,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                mask_buffer_0,
                kv_num_blocks,
                kv_num_blocks,
                kv_num_blocks,
                output,
                stream,
            )

    elif mask_buffer_count == 2:

        @flyc.jit
        def launch(
            query: fx.Tensor,
            key: fx.Tensor,
            value: fx.Tensor,
            logsumexp: fx.Tensor,
            max_scores: fx.Tensor,
            kv_num_blocks: fx.Tensor,
            kv_indices: fx.Tensor,
            full_kv_num_blocks: fx.Tensor,
            full_kv_indices: fx.Tensor,
            mask_buffer_0: fx.Tensor,
            mask_buffer_1: fx.Tensor,
            output: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                query,
                key,
                value,
                logsumexp,
                max_scores,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                mask_buffer_0,
                mask_buffer_1,
                kv_num_blocks,
                kv_num_blocks,
                output,
                stream,
            )

    elif mask_buffer_count == 3:

        @flyc.jit
        def launch(
            query: fx.Tensor,
            key: fx.Tensor,
            value: fx.Tensor,
            logsumexp: fx.Tensor,
            max_scores: fx.Tensor,
            kv_num_blocks: fx.Tensor,
            kv_indices: fx.Tensor,
            full_kv_num_blocks: fx.Tensor,
            full_kv_indices: fx.Tensor,
            mask_buffer_0: fx.Tensor,
            mask_buffer_1: fx.Tensor,
            mask_buffer_2: fx.Tensor,
            output: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                query,
                key,
                value,
                logsumexp,
                max_scores,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                mask_buffer_0,
                mask_buffer_1,
                mask_buffer_2,
                kv_num_blocks,
                output,
                stream,
            )

    else:

        @flyc.jit
        def launch(
            query: fx.Tensor,
            key: fx.Tensor,
            value: fx.Tensor,
            logsumexp: fx.Tensor,
            max_scores: fx.Tensor,
            kv_num_blocks: fx.Tensor,
            kv_indices: fx.Tensor,
            full_kv_num_blocks: fx.Tensor,
            full_kv_indices: fx.Tensor,
            mask_buffer_0: fx.Tensor,
            mask_buffer_1: fx.Tensor,
            mask_buffer_2: fx.Tensor,
            mask_buffer_3: fx.Tensor,
            output: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                query,
                key,
                value,
                logsumexp,
                max_scores,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                mask_buffer_0,
                mask_buffer_1,
                mask_buffer_2,
                mask_buffer_3,
                output,
                stream,
            )

    launch.compile_hints = {"waves_per_eu": waves_per_eu}
    return launch
