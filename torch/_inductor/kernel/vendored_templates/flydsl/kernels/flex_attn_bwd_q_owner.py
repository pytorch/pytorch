"""Parameterised dQ owner pipeline for FlexAttention backward."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from .flex_attn_utils import (
    make_global_view,
    make_mask_buffers,
    make_mask_evaluator,
    make_metadata_view,
    make_mfma32_ops,
    make_qk_shared_layout,
    make_value_shared_layout,
    schedule_pack0_pipeline,
    schedule_pack1_pipeline,
    schedule_score_pipeline,
    schedule_update_tail,
    scheduled_workgroup_barrier,
)

_LOG2E = 1.4426950408889634
_NEG_BIG = -1e30


def make_dq_mfma32_body(context, f32, exp2):
    batch_size = context["batch_size"]
    dense_mask = context["dense_mask"]
    grad_output_stride = context["grad_output_stride"]
    dq_query_rows = context["dq_query_rows"]
    dq_kv_rows = context["dq_kv_rows"]
    dq_key_load_iterations = context["dq_key_load_iterations"]
    dq_value_load_iterations = context["dq_value_load_iterations"]
    dq_num_kv_chunks = context["dq_num_kv_chunks"]
    dq_num_threads = context["dq_num_threads"]
    qk_head_dim = context["qk_head_dim"]
    grad_query_stride = context["grad_query_stride"]
    value_head_dim = context["value_head_dim"]
    num_heads = context["num_heads"]
    k_stride = context["k_stride"]
    lse_in_log2 = context["lse_in_log2"]
    stats_stride = context["stats_stride"]
    mask_buffer_count = context["mask_buffer_count"]
    mask_buffer_sizes = context["mask_buffer_sizes"]
    mask_buffer_strides = context["mask_buffer_strides"]
    mask_program = context["mask_program"]
    mask_program_output = context["mask_program_output"]
    num_query_chunks = context["num_query_chunks"]
    num_kv_chunks = context["num_kv_chunks"]
    q_stride = context["q_stride"]
    sequence_length = context["sequence_length"]
    scale_log2 = context["scale_log2"]
    softmax_scale = context["softmax_scale"]
    values_per_thread = context["values_per_thread"]
    v_stride = context["v_stride"]
    _f32 = f32
    _exp2 = exp2

    @flyc.jit
    def _emit_dq_mfma32_body(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        logsumexp: fx.Tensor,
        delta: fx.Tensor,
        grad_output: fx.Tensor,
        grad_query: fx.Tensor,
        partial_counts: fx.Tensor,
        partial_indices: fx.Tensor,
        full_counts: fx.Tensor,
        full_indices: fx.Tensor,
        mask_buffer_0: fx.Tensor,
        mask_buffer_1: fx.Tensor,
        mask_buffer_2: fx.Tensor,
        mask_buffer_3: fx.Tensor,
        query_chunk: fx.Int32,
        batch_head: fx.Int32,
        persistent_shared_pointer,
        short_shared_pointer,
    ):
        """dQ owner with asymmetric operand-native LDS staging.

        query and dO are loaded directly into persistent MFMA registers. Each KV
        tile keeps key in two alternating LDS slots because both score and dQ
        consume it. value is score-only and reuses one LDS slot after score/dP.
        """
        tid = fx.Int32(fx.thread_idx.x)
        batch = batch_head // fx.Int32(num_heads)
        head = batch_head % fx.Int32(num_heads)
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)
        (
            dma128,
            tr16,
            o64,
            tiled_mma,
            thread_mma,
            accumulator_coordinates,
            row_coordinates,
            gload_f32,
            gload_i32,
            load_global_pack,
            make_b_fragment,
        ) = make_mfma32_ops(lane, values_per_thread)
        qk_coordinates = thread_mma.partition_B(
            fx.make_view(
                0,
                fx.make_layout((32, qk_head_dim), (0, 1)),
            )
        )
        value_coordinates = thread_mma.partition_B(
            fx.make_view(
                0,
                fx.make_layout((32, value_head_dim), (0, 1)),
            )
        )
        query_base = query_chunk * fx.Int32(dq_query_rows)
        query_local = fx.Int32(32) * wave + fx.Int32(
            fx.get_scalar(row_coordinates[0])
        )
        query_position = query_base + query_local
        batch_i64 = fx.Int64(batch)
        head_i64 = fx.Int64(head)
        bh_i64 = fx.Int64(batch_head)
        head_coord = (batch_i64, head_i64, None, None)
        query_view = make_global_view(
            query,
            head_coord,
            (batch_size, num_heads, sequence_length, qk_head_dim),
            q_stride,
        )
        key_view = make_global_view(
            key,
            head_coord,
            (batch_size, num_heads, sequence_length, qk_head_dim),
            k_stride,
        )
        value_view = make_global_view(
            value,
            head_coord,
            (batch_size, num_heads, sequence_length, value_head_dim),
            v_stride,
        )
        grad_output_view = make_global_view(
            grad_output,
            head_coord,
            (batch_size, num_heads, sequence_length, value_head_dim),
            grad_output_stride,
        )
        grad_query_view = make_global_view(
            grad_query,
            head_coord,
            (batch_size, num_heads, sequence_length, qk_head_dim),
            grad_query_stride,
        )
        logsumexp_view = make_global_view(
            logsumexp,
            (bh_i64, None),
            (batch_size * num_heads, sequence_length),
            (stats_stride, 1),
        )
        delta_view = make_global_view(
            delta,
            (bh_i64, None),
            (batch_size * num_heads, sequence_length),
            (stats_stride, 1),
        )
        partial_counts_view = make_metadata_view(
            partial_counts,
            batch_head * fx.Int32(num_query_chunks),
            num_query_chunks,
        )
        full_counts_view = make_metadata_view(
            full_counts,
            batch_head * fx.Int32(num_query_chunks),
            num_query_chunks,
        )
        partial_indices_view = make_metadata_view(
            partial_indices,
            batch_head * fx.Int32(num_query_chunks * num_kv_chunks),
            num_query_chunks * num_kv_chunks,
        )
        full_indices_view = make_metadata_view(
            full_indices,
            batch_head * fx.Int32(num_query_chunks * num_kv_chunks),
            num_query_chunks * num_kv_chunks,
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
        evaluate_mask = make_mask_evaluator(
            mask_program,
            mask_program_output,
            mask_buffer_strides,
            mask_buffers,
            gload_i32,
            batch,
            head,
        )

        global_to_shared_copy = dma128
        shared_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        score_copy = fx.make_tiled_copy_A(shared_copy, tiled_mma).get_slice(lane)
        update_copy = fx.make_tiled_copy_A(tr16, tiled_mma).get_slice(lane)

        key_shared_layout = make_value_shared_layout(dq_kv_rows, qk_head_dim)
        value_shared_layout = make_qk_shared_layout(dq_kv_rows, value_head_dim)
        key_stage_pointers = [
            fx.add_offset(
                persistent_shared_pointer,
                fx.make_int_tuple(stage * dq_kv_rows * qk_head_dim),
            )
            for stage in fx.range_constexpr(2)
        ]
        shared_keys = [
            fx.make_view(pointer, key_shared_layout) for pointer in key_stage_pointers
        ]
        shared_value = fx.make_view(short_shared_pointer, value_shared_layout)
        key_copy_coordinates = fx.right_inverse(key_shared_layout)
        value_copy_coordinates = fx.make_composed_layout(
            fx.right_inverse(value_shared_layout.outer),
            fx.make_composed_layout(
                value_shared_layout.inner,
                fx.make_layout(dq_kv_rows * value_head_dim, 1),
            ),
        )
        key_copy_destinations = [
            fx.logical_divide(
                fx.make_view(
                    pointer,
                    fx.make_layout(dq_kv_rows * qk_head_dim, 1),
                ),
                fx.make_layout(values_per_thread, 1),
            )
            for pointer in key_stage_pointers
        ]
        value_copy_destination = fx.logical_divide(
            fx.make_view(
                short_shared_pointer,
                fx.make_layout(dq_kv_rows * value_head_dim, 1),
            ),
            fx.make_layout(values_per_thread, 1),
        )
        key_score_tiles = [
            fx.flat_divide(shared_key, (32, 16)) for shared_key in shared_keys
        ]
        value_score_tiles = fx.flat_divide(shared_value, (32, 16))
        key_update_layout = fx.composition(
            fx.select(key_shared_layout, [1, 0]),
            fx.make_tile(
                fx.make_layout(qk_head_dim, 1),
                fx.make_layout(
                    (4, 2, 2, dq_kv_rows // 16),
                    (1, 8, 4, 16),
                ),
            ),
        )
        key_update_tiles = [
            fx.flat_divide(
                fx.make_view(pointer, key_update_layout),
                (32, 16),
            )
            for pointer in key_stage_pointers
        ]

        def stage_operand(
            view,
            copy_coordinates,
            destination,
            chunk,
            dimension: fx.Constexpr[int],
            load_iterations: fx.Constexpr[int],
        ):
            row_base = chunk * fx.Int32(dq_kv_rows)
            for load_step in fx.range_constexpr(load_iterations):
                linear = fx.Int32(load_step * dq_num_threads) + tid
                logical = fx.Int32(
                    fx.get_scalar(
                        fx.crd2idx(
                            linear * fx.Int32(values_per_thread),
                            copy_coordinates,
                        )
                    )
                )
                row = logical % fx.Int32(dq_kv_rows)
                pack = logical // fx.Int32(dq_kv_rows * values_per_thread)
                source = fx.logical_divide(
                    fx.slice(view, (row_base + row, None)),
                    fx.make_layout(values_per_thread, 1),
                )
                fx.copy(
                    global_to_shared_copy,
                    fx.slice(source, (None, pack)),
                    fx.slice(destination, (None, linear)),
                )

        def stage_persistent(kv_chunk, stage: fx.Constexpr[int]):
            stage_operand(
                key_view,
                key_copy_coordinates,
                key_copy_destinations[stage],
                kv_chunk,
                qk_head_dim,
                dq_key_load_iterations,
            )

        def stage_short(kv_chunk):
            stage_operand(
                value_view,
                value_copy_coordinates,
                value_copy_destination,
                kv_chunk,
                value_head_dim,
                dq_value_load_iterations,
            )

        def load_score_fragment(tiles, row_block, k_step):
            tile = fx.slice(tiles, (None, None, row_block, k_step))
            fragment = thread_mma.make_fragment_A(tile)
            fx.copy(
                shared_copy,
                score_copy.partition_S(tile),
                score_copy.retile(fragment),
            )
            return fragment

        def load_update_fragment(
            stage: fx.Constexpr[int],
            row_block,
            half: fx.Constexpr[int],
            d_chunk,
        ):
            probability_pack = row_block * fx.Int32(2) + fx.Int32(half)
            tile = fx.slice(
                key_update_tiles[stage],
                (None, None, d_chunk, probability_pack),
            )
            fragment = thread_mma.make_fragment_A(tile)
            fx.copy(
                tr16,
                update_copy.partition_S(tile),
                update_copy.retile(fragment),
            )
            return fragment

        def mma_fragments(a_fragment, b_fragment, accumulator):
            accumulator_fragment = fx.make_fragment_like(
                accumulator_coordinates,
                fx.Float32,
            )
            accumulator_fragment.store(fx.Vector(accumulator))
            fx.gemm(
                tiled_mma,
                accumulator_fragment,
                a_fragment,
                b_fragment,
                accumulator_fragment,
            )
            return accumulator_fragment.load()

        q_packs = []
        for k_step in fx.range_constexpr(qk_head_dim // 16):
            column = fx.Int32(fx.get_scalar(qk_coordinates[0, 0, k_step]))
            q_packs.append(load_global_pack(query_view, query_position, column))
        do_packs = []
        for k_step in fx.range_constexpr(value_head_dim // 16):
            column = fx.Int32(fx.get_scalar(value_coordinates[0, 0, k_step]))
            do_packs.append(load_global_pack(grad_output_view, query_position, column))
        lse = gload_f32(logsumexp_view, query_position)
        delta = gload_f32(delta_view, query_position)
        lse_log2 = lse if lse_in_log2 else lse * _f32(_LOG2E)
        lse_log2 = (lse < _f32(_NEG_BIG)).select(_f32(0.0), lse_log2)
        zero16 = fx.Vector.filled(16, 0.0, fx.Float32).ir_value()
        grad_query_accumulators = [
            fx.make_fragment_like(accumulator_coordinates, fx.Float32)
            for _ in fx.range_constexpr(qk_head_dim // 32)
        ]
        for d_chunk in fx.range_constexpr(qk_head_dim // 32):
            grad_query_accumulators[d_chunk].fill(0)

        def process_tile(
            kv_chunk,
            masked,
            stage: fx.Constexpr[int],
            next_kchunk,
            overlap_next,
            handoff: fx.Constexpr[bool],
        ):
            for reduction_sub in fx.range_constexpr(dq_kv_rows // 32):
                score = zero16
                dp = zero16
                for k_step in fx.range_constexpr(qk_head_dim // 16):
                    score = mma_fragments(
                        load_score_fragment(
                            key_score_tiles[stage], reduction_sub, k_step
                        ),
                        make_b_fragment(q_packs[k_step]),
                        score,
                    )
                for k_step in fx.range_constexpr(value_head_dim // 16):
                    dp = mma_fragments(
                        load_score_fragment(value_score_tiles, reduction_sub, k_step),
                        make_b_fragment(do_packs[k_step]),
                        dp,
                    )
                schedule_score_pipeline(
                    mfma_count=(qk_head_dim + value_head_dim) // 16,
                    dsrd_count=(qk_head_dim + value_head_dim) // 16,
                    vmem_count=(
                        dq_key_load_iterations
                        if overlap_next and reduction_sub == 0
                        else 0
                    ),
                )
                recycle_short = overlap_next and reduction_sub == dq_kv_rows // 32 - 1
                if const_expr(recycle_short):
                    scheduled_workgroup_barrier()
                    stage_short(next_kchunk)
                kv_base = kv_chunk * fx.Int32(dq_kv_rows) + fx.Int32(reduction_sub * 32)
                score_values = fx.Vector(score)
                dp_values = fx.Vector(dp)
                first_ds_values = []
                for element in fx.range_constexpr(8):
                    score_index = element
                    key_local = fx.Int32(
                        fx.get_scalar(accumulator_coordinates[score_index])
                    )
                    key_pos = kv_base + key_local
                    probability = _exp2(
                        _f32(score_values[score_index]) * _f32(scale_log2) - lse_log2
                    )
                    if not isinstance(masked, bool):
                        raise AssertionError(
                            "generic mask evaluation requires a static mask flag"
                        )
                    if masked:
                        probability = evaluate_mask(query_position, key_pos).select(
                            probability, _f32(0.0)
                        )
                    first_ds_values.append(
                        probability * (_f32(dp_values[score_index]) - delta)
                    )
                first_ds_fragment = make_b_fragment(
                    fx.Vector.from_elements(first_ds_values, fx.Float32)
                    .to(fx.BFloat16)
                )
                first_update_fragments = []
                for d_chunk in fx.range_constexpr(qk_head_dim // 32):
                    first_update_fragments.append(
                        load_update_fragment(
                            stage,
                            reduction_sub,
                            0,
                            d_chunk,
                        )
                    )
                schedule_pack0_pipeline(
                    vmem_count=dq_key_load_iterations if recycle_short else 0,
                    exp_count=8,
                    dsrd_count=4 * (qk_head_dim // 32),
                )
                second_ds_values = []
                for element in fx.range_constexpr(8):
                    score_index = 8 + element
                    key_local = fx.Int32(
                        fx.get_scalar(accumulator_coordinates[score_index])
                    )
                    key_pos = kv_base + key_local
                    probability = _exp2(
                        _f32(score_values[score_index]) * _f32(scale_log2) - lse_log2
                    )
                    if not isinstance(masked, bool):
                        raise AssertionError(
                            "generic mask evaluation requires a static mask flag"
                        )
                    if masked:
                        probability = evaluate_mask(query_position, key_pos).select(
                            probability, _f32(0.0)
                        )
                    second_ds_values.append(
                        probability * (_f32(dp_values[score_index]) - delta)
                    )
                second_ds_fragment = make_b_fragment(
                    fx.Vector.from_elements(second_ds_values, fx.Float32)
                    .to(fx.BFloat16)
                )
                second_update_fragments = []
                for d_chunk in fx.range_constexpr(min(2, qk_head_dim // 32)):
                    second_update_fragments.append(
                        load_update_fragment(
                            stage,
                            reduction_sub,
                            1,
                            d_chunk,
                        )
                    )
                for d_chunk in fx.range_constexpr(qk_head_dim // 32):
                    fx.gemm(
                        tiled_mma,
                        grad_query_accumulators[d_chunk],
                        first_update_fragments[d_chunk],
                        first_ds_fragment,
                        grad_query_accumulators[d_chunk],
                    )
                schedule_pack1_pipeline(
                    mfma_count=qk_head_dim // 32,
                    exp_count=8,
                    dsrd_count=2 * (qk_head_dim // 32),
                )
                for d_chunk in fx.range_constexpr(qk_head_dim // 32):
                    fx.gemm(
                        tiled_mma,
                        grad_query_accumulators[d_chunk],
                        second_update_fragments[d_chunk],
                        second_ds_fragment,
                        grad_query_accumulators[d_chunk],
                    )
                    if const_expr(d_chunk + 2 < qk_head_dim // 32):
                        second_update_fragments.append(
                            load_update_fragment(
                                stage,
                                reduction_sub,
                                1,
                                d_chunk + 2,
                            )
                        )
                schedule_update_tail(
                    mfma_count=qk_head_dim // 32, dsrd_count=2 * (qk_head_dim // 32)
                )
            if const_expr(handoff):
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()

        if const_expr(dense_mask):
            first_tile = fx.Int32(0)
            tile_count = fx.Int32(dq_num_kv_chunks)
            stage_persistent(first_tile, 0)
            stage_short(first_tile)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            stage_persistent(first_tile + fx.Int32(1), 1)
            process_tile(
                first_tile,
                False,
                0,
                first_tile + fx.Int32(1),
                True,
                True,
            )
            remaining_tiles = tile_count - fx.Int32(1)
            double_buffered_tiles = remaining_tiles - remaining_tiles % fx.Int32(2)
            for tile_index in range(fx.Int32(0), double_buffered_tiles, fx.Int32(2)):
                local_tile = fx.Int32(tile_index) + fx.Int32(1)
                current_tile = first_tile + local_tile
                stage_persistent(current_tile + fx.Int32(1), 0)
                process_tile(
                    current_tile,
                    False,
                    1,
                    current_tile + fx.Int32(1),
                    True,
                    True,
                )
                next_pair_tile = local_tile + fx.Int32(2)
                if next_pair_tile < tile_count:
                    stage_persistent(first_tile + next_pair_tile, 1)
                process_tile(
                    current_tile + fx.Int32(1),
                    False,
                    0,
                    first_tile + next_pair_tile,
                    True,
                    True,
                )
            if double_buffered_tiles < remaining_tiles:
                current_tile = first_tile + double_buffered_tiles + fx.Int32(1)
                process_tile(current_tile, False, 1, current_tile, False, False)
        else:
            worklist_row = query_chunk * fx.Int32(num_kv_chunks)
            partial_count = gload_i32(partial_counts_view, query_chunk)
            full_count = gload_i32(full_counts_view, query_chunk)
            for ti in range(fx.Int32(0), partial_count, fx.Int32(1)):
                kv_chunk = gload_i32(partial_indices_view, worklist_row + fx.Int32(ti))
                stage_persistent(kv_chunk, 0)
                stage_short(kv_chunk)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                process_tile(kv_chunk, True, 0, kv_chunk, False, False)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
            for ti in range(fx.Int32(0), full_count, fx.Int32(1)):
                kv_chunk = gload_i32(full_indices_view, worklist_row + fx.Int32(ti))
                stage_persistent(kv_chunk, 0)
                stage_short(kv_chunk)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                process_tile(kv_chunk, False, 0, kv_chunk, False, False)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
        output_row = fx.logical_divide(
            fx.slice(grad_query_view, (query_position, None)), fx.make_layout(4, 1)
        )
        output_fragment = fx.make_rmem_tensor(4, fx.BFloat16)
        scale_vector = fx.Vector.filled(16, softmax_scale, fx.Float32)
        for d_chunk in fx.range_constexpr(qk_head_dim // 32):
            values = fx.Vector(grad_query_accumulators[d_chunk].load()) * scale_vector
            values_bf16 = values.to(fx.BFloat16)
            for pack in fx.range_constexpr(4):
                output = fx.Vector.from_elements(
                    [
                        values_bf16[pack * 4 + element]
                        for element in fx.range_constexpr(4)
                    ],
                    fx.BFloat16,
                )
                output_fragment.store(output.ir_value())
                column = fx.Int32(d_chunk * 32) + fx.Int32(
                    fx.get_scalar(accumulator_coordinates[pack * 4])
                )
                fx.copy(
                    o64,
                    output_fragment,
                    fx.slice(output_row, (None, column // fx.Int32(4))),
                )

    return _emit_dq_mfma32_body
