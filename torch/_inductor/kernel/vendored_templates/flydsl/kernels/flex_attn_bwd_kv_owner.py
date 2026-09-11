"""Parameterised paired dK/dV owner pipeline for FlexAttention backward."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from .flex_attn_utils import (
    make_global_view,
    make_mask_buffers,
    make_mask_evaluator,
    make_metadata_view,
    make_mfma32_ops,
    make_shared_view,
    make_value_shared_layout,
    schedule_fence,
    schedule_pack0_pipeline,
    schedule_pack1_pipeline,
    schedule_score_pipeline,
    schedule_update_tail,
)

_LOG2E = 1.4426950408889634
_NEG_BIG = -1e30
_MASK_NONE = 0
_MASK_ALL = 1


def make_dkdv_mfma32_body(context, f32, exp2):
    batch_size = context["batch_size"]
    dense_mask = context["dense_mask"]
    pair_query_rows = context["pair_query_rows"]
    pair_kv_rows = context["pair_kv_rows"]
    pair_grad_output_load_iterations = context["pair_grad_output_load_iterations"]
    pair_key_split = context["pair_key_split"]
    pair_list_query_split = context["pair_list_query_split"]
    pair_logical_waves = context["pair_logical_waves"]
    pair_num_query_chunks = context["pair_num_query_chunks"]
    pair_query_load_iterations = context["pair_query_load_iterations"]
    grad_key_stride = context["grad_key_stride"]
    grad_output_stride = context["grad_output_stride"]
    qk_head_dim = context["qk_head_dim"]
    value_head_dim = context["value_head_dim"]
    grad_value_stride = context["grad_value_stride"]
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
    compute_metadata_elements = context["compute_metadata_elements"]
    q_stride = context["q_stride"]
    sequence_length = context["sequence_length"]
    scale_log2 = context["scale_log2"]
    softmax_scale = context["softmax_scale"]
    values_per_thread = context["values_per_thread"]
    v_stride = context["v_stride"]
    _f32 = f32
    _exp2 = exp2

    @flyc.jit
    def _emit_dkdv_mfma32_body(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        logsumexp: fx.Tensor,
        delta: fx.Tensor,
        grad_output: fx.Tensor,
        grad_key: fx.Tensor,
        grad_value: fx.Tensor,
        partial_counts: fx.Tensor,
        partial_indices: fx.Tensor,
        full_counts: fx.Tensor,
        full_indices: fx.Tensor,
        mask_buffer_0: fx.Tensor,
        mask_buffer_1: fx.Tensor,
        mask_buffer_2: fx.Tensor,
        mask_buffer_3: fx.Tensor,
        kv_chunk: fx.Int32,
        batch_head: fx.Int32,
        query_grad_output_shared_pointer,
        probability_shared_pointer,
        metadata_shared_pointer,
    ):
        """Compute dK and dV with paired producer and consumer waves.

        Producers calculate P/dS/dK; consumers reuse P for dV while loading the
        next tile. qk_head_dim and value_head_dim use independent compile-time
        chunks and layouts.
        """
        tid = fx.Int32(fx.thread_idx.x)
        batch = batch_head // fx.Int32(num_heads)
        head = batch_head % fx.Int32(num_heads)
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)
        logical_wave = wave % fx.Int32(pair_logical_waves)
        producer_wave = wave < fx.Int32(pair_logical_waves)
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
        kv_base = kv_chunk * fx.Int32(pair_kv_rows)
        key_pos = (
            kv_base
            + logical_wave * fx.Int32(32)
            + fx.Int32(fx.get_scalar(row_coordinates[0]))
        )
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
        grad_key_view = make_global_view(
            grad_key,
            head_coord,
            (batch_size, num_heads, sequence_length, qk_head_dim),
            grad_key_stride,
        )
        grad_value_view = make_global_view(
            grad_value,
            head_coord,
            (batch_size, num_heads, sequence_length, value_head_dim),
            grad_value_stride,
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
        logsumexp_packs = fx.logical_divide(logsumexp_view, fx.make_layout(4, 1))
        delta_packs_view = fx.logical_divide(delta_view, fx.make_layout(4, 1))
        partial_counts_view = make_metadata_view(
            partial_counts,
            batch_head * fx.Int32(num_kv_chunks),
            num_kv_chunks,
        )
        full_counts_view = make_metadata_view(
            full_counts,
            batch_head * fx.Int32(num_kv_chunks),
            num_kv_chunks,
        )
        partial_indices_view = make_metadata_view(
            partial_indices,
            batch_head * fx.Int32(num_kv_chunks * num_query_chunks),
            num_kv_chunks * num_query_chunks,
        )
        full_indices_view = make_metadata_view(
            full_indices,
            batch_head * fx.Int32(num_kv_chunks * num_query_chunks),
            num_kv_chunks * num_query_chunks,
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
        metadata_shared = make_shared_view(
            metadata_shared_pointer, compute_metadata_elements, 1
        )
        metadata_packs = fx.logical_divide(metadata_shared, fx.make_layout(4, 1))
        evaluate_mask = make_mask_evaluator(
            mask_program,
            mask_program_output,
            mask_buffer_strides,
            mask_buffers,
            gload_i32,
            batch,
            head,
        )
        q_stage_elems = pair_query_rows * qk_head_dim
        q_all_stages_elems = 2 * q_stage_elems
        probability_wave_elems = 64 * 16
        probability_stage_elems = pair_logical_waves * probability_wave_elems
        grad_output_shared_pointer = query_grad_output_shared_pointer + fx.Int32(
            q_all_stages_elems
        )
        loader_tid = (wave - fx.Int32(pair_logical_waves)) * fx.Int32(64) + lane

        global_to_shared_copy = dma128
        shared_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        score_copy = fx.make_tiled_copy_A(shared_copy, tiled_mma).get_slice(lane)
        update_copy = fx.make_tiled_copy_A(tr16, tiled_mma).get_slice(lane)
        query_shared_layout = make_value_shared_layout(pair_query_rows, qk_head_dim)
        grad_output_shared_layout = make_value_shared_layout(
            pair_query_rows, value_head_dim
        )
        query_copy_coordinates = fx.right_inverse(query_shared_layout)
        grad_output_copy_coordinates = fx.right_inverse(grad_output_shared_layout)

        def make_update_layout(shared_layout, dimension):
            return fx.composition(
                fx.select(shared_layout, [1, 0]),
                fx.make_tile(
                    fx.make_layout(dimension, 1),
                    fx.make_layout(
                        (4, 2, 2, pair_query_rows // 16),
                        (1, 8, 4, 16),
                    ),
                ),
            )

        query_update_layout = make_update_layout(query_shared_layout, qk_head_dim)
        grad_output_update_layout = make_update_layout(
            grad_output_shared_layout,
            value_head_dim,
        )

        def stage_pointer(base, stage, stage_elements: fx.Constexpr[int]):
            stages = fx.make_view(
                base,
                fx.make_layout(
                    (2, stage_elements),
                    (stage_elements, 1),
                ),
            )
            return fx.get_iter(fx.slice(stages, (stage, None)))

        def copy_destination(base, stage, stage_elements: fx.Constexpr[int]):
            return fx.logical_divide(
                fx.make_view(
                    stage_pointer(base, stage, stage_elements),
                    fx.make_layout(stage_elements, 1),
                ),
                fx.make_layout(values_per_thread, 1),
            )

        def stage_operand(
            view,
            copy_coordinates,
            destination,
            query_chunk,
            load_iterations: fx.Constexpr[int],
        ):
            query_base = query_chunk * fx.Int32(pair_query_rows)
            for load_step in fx.range_constexpr(load_iterations):
                linear = fx.Int32(load_step * pair_logical_waves * 64) + loader_tid
                logical = fx.Int32(
                    fx.get_scalar(
                        fx.crd2idx(
                            linear * fx.Int32(values_per_thread),
                            copy_coordinates,
                        )
                    )
                )
                row = logical % fx.Int32(pair_query_rows)
                pack = logical // fx.Int32(pair_query_rows * values_per_thread)
                source = fx.logical_divide(
                    fx.slice(view, (query_base + row, None)),
                    fx.make_layout(values_per_thread, 1),
                )
                fx.copy(
                    global_to_shared_copy,
                    fx.slice(source, (None, pack)),
                    fx.slice(destination, (None, linear)),
                )

        def stage_metadata(query_chunk, stage):
            query_base = query_chunk * fx.Int32(pair_query_rows)
            stat_base = fx.Int32(stage * 2 * pair_query_rows)
            if const_expr(lse_in_log2):
                if loader_tid < fx.Int32(pair_query_rows // 4):
                    source_pack = (
                        query_chunk * fx.Int32(pair_query_rows // 4) + loader_tid
                    )
                    destination_pack = (
                        fx.Int32(stage * (pair_query_rows // 2)) + loader_tid
                    )
                    fx.copy(
                        dma128,
                        fx.slice(logsumexp_packs, (None, source_pack)),
                        fx.slice(metadata_packs, (None, destination_pack)),
                    )
                    fx.copy(
                        dma128,
                        fx.slice(delta_packs_view, (None, source_pack)),
                        fx.slice(
                            metadata_packs,
                            (None, destination_pack + fx.Int32(pair_query_rows // 4)),
                        ),
                    )
            elif loader_tid < fx.Int32(pair_query_rows):
                lse = gload_f32(logsumexp_view, query_base + loader_tid)
                lse_log2 = lse * _f32(_LOG2E)
                lse_log2 = (lse < _f32(_NEG_BIG)).select(_f32(0.0), lse_log2)
                fx.ptr_store(lse_log2, metadata_shared_pointer + stat_base + loader_tid)
                delta_value = gload_f32(delta_view, query_base + loader_tid)
                fx.ptr_store(
                    delta_value,
                    metadata_shared_pointer
                    + stat_base
                    + fx.Int32(pair_query_rows)
                    + loader_tid,
                )

        def stage_q_metadata(query_chunk, stage):
            stage_operand(
                query_view,
                query_copy_coordinates,
                copy_destination(
                    query_grad_output_shared_pointer,
                    stage,
                    pair_query_rows * qk_head_dim,
                ),
                query_chunk,
                pair_query_load_iterations,
            )
            stage_metadata(query_chunk, stage)

        def stage_do(query_chunk, stage):
            stage_operand(
                grad_output_view,
                grad_output_copy_coordinates,
                copy_destination(
                    grad_output_shared_pointer,
                    stage,
                    pair_query_rows * value_head_dim,
                ),
                query_chunk,
                pair_grad_output_load_iterations,
            )

        def stage_qdo_metadata(query_chunk, stage):
            stage_q_metadata(query_chunk, stage)
            stage_do(query_chunk, stage)

        def load_score_fragment(
            base,
            shared_layout,
            stage,
            stage_elements: fx.Constexpr[int],
            k_step,
        ):
            shared_operand = fx.make_view(
                stage_pointer(base, stage, stage_elements),
                shared_layout,
            )
            tiles = fx.flat_divide(shared_operand, (32, 16))
            tile = fx.slice(tiles, (None, None, 0, k_step))
            fragment = thread_mma.make_fragment_A(tile)
            fx.copy(
                shared_copy,
                score_copy.partition_S(tile),
                score_copy.retile(fragment),
            )
            return fragment

        def load_update_fragment(
            base,
            update_layout,
            stage,
            stage_elements: fx.Constexpr[int],
            q_step,
            d_chunk,
        ):
            tiles = fx.flat_divide(
                fx.make_view(
                    stage_pointer(base, stage, stage_elements),
                    update_layout,
                ),
                (32, 16),
            )
            tile = fx.slice(
                tiles,
                (None, None, d_chunk, q_step),
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

        def probability_pointer(stage, half):
            element_offset = (
                fx.Int32(stage * probability_stage_elems)
                + logical_wave * fx.Int32(probability_wave_elems)
                + lane * fx.Int32(16)
                + fx.Int32(half * 8)
            )
            return probability_shared_pointer + element_offset

        def store_probability(stage, first, second):
            fx.ptr_store(first, probability_pointer(stage, 0))
            fx.ptr_store(second, probability_pointer(stage, 1))

        def load_probability(stage, half):
            return fx.ptr_load(
                probability_pointer(stage, half),
                result_type=fx.Vector.make_type(8, fx.BFloat16),
            )

        def qchunk_at(
            tile_index,
            first_tile,
            indices,
            list_row,
            indexed: fx.Constexpr[bool],
        ):
            if const_expr(indexed):
                list_index = tile_index // fx.Int32(pair_list_query_split)
                within_list_tile = tile_index % fx.Int32(pair_list_query_split)
                return (
                    gload_i32(indices, list_row + list_index)
                    * fx.Int32(pair_list_query_split)
                    + within_list_tile
                )
            return first_tile + tile_index

        def tile_is_masked(tile_index, mask_mode: fx.Constexpr[int]):
            if const_expr(mask_mode == _MASK_NONE):
                return False
            return True

        zero16 = fx.Vector.filled(16, 0.0, fx.Float32).ir_value()
        if producer_wave:
            k_packs = []
            v_packs = []
            for k_step in fx.range_constexpr(max(qk_head_dim, value_head_dim) // 16):
                if const_expr(k_step < qk_head_dim // 16):
                    column = fx.Int32(fx.get_scalar(qk_coordinates[0, 0, k_step]))
                    k_packs.append(load_global_pack(key_view, key_pos, column))
                if const_expr(k_step < value_head_dim // 16):
                    column = fx.Int32(fx.get_scalar(value_coordinates[0, 0, k_step]))
                    v_packs.append(load_global_pack(value_view, key_pos, column))
            grad_key_accumulators = [
                fx.make_fragment_like(accumulator_coordinates, fx.Float32)
                for _ in fx.range_constexpr(qk_head_dim // 32)
            ]
            for d_chunk in fx.range_constexpr(qk_head_dim // 32):
                grad_key_accumulators[d_chunk].fill(0)

            def produce_tile(query_chunk, masked, stage):
                score = zero16
                dp = zero16
                for k_step in fx.range_constexpr(qk_head_dim // 16):
                    score = mma_fragments(
                        load_score_fragment(
                            query_grad_output_shared_pointer,
                            query_shared_layout,
                            stage,
                            pair_query_rows * qk_head_dim,
                            k_step,
                        ),
                        make_b_fragment(k_packs[k_step]),
                        score,
                    )
                for k_step in fx.range_constexpr(value_head_dim // 16):
                    dp = mma_fragments(
                        load_score_fragment(
                            grad_output_shared_pointer,
                            grad_output_shared_layout,
                            stage,
                            pair_query_rows * value_head_dim,
                            k_step,
                        ),
                        make_b_fragment(v_packs[k_step]),
                        dp,
                    )

                stat_base = fx.Int32(stage * 2 * pair_query_rows)
                prefetch_all_metadata = pair_num_query_chunks <= 32
                prefetch_lse = pair_num_query_chunks >= 64
                first_lse_values = []
                first_delta_values = []
                if const_expr(prefetch_all_metadata or prefetch_lse):
                    for element in fx.range_constexpr(4):
                        query_local = fx.Int32(
                            fx.get_scalar(accumulator_coordinates[element])
                        )
                        first_lse_values.append(
                            fx.Float32(
                                fx.ptr_load(
                                    metadata_shared_pointer + stat_base + query_local
                                )
                            )
                        )
                        if const_expr(prefetch_all_metadata):
                            first_delta_values.append(
                                fx.Float32(
                                    fx.ptr_load(
                                        metadata_shared_pointer
                                        + stat_base
                                        + fx.Int32(pair_query_rows)
                                        + query_local
                                    )
                                )
                            )
                schedule_score_pipeline(
                    mfma_count=(qk_head_dim + value_head_dim) // 16,
                    dsrd_count=(qk_head_dim + value_head_dim) // 16
                    + (2 if prefetch_all_metadata else 1 if prefetch_lse else 0),
                    vmem_count=0,
                )

                score_values = fx.Vector(score)
                dp_values = fx.Vector(dp)
                first_probabilities = []
                first_ds_values = []
                for element in fx.range_constexpr(8):
                    query_local = fx.Int32(
                        fx.get_scalar(accumulator_coordinates[element])
                    )
                    query_position = (
                        query_chunk * fx.Int32(pair_query_rows) + query_local
                    )
                    if const_expr(
                        (prefetch_all_metadata or prefetch_lse) and element < 4
                    ):
                        lse_log2 = first_lse_values[element]
                    else:
                        lse_log2 = fx.Float32(
                            fx.ptr_load(
                                metadata_shared_pointer + stat_base + query_local
                            )
                        )
                    if const_expr(prefetch_all_metadata and element < 4):
                        delta_value = first_delta_values[element]
                    else:
                        delta_value = fx.Float32(
                            fx.ptr_load(
                                metadata_shared_pointer
                                + stat_base
                                + fx.Int32(pair_query_rows)
                                + query_local
                            )
                        )
                    probability = _exp2(
                        _f32(score_values[element]) * _f32(scale_log2) - lse_log2
                    )
                    if not isinstance(masked, bool):
                        raise AssertionError(
                            "generic mask evaluation requires a static mask flag"
                        )
                    if masked:
                        probability = evaluate_mask(query_position, key_pos).select(
                            probability, _f32(0.0)
                        )
                    first_probabilities.append(probability)
                    first_ds_values.append(
                        probability * (_f32(dp_values[element]) - delta_value)
                    )
                first_probability = (
                    fx.Vector.from_elements(first_probabilities, fx.Float32)
                    .to(fx.BFloat16)
                    .ir_value()
                )
                first_ds_fragment = make_b_fragment(
                    fx.Vector.from_elements(
                        first_ds_values,
                        fx.Float32,
                    ).to(fx.BFloat16)
                )
                first_update_fragments = []
                for d_chunk in fx.range_constexpr(qk_head_dim // 32):
                    first_update_fragments.append(
                        load_update_fragment(
                            query_grad_output_shared_pointer,
                            query_update_layout,
                            stage,
                            pair_query_rows * qk_head_dim,
                            0,
                            d_chunk,
                        )
                    )

                prefetch_second_metadata = pair_num_query_chunks >= 64
                second_lse_values = []
                second_delta_values = []
                if const_expr(prefetch_second_metadata):
                    for element in fx.range_constexpr(4):
                        score_index = 8 + element
                        query_local = fx.Int32(
                            fx.get_scalar(accumulator_coordinates[score_index])
                        )
                        second_lse_values.append(
                            fx.Float32(
                                fx.ptr_load(
                                    metadata_shared_pointer + stat_base + query_local
                                )
                            )
                        )
                        second_delta_values.append(
                            fx.Float32(
                                fx.ptr_load(
                                    metadata_shared_pointer
                                    + stat_base
                                    + fx.Int32(pair_query_rows)
                                    + query_local
                                )
                            )
                        )
                schedule_pack0_pipeline(
                    vmem_count=0,
                    exp_count=8,
                    dsrd_count=4 * (qk_head_dim // 32)
                    + (2 if prefetch_second_metadata else 0),
                )

                second_probabilities = []
                second_ds_values = []
                for element in fx.range_constexpr(8):
                    score_index = 8 + element
                    query_local = fx.Int32(
                        fx.get_scalar(accumulator_coordinates[score_index])
                    )
                    query_position = (
                        query_chunk * fx.Int32(pair_query_rows) + query_local
                    )
                    if const_expr(prefetch_second_metadata and element < 4):
                        lse_log2 = second_lse_values[element]
                        delta_value = second_delta_values[element]
                    else:
                        lse_log2 = fx.Float32(
                            fx.ptr_load(
                                metadata_shared_pointer + stat_base + query_local
                            )
                        )
                        delta_value = fx.Float32(
                            fx.ptr_load(
                                metadata_shared_pointer
                                + stat_base
                                + fx.Int32(pair_query_rows)
                                + query_local
                            )
                        )
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
                    second_probabilities.append(probability)
                    second_ds_values.append(
                        probability * (_f32(dp_values[score_index]) - delta_value)
                    )
                second_probability = (
                    fx.Vector.from_elements(second_probabilities, fx.Float32)
                    .to(fx.BFloat16)
                    .ir_value()
                )
                second_ds_fragment = make_b_fragment(
                    fx.Vector.from_elements(
                        second_ds_values,
                        fx.Float32,
                    ).to(fx.BFloat16)
                )
                store_probability(stage, first_probability, second_probability)
                fx.rocdl.sched_dswr(2)

                second_update_fragments = []
                for d_chunk in fx.range_constexpr(min(2, qk_head_dim // 32)):
                    second_update_fragments.append(
                        load_update_fragment(
                            query_grad_output_shared_pointer,
                            query_update_layout,
                            stage,
                            pair_query_rows * qk_head_dim,
                            1,
                            d_chunk,
                        )
                    )
                for d_chunk in fx.range_constexpr(qk_head_dim // 32):
                    fx.gemm(
                        tiled_mma,
                        grad_key_accumulators[d_chunk],
                        first_update_fragments[d_chunk],
                        first_ds_fragment,
                        grad_key_accumulators[d_chunk],
                    )
                schedule_pack1_pipeline(
                    mfma_count=qk_head_dim // 32,
                    exp_count=8,
                    dsrd_count=2 * (qk_head_dim // 32),
                )
                for d_chunk in fx.range_constexpr(qk_head_dim // 32):
                    fx.gemm(
                        tiled_mma,
                        grad_key_accumulators[d_chunk],
                        second_update_fragments[d_chunk],
                        second_ds_fragment,
                        grad_key_accumulators[d_chunk],
                    )
                    if const_expr(d_chunk + 2 < qk_head_dim // 32):
                        second_update_fragments.append(
                            load_update_fragment(
                                query_grad_output_shared_pointer,
                                query_update_layout,
                                stage,
                                pair_query_rows * qk_head_dim,
                                1,
                                d_chunk + 2,
                            )
                        )
                schedule_update_tail(
                    mfma_count=qk_head_dim // 32, dsrd_count=2 * (qk_head_dim // 32)
                )
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                fx.gpu.barrier()

            def run_producer(
                tile_count,
                first_tile,
                indices,
                list_row,
                indexed: fx.Constexpr[bool],
                mask_mode: fx.Constexpr[int],
            ):
                if tile_count > fx.Int32(0):
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    fx.gpu.barrier()
                    for tile_index in range(fx.Int32(0), tile_count, fx.Int32(1)):
                        query_chunk = qchunk_at(
                            tile_index, first_tile, indices, list_row, indexed
                        )
                        produce_tile(
                            query_chunk,
                            tile_is_masked(tile_index, mask_mode),
                            tile_index % fx.Int32(2),
                        )

            if const_expr(dense_mask):
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                fx.gpu.barrier()
                for qpair in range(
                    fx.Int32(0), fx.Int32(pair_num_query_chunks), fx.Int32(2)
                ):
                    produce_tile(qpair, False, 0)
                    produce_tile(qpair + fx.Int32(1), False, 1)
            else:
                list_nc = kv_chunk // fx.Int32(pair_key_split)
                list_row = list_nc * fx.Int32(num_query_chunks)
                partial_count = gload_i32(partial_counts_view, list_nc) * fx.Int32(
                    pair_list_query_split
                )
                full_count = gload_i32(full_counts_view, list_nc) * fx.Int32(
                    pair_list_query_split
                )
                run_producer(
                    partial_count,
                    fx.Int32(0),
                    partial_indices_view,
                    list_row,
                    True,
                    _MASK_ALL,
                )
                run_producer(
                    full_count,
                    fx.Int32(0),
                    full_indices_view,
                    list_row,
                    True,
                    _MASK_NONE,
                )

            output_fragment = fx.make_rmem_tensor(4, fx.BFloat16)
            output_row = fx.logical_divide(
                fx.slice(grad_key_view, (key_pos, None)), fx.make_layout(4, 1)
            )
            scale_vector = fx.Vector.filled(16, softmax_scale, fx.Float32)
            for d_chunk in fx.range_constexpr(qk_head_dim // 32):
                values_bf16 = (
                    fx.Vector(grad_key_accumulators[d_chunk].load()) * scale_vector
                ).to(fx.BFloat16)
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
        else:
            grad_value_accumulators = [
                fx.make_fragment_like(accumulator_coordinates, fx.Float32)
                for _ in fx.range_constexpr(value_head_dim // 32)
            ]
            for d_chunk in fx.range_constexpr(value_head_dim // 32):
                grad_value_accumulators[d_chunk].fill(0)

            def prefetch_dv_pack(stage, half: fx.Constexpr[int]):
                probability = make_b_fragment(load_probability(stage, half))
                update_fragments = []
                for d_chunk in fx.range_constexpr(value_head_dim // 32):
                    update_fragments.append(
                        load_update_fragment(
                            grad_output_shared_pointer,
                            grad_output_update_layout,
                            stage,
                            pair_query_rows * value_head_dim,
                            half,
                            d_chunk,
                        )
                    )
                if const_expr(half == 0):
                    fx.rocdl.sched_dsrd(1 + 2 * (value_head_dim // 32))
                    schedule_fence()
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                return probability, update_fragments

            def consume_dv_tile(
                first_probability,
                second_probability,
                first_update_fragments,
                second_update_fragments,
                overlap_q_metadata: fx.Constexpr[bool],
            ):
                for d_chunk in fx.range_constexpr(value_head_dim // 32):
                    fx.gemm(
                        tiled_mma,
                        grad_value_accumulators[d_chunk],
                        first_update_fragments[d_chunk],
                        first_probability,
                        grad_value_accumulators[d_chunk],
                    )
                second_read_count = 1 + 2 * (value_head_dim // 32)
                preload_count = min(3, second_read_count)
                remaining_reads = second_read_count - preload_count
                fx.rocdl.sched_dsrd(preload_count)
                for mfma_index in fx.range_constexpr(value_head_dim // 32):
                    fx.rocdl.sched_mfma(1)
                    reads = min(2, max(0, remaining_reads - 2 * mfma_index))
                    if const_expr(reads):
                        fx.rocdl.sched_dsrd(reads)
                schedule_fence()
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                for d_chunk in fx.range_constexpr(value_head_dim // 32):
                    fx.gemm(
                        tiled_mma,
                        grad_value_accumulators[d_chunk],
                        second_update_fragments[d_chunk],
                        second_probability,
                        grad_value_accumulators[d_chunk],
                    )
                next_vmem_count = (
                    pair_query_load_iterations + 2 if overlap_q_metadata else 0
                )
                for mfma_index in fx.range_constexpr(value_head_dim // 32):
                    if const_expr(mfma_index < next_vmem_count):
                        fx.rocdl.sched_vmem(1)
                    fx.rocdl.sched_mfma(1)
                for _ in fx.range_constexpr(
                    max(0, next_vmem_count - value_head_dim // 32)
                ):
                    fx.rocdl.sched_vmem(1)
                schedule_fence()

            def run_consumer(
                tile_count,
                first_tile,
                indices,
                list_row,
                indexed: fx.Constexpr[bool],
            ):
                if tile_count > fx.Int32(0):
                    stage_qdo_metadata(
                        qchunk_at(fx.Int32(0), first_tile, indices, list_row, indexed),
                        0,
                    )
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    fx.gpu.barrier()
                    for tile_index in range(fx.Int32(0), tile_count, fx.Int32(1)):
                        if tile_index == fx.Int32(0):
                            if tile_count > fx.Int32(1):
                                stage_qdo_metadata(
                                    qchunk_at(
                                        fx.Int32(1),
                                        first_tile,
                                        indices,
                                        list_row,
                                        indexed,
                                    ),
                                    1,
                                )
                        else:
                            previous_stage = (tile_index - fx.Int32(1)) % fx.Int32(2)
                            first_probability, first_updates = prefetch_dv_pack(
                                previous_stage, 0
                            )
                            second_probability, second_updates = prefetch_dv_pack(
                                previous_stage, 1
                            )
                            has_next = tile_index + fx.Int32(1) < tile_count
                            if has_next:
                                next_qchunk = qchunk_at(
                                    tile_index + fx.Int32(1),
                                    first_tile,
                                    indices,
                                    list_row,
                                    indexed,
                                )
                                stage_q_metadata(next_qchunk, previous_stage)
                                consume_dv_tile(
                                    first_probability,
                                    second_probability,
                                    first_updates,
                                    second_updates,
                                    True,
                                )
                                stage_do(next_qchunk, previous_stage)
                                fx.rocdl.sched_vmem(pair_grad_output_load_iterations)
                                schedule_fence()
                            else:
                                consume_dv_tile(
                                    first_probability,
                                    second_probability,
                                    first_updates,
                                    second_updates,
                                    False,
                                )
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        fx.gpu.barrier()
                    last_stage = (tile_count - fx.Int32(1)) % fx.Int32(2)
                    first_probability, first_updates = prefetch_dv_pack(last_stage, 0)
                    second_probability, second_updates = prefetch_dv_pack(last_stage, 1)
                    consume_dv_tile(
                        first_probability,
                        second_probability,
                        first_updates,
                        second_updates,
                        False,
                    )

            if const_expr(dense_mask):
                run_consumer(
                    fx.Int32(pair_num_query_chunks),
                    fx.Int32(0),
                    partial_indices_view,
                    fx.Int32(0),
                    False,
                )
            else:
                list_nc = kv_chunk // fx.Int32(pair_key_split)
                list_row = list_nc * fx.Int32(num_query_chunks)
                partial_count = gload_i32(partial_counts_view, list_nc) * fx.Int32(
                    pair_list_query_split
                )
                full_count = gload_i32(full_counts_view, list_nc) * fx.Int32(
                    pair_list_query_split
                )
                run_consumer(
                    partial_count, fx.Int32(0), partial_indices_view, list_row, True
                )
                run_consumer(full_count, fx.Int32(0), full_indices_view, list_row, True)

            output_fragment = fx.make_rmem_tensor(4, fx.BFloat16)
            output_row = fx.logical_divide(
                fx.slice(grad_value_view, (key_pos, None)), fx.make_layout(4, 1)
            )
            for d_chunk in fx.range_constexpr(value_head_dim // 32):
                values_bf16 = fx.Vector(grad_value_accumulators[d_chunk].load()).to(
                    fx.BFloat16
                )
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

    return _emit_dkdv_mfma32_body
