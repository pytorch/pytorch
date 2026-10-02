"""Producer waves overlap current QK/dP/dK/dV with two prior-dQ waves."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from .flex_attn_utils import (
    make_dq_workspace_layout,
    make_global_view,
    make_mask_buffers,
    make_mask_evaluator,
    make_value_shared_layout,
)


def make_pipeline_kernel(context, f32, exp2):
    batch_size = context["batch_size"]
    num_heads = context["num_heads"]
    sequence_length = context["sequence_length"]
    qk_dim = context["qk_head_dim"]
    v_dim = context["value_head_dim"]
    num_waves = context["num_waves"]
    heads_are_inner = context["heads_are_inner"]
    q_stride = context["q_stride"]
    k_stride = context["k_stride"]
    v_stride = context["v_stride"]
    do_stride = context["grad_output_stride"]
    dq_stride = (num_heads * sequence_length * qk_dim, sequence_length * qk_dim, qk_dim, 1)
    dk_stride = context["grad_key_stride"]
    dv_stride = context["grad_value_stride"]
    lse_in_log2 = context["lse_in_log2"]
    scale = context["softmax_scale"]
    scale_log2 = context["scale_log2"]
    unmasked = context["unmasked_traversal"]
    direct_range = context["direct_range_traversal"]
    block_list = context["block_list_traversal"]
    metadata_chunks = context["metadata_chunks"]
    query_split = context["sparse_q_block_size"] // 16
    mask_buffer_count = context["mask_buffer_count"]
    mask_buffer_sizes = context["mask_buffer_sizes"]
    mask_buffer_strides = context["mask_buffer_strides"]
    mask_program = context["mask_program"]
    mask_output = context["mask_program_output"]
    threads = num_waves * 64
    key_groups = context["key_groups"]
    producer_waves = context["producer_waves"]
    consumer_waves = 2
    key_rows = producer_waves * key_groups * 16
    dq_phase_tiles = qk_dim // (16 * consumer_waves * 2)
    if qk_dim % 64:
        raise ValueError(f"QK head dimension must be divisible by 64, got {qk_dim}")

    @fx.struct
    class SharedMemory:
        key: fx.Array[fx.BFloat16, key_rows * qk_dim, 16]
        value: fx.Array[fx.BFloat16, key_rows * v_dim, 16]
        query: fx.Array[fx.BFloat16, 2 * 16 * qk_dim, 16]
        grad_output: fx.Array[fx.BFloat16, 2 * 16 * v_dim, 16]
        ds: fx.Array[fx.BFloat16, 2 * key_rows * 16, 16]
        probability: fx.Array[fx.BFloat16, key_rows * 16, 16]
        stats: fx.Array[fx.Float32, 2 * 32, 16]

    @flyc.kernel
    def compute_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        logsumexp: fx.Tensor,
        delta: fx.Tensor,
        grad_output: fx.Tensor,
        grad_query: fx.Tensor,
        grad_key: fx.Tensor,
        grad_value: fx.Tensor,
        partial_kv_counts: fx.Tensor,
        partial_kv_indices: fx.Tensor,
        full_kv_counts: fx.Tensor,
        full_kv_indices_transposed: fx.Tensor,
        mask_buffer_0: fx.Tensor,
        mask_buffer_1: fx.Tensor,
        mask_buffer_2: fx.Tensor,
        mask_buffer_3: fx.Tensor,
    ):
        if const_expr(heads_are_inner):
            batch_head = fx.Int32(fx.block_idx.x)
            owner = fx.Int32(fx.block_idx.y)
        else:
            owner = fx.Int32(fx.block_idx.x)
            batch_head = fx.Int32(fx.block_idx.y)
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % fx.Int32(64)
        wave = fx.Int32(fx.gpu.shuffle_idx(tid // fx.Int32(64), 0, 64))
        batch = batch_head // fx.Int32(num_heads)
        head = batch_head % fx.Int32(num_heads)
        head_coord = (fx.Int64(batch), fx.Int64(head), None, None)
        qview = make_global_view(query, head_coord, (batch_size, num_heads, sequence_length, qk_dim), q_stride)
        kview = make_global_view(key, head_coord, (batch_size, num_heads, sequence_length, qk_dim), k_stride)
        vview = make_global_view(value, head_coord, (batch_size, num_heads, sequence_length, v_dim), v_stride)
        doview = make_global_view(grad_output, head_coord, (batch_size, num_heads, sequence_length, v_dim), do_stride)
        dqview = make_global_view(grad_query, head_coord, (batch_size, num_heads, sequence_length, qk_dim), dq_stride)
        dkview = make_global_view(grad_key, head_coord, (batch_size, num_heads, sequence_length, qk_dim), dk_stride)
        dvview = make_global_view(grad_value, head_coord, (batch_size, num_heads, sequence_length, v_dim), dv_stride)
        lseview = make_global_view(
            logsumexp, (fx.Int64(batch_head), None), (batch_size * num_heads, sequence_length), (sequence_length, 1)
        )
        deltaview = make_global_view(
            delta, (fx.Int64(batch_head), None), (batch_size * num_heads, sequence_length), (sequence_length, 1)
        )

        def load_i32(view, index):
            return fx.Int32(fx.get_iter(view)[index])

        masks = make_mask_buffers(
            make_global_view,
            mask_buffer_count,
            mask_buffer_sizes,
            mask_buffer_0,
            mask_buffer_1,
            mask_buffer_2,
            mask_buffer_3,
        )
        evaluate_mask = make_mask_evaluator(
            mask_program, mask_output, mask_buffer_strides, masks, load_i32, batch, head
        )
        shared = fx.SharedAllocator().allocate(SharedMemory).peek()
        g64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        read128 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        tr16 = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(16, 64), fx.BFloat16)
        atomic = fx.make_copy_atom(fx.rocdl.BufferAtomicPkAdd(fx.BFloat16), fx.BFloat16)
        score_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16)), fx.make_layout((1, 1, 1), (1, 1, 1))
        )
        wide_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16)), fx.make_layout((1, 1, 1), (1, 1, 1))
        )
        wide_thread = wide_mma.get_slice(lane)
        wide_dim_coords = wide_thread.partition_C(fx.make_view(0, fx.make_layout((32, 32), (1, 0))))
        wide_row_coords = wide_thread.partition_C(fx.make_view(0, fx.make_layout((32, 32), (0, 1))))
        wide_a_copy = fx.make_tiled_copy_A(tr16, wide_mma).get_slice(lane)
        score_thread = score_mma.get_slice(lane)
        ccoords = score_thread.partition_C(fx.make_view(0, fx.make_layout((16, 16), (1, 0))))
        rowcoords = score_thread.partition_C(fx.make_view(0, fx.make_layout((16, 16), (0, 1))))
        dq_key_copy = fx.make_tiled_copy_A(tr16, score_mma).get_slice(lane)
        dq_ds_copy = fx.make_tiled_copy_B(tr16, score_mma).get_slice(lane)
        score_copy = fx.make_tiled_copy_A(read128, score_mma).get_slice(lane)
        score_b_copy = fx.make_tiled_copy_B(read128, score_mma).get_slice(lane)
        lds_copy = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.BFloat16)
        stats_copy = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Float32)
        write64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        score_store = fx.make_tiled_copy_C(write64, score_mma).get_slice(lane)
        dq_store = fx.make_tiled_copy_C(atomic, score_mma).get_slice(lane)
        dq_tiles = fx.flat_divide(
            fx.make_view(fx.get_iter(dqview), fx.select(make_dq_workspace_layout(sequence_length, qk_dim), [1, 0])),
            (16, 16),
        )
        wide_b_copy = fx.make_tiled_copy_B(read128, wide_mma).get_slice(lane)
        key_base = owner * fx.Int32(key_rows)
        key_pos = key_base + wave * fx.Int32(key_groups * 16) + fx.Int32(fx.get_scalar(rowcoords[0]))
        klayout = make_value_shared_layout(key_rows, qk_dim)
        vlayout = make_value_shared_layout(key_rows, v_dim)
        qlayout = make_value_shared_layout(16, qk_dim)
        dolayout = make_value_shared_layout(16, v_dim)
        score_layout = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(2, 3, 3)),
            fx.make_layout((16, key_rows), (1, 16)),
        )

        def stage_operand(view, layout, pointer, first_row, rows, dimension, loader_threads):
            # Distribute physical LDS packs; invert the swizzled layout to
            # recover their logical global coordinates, as in forward.
            coordinates = fx.make_composed_layout(
                fx.right_inverse(layout.outer),
                fx.make_composed_layout(layout.inner, fx.make_layout(rows * dimension, 1)),
            )
            destinations = fx.logical_divide(
                fx.make_view(pointer, fx.make_layout(rows * dimension, 1)),
                fx.make_layout(8, 1),
            )
            for step in fx.range_constexpr((rows * dimension // 8 + loader_threads - 1) // loader_threads):
                linear = tid + fx.Int32(step * loader_threads)
                if linear < fx.Int32(rows * dimension // 8):
                    logical = fx.Int32(fx.get_scalar(fx.crd2idx(linear * fx.Int32(8), coordinates)))
                    row = logical % fx.Int32(rows)
                    column = logical // fx.Int32(rows * 8)
                    destination = fx.slice(destinations, (None, linear))
                    if first_row + row < fx.Int32(sequence_length):
                        source = fx.logical_divide(fx.slice(view, (first_row + row, None)), fx.make_layout(8, 1))
                        fx.copy(lds_copy, fx.slice(source, (None, column)), destination)
                    else:
                        zeros = fx.make_rmem_tensor(8, fx.BFloat16)
                        zeros.fill(0)
                        fx.copy(read128, zeros, destination)

        def stage_query(query_tile, stage):
            first_row = query_tile * fx.Int32(16)
            if tid < fx.Int32(4):
                source_index = query_tile * fx.Int32(4) + tid
                for source_view, stats_offset in [(lseview, 0), (deltaview, 16)]:
                    source = fx.logical_divide(source_view, fx.make_layout(4, 1))
                    destination = fx.logical_divide(
                        fx.make_view(
                            shared.stats.ptr + stage * fx.Int32(32) + fx.Int32(stats_offset), fx.make_layout(16, 1)
                        ),
                        fx.make_layout(4, 1),
                    )
                    fx.copy(stats_copy, fx.slice(source, (None, source_index)), fx.slice(destination, (None, tid)))
            stage_operand(
                qview,
                qlayout,
                shared.query.ptr + stage * fx.Int32(16 * qk_dim),
                first_row,
                16,
                qk_dim,
                producer_waves * 64,
            )
            stage_operand(
                doview,
                dolayout,
                shared.grad_output.ptr + stage * fx.Int32(16 * v_dim),
                first_row,
                16,
                v_dim,
                producer_waves * 64,
            )

        stage_operand(kview, klayout, shared.key.ptr, key_base, key_rows, qk_dim, threads)
        stage_operand(vview, vlayout, shared.value.ptr, key_base, key_rows, v_dim, threads)

        def load_score(pointer, layout, step):
            tiles = fx.flat_divide(fx.make_view(pointer, layout), (16, 32))
            tile = fx.slice(tiles, (None, None, 0, step))
            fragment = score_thread.make_fragment_A(tile)
            fx.copy(read128, score_copy.partition_S(tile), score_copy.retile(fragment))
            return fragment

        def load_score_operand(pointer, layout, step, group):
            tiles = fx.flat_divide(fx.make_view(pointer, layout), (16, 32))
            row_tile = wave * fx.Int32(key_groups) + fx.Int32(group)
            tile = fx.slice(tiles, (None, None, row_tile, step))
            fragment = score_thread.make_fragment_B(tile)
            fx.copy(read128, score_b_copy.partition_S(tile), score_b_copy.retile(fragment))
            return fragment

        def load_wide(pointer, layout, dim_chunk):
            tiles = fx.flat_divide(fx.make_view(pointer, fx.select(layout, [1, 0])), (32, 16))
            tile = fx.slice(tiles, (None, None, dim_chunk, 0))
            fragment = wide_thread.make_fragment_A(tile)
            fx.copy(tr16, wide_a_copy.partition_S(tile), wide_a_copy.retile(fragment))
            return fragment

        def store_score(pointer, group, values):
            tiles = fx.flat_divide(fx.make_view(pointer, score_layout), (16, 16))
            tile = fx.slice(tiles, (None, None, 0, wave * fx.Int32(key_groups) + fx.Int32(group)))
            fragment = fx.make_fragment_like(score_thread.partition_C(tile), fx.BFloat16)
            fragment.store(values.ir_value())
            fx.copy(write64, score_store.retile(fragment), score_store.partition_D(tile))

        def load_wide_score(pointer):
            tiles = fx.flat_divide(fx.make_view(pointer, fx.select(score_layout, [1, 0])), (32, 16))
            tile = fx.slice(tiles, (None, None, wave, 0))
            fragment = wide_thread.make_fragment_B(tile)
            fx.copy(read128, wide_b_copy.partition_S(tile), wide_b_copy.retile(fragment))
            return fragment

        def load_dq_operands(stage, step, phase: fx.Constexpr[int], cached_keys):
            ds_pointer = shared.ds.ptr + stage * fx.Int32(key_rows * 16)
            ds_tiles = fx.flat_divide(fx.make_view(ds_pointer, score_layout), (16, 32))
            ds_tile = fx.slice(ds_tiles, (None, None, 0, step))
            ds_fragment = score_thread.make_fragment_B(ds_tile)
            fx.copy(tr16, dq_ds_copy.partition_S(ds_tile), dq_ds_copy.retile(ds_fragment))
            key_fragments = cached_keys[phase][step]
            return ds_fragment, key_fragments

        def store_dq(query_tile, accumulator, tile):
            values = (fx.Vector(accumulator.load()) * fx.Vector.filled(4, scale, fx.Float32)).to(fx.BFloat16)
            destination = fx.slice(dq_tiles, (None, None, tile, query_tile))
            fragment = fx.make_fragment_like(score_thread.partition_C(destination), fx.BFloat16)
            fragment.store(values.ir_value())
            fx.copy(atomic, dq_store.retile(fragment), dq_store.partition_D(destination))

        def consume(query_tile, stage, cached_keys):
            # Two phases keep dQ state small while producer accumulators live.
            for phase in fx.range_constexpr(2):
                accumulators = [fx.make_fragment_like(ccoords, fx.Float32) for _ in fx.range_constexpr(dq_phase_tiles)]
                for d in fx.range_constexpr(dq_phase_tiles):
                    accumulators[d].fill(0)
                operands = load_dq_operands(stage, 0, phase, cached_keys)
                for step in fx.range_constexpr(key_rows // 32):
                    if const_expr(step + 1 < key_rows // 32):
                        next_operands = load_dq_operands(stage, step + 1, phase, cached_keys)
                    for d in fx.range_constexpr(dq_phase_tiles):
                        fx.gemm(score_mma, accumulators[d], operands[1][d], operands[0], accumulators[d])
                    if const_expr(step + 1 < key_rows // 32):
                        operands = next_operands
                for d in fx.range_constexpr(dq_phase_tiles):
                    tile = (wave - fx.Int32(producer_waves)) * fx.Int32(2 * dq_phase_tiles) + fx.Int32(
                        phase * dq_phase_tiles + d
                    )
                    store_dq(query_tile, accumulators[d], tile)

        def finish_stage(producer: fx.Constexpr[bool]):
            # One common barrier publishes dS(i) and Q/dO(i+1); all waves attend.
            if const_expr(producer):
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            else:
                fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()

        def produce(query_tile, stage, masked: fx.Constexpr[bool], dk_wide, dv_wide):
            qpointer = shared.query.ptr + stage * fx.Int32(16 * qk_dim)
            dopointer = shared.grad_output.ptr + stage * fx.Int32(16 * v_dim)
            scores = [fx.make_fragment_like(ccoords, fx.Float32) for _ in fx.range_constexpr(key_groups)]
            probabilities = [fx.make_fragment_like(ccoords, fx.Float32) for _ in fx.range_constexpr(key_groups)]
            for group in fx.range_constexpr(key_groups):
                scores[group].fill(0)
            keys = [load_score_operand(shared.key.ptr, klayout, 0, group) for group in fx.range_constexpr(key_groups)]
            query_operand = load_score(qpointer, qlayout, 0)
            for step in fx.range_constexpr(qk_dim // 32):
                if const_expr(step + 1 < qk_dim // 32):
                    next_keys = [
                        load_score_operand(shared.key.ptr, klayout, step + 1, group)
                        for group in fx.range_constexpr(key_groups)
                    ]
                    next_query = load_score(qpointer, qlayout, step + 1)
                for group in fx.range_constexpr(key_groups):
                    fx.gemm(score_mma, scores[group], query_operand, keys[group], scores[group])
                if const_expr(step + 1 < qk_dim // 32):
                    keys = next_keys
                    query_operand = next_query
            values_operand = [
                load_score_operand(shared.value.ptr, vlayout, 0, group) for group in fx.range_constexpr(key_groups)
            ]
            grad_output_operand = load_score(dopointer, dolayout, 0)
            for group in fx.range_constexpr(key_groups):
                score_values = fx.Vector(scores[group].load())
                values = []
                for element in fx.range_constexpr(4):
                    local_row = fx.Int32(fx.get_scalar(ccoords[element]))
                    position = query_tile * fx.Int32(16) + local_row
                    lse = fx.Float32(shared.stats.ptr[stage * fx.Int32(32) + local_row])
                    if const_expr(not lse_in_log2):
                        lse = lse * f32(1.4426950408889634)
                    probability = exp2(f32(fx.math.fma(f32(score_values[element]), f32(scale_log2), -lse)))
                    if masked:
                        probability = evaluate_mask(position, key_pos + fx.Int32(group * 16)).select(
                            probability, f32(0.0)
                        )
                    if const_expr(not direct_range):
                        probability = (key_pos + fx.Int32(group * 16) < fx.Int32(sequence_length)).select(
                            probability, f32(0.0)
                        )
                    values.append(probability)
                probabilities[group].store(fx.Vector.from_elements(values, fx.Float32).ir_value())
            dps = [fx.make_fragment_like(ccoords, fx.Float32) for _ in fx.range_constexpr(key_groups)]
            for group in fx.range_constexpr(key_groups):
                dps[group].fill(0)
            for step in fx.range_constexpr(v_dim // 32):
                if const_expr(step + 1 < v_dim // 32):
                    next_values = [
                        load_score_operand(shared.value.ptr, vlayout, step + 1, group)
                        for group in fx.range_constexpr(key_groups)
                    ]
                    next_grad_output = load_score(dopointer, dolayout, step + 1)
                for group in fx.range_constexpr(key_groups):
                    fx.gemm(score_mma, dps[group], grad_output_operand, values_operand[group], dps[group])
                if const_expr(step + 1 < v_dim // 32):
                    values_operand = next_values
                    grad_output_operand = next_grad_output
            for group in fx.range_constexpr(key_groups):
                probability_values = fx.Vector(probabilities[group].load())
                dp_values = fx.Vector(dps[group].load())
                ds_values = []
                for element in fx.range_constexpr(4):
                    local_row = fx.Int32(fx.get_scalar(ccoords[element]))
                    delta_value = fx.Float32(shared.stats.ptr[stage * fx.Int32(32) + fx.Int32(16) + local_row])
                    ds_values.append(f32(probability_values[element]) * (f32(dp_values[element]) - delta_value))
                store_score(shared.probability.ptr, group, probability_values.to(fx.BFloat16))
                ds_vector = fx.Vector.from_elements(ds_values, fx.Float32).to(fx.BFloat16)
                store_score(shared.ds.ptr + stage * fx.Int32(key_rows * 16), group, ds_vector)
            # Each wave converts its own 32-key tile through LDS. No other
            # wave reads probability; dS is published by finish_stage.
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            ds_wide = load_wide_score(shared.ds.ptr + stage * fx.Int32(key_rows * 16))
            p_wide = load_wide_score(shared.probability.ptr)
            qa = load_wide(qpointer, qlayout, 0)
            for d in fx.range_constexpr(qk_dim // 32):
                if const_expr(d + 1 < qk_dim // 32):
                    next_qa = load_wide(qpointer, qlayout, d + 1)
                if const_expr(d < v_dim // 32):
                    doa = load_wide(dopointer, dolayout, d)
                fx.gemm(wide_mma, dk_wide[d], qa, ds_wide, dk_wide[d])
                if const_expr(d < v_dim // 32):
                    fx.gemm(wide_mma, dv_wide[d], doa, p_wide, dv_wide[d])
                if const_expr(d + 1 < qk_dim // 32):
                    qa = next_qa

        def run(
            count,
            first_query,
            query_indices,
            masked: fx.Constexpr[bool],
            producer: fx.Constexpr[bool],
            dk_wide,
            dv_wide,
            cached_keys,
        ):
            def query_at(index):
                if const_expr(block_list):
                    list_offset = (fx.Int64(batch_head) * fx.Int64(metadata_chunks) + fx.Int64(owner)) * fx.Int64(
                        metadata_chunks
                    )
                    sparse_query = fx.Int32(
                        fx.get_iter(query_indices)[list_offset + fx.Int64(index // fx.Int32(query_split))]
                    )
                    return sparse_query * fx.Int32(query_split) + index % fx.Int32(query_split)
                else:
                    return first_query + index

            if count > fx.Int32(0):
                if const_expr(producer):
                    stage_query(query_at(fx.Int32(0)), fx.Int32(0))
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                fx.gpu.barrier()
                if const_expr(producer):
                    if count > fx.Int32(1):
                        stage_query(query_at(fx.Int32(1)), fx.Int32(1))
                    produce(query_at(fx.Int32(0)), fx.Int32(0), masked, dk_wide, dv_wide)
                finish_stage(producer)
                for index in range(fx.Int32(1), count, fx.Int32(1)):
                    stage = index % fx.Int32(2)
                    if const_expr(producer):
                        if index + fx.Int32(1) < count:
                            stage_query(query_at(index + fx.Int32(1)), fx.Int32(1) - stage)
                        produce(query_at(index), stage, masked, dk_wide, dv_wide)
                    else:
                        consume(query_at(index - fx.Int32(1)), fx.Int32(1) - stage, cached_keys)
                    finish_stage(producer)
                if const_expr(not producer):
                    consume(query_at(count - fx.Int32(1)), (count - fx.Int32(1)) % fx.Int32(2), cached_keys)
                # A subsequent masked/full list may reuse either dS stage.
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()

        def execute_runs(producer: fx.Constexpr[bool], dk_wide, dv_wide, cached_keys):
            if const_expr(unmasked):
                run(
                    fx.Int32(sequence_length // 16),
                    fx.Int32(0),
                    partial_kv_indices,
                    False,
                    producer,
                    dk_wide,
                    dv_wide,
                    cached_keys,
                )
            elif const_expr(direct_range):
                first_query = key_base // fx.Int32(16)
                masked_count = fx.min(fx.Int32(key_rows // 16), fx.Int32(sequence_length // 16) - first_query)
                run(masked_count, first_query, partial_kv_indices, True, producer, dk_wide, dv_wide, cached_keys)
                run(
                    fx.Int32(sequence_length // 16) - first_query - masked_count,
                    first_query + masked_count,
                    full_kv_indices_transposed,
                    False,
                    producer,
                    dk_wide,
                    dv_wide,
                    cached_keys,
                )

            else:
                count_index = fx.Int64(batch_head) * fx.Int64(metadata_chunks) + fx.Int64(owner)
                partial_count = fx.Int32(fx.get_iter(partial_kv_counts)[count_index])
                full_count = fx.Int32(fx.get_iter(full_kv_counts)[count_index])
                run(
                    partial_count * fx.Int32(query_split),
                    fx.Int32(0),
                    partial_kv_indices,
                    True,
                    producer,
                    dk_wide,
                    dv_wide,
                    cached_keys,
                )
                run(
                    full_count * fx.Int32(query_split),
                    fx.Int32(0),
                    full_kv_indices_transposed,
                    False,
                    producer,
                    dk_wide,
                    dv_wide,
                    cached_keys,
                )

        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        fx.gpu.barrier()

        def cache_consumer_keys():
            key_tiles = fx.flat_divide(fx.make_view(shared.key.ptr, fx.select(klayout, [1, 0])), (16, 32))
            cached_keys = []
            for phase in fx.range_constexpr(2):
                phase_keys = []
                for step in fx.range_constexpr(key_rows // 32):
                    key_fragments = []
                    for d in fx.range_constexpr(dq_phase_tiles):
                        dimension_tile = (wave - fx.Int32(producer_waves)) * fx.Int32(2 * dq_phase_tiles) + fx.Int32(
                            phase * dq_phase_tiles + d
                        )
                        key_tile = fx.slice(key_tiles, (None, None, dimension_tile, step))
                        fragment = score_thread.make_fragment_A(key_tile)
                        fx.copy(tr16, dq_key_copy.partition_S(key_tile), dq_key_copy.retile(fragment))
                        key_fragments.append(fragment)
                    phase_keys.append(key_fragments)
                cached_keys.append(phase_keys)
            return cached_keys

        if wave < fx.Int32(producer_waves):
            dk_wide = [fx.make_fragment_like(wide_dim_coords, fx.Float32) for _ in fx.range_constexpr(qk_dim // 32)]
            dv_wide = [fx.make_fragment_like(wide_dim_coords, fx.Float32) for _ in fx.range_constexpr(v_dim // 32)]
            for d in fx.range_constexpr(qk_dim // 32):
                dk_wide[d].fill(0)
            for d in fx.range_constexpr(v_dim // 32):
                dv_wide[d].fill(0)
            execute_runs(True, dk_wide, dv_wide, [])
            wide_key_pos = key_base + wave * fx.Int32(key_groups * 16) + fx.Int32(fx.get_scalar(wide_row_coords[0]))
            if wide_key_pos < fx.Int32(sequence_length):
                dkrow = fx.logical_divide(fx.slice(dkview, (wide_key_pos, None)), fx.make_layout(4, 1))
                dvrow = fx.logical_divide(fx.slice(dvview, (wide_key_pos, None)), fx.make_layout(4, 1))
                for d in fx.range_constexpr(qk_dim // 32):
                    values = (fx.Vector(dk_wide[d].load()) * fx.Vector.filled(16, scale, fx.Float32)).to(fx.BFloat16)
                    for part in fx.range_constexpr(4):
                        output = fx.make_rmem_tensor(4, fx.BFloat16)
                        output.store(
                            fx.Vector.from_elements(
                                [values[part * 4 + e] for e in fx.range_constexpr(4)], fx.BFloat16
                            ).ir_value()
                        )
                        column = fx.Int32(d * 32) + fx.Int32(fx.get_scalar(wide_dim_coords[part * 4]))
                        fx.copy(g64, output, fx.slice(dkrow, (None, column // fx.Int32(4))))
                for d in fx.range_constexpr(v_dim // 32):
                    values = fx.Vector(dv_wide[d].load()).to(fx.BFloat16)
                    for part in fx.range_constexpr(4):
                        output = fx.make_rmem_tensor(4, fx.BFloat16)
                        output.store(
                            fx.Vector.from_elements(
                                [values[part * 4 + e] for e in fx.range_constexpr(4)], fx.BFloat16
                            ).ir_value()
                        )
                        column = fx.Int32(d * 32) + fx.Int32(fx.get_scalar(wide_dim_coords[part * 4]))
                        fx.copy(g64, output, fx.slice(dvrow, (None, column // fx.Int32(4))))

        else:
            cached_keys = cache_consumer_keys()
            execute_runs(False, [], [], cached_keys)

    return compute_kernel
