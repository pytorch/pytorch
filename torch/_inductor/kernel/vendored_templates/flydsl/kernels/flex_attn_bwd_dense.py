"""Six producer waves overlap current QK/dP/dK/dV with two prior-dQ waves."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr

from .flex_attn_utils import (
    make_global_view,
    make_mask_buffers,
    make_mask_evaluator,
    make_value_shared_layout,
)


def make_dense_kernel(context, f32, exp2):
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
        wave = fx.Int32(fx.rocdl.readfirstlane(fx.Int32.ir_type, (tid // fx.Int32(64)).ir_value()))
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
        lseview = make_global_view(logsumexp, (fx.Int64(batch_head), None), (batch_size * num_heads, sequence_length), (sequence_length, 1))
        deltaview = make_global_view(delta, (fx.Int64(batch_head), None), (batch_size * num_heads, sequence_length), (sequence_length, 1))

        def load_i32(view, index):
            return fx.Int32(fx.get_iter(view)[index])

        masks = make_mask_buffers(make_global_view, mask_buffer_count, mask_buffer_sizes, mask_buffer_0, mask_buffer_1, mask_buffer_2, mask_buffer_3)
        evaluate_mask = make_mask_evaluator(mask_program, mask_output, mask_buffer_strides, masks, load_i32, batch, head)
        shared = fx.SharedAllocator().allocate(SharedMemory).peek()
        g64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        read128 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        tr16 = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(16, 64), fx.BFloat16)
        atomic = fx.make_copy_atom(fx.rocdl.BufferAtomicPkAdd(fx.BFloat16), fx.BFloat16)
        score_mma = fx.make_tiled_mma(fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16)), fx.make_layout((1, 1, 1), (1, 1, 1)))
        update_mma = fx.make_tiled_mma(fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)), fx.make_layout((1, 1, 1), (1, 1, 1)))
        wide_mma = fx.make_tiled_mma(fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16)), fx.make_layout((1, 1, 1), (1, 1, 1)))
        wide_thread = wide_mma.get_slice(lane)
        wide_dim_coords = wide_thread.partition_C(fx.make_view(0, fx.make_layout((32, 32), (1, 0))))
        wide_row_coords = wide_thread.partition_C(fx.make_view(0, fx.make_layout((32, 32), (0, 1))))
        wide_a_copy = fx.make_tiled_copy_A(tr16, wide_mma).get_slice(lane)
        score_thread = score_mma.get_slice(lane)
        update_thread = update_mma.get_slice(lane)
        ccoords = score_thread.partition_C(fx.make_view(0, fx.make_layout((16, 16), (1, 0))))
        rowcoords = score_thread.partition_C(fx.make_view(0, fx.make_layout((16, 16), (0, 1))))
        update_bcoords = update_thread.partition_B(fx.make_view(0, fx.make_layout((16, 16), (16, 1))))
        dq_key_copy = fx.make_tiled_copy_A(tr16, score_mma).get_slice(lane)
        dq_ds_copy = fx.make_tiled_copy_B(tr16, score_mma).get_slice(lane)
        score_copy = fx.make_tiled_copy_A(read128, score_mma).get_slice(lane)
        score_b_copy = fx.make_tiled_copy_B(read128, score_mma).get_slice(lane)
        key_base = owner * fx.Int32(key_rows)
        key_pos = key_base + wave * fx.Int32(key_groups * 16) + fx.Int32(fx.get_scalar(rowcoords[0]))
        klayout = make_value_shared_layout(key_rows, qk_dim)
        vlayout = make_value_shared_layout(key_rows, v_dim)
        qlayout = make_value_shared_layout(16, qk_dim)
        dolayout = make_value_shared_layout(16, v_dim)

        def stage_operand(view, layout, pointer, first_row, rows, dimension, row_stride):
            # The public BufferCopyLDS atom is synchronous with respect to the
            # compiler's VMEM wait tracking. Asyncmark-native DMA preserves
            # outstanding prior-dQ atomics at the cross-iteration boundary.
            base_layout = fx.make_layout(
                ((16, rows // 16), (32, dimension // 32)),
                ((32, 16 * dimension), (1, 16 * 32)),
            )
            coordinates = fx.right_inverse(base_layout)
            for step in fx.range_constexpr((rows * dimension // 8 + threads - 1) // threads):
                linear = tid + fx.Int32(step * threads)
                if linear < fx.Int32(rows * dimension // 8):
                    physical = linear * fx.Int32(8)
                    unswizzled = physical ^ (physical // fx.Int32(256) % fx.Int32(2) * fx.Int32(16))
                    logical = fx.Int32(fx.get_scalar(fx.crd2idx(unswizzled, coordinates)))
                    row = logical % fx.Int32(rows)
                    column = logical // fx.Int32(rows * 8)
                    if const_expr(rows == 16):
                        fx.rocdl.raw_ptr_buffer_load_async_lds(
                            fx.rocdl.get_buffer_rsrc(fx.get_iter(view)),
                            fx.to_llvm_ptr(pointer + (linear - lane) * fx.Int32(8)),
                            fx.Int32(16).ir_value(),
                            fx.Int32((first_row + row) * fx.Int32(row_stride * 2) + column * fx.Int32(16)).ir_value(),
                            fx.Int32(0).ir_value(), fx.Int32(0).ir_value(),
                        )
                    elif first_row + row < fx.Int32(sequence_length):
                        fx.rocdl.raw_ptr_buffer_load_async_lds(
                            fx.rocdl.get_buffer_rsrc(fx.get_iter(view)),
                            fx.to_llvm_ptr(pointer + (linear - lane) * fx.Int32(8)),
                            fx.Int32(16).ir_value(),
                            fx.Int32((first_row + row) * fx.Int32(row_stride * 2) + column * fx.Int32(16)).ir_value(),
                            fx.Int32(0).ir_value(), fx.Int32(0).ir_value(),
                        )
                    else:
                        fx.ptr_store(fx.Vector.filled(8, 0.0, fx.BFloat16).ir_value(), pointer + linear * fx.Int32(8))

        def stage_query(query_tile, stage):
            first_row = query_tile * fx.Int32(16)
            if tid < fx.Int32(4):
                source_index = query_tile * fx.Int32(4) + tid
                for source_view, stats_offset in [(lseview, 0), (deltaview, 16)]:
                    fx.rocdl.raw_ptr_buffer_load_async_lds(
                        fx.rocdl.get_buffer_rsrc(fx.get_iter(source_view)),
                        fx.to_llvm_ptr(shared.stats.ptr + stage * fx.Int32(32) + fx.Int32(stats_offset)),
                        fx.Int32(16).ir_value(),
                        fx.Int32(source_index * fx.Int32(16)).ir_value(),
                        fx.Int32(0).ir_value(), fx.Int32(0).ir_value(),
                    )
            stage_operand(qview, qlayout, shared.query.ptr + stage * fx.Int32(16 * qk_dim), first_row, 16, qk_dim, q_stride[2])
            stage_operand(doview, dolayout, shared.grad_output.ptr + stage * fx.Int32(16 * v_dim), first_row, 16, v_dim, do_stride[2])
            fx.rocdl.asyncmark()

        stage_operand(kview, klayout, shared.key.ptr, key_base, key_rows, qk_dim, k_stride[2])
        stage_operand(vview, vlayout, shared.value.ptr, key_base, key_rows, v_dim, v_stride[2])
        fx.rocdl.asyncmark()

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

        def swap16_vectors(first, second):
            first_packed = first.bitcast(fx.Int32)
            second_packed = second.bitcast(fx.Int32)
            low = []
            high = []
            for part in fx.range_constexpr(2):
                swapped = fx.rocdl.permlane16_swap(
                    ir.Type.parse("!llvm.struct<(i32, i32)>"),
                    fx.Int32(first_packed[part]).ir_value(),
                    fx.Int32(second_packed[part]).ir_value(), False, False,
                )
                low.append(fx.Int32(llvm.extractvalue(fx.Int32.ir_type, swapped, [0])))
                high.append(fx.Int32(llvm.extractvalue(fx.Int32.ir_type, swapped, [1])))
            return fx.Vector.from_elements(low + high, fx.Int32).bitcast(fx.BFloat16)

        def load_dq_operands(stage, step, phase: fx.Constexpr[int], cached_keys):
            ds_pointer = shared.ds.ptr + stage * fx.Int32(key_rows * 16)
            ds_tiles = fx.flat_divide(fx.make_view(ds_pointer, fx.make_layout((16, key_rows), (1, 16))), (16, 32))
            ds_tile = fx.slice(ds_tiles, (None, None, 0, step))
            ds_fragment = score_thread.make_fragment_B(ds_tile)
            fx.copy(tr16, dq_ds_copy.partition_S(ds_tile), dq_ds_copy.retile(ds_fragment))
            key_fragments = cached_keys[phase][step]
            return ds_fragment, key_fragments

        def store_dq(query_tile, accumulator, tile):
            values = (fx.Vector(accumulator.load()) * fx.Vector.filled(4, scale, fx.Float32)).to(fx.BFloat16)
            for pair in fx.range_constexpr(2):
                packed = fx.Vector.from_elements([values[pair * 2], values[pair * 2 + 1]], fx.BFloat16)
                output = fx.make_rmem_tensor(2, fx.BFloat16)
                output.store(packed.ir_value())
                linear = tile * fx.Int32(256) + lane * fx.Int32(2) + fx.Int32(pair * 128)
                row = query_tile * fx.Int32(16) + linear // fx.Int32(qk_dim)
                column = linear % fx.Int32(qk_dim)
                destination = fx.logical_divide(fx.slice(dqview, (row, None)), fx.make_layout(2, 1))
                fx.copy(atomic, output, fx.slice(destination, (None, column // fx.Int32(2))))

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
                    tile = (wave - fx.Int32(producer_waves)) * fx.Int32(2 * dq_phase_tiles) + fx.Int32(phase * dq_phase_tiles + d)
                    store_dq(query_tile, accumulators[d], tile)

        def finish_stage():
            # One common barrier publishes dS(i) and Q/dO(i+1); all waves attend.
            fx.rocdl.wait_asyncmark(0)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.rocdl.s_barrier()

        def update_in_bank(accumulator, a, b, bank: fx.Constexpr[str]):
            # FlyDSL exposes tiled MMA but currently no accumulator-bank API.
            # Keep only this register-class boundary around high-level gemm;
            # replacing it changes AGPR/VGPR allocation and needs a perf A/B.
            c = accumulator.load()
            accumulator.store(llvm.inline_asm(c.type, [c], "", "=" + bank + ",0", has_side_effects=False))
            fx.gemm(wide_mma, accumulator, a, b, accumulator)
            c = accumulator.load()
            accumulator.store(llvm.inline_asm(c.type, [c], "", "=" + bank + ",0", has_side_effects=False))

        def produce(query_tile, stage, masked: fx.Constexpr[bool], dk_wide, dv_wide):
            qpointer = shared.query.ptr + stage * fx.Int32(16 * qk_dim)
            dopointer = shared.grad_output.ptr + stage * fx.Int32(16 * v_dim)
            scores = [fx.make_fragment_like(ccoords, fx.Float32) for _ in fx.range_constexpr(key_groups)]
            probabilities = [fx.make_fragment_like(ccoords, fx.Float32) for _ in fx.range_constexpr(key_groups)]
            ps = [fx.make_fragment_like(update_bcoords, fx.BFloat16) for _ in fx.range_constexpr(key_groups)]
            dss = [fx.make_fragment_like(update_bcoords, fx.BFloat16) for _ in fx.range_constexpr(key_groups)]
            for group in fx.range_constexpr(key_groups):
                scores[group].fill(0)
            keys = [load_score_operand(shared.key.ptr, klayout, 0, group) for group in fx.range_constexpr(key_groups)]
            query_operand = load_score(qpointer, qlayout, 0)
            for step in fx.range_constexpr(qk_dim // 32):
                if const_expr(step + 1 < qk_dim // 32):
                    next_keys = [load_score_operand(shared.key.ptr, klayout, step + 1, group) for group in fx.range_constexpr(key_groups)]
                    next_query = load_score(qpointer, qlayout, step + 1)
                for group in fx.range_constexpr(key_groups):
                    fx.gemm(score_mma, scores[group], query_operand, keys[group], scores[group])
                if const_expr(step + 1 < qk_dim // 32):
                    keys = next_keys
                    query_operand = next_query
            values_operand = [load_score_operand(shared.value.ptr, vlayout, 0, group) for group in fx.range_constexpr(key_groups)]
            grad_output_operand = load_score(dopointer, dolayout, 0)
            for group in fx.range_constexpr(key_groups):
                score_values = fx.Vector(scores[group].load())
                values = []
                for element in fx.range_constexpr(4):
                    local_row = fx.Int32(fx.get_scalar(ccoords[element]))
                    position = query_tile * fx.Int32(16) + local_row
                    lse = fx.Float32(fx.ptr_load(shared.stats.ptr + stage * fx.Int32(32) + local_row))
                    if const_expr(not lse_in_log2):
                        lse = lse * f32(1.4426950408889634)
                    probability = exp2(f32(fx.math.fma(f32(score_values[element]), f32(scale_log2), -lse)))
                    if masked:
                        probability = evaluate_mask(position, key_pos + fx.Int32(group * 16)).select(probability, f32(0.0))
                    if const_expr(not direct_range):
                        probability = (key_pos + fx.Int32(group * 16) < fx.Int32(sequence_length)).select(probability, f32(0.0))
                    values.append(probability)
                probabilities[group].store(fx.Vector.from_elements(values, fx.Float32).ir_value())
            dps = [fx.make_fragment_like(ccoords, fx.Float32) for _ in fx.range_constexpr(key_groups)]
            for group in fx.range_constexpr(key_groups):
                dps[group].fill(0)
            for step in fx.range_constexpr(v_dim // 32):
                if const_expr(step + 1 < v_dim // 32):
                    next_values = [load_score_operand(shared.value.ptr, vlayout, step + 1, group) for group in fx.range_constexpr(key_groups)]
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
                    delta_value = fx.Float32(fx.ptr_load(shared.stats.ptr + stage * fx.Int32(32) + fx.Int32(16) + local_row))
                    ds_values.append(f32(probability_values[element]) * (f32(dp_values[element]) - delta_value))
                ps[group].store(probability_values.to(fx.BFloat16).ir_value())
                ds_vector = fx.Vector.from_elements(ds_values, fx.Float32).to(fx.BFloat16)
                dss[group].store(ds_vector.ir_value())
                dspointer = shared.ds.ptr + stage * fx.Int32(key_rows * 16) + (wave * fx.Int32(key_groups) + fx.Int32(group)) * fx.Int32(256)
                ds_offset = fx.Int32(fx.get_scalar(rowcoords[0])) * fx.Int32(16) + fx.Int32(fx.get_scalar(ccoords[0]))
                fx.ptr_store(ds_vector.ir_value(), dspointer + ds_offset)
            wide_operand = wide_thread.partition_B(fx.make_view(0, fx.make_layout((32, 16), (16, 1))))
            ds_wide = fx.make_fragment_like(wide_operand, fx.BFloat16)
            p_wide = fx.make_fragment_like(wide_operand, fx.BFloat16)
            ds_wide.store(swap16_vectors(fx.Vector(dss[0].load()), fx.Vector(dss[1].load())).ir_value())
            p_wide.store(swap16_vectors(fx.Vector(ps[0].load()), fx.Vector(ps[1].load())).ir_value())
            qa = load_wide(qpointer, qlayout, 0)
            for d in fx.range_constexpr(qk_dim // 32):
                if const_expr(d + 1 < qk_dim // 32):
                    next_qa = load_wide(qpointer, qlayout, d + 1)
                if const_expr(d < v_dim // 32):
                    doa = load_wide(dopointer, dolayout, d)
                update_in_bank(dk_wide[d], qa, ds_wide, "a")
                if const_expr(d < v_dim // 32):
                    update_in_bank(dv_wide[d], doa, p_wide, "v")
                if const_expr(d + 1 < qk_dim // 32):
                    qa = next_qa

        def run(count, first_query, masked: fx.Constexpr[bool], producer: fx.Constexpr[bool], dk_wide, dv_wide, cached_keys):
            def query_at(index):
                return first_query + index
            if count > fx.Int32(0):
                if const_expr(producer):
                    stage_query(query_at(fx.Int32(0)), fx.Int32(0))
                fx.rocdl.wait_asyncmark(0)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                fx.gpu.barrier()
                if const_expr(producer):
                    if count > fx.Int32(1):
                        stage_query(query_at(fx.Int32(1)), fx.Int32(1))
                    produce(query_at(fx.Int32(0)), fx.Int32(0), masked, dk_wide, dv_wide)
                finish_stage()
                for index in range(fx.Int32(1), count, fx.Int32(1)):
                    stage = index % fx.Int32(2)
                    if const_expr(producer):
                        if index + fx.Int32(1) < count:
                            stage_query(query_at(index + fx.Int32(1)), fx.Int32(1) - stage)
                        produce(query_at(index), stage, masked, dk_wide, dv_wide)
                    else:
                        consume(query_at(index - fx.Int32(1)), fx.Int32(1) - stage, cached_keys)
                    finish_stage()
                if const_expr(not producer):
                    consume(query_at(count - fx.Int32(1)), (count - fx.Int32(1)) % fx.Int32(2), cached_keys)
                # A subsequent masked/full list may reuse either dS stage.
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.rocdl.s_barrier()

        def execute_runs(producer: fx.Constexpr[bool], dk_wide, dv_wide, cached_keys):
            if const_expr(unmasked):
                run(fx.Int32(sequence_length // 16), fx.Int32(0), False, producer, dk_wide, dv_wide, cached_keys)
            elif const_expr(direct_range):
                first_query = key_base // fx.Int32(16)
                masked_count = fx.min(fx.Int32(key_rows // 16), fx.Int32(sequence_length // 16) - first_query)
                run(masked_count, first_query, True, producer, dk_wide, dv_wide, cached_keys)
                run(fx.Int32(sequence_length // 16) - first_query - masked_count, first_query + masked_count, False, producer, dk_wide, dv_wide, cached_keys)

        fx.rocdl.wait_asyncmark(0)
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        fx.rocdl.s_barrier()
        def cache_consumer_keys():
            key_tiles = fx.flat_divide(fx.make_view(shared.key.ptr, fx.select(klayout, [1, 0])), (16, 32))
            cached_keys = []
            for phase in fx.range_constexpr(2):
                phase_keys = []
                for step in fx.range_constexpr(key_rows // 32):
                    key_fragments = []
                    for d in fx.range_constexpr(dq_phase_tiles):
                        dimension_tile = (wave - fx.Int32(producer_waves)) * fx.Int32(2 * dq_phase_tiles) + fx.Int32(phase * dq_phase_tiles + d)
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
                        output.store(fx.Vector.from_elements([values[part * 4 + e] for e in fx.range_constexpr(4)], fx.BFloat16).ir_value())
                        column = fx.Int32(d * 32) + fx.Int32(fx.get_scalar(wide_dim_coords[part * 4]))
                        fx.copy(g64, output, fx.slice(dkrow, (None, column // fx.Int32(4))))
                for d in fx.range_constexpr(v_dim // 32):
                    values = fx.Vector(dv_wide[d].load()).to(fx.BFloat16)
                    for part in fx.range_constexpr(4):
                        output = fx.make_rmem_tensor(4, fx.BFloat16)
                        output.store(fx.Vector.from_elements([values[part * 4 + e] for e in fx.range_constexpr(4)], fx.BFloat16).ir_value())
                        column = fx.Int32(d * 32) + fx.Int32(fx.get_scalar(wide_dim_coords[part * 4]))
                        fx.copy(g64, output, fx.slice(dvrow, (None, column // fx.Int32(4))))

        else:
            cached_keys = cache_consumer_keys()
            execute_runs(False, [], [], cached_keys)

    return compute_kernel
