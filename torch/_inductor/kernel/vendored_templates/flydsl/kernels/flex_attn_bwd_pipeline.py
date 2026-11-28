"""Producer waves overlap current QK/dP/dK/dV with two prior-dQ waves."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr

from .flex_attn_bwd_utils import (
    CAPTURE_GRAD_FULL,
    CAPTURE_GRAD_INVARIANT,
    CAPTURE_GRAD_KEY,
    CAPTURE_GRAD_QUERY,
    CAPTURE_GRAD_RELATIVE,
    make_bwd_shared_layout,
)
from .flex_attn_utils import (
    make_capture_buffers,
    make_global_view,
    make_mask_buffers,
    make_mask_evaluator,
    make_score_evaluator,
    store_element,
)


def make_pipeline_kernel(context, f32, exp2):
    """Build the block-list producer/consumer pipeline."""
    batch_size = context["batch_size"]
    num_heads = context["num_heads"]
    sequence_length = context["sequence_length"]
    qk_dim = context["qk_head_dim"]
    v_dim = context["value_head_dim"]
    num_waves = context["num_waves"]
    q_stride = context["q_stride"]
    k_stride = context["k_stride"]
    v_stride = context["v_stride"]
    do_stride = context["grad_output_stride"]
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
    has_score_mod = context["has_score_mod"]
    score_unit_gradient = context["score_unit_gradient"]
    score_negative_infinity = context["score_negative_infinity"]
    score_bias_slot = context["score_bias_slot"]
    score_program = context["score_program"]
    score_output = context["score_program_output"]
    score_buffer_sizes = context["score_buffer_sizes"]
    score_buffer_strides = context["score_buffer_strides"]
    dq_partitions = context["dq_partitions"]
    dq_workspace_dtype = context["dq_workspace_dtype"]
    dq_copy_op = context["dq_copy_op"]
    workspace_head_group = context["workspace_head_group"]
    cta_head_group = context["cta_head_group"]
    batch_heads = batch_size * num_heads
    owner_count = context["fused_grid"]
    threads = num_waves * 64
    key_groups = context["key_groups"]
    producer_waves = context["producer_waves"]
    consumer_waves = 2
    key_rows = producer_waves * key_groups * 16
    dq_phase_tiles = qk_dim // (16 * consumer_waves * 2)
    if qk_dim % 64:
        raise ValueError(f"QK head dimension must be divisible by 64, got {qk_dim}")
    # (kind, load slot, relative sign) per trainable score capture. Gradients go
    # to private fp32 workspaces in fixed order; Inductor finishes the reduction.
    capture_grads = context["capture_grads"]
    seeded_loads = tuple(slot for _, slot, _ in capture_grads if slot >= 0)
    staged_captures = [
        index
        for index, (kind, _, _) in enumerate(capture_grads)
        if kind in (CAPTURE_GRAD_QUERY, CAPTURE_GRAD_RELATIVE)
    ]
    capture_stage_tile = 16 * key_rows
    # Several owners share each CTA's private dQ slot.
    shared_dq_slots = block_list and dq_partitions < owner_count
    dq_view_elements = dq_partitions * sequence_length * qk_dim * workspace_head_group
    relative_extent = 2 * sequence_length - 1

    @fx.struct
    class SharedMemory:
        key: fx.Array[fx.BFloat16, key_rows * qk_dim, 16]
        value: fx.Array[fx.BFloat16, key_rows * v_dim, 16]
        query: fx.Array[fx.BFloat16, 2 * 16 * qk_dim, 16]
        grad_output: fx.Array[fx.BFloat16, 2 * 16 * v_dim, 16]
        ds: fx.Array[fx.BFloat16, 2 * key_rows * 16, 16]
        probability: fx.Array[fx.BFloat16, key_rows * 16, 16]
        stats: fx.Array[fx.Float32, 2 * 32, 16]
        # Double-buffered fp32 contribution tiles for row/diagonal reductions.
        capture_stage: fx.Array[
            fx.Float32, max(1, 2 * len(staged_captures) * capture_stage_tile), 16
        ]
        # Query blocks this CTA's dQ slot already holds, so the first owner to
        # touch a block writes it without reading (or zeroing) the workspace.
        written_blocks: fx.Array[fx.Int32, max(16, metadata_chunks), 16]

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
        capture_grad_0: fx.Tensor,
        capture_grad_1: fx.Tensor,
    ):
        batch_head = fx.Int32(fx.block_idx.z) * fx.Int32(cta_head_group) + fx.Int32(
            fx.block_idx.x
        )
        partial_slot = fx.Int32(fx.block_idx.y)
        tid = fx.Int32(fx.thread_idx.x)
        wave = fx.Int32(fx.gpu.shuffle_idx(tid // fx.Int32(64), 0, 64))
        shared = fx.SharedAllocator().allocate(SharedMemory).peek()
        if const_expr(shared_dq_slots):
            for step in fx.range_constexpr((metadata_chunks + threads - 1) // threads):
                block = tid + fx.Int32(step * threads)
                if block < fx.Int32(metadata_chunks):
                    fx.ptr_store(
                        fx.Int32(0).ir_value(), shared.written_blocks.ptr + block
                    )
            fx.gpu.barrier()
        capture_outputs = [
            fx.get_iter(tensor) for tensor in (capture_grad_0, capture_grad_1)
        ]
        # Batch/head-invariant captures accumulate across every owner of the CTA.
        invariant_sums = []
        for kind, _, _ in capture_grads:
            accumulator = fx.make_rmem_tensor(1, fx.Float32)
            accumulator.store(fx.Vector.filled(1, 0.0, fx.Float32).ir_value())
            invariant_sums.append(accumulator)

        def process_owner(owner, producer: fx.Constexpr[bool]):
            tid = fx.Int32(
                llvm.inline_asm(
                    fx.Int32.ir_type,
                    [fx.Int32(fx.thread_idx.x).ir_value()],
                    "",
                    "=v,0",
                    has_side_effects=True,
                )
            ) & fx.Int32((1 << (threads - 1).bit_length()) - 1)
            lane = tid % fx.Int32(64)
            wave = fx.Int32(
                fx.rocdl.readfirstlane(
                    fx.Int32.ir_type, (tid // fx.Int32(64)).ir_value()
                )
            )
            batch = batch_head // fx.Int32(num_heads)
            head = batch_head % fx.Int32(num_heads)
            head_coord = (fx.Int64(batch), fx.Int64(head), None, None)
            qview = make_global_view(
                query,
                head_coord,
                (batch_size, num_heads, sequence_length, qk_dim),
                q_stride,
            )
            kview = make_global_view(
                key,
                head_coord,
                (batch_size, num_heads, sequence_length, qk_dim),
                k_stride,
            )
            vview = make_global_view(
                value,
                head_coord,
                (batch_size, num_heads, sequence_length, v_dim),
                v_stride,
            )
            doview = make_global_view(
                grad_output,
                head_coord,
                (batch_size, num_heads, sequence_length, v_dim),
                do_stride,
            )
            dqview = make_global_view(
                grad_query,
                (fx.Int64(batch_head // fx.Int32(workspace_head_group)), None),
                (
                    (batch_heads + workspace_head_group - 1) // workspace_head_group,
                    dq_partitions * sequence_length * qk_dim * workspace_head_group,
                ),
                (dq_partitions * sequence_length * qk_dim * workspace_head_group, 1),
            )
            dkview = make_global_view(
                grad_key,
                head_coord,
                (batch_size, num_heads, sequence_length, qk_dim),
                dk_stride,
            )
            dvview = make_global_view(
                grad_value,
                head_coord,
                (batch_size, num_heads, sequence_length, v_dim),
                dv_stride,
            )
            lseview = make_global_view(
                logsumexp,
                (fx.Int64(batch_head), None),
                (batch_size * num_heads, sequence_length),
                (sequence_length, 1),
            )
            deltaview = make_global_view(
                delta,
                (fx.Int64(batch_head), None),
                (batch_size * num_heads, sequence_length),
                (sequence_length, 1),
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
                mask_program,
                mask_output,
                mask_buffer_strides,
                masks,
                load_i32,
                batch,
                head,
            )
            # Score captures follow the mask captures in the buffer slots.
            evaluate_score = make_score_evaluator(
                score_program,
                score_output,
                score_buffer_strides,
                make_capture_buffers(
                    make_global_view,
                    score_buffer_sizes,
                    mask_buffer_count,
                    mask_buffer_0,
                    mask_buffer_1,
                    mask_buffer_2,
                    mask_buffer_3,
                ),
                lambda view, index: fx.get_iter(view)[index],
                batch,
                head,
                seeded_loads=seeded_loads,
            )
            # Key-indexed captures accumulate per owner, like dK and dV.
            key_sums = []
            for kind, _, _ in capture_grads:
                accumulator = fx.make_rmem_tensor(key_groups, fx.Float32)
                accumulator.store(
                    fx.Vector.filled(key_groups, 0.0, fx.Float32).ir_value()
                )
                key_sums.append(accumulator)
            g64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
            dq_copy = fx.make_copy_atom(dq_copy_op(), dq_workspace_dtype)
            read128 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            tr16 = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(16, 64), fx.BFloat16)
            score_mma = fx.make_tiled_mma(
                fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16)),
                fx.make_layout((1, 1, 1), (1, 1, 1)),
            )
            wide_mma = fx.make_tiled_mma(
                fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16)),
                fx.make_layout((1, 1, 1), (1, 1, 1)),
            )
            wide_thread = wide_mma.get_slice(lane)
            wide_dim_coords = wide_thread.partition_C(
                fx.make_view(0, fx.make_layout((32, 32), (1, 0)))
            )
            wide_row_coords = wide_thread.partition_C(
                fx.make_view(0, fx.make_layout((32, 32), (0, 1)))
            )
            wide_a_copy = fx.make_tiled_copy_A(tr16, wide_mma).get_slice(lane)
            score_thread = score_mma.get_slice(lane)
            ccoords = score_thread.partition_C(
                fx.make_view(0, fx.make_layout((16, 16), (1, 0)))
            )
            rowcoords = score_thread.partition_C(
                fx.make_view(0, fx.make_layout((16, 16), (0, 1)))
            )
            dq_key_copy = fx.make_tiled_copy_A(tr16, score_mma).get_slice(lane)
            dq_ds_copy = fx.make_tiled_copy_B(tr16, score_mma).get_slice(lane)
            score_copy = fx.make_tiled_copy_A(read128, score_mma).get_slice(lane)
            score_b_copy = fx.make_tiled_copy_B(read128, score_mma).get_slice(lane)
            write64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
            score_store = fx.make_tiled_copy_C(write64, score_mma).get_slice(lane)
            packed_dq = fx.logical_divide(dqview, fx.make_layout(4, 1))
            wide_b_copy = fx.make_tiled_copy_B(read128, wide_mma).get_slice(lane)
            key_base = owner * fx.Int32(key_rows)
            key_pos = (
                key_base
                + wave * fx.Int32(key_groups * 16)
                + fx.Int32(fx.get_scalar(rowcoords[0]))
            )
            klayout = make_bwd_shared_layout(key_rows, qk_dim)
            vlayout = make_bwd_shared_layout(key_rows, v_dim)
            qlayout = make_bwd_shared_layout(16, qk_dim)
            dolayout = make_bwd_shared_layout(16, v_dim)
            score_layout = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(2, 3, 3)),
                fx.make_layout((16, key_rows), (1, 16)),
            )

            def stage_operand(
                view,
                layout,
                pointer,
                first_row,
                rows,
                dimension,
                loader_threads,
                loader_stride,
            ):
                # Distribute physical LDS packs; invert the swizzled layout to
                # recover their logical global coordinates, as in forward.
                coordinates = fx.make_composed_layout(
                    fx.right_inverse(layout.outer),
                    fx.make_composed_layout(
                        layout.inner, fx.make_layout(rows * dimension, 1)
                    ),
                )
                destinations = fx.logical_divide(
                    fx.make_view(pointer, fx.make_layout(rows * dimension, 1)),
                    fx.make_layout(8, 1),
                )
                for step in fx.range_constexpr(
                    (rows * dimension // 8 + loader_threads - 1) // loader_threads
                ):
                    linear = tid + fx.Int32(step * loader_threads)
                    if linear < fx.Int32(rows * dimension // 8):
                        logical = fx.Int32(
                            fx.get_scalar(fx.crd2idx(linear * fx.Int32(8), coordinates))
                        )
                        row = logical % fx.Int32(rows)
                        column = logical // fx.Int32(rows * 8)
                        destination = fx.slice(destinations, (None, linear))
                        if first_row + row < fx.Int32(sequence_length):
                            fx.rocdl.raw_ptr_buffer_load_async_lds(
                                fx.rocdl.get_buffer_rsrc(fx.get_iter(view)),
                                fx.to_llvm_ptr(pointer + (linear - lane) * fx.Int32(8)),
                                fx.Int32(16).ir_value(),
                                fx.Int32(
                                    (first_row + row) * fx.Int32(loader_stride * 2)
                                    + column * fx.Int32(16)
                                ).ir_value(),
                                fx.Int32(0).ir_value(),
                                fx.Int32(0).ir_value(),
                            )
                        else:
                            zeros = fx.make_rmem_tensor(8, fx.BFloat16)
                            zeros.fill(0)
                            fx.copy(read128, zeros, destination)

            def stage_query(query_tile, stage):
                first_row = query_tile * fx.Int32(16)
                if tid < fx.Int32(4):
                    source_index = query_tile * fx.Int32(4) + tid
                    for source_view, stats_offset in [(lseview, 0), (deltaview, 16)]:
                        fx.rocdl.raw_ptr_buffer_load_async_lds(
                            fx.rocdl.get_buffer_rsrc(fx.get_iter(source_view)),
                            fx.to_llvm_ptr(
                                shared.stats.ptr
                                + stage * fx.Int32(32)
                                + fx.Int32(stats_offset)
                            ),
                            fx.Int32(16).ir_value(),
                            fx.Int32(source_index * fx.Int32(16)).ir_value(),
                            fx.Int32(0).ir_value(),
                            fx.Int32(0).ir_value(),
                        )
                stage_operand(
                    qview,
                    qlayout,
                    shared.query.ptr + stage * fx.Int32(16 * qk_dim),
                    first_row,
                    16,
                    qk_dim,
                    producer_waves * 64,
                    q_stride[2],
                )
                stage_operand(
                    doview,
                    dolayout,
                    shared.grad_output.ptr + stage * fx.Int32(16 * v_dim),
                    first_row,
                    16,
                    v_dim,
                    producer_waves * 64,
                    do_stride[2],
                )

            stage_operand(
                kview,
                klayout,
                shared.key.ptr,
                key_base,
                key_rows,
                qk_dim,
                threads,
                k_stride[2],
            )
            stage_operand(
                vview,
                vlayout,
                shared.value.ptr,
                key_base,
                key_rows,
                v_dim,
                threads,
                v_stride[2],
            )

            def load_score(pointer, layout, step):
                tiles = fx.flat_divide(fx.make_view(pointer, layout), (16, 32))
                tile = fx.slice(tiles, (None, None, 0, step))
                fragment = score_thread.make_fragment_A(tile)
                fx.copy(
                    read128, score_copy.partition_S(tile), score_copy.retile(fragment)
                )
                return fragment

            def load_score_operand(pointer, layout, step, group):
                tiles = fx.flat_divide(fx.make_view(pointer, layout), (16, 32))
                row_tile = wave * fx.Int32(key_groups) + fx.Int32(group)
                tile = fx.slice(tiles, (None, None, row_tile, step))
                fragment = score_thread.make_fragment_B(tile)
                fx.copy(
                    read128,
                    score_b_copy.partition_S(tile),
                    score_b_copy.retile(fragment),
                )
                return fragment

            def load_wide(pointer, layout, dim_chunk):
                tiles = fx.flat_divide(
                    fx.make_view(pointer, fx.select(layout, [1, 0])), (32, 16)
                )
                tile = fx.slice(tiles, (None, None, dim_chunk, 0))
                fragment = wide_thread.make_fragment_A(tile)
                fx.copy(
                    tr16, wide_a_copy.partition_S(tile), wide_a_copy.retile(fragment)
                )
                return fragment

            def store_score(pointer, group, values):
                tiles = fx.flat_divide(fx.make_view(pointer, score_layout), (16, 16))
                tile = fx.slice(
                    tiles,
                    (None, None, 0, wave * fx.Int32(key_groups) + fx.Int32(group)),
                )
                fragment = fx.make_fragment_like(
                    score_thread.partition_C(tile), fx.BFloat16
                )
                fragment.store(values.ir_value())
                fx.copy(
                    write64, score_store.retile(fragment), score_store.partition_D(tile)
                )

            def load_wide_score(pointer):
                tiles = fx.flat_divide(
                    fx.make_view(pointer, fx.select(score_layout, [1, 0])), (32, 16)
                )
                tile = fx.slice(tiles, (None, None, wave, 0))
                fragment = wide_thread.make_fragment_B(tile)
                fx.copy(
                    read128, wide_b_copy.partition_S(tile), wide_b_copy.retile(fragment)
                )
                return fragment

            def load_dq_operands(stage, step, phase: fx.Constexpr[int], cached_keys):
                ds_pointer = shared.ds.ptr + stage * fx.Int32(key_rows * 16)
                ds_tiles = fx.flat_divide(
                    fx.make_view(ds_pointer, score_layout), (16, 32)
                )
                ds_tile = fx.slice(ds_tiles, (None, None, 0, step))
                ds_fragment = score_thread.make_fragment_B(ds_tile)
                fx.copy(
                    tr16,
                    dq_ds_copy.partition_S(ds_tile),
                    dq_ds_copy.retile(ds_fragment),
                )
                key_fragments = cached_keys[phase][step]
                return ds_fragment, key_fragments

            previous_dq = [
                fx.make_rmem_tensor(4, dq_workspace_dtype)
                for _ in fx.range_constexpr(2 * dq_phase_tiles)
            ]

            def load_previous_dq_phase(query_tile, phase: fx.Constexpr[int]):
                if const_expr(shared_dq_slots):
                    written = fx.Int32(
                        fx.ptr_load(
                            shared.written_blocks.ptr
                            + query_tile // fx.Int32(query_split)
                        )
                    ) != fx.Int32(0)
                for d in fx.range_constexpr(dq_phase_tiles if shared_dq_slots else 0):
                    part = phase * dq_phase_tiles + d
                    tile = (wave - fx.Int32(producer_waves)) * fx.Int32(
                        2 * dq_phase_tiles
                    ) + fx.Int32(part)
                    offset = (
                        (
                            (query_tile * fx.Int32(dq_partitions) + partial_slot)
                            * fx.Int32(workspace_head_group)
                            + batch_head % fx.Int32(workspace_head_group)
                        )
                        * fx.Int32(16 * qk_dim)
                        + tile * fx.Int32(256)
                        + lane * fx.Int32(4)
                    )
                    # An unwritten block reads past the buffer view, which
                    # returns zeros without memory traffic.
                    offset = written.select(offset, fx.Int32(dq_view_elements))
                    fx.copy(
                        dq_copy,
                        fx.slice(packed_dq, (None, offset // fx.Int32(4))),
                        previous_dq[part],
                    )

            def load_previous_dq(query_tile):
                for phase in fx.range_constexpr(2):
                    load_previous_dq_phase(query_tile, phase)

            def store_dq(query_tile, accumulator, tile, part: fx.Constexpr[int]):
                offset = (
                    (
                        (query_tile * fx.Int32(dq_partitions) + partial_slot)
                        * fx.Int32(workspace_head_group)
                        + batch_head % fx.Int32(workspace_head_group)
                    )
                    * fx.Int32(16 * qk_dim)
                    + tile * fx.Int32(256)
                    + lane * fx.Int32(4)
                )
                destination = fx.slice(packed_dq, (None, offset // fx.Int32(4)))
                values = fx.Vector(accumulator.load()) * fx.Vector.filled(
                    4, scale, fx.Float32
                )
                if const_expr(shared_dq_slots):
                    values = values + fx.Vector(previous_dq[part].load()).to(fx.Float32)
                output = fx.make_rmem_tensor(4, dq_workspace_dtype)
                output.store(values.to(dq_workspace_dtype).ir_value())
                fx.copy(dq_copy, output, destination)

            def consume(
                query_tile, stage, cached_keys, next_query, prefetch: fx.Constexpr[bool]
            ):
                # Two phases keep dQ state small while producer accumulators live.
                for phase in fx.range_constexpr(2):
                    accumulators = [
                        fx.make_fragment_like(ccoords, fx.Float32)
                        for _ in fx.range_constexpr(dq_phase_tiles)
                    ]
                    for d in fx.range_constexpr(dq_phase_tiles):
                        accumulators[d].fill(0)
                    operands = load_dq_operands(stage, 0, phase, cached_keys)
                    for step in fx.range_constexpr(key_rows // 32):
                        if const_expr(step + 1 < key_rows // 32):
                            next_operands = load_dq_operands(
                                stage, step + 1, phase, cached_keys
                            )
                        for d in fx.range_constexpr(dq_phase_tiles):
                            fx.gemm(
                                score_mma,
                                accumulators[d],
                                operands[1][d],
                                operands[0],
                                accumulators[d],
                            )
                        if const_expr(step + 1 < key_rows // 32):
                            operands = next_operands
                    for d in fx.range_constexpr(dq_phase_tiles):
                        tile = (wave - fx.Int32(producer_waves)) * fx.Int32(
                            2 * dq_phase_tiles
                        ) + fx.Int32(phase * dq_phase_tiles + d)
                        store_dq(
                            query_tile,
                            accumulators[d],
                            tile,
                            phase * dq_phase_tiles + d,
                        )
                    if const_expr(prefetch):
                        load_previous_dq_phase(next_query, phase)
                if const_expr(bool(staged_captures)):
                    reduce_staged_captures(query_tile, stage)

            def reduce_staged_captures(query_tile, stage):
                # Consumers sum the producers' contribution tile in a fixed order
                # and update this CTA's private slot; no other CTA writes it.
                consumer_lane = tid - fx.Int32(producer_waves * 64)
                slot_base = batch_head * fx.Int32(dq_partitions) + partial_slot
                for position, index in enumerate(staged_captures):
                    kind, _, sign = capture_grads[index]
                    tile = (
                        shared.capture_stage.ptr
                        + fx.Int32(2 * position * capture_stage_tile)
                        + stage * fx.Int32(capture_stage_tile)
                    )
                    output = capture_outputs[index]
                    if const_expr(kind == CAPTURE_GRAD_QUERY):
                        if consumer_lane < fx.Int32(16):
                            row_total = f32(0.0)
                            for column in fx.range_constexpr(key_rows):
                                row_total = row_total + f32(
                                    fx.ptr_load(
                                        tile
                                        + consumer_lane * fx.Int32(key_rows)
                                        + fx.Int32(column)
                                    )
                                )
                            target = (
                                slot_base * fx.Int32(sequence_length)
                                + query_tile * fx.Int32(16)
                                + consumer_lane
                            )
                            store_element(
                                output, target, f32(output[target]) + row_total
                            )
                    else:
                        for part in fx.range_constexpr(
                            (16 + key_rows - 1 + consumer_waves * 64 - 1)
                            // (consumer_waves * 64)
                        ):
                            diagonal = consumer_lane + fx.Int32(
                                part * consumer_waves * 64
                            )
                            if diagonal < fx.Int32(16 + key_rows - 1):
                                # row - column along this diagonal of the tile.
                                shift = diagonal - fx.Int32(key_rows - 1)
                                row_total = f32(0.0)
                                for row in fx.range_constexpr(16):
                                    column = fx.Int32(row) - shift
                                    valid = (column >= fx.Int32(0)) & (
                                        column < fx.Int32(key_rows)
                                    )
                                    entry = f32(
                                        fx.ptr_load(
                                            tile
                                            + fx.Int32(row * key_rows)
                                            + valid.select(column, fx.Int32(0))
                                        )
                                    )
                                    row_total = row_total + valid.select(
                                        entry, f32(0.0)
                                    )
                                relative = query_tile * fx.Int32(16) - key_base + shift
                                target = (
                                    slot_base * fx.Int32(relative_extent)
                                    + fx.Int32(sign) * relative
                                    + fx.Int32(sequence_length - 1)
                                )
                                store_element(
                                    output, target, f32(output[target]) + row_total
                                )
                # The next tile may update the same entries.
                fx.rocdl.s_waitcnt(vmcnt=0)

            def finish_stage(producer: fx.Constexpr[bool]):
                # One common barrier publishes dS(i) and Q/dO(i+1); all waves attend.
                if const_expr(producer):
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                else:
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()

            def accumulate_capture_grads(
                stage,
                group,
                element,
                local_row,
                probability,
                ds_value,
                load_gradients,
                load_offsets,
                capture_parts,
            ):
                """Route one score's capture-gradient contributions by class."""
                trainable = [
                    (index, kind)
                    for index, (kind, slot, _) in enumerate(capture_grads)
                    if slot >= 0
                ]
                for seed, (index, kind) in enumerate(trainable):
                    contribution = (probability == f32(0.0)).select(
                        f32(0.0), ds_value * load_gradients[seed][group][element]
                    )
                    if const_expr(kind in (CAPTURE_GRAD_INVARIANT, CAPTURE_GRAD_KEY)):
                        capture_parts[index][group].append(contribution)
                    elif const_expr(kind == CAPTURE_GRAD_FULL):
                        # Each (b, h, q, kv) element is written exactly once.
                        store_element(
                            capture_outputs[index],
                            load_offsets[seed][group][element],
                            contribution,
                        )
                    else:
                        offset = (
                            fx.Int32(
                                2 * staged_captures.index(index) * capture_stage_tile
                            )
                            + stage * fx.Int32(capture_stage_tile)
                            + local_row * fx.Int32(key_rows)
                            + (key_pos - key_base)
                            + fx.Int32(group * 16)
                        )
                        fx.ptr_store(
                            contribution.ir_value(), shared.capture_stage.ptr + offset
                        )

            def produce(
                query_tile, stage, masked: fx.Constexpr[bool], dk_wide, dv_wide
            ):
                qpointer = shared.query.ptr + stage * fx.Int32(16 * qk_dim)
                dopointer = shared.grad_output.ptr + stage * fx.Int32(16 * v_dim)
                scores = [
                    fx.make_fragment_like(ccoords, fx.Float32)
                    for _ in fx.range_constexpr(key_groups)
                ]
                probabilities = [
                    fx.make_fragment_like(ccoords, fx.Float32)
                    for _ in fx.range_constexpr(key_groups)
                ]
                for group in fx.range_constexpr(key_groups):
                    scores[group].fill(0)
                # Mask bits and score-independent score_mod work (table loads,
                # position arithmetic) overlap the QK MFMAs below.
                early_keep = [[] for _ in fx.range_constexpr(key_groups)]
                early_score = [[] for _ in fx.range_constexpr(key_groups)]
                # Additive part of each exponent for identity and score + bias:
                # column terms + (row terms - LSE) + joint terms, in log2 units.
                early_exponent = [[] for _ in fx.range_constexpr(key_groups)]
                # LSE per row, read before the MFMAs for every score path.
                row_lses = []
                for element in fx.range_constexpr(4):
                    row_lse = fx.Float32(
                        shared.stats.ptr[
                            stage * fx.Int32(32)
                            + fx.Int32(fx.get_scalar(ccoords[element]))
                        ]
                    )
                    if const_expr(not lse_in_log2):
                        row_lse = row_lse * f32(1.4426950408889634)
                    if const_expr(score_negative_infinity):
                        # score_mod may empty a row with -inf scores.
                        row_lse = (row_lse == f32(float("-inf"))).select(
                            f32(0.0), row_lse
                        )
                    row_lses.append(row_lse)
                if const_expr(score_bias_slot is not None):
                    column_bias = [
                        evaluate_score.bias_part(
                            score_bias_slot,
                            ("col", "inv"),
                            kv_pos=key_pos + fx.Int32(group * 16),
                        )
                        for group in fx.range_constexpr(key_groups)
                    ]
                    row_bias = []
                    for element in fx.range_constexpr(4):
                        row_part = evaluate_score.bias_part(
                            score_bias_slot,
                            ("row",),
                            q_pos=query_tile * fx.Int32(16)
                            + fx.Int32(fx.get_scalar(ccoords[element])),
                        )
                        row_bias.append(
                            -row_lses[element]
                            if row_part is None
                            else row_part - row_lses[element]
                        )
                if const_expr(masked):
                    # Row- and column-only mask terms (e.g. doc_id[q], doc_id[kv])
                    # once per row/column; only the comparison is per element.
                    mask_rows = [
                        evaluate_mask.row(
                            query_tile * fx.Int32(16)
                            + fx.Int32(fx.get_scalar(ccoords[element]))
                        )
                        for element in fx.range_constexpr(4)
                    ]
                    mask_columns = [
                        evaluate_mask.column(key_pos + fx.Int32(group * 16))
                        for group in fx.range_constexpr(key_groups)
                    ]
                for group in fx.range_constexpr(key_groups):
                    for element in fx.range_constexpr(4):
                        position = query_tile * fx.Int32(16) + fx.Int32(
                            fx.get_scalar(ccoords[element])
                        )
                        if masked:
                            early_keep[group].append(
                                evaluate_mask(
                                    position,
                                    key_pos + fx.Int32(group * 16),
                                    {**mask_rows[element], **mask_columns[group]},
                                )
                            )
                        if const_expr(score_bias_slot is not None):
                            additive = row_bias[element]
                            if const_expr(column_bias[group] is not None):
                                additive = column_bias[group] + additive
                            joint = evaluate_score.bias_part(
                                score_bias_slot,
                                ("joint",),
                                q_pos=position,
                                kv_pos=key_pos + fx.Int32(group * 16),
                            )
                            if const_expr(joint is not None):
                                additive = additive + joint
                            early_exponent[group].append(additive)
                        elif const_expr(has_score_mod):
                            early_score[group].append(
                                evaluate_score.prepare(
                                    position, key_pos + fx.Int32(group * 16)
                                )
                            )
                        else:
                            early_exponent[group].append(-row_lses[element])
                keys = [
                    load_score_operand(shared.key.ptr, klayout, 0, group)
                    for group in fx.range_constexpr(key_groups)
                ]
                query_operand = load_score(qpointer, qlayout, 0)
                for step in fx.range_constexpr(qk_dim // 32):
                    if const_expr(step + 1 < qk_dim // 32):
                        next_keys = [
                            load_score_operand(shared.key.ptr, klayout, step + 1, group)
                            for group in fx.range_constexpr(key_groups)
                        ]
                        next_query = load_score(qpointer, qlayout, step + 1)
                    for group in fx.range_constexpr(key_groups):
                        fx.gemm(
                            score_mma,
                            scores[group],
                            query_operand,
                            keys[group],
                            scores[group],
                        )
                    if const_expr(step + 1 < qk_dim // 32):
                        keys = next_keys
                        query_operand = next_query
                values_operand = [
                    load_score_operand(shared.value.ptr, vlayout, 0, group)
                    for group in fx.range_constexpr(key_groups)
                ]
                grad_output_operand = load_score(dopointer, dolayout, 0)
                score_gradients = [[] for _ in fx.range_constexpr(key_groups)]
                load_gradients = [
                    [[] for _ in fx.range_constexpr(key_groups)] for _ in seeded_loads
                ]
                load_offsets = [
                    [[] for _ in fx.range_constexpr(key_groups)] for _ in seeded_loads
                ]
                for group in fx.range_constexpr(key_groups):
                    score_values = fx.Vector(scores[group].load())
                    values = []
                    for element in fx.range_constexpr(4):
                        local_row = fx.Int32(fx.get_scalar(ccoords[element]))
                        position = query_tile * fx.Int32(16) + local_row
                        lse = row_lses[element]
                        if const_expr(score_bias_slot is not None or not has_score_mod):
                            exponent = f32(
                                fx.math.fma(
                                    f32(score_values[element]),
                                    f32(scale_log2),
                                    early_exponent[group][element],
                                )
                            )
                        elif const_expr(has_score_mod):
                            evaluated = evaluate_score(
                                f32(score_values[element]) * f32(scale),
                                position,
                                key_pos + fx.Int32(group * 16),
                                with_gradient=True,
                                prepared=early_score[group][element],
                            )
                            modified, gradient = evaluated[0], evaluated[1]
                            score_gradients[group].append(gradient)
                            for seed in fx.range_constexpr(len(seeded_loads)):
                                load_gradients[seed][group].append(evaluated[2][seed])
                                load_offsets[seed][group].append(evaluated[3][seed])
                            exponent = f32(
                                fx.math.fma(modified, f32(1.4426950408889634), -lse)
                            )
                        else:
                            exponent = f32(
                                fx.math.fma(
                                    f32(score_values[element]), f32(scale_log2), -lse
                                )
                            )
                        # Mask the exponent, not the probability: masked scores
                        # with large biases would otherwise overflow to inf.
                        if masked:
                            exponent = (
                                early_keep[group][element] != fx.Int32(0)
                            ).select(exponent, f32(float("-inf")))
                        if const_expr(not direct_range):
                            exponent = (
                                key_pos + fx.Int32(group * 16)
                                < fx.Int32(sequence_length)
                            ).select(exponent, f32(float("-inf")))
                        probability = exp2(exponent)
                        values.append(probability)
                    probabilities[group].store(
                        fx.Vector.from_elements(values, fx.Float32).ir_value()
                    )
                dps = [
                    fx.make_fragment_like(ccoords, fx.Float32)
                    for _ in fx.range_constexpr(key_groups)
                ]
                for group in fx.range_constexpr(key_groups):
                    dps[group].fill(0)
                for step in fx.range_constexpr(v_dim // 32):
                    if const_expr(step + 1 < v_dim // 32):
                        next_values = [
                            load_score_operand(
                                shared.value.ptr, vlayout, step + 1, group
                            )
                            for group in fx.range_constexpr(key_groups)
                        ]
                        next_grad_output = load_score(dopointer, dolayout, step + 1)
                    for group in fx.range_constexpr(key_groups):
                        fx.gemm(
                            score_mma,
                            dps[group],
                            grad_output_operand,
                            values_operand[group],
                            dps[group],
                        )
                    if const_expr(step + 1 < v_dim // 32):
                        values_operand = next_values
                        grad_output_operand = next_grad_output
                capture_parts = [
                    [[] for _ in fx.range_constexpr(key_groups)] for _ in capture_grads
                ]
                for group in fx.range_constexpr(key_groups):
                    probability_values = fx.Vector(probabilities[group].load())
                    dp_values = fx.Vector(dps[group].load())
                    ds_values = []
                    for element in fx.range_constexpr(4):
                        local_row = fx.Int32(fx.get_scalar(ccoords[element]))
                        delta_value = fx.Float32(
                            shared.stats.ptr[
                                stage * fx.Int32(32) + fx.Int32(16) + local_row
                            ]
                        )
                        probability = f32(probability_values[element])
                        ds_value = probability * (f32(dp_values[element]) - delta_value)
                        if const_expr(bool(seeded_loads)):
                            accumulate_capture_grads(
                                stage,
                                group,
                                element,
                                local_row,
                                probability,
                                ds_value,
                                load_gradients,
                                load_offsets,
                                capture_parts,
                            )
                        if const_expr(has_score_mod and not score_unit_gradient):
                            # Chain through score_mod; skip zero-probability
                            # entries whose derivative may be non-finite.
                            ds_value = (probability == f32(0.0)).select(
                                f32(0.0), ds_value * score_gradients[group][element]
                            )
                        ds_values.append(ds_value)
                    store_score(
                        shared.probability.ptr,
                        group,
                        probability_values.to(fx.BFloat16),
                    )
                    ds_vector = fx.Vector.from_elements(ds_values, fx.Float32).to(
                        fx.BFloat16
                    )
                    store_score(
                        shared.ds.ptr + stage * fx.Int32(key_rows * 16),
                        group,
                        ds_vector,
                    )
                for index, (kind, _, _) in enumerate(capture_grads):
                    # Fixed-order register sums; flushed per owner or per CTA.
                    if const_expr(kind == CAPTURE_GRAD_INVARIANT):
                        total = f32(fx.Vector(invariant_sums[index].load())[0])
                        for group in fx.range_constexpr(key_groups):
                            for part in capture_parts[index][group]:
                                total = total + part
                        invariant_sums[index].store(
                            fx.Vector.from_elements([total], fx.Float32).ir_value()
                        )
                    elif const_expr(kind == CAPTURE_GRAD_KEY):
                        sums = fx.Vector(key_sums[index].load())
                        totals = []
                        for group in fx.range_constexpr(key_groups):
                            total = f32(sums[group])
                            for part in capture_parts[index][group]:
                                total = total + part
                            totals.append(total)
                        key_sums[index].store(
                            fx.Vector.from_elements(totals, fx.Float32).ir_value()
                        )
                # Each wave converts its own 32-key tile through LDS. No other
                # wave reads probability; dS is published by finish_stage.
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                ds_wide = load_wide_score(
                    shared.ds.ptr + stage * fx.Int32(key_rows * 16)
                )
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
                        list_offset = (
                            fx.Int64(batch_head) * fx.Int64(metadata_chunks)
                            + fx.Int64(owner)
                        ) * fx.Int64(metadata_chunks)
                        sparse_query = fx.Int32(
                            fx.get_iter(query_indices)[
                                list_offset + fx.Int64(index // fx.Int32(query_split))
                            ]
                        )
                        return sparse_query * fx.Int32(query_split) + index % fx.Int32(
                            query_split
                        )
                    else:
                        return first_query + index

                if count > fx.Int32(0):
                    if const_expr(producer):
                        stage_query(query_at(fx.Int32(0)), fx.Int32(0))
                    else:
                        load_previous_dq(query_at(fx.Int32(0)))
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    fx.gpu.barrier()
                    if const_expr(producer):
                        if count > fx.Int32(1):
                            stage_query(query_at(fx.Int32(1)), fx.Int32(1))
                        produce(
                            query_at(fx.Int32(0)), fx.Int32(0), masked, dk_wide, dv_wide
                        )
                    finish_stage(producer)

                    def clamped_query(index):
                        last = count - fx.Int32(1)
                        return query_at((index < last).select(index, last))

                    # Query tiles are loop-carried one step ahead, so a tile's
                    # block-list index load never stalls its own iteration.
                    tiles = [
                        clamped_query(fx.Int32(1)),
                        query_at(fx.Int32(0)),
                        clamped_query(fx.Int32(2)),
                    ]
                    for index, carried in range(
                        fx.Int32(1), count, fx.Int32(1), init=tiles
                    ):
                        current = fx.Int32(carried[0])
                        previous = fx.Int32(carried[1])
                        following = fx.Int32(carried[2])
                        lookahead = clamped_query(fx.Int32(index) + fx.Int32(2))
                        stage = fx.Int32(index) % fx.Int32(2)
                        if const_expr(producer):
                            if fx.Int32(index) + fx.Int32(1) < count:
                                stage_query(following, fx.Int32(1) - stage)
                            produce(current, stage, masked, dk_wide, dv_wide)
                        else:
                            consume(
                                previous,
                                fx.Int32(1) - stage,
                                cached_keys,
                                current,
                                True,
                            )
                        finish_stage(producer)
                        tiles = yield [following, current, lookahead]
                    if const_expr(not producer):
                        consume(
                            query_at(count - fx.Int32(1)),
                            (count - fx.Int32(1)) % fx.Int32(2),
                            cached_keys,
                            fx.Int32(0),
                            False,
                        )
                    # A subsequent masked/full list may reuse either dS stage.
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                    fx.gpu.barrier()

            def execute_runs(
                producer: fx.Constexpr[bool], dk_wide, dv_wide, cached_keys
            ):
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
                    masked_count = fx.min(
                        fx.Int32(key_rows // 16),
                        fx.Int32(sequence_length // 16) - first_query,
                    )
                    run(
                        masked_count,
                        first_query,
                        partial_kv_indices,
                        True,
                        producer,
                        dk_wide,
                        dv_wide,
                        cached_keys,
                    )
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
                    count_index = fx.Int64(batch_head) * fx.Int64(
                        metadata_chunks
                    ) + fx.Int64(owner)
                    partial_count = fx.Int32(
                        fx.get_iter(partial_kv_counts)[count_index]
                    )
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
                key_tiles = fx.flat_divide(
                    fx.make_view(shared.key.ptr, fx.select(klayout, [1, 0])), (16, 32)
                )
                cached_keys = []
                for phase in fx.range_constexpr(2):
                    phase_keys = []
                    for step in fx.range_constexpr(key_rows // 32):
                        key_fragments = []
                        for d in fx.range_constexpr(dq_phase_tiles):
                            dimension_tile = (
                                wave - fx.Int32(producer_waves)
                            ) * fx.Int32(2 * dq_phase_tiles) + fx.Int32(
                                phase * dq_phase_tiles + d
                            )
                            key_tile = fx.slice(
                                key_tiles, (None, None, dimension_tile, step)
                            )
                            fragment = score_thread.make_fragment_A(key_tile)
                            fx.copy(
                                tr16,
                                dq_key_copy.partition_S(key_tile),
                                dq_key_copy.retile(fragment),
                            )
                            key_fragments.append(fragment)
                        phase_keys.append(key_fragments)
                    cached_keys.append(phase_keys)
                return cached_keys

            if const_expr(producer):
                dk_wide = [
                    fx.make_fragment_like(wide_dim_coords, fx.Float32)
                    for _ in fx.range_constexpr(qk_dim // 32)
                ]
                dv_wide = [
                    fx.make_fragment_like(wide_dim_coords, fx.Float32)
                    for _ in fx.range_constexpr(v_dim // 32)
                ]
                for d in fx.range_constexpr(qk_dim // 32):
                    dk_wide[d].fill(0)
                for d in fx.range_constexpr(v_dim // 32):
                    dv_wide[d].fill(0)
                execute_runs(True, dk_wide, dv_wide, [])
                for index, (kind, _, _) in enumerate(capture_grads):
                    if const_expr(kind == CAPTURE_GRAD_KEY):
                        # Lanes 16 apart hold the same key; this owner is the
                        # key's only writer.
                        sums = fx.Vector(key_sums[index].load())
                        for group in fx.range_constexpr(key_groups):
                            total = f32(sums[group])
                            total = total + f32(fx.gpu.shuffle_xor(total, 16, 64))
                            total = total + f32(fx.gpu.shuffle_xor(total, 32, 64))
                            if lane < fx.Int32(16):
                                store_element(
                                    capture_outputs[index],
                                    batch_head * fx.Int32(sequence_length)
                                    + key_pos
                                    + fx.Int32(group * 16),
                                    total,
                                )
                wide_key_pos = (
                    key_base
                    + wave * fx.Int32(key_groups * 16)
                    + fx.Int32(fx.get_scalar(wide_row_coords[0]))
                )
                if wide_key_pos < fx.Int32(sequence_length):
                    dkrow = fx.logical_divide(
                        fx.slice(dkview, (wide_key_pos, None)), fx.make_layout(4, 1)
                    )
                    dvrow = fx.logical_divide(
                        fx.slice(dvview, (wide_key_pos, None)), fx.make_layout(4, 1)
                    )
                    for d in fx.range_constexpr(qk_dim // 32):
                        values = (
                            fx.Vector(dk_wide[d].load())
                            * fx.Vector.filled(16, scale, fx.Float32)
                        ).to(fx.BFloat16)
                        for part in fx.range_constexpr(4):
                            output = fx.make_rmem_tensor(4, fx.BFloat16)
                            output.store(
                                fx.Vector.from_elements(
                                    [
                                        values[part * 4 + e]
                                        for e in fx.range_constexpr(4)
                                    ],
                                    fx.BFloat16,
                                ).ir_value()
                            )
                            column = fx.Int32(d * 32) + fx.Int32(
                                fx.get_scalar(wide_dim_coords[part * 4])
                            )
                            fx.copy(
                                g64,
                                output,
                                fx.slice(dkrow, (None, column // fx.Int32(4))),
                            )
                    for d in fx.range_constexpr(v_dim // 32):
                        values = fx.Vector(dv_wide[d].load()).to(fx.BFloat16)
                        for part in fx.range_constexpr(4):
                            output = fx.make_rmem_tensor(4, fx.BFloat16)
                            output.store(
                                fx.Vector.from_elements(
                                    [
                                        values[part * 4 + e]
                                        for e in fx.range_constexpr(4)
                                    ],
                                    fx.BFloat16,
                                ).ir_value()
                            )
                            column = fx.Int32(d * 32) + fx.Int32(
                                fx.get_scalar(wide_dim_coords[part * 4])
                            )
                            fx.copy(
                                g64,
                                output,
                                fx.slice(dvrow, (None, column // fx.Int32(4))),
                            )

            else:
                cached_keys = cache_consumer_keys()
                execute_runs(False, [], [], cached_keys)

            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            fx.gpu.barrier()
            if const_expr(shared_dq_slots):
                # Publish the query blocks this owner wrote; each list holds a
                # block at most once and counts never exceed metadata_chunks.
                count_index = fx.Int64(batch_head) * fx.Int64(
                    metadata_chunks
                ) + fx.Int64(owner)
                list_offset = count_index * fx.Int64(metadata_chunks)
                for indices, counts in (
                    (partial_kv_indices, partial_kv_counts),
                    (full_kv_indices_transposed, full_kv_counts),
                ):
                    count = fx.Int32(fx.get_iter(counts)[count_index])
                    if tid < count:
                        block = fx.Int32(
                            fx.get_iter(indices)[list_offset + fx.Int64(tid)]
                        )
                        fx.ptr_store(
                            fx.Int32(1).ir_value(), shared.written_blocks.ptr + block
                        )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()

        # Zig-zag owner order balances causal work across the CTA slots.
        for round_index in range(
            fx.Int32(0),
            fx.Int32((owner_count + dq_partitions - 1) // dq_partitions),
            fx.Int32(1),
        ):
            owner = fx.Int32(round_index) * fx.Int32(dq_partitions) + (
                fx.Int32(round_index) % fx.Int32(2) == fx.Int32(0)
            ).select(partial_slot, fx.Int32(dq_partitions - 1) - partial_slot)
            if owner < fx.Int32(owner_count):
                if wave < fx.Int32(producer_waves):
                    process_owner(owner, True)
                else:
                    process_owner(owner, False)

        for index, (kind, _, _) in enumerate(capture_grads):
            if const_expr(kind == CAPTURE_GRAD_INVARIANT):
                if wave < fx.Int32(producer_waves):
                    total = f32(fx.Vector(invariant_sums[index].load())[0])
                    for offset in (32, 16, 8, 4, 2, 1):
                        total = total + f32(fx.gpu.shuffle_xor(total, offset, 64))
                    if tid % fx.Int32(64) == fx.Int32(0):
                        store_element(
                            capture_outputs[index],
                            (batch_head * fx.Int32(dq_partitions) + partial_slot)
                            * fx.Int32(producer_waves)
                            + wave,
                            total,
                        )

    return compute_kernel
