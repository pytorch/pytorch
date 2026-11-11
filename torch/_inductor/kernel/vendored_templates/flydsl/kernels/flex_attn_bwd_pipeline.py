"""Producer waves overlap current QK/dP/dK/dV with two prior-dQ waves."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr

from .flex_attn_bwd_utils import LSE_FLOOR, make_bwd_shared_layout, split_dq_tiles
from .flex_attn_target import WAVE_SIZE
from .flex_attn_utils import (
    make_capture_buffers,
    make_global_view,
    make_mask_evaluator,
    make_score_mod_readers,
    make_tile_score_mod_readers,
    runtime_strides,
    tied_nop,
)


_LOG2E = 1.4426950408889634


def make_pipeline_kernel(context, f32, exp2):
    """Build the block-list producer/consumer pipeline."""
    num_heads = context["num_heads"]
    q_ragged = context["q_ragged"]
    num_kv_heads = context["num_kv_heads"]
    query_group = context["query_group"]
    qk_dim = context["qk_head_dim"]
    v_dim = context["value_head_dim"]
    element_type = context["element_type"]
    num_waves = context["num_waves"]
    lse_in_log2 = context["lse_in_log2"]
    scale = context["softmax_scale"]
    scale_log2 = context["scale_log2"]
    unmasked = context["unmasked_traversal"]
    direct_range = context["direct_range_traversal"]
    block_list = context["block_list_traversal"]
    query_split = context["sparse_q_block_size"] // 16
    sparse_kv_block_size = context["sparse_kv_block_size"]
    mask_buffer_count = context["mask_buffer_count"]
    capture_table = context["capture_table"]
    mask_program = context["mask_program"]
    mask_output = context["mask_program_output"]
    has_score_mod = context["has_score_mod"]
    score_mod = context["score_mod"]
    joint_mod = context["joint_mod"]
    score_mod_aux = context["score_mod_aux"]
    dq_partitions = context["dq_partitions"]
    dq_workspace_dtype = context["dq_workspace_dtype"]
    dq_copy_op = context["dq_copy_op"]
    workspace_head_group = context["workspace_head_group"]
    cta_head_group = context["cta_head_group"]
    dq_accumulates = context["dq_accumulates"]
    direct_dq = context["direct_dq"]
    dq_stride_static = context["static_strides"]["dq"]
    resolve_compute = context["resolve_compute"]
    threads = num_waves * WAVE_SIZE
    key_groups = context["key_groups"]
    producer_waves = context["producer_waves"]
    consumer_waves = 2
    key_rows = context["key_rows"]
    column_slices = context["column_slices"]
    key_slice_chunks = context["key_slice_chunks"]
    value_slice_chunks = context["value_slice_chunks"]
    single_pass = context["single_pass"]
    step_rows = context["step_rows"]
    ds_stride = context["ds_stride"]
    # Query-list entries per sparse Q block, one per producer step.
    step_split = query_split * 16 // step_rows
    # Each producer wave converts its P and dS tiles through its own rows.
    score_rows = producer_waves * key_groups * 16
    # LLVM cannot merge table reads, so these builds read each capture once
    # per row or key of a tile; two producers per SIMD would spill the values.
    tile_capture_reads = capture_table is not None and producer_waves == 2
    if qk_dim % 32:
        raise ValueError(f"QK head dimension must be divisible by 32, got {qk_dim}")
    dq_phases, dq_phase_tiles = split_dq_tiles(qk_dim, consumer_waves)
    dq_wave_tiles = dq_phases * dq_phase_tiles

    @fx.struct
    class SharedMemory:
        key: fx.Array[element_type, key_rows * qk_dim, 16]
        value: fx.Array[element_type, key_rows * v_dim, 16]
        query: fx.Array[element_type, 2 * step_rows * qk_dim, 16]
        grad_output: fx.Array[element_type, 2 * step_rows * v_dim, 16]
        ds: fx.Array[
            element_type,
            2 * key_rows * ds_stride if single_pass else 2 * score_rows * 16,
            16,
        ]
        probability: fx.Array[element_type, score_rows * 16, 16]
        stats: fx.Array[fx.Float32, 2 * 2 * step_rows, 16]

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
        (
            sequence_length,
            padded_sequence_length,
            query_tiles,
            batch_heads,
            kv_sequence_length,
            owner_count,
            key_metadata_chunks,
            metadata_chunks,
            query_shape,
            key_shape,
            q_stride,
            k_stride,
            v_stride,
            do_stride,
            dk_stride,
            dv_stride,
            mask_buffer_strides,
            mask_buffer_sizes,
        ) = resolve_compute(
            query,
            key,
            value,
            grad_output,
            grad_key,
            grad_value,
            (mask_buffer_0, mask_buffer_1, mask_buffer_2, mask_buffer_3),
        )
        if const_expr(direct_dq):
            dq_output_stride = runtime_strides(
                dq_stride_static, grad_query, sequence_length
            )
        # Steps per listed Q block. A static Sq under one sparse block lists
        # only block 0, so steps past its rows are skipped.
        list_split = (
            min(step_split, -(-query_tiles * 16 // step_rows))
            if isinstance(query_tiles, int)
            else step_split
        )
        batch_head = fx.Int32(fx.block_idx.z) * fx.Int32(cta_head_group) + fx.Int32(
            fx.block_idx.x
        )
        partial_slot = fx.Int32(fx.block_idx.y)
        tid = fx.Int32(fx.thread_idx.x)
        wave = fx.Int32(fx.gpu.shuffle_idx(tid // fx.Int32(WAVE_SIZE), 0, WAVE_SIZE))
        shared = fx.SharedAllocator().allocate(SharedMemory).peek()

        def process_owner(
            owner, producer: fx.Constexpr[bool], column_slice: fx.Constexpr[int] = 0
        ):
            key_chunks = range(
                column_slice * key_slice_chunks, (column_slice + 1) * key_slice_chunks
            )
            value_chunks = range(
                column_slice * value_slice_chunks,
                (column_slice + 1) * value_slice_chunks,
            )
            tid = fx.Int32(
                llvm.inline_asm(
                    fx.Int32.ir_type,
                    [fx.Int32(fx.thread_idx.x).ir_value()],
                    "",
                    "=v,0",
                    has_side_effects=True,
                )
            ) & fx.Int32((1 << (threads - 1).bit_length()) - 1)
            lane = tid % fx.Int32(WAVE_SIZE)

            def pin_lane_coordinates():
                # Rebuilt per query head so staging addresses are not hoisted
                # across the group loop and spilled.
                nonlocal tid
                tid = fx.Int32(
                    llvm.inline_asm(
                        fx.Int32.ir_type,
                        [fx.Int32(fx.thread_idx.x).ir_value()],
                        "",
                        "=v,0",
                        has_side_effects=True,
                    )
                ) & fx.Int32((1 << (threads - 1).bit_length()) - 1)

            wave = fx.Int32(
                fx.rocdl.readfirstlane(
                    fx.Int32.ir_type, (tid // fx.Int32(WAVE_SIZE)).ir_value()
                )
            )
            # The producers of one key tile share its rows.
            key_wave = wave
            if const_expr(column_slices > 1):
                key_wave = fx.Int32(0)
            batch = batch_head // fx.Int32(num_kv_heads)
            head = batch_head % fx.Int32(num_kv_heads)
            head_coord = (fx.Int64(batch), fx.Int64(head), None, None)

            def load_i32(view, index):
                return fx.Int32(fx.get_iter(view)[index])

            # K-side views are bound once per owner and Q-side views once per
            # query head of the KV group; the closures below read them when
            # traced. Creation follows one fixed order so MHA codegen is stable.
            kview = vview = dkview = dvview = masks = None
            query_batch_head = qview = doview = dqview = packed_dq = None
            lseview = deltaview = evaluate_mask = None
            mod_head = score_mod_readers = score_mod_table = None

            def bind_views(
                query_head,
                query_head_index,
                query_views: fx.Constexpr[bool],
                key_views: fx.Constexpr[bool],
            ):
                nonlocal kview, vview, dkview, dvview, masks
                nonlocal query_batch_head, qview, doview, dqview, packed_dq
                nonlocal lseview, deltaview, evaluate_mask
                nonlocal mod_head, score_mod_readers, score_mod_table
                if const_expr(query_views):
                    query_batch_head = query_head_index
                    query_coord = (
                        head_coord
                        if const_expr(query_group == 1)
                        else (fx.Int64(batch), fx.Int64(query_head), None, None)
                    )
                    qview = make_global_view(
                        query, query_coord, (*query_shape, qk_dim), q_stride
                    )
                if const_expr(key_views):
                    kview = make_global_view(
                        key, head_coord, (*key_shape, qk_dim), k_stride
                    )
                    vview = make_global_view(
                        value, head_coord, (*key_shape, v_dim), v_stride
                    )
                if const_expr(query_views):
                    doview = make_global_view(
                        grad_output, query_coord, (*query_shape, v_dim), do_stride
                    )
                if const_expr(query_views and direct_dq):
                    dqview = make_global_view(
                        grad_query,
                        query_coord,
                        (*query_shape, qk_dim),
                        dq_output_stride,
                    )
                elif const_expr(query_views):
                    dqview = make_global_view(
                        grad_query,
                        (
                            fx.Int64(
                                query_batch_head // fx.Int32(workspace_head_group)
                            ),
                            None,
                        ),
                        (
                            (batch_heads + workspace_head_group - 1)
                            // workspace_head_group,
                            dq_partitions
                            * padded_sequence_length
                            * qk_dim
                            * workspace_head_group,
                        ),
                        (
                            dq_partitions
                            * padded_sequence_length
                            * qk_dim
                            * workspace_head_group,
                            1,
                        ),
                    )
                if const_expr(key_views):
                    dkview = make_global_view(
                        grad_key, head_coord, (*key_shape, qk_dim), dk_stride
                    )
                    dvview = make_global_view(
                        grad_value, head_coord, (*key_shape, v_dim), dv_stride
                    )
                if const_expr(query_views):
                    lseview = make_global_view(
                        logsumexp,
                        (fx.Int64(query_batch_head), None),
                        (batch_heads, sequence_length),
                        (sequence_length, 1),
                    )
                    deltaview = make_global_view(
                        delta,
                        (fx.Int64(query_batch_head), None),
                        (batch_heads, padded_sequence_length),
                        (padded_sequence_length, 1),
                    )
                if const_expr(key_views):
                    masks, score_mod_buffers = make_capture_buffers(
                        make_global_view,
                        mask_buffer_count,
                        mask_buffer_sizes,
                        capture_table,
                        (mask_buffer_0, mask_buffer_1, mask_buffer_2, mask_buffer_3),
                    )
                    if const_expr(has_score_mod):
                        score_mod_table = score_mod_buffers
                        score_mod_readers = make_score_mod_readers(
                            make_global_view, score_mod_aux, score_mod_buffers
                        )
                if const_expr(query_views):
                    mod_head = query_head
                    evaluate_mask = make_mask_evaluator(
                        mask_program,
                        mask_output,
                        mask_buffer_strides,
                        masks,
                        load_i32,
                        batch,
                        query_head,
                    )
                    if const_expr(query_group > 1 and not direct_dq):
                        packed_dq = fx.logical_divide(dqview, fx.make_layout(4, 1))

            bind_views(head, batch_head, query_group == 1, True)
            g64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), element_type)
            dq_copy = fx.make_copy_atom(dq_copy_op(), dq_workspace_dtype)
            dq_output_copy = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), element_type)
            read128 = fx.make_copy_atom(fx.UniversalCopy128b(), element_type)
            tr16 = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(16, 64), element_type)
            score_mma = fx.make_tiled_mma(
                fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, element_type)),
                fx.make_layout((1, 1, 1), (1, 1, 1)),
            )
            wide_mma = fx.make_tiled_mma(
                fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, element_type)),
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
            write64 = fx.make_copy_atom(fx.UniversalCopy64b(), element_type)
            score_store = fx.make_tiled_copy_C(write64, score_mma).get_slice(lane)
            if const_expr(query_group == 1 and not direct_dq):
                packed_dq = fx.logical_divide(dqview, fx.make_layout(4, 1))
            wide_b_copy = fx.make_tiled_copy_B(read128, wide_mma).get_slice(lane)
            key_base = owner * fx.Int32(key_rows)
            sparse_key = owner
            if const_expr(key_rows < sparse_kv_block_size):
                sparse_key = owner // fx.Int32(sparse_kv_block_size // key_rows)
            key_pos = (
                key_base
                + key_wave * fx.Int32(key_groups * 16)
                + fx.Int32(fx.get_scalar(rowcoords[0]))
            )
            klayout = make_bwd_shared_layout(key_rows, qk_dim)
            vlayout = make_bwd_shared_layout(key_rows, v_dim)
            qlayout = make_bwd_shared_layout(16, qk_dim)
            dolayout = make_bwd_shared_layout(16, v_dim)
            score_layout = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(2, 3, 3)),
                fx.make_layout((16, score_rows), (1, 16)),
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
                row_limit,
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
                        if first_row + row < fx.Int32(row_limit):
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
                            zeros = fx.make_rmem_tensor(8, element_type)
                            zeros.fill(0)
                            fx.copy(read128, zeros, destination)

            def stage_query(query_tile, stage):
                first_row = query_tile * fx.Int32(16)
                if tid < fx.Int32(4):
                    source_index = query_tile * fx.Int32(4) + tid
                    for source_view, stats_offset in [(lseview, 0), (deltaview, 16)][
                        int(q_ragged) :
                    ]:
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
                if const_expr(q_ragged):
                    # A 16-byte load straddling Sq may read as fully out of
                    # bounds; load the unpadded LSE one row per lane.
                    if tid < fx.Int32(16):
                        fx.rocdl.raw_ptr_buffer_load_async_lds(
                            fx.rocdl.get_buffer_rsrc(fx.get_iter(lseview)),
                            fx.to_llvm_ptr(shared.stats.ptr + stage * fx.Int32(32)),
                            fx.Int32(4).ir_value(),
                            fx.Int32((first_row + tid) * fx.Int32(4)).ir_value(),
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
                    producer_waves * WAVE_SIZE,
                    q_stride[2],
                    sequence_length,
                )
                stage_operand(
                    doview,
                    dolayout,
                    shared.grad_output.ptr + stage * fx.Int32(16 * v_dim),
                    first_row,
                    16,
                    v_dim,
                    producer_waves * WAVE_SIZE,
                    do_stride[2],
                    sequence_length,
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
                kv_sequence_length,
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
                kv_sequence_length,
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
                row_tile = key_wave * fx.Int32(key_groups) + fx.Int32(group)
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
                    score_thread.partition_C(tile), element_type
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
                ds_pointer = shared.ds.ptr + stage * fx.Int32(score_rows * 16)
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
                for _ in fx.range_constexpr(dq_wave_tiles)
            ]

            def load_previous_dq_phase(
                query_tile, phase: fx.Constexpr[int], previous=None
            ):
                if const_expr(previous is None):
                    previous = previous_dq
                for d in fx.range_constexpr(dq_phase_tiles if dq_accumulates else 0):
                    part = phase * dq_phase_tiles + d
                    tile = (wave - fx.Int32(producer_waves)) * fx.Int32(
                        dq_wave_tiles
                    ) + fx.Int32(part)
                    offset = (
                        (
                            (query_tile * fx.Int32(dq_partitions) + partial_slot)
                            * fx.Int32(workspace_head_group)
                            + query_batch_head % fx.Int32(workspace_head_group)
                        )
                        * fx.Int32(16 * qk_dim)
                        + tile * fx.Int32(256)
                        + lane * fx.Int32(4)
                    )
                    fx.copy(
                        dq_copy,
                        fx.slice(packed_dq, (None, offset // fx.Int32(4))),
                        previous[part],
                    )

            def load_previous_dq(query_tile):
                for phase in fx.range_constexpr(dq_phases):
                    load_previous_dq_phase(query_tile, phase)

            def store_dq(
                query_tile, accumulator, tile, part: fx.Constexpr[int], previous=None
            ):
                if const_expr(direct_dq):
                    # Lane l holds row l % 16, columns 4 * (l // 16) + 0..3 of
                    # its 16-column tile; dq_layout reads the same order.
                    values = fx.Vector(accumulator.load()) * fx.Vector.filled(
                        4, scale, fx.Float32
                    )
                    output = fx.make_rmem_tensor(4, element_type)
                    output.store(values.to(element_type).ir_value())
                    row = query_tile * fx.Int32(16) + lane % fx.Int32(16)
                    column = tile * fx.Int32(16) + lane // fx.Int32(16) * fx.Int32(4)
                    destination = fx.logical_divide(
                        fx.slice(dqview, (row, None)), fx.make_layout(4, 1)
                    )
                    fx.copy(
                        dq_output_copy,
                        output,
                        fx.slice(destination, (None, column // fx.Int32(4))),
                    )
                    return
                offset = (
                    (
                        (query_tile * fx.Int32(dq_partitions) + partial_slot)
                        * fx.Int32(workspace_head_group)
                        + query_batch_head % fx.Int32(workspace_head_group)
                    )
                    * fx.Int32(16 * qk_dim)
                    + tile * fx.Int32(256)
                    + lane * fx.Int32(4)
                )
                destination = fx.slice(packed_dq, (None, offset // fx.Int32(4)))
                values = fx.Vector(accumulator.load()) * fx.Vector.filled(
                    4, scale, fx.Float32
                )
                if const_expr(dq_accumulates):
                    if const_expr(previous is None):
                        previous = previous_dq
                    values = values + fx.Vector(previous[part].load()).to(fx.Float32)
                output = fx.make_rmem_tensor(4, dq_workspace_dtype)
                output.store(values.to(dq_workspace_dtype).ir_value())
                fx.copy(dq_copy, output, destination)

            def consume(
                query_tile, stage, cached_keys, next_query, prefetch: fx.Constexpr[bool]
            ):
                # Up to two phases keep dQ state small while producer accumulators live.
                for phase in fx.range_constexpr(dq_phases):
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
                            dq_wave_tiles
                        ) + fx.Int32(phase * dq_phase_tiles + d)
                        store_dq(
                            query_tile,
                            accumulators[d],
                            tile,
                            phase * dq_phase_tiles + d,
                        )
                    if const_expr(prefetch):
                        load_previous_dq_phase(next_query, phase)

            def finish_stage(producer: fx.Constexpr[bool]):
                # One common barrier publishes dS(i) and Q/dO(i+1); all waves attend.
                if const_expr(producer):
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                else:
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()

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
                pre_mod_scores = []
                keep_masks = []
                readers = score_mod_readers
                if const_expr(tile_capture_reads):
                    positions = [
                        query_tile * fx.Int32(16)
                        + fx.Int32(fx.get_scalar(ccoords[element]))
                        for element in fx.range_constexpr(4)
                    ]
                    group_keys = [
                        key_pos + fx.Int32(group * 16)
                        for group in fx.range_constexpr(key_groups)
                    ]
                    if const_expr(has_score_mod):
                        readers = make_tile_score_mod_readers(
                            score_mod_aux, score_mod_table
                        )
                    if const_expr(masked):
                        row_masks = [
                            evaluate_mask(position, None, independent_of=3)
                            for position in positions
                        ]
                        key_masks = [
                            evaluate_mask(None, group_key, independent_of=2)
                            for group_key in group_keys
                        ]
                for group in fx.range_constexpr(key_groups):
                    score_values = fx.Vector(scores[group].load())
                    values = []
                    for element in fx.range_constexpr(4):
                        local_row = fx.Int32(fx.get_scalar(ccoords[element]))
                        group_key = cached_mask = None
                        if const_expr(tile_capture_reads):
                            position = positions[element]
                            group_key = group_keys[group]
                            if const_expr(masked):
                                cached_mask = {**row_masks[element], **key_masks[group]}
                        else:
                            position = query_tile * fx.Int32(16) + local_row
                        lse = fx.Float32(
                            shared.stats.ptr[stage * fx.Int32(32) + local_row]
                        )
                        if const_expr(not lse_in_log2):
                            lse = lse * f32(1.4426950408889634)
                        lse = lse.maximumf(f32(LSE_FLOOR))
                        if const_expr(has_score_mod):
                            pre_mod = f32(score_values[element]) * f32(scale)
                            post_mod = f32(
                                score_mod(
                                    pre_mod,
                                    batch,
                                    mod_head,
                                    position,
                                    key_pos + fx.Int32(group * 16)
                                    if group_key is None
                                    else group_key,
                                    readers,
                                )
                            )
                            pre_mod_scores.append(pre_mod)
                            probability = exp2(post_mod * f32(_LOG2E) - lse)
                        else:
                            probability = exp2(
                                f32(score_values[element]) * f32(scale_log2) - lse
                            )
                        if const_expr(producer_waves == 2 or capture_table is not None):
                            probability = tied_nop(probability, 0)
                        elif const_expr(has_score_mod):
                            probability = tied_nop(
                                probability, 0, has_side_effects=False
                            )
                        keep = None
                        if masked:
                            keep = evaluate_mask(
                                position,
                                key_pos + fx.Int32(group * 16)
                                if group_key is None
                                else group_key,
                                cached_mask,
                            )
                            probability = keep.select(probability, f32(0.0))
                        if const_expr(not direct_range):
                            in_bounds = key_pos + fx.Int32(group * 16) < fx.Int32(
                                kv_sequence_length
                            )
                            probability = in_bounds.select(probability, f32(0.0))
                            if const_expr(has_score_mod):
                                keep = in_bounds if keep is None else keep & in_bounds
                        if const_expr(q_ragged):
                            in_rows = position < fx.Int32(sequence_length)
                            probability = in_rows.select(probability, f32(0.0))
                            if const_expr(has_score_mod):
                                keep = in_rows if keep is None else keep & in_rows
                        keep_masks.append(keep)
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
                        ds_value = f32(probability_values[element]) * (
                            f32(dp_values[element]) - delta_value
                        )
                        if const_expr(has_score_mod and tile_capture_reads):
                            ds_value = f32(
                                joint_mod(
                                    pre_mod_scores[group * 4 + element],
                                    batch,
                                    mod_head,
                                    positions[element],
                                    group_keys[group],
                                    ds_value,
                                    readers,
                                )
                            )
                        elif const_expr(has_score_mod):
                            ds_value = f32(
                                joint_mod(
                                    pre_mod_scores[group * 4 + element],
                                    batch,
                                    mod_head,
                                    query_tile * fx.Int32(16) + local_row,
                                    key_pos + fx.Int32(group * 16),
                                    ds_value,
                                    score_mod_readers,
                                )
                            )
                        if const_expr(has_score_mod):
                            keep = keep_masks[group * 4 + element]
                            if const_expr(keep is not None):
                                ds_value = keep.select(ds_value, f32(0.0))
                        ds_values.append(ds_value)
                    store_score(
                        shared.probability.ptr,
                        group,
                        probability_values.to(element_type),
                    )
                    ds_vector = fx.Vector.from_elements(ds_values, fx.Float32).to(
                        element_type
                    )
                    store_score(
                        shared.ds.ptr + stage * fx.Int32(score_rows * 16),
                        group,
                        ds_vector,
                    )
                # Each wave converts its own 32-key tile through LDS. No other
                # wave reads probability; dS is published by finish_stage.
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                ds_wide = load_wide_score(
                    shared.ds.ptr + stage * fx.Int32(score_rows * 16)
                )
                p_wide = load_wide_score(shared.probability.ptr)
                qa = load_wide(qpointer, qlayout, key_chunks[0])
                for d in fx.range_constexpr(
                    min(key_chunks[0], value_chunks[0]),
                    max(key_chunks[-1], value_chunks[-1]) + 1,
                ):
                    prefetch = d + 1 in key_chunks[1:]
                    if const_expr(prefetch):
                        next_qa = load_wide(qpointer, qlayout, d + 1)
                    if const_expr(d in value_chunks):
                        doa = load_wide(dopointer, dolayout, d)
                    if const_expr(d in key_chunks):
                        fx.gemm(wide_mma, dk_wide[d], qa, ds_wide, dk_wide[d])
                    if const_expr(d in value_chunks):
                        fx.gemm(wide_mma, dv_wide[d], doa, p_wide, dv_wide[d])
                    if const_expr(prefetch):
                        qa = next_qa

            if const_expr(single_pass):
                query_pair_layout = make_bwd_shared_layout(32, qk_dim)
                grad_output_pair_layout = make_bwd_shared_layout(32, v_dim)
                # Row m of score tile s reads Q row 8 * (m // 4) + 4 * s + m % 4
                # of the pair, so the two tiles' accumulator rows concatenate
                # into the 32-row reduction operand of the dK/dV MFMAs.
                pair_rows = fx.make_layout((4, 4, 2), (1, 8, 4))
                query_score_layout = fx.composition(
                    query_pair_layout,
                    fx.make_tile(pair_rows, fx.make_layout(qk_dim, 1)),
                )
                grad_output_score_layout = fx.composition(
                    grad_output_pair_layout,
                    fx.make_tile(pair_rows, fx.make_layout(v_dim, 1)),
                )
                operand_template = fx.slice(
                    fx.flat_divide(fx.make_view(shared.key.ptr, klayout), (16, 32)),
                    (None, None, 0, 0),
                )
                previous_pair = [
                    previous_dq,
                    [
                        fx.make_rmem_tensor(4, dq_workspace_dtype)
                        for _ in fx.range_constexpr(dq_wave_tiles)
                    ],
                ]

            def stage_query_pair(pair, stage):
                first_row = pair * fx.Int32(32)
                stats = shared.stats.ptr + stage * fx.Int32(64)
                if tid < fx.Int32(8):
                    source_index = pair * fx.Int32(8) + tid
                    for source_view, stats_offset in [(lseview, 0), (deltaview, 32)][
                        int(q_ragged) :
                    ]:
                        fx.rocdl.raw_ptr_buffer_load_async_lds(
                            fx.rocdl.get_buffer_rsrc(fx.get_iter(source_view)),
                            fx.to_llvm_ptr(stats + fx.Int32(stats_offset)),
                            fx.Int32(16).ir_value(),
                            fx.Int32(source_index * fx.Int32(16)).ir_value(),
                            fx.Int32(0).ir_value(),
                            fx.Int32(0).ir_value(),
                        )
                if const_expr(q_ragged):
                    if tid < fx.Int32(32):
                        fx.rocdl.raw_ptr_buffer_load_async_lds(
                            fx.rocdl.get_buffer_rsrc(fx.get_iter(lseview)),
                            fx.to_llvm_ptr(stats),
                            fx.Int32(4).ir_value(),
                            fx.Int32((first_row + tid) * fx.Int32(4)).ir_value(),
                            fx.Int32(0).ir_value(),
                            fx.Int32(0).ir_value(),
                        )
                stage_operand(
                    qview,
                    query_pair_layout,
                    shared.query.ptr + stage * fx.Int32(32 * qk_dim),
                    first_row,
                    32,
                    qk_dim,
                    producer_waves * WAVE_SIZE,
                    q_stride[2],
                    sequence_length,
                )
                stage_operand(
                    doview,
                    grad_output_pair_layout,
                    shared.grad_output.ptr + stage * fx.Int32(32 * v_dim),
                    first_row,
                    32,
                    v_dim,
                    producer_waves * WAVE_SIZE,
                    do_stride[2],
                    sequence_length,
                )

            def pair_row(local_row, tile: fx.Constexpr[int]):
                row = local_row + local_row // fx.Int32(4) * fx.Int32(4)
                if const_expr(tile):
                    row = row + fx.Int32(4 * tile)
                return row

            def load_pair_score(pointer, layout, step, tile: fx.Constexpr[int]):
                tiles = fx.flat_divide(fx.make_view(pointer, layout), (16, 32))
                operand = fx.slice(tiles, (None, None, tile, step))
                fragment = score_thread.make_fragment_A(operand)
                fx.copy(
                    read128,
                    score_copy.partition_S(operand),
                    score_copy.retile(fragment),
                )
                return fragment

            def load_pair_transposed(pointer, layout, dim_tile: fx.Constexpr[int]):
                tiles = fx.flat_divide(
                    fx.make_view(pointer, fx.select(layout, [1, 0])), (16, 32)
                )
                operand = fx.slice(tiles, (None, None, dim_tile, 0))
                fragment = score_thread.make_fragment_A(operand)
                fx.copy(
                    tr16, dq_key_copy.partition_S(operand), dq_key_copy.retile(fragment)
                )
                return fragment

            def produce_pair(
                pair, stage, masked: fx.Constexpr[bool], dk_tiles, dv_tiles
            ):
                qpointer = shared.query.ptr + stage * fx.Int32(32 * qk_dim)
                dopointer = shared.grad_output.ptr + stage * fx.Int32(32 * v_dim)
                stats = stage * fx.Int32(64)
                scores = [
                    fx.make_fragment_like(ccoords, fx.Float32)
                    for _ in fx.range_constexpr(2)
                ]
                dps = [
                    fx.make_fragment_like(ccoords, fx.Float32)
                    for _ in fx.range_constexpr(2)
                ]
                for tile in fx.range_constexpr(2):
                    scores[tile].fill(0)
                    dps[tile].fill(0)
                rows = [
                    [
                        pair_row(fx.Int32(fx.get_scalar(ccoords[element])), tile)
                        for element in fx.range_constexpr(4)
                    ]
                    for tile in fx.range_constexpr(2)
                ]
                keys = load_score_operand(shared.key.ptr, klayout, 0, 0)
                query_operands = [
                    load_pair_score(qpointer, query_score_layout, 0, tile)
                    for tile in fx.range_constexpr(2)
                ]
                for step in fx.range_constexpr(qk_dim // 32):
                    if const_expr(step + 1 < qk_dim // 32):
                        next_keys = load_score_operand(
                            shared.key.ptr, klayout, step + 1, 0
                        )
                        next_queries = [
                            load_pair_score(
                                qpointer, query_score_layout, step + 1, tile
                            )
                            for tile in fx.range_constexpr(2)
                        ]
                    for tile in fx.range_constexpr(2):
                        fx.gemm(
                            score_mma,
                            scores[tile],
                            query_operands[tile],
                            keys,
                            scores[tile],
                        )
                    if const_expr(step + 1 < qk_dim // 32):
                        keys = next_keys
                        query_operands = next_queries
                values_operand = load_score_operand(shared.value.ptr, vlayout, 0, 0)
                grad_output_operands = [
                    load_pair_score(dopointer, grad_output_score_layout, 0, tile)
                    for tile in fx.range_constexpr(2)
                ]
                positions = [
                    pair * fx.Int32(32) + rows[tile][element]
                    for tile in fx.range_constexpr(2)
                    for element in fx.range_constexpr(4)
                ]
                pre_mod_scores = []
                keep_masks = []
                probabilities = []
                readers = score_mod_readers
                if const_expr(tile_capture_reads):
                    if const_expr(has_score_mod):
                        readers = make_tile_score_mod_readers(
                            score_mod_aux, score_mod_table
                        )
                    if const_expr(masked):
                        row_masks = [
                            evaluate_mask(position, None, independent_of=3)
                            for position in positions
                        ]
                        key_masks = evaluate_mask(None, key_pos, independent_of=2)
                for tile in fx.range_constexpr(2):
                    score_values = fx.Vector(scores[tile].load())
                    for element in fx.range_constexpr(4):
                        index = tile * 4 + element
                        position = positions[index]
                        cached_mask = None
                        if const_expr(tile_capture_reads and masked):
                            cached_mask = {**row_masks[index], **key_masks}
                        lse = fx.Float32(shared.stats.ptr[stats + rows[tile][element]])
                        if const_expr(not lse_in_log2):
                            lse = lse * f32(_LOG2E)
                        lse = lse.maximumf(f32(LSE_FLOOR))
                        if const_expr(has_score_mod):
                            pre_mod = f32(score_values[element]) * f32(scale)
                            post_mod = f32(
                                score_mod(
                                    pre_mod, batch, mod_head, position, key_pos, readers
                                )
                            )
                            pre_mod_scores.append(pre_mod)
                            probability = exp2(post_mod * f32(_LOG2E) - lse)
                        else:
                            probability = exp2(
                                f32(score_values[element]) * f32(scale_log2) - lse
                            )
                        probability = tied_nop(probability, 0)
                        keep = None
                        if masked:
                            keep = evaluate_mask(position, key_pos, cached_mask)
                            probability = keep.select(probability, f32(0.0))
                        if const_expr(not direct_range):
                            in_bounds = key_pos < fx.Int32(kv_sequence_length)
                            probability = in_bounds.select(probability, f32(0.0))
                            if const_expr(has_score_mod):
                                keep = in_bounds if keep is None else keep & in_bounds
                        if const_expr(q_ragged):
                            in_rows = position < fx.Int32(sequence_length)
                            probability = in_rows.select(probability, f32(0.0))
                            if const_expr(has_score_mod):
                                keep = in_rows if keep is None else keep & in_rows
                        keep_masks.append(keep)
                        probabilities.append(probability)
                for step in fx.range_constexpr(v_dim // 32):
                    if const_expr(step + 1 < v_dim // 32):
                        next_values = load_score_operand(
                            shared.value.ptr, vlayout, step + 1, 0
                        )
                        next_grad_outputs = [
                            load_pair_score(
                                dopointer, grad_output_score_layout, step + 1, tile
                            )
                            for tile in fx.range_constexpr(2)
                        ]
                    for tile in fx.range_constexpr(2):
                        fx.gemm(
                            score_mma,
                            dps[tile],
                            grad_output_operands[tile],
                            values_operand,
                            dps[tile],
                        )
                    if const_expr(step + 1 < v_dim // 32):
                        values_operand = next_values
                        grad_output_operands = next_grad_outputs
                ds_values = []
                for tile in fx.range_constexpr(2):
                    dp_values = fx.Vector(dps[tile].load())
                    for element in fx.range_constexpr(4):
                        index = tile * 4 + element
                        delta_value = fx.Float32(
                            shared.stats.ptr[stats + fx.Int32(32) + rows[tile][element]]
                        )
                        ds_value = probabilities[index] * (
                            f32(dp_values[element]) - delta_value
                        )
                        if const_expr(has_score_mod):
                            ds_value = f32(
                                joint_mod(
                                    pre_mod_scores[index],
                                    batch,
                                    mod_head,
                                    positions[index],
                                    key_pos,
                                    ds_value,
                                    readers,
                                )
                            )
                            keep = keep_masks[index]
                            if const_expr(keep is not None):
                                ds_value = keep.select(ds_value, f32(0.0))
                        ds_values.append(ds_value)
                ds_vector = fx.Vector.from_elements(ds_values, fx.Float32).to(
                    element_type
                )
                fx.ptr_store(
                    ds_vector.ir_value(),
                    shared.ds.ptr
                    + stage * fx.Int32(key_rows * ds_stride)
                    + (key_wave * fx.Int32(16) + fx.Int32(fx.get_scalar(rowcoords[0])))
                    * fx.Int32(ds_stride)
                    + fx.Int32(fx.get_scalar(ccoords[0])) * fx.Int32(2),
                )
                ds_operand = score_thread.make_fragment_B(operand_template)
                ds_operand.store(
                    tied_nop(ds_vector, 1, has_side_effects=False).ir_value()
                )
                p_operand = score_thread.make_fragment_B(operand_template)
                p_operand.store(
                    tied_nop(
                        fx.Vector.from_elements(probabilities, fx.Float32).to(
                            element_type
                        ),
                        1,
                        has_side_effects=False,
                    ).ir_value()
                )
                for dim_tile in fx.range_constexpr(qk_dim // 16):
                    operand = load_pair_transposed(
                        qpointer, query_pair_layout, dim_tile
                    )
                    fx.gemm(
                        score_mma,
                        dk_tiles[dim_tile],
                        operand,
                        ds_operand,
                        dk_tiles[dim_tile],
                    )
                for dim_tile in fx.range_constexpr(v_dim // 16):
                    operand = load_pair_transposed(
                        dopointer, grad_output_pair_layout, dim_tile
                    )
                    fx.gemm(
                        score_mma,
                        dv_tiles[dim_tile],
                        operand,
                        p_operand,
                        dv_tiles[dim_tile],
                    )

            def consume_pair(
                pair, stage, cached_keys, next_pair, prefetch: fx.Constexpr[bool]
            ):
                for tile in fx.range_constexpr(2):
                    ds_tiles = fx.flat_divide(
                        fx.make_view(
                            shared.ds.ptr
                            + stage * fx.Int32(key_rows * ds_stride)
                            + fx.Int32(16 * tile),
                            fx.make_layout((16, key_rows), (1, ds_stride)),
                        ),
                        (16, 32),
                    )
                    ds_operands = []
                    for step in fx.range_constexpr(key_rows // 32):
                        ds_tile = fx.slice(ds_tiles, (None, None, 0, step))
                        ds_fragment = score_thread.make_fragment_B(ds_tile)
                        fx.copy(
                            tr16,
                            dq_ds_copy.partition_S(ds_tile),
                            dq_ds_copy.retile(ds_fragment),
                        )
                        ds_operands.append(ds_fragment)
                    query_tile = pair * fx.Int32(2) + fx.Int32(tile)
                    for phase in fx.range_constexpr(dq_phases):
                        accumulators = [
                            fx.make_fragment_like(ccoords, fx.Float32)
                            for _ in fx.range_constexpr(dq_phase_tiles)
                        ]
                        for d in fx.range_constexpr(dq_phase_tiles):
                            accumulators[d].fill(0)
                        for step in fx.range_constexpr(key_rows // 32):
                            for d in fx.range_constexpr(dq_phase_tiles):
                                fx.gemm(
                                    score_mma,
                                    accumulators[d],
                                    cached_keys[phase][step][d],
                                    ds_operands[step],
                                    accumulators[d],
                                )
                        for d in fx.range_constexpr(dq_phase_tiles):
                            store_dq(
                                query_tile,
                                accumulators[d],
                                (wave - fx.Int32(producer_waves))
                                * fx.Int32(dq_wave_tiles)
                                + fx.Int32(phase * dq_phase_tiles + d),
                                phase * dq_phase_tiles + d,
                                previous_pair[tile],
                            )
                        if const_expr(prefetch):
                            load_previous_dq_phase(
                                next_pair * fx.Int32(2) + fx.Int32(tile),
                                phase,
                                previous_pair[tile],
                            )

            def load_previous_dq_pair(pair):
                for tile in fx.range_constexpr(2):
                    for phase in fx.range_constexpr(dq_phases):
                        load_previous_dq_phase(
                            pair * fx.Int32(2) + fx.Int32(tile),
                            phase,
                            previous_pair[tile],
                        )

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
                            fx.Int64(query_batch_head) * fx.Int64(key_metadata_chunks)
                            + fx.Int64(sparse_key)
                        ) * fx.Int64(metadata_chunks)
                        sparse_query = fx.Int32(
                            fx.get_iter(query_indices)[
                                list_offset + fx.Int64(index // fx.Int32(list_split))
                            ]
                        )
                        return sparse_query * fx.Int32(step_split) + index % fx.Int32(
                            list_split
                        )
                    else:
                        return first_query + index

                stage_step, produce_step, consume_step, load_previous = (
                    (
                        stage_query_pair,
                        produce_pair,
                        consume_pair,
                        load_previous_dq_pair,
                    )
                    if const_expr(single_pass)
                    else (stage_query, produce, consume, load_previous_dq)
                )
                if count > fx.Int32(0):
                    if const_expr(producer):
                        stage_step(query_at(fx.Int32(0)), fx.Int32(0))
                    else:
                        load_previous(query_at(fx.Int32(0)))
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    fx.gpu.barrier()
                    if const_expr(producer):
                        if count > fx.Int32(1):
                            stage_step(query_at(fx.Int32(1)), fx.Int32(1))
                        produce_step(
                            query_at(fx.Int32(0)), fx.Int32(0), masked, dk_wide, dv_wide
                        )
                    finish_stage(producer)
                    for index in range(fx.Int32(1), count, fx.Int32(1)):
                        stage = index % fx.Int32(2)
                        if const_expr(producer):
                            if index + fx.Int32(1) < count:
                                stage_step(
                                    query_at(index + fx.Int32(1)), fx.Int32(1) - stage
                                )
                            produce_step(
                                query_at(index), stage, masked, dk_wide, dv_wide
                            )
                        else:
                            consume_step(
                                query_at(index - fx.Int32(1)),
                                fx.Int32(1) - stage,
                                cached_keys,
                                query_at(index),
                                True,
                            )
                        finish_stage(producer)
                    if const_expr(not producer):
                        consume_step(
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
                        fx.Int32(query_tiles),
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
                        fx.Int32(query_tiles) - first_query,
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
                        fx.Int32(query_tiles) - first_query - masked_count,
                        first_query + masked_count,
                        full_kv_indices_transposed,
                        False,
                        producer,
                        dk_wide,
                        dv_wide,
                        cached_keys,
                    )

                else:
                    count_index = fx.Int64(query_batch_head) * fx.Int64(
                        key_metadata_chunks
                    ) + fx.Int64(sparse_key)
                    partial_count = fx.Int32(
                        fx.get_iter(partial_kv_counts)[count_index]
                    )
                    full_count = fx.Int32(fx.get_iter(full_kv_counts)[count_index])
                    if const_expr(direct_dq and not producer):
                        if partial_count + full_count == fx.Int32(0):
                            zero = fx.make_fragment_like(ccoords, fx.Float32)
                            zero.fill(0)
                            for query_tile in fx.range_constexpr(query_tiles):
                                for part in fx.range_constexpr(dq_wave_tiles):
                                    tile = (wave - fx.Int32(producer_waves)) * fx.Int32(
                                        dq_wave_tiles
                                    ) + fx.Int32(part)
                                    store_dq(fx.Int32(query_tile), zero, tile, part)
                    run(
                        partial_count * fx.Int32(list_split),
                        fx.Int32(0),
                        partial_kv_indices,
                        True,
                        producer,
                        dk_wide,
                        dv_wide,
                        cached_keys,
                    )
                    run(
                        full_count * fx.Int32(list_split),
                        fx.Int32(0),
                        full_kv_indices_transposed,
                        False,
                        producer,
                        dk_wide,
                        dv_wide,
                        cached_keys,
                    )

            def execute_query_heads(
                producer: fx.Constexpr[bool], dk_wide, dv_wide, cached_keys
            ):
                if const_expr(query_group == 1):
                    execute_runs(producer, dk_wide, dv_wide, cached_keys)
                else:
                    # K/V stay staged while every Q head of the group walks its
                    # own reverse lists into the same dK/dV registers.
                    for group_head in range(
                        fx.Int32(0), fx.Int32(query_group), fx.Int32(1)
                    ):
                        pin_lane_coordinates()
                        query_head = head * fx.Int32(query_group) + group_head
                        bind_views(
                            query_head,
                            batch * fx.Int32(num_heads) + query_head,
                            True,
                            False,
                        )
                        execute_runs(producer, dk_wide, dv_wide, cached_keys)

            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            fx.gpu.barrier()

            def cache_consumer_keys():
                key_tiles = fx.flat_divide(
                    fx.make_view(shared.key.ptr, fx.select(klayout, [1, 0])), (16, 32)
                )
                cached_keys = []
                for phase in fx.range_constexpr(dq_phases):
                    phase_keys = []
                    for step in fx.range_constexpr(key_rows // 32):
                        key_fragments = []
                        for d in fx.range_constexpr(dq_phase_tiles):
                            dimension_tile = (
                                wave - fx.Int32(producer_waves)
                            ) * fx.Int32(dq_wave_tiles) + fx.Int32(
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

            if const_expr(producer and single_pass):
                dk_tiles = [
                    fx.make_fragment_like(ccoords, fx.Float32)
                    for _ in fx.range_constexpr(qk_dim // 16)
                ]
                dv_tiles = [
                    fx.make_fragment_like(ccoords, fx.Float32)
                    for _ in fx.range_constexpr(v_dim // 16)
                ]
                for gradient_tile in dk_tiles + dv_tiles:
                    gradient_tile.fill(0)
                execute_query_heads(True, dk_tiles, dv_tiles, [])
                gradient_column = fx.Int32(fx.get_scalar(ccoords[0]))
                if key_pos < fx.Int32(kv_sequence_length):
                    for gradient_tiles, gradient_view, gradient_scale in (
                        (dk_tiles, dkview, scale),
                        (dv_tiles, dvview, None),
                    ):
                        gradient_row = fx.logical_divide(
                            fx.slice(gradient_view, (key_pos, None)),
                            fx.make_layout(4, 1),
                        )
                        for dim_tile in fx.range_constexpr(len(gradient_tiles)):
                            gradient_values = fx.Vector(gradient_tiles[dim_tile].load())
                            if const_expr(gradient_scale is not None):
                                gradient_values = gradient_values * fx.Vector.filled(
                                    4, gradient_scale, fx.Float32
                                )
                            gradient_output = fx.make_rmem_tensor(4, element_type)
                            gradient_output.store(
                                gradient_values.to(element_type).ir_value()
                            )
                            fx.copy(
                                g64,
                                gradient_output,
                                fx.slice(
                                    gradient_row,
                                    (
                                        None,
                                        (fx.Int32(dim_tile * 16) + gradient_column)
                                        // fx.Int32(4),
                                    ),
                                ),
                            )
            elif const_expr(producer):
                dk_wide = [
                    fx.make_fragment_like(wide_dim_coords, fx.Float32)
                    if d in key_chunks
                    else None
                    for d in fx.range_constexpr(qk_dim // 32)
                ]
                dv_wide = [
                    fx.make_fragment_like(wide_dim_coords, fx.Float32)
                    if d in value_chunks
                    else None
                    for d in fx.range_constexpr(v_dim // 32)
                ]
                for d in key_chunks:
                    dk_wide[d].fill(0)
                for d in value_chunks:
                    dv_wide[d].fill(0)
                execute_query_heads(True, dk_wide, dv_wide, [])
                wide_key_pos = (
                    key_base
                    + key_wave * fx.Int32(key_groups * 16)
                    + fx.Int32(fx.get_scalar(wide_row_coords[0]))
                )
                if wide_key_pos < fx.Int32(kv_sequence_length):
                    dkrow = fx.logical_divide(
                        fx.slice(dkview, (wide_key_pos, None)), fx.make_layout(4, 1)
                    )
                    dvrow = fx.logical_divide(
                        fx.slice(dvview, (wide_key_pos, None)), fx.make_layout(4, 1)
                    )
                    for d in key_chunks:
                        values = (
                            fx.Vector(dk_wide[d].load())
                            * fx.Vector.filled(16, scale, fx.Float32)
                        ).to(element_type)
                        for part in fx.range_constexpr(4):
                            output = fx.make_rmem_tensor(4, element_type)
                            output.store(
                                fx.Vector.from_elements(
                                    [
                                        values[part * 4 + e]
                                        for e in fx.range_constexpr(4)
                                    ],
                                    element_type,
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
                    for d in value_chunks:
                        values = fx.Vector(dv_wide[d].load()).to(element_type)
                        for part in fx.range_constexpr(4):
                            output = fx.make_rmem_tensor(4, element_type)
                            output.store(
                                fx.Vector.from_elements(
                                    [
                                        values[part * 4 + e]
                                        for e in fx.range_constexpr(4)
                                    ],
                                    element_type,
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
                execute_query_heads(False, [], [], cached_keys)

            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            fx.gpu.barrier()

        for owner in range(
            partial_slot, fx.Int32(owner_count), fx.Int32(dq_partitions)
        ):
            if wave < fx.Int32(producer_waves):
                if const_expr(column_slices > 1):
                    if wave == fx.Int32(0):
                        process_owner(owner, True, 0)
                    else:
                        process_owner(owner, True, 1)
                else:
                    process_owner(owner, True)
            else:
                process_owner(owner, False)

    return compute_kernel
