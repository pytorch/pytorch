"""BF16 backward for gfx950, DQK128/192 and DV128.

The launcher computes delta and zeros a BF16 dQ workspace, runs the
cross-iteration MFMA pipeline, then scatters dQ into its requested layout.
Dense masks use the register-tuned producer/consumer kernel over contiguous
query ranges. Block masks use the shared pipeline over transposed partial/full
query lists. Mask semantics select the kernel and tile geometry.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr

from .flex_attn_bwd_pipeline import make_pipeline_kernel
from .flex_attn_utils import (
    DIRECT_RANGE_CAUSAL,
    MASK_TRAVERSAL_BLOCK_LIST,
    MASK_TRAVERSAL_DIRECT_RANGE,
    MASK_TRAVERSAL_UNMASKED,
    classify_mask_traversal,
    fast_exp2,
    make_global_view,
    make_mask_buffers,
    make_mask_evaluator,
    make_value_shared_layout,
)

_LOG2E = 1.4426950408889634


def _f32(x):
    return fx.Float32(x)


def build_flex_attn_bwd_module(
    batch_size,
    num_heads,
    sequence_length,
    qk_head_dim,
    value_head_dim,
    dtype_str,
    sparse_q_block_size,
    sparse_kv_block_size,
    *,
    scale=None,
    max_partial_blocks=None,
    max_full_blocks=None,
    block_mask_batch=1,
    block_mask_heads=1,
    mask_program=(),
    mask_program_output=0,
    mask_buffer_shapes=(),
    mask_buffer_strides=(),
    q_stride=None,
    k_stride=None,
    v_stride=None,
    out_stride=None,
    do_stride=None,
    dq_stride=None,
    dk_stride=None,
    dv_stride=None,
    lse_in_log2=False,
):
    # Keep standalone entry-point validation even though Inductor checks these
    # constraints before registering the vendored kernel.
    if dtype_str != "bf16":
        raise ValueError(f"unsupported dtype {dtype_str}")
    if (qk_head_dim, value_head_dim) not in ((128, 128), (192, 128)):
        raise ValueError("FlyDSL backward requires (qk_head_dim, value_head_dim) to be (128, 128) or (192, 128)")
    sparse_q_block_size = int(sparse_q_block_size)
    sparse_kv_block_size = int(sparse_kv_block_size)
    if sparse_q_block_size != 128 or sparse_kv_block_size != 128:
        raise ValueError("FlyDSL backward requires a 128x128 sparse block")
    if sequence_length <= 0 or sequence_length % sparse_q_block_size:
        raise ValueError("FlyDSL backward requires sequence_length to be divisible by sparse_q_block_size")
    if block_mask_batch not in (1, batch_size):
        raise ValueError("BlockMask batch dimension must be 1 or batch_size")
    if block_mask_heads not in (1, num_heads):
        raise ValueError("MHA backward BlockMask head dimension must be 1 or num_heads")
    if len(mask_buffer_shapes) != len(mask_buffer_strides):
        raise ValueError("mask buffer shape/stride descriptors must match")
    if len(mask_buffer_shapes) > 4:
        raise ValueError("FlyDSL backward supports at most four mask buffers")
    mask_program = tuple(mask_program)
    mask_program_output = int(mask_program_output)
    mask_buffer_shapes = tuple((tuple(shape) for shape in mask_buffer_shapes))
    mask_buffer_strides = tuple((tuple(stride) for stride in mask_buffer_strides))
    mask_traversal, direct_range_kind = classify_mask_traversal(
        mask_program, mask_program_output, mask_buffer_shapes,
        sequence_length=sequence_length,
    )
    mask_buffer_count = len(mask_buffer_shapes)
    if mask_traversal not in (
        MASK_TRAVERSAL_UNMASKED,
        MASK_TRAVERSAL_DIRECT_RANGE,
        MASK_TRAVERSAL_BLOCK_LIST,
    ):
        raise ValueError(f"unsupported mask traversal {mask_traversal}")
    if mask_traversal == MASK_TRAVERSAL_DIRECT_RANGE and direct_range_kind != DIRECT_RANGE_CAUSAL:
        raise ValueError(f"unsupported direct range {direct_range_kind}")
    unmasked_traversal = mask_traversal == MASK_TRAVERSAL_UNMASKED
    direct_range_traversal = mask_traversal == MASK_TRAVERSAL_DIRECT_RANGE
    block_list_traversal = mask_traversal == MASK_TRAVERSAL_BLOCK_LIST
    num_waves = 6 if block_list_traversal else 8
    key_groups = 2
    producer_waves = num_waves - 2
    key_rows = producer_waves * key_groups * 16
    # Sparse owners divide metadata blocks exactly; dense owners need no lists.
    if block_list_traversal and sparse_kv_block_size % key_rows:
        raise ValueError("KV owner tile must divide the sparse KV block")
    num_sparse_blocks = sequence_length // sparse_q_block_size
    metadata_chunks = sequence_length // sparse_q_block_size
    max_partial_blocks_limit = num_sparse_blocks if max_partial_blocks is None else int(max_partial_blocks)
    max_full_blocks_limit = num_sparse_blocks if max_full_blocks is None else int(max_full_blocks)
    compute_threads = 256
    softmax_scale = float(qk_head_dim) ** (-0.5) if scale is None else float(scale)
    scale_log2 = softmax_scale * _LOG2E
    lse_in_log2 = bool(lse_in_log2)
    block_mask_batch = int(block_mask_batch)
    block_mask_heads = int(block_mask_heads)
    mask_buffer_sizes = tuple(
        (
            1 + sum(((size - 1) * stride for (size, stride) in zip(shape, strides)))
            for (shape, strides) in zip(mask_buffer_shapes, mask_buffer_strides)
        )
    )

    def contiguous_stride(dim):
        return (num_heads * sequence_length * dim, sequence_length * dim, dim, 1)

    q_stride = tuple(q_stride or contiguous_stride(qk_head_dim))
    k_stride = tuple(k_stride or contiguous_stride(qk_head_dim))
    v_stride = tuple(v_stride or contiguous_stride(value_head_dim))
    out_stride = tuple(out_stride or contiguous_stride(value_head_dim))
    grad_output_stride = tuple(do_stride or contiguous_stride(value_head_dim))
    grad_query_stride = tuple(dq_stride or contiguous_stride(qk_head_dim))
    grad_query_workspace_stride = contiguous_stride(qk_head_dim)
    grad_key_stride = tuple(dk_stride or contiguous_stride(qk_head_dim))
    grad_value_stride = tuple(dv_stride or contiguous_stride(value_head_dim))
    heads_are_inner = all(
        stride[1] < stride[2]
        for stride in (
            q_stride,
            k_stride,
            v_stride,
            out_stride,
            grad_output_stride,
            grad_query_stride,
            grad_key_stride,
            grad_value_stride,
        )
    )
    delta_slice_bytes = tuple(
        (1 + sum((size - 1) * stride for size, stride in zip(
            (batch_size, num_heads, sequence_length, value_head_dim), strides,
        ))) * 2
        for strides in (out_stride, grad_output_stride)
    )
    dense_linear_delta = not block_list_traversal and all(
        stride % 8 == 0 for strides in (out_stride, grad_output_stride) for stride in strides[:-1]
    ) and max(
        *delta_slice_bytes, batch_size * num_heads * sequence_length * qk_head_dim * 2,
    ) <= 0x7FFFFFFF
    batch_heads = batch_size * num_heads
    delta_packs = value_head_dim // 8
    lanes_per_row = delta_packs if delta_packs <= 64 and delta_packs & delta_packs - 1 == 0 else 4
    delta_load_iterations = delta_packs // lanes_per_row
    delta_rows_per_block = compute_threads // lanes_per_row
    delta_grid = sequence_length // delta_rows_per_block
    delta_shuffle_offsets = []
    _s = 1
    while _s < lanes_per_row:
        delta_shuffle_offsets.append(_s)
        _s *= 2

    @fx.struct
    class MaskFlags:
        flags: fx.Array[fx.Int32, num_sparse_blocks, 16]

    @flyc.kernel
    def delta_kernel(
        attention_output: fx.Tensor, grad_output: fx.Tensor, delta: fx.Tensor, grad_query: fx.Tensor,
        kv_num_blocks: fx.Tensor, kv_indices: fx.Tensor, full_kv_num_blocks: fx.Tensor, full_kv_indices: fx.Tensor,
        partial_counts: fx.Tensor, partial_indices: fx.Tensor, full_counts: fx.Tensor, full_indices: fx.Tensor,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        chunk = tid % fx.Int32(lanes_per_row)
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        if const_expr(dense_linear_delta):
            output_packs = fx.logical_divide(fx.make_view(fx.get_iter(make_global_view(attention_output, None, (batch_size, num_heads, sequence_length, value_head_dim), out_stride)), fx.make_layout(batch_heads * sequence_length * value_head_dim, 1)), fx.make_layout(8, 1))
            grad_packs = fx.logical_divide(fx.make_view(fx.get_iter(make_global_view(grad_output, None, (batch_size, num_heads, sequence_length, value_head_dim), grad_output_stride)), fx.make_layout(batch_heads * sequence_length * value_head_dim, 1)), fx.make_layout(8, 1))
            zero_packs = fx.logical_divide(make_global_view(grad_query, None, (batch_heads * sequence_length * qk_head_dim,), (1,)), fx.make_layout(8, 1))
            logical_row = bid * fx.Int32(delta_rows_per_block) + tid // fx.Int32(lanes_per_row)
            if const_expr(heads_are_inner):
                batch = logical_row // fx.Int32(num_heads * sequence_length)
                head = logical_row % fx.Int32(num_heads)
                row = logical_row // fx.Int32(num_heads) % fx.Int32(sequence_length)
            else:
                batch = logical_row // fx.Int32(num_heads * sequence_length)
                head = logical_row // fx.Int32(sequence_length) % fx.Int32(num_heads)
                row = logical_row % fx.Int32(sequence_length)
            batch_head = batch * fx.Int32(num_heads) + head
            row_tile = bid
            output_row = batch * fx.Int32(out_stride[0]) + head * fx.Int32(out_stride[1]) + row * fx.Int32(out_stride[2])
            grad_row = batch * fx.Int32(grad_output_stride[0]) + head * fx.Int32(grad_output_stride[1]) + row * fx.Int32(grad_output_stride[2])
            zero_row = logical_row
        else:
            if const_expr(heads_are_inner):
                batch_head = bid % fx.Int32(batch_heads)
                row_tile = bid // fx.Int32(batch_heads)
            else:
                batch_head = bid // fx.Int32(delta_grid)
                row_tile = bid % fx.Int32(delta_grid)
            batch = fx.Int64(batch_head // fx.Int32(num_heads))
            head = fx.Int64(batch_head % fx.Int32(num_heads))
            row = row_tile * fx.Int32(delta_rows_per_block) + tid // fx.Int32(lanes_per_row)
            output_view = make_global_view(
                attention_output, (batch, head, None, None),
                (batch_size, num_heads, sequence_length, value_head_dim), out_stride
            )
            grad_view = make_global_view(
                grad_output, (batch, head, None, None),
                (batch_size, num_heads, sequence_length, value_head_dim), grad_output_stride
            )
            zero_view = make_global_view(
                grad_query, (batch, head, None, None),
                (batch_size, num_heads, sequence_length, qk_head_dim),
                contiguous_stride(qk_head_dim),
            )
            output_packs = fx.logical_divide(fx.slice(output_view, (row, None)), fx.make_layout(8, 1))
            grad_packs = fx.logical_divide(fx.slice(grad_view, (row, None)), fx.make_layout(8, 1))
            zero_row = row
        if const_expr(not dense_linear_delta):
            zero_packs = fx.logical_divide(
                fx.make_view(fx.get_iter(zero_view), fx.make_layout(sequence_length * qk_head_dim, 1)),
                fx.make_layout(8, 1),
            )
        zeros = fx.make_rmem_tensor(8, fx.BFloat16)
        zeros.store(fx.Vector.filled(8, 0.0, fx.BFloat16).ir_value())
        delta_view = fx.make_view(fx.get_iter(delta), fx.make_layout(batch_heads * sequence_length, 1))
        output_fragment = fx.make_rmem_tensor(8, fx.BFloat16)
        grad_fragment = fx.make_rmem_tensor(8, fx.BFloat16)
        row_sum = _f32(0.0)
        for load_step in fx.range_constexpr(delta_load_iterations):
            column = (chunk + fx.Int32(load_step * lanes_per_row)) * fx.Int32(8)
            if const_expr(dense_linear_delta):
                fx.copy(atom, fx.slice(output_packs, (None, (output_row + column) // fx.Int32(8))), output_fragment)
                fx.copy(atom, fx.slice(grad_packs, (None, (grad_row + column) // fx.Int32(8))), grad_fragment)
            else:
                fx.copy(atom, fx.slice(output_packs, (None, column // fx.Int32(8))), output_fragment)
                fx.copy(atom, fx.slice(grad_packs, (None, column // fx.Int32(8))), grad_fragment)
            products = fx.Vector(output_fragment.load()).to(fx.Float32) * fx.Vector(grad_fragment.load()).to(fx.Float32)
            for i in fx.range_constexpr(8):
                row_sum = row_sum + _f32(products[i])
        for shuffle_offset in delta_shuffle_offsets:
            row_sum = row_sum + _f32(fx.gpu.shuffle_xor(row_sum, shuffle_offset, 64))
        if chunk == fx.Int32(0):
            if const_expr(dense_linear_delta):
                fx.get_iter(delta_view)[batch_head * fx.Int32(sequence_length) + row] = row_sum
            else:
                fx.get_iter(delta_view)[fx.Int64(batch_head) * fx.Int64(sequence_length) + fx.Int64(row)] = row_sum

        for part in fx.range_constexpr((qk_head_dim // 8 + lanes_per_row - 1) // lanes_per_row):
            pack = chunk + fx.Int32(part * lanes_per_row)
            if pack < fx.Int32(qk_head_dim // 8):
                fx.copy(atom, zeros, fx.slice(zero_packs, (None, zero_row * fx.Int32(qk_head_dim // 8) + pack)))

        # Reuse the first delta CTAs in each head for the reverse query lists.
        if const_expr(block_list_traversal):
            if row_tile < fx.Int32((sequence_length + key_rows - 1) // key_rows):
                owner = row_tile
                mask_batch = fx.Int32(0) if const_expr(block_mask_batch == 1) else batch
                mask_head = fx.Int32(0) if const_expr(block_mask_heads == 1) else head
                mask_group = fx.Int64(mask_batch) * fx.Int64(block_mask_heads) + fx.Int64(mask_head)
                sparse_key = owner // fx.Int32(sparse_kv_block_size // key_rows)
                flags = fx.SharedAllocator().allocate(MaskFlags).peek().flags.ptr
                for step in fx.range_constexpr((num_sparse_blocks + compute_threads - 1) // compute_threads):
                    query_block = tid + fx.Int32(step * compute_threads)
                    if query_block < fx.Int32(num_sparse_blocks):
                        count_index = mask_group * fx.Int64(num_sparse_blocks) + fx.Int64(query_block)
                        partial_count = fx.Int32(fx.get_iter(kv_num_blocks)[count_index])
                        full_count = fx.Int32(fx.get_iter(full_kv_num_blocks)[count_index])
                        flag = fx.Int32(0)
                        for item in range(partial_count):
                            index = count_index * fx.Int64(max_partial_blocks_limit) + fx.Int64(item)
                            if fx.Int32(fx.get_iter(kv_indices)[index]) == sparse_key:
                                flag = fx.Int32(1)
                        for item in range(full_count):
                            index = count_index * fx.Int64(max_full_blocks_limit) + fx.Int64(item)
                            if fx.Int32(fx.get_iter(full_kv_indices)[index]) == sparse_key:
                                flag = fx.Int32(2)
                        flags[query_block] = flag
                fx.gpu.barrier()
                if tid == fx.Int32(0):
                    num_partial = fx.Int32(0)
                    num_full = fx.Int32(0)
                    owner_index = fx.Int64(batch_head) * fx.Int64(metadata_chunks) + fx.Int64(owner)
                    list_offset = owner_index * fx.Int64(metadata_chunks)
                    for compact_query in range(fx.Int32(num_sparse_blocks)):
                        compact_flag = fx.Int32(flags[compact_query])
                        if compact_flag == fx.Int32(1):
                            fx.get_iter(partial_indices)[list_offset + fx.Int64(num_partial)] = compact_query
                            num_partial = num_partial + fx.Int32(1)
                        elif compact_flag == fx.Int32(2):
                            fx.get_iter(full_indices)[list_offset + fx.Int64(num_full)] = compact_query
                            num_full = num_full + fx.Int32(1)
                    fx.get_iter(partial_counts)[owner_index] = num_partial
                    fx.get_iter(full_counts)[owner_index] = num_full

    fused_grid = (sequence_length + key_rows - 1) // key_rows
    compute_num_threads = num_waves * 64
    if block_list_traversal:
        compute_kernel = make_pipeline_kernel(locals(), _f32, fast_exp2)
        compute_attrs = {}
    else:
        consumer_waves = 2
        dq_phase_tiles = qk_head_dim // (16 * consumer_waves * 2)

        @fx.struct
        class SharedMemory:
            key: fx.Array[fx.BFloat16, key_rows * qk_head_dim, 16]
            value: fx.Array[fx.BFloat16, key_rows * value_head_dim, 16]
            query: fx.Array[fx.BFloat16, 2 * 16 * qk_head_dim, 16]
            grad_output: fx.Array[fx.BFloat16, 2 * 16 * value_head_dim, 16]
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
            qview = make_global_view(query, head_coord, (batch_size, num_heads, sequence_length, qk_head_dim), q_stride)
            kview = make_global_view(key, head_coord, (batch_size, num_heads, sequence_length, qk_head_dim), k_stride)
            vview = make_global_view(value, head_coord, (batch_size, num_heads, sequence_length, value_head_dim), v_stride)
            doview = make_global_view(grad_output, head_coord, (batch_size, num_heads, sequence_length, value_head_dim), grad_output_stride)
            dqview = make_global_view(grad_query, head_coord, (batch_size, num_heads, sequence_length, qk_head_dim), grad_query_workspace_stride)
            dkview = make_global_view(grad_key, head_coord, (batch_size, num_heads, sequence_length, qk_head_dim), grad_key_stride)
            dvview = make_global_view(grad_value, head_coord, (batch_size, num_heads, sequence_length, value_head_dim), grad_value_stride)
            lseview = make_global_view(logsumexp, (fx.Int64(batch_head), None), (batch_size * num_heads, sequence_length), (sequence_length, 1))
            deltaview = make_global_view(delta, (fx.Int64(batch_head), None), (batch_size * num_heads, sequence_length), (sequence_length, 1))

            def load_i32(view, index):
                return fx.Int32(fx.get_iter(view)[index])

            masks = make_mask_buffers(make_global_view, mask_buffer_count, mask_buffer_sizes, mask_buffer_0, mask_buffer_1, mask_buffer_2, mask_buffer_3)
            evaluate_mask = make_mask_evaluator(mask_program, mask_program_output, mask_buffer_strides, masks, load_i32, batch, head)
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
            klayout = make_value_shared_layout(key_rows, qk_head_dim)
            vlayout = make_value_shared_layout(key_rows, value_head_dim)
            qlayout = make_value_shared_layout(16, qk_head_dim)
            dolayout = make_value_shared_layout(16, value_head_dim)

            def stage_operand(view, layout, pointer, first_row, rows, dimension, row_stride):
                # The public BufferCopyLDS atom is synchronous with respect to the
                # compiler's VMEM wait tracking. Asyncmark-native DMA preserves
                # outstanding prior-dQ atomics at the cross-iteration boundary.
                base_layout = fx.make_layout(
                    ((16, rows // 16), (32, dimension // 32)),
                    ((32, 16 * dimension), (1, 16 * 32)),
                )
                coordinates = fx.right_inverse(base_layout)
                for step in fx.range_constexpr((rows * dimension // 8 + compute_num_threads - 1) // compute_num_threads):
                    linear = tid + fx.Int32(step * compute_num_threads)
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
                stage_operand(qview, qlayout, shared.query.ptr + stage * fx.Int32(16 * qk_head_dim), first_row, 16, qk_head_dim, q_stride[2])
                stage_operand(doview, dolayout, shared.grad_output.ptr + stage * fx.Int32(16 * value_head_dim), first_row, 16, value_head_dim, grad_output_stride[2])
                fx.rocdl.asyncmark()

            stage_operand(kview, klayout, shared.key.ptr, key_base, key_rows, qk_head_dim, k_stride[2])
            stage_operand(vview, vlayout, shared.value.ptr, key_base, key_rows, value_head_dim, v_stride[2])
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
                values = (fx.Vector(accumulator.load()) * fx.Vector.filled(4, softmax_scale, fx.Float32)).to(fx.BFloat16)
                for pair in fx.range_constexpr(2):
                    packed = fx.Vector.from_elements([values[pair * 2], values[pair * 2 + 1]], fx.BFloat16)
                    output = fx.make_rmem_tensor(2, fx.BFloat16)
                    output.store(packed.ir_value())
                    linear = tile * fx.Int32(256) + lane * fx.Int32(2) + fx.Int32(pair * 128)
                    row = query_tile * fx.Int32(16) + linear // fx.Int32(qk_head_dim)
                    column = linear % fx.Int32(qk_head_dim)
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
                qpointer = shared.query.ptr + stage * fx.Int32(16 * qk_head_dim)
                dopointer = shared.grad_output.ptr + stage * fx.Int32(16 * value_head_dim)
                scores = [fx.make_fragment_like(ccoords, fx.Float32) for _ in fx.range_constexpr(key_groups)]
                probabilities = [fx.make_fragment_like(ccoords, fx.Float32) for _ in fx.range_constexpr(key_groups)]
                ps = [fx.make_fragment_like(update_bcoords, fx.BFloat16) for _ in fx.range_constexpr(key_groups)]
                dss = [fx.make_fragment_like(update_bcoords, fx.BFloat16) for _ in fx.range_constexpr(key_groups)]
                for group in fx.range_constexpr(key_groups):
                    scores[group].fill(0)
                keys = [load_score_operand(shared.key.ptr, klayout, 0, group) for group in fx.range_constexpr(key_groups)]
                query_operand = load_score(qpointer, qlayout, 0)
                for step in fx.range_constexpr(qk_head_dim // 32):
                    if const_expr(step + 1 < qk_head_dim // 32):
                        next_keys = [load_score_operand(shared.key.ptr, klayout, step + 1, group) for group in fx.range_constexpr(key_groups)]
                        next_query = load_score(qpointer, qlayout, step + 1)
                    for group in fx.range_constexpr(key_groups):
                        fx.gemm(score_mma, scores[group], query_operand, keys[group], scores[group])
                    if const_expr(step + 1 < qk_head_dim // 32):
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
                            lse = lse * _f32(1.4426950408889634)
                        probability = fast_exp2(_f32(fx.math.fma(_f32(score_values[element]), _f32(scale_log2), -lse)))
                        if masked:
                            probability = evaluate_mask(position, key_pos + fx.Int32(group * 16)).select(probability, _f32(0.0))
                        if const_expr(not direct_range_traversal):
                            probability = (key_pos + fx.Int32(group * 16) < fx.Int32(sequence_length)).select(probability, _f32(0.0))
                        values.append(probability)
                    probabilities[group].store(fx.Vector.from_elements(values, fx.Float32).ir_value())
                dps = [fx.make_fragment_like(ccoords, fx.Float32) for _ in fx.range_constexpr(key_groups)]
                for group in fx.range_constexpr(key_groups):
                    dps[group].fill(0)
                for step in fx.range_constexpr(value_head_dim // 32):
                    if const_expr(step + 1 < value_head_dim // 32):
                        next_values = [load_score_operand(shared.value.ptr, vlayout, step + 1, group) for group in fx.range_constexpr(key_groups)]
                        next_grad_output = load_score(dopointer, dolayout, step + 1)
                    for group in fx.range_constexpr(key_groups):
                        fx.gemm(score_mma, dps[group], grad_output_operand, values_operand[group], dps[group])
                    if const_expr(step + 1 < value_head_dim // 32):
                        values_operand = next_values
                        grad_output_operand = next_grad_output
                for group in fx.range_constexpr(key_groups):
                    probability_values = fx.Vector(probabilities[group].load())
                    dp_values = fx.Vector(dps[group].load())
                    ds_values = []
                    for element in fx.range_constexpr(4):
                        local_row = fx.Int32(fx.get_scalar(ccoords[element]))
                        delta_value = fx.Float32(fx.ptr_load(shared.stats.ptr + stage * fx.Int32(32) + fx.Int32(16) + local_row))
                        ds_values.append(_f32(probability_values[element]) * (_f32(dp_values[element]) - delta_value))
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
                for d in fx.range_constexpr(qk_head_dim // 32):
                    if const_expr(d + 1 < qk_head_dim // 32):
                        next_qa = load_wide(qpointer, qlayout, d + 1)
                    if const_expr(d < value_head_dim // 32):
                        doa = load_wide(dopointer, dolayout, d)
                    update_in_bank(dk_wide[d], qa, ds_wide, "a")
                    if const_expr(d < value_head_dim // 32):
                        update_in_bank(dv_wide[d], doa, p_wide, "v")
                    if const_expr(d + 1 < qk_head_dim // 32):
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
                if const_expr(unmasked_traversal):
                    run(fx.Int32(sequence_length // 16), fx.Int32(0), False, producer, dk_wide, dv_wide, cached_keys)
                elif const_expr(direct_range_traversal):
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
                dk_wide = [fx.make_fragment_like(wide_dim_coords, fx.Float32) for _ in fx.range_constexpr(qk_head_dim // 32)]
                dv_wide = [fx.make_fragment_like(wide_dim_coords, fx.Float32) for _ in fx.range_constexpr(value_head_dim // 32)]
                for d in fx.range_constexpr(qk_head_dim // 32):
                    dk_wide[d].fill(0)
                for d in fx.range_constexpr(value_head_dim // 32):
                    dv_wide[d].fill(0)
                execute_runs(True, dk_wide, dv_wide, [])
                wide_key_pos = key_base + wave * fx.Int32(key_groups * 16) + fx.Int32(fx.get_scalar(wide_row_coords[0]))
                if wide_key_pos < fx.Int32(sequence_length):
                    dkrow = fx.logical_divide(fx.slice(dkview, (wide_key_pos, None)), fx.make_layout(4, 1))
                    dvrow = fx.logical_divide(fx.slice(dvview, (wide_key_pos, None)), fx.make_layout(4, 1))
                    for d in fx.range_constexpr(qk_head_dim // 32):
                        values = (fx.Vector(dk_wide[d].load()) * fx.Vector.filled(16, softmax_scale, fx.Float32)).to(fx.BFloat16)
                        for part in fx.range_constexpr(4):
                            output = fx.make_rmem_tensor(4, fx.BFloat16)
                            output.store(fx.Vector.from_elements([values[part * 4 + e] for e in fx.range_constexpr(4)], fx.BFloat16).ir_value())
                            column = fx.Int32(d * 32) + fx.Int32(fx.get_scalar(wide_dim_coords[part * 4]))
                            fx.copy(g64, output, fx.slice(dkrow, (None, column // fx.Int32(4))))
                    for d in fx.range_constexpr(value_head_dim // 32):
                        values = fx.Vector(dv_wide[d].load()).to(fx.BFloat16)
                        for part in fx.range_constexpr(4):
                            output = fx.make_rmem_tensor(4, fx.BFloat16)
                            output.store(fx.Vector.from_elements([values[part * 4 + e] for e in fx.range_constexpr(4)], fx.BFloat16).ir_value())
                            column = fx.Int32(d * 32) + fx.Int32(fx.get_scalar(wide_dim_coords[part * 4]))
                            fx.copy(g64, output, fx.slice(dvrow, (None, column // fx.Int32(4))))

            else:
                cached_keys = cache_consumer_keys()
                execute_runs(False, [], [], cached_keys)

        compute_attrs = {"llvm.passthrough": [["amdgpu-agpr-alloc", "256"]]}

    @flyc.kernel
    def dq_layout_kernel(grad_query: fx.Tensor, grad_query_workspace: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        query_tile = fx.Int32(fx.block_idx.x)
        batch_head = fx.Int32(fx.block_idx.y)
        batch = batch_head // fx.Int32(num_heads)
        head = batch_head % fx.Int32(num_heads)
        view = make_global_view(
            grad_query,
            (fx.Int64(batch), fx.Int64(head), None, None),
            (batch_size, num_heads, sequence_length, qk_head_dim),
            grad_query_stride,
        )
        source_view = make_global_view(
            grad_query_workspace,
            (fx.Int64(batch), fx.Int64(head), None, None),
            (batch_size, num_heads, sequence_length, qk_head_dim),
            (num_heads * sequence_length * qk_head_dim, sequence_length * qk_head_dim, qk_head_dim, 1),
        )
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)
        query_base = query_tile * fx.Int32(64) + wave * fx.Int32(16)
        row_base = lane // fx.Int32(32) * fx.Int32(8)
        load = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        store = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.BFloat16)
        packed_source = fx.logical_divide(
            fx.make_view(fx.get_iter(source_view), fx.make_layout(sequence_length * qk_head_dim, 1)),
            fx.make_layout(8, 1),
        )
        fragments = []
        for block in fx.range_constexpr(qk_head_dim // 64):
            column = lane % fx.Int32(32) * fx.Int32(2) + fx.Int32(block * 64)
            native_base = query_base * fx.Int32(qk_head_dim) + column // fx.Int32(16) * fx.Int32(256) + row_base * fx.Int32(2) + column % fx.Int32(16) // fx.Int32(4) * fx.Int32(32) + column % fx.Int32(4) // fx.Int32(2) * fx.Int32(128)
            for half in fx.range_constexpr(2):
                fragment = fx.make_rmem_tensor(8, fx.BFloat16)
                fx.copy(load, fx.slice(packed_source, (None, native_base // fx.Int32(8) + fx.Int32(half))), fragment)
                fragments.append(fragment)
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.rocdl.s_barrier()
        for block in fx.range_constexpr(qk_head_dim // 64):
            column = lane % fx.Int32(32) * fx.Int32(2) + fx.Int32(block * 64)
            for half in fx.range_constexpr(2):
                fragment = fragments[block * 2 + half]
                values = fx.Vector(fragment.load())
                for word in fx.range_constexpr(4):
                    pair = fx.make_rmem_tensor(2, fx.BFloat16)
                    pair.store(
                        fx.Vector.from_elements([values[2 * word], values[2 * word + 1]], fx.BFloat16).ir_value()
                    )
                    row = query_base + row_base + fx.Int32(half * 4 + word)
                    destination = fx.logical_divide(fx.slice(view, (row, None)), fx.make_layout(2, 1))
                    fx.copy(store, pair, fx.slice(destination, (None, column // fx.Int32(2))))

    @flyc.jit
    def _launch(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        attention_output: fx.Tensor,
        logsumexp: fx.Tensor,
        grad_output: fx.Tensor,
        grad_query: fx.Tensor,
        grad_key: fx.Tensor,
        grad_value: fx.Tensor,
        kv_num_blocks: fx.Tensor,
        kv_indices: fx.Tensor,
        full_kv_num_blocks: fx.Tensor,
        full_kv_indices: fx.Tensor,
        delta: fx.Tensor,
        partial_kv_counts: fx.Tensor,
        partial_kv_indices: fx.Tensor,
        full_kv_counts: fx.Tensor,
        full_kv_indices_transposed: fx.Tensor,
        mask_buffer_0: fx.Tensor,
        mask_buffer_1: fx.Tensor,
        mask_buffer_2: fx.Tensor,
        mask_buffer_3: fx.Tensor,
        grad_query_workspace: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        delta_kernel(
            attention_output, grad_output, delta, grad_query_workspace,
            kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices,
            partial_kv_counts, partial_kv_indices, full_kv_counts, full_kv_indices_transposed,
        ).launch(
            grid=(delta_grid * batch_heads, 1, 1),
            block=(compute_threads, 1, 1),
            stream=stream,
        )
        compute_kernel(
            query,
            key,
            value,
            logsumexp,
            delta,
            grad_output,
            grad_query_workspace,
            grad_key,
            grad_value,
            partial_kv_counts,
            partial_kv_indices,
            full_kv_counts,
            full_kv_indices_transposed,
            mask_buffer_0,
            mask_buffer_1,
            mask_buffer_2,
            mask_buffer_3,
        ).launch(
            value_attrs=compute_attrs,
            grid=((batch_heads, fused_grid, 1) if heads_are_inner else (fused_grid, batch_heads, 1)),
            block=(compute_num_threads, 1, 1),
            stream=stream,
        )
        dq_layout_kernel(grad_query, grad_query_workspace).launch(
            grid=(sequence_length // 64, batch_heads, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    if num_waves == 8:
        _launch.compile_hints["waves_per_eu"] = 1
    if not block_list_traversal:
        _launch.compile_hints["llvm_options"] = {"amdgpu-mfma-vgpr-form": True}
    return _launch
