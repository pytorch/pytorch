"""BF16 backward for gfx950, DQK128/192 and DV128.

The launcher computes delta and zeros a BF16 dQ workspace, runs the
cross-iteration MFMA pipeline, then scatters dQ into its requested layout.
Dense masks use the register-tuned producer/consumer kernel over contiguous
query ranges. Block masks use the shared pipeline over transposed partial/full
query lists. Mask semantics select the kernel and tile geometry.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from .flex_attn_bwd_dense import make_dense_kernel
from .flex_attn_bwd_pipeline import make_pipeline_kernel
from .flex_attn_utils import (
    DIRECT_RANGE_CAUSAL,
    MASK_TRAVERSAL_BLOCK_LIST,
    MASK_TRAVERSAL_DIRECT_RANGE,
    MASK_TRAVERSAL_UNMASKED,
    classify_mask_traversal,
    fast_exp2,
    make_dq_workspace_layout,
    make_global_view,
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
        if const_expr(heads_are_inner):
            batch_head = bid % fx.Int32(batch_heads)
            row_tile = bid // fx.Int32(batch_heads)
        else:
            batch_head = bid // fx.Int32(delta_grid)
            row_tile = bid % fx.Int32(delta_grid)
        batch = fx.Int64(batch_head // fx.Int32(num_heads))
        head = fx.Int64(batch_head % fx.Int32(num_heads))
        row = row_tile * fx.Int32(delta_rows_per_block) + tid // fx.Int32(lanes_per_row)
        chunk = tid % fx.Int32(lanes_per_row)
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
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
        zero_packs = fx.logical_divide(
            fx.make_view(fx.get_iter(zero_view), fx.make_layout(sequence_length * qk_head_dim, 1)),
            fx.make_layout(8, 1),
        )
        zeros = fx.make_rmem_tensor(8, fx.BFloat16)
        zeros.store(fx.Vector.filled(8, 0.0, fx.BFloat16).ir_value())
        delta_view = fx.make_view(fx.get_iter(delta), fx.make_layout(batch_heads * sequence_length, 1))
        output_packs = fx.logical_divide(fx.slice(output_view, (row, None)), fx.make_layout(8, 1))
        grad_packs = fx.logical_divide(fx.slice(grad_view, (row, None)), fx.make_layout(8, 1))
        output_fragment = fx.make_rmem_tensor(8, fx.BFloat16)
        grad_fragment = fx.make_rmem_tensor(8, fx.BFloat16)
        row_sum = _f32(0.0)
        for load_step in fx.range_constexpr(delta_load_iterations):
            column = (chunk + fx.Int32(load_step * lanes_per_row)) * fx.Int32(8)
            fx.copy(atom, fx.slice(output_packs, (None, column // fx.Int32(8))), output_fragment)
            fx.copy(atom, fx.slice(grad_packs, (None, column // fx.Int32(8))), grad_fragment)
            products = fx.Vector(output_fragment.load()).to(fx.Float32) * fx.Vector(grad_fragment.load()).to(fx.Float32)
            for i in fx.range_constexpr(8):
                row_sum = row_sum + _f32(products[i])
        for shuffle_offset in delta_shuffle_offsets:
            row_sum = row_sum + _f32(fx.gpu.shuffle_xor(row_sum, shuffle_offset, 64))
        if chunk == fx.Int32(0):
            fx.get_iter(delta_view)[fx.Int64(batch_head) * fx.Int64(sequence_length) + fx.Int64(row)] = row_sum

        for part in fx.range_constexpr((qk_head_dim // 8 + lanes_per_row - 1) // lanes_per_row):
            pack = chunk + fx.Int32(part * lanes_per_row)
            if pack < fx.Int32(qk_head_dim // 8):
                fx.copy(atom, zeros, fx.slice(zero_packs, (None, row * fx.Int32(qk_head_dim // 8) + pack)))

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
        compute_kernel = make_dense_kernel(locals(), _f32, fast_exp2)
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
        workspace_layout = make_dq_workspace_layout(sequence_length, qk_head_dim)
        fragments = []
        for block in fx.range_constexpr(qk_head_dim // 64):
            column = lane % fx.Int32(32) * fx.Int32(2) + fx.Int32(block * 64)
            for half in fx.range_constexpr(2):
                native_index = fx.Int32(
                    fx.get_scalar(fx.crd2idx((query_base + row_base + fx.Int32(half * 4), column), workspace_layout))
                )
                fragment = fx.make_rmem_tensor(8, fx.BFloat16)
                fx.copy(load, fx.slice(packed_source, (None, native_index // fx.Int32(8))), fragment)
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
