"""FlexAttention backward implementation for AMD gfx950 GPUs.

``build_flex_attn_bwd_module`` returns a FlyDSL JIT launcher that computes the
query, key, and value gradients for BF16 attention. Query/key and value head
dimensions must be either (128, 128) or (192, 128). BlockMask metadata uses
128 x 128 blocks, and tensor layouts are described by explicit strides.

The launcher submits the following kernels to the caller-provided stream:

1. ``delta_kernel`` computes the row-wise correction
   ``delta[b, h, i] = sum_d(out[b, h, i, d] * grad_out[b, h, i, d])``.
2. ``prologue`` is used for every non-dense mask. It converts
   BlockMask metadata into compact query-major and key/value-major work lists.
3. ``compute_kernel`` dispatches workgroups to dQ owners or paired dK/dV
   owners. In a paired owner, producer waves compute ``P``, ``dS``, and dK;
   consumer waves reuse the staged ``P`` fragments for dV. Only the dense path
   skips the prologue; all other masks use the generic lowered mask program.

For attention probabilities ``P`` and score gradients ``dS``, the kernel
computes:

    P  = exp2(query @ key.T * (scale * log2(e)) - logsumexp * log2(e))
    dV = P.T @ dO
    dP = dO @ value.T
    dS = P * (dP - delta)
    dQ = (dS @ key) * scale
    dK = (dS.T @ query) * scale

Masked probability entries are zero. Partial tiles evaluate the same bounded
mask bytecode as the forward kernel, including captured int32 mask buffers.

The MFMA32 pipeline uses the compatible accumulator and transposed-operand
fragment layouts of the 32 x 32 x 16 BF16 instruction. Probability and score
gradient fragments can therefore feed the gradient update directly from
registers without an intermediate LDS round trip.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from .flex_attn_bwd_kv_owner import make_dkdv_mfma32_body
from .flex_attn_bwd_q_owner import make_dq_mfma32_body
from .flex_attn_utils import (
    fast_exp2,
    is_causal_document_mask_program,
    make_global_view,
    make_metadata_view,
)

_LOG2E = 1.4426950408889634
_BATCHED_CAUSAL_DOCUMENT_MASK_PROGRAM = (
    ("const_bool", True),
    ("ge", 2, 3),
    ("and", 4, 5),
    ("load_i32", 0, (0, 3)),
    ("le", 2, 7),
    ("and", 6, 8),
)


def _is_batched_causal_document_mask_program(
    mask_program,
    mask_program_output,
    mask_buffer_shapes,
    mask_buffer_strides,
    *,
    batch_size,
    sequence_length,
    block_mask_batch,
    block_mask_heads,
):
    return (
        tuple(mask_program) == _BATCHED_CAUSAL_DOCUMENT_MASK_PROGRAM
        and int(mask_program_output) == 9
        and tuple(tuple(shape) for shape in mask_buffer_shapes)
        == ((batch_size, sequence_length),)
        and tuple(tuple(stride) for stride in mask_buffer_strides)
        == ((sequence_length, 1),)
        and block_mask_batch == batch_size
        and block_mask_heads == 1
    )


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
    mask_buffer_shapes = tuple((tuple(shape) for shape in mask_buffer_shapes))
    mask_buffer_strides = tuple((tuple(stride) for stride in mask_buffer_strides))
    batched_causal_document_mask = _is_batched_causal_document_mask_program(
        mask_program,
        mask_program_output,
        mask_buffer_shapes,
        mask_buffer_strides,
        batch_size=batch_size,
        sequence_length=sequence_length,
        block_mask_batch=block_mask_batch,
        block_mask_heads=block_mask_heads,
    )
    # Exact document-program matches only select a reduction tuning. If lowering
    # changes, the generic mask path remains correct.
    narrow_dq_reduction = (
        sequence_length == 4096
        and qk_head_dim == 192
        and (value_head_dim == 128)
        and (int(sparse_q_block_size) == 128)
        and (int(sparse_kv_block_size) == 128)
        and (
            is_causal_document_mask_program(
                tuple(mask_program), int(mask_program_output), mask_buffer_strides
            )
            or batched_causal_document_mask
        )
    )
    return _build_flex_attn_bwd_module(
        batch_size,
        num_heads,
        sequence_length,
        qk_head_dim,
        value_head_dim,
        dtype_str,
        sparse_q_block_size,
        sparse_kv_block_size,
        scale=scale,
        max_partial_blocks=max_partial_blocks,
        max_full_blocks=max_full_blocks,
        block_mask_batch=block_mask_batch,
        block_mask_heads=block_mask_heads,
        mask_program=mask_program,
        mask_program_output=mask_program_output,
        mask_buffer_shapes=mask_buffer_shapes,
        mask_buffer_strides=mask_buffer_strides,
        q_stride=q_stride,
        k_stride=k_stride,
        v_stride=v_stride,
        out_stride=out_stride,
        do_stride=do_stride,
        dq_stride=dq_stride,
        dk_stride=dk_stride,
        dv_stride=dv_stride,
        lse_in_log2=lse_in_log2,
        dq_reduction_rows=32 if narrow_dq_reduction else 64,
    )


def _build_flex_attn_bwd_module(
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
    dq_reduction_rows=64,
):
    # Keep standalone entry-point validation even though Inductor checks these
    # constraints before registering the vendored kernel.
    if dtype_str != "bf16":
        raise ValueError(f"unsupported dtype {dtype_str}")
    if (qk_head_dim, value_head_dim) not in ((128, 128), (192, 128)):
        raise ValueError(
            "FlyDSL backward requires (qk_head_dim, value_head_dim) to be "
            "(128, 128) or (192, 128)"
        )
    sparse_q_block_size = int(sparse_q_block_size)
    sparse_kv_block_size = int(sparse_kv_block_size)
    if sparse_q_block_size != 128 or sparse_kv_block_size != 128:
        raise ValueError("FlyDSL backward requires a 128x128 sparse block")
    if sequence_length <= 0 or sequence_length % sparse_q_block_size:
        raise ValueError(
            "FlyDSL backward requires sequence_length to be divisible by sparse_q_block_size"
        )
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
    mask_buffer_count = len(mask_buffer_shapes)
    dense_mask = not mask_program and mask_buffer_count == 0
    indirect_mask = not dense_mask
    wide_owner = (
        dense_mask
        and qk_head_dim + value_head_dim <= 256
        and (sequence_length % 256 == 0)
        and (64 * qk_head_dim // 8 % (8 * 64) == 0)
        and (64 * value_head_dim // 8 % (8 * 64) == 0)
    )
    eight_wave_compute = wide_owner
    reduction_block_rows = 64 if wide_owner or indirect_mask else 32
    num_waves = 8 if eight_wave_compute else 2
    grad_key_direction = 0
    grad_value_direction = 1
    grad_query_direction = 2
    sparse_block_rows = sparse_q_block_size
    query_chunk_rows = 64 if indirect_mask else 32
    kv_chunk_rows = 64 if indirect_mask else 32
    compute_waves = 4 if wide_owner or indirect_mask else 2
    compute_threads = compute_waves * 64
    num_sparse_blocks = sequence_length // sparse_block_rows
    num_query_chunks = sequence_length // query_chunk_rows
    num_kv_chunks = sequence_length // kv_chunk_rows
    chunks_per_sparse_block = sparse_block_rows // query_chunk_rows
    max_partial_blocks_limit = (
        num_sparse_blocks if max_partial_blocks is None else int(max_partial_blocks)
    )
    max_full_blocks_limit = (
        num_sparse_blocks if max_full_blocks is None else int(max_full_blocks)
    )
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
    batch_heads = batch_size * num_heads
    stats_stride = sequence_length
    delta_packs = value_head_dim // 8
    lanes_per_row = (
        delta_packs if delta_packs <= 64 and delta_packs & delta_packs - 1 == 0 else 4
    )
    delta_load_iterations = delta_packs // lanes_per_row
    delta_rows_per_block = compute_threads // lanes_per_row
    delta_grid = sequence_length // delta_rows_per_block
    delta_shuffle_offsets = []
    _s = 1
    while _s < lanes_per_row:
        delta_shuffle_offsets.append(_s)
        _s *= 2

    @flyc.kernel
    def delta_kernel(
        attention_output: fx.Tensor, grad_output: fx.Tensor, delta: fx.Tensor
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        batch_head = fx.Int32(fx.block_idx.y)
        batch = batch_head // fx.Int32(num_heads)
        head = batch_head % fx.Int32(num_heads)
        batch_i64 = fx.Int64(batch)
        head_i64 = fx.Int64(head)
        output_view = make_global_view(
            attention_output,
            (batch_i64, head_i64, None, None),
            (batch_size, num_heads, sequence_length, value_head_dim),
            out_stride,
        )
        grad_output_view = make_global_view(
            grad_output,
            (batch_i64, head_i64, None, None),
            (batch_size, num_heads, sequence_length, value_head_dim),
            grad_output_stride,
        )
        delta_view = make_global_view(
            delta,
            (fx.Int64(batch_head), None),
            (batch_heads, sequence_length),
            (sequence_length, 1),
        )
        chunk = tid % fx.Int32(lanes_per_row)
        row = bid * fx.Int32(delta_rows_per_block) + tid // fx.Int32(lanes_per_row)
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        output_fragment = fx.make_rmem_tensor(8, fx.BFloat16)
        grad_output_fragment = fx.make_rmem_tensor(8, fx.BFloat16)
        output_row_packs = fx.logical_divide(
            fx.slice(output_view, (row, None)), fx.make_layout(8, 1)
        )
        grad_output_row_packs = fx.logical_divide(
            fx.slice(grad_output_view, (row, None)), fx.make_layout(8, 1)
        )
        row_sum = _f32(0.0)
        for load_step in fx.range_constexpr(delta_load_iterations):
            pack = chunk + fx.Int32(load_step * lanes_per_row)
            fx.copy(atom, fx.slice(output_row_packs, (None, pack)), output_fragment)
            fx.copy(
                atom,
                fx.slice(grad_output_row_packs, (None, pack)),
                grad_output_fragment,
            )
            products = fx.Vector(output_fragment.load()).to(fx.Float32) * fx.Vector(
                grad_output_fragment.load()
            ).to(fx.Float32)
            for i in fx.range_constexpr(8):
                row_sum = row_sum + _f32(products[i])
        for shuffle_offset in delta_shuffle_offsets:
            row_sum = row_sum + _f32(fx.gpu.shuffle_xor(row_sum, shuffle_offset, 64))
        if chunk == fx.Int32(0):
            fx.get_iter(delta_view)[row] = row_sum

    flag_elements = (
        (num_sparse_blocks * num_sparse_blocks + 1 + compute_threads - 1)
        // compute_threads
        * compute_threads
    )
    flag_fill_iterations = (
        num_sparse_blocks * num_sparse_blocks + compute_threads - 1
    ) // compute_threads

    @fx.struct
    class PrologueSharedMemory:
        flags: fx.Array[fx.Int32, flag_elements, 16]

    @flyc.kernel
    def prologue(
        kv_num_blocks: fx.Tensor,
        kv_indices: fx.Tensor,
        full_kv_num_blocks: fx.Tensor,
        full_kv_indices: fx.Tensor,
        partial_q_counts: fx.Tensor,
        partial_q_indices: fx.Tensor,
        full_q_counts: fx.Tensor,
        full_q_indices: fx.Tensor,
        partial_kv_counts: fx.Tensor,
        partial_kv_indices: fx.Tensor,
        full_kv_counts: fx.Tensor,
        full_kv_indices_transposed: fx.Tensor,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        batch_head = fx.Int32(fx.block_idx.x)
        batch = batch_head // fx.Int32(num_heads)
        head = batch_head % fx.Int32(num_heads)
        mask_batch = fx.Int32(0) if block_mask_batch == 1 else batch
        mask_head = fx.Int32(0) if block_mask_heads == 1 else head
        mask_batch_i64 = fx.Int64(mask_batch)
        mask_head_i64 = fx.Int64(mask_head)
        mask_group_i64 = (
            mask_batch_i64 * fx.Int64(block_mask_heads) + mask_head_i64
        )
        kv_num_blocks_offset = mask_group_i64 * fx.Int64(num_sparse_blocks)
        kv_indices_offset = (
            mask_group_i64
            * fx.Int64(num_sparse_blocks * max_partial_blocks_limit)
        )
        full_kv_num_blocks_offset = mask_group_i64 * fx.Int64(num_sparse_blocks)
        full_kv_indices_offset = (
            mask_group_i64 * fx.Int64(num_sparse_blocks * max_full_blocks_limit)
        )
        shared_memory = fx.SharedAllocator().allocate(PrologueSharedMemory).peek()
        flags_pointer = shared_memory.flags.ptr
        partial_q_counts_view = make_metadata_view(
            partial_q_counts,
            batch_head * fx.Int32(num_query_chunks),
            num_query_chunks,
        )
        full_q_counts_view = make_metadata_view(
            full_q_counts,
            batch_head * fx.Int32(num_query_chunks),
            num_query_chunks,
        )
        partial_q_indices_view = make_metadata_view(
            partial_q_indices,
            batch_head * fx.Int32(num_query_chunks * num_kv_chunks),
            num_query_chunks * num_kv_chunks,
        )
        full_q_indices_view = make_metadata_view(
            full_q_indices,
            batch_head * fx.Int32(num_query_chunks * num_kv_chunks),
            num_query_chunks * num_kv_chunks,
        )
        partial_kv_counts_view = make_metadata_view(
            partial_kv_counts,
            batch_head * fx.Int32(num_kv_chunks),
            num_kv_chunks,
        )
        full_kv_counts_view = make_metadata_view(
            full_kv_counts,
            batch_head * fx.Int32(num_kv_chunks),
            num_kv_chunks,
        )
        partial_kv_indices_view = make_metadata_view(
            partial_kv_indices,
            batch_head * fx.Int32(num_kv_chunks * num_query_chunks),
            num_kv_chunks * num_query_chunks,
        )
        full_kv_indices_view = make_metadata_view(
            full_kv_indices_transposed,
            batch_head * fx.Int32(num_kv_chunks * num_query_chunks),
            num_kv_chunks * num_query_chunks,
        )

        def store_i32(view, idx, val):
            fx.get_iter(view)[idx] = val

        def load_tensor_i32(tensor, idx):
            return fx.Int32(fx.get_iter(tensor)[idx])

        zero = fx.Int32(0)
        for l in fx.range_constexpr(flag_elements // compute_threads):
            fx.ptr_store(zero, flags_pointer + (fx.Int32(l * compute_threads) + tid))
        fx.gpu.barrier()
        one = fx.Int32(1)
        two = fx.Int32(2)
        dump = fx.Int32(num_sparse_blocks * num_sparse_blocks)
        for l in fx.range_constexpr(flag_fill_iterations):
            i = fx.Int32(l * compute_threads) + tid
            sparse_query_block_raw = i // fx.Int32(num_sparse_blocks)
            sparse_query_block = (
                sparse_query_block_raw < fx.Int32(num_sparse_blocks)
            ).select(sparse_query_block_raw, fx.Int32(num_sparse_blocks - 1))
            t = i % fx.Int32(num_sparse_blocks)
            partial_block_count = load_tensor_i32(
                kv_num_blocks,
                kv_num_blocks_offset + fx.Int64(sparse_query_block),
            )
            partial_list_index = (t < fx.Int32(max_partial_blocks_limit)).select(
                t, fx.Int32(max_partial_blocks_limit - 1)
            )
            kv_block = load_tensor_i32(
                kv_indices,
                kv_indices_offset
                + fx.Int64(sparse_query_block)
                * fx.Int64(max_partial_blocks_limit)
                + fx.Int64(partial_list_index),
            )
            partial_active = (t < partial_block_count) & (
                t < fx.Int32(max_partial_blocks_limit)
            )
            partial_flag_offset = partial_active.select(
                sparse_query_block * fx.Int32(num_sparse_blocks) + kv_block, dump
            )
            fx.ptr_store(one, flags_pointer + partial_flag_offset)
            full_block_count = load_tensor_i32(
                full_kv_num_blocks,
                full_kv_num_blocks_offset + fx.Int64(sparse_query_block),
            )
            full_list_index = (t < fx.Int32(max_full_blocks_limit)).select(
                t, fx.Int32(max_full_blocks_limit - 1)
            )
            full_kv_block = load_tensor_i32(
                full_kv_indices,
                full_kv_indices_offset
                + fx.Int64(sparse_query_block) * fx.Int64(max_full_blocks_limit)
                + fx.Int64(full_list_index),
            )
            full_active = (t < full_block_count) & (t < fx.Int32(max_full_blocks_limit))
            full_flag_offset = full_active.select(
                sparse_query_block * fx.Int32(num_sparse_blocks) + full_kv_block, dump
            )
            fx.ptr_store(two, flags_pointer + full_flag_offset)
        fx.gpu.barrier()
        query_chunk = (tid < fx.Int32(num_query_chunks)).select(
            tid, fx.Int32(num_query_chunks - 1)
        )
        sparse_query_block = query_chunk // fx.Int32(chunks_per_sparse_block)
        partial_q_count = fx.Int32(0)
        full_q_count = fx.Int32(0)
        partial_kv_count = fx.Int32(0)
        full_kv_count = fx.Int32(0)
        query_worklist_offset = query_chunk * fx.Int32(num_kv_chunks)
        kv_worklist_offset = query_chunk * fx.Int32(num_query_chunks)
        for kv_block in fx.range_constexpr(num_sparse_blocks):
            query_to_kv_flag = fx.Int32(
                fx.ptr_load(
                    flags_pointer
                    + (
                        sparse_query_block * fx.Int32(num_sparse_blocks)
                        + fx.Int32(kv_block)
                    )
                )
            )
            kv_to_query_flag = fx.Int32(
                fx.ptr_load(
                    flags_pointer
                    + (fx.Int32(kv_block * num_sparse_blocks) + sparse_query_block)
                )
            )
            for s in fx.range_constexpr(chunks_per_sparse_block):
                chunk_index = fx.Int32(kv_block * chunks_per_sparse_block + s)
                store_i32(
                    partial_q_indices_view,
                    query_worklist_offset + partial_q_count,
                    chunk_index,
                )
                partial_q_count = partial_q_count + (query_to_kv_flag == one).select(
                    one, zero
                )
                store_i32(
                    full_q_indices_view,
                    query_worklist_offset + full_q_count,
                    chunk_index,
                )
                full_q_count = full_q_count + (query_to_kv_flag == two).select(
                    one, zero
                )
                store_i32(
                    partial_kv_indices_view,
                    kv_worklist_offset + partial_kv_count,
                    chunk_index,
                )
                partial_kv_count = partial_kv_count + (kv_to_query_flag == one).select(
                    one, zero
                )
                store_i32(
                    full_kv_indices_view,
                    kv_worklist_offset + full_kv_count,
                    chunk_index,
                )
                full_kv_count = full_kv_count + (kv_to_query_flag == two).select(
                    one, zero
                )
        store_i32(partial_q_counts_view, query_chunk, partial_q_count)
        store_i32(full_q_counts_view, query_chunk, full_q_count)
        store_i32(partial_kv_counts_view, query_chunk, partial_kv_count)
        store_i32(full_kv_counts_view, query_chunk, full_kv_count)

    values_per_thread = 8
    dq_query_rows = 256 if eight_wave_compute else 64
    dq_kv_rows = reduction_block_rows
    dq_num_waves = num_waves
    dq_num_threads = dq_num_waves * 64
    dq_num_query_chunks = sequence_length // dq_query_rows
    dq_num_kv_chunks = sequence_length // dq_kv_rows
    dq_key_load_iterations = (
        dq_kv_rows * qk_head_dim // values_per_thread // dq_num_threads
    )
    dq_value_load_iterations = (
        dq_kv_rows * value_head_dim // values_per_thread // dq_num_threads
    )
    pair_query_rows = 32
    pair_logical_waves = num_waves // 2
    pair_kv_rows = pair_logical_waves * 32
    pair_num_query_chunks = sequence_length // pair_query_rows
    pair_num_kv_chunks = sequence_length // pair_kv_rows
    pair_query_load_iterations = (
        pair_query_rows * qk_head_dim // values_per_thread // (pair_logical_waves * 64)
    )
    pair_grad_output_load_iterations = (
        pair_query_rows
        * value_head_dim
        // values_per_thread
        // (pair_logical_waves * 64)
    )
    if (dq_kv_rows * (qk_head_dim // values_per_thread)) % dq_num_threads:
        raise ValueError("FlyDSL backward key staging must evenly cover its tile")
    if (dq_kv_rows * (value_head_dim // values_per_thread)) % dq_num_threads:
        raise ValueError("FlyDSL backward value staging must evenly cover its tile")
    pair_load_threads = pair_logical_waves * 64
    if (pair_query_rows * (qk_head_dim // values_per_thread)) % pair_load_threads:
        raise ValueError("FlyDSL backward query staging must evenly cover its tile")
    if (pair_query_rows * (value_head_dim // values_per_thread)) % pair_load_threads:
        raise ValueError("FlyDSL backward dO staging must evenly cover its tile")
    pair_key_split = max(1, kv_chunk_rows // pair_kv_rows)
    pair_list_query_split = max(1, query_chunk_rows // pair_query_rows)
    fused_grid = pair_num_kv_chunks + dq_num_query_chunks
    pair_query_grad_output_elements = (
        2 * pair_query_rows * (qk_head_dim + value_head_dim)
    )
    pair_probability_elements = 2 * pair_logical_waves * 64 * 16
    compute_a_elements = max(
        query_chunk_rows * (qk_head_dim + 8),
        2 * dq_kv_rows * qk_head_dim,
        pair_query_grad_output_elements,
    )
    compute_b_elements = max(
        query_chunk_rows * (value_head_dim + 8),
        dq_kv_rows * value_head_dim,
        pair_probability_elements,
    )
    compute_metadata_elements = max(2 * query_chunk_rows, 4 * pair_query_rows)
    compute_num_threads = dq_num_threads

    @fx.struct
    class ComputeSharedMemory:
        a: fx.Array[fx.BFloat16, compute_a_elements, 16]
        b: fx.Array[fx.BFloat16, compute_b_elements, 16]
        ld: fx.Array[fx.Float32, compute_metadata_elements, 16]

    _emit_dq_mfma32_body = make_dq_mfma32_body(locals(), _f32, fast_exp2)
    _emit_dkdv_mfma32_body = make_dkdv_mfma32_body(locals(), _f32, fast_exp2)

    @flyc.jit
    def _emit_owner(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        logsumexp: fx.Tensor,
        delta: fx.Tensor,
        grad_output: fx.Tensor,
        grad_query: fx.Tensor,
        grad_key: fx.Tensor,
        grad_value: fx.Tensor,
        partial_q_counts: fx.Tensor,
        partial_q_indices: fx.Tensor,
        full_q_counts: fx.Tensor,
        full_q_indices: fx.Tensor,
        partial_kv_counts: fx.Tensor,
        partial_kv_indices: fx.Tensor,
        full_kv_counts: fx.Tensor,
        full_kv_indices_transposed: fx.Tensor,
        mask_buffer_0: fx.Tensor,
        mask_buffer_1: fx.Tensor,
        mask_buffer_2: fx.Tensor,
        mask_buffer_3: fx.Tensor,
        task: fx.Int32,
        batch_head: fx.Int32,
        shared_a_pointer,
        shared_b_pointer,
        shared_metadata_pointer,
    ):
        owner_index = task // fx.Int32(3)
        output_direction = task % fx.Int32(3)
        if output_direction < fx.Int32(grad_query_direction):
            paired_kv_chunk = owner_index * fx.Int32(2) + output_direction
            _emit_dkdv_mfma32_body(
                query,
                key,
                value,
                logsumexp,
                delta,
                grad_output,
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
                paired_kv_chunk,
                batch_head,
                shared_a_pointer,
                shared_b_pointer,
                shared_metadata_pointer,
            )
        else:
            _emit_dq_mfma32_body(
                query,
                key,
                value,
                logsumexp,
                delta,
                grad_output,
                grad_query,
                partial_q_counts,
                partial_q_indices,
                full_q_counts,
                full_q_indices,
                mask_buffer_0,
                mask_buffer_1,
                mask_buffer_2,
                mask_buffer_3,
                owner_index,
                batch_head,
                shared_a_pointer,
                shared_b_pointer,
            )

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
        partial_q_counts: fx.Tensor,
        partial_q_indices: fx.Tensor,
        full_q_counts: fx.Tensor,
        full_q_indices: fx.Tensor,
        partial_kv_counts: fx.Tensor,
        partial_kv_indices: fx.Tensor,
        full_kv_counts: fx.Tensor,
        full_kv_indices_transposed: fx.Tensor,
        mask_buffer_0: fx.Tensor,
        mask_buffer_1: fx.Tensor,
        mask_buffer_2: fx.Tensor,
        mask_buffer_3: fx.Tensor,
    ):
        task = fx.Int32(fx.block_idx.x)
        batch_head = fx.Int32(fx.block_idx.y)
        shared_memory = fx.SharedAllocator().allocate(ComputeSharedMemory).peek()
        _emit_owner(
            query,
            key,
            value,
            logsumexp,
            delta,
            grad_output,
            grad_query,
            grad_key,
            grad_value,
            partial_q_counts,
            partial_q_indices,
            full_q_counts,
            full_q_indices,
            partial_kv_counts,
            partial_kv_indices,
            full_kv_counts,
            full_kv_indices_transposed,
            mask_buffer_0,
            mask_buffer_1,
            mask_buffer_2,
            mask_buffer_3,
            task,
            batch_head,
            shared_memory.a.ptr,
            shared_memory.b.ptr,
            shared_memory.ld.ptr,
        )

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
        partial_q_counts: fx.Tensor,
        partial_q_indices: fx.Tensor,
        full_q_counts: fx.Tensor,
        full_q_indices: fx.Tensor,
        partial_kv_counts: fx.Tensor,
        partial_kv_indices: fx.Tensor,
        full_kv_counts: fx.Tensor,
        full_kv_indices_transposed: fx.Tensor,
        mask_buffer_0: fx.Tensor,
        mask_buffer_1: fx.Tensor,
        mask_buffer_2: fx.Tensor,
        mask_buffer_3: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        delta_kernel(attention_output, grad_output, delta).launch(
            grid=(delta_grid, batch_heads, 1),
            block=(compute_threads, 1, 1),
            stream=stream,
        )
        if const_expr(indirect_mask):
            prologue(
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                partial_q_counts,
                partial_q_indices,
                full_q_counts,
                full_q_indices,
                partial_kv_counts,
                partial_kv_indices,
                full_kv_counts,
                full_kv_indices_transposed,
            ).launch(
                grid=(batch_heads, 1, 1), block=(compute_threads, 1, 1), stream=stream
            )
        compute_kernel(
            query,
            key,
            value,
            logsumexp,
            delta,
            grad_output,
            grad_query,
            grad_key,
            grad_value,
            partial_q_counts,
            partial_q_indices,
            full_q_counts,
            full_q_indices,
            partial_kv_counts,
            partial_kv_indices,
            full_kv_counts,
            full_kv_indices_transposed,
            mask_buffer_0,
            mask_buffer_1,
            mask_buffer_2,
            mask_buffer_3,
        ).launch(
            grid=(fused_grid, batch_heads, 1),
            block=(compute_num_threads, 1, 1),
            stream=stream,
        )

    return _launch
