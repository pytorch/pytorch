"""BF16/FP16 backward with private dQ slots and fixed-order reduction on gfx950.

Each CTA processes its KV owners in a fixed order and writes only its private
dQ slot. The final kernel adds slots in order and scatters to the output layout.
Dense ranges and block lists select traversal and tile geometry.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr

from .flex_attn_bwd_pipeline import make_pipeline_kernel
from .flex_attn_bwd_utils import (
    backward_column_slices,
    backward_producer_waves,
    backward_single_pass,
    bwd_exp2,
    classify_mask_traversal,
    DIRECT_RANGE_CAUSAL,
    LSE_FLOOR,
    make_bwd_shared_layout,
    MASK_TRAVERSAL_BLOCK_LIST,
    MASK_TRAVERSAL_DIRECT_RANGE,
    MASK_TRAVERSAL_UNMASKED,
    MAX_REVERSE_LIST_QUERY_BLOCKS,
    split_dq_tiles,
    SUPPORTED_HEAD_DIMS,
    WIDE_HEAD_DIMS,
)
from .flex_attn_plan import backward_plan
from .flex_attn_target import resolve_target, WAVE_SIZE
from .flex_attn_utils import (
    capture_extent_count,
    capture_table_layout,
    ceil_div,
    ELEMENT_TYPES,
    make_capture_buffers,
    make_global_view,
    make_mask_evaluator,
    make_score_mod_readers,
    make_tile_score_mod_readers,
    mask_capture_sizes,
    optional_int,
    resolve_capture_extents,
    runtime_extent,
    runtime_strides,
    tied_nop,
    validate_score_mod_args,
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
    block_mask_is_dense=False,
    q_stride=None,
    k_stride=None,
    v_stride=None,
    out_stride=None,
    do_stride=None,
    dq_stride=None,
    dk_stride=None,
    dv_stride=None,
    lse_in_log2=False,
    dq_accum_fp32=True,
    num_kv_heads=None,
    kv_sequence_length=None,
    score_mod=None,
    joint_mod=None,
    score_mod_aux=(),
    score_mod_buffer_count=0,
    mod_key=None,
    has_grad_logsumexp=False,
    mask_buffer_dtypes=(),
    score_mod_buffer_dtypes=(),
    plan=None,
    single_pass=False,
    target_limits=None,
):
    """Build the three-kernel deterministic BF16/FP16 backward launcher.

    ``num_kv_heads``/``kv_sequence_length`` describe K/V and default to Q's.
    Each compute CTA owns one KV head and walks its group's Q heads, so dK/dV
    reduce in registers. With a ``plan``, None sizes/strides are read at
    runtime. ``joint_mod(pre, b, h, q_idx, kv_idx, dS, aux)`` maps dS with a
    ``score_mod``. With ``has_grad_logsumexp``, ``delta`` holds the log2-LSE
    gradient on entry (finite past Sq) and becomes
    ``rowsum(dO * O) - grad * log2(e)``. ``single_pass`` (``backward_single_pass``
    head dims) gives each producer 16 key rows at full width and steps over Q
    in 32-row pairs of tiles, so S and dP are computed once. ``target_limits``
    are the ``TargetLimits`` to plan for, by default the current device's.
    """
    target = resolve_target(target_limits)
    # Keep standalone entry-point validation even though Inductor checks these
    # constraints before registering the vendored kernel.
    if dtype_str not in ELEMENT_TYPES:
        raise ValueError(f"unsupported dtype {dtype_str}")
    element_type = ELEMENT_TYPES[dtype_str]
    has_score_mod = score_mod is not None
    if has_score_mod != (joint_mod is not None):
        raise ValueError("score_mod and joint_mod must be given together")
    score_mod_aux, score_mod_buffer_count = validate_score_mod_args(
        score_mod, score_mod_aux, score_mod_buffer_count, mod_key
    )
    if not isinstance(dq_accum_fp32, bool):
        raise ValueError("dq_accum_fp32 must be a bool")
    dq_workspace_dtype = fx.Float32 if dq_accum_fp32 else element_type
    dq_copy_op = fx.rocdl.BufferCopy128b if dq_accum_fp32 else fx.rocdl.BufferCopy64b
    dq_workspace_bytes = 4 if dq_accum_fp32 else 2
    single_pass = bool(single_pass)
    if single_pass and not backward_single_pass(qk_head_dim, value_head_dim, target):
        raise ValueError(
            "the FlyDSL single-pass backward does not fit "
            f"(qk_head_dim, value_head_dim) = {(qk_head_dim, value_head_dim)}"
        )
    if (qk_head_dim, value_head_dim) not in SUPPORTED_HEAD_DIMS + WIDE_HEAD_DIMS:
        raise ValueError(
            "FlyDSL backward requires (qk_head_dim, value_head_dim) "
            f"in {SUPPORTED_HEAD_DIMS + WIDE_HEAD_DIMS}"
        )
    sparse_q_block_size = int(sparse_q_block_size)
    sparse_kv_block_size = int(sparse_kv_block_size)
    if sparse_q_block_size != 128 or sparse_kv_block_size != 128:
        raise ValueError("FlyDSL backward requires a 128x128 sparse block")
    runtime = plan is not None

    batch_size = optional_int(batch_size)
    sequence_length = optional_int(sequence_length)
    num_kv_heads = num_heads if num_kv_heads is None else int(num_kv_heads)
    if kv_sequence_length is None and not runtime:
        kv_sequence_length = sequence_length
    kv_sequence_length = optional_int(kv_sequence_length)
    if not runtime and None in (batch_size, sequence_length):
        raise ValueError("FlyDSL backward needs a backward_plan for runtime sizes")
    if sequence_length is not None and sequence_length <= 0:
        raise ValueError("FlyDSL backward requires a positive sequence_length")
    if kv_sequence_length is not None and kv_sequence_length <= 0:
        raise ValueError("FlyDSL backward requires a positive kv_sequence_length")
    if num_kv_heads <= 0 or num_heads % num_kv_heads:
        raise ValueError("FlyDSL backward requires num_heads divisible by num_kv_heads")
    query_group = num_heads // num_kv_heads
    if block_mask_batch not in (1, batch_size, None):
        raise ValueError("BlockMask batch dimension must be 1 or batch_size")
    if block_mask_heads not in (1, num_kv_heads, num_heads):
        raise ValueError(
            "BlockMask head dimension must be 1, num_kv_heads, or num_heads"
        )
    if len(mask_buffer_shapes) != len(mask_buffer_strides):
        raise ValueError("Mask buffer shape/stride descriptors must match")
    mask_program = tuple(mask_program)
    mask_program_output = int(mask_program_output)
    mask_buffer_shapes = tuple(tuple(shape) for shape in mask_buffer_shapes)
    mask_buffer_strides = tuple(tuple(stride) for stride in mask_buffer_strides)
    mask_traversal, direct_range_kind = classify_mask_traversal(
        mask_program,
        mask_program_output,
        mask_buffer_shapes,
        sequence_length=sequence_length,
        key_sequence_length=kv_sequence_length,
        block_mask_is_dense=block_mask_is_dense,
    )
    mask_buffer_count = len(mask_buffer_shapes)
    if mask_traversal not in (
        MASK_TRAVERSAL_UNMASKED,
        MASK_TRAVERSAL_DIRECT_RANGE,
        MASK_TRAVERSAL_BLOCK_LIST,
    ):
        raise ValueError(f"unsupported mask traversal {mask_traversal}")
    if (
        mask_traversal == MASK_TRAVERSAL_DIRECT_RANGE
        and direct_range_kind != DIRECT_RANGE_CAUSAL
    ):
        raise ValueError(f"unsupported direct range {direct_range_kind}")
    unmasked_traversal = mask_traversal == MASK_TRAVERSAL_UNMASKED
    direct_range_traversal = mask_traversal == MASK_TRAVERSAL_DIRECT_RANGE
    block_list_traversal = mask_traversal == MASK_TRAVERSAL_BLOCK_LIST
    captures = mask_buffer_count + score_mod_buffer_count
    producer_waves = backward_producer_waves(
        block_list_traversal, qk_head_dim, value_head_dim, captures
    )
    num_waves = producer_waves + 2
    key_groups = 1 if single_pass else 2
    column_slices = (
        1 if single_pass else backward_column_slices(qk_head_dim, value_head_dim)
    )
    key_rows = producer_waves * key_groups * 16 // column_slices
    # Q rows of one producer step: a pair of 16-row tiles in single pass.
    step_rows = 32 if single_pass else 16
    ds_stride = step_rows + 8
    key_slice_chunks = qk_head_dim // 32 // column_slices
    value_slice_chunks = value_head_dim // 32 // column_slices

    def contiguous_stride(dim, heads=num_heads, length=sequence_length):
        return (heads * length * dim, length * dim, dim, 1)

    def kv_contiguous_stride(dim):
        return contiguous_stride(dim, num_kv_heads, kv_sequence_length)

    q_stride = tuple(q_stride or contiguous_stride(qk_head_dim))
    k_stride = tuple(k_stride or kv_contiguous_stride(qk_head_dim))
    v_stride = tuple(v_stride or kv_contiguous_stride(value_head_dim))
    out_stride = tuple(out_stride or contiguous_stride(value_head_dim))
    grad_output_stride = tuple(do_stride or contiguous_stride(value_head_dim))
    grad_query_stride = tuple(dq_stride or contiguous_stride(qk_head_dim))
    grad_key_stride = tuple(dk_stride or kv_contiguous_stride(qk_head_dim))
    grad_value_stride = tuple(dv_stride or kv_contiguous_stride(value_head_dim))
    if plan is None:
        plan = backward_plan(
            batch_size=batch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            sequence_length=sequence_length,
            kv_sequence_length=kv_sequence_length,
            qk_head_dim=qk_head_dim,
            value_head_dim=value_head_dim,
            mask_traversal=mask_traversal,
            workspace_element_bytes=dq_workspace_bytes,
            strides=(
                q_stride,
                k_stride,
                v_stride,
                out_stride,
                grad_output_stride,
                grad_query_stride,
                grad_key_stride,
                grad_value_stride,
            ),
            captures=captures,
            target=target,
        )
    # Delta and the dQ workspace cover whole 128-row Q blocks. Rows past Sq
    # load zeros and get zero probability, so they add nothing to dK/dV.
    q_ragged = plan["q_ragged"]
    kv_ragged = plan["kv_ragged"]
    dq_partitions = plan["dq_partitions"]
    dq_accumulates = plan["dq_accumulates"]
    workspace_head_group = plan["workspace_head_group"]
    cta_head_group = plan["cta_head_group"]
    heads_are_inner = plan["heads_are_inner"]
    dense_linear_delta = plan["dense_linear_delta"]
    # A runtime Sk may leave dQ slots without an owner.
    runtime_owners = kv_sequence_length is None
    zero_dq_workspace = dq_accumulates and (unmasked_traversal or block_list_traversal)
    accumulate_every_owner = dq_accumulates and unmasked_traversal

    def query_extents(sequence_length):
        padded_sequence_length = ceil_div(sequence_length, sparse_q_block_size) * (
            sparse_q_block_size
        )
        return (
            padded_sequence_length,
            ceil_div(sequence_length, 16),
            padded_sequence_length // sparse_q_block_size,
        )

    def key_extents(kv_sequence_length):
        return (
            ceil_div(kv_sequence_length, key_rows),
            ceil_div(kv_sequence_length, sparse_kv_block_size),
        )

    padded_sequence_length = query_tiles = num_sparse_blocks = None
    if sequence_length is not None:
        padded_sequence_length, query_tiles, num_sparse_blocks = query_extents(
            sequence_length
        )
    owner_count = key_metadata_chunks = None
    if kv_sequence_length is not None:
        owner_count, key_metadata_chunks = key_extents(kv_sequence_length)
    # With one owner and one slot, the compute kernel stores dQ itself and the
    # layout pass is skipped. Block lists need a single Q block, which the
    # owner zeroes when it lists nothing.
    direct_dq = (
        dq_partitions == 1
        and owner_count == 1
        and qk_head_dim % 16 == 0
        and (
            not block_list_traversal
            or sequence_length is not None
            and sequence_length <= sparse_q_block_size
        )
    )
    # Q blocks per BlockMask row (and entries per reverse list).
    metadata_chunks = num_sparse_blocks
    if (
        block_list_traversal
        and num_sparse_blocks is not None
        and num_sparse_blocks > MAX_REVERSE_LIST_QUERY_BLOCKS
    ):
        raise ValueError(
            "FlyDSL backward block lists support at most "
            f"{MAX_REVERSE_LIST_QUERY_BLOCKS} query blocks"
        )
    max_partial_blocks_limit = (
        key_metadata_chunks if max_partial_blocks is None else int(max_partial_blocks)
    )
    max_full_blocks_limit = (
        key_metadata_chunks if max_full_blocks is None else int(max_full_blocks)
    )
    compute_threads = 256
    softmax_scale = float(qk_head_dim) ** (-0.5) if scale is None else float(scale)
    scale_log2 = softmax_scale * _LOG2E
    lse_in_log2 = bool(lse_in_log2)
    has_grad_logsumexp = bool(has_grad_logsumexp)
    block_mask_batch = optional_int(block_mask_batch)
    block_mask_heads = int(block_mask_heads)
    mask_buffer_sizes = mask_capture_sizes(mask_buffer_shapes, mask_buffer_strides)
    capture_table = capture_table_layout(
        mask_buffer_sizes,
        tuple(mask_buffer_dtypes),
        score_mod_aux,
        score_mod_buffer_count,
        tuple(score_mod_buffer_dtypes),
    )
    capture_extent_count(mask_buffer_sizes, capture_table)
    query_shape = (batch_size, num_heads, sequence_length)
    key_shape = (batch_size, num_kv_heads, kv_sequence_length)
    batch_heads = None if batch_size is None else batch_size * num_heads
    kv_batch_heads = None if batch_size is None else batch_size * num_kv_heads

    delta_packs = value_head_dim // 8
    lanes_per_row = 4
    delta_load_iterations = delta_packs // lanes_per_row
    delta_rows_per_block = compute_threads // lanes_per_row
    # A static Sq under one Q block stops the compute kernel at its last step
    # and dq_layout drops rows past Sq, so only that prefix of each workspace
    # group, whose rows lead its layout, needs zeros. Each delta CTA zeroes
    # one chunk: (chunks per group, leading chunks to zero).
    zero_chunks = None
    if sequence_length is not None and sequence_length < sparse_q_block_size:
        zero_rows = ceil_div(sequence_length, step_rows) * step_rows
        zero_chunks = (
            padded_sequence_length * workspace_head_group // delta_rows_per_block,
            ceil_div(zero_rows * workspace_head_group, delta_rows_per_block),
        )

    def delta_rows(padded_sequence_length):
        return padded_sequence_length // delta_rows_per_block

    delta_grid = (
        None if padded_sequence_length is None else delta_rows(padded_sequence_length)
    )
    delta_shuffle_offsets = []
    _s = 1
    while _s < lanes_per_row:
        delta_shuffle_offsets.append(_s)
        _s *= 2

    static_query_blocks = num_sparse_blocks is not None
    static_key_lists = (
        None not in (key_metadata_chunks, delta_grid)
        and key_metadata_chunks <= delta_grid
    )
    # Kernels that need Sk but take no K/V tensor read it from K when it is a
    # runtime size.
    key_argument = fx.Tensor if runtime_owners else fx.Constexpr
    out_stride_static, grad_output_stride_static = out_stride, grad_output_stride
    max_partial_blocks_static = max_partial_blocks_limit
    max_full_blocks_static = max_full_blocks_limit

    def resolve_block_limits(kv_indices, full_kv_indices):
        """BlockMask list widths, read from the index tensors if runtime."""
        return (
            runtime_extent(max_partial_blocks_static, kv_indices, 3),
            runtime_extent(max_full_blocks_static, full_kv_indices, 3),
        )

    def resolve_query(tensor):
        """Sizes on the query side, read from a [B, Hq, Sq, D] tensor if runtime."""
        batch = runtime_extent(batch_size, tensor, 0)
        length = runtime_extent(sequence_length, tensor, 2)
        if sequence_length is None:
            padded, tiles, blocks = query_extents(length)
        else:
            padded, tiles, blocks = (
                padded_sequence_length,
                query_tiles,
                num_sparse_blocks,
            )
        return (
            batch,
            length,
            padded,
            blocks,
            batch * num_heads,
            delta_rows(padded),
            tiles,
            batch * num_kv_heads,
        )

    def resolve_key(key):
        """(Sk, owners, KV blocks), read from K if Sk is runtime."""
        if kv_sequence_length is not None:
            return kv_sequence_length, owner_count, key_metadata_chunks
        length = runtime_extent(None, key, 2)
        return (length, *key_extents(length))

    static_strides = {
        "q": q_stride,
        "k": k_stride,
        "v": v_stride,
        "do": grad_output_stride,
        "dq": grad_query_stride,
        "dk": grad_key_stride,
        "dv": grad_value_stride,
    }
    static_mask_buffer_strides = mask_buffer_strides
    static_mask_buffer_sizes = mask_buffer_sizes

    def resolve_compute(query, key, value, grad_output, grad_key, grad_value, masks):
        """Sizes, strides and capture extents of a compute kernel; runtime ones
        are read from its tensors."""
        batch, length, padded, blocks, heads, _, tiles, _ = resolve_query(query)
        key_length, owners, key_blocks = resolve_key(key)
        strides = tuple(
            runtime_strides(static_strides[name], tensor, side_length)
            for name, tensor, side_length in (
                ("q", query, length),
                ("k", key, key_length),
                ("v", value, key_length),
                ("do", grad_output, length),
                ("dk", grad_key, key_length),
                ("dv", grad_value, key_length),
            )
        )
        return (
            length,
            padded,
            tiles,
            heads,
            key_length,
            owners,
            key_blocks,
            blocks,
            (batch, num_heads, length),
            (batch, num_kv_heads, key_length),
            *strides,
            *resolve_mask_buffers(masks),
        )

    def resolve_mask_buffers(masks):
        return resolve_capture_extents(
            static_mask_buffer_strides, static_mask_buffer_sizes, masks, capture_table
        )

    @fx.struct
    class MaskFlags:
        # A runtime Sq builds the reverse lists one compute_threads chunk at a time.
        flags: fx.Array[
            fx.Int32,
            compute_threads if num_sparse_blocks is None else num_sparse_blocks,
            16,
        ]

    @flyc.kernel
    def delta_kernel(
        attention_output: fx.Tensor,
        grad_output: fx.Tensor,
        delta: fx.Tensor,
        grad_query: fx.Tensor,
        kv_num_blocks: fx.Tensor,
        kv_indices: fx.Tensor,
        full_kv_num_blocks: fx.Tensor,
        full_kv_indices: fx.Tensor,
        partial_counts: fx.Tensor,
        partial_indices: fx.Tensor,
        full_counts: fx.Tensor,
        full_indices: fx.Tensor,
        key: key_argument,
    ):
        (
            batch_size,
            sequence_length,
            padded_sequence_length,
            num_sparse_blocks,
            batch_heads,
            delta_grid,
        ) = resolve_query(attention_output)[:6]
        metadata_chunks = num_sparse_blocks
        key_metadata_chunks = resolve_key(key)[2]
        out_stride, grad_output_stride = (
            runtime_strides(out_stride_static, attention_output, sequence_length),
            runtime_strides(grad_output_stride_static, grad_output, sequence_length),
        )
        max_partial_blocks_limit, max_full_blocks_limit = resolve_block_limits(
            kv_indices, full_kv_indices
        )
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        chunk = tid % fx.Int32(lanes_per_row)
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), element_type)
        if const_expr(dense_linear_delta):
            output_packs = fx.logical_divide(
                fx.make_view(
                    fx.get_iter(
                        make_global_view(
                            attention_output,
                            None,
                            (batch_size, num_heads, sequence_length, value_head_dim),
                            out_stride,
                        )
                    ),
                    fx.make_layout(batch_heads * sequence_length * value_head_dim, 1),
                ),
                fx.make_layout(8, 1),
            )
            grad_packs = fx.logical_divide(
                fx.make_view(
                    fx.get_iter(
                        make_global_view(
                            grad_output,
                            None,
                            (batch_size, num_heads, sequence_length, value_head_dim),
                            grad_output_stride,
                        )
                    ),
                    fx.make_layout(batch_heads * sequence_length * value_head_dim, 1),
                ),
                fx.make_layout(8, 1),
            )
            logical_row = bid * fx.Int32(delta_rows_per_block) + tid // fx.Int32(
                lanes_per_row
            )
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
            output_row = (
                batch * fx.Int32(out_stride[0])
                + head * fx.Int32(out_stride[1])
                + row * fx.Int32(out_stride[2])
            )
            grad_row = (
                batch * fx.Int32(grad_output_stride[0])
                + head * fx.Int32(grad_output_stride[1])
                + row * fx.Int32(grad_output_stride[2])
            )
        else:
            if const_expr(heads_are_inner):
                batch_head = bid % fx.Int32(batch_heads)
                row_tile = bid // fx.Int32(batch_heads)
            else:
                batch_head = bid // fx.Int32(delta_grid)
                row_tile = bid % fx.Int32(delta_grid)
            batch = fx.Int64(batch_head // fx.Int32(num_heads))
            head = fx.Int64(batch_head % fx.Int32(num_heads))
            row = row_tile * fx.Int32(delta_rows_per_block) + tid // fx.Int32(
                lanes_per_row
            )
            output_view = make_global_view(
                attention_output,
                (batch, head, None, None),
                (batch_size, num_heads, sequence_length, value_head_dim),
                out_stride,
            )
            grad_view = make_global_view(
                grad_output,
                (batch, head, None, None),
                (batch_size, num_heads, sequence_length, value_head_dim),
                grad_output_stride,
            )
            output_packs = fx.logical_divide(
                fx.slice(output_view, (row, None)), fx.make_layout(8, 1)
            )
            grad_packs = fx.logical_divide(
                fx.slice(grad_view, (row, None)), fx.make_layout(8, 1)
            )
        delta_view = fx.make_view(
            fx.get_iter(delta), fx.make_layout(batch_heads * padded_sequence_length, 1)
        )
        output_fragment = fx.make_rmem_tensor(8, element_type)
        grad_fragment = fx.make_rmem_tensor(8, element_type)
        row_sum = _f32(0.0)
        for load_step in fx.range_constexpr(delta_load_iterations):
            column = (chunk + fx.Int32(load_step * lanes_per_row)) * fx.Int32(8)
            if const_expr(dense_linear_delta):
                fx.copy(
                    atom,
                    fx.slice(
                        output_packs, (None, (output_row + column) // fx.Int32(8))
                    ),
                    output_fragment,
                )
                fx.copy(
                    atom,
                    fx.slice(grad_packs, (None, (grad_row + column) // fx.Int32(8))),
                    grad_fragment,
                )
            else:
                fx.copy(
                    atom,
                    fx.slice(output_packs, (None, column // fx.Int32(8))),
                    output_fragment,
                )
                fx.copy(
                    atom,
                    fx.slice(grad_packs, (None, column // fx.Int32(8))),
                    grad_fragment,
                )
            products = fx.Vector(output_fragment.load()).to(fx.Float32) * fx.Vector(
                grad_fragment.load()
            ).to(fx.Float32)
            for i in fx.range_constexpr(8):
                row_sum = row_sum + _f32(products[i])
        for shuffle_offset in delta_shuffle_offsets:
            row_sum = row_sum + _f32(fx.gpu.shuffle_xor(row_sum, shuffle_offset, 64))
        if const_expr(has_grad_logsumexp):
            if const_expr(dense_linear_delta):
                grad_logsumexp = fx.get_iter(delta_view)[
                    batch_head * fx.Int32(sequence_length) + row
                ]
            else:
                grad_logsumexp = fx.get_iter(delta_view)[
                    fx.Int64(batch_head) * fx.Int64(padded_sequence_length)
                    + fx.Int64(row)
                ]
            row_sum = row_sum - _f32(grad_logsumexp) * _f32(_LOG2E)
        if chunk == fx.Int32(0):
            if const_expr(dense_linear_delta):
                fx.get_iter(delta_view)[
                    batch_head * fx.Int32(sequence_length) + row
                ] = row_sum
            else:
                fx.get_iter(delta_view)[
                    fx.Int64(batch_head) * fx.Int64(padded_sequence_length)
                    + fx.Int64(row)
                ] = row_sum

        def zero_workspace_rows():
            zero_elements = 16 // dq_workspace_bytes
            zero_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), dq_workspace_dtype)
            clear_elements = dq_partitions * delta_rows_per_block * qk_head_dim
            zero_view = make_global_view(
                grad_query,
                (fx.Int64(bid), None),
                (delta_grid * batch_heads, clear_elements),
                (clear_elements, 1),
            )
            zero_packs = fx.logical_divide(zero_view, fx.make_layout(zero_elements, 1))
            zero = fx.make_rmem_tensor(zero_elements, dq_workspace_dtype)
            zero.store(
                fx.Vector.filled(zero_elements, 0.0, dq_workspace_dtype).ir_value()
            )
            for part in fx.range_constexpr(
                dq_partitions
                * delta_rows_per_block
                * qk_head_dim
                // (compute_threads * zero_elements)
            ):
                index = tid + fx.Int32(part * compute_threads)
                fx.copy(zero_copy, zero, fx.slice(zero_packs, (None, index)))

        if const_expr(zero_dq_workspace and zero_chunks is not None):
            if bid % fx.Int32(zero_chunks[0]) < fx.Int32(zero_chunks[1]):
                zero_workspace_rows()
        elif const_expr(zero_dq_workspace):
            zero_workspace_rows()

        def build_chunked_reverse_lists(
            sparse_key,
            block_flag,
            flags,
            partial_indices,
            full_indices,
            partial_counts,
            full_counts,
        ):
            owner_index = fx.Int64(batch_head) * fx.Int64(
                key_metadata_chunks
            ) + fx.Int64(sparse_key)
            list_offset = owner_index * fx.Int64(metadata_chunks)
            num_partial = fx.Int32(0)
            num_full = fx.Int32(0)
            for first in range(
                fx.Int32(0), fx.Int32(num_sparse_blocks), fx.Int32(compute_threads)
            ):
                query_block = first + tid
                flag = fx.Int32(0)
                if query_block < fx.Int32(num_sparse_blocks):
                    flag = block_flag(query_block)
                flags[tid] = flag
                fx.gpu.barrier()
                if tid == fx.Int32(0):
                    for offset in range(fx.Int32(compute_threads)):
                        compact_flag = fx.Int32(flags[offset])
                        compact_query = first + offset
                        if compact_flag == fx.Int32(1):
                            fx.get_iter(partial_indices)[
                                list_offset + fx.Int64(num_partial)
                            ] = compact_query
                            num_partial = num_partial + fx.Int32(1)
                        elif compact_flag == fx.Int32(2):
                            fx.get_iter(full_indices)[
                                list_offset + fx.Int64(num_full)
                            ] = compact_query
                            num_full = num_full + fx.Int32(1)
                fx.gpu.barrier()
            if tid == fx.Int32(0):
                fx.get_iter(partial_counts)[owner_index] = num_partial
                fx.get_iter(full_counts)[owner_index] = num_full

        # Reuse the delta CTAs of each Q head to build its reverse query lists,
        # one per sparse KV block; every owner inside that block reads it.
        def build_reverse_lists(
            sparse_key, partial_indices, full_indices, partial_counts, full_counts
        ):
            mask_batch = fx.Int32(0) if const_expr(block_mask_batch == 1) else batch
            if const_expr(block_mask_heads == 1):
                mask_head = fx.Int32(0)
            elif const_expr(block_mask_heads == num_heads):
                mask_head = head
            else:
                mask_head = head // fx.Int64(query_group)
            mask_group = fx.Int64(mask_batch) * fx.Int64(block_mask_heads) + fx.Int64(
                mask_head
            )
            flags = fx.SharedAllocator().allocate(MaskFlags).peek().flags.ptr

            def block_flag(query_block):
                count_index = mask_group * fx.Int64(num_sparse_blocks) + fx.Int64(
                    query_block
                )
                partial_count = fx.Int32(fx.get_iter(kv_num_blocks)[count_index])
                full_count = fx.Int32(fx.get_iter(full_kv_num_blocks)[count_index])
                flag = fx.Int32(0)
                for item in range(partial_count):
                    index = count_index * fx.Int64(max_partial_blocks_limit) + fx.Int64(
                        item
                    )
                    if fx.Int32(fx.get_iter(kv_indices)[index]) == sparse_key:
                        flag = fx.Int32(1)
                for item in range(full_count):
                    index = count_index * fx.Int64(max_full_blocks_limit) + fx.Int64(
                        item
                    )
                    if fx.Int32(fx.get_iter(full_kv_indices)[index]) == sparse_key:
                        flag = fx.Int32(2)
                return flag

            if const_expr(not static_query_blocks):
                build_chunked_reverse_lists(
                    sparse_key,
                    block_flag,
                    flags,
                    partial_indices,
                    full_indices,
                    partial_counts,
                    full_counts,
                )
                return
            for step in fx.range_constexpr(
                (num_sparse_blocks + compute_threads - 1) // compute_threads
            ):
                query_block = tid + fx.Int32(step * compute_threads)
                if query_block < fx.Int32(num_sparse_blocks):
                    flags[query_block] = block_flag(query_block)
            fx.gpu.barrier()
            if tid == fx.Int32(0):
                num_partial = fx.Int32(0)
                num_full = fx.Int32(0)
                owner_index = fx.Int64(batch_head) * fx.Int64(
                    key_metadata_chunks
                ) + fx.Int64(sparse_key)
                list_offset = owner_index * fx.Int64(metadata_chunks)
                for compact_query in range(fx.Int32(num_sparse_blocks)):
                    compact_flag = fx.Int32(flags[compact_query])
                    if compact_flag == fx.Int32(1):
                        fx.get_iter(partial_indices)[
                            list_offset + fx.Int64(num_partial)
                        ] = compact_query
                        num_partial = num_partial + fx.Int32(1)
                    elif compact_flag == fx.Int32(2):
                        fx.get_iter(full_indices)[list_offset + fx.Int64(num_full)] = (
                            compact_query
                        )
                        num_full = num_full + fx.Int32(1)
                fx.get_iter(partial_counts)[owner_index] = num_partial
                fx.get_iter(full_counts)[owner_index] = num_full

        reverse_lists = (partial_indices, full_indices, partial_counts, full_counts)
        if const_expr(block_list_traversal):
            if const_expr(static_key_lists):
                if row_tile < fx.Int32(key_metadata_chunks):
                    build_reverse_lists(row_tile, *reverse_lists)
            else:
                for sparse_key in range(
                    row_tile, fx.Int32(key_metadata_chunks), fx.Int32(delta_grid)
                ):
                    build_reverse_lists(sparse_key, *reverse_lists)
                    # The next block rewrites the flags thread 0 just compacted.
                    fx.gpu.barrier()

    fused_grid = owner_count
    compute_num_threads = num_waves * WAVE_SIZE
    if block_list_traversal:
        compute_kernel = make_pipeline_kernel(locals(), _f32, bwd_exp2)
        compute_attrs = {}
    else:
        consumer_waves = 2
        dq_phases, dq_phase_tiles = split_dq_tiles(qk_head_dim, consumer_waves)
        dq_wave_tiles = dq_phases * dq_phase_tiles
        # With a Q-head group loop the extra AGPR-cached K step spills (D=192).
        cache_extra_key = qk_head_dim > 128 and query_group == 1
        # LLVM cannot merge table reads, so these builds read each score_mod
        # capture once per row or key of a tile.
        tile_capture_reads = (
            has_score_mod and capture_table is not None and producer_waves == 2
        )
        swap_banks = value_head_dim > qk_head_dim or (
            unmasked_traversal and qk_head_dim == value_head_dim
        )
        # A producer alone on its SIMD keeps up to 192 accumulators in VGPRs.
        vgpr_accumulators = (
            producer_waves == 2 and (qk_head_dim + value_head_dim) // 2 <= 192
        )
        key_bank = "v" if vgpr_accumulators or swap_banks else "a"
        value_bank = "v" if vgpr_accumulators or not swap_banks else "a"

        @fx.struct
        class SharedMemory:
            key: fx.Array[element_type, key_rows * qk_head_dim, 16]
            value: fx.Array[element_type, key_rows * value_head_dim, 16]
            query: fx.Array[element_type, 2 * step_rows * qk_head_dim, 16]
            grad_output: fx.Array[element_type, 2 * step_rows * value_head_dim, 16]
            ds: fx.Array[element_type, 2 * key_rows * ds_stride, 16]
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
                fused_grid,
                _,
                _,
                query_shape,
                key_shape,
                q_stride,
                k_stride,
                v_stride,
                grad_output_stride,
                grad_key_stride,
                grad_value_stride,
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
            batch_head = fx.Int32(fx.block_idx.z) * fx.Int32(cta_head_group) + fx.Int32(
                fx.block_idx.x
            )
            partial_slot = fx.Int32(fx.block_idx.y)
            tid = fx.Int32(fx.thread_idx.x)
            wave = fx.Int32(
                fx.rocdl.readfirstlane(
                    fx.Int32.ir_type, (tid // fx.Int32(WAVE_SIZE)).ir_value()
                )
            )
            shared = fx.SharedAllocator().allocate(SharedMemory).peek()
            if const_expr(direct_dq):
                dq_output_stride = runtime_strides(
                    static_strides["dq"], grad_query, sequence_length
                )

            def process_owner(
                owner,
                producer: fx.Constexpr[bool],
                thread_id,
                owner_head,
                initialize: fx.Constexpr[bool],
                column_slice: fx.Constexpr[int] = 0,
            ):
                key_chunks = range(
                    column_slice * key_slice_chunks,
                    (column_slice + 1) * key_slice_chunks,
                )
                value_chunks = range(
                    column_slice * value_slice_chunks,
                    (column_slice + 1) * value_slice_chunks,
                )
                batch_head = fx.Int32(
                    llvm.inline_asm(
                        fx.Int32.ir_type,
                        [owner_head.ir_value()],
                        "",
                        "=s,0",
                        has_side_effects=True,
                    )
                )
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
                            query, query_coord, (*query_shape, qk_head_dim), q_stride
                        )
                    if const_expr(key_views):
                        kview = make_global_view(
                            key, head_coord, (*key_shape, qk_head_dim), k_stride
                        )
                        vview = make_global_view(
                            value, head_coord, (*key_shape, value_head_dim), v_stride
                        )
                    if const_expr(query_views):
                        doview = make_global_view(
                            grad_output,
                            query_coord,
                            (*query_shape, value_head_dim),
                            grad_output_stride,
                        )
                    if const_expr(query_views and direct_dq):
                        dqview = make_global_view(
                            grad_query,
                            query_coord,
                            (*query_shape, qk_head_dim),
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
                                * qk_head_dim
                                * workspace_head_group,
                            ),
                            (
                                dq_partitions
                                * padded_sequence_length
                                * qk_head_dim
                                * workspace_head_group,
                                1,
                            ),
                        )
                    if const_expr(key_views):
                        dkview = make_global_view(
                            grad_key,
                            head_coord,
                            (*key_shape, qk_head_dim),
                            grad_key_stride,
                        )
                        dvview = make_global_view(
                            grad_value,
                            head_coord,
                            (*key_shape, value_head_dim),
                            grad_value_stride,
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
                            (
                                mask_buffer_0,
                                mask_buffer_1,
                                mask_buffer_2,
                                mask_buffer_3,
                            ),
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
                            mask_program_output,
                            mask_buffer_strides,
                            masks,
                            load_i32,
                            batch,
                            query_head,
                        )
                        if const_expr(query_group > 1 and not direct_dq):
                            packed_dq = fx.logical_divide(dqview, fx.make_layout(4, 1))

                bind_views(head, batch_head, query_group == 1, True)
                # Rebuild lane coordinates within each owner to bound register lifetimes.
                tid = fx.Int32(
                    llvm.inline_asm(
                        fx.Int32.ir_type,
                        [thread_id.ir_value()],
                        "",
                        "=v,0",
                        has_side_effects=True,
                    )
                ) & fx.Int32(compute_num_threads - 1)
                lane = tid % fx.Int32(WAVE_SIZE)
                wave = fx.Int32(
                    fx.rocdl.readfirstlane(
                        fx.Int32.ir_type, (tid // fx.Int32(WAVE_SIZE)).ir_value()
                    )
                )
                # The producers of one key tile share its rows.
                key_wave = wave
                if const_expr(column_slices > 1):
                    key_wave = fx.Int32(0)

                def pin_lane_coordinates():
                    # Likewise per query head, so staging addresses are not
                    # hoisted across the group loop and spilled.
                    nonlocal tid
                    tid = fx.Int32(
                        llvm.inline_asm(
                            fx.Int32.ir_type,
                            [thread_id.ir_value()],
                            "",
                            "=v,0",
                            has_side_effects=True,
                        )
                    ) & fx.Int32(compute_num_threads - 1)

                dq_copy = fx.make_copy_atom(dq_copy_op(), dq_workspace_dtype)
                dq_output_copy = fx.make_copy_atom(
                    fx.rocdl.BufferCopy64b(), element_type
                )
                read128 = fx.make_copy_atom(fx.UniversalCopy128b(), element_type)
                tr16 = fx.make_copy_atom(
                    fx.rocdl.cdna4.LDSReadTrans(16, 64), element_type
                )
                score_mma = fx.make_tiled_mma(
                    fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, element_type)),
                    fx.make_layout((1, 1, 1), (1, 1, 1)),
                )
                update_mma = fx.make_tiled_mma(
                    fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, element_type)),
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
                update_thread = update_mma.get_slice(lane)
                ccoords = score_thread.partition_C(
                    fx.make_view(0, fx.make_layout((16, 16), (1, 0)))
                )
                rowcoords = score_thread.partition_C(
                    fx.make_view(0, fx.make_layout((16, 16), (0, 1)))
                )
                update_bcoords = update_thread.partition_B(
                    fx.make_view(0, fx.make_layout((16, 16), (16, 1)))
                )
                dq_key_copy = fx.make_tiled_copy_A(tr16, score_mma).get_slice(lane)
                dq_ds_copy = fx.make_tiled_copy_B(tr16, score_mma).get_slice(lane)
                score_copy = fx.make_tiled_copy_A(read128, score_mma).get_slice(lane)
                score_b_copy = fx.make_tiled_copy_B(read128, score_mma).get_slice(lane)
                key_base = owner * fx.Int32(key_rows)
                key_pos = (
                    key_base
                    + key_wave * fx.Int32(key_groups * 16)
                    + fx.Int32(fx.get_scalar(rowcoords[0]))
                )
                klayout = make_bwd_shared_layout(key_rows, qk_head_dim)
                vlayout = make_bwd_shared_layout(key_rows, value_head_dim)
                qlayout = make_bwd_shared_layout(16, qk_head_dim)
                dolayout = make_bwd_shared_layout(16, value_head_dim)
                if const_expr(single_pass):
                    query_pair_layout = make_bwd_shared_layout(32, qk_head_dim)
                    grad_output_pair_layout = make_bwd_shared_layout(32, value_head_dim)

                def stage_operand(
                    view,
                    layout,
                    pointer,
                    first_row,
                    rows,
                    dimension,
                    row_stride,
                    query_rows: fx.Constexpr[bool] = None,
                ):
                    # The public BufferCopyLDS atom is synchronous with respect to the
                    # compiler's VMEM wait tracking. Asyncmark-native DMA preserves
                    # outstanding private-dQ loads at the cross-iteration boundary.
                    base_layout = fx.make_layout(
                        ((16, rows // 16), (32, dimension // 32)),
                        ((32, 16 * dimension), (1, 16 * 32)),
                    )
                    coordinates = fx.right_inverse(base_layout)
                    if const_expr(query_rows is None):
                        query_rows = rows == 16
                    staging_threads = (
                        producer_waves * WAVE_SIZE
                        if const_expr(query_rows)
                        else compute_num_threads
                    )
                    for step in fx.range_constexpr(
                        (rows * dimension // 8 + staging_threads - 1) // staging_threads
                    ):
                        linear = tid + fx.Int32(step * staging_threads)
                        if linear < fx.Int32(rows * dimension // 8):
                            physical = linear * fx.Int32(8)
                            unswizzled = physical ^ (
                                physical // fx.Int32(256) % fx.Int32(2) * fx.Int32(16)
                            )
                            logical = fx.Int32(
                                fx.get_scalar(fx.crd2idx(unswizzled, coordinates))
                            )
                            row = logical % fx.Int32(rows)
                            column = logical // fx.Int32(rows * 8)
                            if const_expr(query_rows):
                                fx.rocdl.raw_ptr_buffer_load_async_lds(
                                    fx.rocdl.get_buffer_rsrc(fx.get_iter(view)),
                                    fx.to_llvm_ptr(
                                        pointer + (linear - lane) * fx.Int32(8)
                                    ),
                                    fx.Int32(16).ir_value(),
                                    fx.Int32(
                                        (first_row + row) * fx.Int32(row_stride * 2)
                                        + column * fx.Int32(16)
                                    ).ir_value(),
                                    fx.Int32(0).ir_value(),
                                    fx.Int32(0).ir_value(),
                                )
                            elif first_row + row < fx.Int32(kv_sequence_length):
                                fx.rocdl.raw_ptr_buffer_load_async_lds(
                                    fx.rocdl.get_buffer_rsrc(fx.get_iter(view)),
                                    fx.to_llvm_ptr(
                                        pointer + (linear - lane) * fx.Int32(8)
                                    ),
                                    fx.Int32(16).ir_value(),
                                    fx.Int32(
                                        (first_row + row) * fx.Int32(row_stride * 2)
                                        + column * fx.Int32(16)
                                    ).ir_value(),
                                    fx.Int32(0).ir_value(),
                                    fx.Int32(0).ir_value(),
                                )
                            else:
                                destination = fx.logical_divide(
                                    fx.make_view(
                                        pointer, fx.make_layout(rows * dimension, 1)
                                    ),
                                    fx.make_layout(8, 1),
                                )
                                zeros = fx.make_rmem_tensor(8, element_type)
                                zeros.fill(0)
                                fx.copy(
                                    read128,
                                    zeros,
                                    fx.slice(destination, (None, linear)),
                                )

                def stage_query(query_tile, stage):
                    first_row = query_tile * fx.Int32(16)
                    if tid < fx.Int32(4):
                        source_index = query_tile * fx.Int32(4) + tid
                        for source_view, stats_offset in [
                            (lseview, 0),
                            (deltaview, 16),
                        ][int(q_ragged) :]:
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
                        shared.query.ptr + stage * fx.Int32(16 * qk_head_dim),
                        first_row,
                        16,
                        qk_head_dim,
                        q_stride[2],
                    )
                    stage_operand(
                        doview,
                        dolayout,
                        shared.grad_output.ptr + stage * fx.Int32(16 * value_head_dim),
                        first_row,
                        16,
                        value_head_dim,
                        grad_output_stride[2],
                    )
                    fx.rocdl.asyncmark()

                def stage_query_pair(pair, stage):
                    first_row = pair * fx.Int32(32)
                    stats = shared.stats.ptr + stage * fx.Int32(64)
                    if tid < fx.Int32(8):
                        source_index = pair * fx.Int32(8) + tid
                        for source_view, stats_offset in [
                            (lseview, 0),
                            (deltaview, 32),
                        ][int(q_ragged) :]:
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
                        shared.query.ptr + stage * fx.Int32(32 * qk_head_dim),
                        first_row,
                        32,
                        qk_head_dim,
                        q_stride[2],
                        True,
                    )
                    stage_operand(
                        doview,
                        grad_output_pair_layout,
                        shared.grad_output.ptr + stage * fx.Int32(32 * value_head_dim),
                        first_row,
                        32,
                        value_head_dim,
                        grad_output_stride[2],
                        True,
                    )
                    fx.rocdl.asyncmark()

                stage_operand(
                    kview,
                    klayout,
                    shared.key.ptr,
                    key_base,
                    key_rows,
                    qk_head_dim,
                    k_stride[2],
                )
                stage_operand(
                    vview,
                    vlayout,
                    shared.value.ptr,
                    key_base,
                    key_rows,
                    value_head_dim,
                    v_stride[2],
                )
                fx.rocdl.asyncmark()

                def load_score(pointer, layout, step):
                    tiles = fx.flat_divide(fx.make_view(pointer, layout), (16, 32))
                    tile = fx.slice(tiles, (None, None, 0, step))
                    fragment = score_thread.make_fragment_A(tile)
                    fx.copy(
                        read128,
                        score_copy.partition_S(tile),
                        score_copy.retile(fragment),
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
                    tile_pointer = pointer + fx.Int32(dim_chunk * 16 * 32)
                    tiles = fx.flat_divide(
                        fx.make_view(
                            tile_pointer,
                            fx.select(make_bwd_shared_layout(16, 32), [1, 0]),
                        ),
                        (32, 16),
                    )
                    tile = fx.slice(tiles, (None, None, 0, 0))
                    fragment = wide_thread.make_fragment_A(tile)
                    fx.copy(
                        tr16,
                        wide_a_copy.partition_S(tile),
                        wide_a_copy.retile(fragment),
                    )
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
                            fx.Int32(second_packed[part]).ir_value(),
                            False,
                            False,
                        )
                        low.append(
                            fx.Int32(llvm.extractvalue(fx.Int32.ir_type, swapped, [0]))
                        )
                        high.append(
                            fx.Int32(llvm.extractvalue(fx.Int32.ir_type, swapped, [1]))
                        )
                    return fx.Vector.from_elements(low + high, fx.Int32).bitcast(
                        element_type
                    )

                def store_wide_gradient(values, row_view, dimension: fx.Constexpr[int]):
                    words = values.bitcast(fx.Int32)
                    store = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), element_type)
                    destination = fx.logical_divide(row_view, fx.make_layout(8, 1))
                    for pair in fx.range_constexpr(2):
                        first = []
                        second = []
                        for word in fx.range_constexpr(2):
                            exchanged = fx.rocdl.permlane32_swap(
                                ir.Type.parse("!llvm.struct<(i32, i32)>"),
                                fx.Int32(words[pair * 4 + word]).ir_value(),
                                fx.Int32(words[pair * 4 + 2 + word]).ir_value(),
                                False,
                                False,
                            )
                            first.append(
                                fx.Int32(
                                    llvm.extractvalue(fx.Int32.ir_type, exchanged, [0])
                                )
                            )
                            second.append(
                                fx.Int32(
                                    llvm.extractvalue(fx.Int32.ir_type, exchanged, [1])
                                )
                            )
                        packed = fx.make_rmem_tensor(8, element_type)
                        packed.store(
                            fx.Vector.from_elements(first + second, fx.Int32)
                            .bitcast(element_type)
                            .ir_value()
                        )
                        column = fx.Int32(
                            dimension * 32 + pair * 16
                        ) + lane // fx.Int32(32) * fx.Int32(8)
                        fx.copy(
                            store,
                            packed,
                            fx.slice(destination, (None, column // fx.Int32(8))),
                        )

                def load_dq_operands(
                    stage, step, phase: fx.Constexpr[int], cached_keys
                ):
                    ds_pointer = shared.ds.ptr + stage * fx.Int32(key_rows * 24)
                    ds_tiles = fx.flat_divide(
                        fx.make_view(
                            ds_pointer, fx.make_layout((16, key_rows), (1, 24))
                        ),
                        (16, 32),
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
                if const_expr(query_group == 1 and not direct_dq):
                    packed_dq = fx.logical_divide(dqview, fx.make_layout(4, 1))

                def load_previous_dq_phase(
                    query_tile, phase: fx.Constexpr[int], previous=None
                ):
                    if const_expr(previous is None):
                        previous = previous_dq
                    if const_expr(not initialize):
                        for d in fx.range_constexpr(dq_phase_tiles):
                            segment_idx = phase * dq_phase_tiles + d
                            tile = (wave - fx.Int32(producer_waves)) * fx.Int32(
                                dq_wave_tiles
                            ) + fx.Int32(segment_idx)
                            offset = (
                                (
                                    (
                                        query_tile * fx.Int32(dq_partitions)
                                        + partial_slot
                                    )
                                    * fx.Int32(workspace_head_group)
                                    + query_batch_head % fx.Int32(workspace_head_group)
                                )
                                * fx.Int32(16 * qk_head_dim)
                                + tile * fx.Int32(256)
                                + lane * fx.Int32(4)
                            )
                            fx.copy(
                                dq_copy,
                                fx.slice(packed_dq, (None, offset // fx.Int32(4))),
                                previous[segment_idx],
                            )

                def load_previous_dq(query_tile):
                    for phase in fx.range_constexpr(dq_phases):
                        load_previous_dq_phase(query_tile, phase)

                def store_dq(
                    query_tile,
                    accumulator,
                    tile,
                    part: fx.Constexpr[int],
                    previous=None,
                ):
                    values = fx.Vector(accumulator.load()) * fx.Vector.filled(
                        4, softmax_scale, fx.Float32
                    )
                    if const_expr(not initialize):
                        if const_expr(previous is None):
                            previous = previous_dq
                        values = values + fx.Vector(previous[part].load()).to(
                            fx.Float32
                        )
                    if const_expr(direct_dq):
                        # Lane l holds row l % 16, columns 4 * (l // 16) + 0..3
                        # of its 16-column tile; dq_layout reads the same order.
                        output = fx.make_rmem_tensor(4, element_type)
                        output.store(values.to(element_type).ir_value())
                        row = query_tile * fx.Int32(16) + lane % fx.Int32(16)
                        column = tile * fx.Int32(16) + lane // fx.Int32(16) * fx.Int32(
                            4
                        )
                        destination = fx.logical_divide(
                            fx.slice(dqview, (row, None)), fx.make_layout(4, 1)
                        )
                        fx.copy(
                            dq_output_copy,
                            output,
                            fx.slice(destination, (None, column // fx.Int32(4))),
                        )
                        return
                    output = fx.make_rmem_tensor(4, dq_workspace_dtype)
                    output.store(values.to(dq_workspace_dtype).ir_value())
                    offset = (
                        (
                            (query_tile * fx.Int32(dq_partitions) + partial_slot)
                            * fx.Int32(workspace_head_group)
                            + query_batch_head % fx.Int32(workspace_head_group)
                        )
                        * fx.Int32(16 * qk_head_dim)
                        + tile * fx.Int32(256)
                        + lane * fx.Int32(4)
                    )
                    fx.copy(
                        dq_copy,
                        output,
                        fx.slice(packed_dq, (None, offset // fx.Int32(4))),
                    )

                def consume(
                    query_tile, stage, cached_keys, prefetch: fx.Constexpr[bool]
                ):
                    ds_operands = [
                        load_dq_operands(stage, step, 0, cached_keys)[0]
                        for step in fx.range_constexpr(key_rows // 32)
                    ]
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
                            load_previous_dq_phase(query_tile + fx.Int32(1), phase)
                            if const_expr(phase + 1 < dq_phases):
                                fx.rocdl.sched_barrier(0)

                def finish_stage():
                    # One common barrier publishes dS(i) and Q/dO(i+1); all waves attend.
                    fx.rocdl.wait_asyncmark(0)
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                    fx.rocdl.s_barrier()

                def update_in_bank(
                    accumulator, a, b, bank: fx.Constexpr[str], mma=None
                ):
                    # FlyDSL exposes tiled MMA but currently no accumulator-bank API.
                    # Keep only this register-class boundary around high-level gemm;
                    # replacing it changes AGPR/VGPR allocation and needs a perf A/B.
                    c = accumulator.load()
                    accumulator.store(
                        llvm.inline_asm(
                            c.type, [c], "", "=" + bank + ",0", has_side_effects=False
                        )
                    )
                    fx.gemm(
                        wide_mma if const_expr(mma is None) else mma,
                        accumulator,
                        a,
                        b,
                        accumulator,
                    )
                    c = accumulator.load()
                    accumulator.store(
                        llvm.inline_asm(
                            c.type, [c], "", "=" + bank + ",0", has_side_effects=False
                        )
                    )

                def producer_key(
                    step: fx.Constexpr[int],
                    group: fx.Constexpr[int],
                    cache,
                    masked: fx.Constexpr[bool],
                ):
                    if const_expr(step < 2):
                        return cache[0][step][group]
                    if const_expr(
                        not masked and cache_extra_key and step == qk_head_dim // 32 - 2
                    ):
                        return cache[2][group]
                    return load_score_operand(shared.key.ptr, klayout, step, group)

                def producer_value(
                    step: fx.Constexpr[int], group: fx.Constexpr[int], cache
                ):
                    if const_expr(step < 0):
                        return cache[1][step][group]
                    return load_score_operand(shared.value.ptr, vlayout, step, group)

                def cache_producer_operand(pointer, layout, steps):
                    cache = []
                    for step in fx.range_constexpr(steps):
                        fragments = []
                        for group in fx.range_constexpr(key_groups):
                            fragment = load_score_operand(pointer, layout, step, group)
                            packed = fragment.load()
                            fragment.store(
                                llvm.inline_asm(
                                    packed.type,
                                    [packed],
                                    "",
                                    "=a,0",
                                    has_side_effects=False,
                                )
                            )
                            fragments.append(fragment)
                        cache.append(fragments)
                    return cache

                def produce(
                    query_tile,
                    stage,
                    masked: fx.Constexpr[bool],
                    dk_wide,
                    dv_wide,
                    cached_values,
                ):
                    qpointer = shared.query.ptr + stage * fx.Int32(16 * qk_head_dim)
                    dopointer = shared.grad_output.ptr + stage * fx.Int32(
                        16 * value_head_dim
                    )
                    scores = [
                        fx.make_fragment_like(ccoords, fx.Float32)
                        for _ in fx.range_constexpr(key_groups)
                    ]
                    probabilities = [
                        fx.make_fragment_like(ccoords, fx.Float32)
                        for _ in fx.range_constexpr(key_groups)
                    ]
                    ps = [
                        fx.make_fragment_like(update_bcoords, element_type)
                        for _ in fx.range_constexpr(key_groups)
                    ]
                    dss = [
                        fx.make_fragment_like(update_bcoords, element_type)
                        for _ in fx.range_constexpr(key_groups)
                    ]
                    for group in fx.range_constexpr(key_groups):
                        scores[group].fill(0)
                    keys = [
                        producer_key(0, group, cached_values, masked)
                        for group in fx.range_constexpr(key_groups)
                    ]
                    query_operand = load_score(qpointer, qlayout, 0)
                    for step in fx.range_constexpr(qk_head_dim // 32):
                        if const_expr(step == min(3, qk_head_dim // 32 - 1)):
                            lse_values = []
                            for element in fx.range_constexpr(4):
                                local_row = fx.Int32(fx.get_scalar(ccoords[element]))
                                lse_value = fx.Float32(
                                    fx.ptr_load(
                                        shared.stats.ptr
                                        + stage * fx.Int32(32)
                                        + local_row
                                    )
                                )
                                if const_expr(not lse_in_log2):
                                    lse_value = lse_value * _f32(1.4426950408889634)
                                lse_values.append(lse_value.maximumf(_f32(LSE_FLOOR)))
                        if const_expr(step + 1 < qk_head_dim // 32):
                            next_keys = [
                                producer_key(step + 1, group, cached_values, masked)
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
                        if const_expr(step + 1 < qk_head_dim // 32):
                            keys = next_keys
                            query_operand = next_query
                    values_operand = [
                        producer_value(0, group, cached_values)
                        for group in fx.range_constexpr(key_groups)
                    ]
                    grad_output_operand = load_score(dopointer, dolayout, 0)
                    dps = [
                        fx.make_fragment_like(ccoords, fx.Float32)
                        for _ in fx.range_constexpr(key_groups)
                    ]
                    for group in fx.range_constexpr(key_groups):
                        dps[group].fill(0)
                    for group in fx.range_constexpr(key_groups):
                        fx.gemm(
                            score_mma,
                            dps[group],
                            grad_output_operand,
                            values_operand[group],
                            dps[group],
                        )
                    fx.rocdl.sched_barrier(0)
                    values_operand = [
                        producer_value(1, group, cached_values)
                        for group in fx.range_constexpr(key_groups)
                    ]
                    grad_output_operand = load_score(dopointer, dolayout, 1)
                    pre_mod_scores = []
                    keep_masks = []
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
                        readers = make_tile_score_mod_readers(
                            score_mod_aux, score_mod_table
                        )
                    for group in fx.range_constexpr(key_groups):
                        score_values = fx.Vector(scores[group].load())
                        values = []
                        for element in fx.range_constexpr(4):
                            local_row = fx.Int32(fx.get_scalar(ccoords[element]))
                            position = query_tile * fx.Int32(16) + local_row
                            lse = lse_values[element]
                            if const_expr(tile_capture_reads):
                                pre_mod = _f32(score_values[element]) * _f32(
                                    softmax_scale
                                )
                                post_mod = _f32(
                                    score_mod(
                                        pre_mod,
                                        batch,
                                        mod_head,
                                        positions[element],
                                        group_keys[group],
                                        readers,
                                    )
                                )
                            elif const_expr(has_score_mod):
                                pre_mod = _f32(score_values[element]) * _f32(
                                    softmax_scale
                                )
                                post_mod = _f32(
                                    score_mod(
                                        pre_mod,
                                        batch,
                                        mod_head,
                                        position,
                                        key_pos + fx.Int32(group * 16),
                                        score_mod_readers,
                                    )
                                )
                            if const_expr(has_score_mod):
                                pre_mod_scores.append(pre_mod)
                                probability = bwd_exp2(post_mod * _f32(_LOG2E) - lse)
                            else:
                                probability = bwd_exp2(
                                    _f32(score_values[element]) * _f32(scale_log2) - lse
                                )
                            # D192/V128 sits at the register limit: a tie there
                            # makes RegAllocGreedy fail the machine verifier, and
                            # none of its selects reads v_exp_f32 in the next slot.
                            if const_expr(
                                has_score_mod
                                and (qk_head_dim, value_head_dim) != (192, 128)
                            ):
                                probability = tied_nop(
                                    probability, 0, has_side_effects=False
                                )
                            keep = None
                            if masked:
                                keep = evaluate_mask(
                                    position, key_pos + fx.Int32(group * 16)
                                )
                                probability = keep.select(probability, _f32(0.0))
                            if const_expr(not direct_range_traversal or kv_ragged):
                                in_bounds = key_pos + fx.Int32(group * 16) < fx.Int32(
                                    kv_sequence_length
                                )
                                probability = in_bounds.select(probability, _f32(0.0))
                                if const_expr(has_score_mod):
                                    keep = (
                                        in_bounds if keep is None else keep & in_bounds
                                    )
                            if const_expr(q_ragged):
                                in_rows = position < fx.Int32(sequence_length)
                                probability = in_rows.select(probability, _f32(0.0))
                                if const_expr(has_score_mod):
                                    keep = in_rows if keep is None else keep & in_rows
                            keep_masks.append(keep)
                            values.append(probability)
                        probabilities[group].store(
                            fx.Vector.from_elements(values, fx.Float32).ir_value()
                        )
                    for step in fx.range_constexpr(1, value_head_dim // 32):
                        if const_expr(step + 1 < value_head_dim // 32):
                            next_values = [
                                producer_value(step + 1, group, cached_values)
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
                        if const_expr(step + 1 < value_head_dim // 32):
                            values_operand = next_values
                            grad_output_operand = next_grad_output
                    if const_expr(qk_head_dim // 32 - 1 in key_chunks):
                        query_tail = load_wide(qpointer, qlayout, qk_head_dim // 32 - 1)
                        packed_tail = query_tail.load()
                        query_tail.store(
                            llvm.inline_asm(
                                packed_tail.type,
                                [packed_tail],
                                "",
                                "=a,0",
                                has_side_effects=False,
                            )
                        )
                    for group in fx.range_constexpr(key_groups):
                        probability_values = fx.Vector(probabilities[group].load())
                        dp_values = fx.Vector(dps[group].load())
                        ds_values = []
                        for element in fx.range_constexpr(4):
                            local_row = fx.Int32(fx.get_scalar(ccoords[element]))
                            delta_value = fx.Float32(
                                fx.ptr_load(
                                    shared.stats.ptr
                                    + stage * fx.Int32(32)
                                    + fx.Int32(16)
                                    + local_row
                                )
                            )
                            ds_value = _f32(probability_values[element]) * (
                                _f32(dp_values[element]) - delta_value
                            )
                            if const_expr(tile_capture_reads):
                                ds_value = _f32(
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
                                ds_value = _f32(
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
                                    ds_value = keep.select(ds_value, _f32(0.0))
                            ds_values.append(ds_value)
                        ps[group].store(probability_values.to(element_type).ir_value())
                        ds_vector = fx.Vector.from_elements(ds_values, fx.Float32).to(
                            element_type
                        )
                        dss[group].store(ds_vector.ir_value())
                        if const_expr(column_slice == 0):
                            dspointer = (
                                shared.ds.ptr
                                + stage * fx.Int32(key_rows * 24)
                                + (key_wave * fx.Int32(key_groups) + fx.Int32(group))
                                * fx.Int32(384)
                            )
                            ds_offset = fx.Int32(
                                fx.get_scalar(rowcoords[0])
                            ) * fx.Int32(24) + fx.Int32(fx.get_scalar(ccoords[0]))
                            fx.ptr_store(ds_vector.ir_value(), dspointer + ds_offset)
                    wide_operand = wide_thread.partition_B(
                        fx.make_view(0, fx.make_layout((32, 16), (16, 1)))
                    )
                    ds_wide = fx.make_fragment_like(wide_operand, element_type)
                    p_wide = fx.make_fragment_like(wide_operand, element_type)
                    ds_wide.store(
                        swap16_vectors(
                            fx.Vector(dss[0].load()), fx.Vector(dss[1].load())
                        ).ir_value()
                    )
                    p_wide.store(
                        swap16_vectors(
                            fx.Vector(ps[0].load()), fx.Vector(ps[1].load())
                        ).ir_value()
                    )
                    qa = load_wide(qpointer, qlayout, key_chunks[0])
                    for d in fx.range_constexpr(
                        min(key_chunks[0], value_chunks[0]),
                        max(key_chunks[-1], value_chunks[-1]) + 1,
                    ):
                        prefetch = d + 1 in key_chunks[1:]
                        if const_expr(prefetch):
                            if const_expr(d + 2 == qk_head_dim // 32):
                                next_qa = query_tail
                            else:
                                next_qa = load_wide(qpointer, qlayout, d + 1)
                        if const_expr(d in value_chunks):
                            doa = load_wide(dopointer, dolayout, d)
                        if const_expr(d in key_chunks):
                            update_in_bank(dk_wide[d], qa, ds_wide, key_bank)
                        if const_expr(d in value_chunks):
                            update_in_bank(dv_wide[d], doa, p_wide, value_bank)
                        if const_expr(prefetch):
                            qa = next_qa

                if const_expr(single_pass):
                    # Row m of score tile s reads Q row 8 * (m // 4) + 4 * s + m % 4
                    # of the pair, so the two tiles' accumulator rows concatenate
                    # into the 32-row reduction operand of the dK/dV MFMAs.
                    pair_rows = fx.make_layout((4, 4, 2), (1, 8, 4))
                    query_score_layout = fx.composition(
                        query_pair_layout,
                        fx.make_tile(pair_rows, fx.make_layout(qk_head_dim, 1)),
                    )
                    grad_output_score_layout = fx.composition(
                        grad_output_pair_layout,
                        fx.make_tile(pair_rows, fx.make_layout(value_head_dim, 1)),
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
                        tr16,
                        dq_key_copy.partition_S(operand),
                        dq_key_copy.retile(fragment),
                    )
                    return fragment

                def produce_pair(
                    pair,
                    stage,
                    masked: fx.Constexpr[bool],
                    dk_tiles,
                    dv_tiles,
                    cached_values,
                ):
                    qpointer = shared.query.ptr + stage * fx.Int32(32 * qk_head_dim)
                    dopointer = shared.grad_output.ptr + stage * fx.Int32(
                        32 * value_head_dim
                    )
                    stats = shared.stats.ptr + stage * fx.Int32(64)
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
                    keys = producer_key(0, 0, cached_values, masked)
                    query_operands = [
                        load_pair_score(qpointer, query_score_layout, 0, tile)
                        for tile in fx.range_constexpr(2)
                    ]
                    for step in fx.range_constexpr(qk_head_dim // 32):
                        if const_expr(step == min(3, qk_head_dim // 32 - 1)):
                            lse_values = []
                            for tile in fx.range_constexpr(2):
                                for element in fx.range_constexpr(4):
                                    lse_value = fx.Float32(
                                        fx.ptr_load(stats + rows[tile][element])
                                    )
                                    if const_expr(not lse_in_log2):
                                        lse_value = lse_value * _f32(_LOG2E)
                                    lse_values.append(
                                        lse_value.maximumf(_f32(LSE_FLOOR))
                                    )
                        if const_expr(step + 1 < qk_head_dim // 32):
                            next_keys = producer_key(step + 1, 0, cached_values, masked)
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
                        if const_expr(step + 1 < qk_head_dim // 32):
                            keys = next_keys
                            query_operands = next_queries
                    values_operand = producer_value(0, 0, cached_values)
                    grad_output_operands = [
                        load_pair_score(dopointer, grad_output_score_layout, 0, tile)
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
                    fx.rocdl.sched_barrier(0)
                    if const_expr(value_head_dim > 32):
                        values_operand = producer_value(1, 0, cached_values)
                        grad_output_operands = [
                            load_pair_score(
                                dopointer, grad_output_score_layout, 1, tile
                            )
                            for tile in fx.range_constexpr(2)
                        ]
                    positions = [
                        [
                            pair * fx.Int32(32) + rows[tile][element]
                            for element in fx.range_constexpr(4)
                        ]
                        for tile in fx.range_constexpr(2)
                    ]
                    if const_expr(tile_capture_reads):
                        readers = make_tile_score_mod_readers(
                            score_mod_aux, score_mod_table
                        )
                    pre_mod_scores = []
                    keep_masks = []
                    probabilities = []
                    for tile in fx.range_constexpr(2):
                        score_values = fx.Vector(scores[tile].load())
                        for element in fx.range_constexpr(4):
                            position = positions[tile][element]
                            lse = lse_values[tile * 4 + element]
                            if const_expr(has_score_mod):
                                pre_mod = _f32(score_values[element]) * _f32(
                                    softmax_scale
                                )
                                post_mod = _f32(
                                    score_mod(
                                        pre_mod,
                                        batch,
                                        mod_head,
                                        position,
                                        key_pos,
                                        readers
                                        if const_expr(tile_capture_reads)
                                        else score_mod_readers,
                                    )
                                )
                                pre_mod_scores.append(pre_mod)
                                probability = bwd_exp2(post_mod * _f32(_LOG2E) - lse)
                                probability = tied_nop(
                                    probability, 0, has_side_effects=False
                                )
                            else:
                                probability = bwd_exp2(
                                    _f32(score_values[element]) * _f32(scale_log2) - lse
                                )
                            keep = None
                            if masked:
                                keep = evaluate_mask(position, key_pos)
                                probability = keep.select(probability, _f32(0.0))
                            if const_expr(not direct_range_traversal or kv_ragged):
                                in_bounds = key_pos < fx.Int32(kv_sequence_length)
                                probability = in_bounds.select(probability, _f32(0.0))
                                if const_expr(has_score_mod):
                                    keep = (
                                        in_bounds if keep is None else keep & in_bounds
                                    )
                            if const_expr(q_ragged):
                                in_rows = position < fx.Int32(sequence_length)
                                probability = in_rows.select(probability, _f32(0.0))
                                if const_expr(has_score_mod):
                                    keep = in_rows if keep is None else keep & in_rows
                            keep_masks.append(keep)
                            probabilities.append(probability)
                    for step in fx.range_constexpr(1, value_head_dim // 32):
                        if const_expr(step + 1 < value_head_dim // 32):
                            next_values = producer_value(step + 1, 0, cached_values)
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
                        if const_expr(step + 1 < value_head_dim // 32):
                            values_operand = next_values
                            grad_output_operands = next_grad_outputs
                    ds_values = []
                    for tile in fx.range_constexpr(2):
                        dp_values = fx.Vector(dps[tile].load())
                        for element in fx.range_constexpr(4):
                            index = tile * 4 + element
                            delta_value = fx.Float32(
                                fx.ptr_load(stats + fx.Int32(32) + rows[tile][element])
                            )
                            ds_value = probabilities[index] * (
                                _f32(dp_values[element]) - delta_value
                            )
                            if const_expr(has_score_mod):
                                ds_value = _f32(
                                    joint_mod(
                                        pre_mod_scores[index],
                                        batch,
                                        mod_head,
                                        positions[tile][element],
                                        key_pos,
                                        ds_value,
                                        readers
                                        if const_expr(tile_capture_reads)
                                        else score_mod_readers,
                                    )
                                )
                                keep = keep_masks[index]
                                if const_expr(keep is not None):
                                    ds_value = keep.select(ds_value, _f32(0.0))
                            ds_values.append(ds_value)
                    ds_vector = fx.Vector.from_elements(ds_values, fx.Float32).to(
                        element_type
                    )
                    fx.ptr_store(
                        ds_vector.ir_value(),
                        shared.ds.ptr
                        + stage * fx.Int32(key_rows * ds_stride)
                        + (
                            key_wave * fx.Int32(16)
                            + fx.Int32(fx.get_scalar(rowcoords[0]))
                        )
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
                    for dim_tile in fx.range_constexpr(qk_head_dim // 16):
                        update_in_bank(
                            dk_tiles[dim_tile],
                            load_pair_transposed(qpointer, query_pair_layout, dim_tile),
                            ds_operand,
                            key_bank,
                            score_mma,
                        )
                    for dim_tile in fx.range_constexpr(value_head_dim // 16):
                        update_in_bank(
                            dv_tiles[dim_tile],
                            load_pair_transposed(
                                dopointer, grad_output_pair_layout, dim_tile
                            ),
                            p_operand,
                            value_bank,
                            score_mma,
                        )

                def consume_pair(
                    pair, stage, cached_keys, prefetch: fx.Constexpr[bool]
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
                                    query_tile + fx.Int32(2), phase, previous_pair[tile]
                                )
                                if const_expr(phase + 1 < dq_phases):
                                    fx.rocdl.sched_barrier(0)

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
                    masked: fx.Constexpr[bool],
                    producer: fx.Constexpr[bool],
                    dk_wide,
                    dv_wide,
                    cached_keys,
                    cached_values,
                ):
                    def query_at(index):
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
                        fx.rocdl.wait_asyncmark(0)
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        fx.gpu.barrier()
                        extra_keys = []
                        if const_expr(producer and not masked and cache_extra_key):
                            for group in fx.range_constexpr(key_groups):
                                fragment = load_score_operand(
                                    shared.key.ptr,
                                    klayout,
                                    qk_head_dim // 32 - 2,
                                    group,
                                )
                                packed = fragment.load()
                                fragment.store(
                                    llvm.inline_asm(
                                        packed.type,
                                        [packed],
                                        "",
                                        "=a,0",
                                        has_side_effects=False,
                                    )
                                )
                                extra_keys.append(fragment)
                        producer_cache = (
                            [cached_values[0], cached_values[1], extra_keys]
                            if const_expr(producer)
                            else []
                        )
                        if const_expr(producer):
                            if count > fx.Int32(1):
                                stage_step(query_at(fx.Int32(1)), fx.Int32(1))
                            produce_step(
                                query_at(fx.Int32(0)),
                                fx.Int32(0),
                                masked,
                                dk_wide,
                                dv_wide,
                                producer_cache,
                            )
                        finish_stage()
                        for index in range(fx.Int32(1), count, fx.Int32(1)):
                            stage = index % fx.Int32(2)
                            if const_expr(producer):
                                if index + fx.Int32(1) < count:
                                    stage_step(
                                        query_at(index + fx.Int32(1)),
                                        fx.Int32(1) - stage,
                                    )
                                produce_step(
                                    query_at(index),
                                    stage,
                                    masked,
                                    dk_wide,
                                    dv_wide,
                                    producer_cache,
                                )
                            else:
                                consume_step(
                                    query_at(index - fx.Int32(1)),
                                    fx.Int32(1) - stage,
                                    cached_keys,
                                    True,
                                )
                            finish_stage()
                        if const_expr(not producer):
                            consume_step(
                                query_at(count - fx.Int32(1)),
                                (count - fx.Int32(1)) % fx.Int32(2),
                                cached_keys,
                                False,
                            )
                        # A subsequent masked/full list may reuse either dS stage.
                        fx.rocdl.s_waitcnt(lgkmcnt=0)
                        fx.rocdl.s_barrier()

                def execute_runs(
                    producer: fx.Constexpr[bool],
                    dk_wide,
                    dv_wide,
                    cached_keys,
                    cached_values,
                ):
                    def steps():
                        tiles = fx.Int32(query_tiles)
                        return (
                            (tiles + fx.Int32(1)) // fx.Int32(2)
                            if const_expr(single_pass)
                            else tiles
                        )

                    if const_expr(unmasked_traversal):
                        run(
                            steps(),
                            fx.Int32(0),
                            False,
                            producer,
                            dk_wide,
                            dv_wide,
                            cached_keys,
                            cached_values,
                        )
                    elif const_expr(direct_range_traversal):
                        first_query = key_base // fx.Int32(step_rows)
                        masked_count = fx.min(
                            fx.Int32(key_rows // step_rows),
                            steps() - first_query,
                        )
                        run(
                            masked_count,
                            first_query,
                            True,
                            producer,
                            dk_wide,
                            dv_wide,
                            cached_keys,
                            cached_values,
                        )
                        run(
                            steps() - first_query - masked_count,
                            first_query + masked_count,
                            False,
                            producer,
                            dk_wide,
                            dv_wide,
                            cached_keys,
                            cached_values,
                        )

                def execute_query_heads(
                    producer: fx.Constexpr[bool],
                    dk_wide,
                    dv_wide,
                    cached_keys,
                    cached_values,
                ):
                    if const_expr(query_group == 1):
                        execute_runs(
                            producer, dk_wide, dv_wide, cached_keys, cached_values
                        )
                    else:
                        # K/V stay staged while every Q head of the group
                        # accumulates into the same dK/dV registers.
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
                            execute_runs(
                                producer, dk_wide, dv_wide, cached_keys, cached_values
                            )

                fx.rocdl.wait_asyncmark(0)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                fx.rocdl.s_barrier()

                def cache_consumer_keys():
                    key_tiles = fx.flat_divide(
                        fx.make_view(shared.key.ptr, fx.select(klayout, [1, 0])),
                        (16, 32),
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
                        for _ in fx.range_constexpr(qk_head_dim // 16)
                    ]
                    dv_tiles = [
                        fx.make_fragment_like(ccoords, fx.Float32)
                        for _ in fx.range_constexpr(value_head_dim // 16)
                    ]
                    for gradient_tile in dk_tiles + dv_tiles:
                        gradient_tile.fill(0)
                    cached_values = [
                        cache_producer_operand(shared.key.ptr, klayout, 2),
                        cache_producer_operand(shared.value.ptr, vlayout, 0),
                    ]
                    execute_query_heads(True, dk_tiles, dv_tiles, [], cached_values)
                    gradient_store = fx.make_copy_atom(
                        fx.rocdl.BufferCopy64b(), element_type
                    )
                    gradient_column = fx.Int32(fx.get_scalar(ccoords[0]))
                    if key_pos < fx.Int32(kv_sequence_length):
                        for gradient_tiles, gradient_view, gradient_scale in (
                            (dk_tiles, dkview, softmax_scale),
                            (dv_tiles, dvview, None),
                        ):
                            gradient_row = fx.logical_divide(
                                fx.slice(gradient_view, (key_pos, None)),
                                fx.make_layout(4, 1),
                            )
                            for dim_tile in fx.range_constexpr(len(gradient_tiles)):
                                gradient_values = fx.Vector(
                                    gradient_tiles[dim_tile].load()
                                )
                                if const_expr(gradient_scale is not None):
                                    gradient_values = (
                                        gradient_values
                                        * fx.Vector.filled(
                                            4, gradient_scale, fx.Float32
                                        )
                                    )
                                gradient_output = fx.make_rmem_tensor(4, element_type)
                                gradient_output.store(
                                    gradient_values.to(element_type).ir_value()
                                )
                                fx.copy(
                                    gradient_store,
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
                        for d in fx.range_constexpr(qk_head_dim // 32)
                    ]
                    dv_wide = [
                        fx.make_fragment_like(wide_dim_coords, fx.Float32)
                        if d in value_chunks
                        else None
                        for d in fx.range_constexpr(value_head_dim // 32)
                    ]
                    for d in key_chunks:
                        dk_wide[d].fill(0)
                    for d in value_chunks:
                        dv_wide[d].fill(0)
                    cached_values = [
                        cache_producer_operand(shared.key.ptr, klayout, 2),
                        cache_producer_operand(shared.value.ptr, vlayout, 0),
                    ]
                    execute_query_heads(True, dk_wide, dv_wide, [], cached_values)
                    wide_key_pos = (
                        key_base
                        + key_wave * fx.Int32(key_groups * 16)
                        + fx.Int32(fx.get_scalar(wide_row_coords[0]))
                    )
                    if wide_key_pos < fx.Int32(kv_sequence_length):
                        for d in key_chunks:
                            values = (
                                fx.Vector(dk_wide[d].load())
                                * fx.Vector.filled(16, softmax_scale, fx.Float32)
                            ).to(element_type)
                            store_wide_gradient(
                                values, fx.slice(dkview, (wide_key_pos, None)), d
                            )
                        for d in value_chunks:
                            values = fx.Vector(dv_wide[d].load()).to(element_type)
                            store_wide_gradient(
                                values, fx.slice(dvview, (wide_key_pos, None)), d
                            )

                else:
                    cached_keys = cache_consumer_keys()
                    execute_query_heads(False, [], [], cached_keys, [])

                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                fx.gpu.barrier()

            def owner_work(
                owner, producer: fx.Constexpr[bool], initialize: fx.Constexpr[bool]
            ):
                if const_expr(producer):
                    if const_expr(column_slices > 1):
                        if wave == fx.Int32(0):
                            process_owner(owner, True, tid, batch_head, initialize, 0)
                        else:
                            process_owner(owner, True, tid, batch_head, initialize, 1)
                    else:
                        process_owner(owner, True, tid, batch_head, initialize)
                else:
                    process_owner(owner, False, tid, batch_head, initialize)

            def remaining_owners(producer: fx.Constexpr[bool]):
                for round_index in range(
                    fx.Int32(1),
                    fx.Int32((fused_grid + dq_partitions - 1) // dq_partitions),
                ):
                    round = round_index
                    slot = (round % fx.Int32(2) == fx.Int32(0)).select(
                        partial_slot, fx.Int32(dq_partitions - 1) - partial_slot
                    )
                    owner = round * fx.Int32(dq_partitions) + slot
                    if owner < fx.Int32(fused_grid):
                        owner_work(owner, producer, False)

            if const_expr(accumulate_every_owner):
                for round_index in range(
                    fx.Int32((fused_grid + dq_partitions - 1) // dq_partitions)
                ):
                    slot = (round_index % fx.Int32(2) == fx.Int32(0)).select(
                        partial_slot, fx.Int32(dq_partitions - 1) - partial_slot
                    )
                    owner = round_index * fx.Int32(dq_partitions) + slot
                    if owner < fx.Int32(fused_grid):
                        if wave < fx.Int32(producer_waves):
                            owner_work(owner, True, False)
                        else:
                            owner_work(owner, False, False)
            else:

                def slot_owners():
                    if wave < fx.Int32(producer_waves):
                        owner_work(partial_slot, True, True)
                        if const_expr(not direct_dq):
                            remaining_owners(True)
                    else:
                        owner_work(partial_slot, False, True)
                        if const_expr(not direct_dq):
                            remaining_owners(False)

                if const_expr(runtime_owners):
                    if partial_slot < fx.Int32(fused_grid):
                        slot_owners()
                else:
                    slot_owners()

        compute_attrs = {"llvm.passthrough": [["amdgpu-agpr-alloc", "256"]]}

    @flyc.kernel
    def dq_layout_kernel(
        grad_query: fx.Tensor,
        grad_query_workspace: fx.Tensor,
        kv_num_blocks: fx.Tensor,
        kv_indices: fx.Tensor,
        full_kv_num_blocks: fx.Tensor,
        full_kv_indices: fx.Tensor,
        key: key_argument,
    ):
        (
            batch_size,
            sequence_length,
            padded_sequence_length,
            num_sparse_blocks,
            batch_heads,
        ) = resolve_query(grad_query)[:5]
        owner_count = resolve_key(key)[1]
        grad_query_stride = runtime_strides(
            static_strides["dq"], grad_query, sequence_length
        )
        max_partial_blocks_limit, max_full_blocks_limit = resolve_block_limits(
            kv_indices, full_kv_indices
        )
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
            (fx.Int64(batch_head // fx.Int32(workspace_head_group)), None),
            (
                (batch_heads + workspace_head_group - 1) // workspace_head_group,
                dq_partitions
                * padded_sequence_length
                * qk_head_dim
                * workspace_head_group,
            ),
            (
                dq_partitions
                * padded_sequence_length
                * qk_head_dim
                * workspace_head_group,
                1,
            ),
        )
        lane = tid % fx.Int32(WAVE_SIZE)
        wave = tid // fx.Int32(WAVE_SIZE)
        query_base = query_tile * fx.Int32(64) + wave * fx.Int32(16)
        row_base = lane // fx.Int32(16) * fx.Int32(4)
        load = fx.make_copy_atom(dq_copy_op(), dq_workspace_dtype)
        store = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), element_type)
        packed_source = fx.logical_divide(
            fx.make_view(
                fx.get_iter(source_view),
                fx.make_layout(
                    dq_partitions
                    * padded_sequence_length
                    * qk_head_dim
                    * workspace_head_group,
                    1,
                ),
            ),
            fx.make_layout(4, 1),
        )
        layout_blocks = (qk_head_dim + 63) // 64

        def block_lanes(block):
            return min(16, (qk_head_dim - block * 64) // 4)

        def layout_column(block):
            # Lanes past a partial block re-read its columns and skip the store.
            return lane % fx.Int32(block_lanes(block)) * fx.Int32(4) + fx.Int32(
                block * 64
            )

        fragments = [
            fx.make_rmem_tensor(4, fx.Float32)
            for _ in fx.range_constexpr(4 * layout_blocks)
        ]
        active_slots = fx.Int32(-1)
        if const_expr(block_list_traversal and dq_partitions <= 32):
            mask_batch = fx.Int32(0) if const_expr(block_mask_batch == 1) else batch
            if const_expr(block_mask_heads == 1):
                mask_head = fx.Int32(0)
            elif const_expr(block_mask_heads == num_heads):
                mask_head = head
            else:
                mask_head = head // fx.Int32(query_group)
            mask_group = mask_batch * fx.Int32(block_mask_heads) + mask_head
            query_block = query_tile // fx.Int32(sparse_q_block_size // 64)
            mask_index = mask_group * fx.Int32(num_sparse_blocks) + query_block
            owners_per_block = sparse_kv_block_size // key_rows

            def mark_owner_slots(slots, key_block):
                for part in fx.range_constexpr(owners_per_block):
                    owner = key_block
                    if const_expr(owners_per_block > 1):
                        owner = key_block * fx.Int32(owners_per_block) + fx.Int32(part)
                    slot = owner % fx.Int32(dq_partitions)
                    bit = fx.Int32(1) << slot
                    if const_expr(runtime_owners or owner_count % owners_per_block):
                        bit = (owner < fx.Int32(owner_count)).select(bit, fx.Int32(0))
                    slots = slots | bit
                return slots

            active_slots = fx.Int32(0)
            count = fx.Int32(fx.get_iter(kv_num_blocks)[mask_index])
            for item in range(count):
                key_block = fx.Int32(
                    fx.get_iter(kv_indices)[
                        mask_index * fx.Int32(max_partial_blocks_limit) + item
                    ]
                )
                active_slots = mark_owner_slots(active_slots, key_block)
            count = fx.Int32(fx.get_iter(full_kv_num_blocks)[mask_index])
            for item in range(count):
                key_block = fx.Int32(
                    fx.get_iter(full_kv_indices)[
                        mask_index * fx.Int32(max_full_blocks_limit) + item
                    ]
                )
                active_slots = mark_owner_slots(active_slots, key_block)
        if const_expr(block_list_traversal):
            for accumulator in fragments:
                accumulator.fill(0)
        for owner in fx.range_constexpr(dq_partitions):
            valid_owner = fx.Int32(1)
            if const_expr(block_list_traversal and dq_partitions <= 32):
                valid_owner = (active_slots & fx.Int32(1 << owner)) != fx.Int32(0)
            if const_expr(direct_range_traversal and owner > 0):
                valid_owner = query_base >= fx.Int32(owner * key_rows)
            if const_expr(runtime_owners and not block_list_traversal and owner > 0):
                owned = fx.Int32(owner) < owner_count
                valid_owner = (
                    valid_owner & owned if const_expr(direct_range_traversal) else owned
                )
            if valid_owner:
                loaded = []
                for block in fx.range_constexpr(layout_blocks):
                    column = layout_column(block)
                    workspace_offset = (
                        query_base
                        * fx.Int32(dq_partitions * qk_head_dim * workspace_head_group)
                        + batch_head
                        % fx.Int32(workspace_head_group)
                        * fx.Int32(16 * qk_head_dim)
                        + column // fx.Int32(16) * fx.Int32(256)
                        + (
                            column % fx.Int32(16) // fx.Int32(4) * fx.Int32(16)
                            + row_base
                        )
                        * fx.Int32(4)
                    )
                    workspace_offset = workspace_offset + owner * fx.Int32(
                        16 * qk_head_dim * workspace_head_group
                    )
                    for row in fx.range_constexpr(4):
                        fragment = fx.make_rmem_tensor(4, dq_workspace_dtype)
                        fx.copy(
                            load,
                            fx.slice(
                                packed_source,
                                (None, workspace_offset // fx.Int32(4) + fx.Int32(row)),
                            ),
                            fragment,
                        )
                        loaded.append(fragment)
                for index in fx.range_constexpr(4 * layout_blocks):
                    accumulator = fragments[index]
                    if const_expr(owner == 0):
                        accumulator.store(
                            fx.Vector(loaded[index].load()).to(fx.Float32).ir_value()
                        )
                    else:
                        accumulator.store(
                            (
                                fx.Vector(accumulator.load())
                                + fx.Vector(loaded[index].load()).to(fx.Float32)
                            ).ir_value()
                        )

        def store_block(block, column):
            for row in fx.range_constexpr(4):
                fragment = fragments[block * 4 + row]
                output = fx.make_rmem_tensor(4, element_type)
                output.store(fx.Vector(fragment.load()).to(element_type).ir_value())
                destination = fx.logical_divide(
                    fx.slice(view, (query_base + row_base + fx.Int32(row), None)),
                    fx.make_layout(4, 1),
                )
                fx.copy(
                    store, output, fx.slice(destination, (None, column // fx.Int32(4)))
                )

        for block in fx.range_constexpr(layout_blocks):
            column = layout_column(block)
            if const_expr(block_lanes(block) == 16):
                store_block(block, column)
            else:
                if lane % fx.Int32(16) < fx.Int32(block_lanes(block)):
                    store_block(block, column)

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
        _ = mod_key  # Closure scalars key FlyDSL's JIT cache; mod bodies do not.
        launch_shape = resolve_query(query)
        sizes_key = key if runtime_owners else None
        delta_kernel(
            attention_output,
            grad_output,
            delta,
            grad_query_workspace,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            partial_kv_counts,
            partial_kv_indices,
            full_kv_counts,
            full_kv_indices_transposed,
            sizes_key,
        ).launch(
            grid=(launch_shape[5] * launch_shape[4], 1, 1),
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
            grad_query if direct_dq else grad_query_workspace,
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
            grid=(cta_head_group, dq_partitions, launch_shape[7] // cta_head_group),
            block=(compute_num_threads, 1, 1),
            stream=stream,
        )
        if const_expr(not direct_dq):
            dq_layout_kernel(
                grad_query,
                grad_query_workspace,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                sizes_key,
            ).launch(
                grid=(launch_shape[2] // 64, launch_shape[4], 1),
                block=(256, 1, 1),
                stream=stream,
            )

    if not block_list_traversal:
        _launch.compile_hints["waves_per_eu"] = 1
    _launch.compile_hints["llvm_options"] = {
        "amdgpu-mfma-vgpr-form": True,
        "disable-machine-licm": True,
    }
    return _launch
