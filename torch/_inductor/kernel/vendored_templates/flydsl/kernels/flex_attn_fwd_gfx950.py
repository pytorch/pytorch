# mypy: allow-untyped-defs

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr
from flydsl.expr.typing import Vector as Vec

from .flex_attn_bwd_utils import (
    forward_head_dims,
    single_pass_value_shards,
    WIDE_HEAD_DIMS,
)
from .flex_attn_plan import _is_wide_head, forward_plan
from .flex_attn_target import resolve_target, WAVE_SIZE
from .flex_attn_utils import (
    capture_extent_count,
    capture_table_layout,
    ceil_div,
    ELEMENT_TYPES,
    fast_exp2,
    is_sliding_window_mask_program,
    make_capture_buffers,
    make_global_view,
    make_mask_evaluator,
    make_qk_shared_layout,
    make_score_mod_readers,
    make_tile_score_mod_readers,
    make_value_shared_layout,
    mask_capture_sizes,
    mask_element_value,
    mask_instruction_operands,
    optional_int,
    resolve_capture_extents,
    runtime_extent,
    runtime_strides,
    schedule_fwd_pv_pipeline,
    schedule_fwd_qk_pipeline,
    schedule_fwd_softmax_pipeline,
    tied_nop,
    tile_axes,
    tile_grid,
    validate_score_mod_args,
)


_LOG2E = 1.4426950408889634
# Element type and bits per load of four consecutive keys of a table build's
# mask capture; int64 takes two loads.
_VECTOR_MASK_LOADS = {
    "bool": (fx.Uint8, 32),
    "uint8": (fx.Uint8, 32),
    "int8": (fx.Int8, 32),
    "int16": (fx.Int16, 64),
    "float16": (fx.Float16, 64),
    "bfloat16": (fx.BFloat16, 64),
    "float32": (fx.Float32, 128),
    "int64": (fx.Int64, 128),
}
_LN2 = math.log(2.0)
# Keep masked exponentials finite under fast math; empty rows store -inf explicitly.
_NEG_BIG = -1.0e30
# Keys past Sk score far below _NEG_BIG, so exp2 flushes them to zero even in
# rows with no other key, without keeping another mask live.
_NEG_PAST_END = -3.0e38


def _f32(value):
    return fx.Float32(value)


def _exp2(value):
    return fast_exp2(_f32(value))


def _maximum(lhs, rhs):
    return lhs.maximumf(rhs)


def _store_output_fragments(
    final_output,
    inverse_vec,
    output_row,
    lane_half,
    output_copy,
    output_chunks,
    mfma_tile_size,
    element_type,
    chunk_columns=None,
    output_columns=None,
):
    for d_chunk in fx.range_constexpr(output_chunks):
        normalized = Vec(final_output[d_chunk]) * inverse_vec
        # Each column group stores 16 columns; skip those past output_columns.
        column_groups = 2
        if output_columns is not None:
            column_groups = min(
                max((output_columns - d_chunk * mfma_tile_size) // 16, 0), 2
            )
        for column_group in fx.range_constexpr(column_groups):
            packed = (
                Vec.from_elements(
                    [normalized[column_group * 8 + i] for i in fx.range_constexpr(8)],
                    fx.Float32,
                )
                .to(element_type)
                .bitcast(fx.Int32)
            )
            lower = []
            upper = []
            for i in fx.range_constexpr(2):
                lhs = fx.Int32(packed[i]).ir_value()
                rhs = fx.Int32(packed[i + 2]).ir_value()
                swapped = fx.rocdl.permlane32_swap(
                    ir.Type.parse("!llvm.struct<(i32, i32)>"),
                    lhs,
                    rhs,
                    False,
                    True,
                )
                lower.append(
                    fx.Int32(llvm.extractvalue(fx.Int32.ir_type, swapped, [0]))
                )
                upper.append(
                    fx.Int32(llvm.extractvalue(fx.Int32.ir_type, swapped, [1]))
                )
            values = Vec.from_elements([*lower, *upper], fx.Int32).bitcast(element_type)
            if chunk_columns is None:
                column = fx.Int32(
                    d_chunk * mfma_tile_size + column_group * 16
                ) + lane_half * fx.Int32(8)
            else:
                column = chunk_columns[d_chunk] + (
                    fx.Int32(column_group * 16) + lane_half * fx.Int32(8)
                )
            fragment = fx.make_rmem_tensor(8, element_type)
            fragment.store(values)
            fx.copy(
                output_copy,
                fragment,
                fx.slice(output_row, (None, column // fx.Int32(8))),
            )
            fx.rocdl.sched_barrier(0)


def _store_partial_output_fragments(
    final_output,
    inverse_vec,
    output_source,
    accumulator_coordinates,
    output_chunks,
    mfma_tile_size,
):
    # Each lane holds runs of four consecutive accumulator rows.
    output_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    output_row = fx.logical_divide(output_source, fx.make_layout(4, 1))
    for d_chunk in fx.range_constexpr(output_chunks):
        normalized = Vec(final_output[d_chunk]) * inverse_vec
        for group in fx.range_constexpr(4):
            column = fx.Int32(d_chunk * mfma_tile_size) + fx.Int32(
                fx.get_scalar(accumulator_coordinates[group * 4])
            )
            fragment = fx.make_rmem_tensor(4, fx.Float32)
            fragment.store(
                Vec.from_elements(
                    [normalized[group * 4 + i] for i in fx.range_constexpr(4)],
                    fx.Float32,
                )
            )
            fx.copy(
                output_copy,
                fragment,
                fx.slice(output_row, (None, column // fx.Int32(4))),
            )


def _make_forward_launch(
    kernel, *, grid, num_threads, mask_buffer_count, waves_per_eu, mod_key=None
):
    def launch_kernel(inputs, mask_buffers, output, stream):
        _ = mod_key  # Closure scalars key FlyDSL's JIT cache; mod bodies do not.
        buffers = (*mask_buffers, *([inputs[5]] * (4 - mask_buffer_count)))
        kernel(*inputs, *buffers, output).launch(
            grid=grid(inputs[0]) if callable(grid) else grid,
            block=(num_threads, 1, 1),
            stream=stream,
        )

    # Each JIT entry point exposes only its live mask buffers. The internal
    # placeholder slots alias kv_num_blocks and must remain unused above count.
    if mask_buffer_count == 0:

        @flyc.jit
        def launch(
            query: fx.Tensor,
            key: fx.Tensor,
            value: fx.Tensor,
            logsumexp: fx.Tensor,
            max_scores: fx.Tensor,
            kv_num_blocks: fx.Tensor,
            kv_indices: fx.Tensor,
            full_kv_num_blocks: fx.Tensor,
            full_kv_indices: fx.Tensor,
            output: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                (
                    query,
                    key,
                    value,
                    logsumexp,
                    max_scores,
                    kv_num_blocks,
                    kv_indices,
                    full_kv_num_blocks,
                    full_kv_indices,
                ),
                (),
                output,
                stream,
            )

    elif mask_buffer_count == 1:

        @flyc.jit
        def launch(
            query: fx.Tensor,
            key: fx.Tensor,
            value: fx.Tensor,
            logsumexp: fx.Tensor,
            max_scores: fx.Tensor,
            kv_num_blocks: fx.Tensor,
            kv_indices: fx.Tensor,
            full_kv_num_blocks: fx.Tensor,
            full_kv_indices: fx.Tensor,
            mask_buffer_0: fx.Tensor,
            output: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                (
                    query,
                    key,
                    value,
                    logsumexp,
                    max_scores,
                    kv_num_blocks,
                    kv_indices,
                    full_kv_num_blocks,
                    full_kv_indices,
                ),
                (mask_buffer_0,),
                output,
                stream,
            )

    elif mask_buffer_count == 2:

        @flyc.jit
        def launch(
            query: fx.Tensor,
            key: fx.Tensor,
            value: fx.Tensor,
            logsumexp: fx.Tensor,
            max_scores: fx.Tensor,
            kv_num_blocks: fx.Tensor,
            kv_indices: fx.Tensor,
            full_kv_num_blocks: fx.Tensor,
            full_kv_indices: fx.Tensor,
            mask_buffer_0: fx.Tensor,
            mask_buffer_1: fx.Tensor,
            output: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                (
                    query,
                    key,
                    value,
                    logsumexp,
                    max_scores,
                    kv_num_blocks,
                    kv_indices,
                    full_kv_num_blocks,
                    full_kv_indices,
                ),
                (
                    mask_buffer_0,
                    mask_buffer_1,
                ),
                output,
                stream,
            )

    elif mask_buffer_count == 3:

        @flyc.jit
        def launch(
            query: fx.Tensor,
            key: fx.Tensor,
            value: fx.Tensor,
            logsumexp: fx.Tensor,
            max_scores: fx.Tensor,
            kv_num_blocks: fx.Tensor,
            kv_indices: fx.Tensor,
            full_kv_num_blocks: fx.Tensor,
            full_kv_indices: fx.Tensor,
            mask_buffer_0: fx.Tensor,
            mask_buffer_1: fx.Tensor,
            mask_buffer_2: fx.Tensor,
            output: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                (
                    query,
                    key,
                    value,
                    logsumexp,
                    max_scores,
                    kv_num_blocks,
                    kv_indices,
                    full_kv_num_blocks,
                    full_kv_indices,
                ),
                (
                    mask_buffer_0,
                    mask_buffer_1,
                    mask_buffer_2,
                ),
                output,
                stream,
            )

    else:

        @flyc.jit
        def launch(
            query: fx.Tensor,
            key: fx.Tensor,
            value: fx.Tensor,
            logsumexp: fx.Tensor,
            max_scores: fx.Tensor,
            kv_num_blocks: fx.Tensor,
            kv_indices: fx.Tensor,
            full_kv_num_blocks: fx.Tensor,
            full_kv_indices: fx.Tensor,
            mask_buffer_0: fx.Tensor,
            mask_buffer_1: fx.Tensor,
            mask_buffer_2: fx.Tensor,
            mask_buffer_3: fx.Tensor,
            output: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            launch_kernel(
                (
                    query,
                    key,
                    value,
                    logsumexp,
                    max_scores,
                    kv_num_blocks,
                    kv_indices,
                    full_kv_num_blocks,
                    full_kv_indices,
                ),
                (
                    mask_buffer_0,
                    mask_buffer_1,
                    mask_buffer_2,
                    mask_buffer_3,
                ),
                output,
                stream,
            )

    launch.compile_hints = {"waves_per_eu": waves_per_eu}
    return launch


def build_flex_attn_fwd_module(
    *,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_q: int,
    seq_kv: int,
    qk_head_dim: int,
    v_head_dim: int,
    block_mask_batch: int,
    block_mask_heads: int,
    num_q_blocks: int,
    max_partial_blocks: int,
    max_full_blocks: int,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    scale: float,
    mask_program=(),
    mask_program_output: int = 0,
    mask_buffer_shapes=(),
    mask_buffer_strides=(),
    q_stride=None,
    k_stride=None,
    v_stride=None,
    o_stride=None,
    output_stats_in_log2: bool = False,
    write_max_scores: bool = True,
    dtype: str = "bf16",
    score_mod=None,
    score_mod_aux=(),
    score_mod_buffer_count: int = 0,
    mod_key=None,
    mask_buffer_dtypes=(),
    score_mod_buffer_dtypes=(),
    kv_head_dims=None,
    qo_head_dims=None,
    tie_streamed_probabilities: bool = False,
    plan=None,
    head_slow_order: bool = False,
    kv_splits: int = 1,
    single_pass: bool = False,
    target_limits=None,
):
    """Build the gfx950 prefill or packed-GQA decode kernel.

    Each wave owns 32 query rows; K/V are staged once per CTA for all owners.
    ``score_mod(score, b, h, q_idx, kv_idx, aux)`` sees ``q.k * scale``; its
    captures follow the mask buffers, or above ``MAX_DIRECT_CAPTURES`` go
    through an int64 table of base addresses. ``kv_head_dims`` are narrower
    stored K/V widths (multiples of 8) whose missing columns read as zeros.
    ``qo_head_dims`` are likewise narrower stored Q/O widths: missing Q columns
    read as zeros and O columns past the stored width (a multiple of 16) are
    not written.
    ``tie_streamed_probabilities`` ties the wide-head wait states between
    masked-tile probabilities and their PV MFMAs at every head dim.
    ``head_slow_order`` launches the grid in ``tile_axes`` head-slow order.
    ``kv_splits`` decode CTAs share each row's full and partial block lists;
    split ``s`` writes its fp32 output, normalized by its own sum, to batch
    ``s * B + b`` of a [kv_splits * B, Hq, Sq, V] output, and its statistics
    to ``s * B + b`` of [kv_splits * B, Hq, Sq] ones.
    ``single_pass`` builds a head-dim pair of ``WIDE_HEAD_DIMS`` as one
    launch: K and V tiles pass through LDS in column stages of at most 256,
    and ``single_pass_value_shards`` waves split each query row's output.
    ``target_limits`` are the ``TargetLimits`` to plan for, by default the
    current device's.
    """

    target = resolve_target(target_limits)
    # Keep standalone entry-point validation even though Inductor checks these
    # constraints before registering the vendored kernel.
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError("FlyDSL forward requires Hq % Hkv == 0")
    has_score_mod = score_mod is not None
    score_mod_aux, score_mod_buffer_count = validate_score_mod_args(
        score_mod, score_mod_aux, score_mod_buffer_count, mod_key
    )

    key_columns, value_columns = (
        (int(qk_head_dim), int(v_head_dim))
        if kv_head_dims is None
        else (int(kv_head_dims[0]), int(kv_head_dims[1]))
    )
    if not (
        0 < key_columns <= qk_head_dim
        and 0 < value_columns <= v_head_dim
        and key_columns % 8 == value_columns % 8 == 0
    ):
        raise ValueError(
            "FlyDSL forward kv_head_dims must be multiples of 8 up to the head dims"
        )
    query_columns, output_columns = (
        (int(qk_head_dim), int(v_head_dim))
        if qo_head_dims is None
        else (int(qo_head_dims[0]), int(qo_head_dims[1]))
    )
    if not (
        0 < query_columns <= qk_head_dim
        and 0 < output_columns <= v_head_dim
        and query_columns % 8 == output_columns % 16 == 0
    ):
        raise ValueError(
            "FlyDSL forward qo_head_dims must be multiples of 8 (Q) and 16 (O) "
            "up to the head dims"
        )
    padded_qo = (query_columns, output_columns) != (qk_head_dim, v_head_dim)
    if padded_qo and (single_pass or (kv_splits > 1 and output_columns < v_head_dim)):
        raise ValueError(
            "FlyDSL forward qo_head_dims needs single_pass off, and kv_splits 1 "
            "for a narrower O"
        )
    padded_kv = (key_columns, value_columns) != (qk_head_dim, v_head_dim)

    if dtype not in ELEMENT_TYPES:
        raise ValueError(f"FlyDSL forward supports bf16/fp16, got {dtype}")
    element_type = ELEMENT_TYPES[dtype]
    single_pass = bool(single_pass)
    if single_pass:
        if (qk_head_dim, v_head_dim) not in WIDE_HEAD_DIMS:
            raise ValueError(
                "FlyDSL single-pass forward requires (qk_head_dim, v_head_dim) "
                f"in {WIDE_HEAD_DIMS}"
            )
    elif (qk_head_dim, v_head_dim) not in forward_head_dims(target):
        raise ValueError(
            "FlyDSL forward requires (qk_head_dim, v_head_dim) in "
            f"{forward_head_dims(target)}"
        )
    if sparse_q_block_size != 128 or sparse_kv_block_size != 128:
        raise ValueError("FlyDSL forward requires sparse block size 128")
    # None sizes and strides are read from the launch's layout-dynamic tensors.
    sizes = (batch_size, seq_q, seq_kv)
    if any(size is not None and size <= 0 for size in sizes):
        raise ValueError("FlyDSL forward requires positive B, Sq and Sk")
    if None in sizes and plan is None:
        raise ValueError("FlyDSL forward needs a forward_plan for runtime sizes")
    if block_mask_batch not in (1, batch_size, None):
        raise ValueError("BlockMask batch dimension must be 1 or B")
    if block_mask_heads not in (1, num_kv_heads, num_q_heads):
        raise ValueError("BlockMask head dimension must be 1, Hkv, or Hq")
    if any(
        blocks is not None and blocks <= 0
        for blocks in (max_partial_blocks, max_full_blocks)
    ):
        raise ValueError("FlyDSL forward requires non-empty index storage")
    if len(mask_buffer_shapes) != len(mask_buffer_strides):
        raise ValueError("mask buffer shape/stride descriptors must match")

    batch_size, seq_q, seq_kv = map(optional_int, sizes)
    num_q_heads = int(num_q_heads)
    num_kv_heads = int(num_kv_heads)
    qk_head_dim = int(qk_head_dim)
    v_head_dim = int(v_head_dim)
    block_mask_batch = optional_int(block_mask_batch)
    block_mask_heads = int(block_mask_heads)
    num_q_blocks = optional_int(num_q_blocks)
    max_partial_blocks = optional_int(max_partial_blocks)
    max_full_blocks = optional_int(max_full_blocks)
    if plan is None:
        plan = forward_plan(
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            has_score_mod=has_score_mod,
            padded_kv=padded_kv or padded_qo,
            has_mask_buffers=bool(mask_buffer_shapes),
            **({"single_pass": True} if single_pass else {}),
            target=target,
        )
    paired = plan["paired"]
    decode = plan["decode"]
    owner_waves = plan["owner_waves"]
    split_kv = plan["split_kv"]
    waves_per_eu = plan["waves_per_eu"]
    pipelined_kv = plan["pipelined_kv"]
    q_ragged = plan["q_ragged"]
    kv_ragged = plan["kv_ragged"]
    heads_per_group = plan["heads_per_group"]
    paired_kv_blocks = plan.get("paired_kv_blocks")
    if paired and None in sizes and paired_kv_blocks is None:
        raise ValueError("the paired FlyDSL forward needs a KV block bound")
    kv_splits = int(kv_splits)
    if kv_splits < 1 or (kv_splits > 1 and (paired or not decode)):
        raise ValueError("FlyDSL forward splits the KV blocks of decode CTAs only")
    wide_head = _is_wide_head(qk_head_dim, v_head_dim)
    if single_pass and not (
        pipelined_kv and owner_waves == 4 and not (paired or split_kv)
    ):
        raise ValueError("the FlyDSL single-pass forward needs a 4-wave pipeline")
    value_shards = (
        single_pass_value_shards(
            qk_head_dim,
            v_head_dim,
            not (
                has_score_mod
                or mask_buffer_shapes
                or decode
                or q_ragged
                or kv_ragged
                or None in sizes
            ),
        )
        if single_pass
        else 1
    )
    if kv_splits > 1 and value_shards > 1:
        raise ValueError("FlyDSL split-KV decode cannot shard V across waves")
    query_tile_rows = owner_waves * 32 // value_shards
    kv_tile_rows = 64
    num_waves = 2 if split_kv else owner_waves
    num_threads = num_waves * WAVE_SIZE
    values_per_thread = 8
    mfma_tile_size = 32
    query_heads_per_kv_head = num_q_heads // num_kv_heads

    def packed_rows(seq_q):
        return query_heads_per_kv_head * seq_q if decode else seq_q

    launch_heads = num_kv_heads if decode else num_q_heads

    def launch_grid(batch_size, seq_q):
        num_query_chunks = ceil_div(packed_rows(seq_q), query_tile_rows)
        if kv_splits > 1:
            num_query_chunks = num_query_chunks * kv_splits
        if paired:
            return tile_grid(num_q_heads, num_query_chunks, batch_size, head_slow_order)
        if isinstance(batch_size, int):
            return tile_grid(
                heads_per_group,
                num_query_chunks,
                batch_size * launch_heads // heads_per_group,
                head_slow_order,
            )
        return tile_grid(
            heads_per_group,
            num_query_chunks,
            batch_size * (launch_heads // heads_per_group),
            head_slow_order,
        )

    packed_query_rows = None if seq_q is None else packed_rows(seq_q)

    if decode:
        if block_mask_heads not in (1, num_kv_heads):
            raise ValueError(
                "FlyDSL forward decode requires a shared or per-KV-head BlockMask"
            )
        if num_q_blocks not in (1, None):
            raise ValueError("FlyDSL forward decode requires one sparse Q block")
        if packed_query_rows is not None and not 0 < packed_query_rows <= 256:
            raise ValueError("FlyDSL forward decode requires 1 <= (Hq/Hkv)*Sq <= 256")
    elif None not in (num_q_blocks, seq_q) and num_q_blocks != ceil_div(
        seq_q, sparse_q_block_size
    ):
        raise ValueError("BlockMask Q rows must cover Sq with 128-row blocks")

    kv_tiles_per_sparse_block = sparse_kv_block_size // kv_tile_rows
    qk_reduction_steps = qk_head_dim // 16
    first_softmax_count = min(24, 2 * qk_reduction_steps)
    output_chunks = v_head_dim // mfma_tile_size // value_shards
    # Single-pass stages: (count, columns) of K and V, two LDS buffers each.
    key_stages = ceil_div(qk_head_dim, 256) if single_pass else 1
    value_stages = ceil_div(v_head_dim, 256) if single_pass else 1
    key_stage_columns = qk_head_dim // key_stages
    value_stage_columns = v_head_dim // value_stages
    key_stage_steps = key_stage_columns // 16
    shard_stage_chunks = value_stage_columns // mfma_tile_size // value_shards
    qk_chunks_per_row = qk_head_dim // values_per_thread
    value_chunks_per_row = v_head_dim // values_per_thread
    # Unpipelined K and V tiles take turns in one buffer.
    shared_kv_tile = kv_tile_rows * max(qk_head_dim, v_head_dim)
    query_load_iterations = (query_tile_rows * qk_chunks_per_row) // num_threads
    kv_load_threads = WAVE_SIZE if split_kv else num_threads
    key_load_iterations = (kv_tile_rows * qk_chunks_per_row) // kv_load_threads
    value_load_iterations = (kv_tile_rows * value_chunks_per_row) // kv_load_threads
    if (query_tile_rows * qk_chunks_per_row) % num_threads:
        raise ValueError("FlyDSL forward Q staging must evenly cover its tile")
    if (kv_tile_rows * qk_chunks_per_row) % kv_load_threads:
        raise ValueError("FlyDSL forward K staging must evenly cover its tile")
    if (kv_tile_rows * value_chunks_per_row) % kv_load_threads:
        raise ValueError("FlyDSL forward V staging must evenly cover its tile")

    def contiguous_stride(heads, sequence, dimension):
        return (heads * sequence * dimension, sequence * dimension, dimension, 1)

    q_stride = tuple(q_stride or contiguous_stride(num_q_heads, seq_q, qk_head_dim))
    k_stride = tuple(k_stride or contiguous_stride(num_kv_heads, seq_kv, qk_head_dim))
    v_stride = tuple(v_stride or contiguous_stride(num_kv_heads, seq_kv, v_head_dim))
    o_stride = tuple(o_stride or contiguous_stride(num_q_heads, seq_q, v_head_dim))
    scale_log2 = float(scale) * _LOG2E
    output_stats_in_log2 = bool(output_stats_in_log2)
    head_slow_order = bool(head_slow_order)
    head_axis, block_axis = tile_axes(head_slow_order)
    mask_program = tuple(mask_program)
    mask_program_output = int(mask_program_output)
    mask_buffer_shapes = tuple(tuple(shape) for shape in mask_buffer_shapes)
    mask_buffer_strides = tuple(tuple(stride) for stride in mask_buffer_strides)
    mask_buffer_count = len(mask_buffer_shapes)
    mask_buffer_sizes = mask_capture_sizes(mask_buffer_shapes, mask_buffer_strides)
    capture_table = capture_table_layout(
        mask_buffer_sizes,
        tuple(mask_buffer_dtypes),
        score_mod_aux,
        score_mod_buffer_count,
        tuple(score_mod_buffer_dtypes),
    )
    capture_slots = capture_extent_count(mask_buffer_sizes, capture_table)
    if single_pass and capture_table is not None:
        raise ValueError("the FlyDSL single-pass forward reads no capture table")
    if capture_table is not None and not paired:
        # With two waves per SIMD, the capture reads of a table build spill
        # inside the KV loops.
        waves_per_eu = 1
    mask_load_width = 4
    vector_mask_loads = []
    for slot, instruction in enumerate(() if paired else mask_program, 4):
        if instruction[0] == "load_i32":
            buffer_index, indices = instruction[1:]
            load_dtype = "int32"
        elif (
            capture_table is not None
            and instruction[0] == "load"
            and instruction[3] in _VECTOR_MASK_LOADS
        ):
            # A table read holds its own descriptor until it issues.
            buffer_index, indices, load_dtype = instruction[1:]
        else:
            continue
        if indices.count(3) != 1 or any(index not in (0, 1, 2, 3) for index in indices):
            continue
        if mask_buffer_sizes[buffer_index] is None:
            continue
        dimension = indices.index(3)
        strides = mask_buffer_strides[buffer_index]
        if (
            strides[dimension] == 1
            and mask_buffer_shapes[buffer_index][dimension] % mask_load_width == 0
            and all(
                stride % mask_load_width == 0
                for i, stride in enumerate(strides)
                if i != dimension
            )
        ):
            vector_mask_loads.append((slot, buffer_index, indices, load_dtype))
    # With no vector mask load, LLVM can schedule a streamed P pack's convert
    # one wait state before the PV MFMA that reads it.
    tie_scalar_mask_loads = not vector_mask_loads

    mask_interval_types = ["int"] * 4
    mask_key_dependent = [False, False, False, True]
    for instruction in mask_program:
        op = instruction[0]
        value_type = None
        dependencies = mask_instruction_operands(instruction)
        if op in ("const_i32", "const_bool"):
            value_type = "int" if op == "const_i32" else "bool"
        elif op == "load_i32":
            if all(
                mask_interval_types[index] == "int" and not mask_key_dependent[index]
                for index in dependencies
            ):
                value_type = "int"
        else:
            if op in ("ge", "gt", "le", "lt", "eq", "ne"):
                if all(mask_interval_types[index] == "int" for index in dependencies):
                    value_type = "bool"
            elif op in ("and", "or", "not"):
                if all(mask_interval_types[index] == "bool" for index in dependencies):
                    value_type = "bool"
        mask_interval_types.append(value_type)
        mask_key_dependent.append(
            any(mask_key_dependent[index] for index in dependencies)
        )
    mask_interval_supported = (
        bool(mask_program)
        and None not in mask_interval_types
        and mask_interval_types[mask_program_output] == "bool"
    )
    flat_work_mask = is_sliding_window_mask_program(
        mask_program,
        mask_program_output,
        mask_buffer_strides,
    )

    key_stage_loads = (
        kv_tile_rows * key_stage_columns // values_per_thread // num_threads
    )
    value_stage_loads = (
        kv_tile_rows * value_stage_columns // values_per_thread // num_threads
    )
    if single_pass:

        @fx.struct
        class ForwardSharedMemory:
            k0: fx.Array[element_type, kv_tile_rows * key_stage_columns, 16]
            k1: fx.Array[element_type, kv_tile_rows * key_stage_columns, 16]
            v0: fx.Array[element_type, kv_tile_rows * value_stage_columns, 16]
            v1: fx.Array[element_type, kv_tile_rows * value_stage_columns, 16]

    elif pipelined_kv:

        @fx.struct
        class ForwardSharedMemory:
            # Keep Q in registers and double-buffer K/V
            # so the next tile's DMA can overlap the current tile's math.
            k0: fx.Array[element_type, kv_tile_rows * qk_head_dim, 16]
            k1: fx.Array[element_type, kv_tile_rows * qk_head_dim, 16]
            v0: fx.Array[element_type, kv_tile_rows * v_head_dim, 16]
            v1: fx.Array[element_type, kv_tile_rows * v_head_dim, 16]

    elif split_kv:

        @fx.struct
        class ForwardSharedMemory:
            query: fx.Array[element_type, query_tile_rows * qk_head_dim, 16]
            # One reusable K/V tile per worker wave keeps the CTA below 64 KiB.
            kv: fx.Array[element_type, num_waves * shared_kv_tile, 16]
            reduction_stats: fx.Array[fx.Float32, 2 * 2, 16]
            reduction_output: fx.Array[fx.Float32, 2 * output_chunks * 16, 16]

    else:

        @fx.struct
        class ForwardSharedMemory:
            query: fx.Array[element_type, query_tile_rows * qk_head_dim, 16]
            # V reuses the K storage after every wave has consumed K into
            # registers.
            kv: fx.Array[element_type, shared_kv_tile, 16]

    single_batch = batch_size == 1

    def resolve_shape(query, key, value, output, metadata, mask_buffers):
        kv_num_blocks, kv_indices, full_kv_indices = metadata
        sizes = (
            runtime_extent(batch_size, query, 0),
            runtime_extent(seq_q, query, 2),
            runtime_extent(seq_kv, key, 2),
            runtime_extent(num_q_blocks, kv_num_blocks, 2),
            runtime_extent(max_partial_blocks, kv_indices, 3),
            runtime_extent(max_full_blocks, full_kv_indices, 3),
        )
        rows = packed_rows(sizes[1])
        mask_rows = (
            runtime_extent(block_mask_batch, kv_num_blocks, 0)
            * block_mask_heads
            * sizes[3]
        )
        return (
            *sizes,
            *(
                runtime_strides(strides, tensor, sizes[length])
                for strides, tensor, length in (
                    (q_stride, query, 1),
                    (k_stride, key, 2),
                    (v_stride, value, 2),
                    (o_stride, output, 1),
                )
            ),
            rows,
            ceil_div(rows, query_tile_rows),
            mask_rows,
            *resolve_mask_buffers(mask_buffers),
        )

    def resolve_mask_buffers(mask_buffers):
        return resolve_capture_extents(
            mask_buffer_strides, mask_buffer_sizes, mask_buffers, capture_table
        )

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        logsumexp: fx.Tensor,
        max_scores: fx.Tensor,
        kv_num_blocks: fx.Tensor,
        kv_indices: fx.Tensor,
        full_kv_num_blocks: fx.Tensor,
        full_kv_indices: fx.Tensor,
        mask_buffer_0: fx.Tensor,
        mask_buffer_1: fx.Tensor,
        mask_buffer_2: fx.Tensor,
        mask_buffer_3: fx.Tensor,
        output: fx.Tensor,
    ):
        (
            batch_size,
            seq_q,
            seq_kv,
            num_q_blocks,
            max_partial_blocks,
            max_full_blocks,
            q_stride,
            k_stride,
            v_stride,
            o_stride,
            packed_query_rows,
            num_query_chunks,
            mask_rows,
            mask_buffer_strides,
            mask_buffer_sizes,
        ) = resolve_shape(
            query,
            key,
            value,
            output,
            (kv_num_blocks, kv_indices, full_kv_indices),
            (mask_buffer_0, mask_buffer_1, mask_buffer_2, mask_buffer_3),
        )
        tid = fx.thread_idx.x
        warp_size = WAVE_SIZE
        wave_index = tid // fx.Int32(warp_size)
        wave = fx.Int32(fx.rocdl.readfirstlane(fx.Int32.ir_type, wave_index.ir_value()))
        if const_expr(paired):
            llvm.intr_assume(
                ((wave >= fx.Int32(0)) & (wave < fx.Int32(num_waves))).ir_value(),
                [],
                [],
            )
        lds = fx.SharedAllocator().allocate(ForwardSharedMemory).peek()

        def run_body(stagger):
            if const_expr(paired):
                lane_value = fx.thread_idx.x % fx.Int32(warp_size)
                lane_raw = llvm.inline_asm(
                    lane_value.ir_value().type,
                    [lane_value.ir_value()],
                    "",
                    "=v,0",
                    has_side_effects=True,
                )
                lane = fx.Int32(lane_raw)
                llvm.intr_assume(
                    ((lane >= fx.Int32(0)) & (lane < fx.Int32(warp_size))).ir_value(),
                    [],
                    [],
                )
                tid = wave * fx.Int32(warp_size) + lane
            else:
                tid = fx.thread_idx.x
                lane = tid % fx.Int32(warp_size)

            lane_half = lane // fx.Int32(mfma_tile_size)
            mma_atom = fx.make_mma_atom(
                fx.rocdl.MFMA(
                    mfma_tile_size,
                    mfma_tile_size,
                    16,
                    element_type,
                )
            )
            tiled_mma = fx.make_tiled_mma(
                mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1))
            )
            thr_mma = tiled_mma.get_slice(lane)
            accumulator_coordinates = thr_mma.partition_C(
                fx.make_view(
                    0,
                    fx.make_layout((mfma_tile_size, mfma_tile_size), (1, 0)),
                )
            )
            query_coordinates = thr_mma.partition_C(
                fx.make_view(
                    0,
                    fx.make_layout((mfma_tile_size, mfma_tile_size), (0, 1)),
                )
            )
            query_k_coordinates = thr_mma.partition_B(
                fx.make_view(
                    0,
                    fx.make_layout((mfma_tile_size, qk_head_dim), (0, 1)),
                )
            )
            if const_expr(paired):
                batch = fx.block_idx.z
            else:
                grouped_head = fx.block_idx.z * fx.Int32(heads_per_group) + getattr(
                    fx.block_idx, head_axis
                )
                batch = (
                    fx.Int32(0)
                    if const_expr(single_batch)
                    else grouped_head // fx.Int32(launch_heads)
                )
            if const_expr(kv_splits > 1):
                kv_split = getattr(fx.block_idx, block_axis) % fx.Int32(kv_splits)
                q_chunk = getattr(fx.block_idx, block_axis) // fx.Int32(kv_splits)
            elif const_expr(decode or flat_work_mask):
                q_chunk = getattr(fx.block_idx, block_axis)
            else:
                # Dispatch longer causal query blocks first to avoid a ragged tail.
                q_chunk = fx.Int32(num_query_chunks - 1) - getattr(
                    fx.block_idx, block_axis
                )
            q_base = q_chunk * fx.Int32(query_tile_rows)
            if const_expr(paired):
                head = getattr(fx.block_idx, head_axis)
                if const_expr(query_heads_per_kv_head > 1):
                    head = (head % fx.Int32(num_kv_heads)) * fx.Int32(
                        query_heads_per_kv_head
                    ) + head // fx.Int32(num_kv_heads)
                kv_head = head // fx.Int32(query_heads_per_kv_head)
            elif const_expr(decode):
                kv_head = grouped_head % fx.Int32(num_kv_heads)
                head = kv_head * fx.Int32(query_heads_per_kv_head)
            else:
                head = grouped_head % fx.Int32(num_q_heads)
                kv_head = head // fx.Int32(query_heads_per_kv_head)

            def row_coordinates(local_row):
                packed_row = q_base + local_row
                if const_expr(decode):
                    valid = packed_row < fx.Int32(packed_query_rows)
                    safe_row = valid.select(packed_row, fx.Int32(0))
                    row_head = kv_head * fx.Int32(
                        query_heads_per_kv_head
                    ) + safe_row // fx.Int32(seq_q)
                    query_position = safe_row % fx.Int32(seq_q)
                else:
                    valid = fx.Int32(0) == fx.Int32(0)
                    row_head = head
                    query_position = packed_row
                return valid, row_head, query_position

            query_row_in_wave = fx.Int32(fx.get_scalar(query_coordinates[0]))
            owner_wave = wave
            if const_expr(value_shards > 1):
                owner_wave = wave // fx.Int32(value_shards)
                value_shard = wave % fx.Int32(value_shards)
            query_row = (
                query_row_in_wave
                if const_expr(split_kv)
                else owner_wave * fx.Int32(mfma_tile_size) + query_row_in_wave
            )
            query_valid, query_head, query_pos = row_coordinates(query_row)

            if const_expr(single_pass):
                shared_query_pointer = None
                shared_key_stages = [lds.k0.ptr, lds.k1.ptr]
                shared_value_stages = [lds.v0.ptr, lds.v1.ptr]
            elif const_expr(pipelined_kv):
                # Q aliases K0/K1 until every wave has cached its fragments.
                shared_query_pointer = lds.k0.ptr
                shared_key_stages = [lds.k0.ptr, lds.k1.ptr]
                shared_value_stages = [lds.v0.ptr, lds.v1.ptr]
            else:
                shared_query_pointer = lds.query.ptr
                shared_kv_pointer = lds.kv.ptr
                if const_expr(split_kv):
                    shared_kv_pointer = fx.get_iter(
                        fx.slice(
                            fx.make_view(
                                shared_kv_pointer,
                                fx.make_layout(
                                    (num_waves, shared_kv_tile), (shared_kv_tile, 1)
                                ),
                            ),
                            (wave, None),
                        )
                    )
                shared_key_stages = [shared_kv_pointer]
                shared_value_stages = [shared_kv_pointer]

            batch_i64 = fx.Int64(batch)
            kv_head_i64 = fx.Int64(kv_head)
            key_view = make_global_view(
                key,
                (batch_i64, kv_head_i64, None, None),
                (batch_size, num_kv_heads, seq_kv, key_columns),
                k_stride,
            )
            value_view = make_global_view(
                value,
                (batch_i64, kv_head_i64, None, None),
                (batch_size, num_kv_heads, seq_kv, value_columns),
                v_stride,
            )
            output_batch, output_batches = batch, batch_size
            output_batch_i64 = batch_i64
            if const_expr(kv_splits > 1):
                output_batch = kv_split * fx.Int32(batch_size) + batch
                output_batch_i64 = fx.Int64(output_batch)
                output_batches = batch_size * kv_splits
            if const_expr(decode):
                query_view = make_global_view(
                    query,
                    (batch_i64, None, kv_head_i64, None, None),
                    (
                        batch_size,
                        query_heads_per_kv_head,
                        num_kv_heads,
                        seq_q,
                        query_columns,
                    ),
                    (
                        q_stride[0],
                        q_stride[1],
                        query_heads_per_kv_head * q_stride[1],
                        q_stride[2],
                        q_stride[3],
                    ),
                )
                output_view = make_global_view(
                    output,
                    (output_batch_i64, None, kv_head_i64, None, None),
                    (
                        output_batches,
                        query_heads_per_kv_head,
                        num_kv_heads,
                        seq_q,
                        output_columns,
                    ),
                    (
                        o_stride[0],
                        o_stride[1],
                        query_heads_per_kv_head * o_stride[1],
                        o_stride[2],
                        o_stride[3],
                    ),
                )
            else:
                head_i64 = fx.Int64(head)
                query_view = make_global_view(
                    query,
                    (batch_i64, head_i64, None, None),
                    (batch_size, num_q_heads, seq_q, query_columns),
                    q_stride,
                )
                output_view = make_global_view(
                    output,
                    (batch_i64, head_i64, None, None),
                    (batch_size, num_q_heads, seq_q, output_columns),
                    o_stride,
                )

            metadata_rows = mask_rows

            def make_metadata_view(tensor, size):
                if const_expr(paired):
                    return fx.make_view(fx.get_iter(tensor), fx.make_layout(size, 1))
                return make_global_view(tensor, None, size, 1)

            kv_num_blocks_view = make_metadata_view(kv_num_blocks, metadata_rows)
            kv_indices_view = make_metadata_view(
                kv_indices, metadata_rows * max_partial_blocks
            )
            full_kv_num_blocks_view = make_metadata_view(
                full_kv_num_blocks, metadata_rows
            )
            full_kv_indices_view = make_metadata_view(
                full_kv_indices, metadata_rows * max_full_blocks
            )
            logsumexp_view = make_global_view(
                logsumexp,
                None,
                output_batches * num_q_heads * seq_q,
                1,
            )
            max_scores_view = make_global_view(
                max_scores,
                None,
                output_batches * num_q_heads * seq_q,
                1,
            )
            mask_buffers, score_mod_buffers = make_capture_buffers(
                make_global_view,
                mask_buffer_count,
                mask_buffer_sizes,
                capture_table,
                (mask_buffer_0, mask_buffer_1, mask_buffer_2, mask_buffer_3),
            )
            if const_expr(has_score_mod):
                score_mod_readers = make_score_mod_readers(
                    make_global_view, score_mod_aux, score_mod_buffers
                )

            output_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), element_type)

            def load_i32(view, index):
                return fx.Int32(view[index])

            def load_uniform_i32(view, index):
                if const_expr(paired):
                    loaded = load_i32(view, index)
                    return fx.Int32(
                        fx.rocdl.readfirstlane(fx.Int32.ir_type, loaded.ir_value())
                    )
                return fx.gpu.shuffle_idx(load_i32(view, index), 0, warp_size)

            evaluate_mask = make_mask_evaluator(
                mask_program,
                mask_program_output,
                mask_buffer_strides,
                mask_buffers,
                load_i32,
                batch,
                query_head,
            )
            # Table reads are opaque to LLVM, which would otherwise hoist the
            # reads that do not depend on the key out of the KV loops. The
            # paired schedule has no registers to keep them in.
            query_mask_values = {}
            if const_expr(
                capture_table is not None and bool(mask_program) and not paired
            ):
                query_mask_values = evaluate_mask(query_pos, None, independent_of=3)

            mask_vector_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)

            def load_mask_group(first_key):
                inputs = [fx.Int32(batch), query_head, query_pos, first_key]
                groups = {}
                for slot, buffer_index, indices, load_dtype in vector_mask_loads:
                    offset = fx.Int32(0)
                    for dimension, index in enumerate(indices):
                        offset = offset + inputs[index] * fx.Int32(
                            mask_buffer_strides[buffer_index][dimension]
                        )
                    element, copy = (fx.Int32, mask_vector_copy)
                    if const_expr(load_dtype != "int32"):
                        element, bits = _VECTOR_MASK_LOADS[load_dtype]
                        copy = fx.make_copy_atom(
                            getattr(fx.rocdl, f"BufferCopy{bits}b")(), element
                        )
                    width = min(mask_load_width, 128 // element.width)
                    source = fx.logical_divide(
                        mask_buffers[buffer_index], fx.make_layout(width, 1)
                    )
                    values = []
                    for part in fx.range_constexpr(mask_load_width // width):
                        index = offset // fx.Int32(width)
                        if const_expr(part):
                            index = index + fx.Int32(part)
                        fragment = fx.make_rmem_tensor(width, element)
                        fx.copy(copy, fx.slice(source, (None, index)), fragment)
                        values.append(Vec(fragment.load()))
                    groups[slot] = (values, width, load_dtype)
                return groups

            def mask_group_value(group, position):
                values, width, load_dtype = group
                value = values[position // width][position % width]
                if const_expr(load_dtype == "int32"):
                    return fx.Int32(value)
                return mask_element_value(value, load_dtype)

            global_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), element_type)
            lds_copy = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), element_type)
            transposed_lds_copy = fx.make_copy_atom(
                fx.rocdl.cdna4.LDSReadTrans(16, 64),
                element_type,
            )
            key_shared_layout = make_qk_shared_layout(kv_tile_rows, key_stage_columns)
            value_shared_layout = make_value_shared_layout(
                kv_tile_rows, value_stage_columns
            )
            shared_keys = [
                fx.make_view(pointer, key_shared_layout)
                for pointer in shared_key_stages
            ]
            key_copy_coordinates = fx.make_composed_layout(
                fx.right_inverse(key_shared_layout.outer),
                fx.make_composed_layout(
                    key_shared_layout.inner,
                    fx.make_layout(kv_tile_rows * key_stage_columns, 1),
                ),
            )
            value_copy_coordinates = fx.right_inverse(value_shared_layout)
            key_copy_destinations = [
                fx.logical_divide(
                    fx.make_view(
                        pointer,
                        fx.make_layout(kv_tile_rows * key_stage_columns, 1),
                    ),
                    fx.make_layout(values_per_thread, 1),
                )
                for pointer in shared_key_stages
            ]
            value_copy_destinations = [
                fx.logical_divide(
                    fx.make_view(
                        pointer,
                        fx.make_layout(kv_tile_rows * value_stage_columns, 1),
                    ),
                    fx.make_layout(values_per_thread, 1),
                )
                for pointer in shared_value_stages
            ]
            if const_expr(paired):
                shared_query = query_view
            elif const_expr((not pipelined_kv or not decode) and not single_pass):
                shared_query = fx.make_view(
                    shared_query_pointer,
                    make_qk_shared_layout(query_tile_rows, qk_head_dim),
                )
            else:
                # Q stays in registers; this view is never read.
                shared_query = fx.make_view(
                    fx.get_iter(query_view),
                    fx.make_layout(
                        (packed_query_rows, qk_head_dim),
                        (qk_head_dim, 1),
                    ),
                )
            q_wave = fx.Int32(0) if const_expr(split_kv) else wave
            query_tiles = fx.flat_divide(shared_query, (mfma_tile_size, 16))
            key_tiles = [
                fx.flat_divide(shared_key, (mfma_tile_size, 16))
                for shared_key in shared_keys
            ]
            copy_q = fx.make_tiled_copy_B(global_copy, tiled_mma).get_slice(lane)
            copy_k = fx.make_tiled_copy_A(global_copy, tiled_mma).get_slice(lane)
            copy_v = fx.make_tiled_copy_A(transposed_lds_copy, tiled_mma).get_slice(
                lane
            )
            shared_copy = fx.make_copy_atom(fx.UniversalCopy128b(), element_type)
            probability_coordinates = thr_mma.partition_B(
                fx.make_view(
                    0,
                    fx.make_layout((mfma_tile_size, 16), (1, mfma_tile_size)),
                )
            )
            # Softmax keeps each lane's C values in register order. Permute V's
            # reduction mode to match that order without a cross-lane P shuffle.
            value_mma_layout = fx.composition(
                fx.select(value_shared_layout, [1, 0]),
                fx.make_tile(
                    fx.make_layout(value_stage_columns, 1),
                    fx.make_layout(
                        (4, 2, 2, kv_tile_rows // 16),
                        (1, 8, 4, 16),
                    ),
                ),
            )
            value_tiles = [
                fx.flat_divide(
                    fx.make_view(pointer, value_mma_layout),
                    (mfma_tile_size, 16),
                )
                for pointer in shared_value_stages
            ]
            shard_value_tiles = value_tiles
            if const_expr(value_shards > 1):
                # A shard's first chunk of each stage, 8 x 32 elements per chunk.
                shard_value_tiles = [
                    fx.flat_divide(
                        fx.make_view(
                            pointer
                            + value_shard
                            * fx.Int32(shard_stage_chunks * 8 * mfma_tile_size),
                            value_mma_layout,
                        ),
                        (mfma_tile_size, 16),
                    )
                    for pointer in shared_value_stages
                ]

            def mfma(a_fragment, b_fragment, c_v16):
                c_fragment = fx.make_fragment_like(accumulator_coordinates, fx.Float32)
                c_fragment.store(Vec(c_v16))
                fx.gemm(
                    tiled_mma,
                    c_fragment,
                    a_fragment,
                    b_fragment,
                    c_fragment,
                )
                return c_fragment.load()

            def get_mask_row():
                if const_expr(block_mask_batch == 1):
                    mask_batch = fx.Int32(0)
                else:
                    mask_batch = batch
                if const_expr(block_mask_heads == 1):
                    mask_head = fx.Int32(0)
                elif const_expr(block_mask_heads == num_kv_heads):
                    mask_head = kv_head
                else:
                    mask_head = head
                if const_expr(decode):
                    mask_q_block = fx.Int32(0)
                else:
                    mask_q_block = q_base // fx.Int32(sparse_q_block_size)
                mask_row = (
                    mask_batch * fx.Int32(block_mask_heads) + mask_head
                ) * fx.Int32(num_q_blocks) + mask_q_block
                return mask_row

            def pair_word(view, row, count, capacity, word):
                bits = fx.Uint32(0)
                slots = capacity if isinstance(capacity, int) else paired_kv_blocks
                for group in fx.range_constexpr((slots + warp_size - 1) // warp_size):
                    slot = fx.Int32(group * warp_size) + lane
                    safe_slot = (slot < fx.Int32(capacity)).select(slot, fx.Int32(0))
                    block = load_i32(view, row * fx.Int32(capacity) + safe_slot)
                    valid = (slot < count) & (block // fx.Int32(32) == fx.Int32(word))
                    bit = fx.Uint32(1) << (fx.Uint32(block) & fx.Uint32(31))
                    bits = bits | valid.select(bit, fx.Uint32(0))
                for step in fx.range_constexpr(6):
                    bits = bits | bits.shuffle_xor(
                        fx.Int32(1 << step), fx.Int32(warp_size)
                    )
                return fx.Uint32(
                    fx.rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(bits).ir_value())
                )

            def pair_cache(words):
                cache = []
                for group in fx.range_constexpr(
                    (paired_kv_blocks + warp_size - 1) // warp_size
                ):
                    index = fx.Int32(group * warp_size) + lane
                    prefix = fx.Int32(0)
                    selected = fx.Int32(0)
                    for word in fx.range_constexpr(len(words)):
                        bits = words[word]
                        count = fx.Int32(fx.math.ctpop(bits))
                        valid = (index >= prefix) & (index < prefix + count)
                        rank = valid.select(index - prefix, fx.Int32(0))
                        position = fx.Int32(word * 32)
                        for level in fx.range_constexpr(5):
                            width = 16 >> level
                            low_count = fx.Int32(
                                fx.math.ctpop(bits & fx.Uint32((1 << width) - 1))
                            )
                            upper = rank >= low_count
                            rank = upper.select(rank - low_count, rank)
                            bits = upper.select(bits >> fx.Uint32(width), bits)
                            position = position + upper.select(
                                fx.Int32(width), fx.Int32(0)
                            )
                        selected = valid.select(position, selected)
                        prefix = prefix + count
                    cache.append(selected)
                return cache

            def read_pair_cache(cache, index):
                source_lane = index % fx.Int32(warp_size)
                selected = fx.Int32(
                    fx.rocdl.readlane(
                        fx.Int32.ir_type, cache[0].ir_value(), source_lane.ir_value()
                    )
                )
                for group in fx.range_constexpr(
                    1, (paired_kv_blocks + warp_size - 1) // warp_size
                ):
                    value = fx.Int32(
                        fx.rocdl.readlane(
                            fx.Int32.ir_type,
                            cache[group].ir_value(),
                            source_lane.ir_value(),
                        )
                    )
                    selected = (index >= fx.Int32(group * warp_size)).select(
                        value, selected
                    )
                return selected

            if const_expr(paired):
                mask_row = get_mask_row()
                full_count = load_uniform_i32(full_kv_num_blocks_view, mask_row)
                partial_count = load_uniform_i32(kv_num_blocks_view, mask_row)
                partial_base = mask_row * fx.Int32(max_partial_blocks)
                full1 = load_uniform_i32(
                    full_kv_num_blocks_view, mask_row + fx.Int32(1)
                )
                partial1 = load_uniform_i32(kv_num_blocks_view, mask_row + fx.Int32(1))
                full_words = []
                partial_words = []
                paired_full_count = fx.Int32(0)
                paired_partial_count = fx.Int32(0)
                for word in fx.range_constexpr((paired_kv_blocks + 31) // 32):
                    f0 = pair_word(
                        full_kv_indices_view,
                        mask_row,
                        full_count,
                        max_full_blocks,
                        word,
                    )
                    f1 = pair_word(
                        full_kv_indices_view,
                        mask_row + fx.Int32(1),
                        full1,
                        max_full_blocks,
                        word,
                    )
                    p0 = pair_word(
                        kv_indices_view,
                        mask_row,
                        partial_count,
                        max_partial_blocks,
                        word,
                    )
                    p1 = pair_word(
                        kv_indices_view,
                        mask_row + fx.Int32(1),
                        partial1,
                        max_partial_blocks,
                        word,
                    )
                    common = f0 & f1
                    partial = (f0 | f1 | p0 | p1) & ~common
                    full_words.append(common)
                    partial_words.append(partial)
                    paired_full_count = paired_full_count + fx.Int32(
                        fx.math.ctpop(common)
                    )
                    paired_partial_count = paired_partial_count + fx.Int32(
                        fx.math.ctpop(partial)
                    )
                full_count = paired_full_count
                partial_count = paired_partial_count
                paired_full_cache = pair_cache(full_words)

            def select_query_pack(keep, pack):
                return Vec.from_elements(
                    [
                        keep.select(pack[element], element_type(0.0))
                        for element in fx.range_constexpr(values_per_thread)
                    ],
                    element_type,
                )

            # Cache Q once so its LDS reads do not compete with K/V staging.
            query_register_packs = []
            if const_expr(
                pipelined_kv and not decode and not paired and not single_pass
            ):
                query_shared_layout = make_qk_shared_layout(
                    query_tile_rows, qk_head_dim
                )
                query_copy_coordinates = fx.make_composed_layout(
                    fx.right_inverse(query_shared_layout.outer),
                    fx.make_composed_layout(
                        query_shared_layout.inner,
                        fx.make_layout(query_tile_rows * qk_head_dim, 1),
                    ),
                )
                query_destinations = fx.logical_divide(
                    fx.make_view(
                        shared_query_pointer,
                        fx.make_layout(query_tile_rows * qk_head_dim, 1),
                    ),
                    fx.make_layout(values_per_thread, 1),
                )
                for load_step in fx.range_constexpr(query_load_iterations):
                    linear = fx.Int32(load_step * num_threads) + tid
                    logical = fx.Int32(
                        fx.get_scalar(
                            fx.crd2idx(
                                linear * fx.Int32(values_per_thread),
                                query_copy_coordinates,
                            )
                        )
                    )
                    row = logical % fx.Int32(query_tile_rows)
                    chunk = logical // fx.Int32(query_tile_rows * values_per_thread)
                    if const_expr(query_columns < qk_head_dim):
                        # Packs past the stored columns load from a row past
                        # the buffer bound, which reads as zeros.
                        row = (
                            chunk < fx.Int32(query_columns // values_per_thread)
                        ).select(
                            row, fx.Int32(seq_q + qk_head_dim // values_per_thread)
                        )
                    source = fx.logical_divide(
                        fx.slice(query_view, (q_base + row, None)),
                        fx.make_layout(values_per_thread, 1),
                    )
                    fx.copy(
                        lds_copy,
                        fx.slice(source, (None, chunk)),
                        fx.slice(query_destinations, (None, linear)),
                    )
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.rocdl.s_barrier()
                for k_step in fx.range_constexpr(qk_reduction_steps):
                    tile = fx.slice(query_tiles, (None, None, q_wave, k_step))
                    fragment = thr_mma.make_fragment_B(tile)
                    fx.copy(
                        shared_copy, copy_q.partition_S(tile), copy_q.retile(fragment)
                    )
                    query_register_packs.append(Vec(fragment.load()))
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.rocdl.s_barrier()
            elif const_expr(pipelined_kv):
                if const_expr(paired or (single_pass and not decode)):
                    query_source = fx.slice(query_view, (query_pos, None))
                else:
                    local_head = query_head - kv_head * fx.Int32(
                        query_heads_per_kv_head
                    )
                    query_source = fx.slice(query_view, (local_head, query_pos, None))
                query_row_packs = fx.logical_divide(
                    query_source,
                    fx.make_layout(values_per_thread, 1),
                )
                for k_step in fx.range_constexpr(qk_reduction_steps):
                    column = fx.Int32(fx.get_scalar(query_k_coordinates[0, 0, k_step]))
                    q_fragment = fx.make_rmem_tensor(values_per_thread, element_type)
                    fx.copy(
                        global_copy,
                        fx.slice(
                            query_row_packs,
                            (None, column // fx.Int32(values_per_thread)),
                        ),
                        q_fragment,
                    )
                    query_pack = Vec(q_fragment.load())
                    if const_expr(query_columns < qk_head_dim):
                        query_pack = select_query_pack(
                            column < fx.Int32(query_columns), query_pack
                        )
                    query_register_packs.append(query_pack)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            else:
                for load_step in fx.range_constexpr(query_load_iterations):
                    linear = fx.Int32(load_step * num_threads) + tid
                    row = linear // fx.Int32(qk_chunks_per_row)
                    chunk = linear % fx.Int32(qk_chunks_per_row)
                    column = chunk * fx.Int32(values_per_thread)
                    row_valid, row_head, row_query_pos = row_coordinates(row)
                    query_fragment = fx.make_rmem_tensor(
                        values_per_thread,
                        element_type,
                    )
                    if const_expr(decode):
                        local_head = row_head - kv_head * fx.Int32(
                            query_heads_per_kv_head
                        )
                        source_row = fx.slice(
                            query_view,
                            (local_head, row_query_pos, None),
                        )
                    else:
                        source_row = fx.slice(query_view, (row_query_pos, None))
                    source = fx.logical_divide(
                        source_row,
                        fx.make_layout(values_per_thread, 1),
                    )
                    fx.copy(
                        global_copy,
                        fx.slice(source, (None, chunk)),
                        query_fragment,
                    )
                    query_value = Vec(query_fragment.load())
                    if const_expr(query_columns < qk_head_dim):
                        column_valid = column < fx.Int32(query_columns)
                        row_valid = (
                            row_valid & column_valid
                            if const_expr(decode)
                            else column_valid
                        )
                    if const_expr(decode or query_columns < qk_head_dim):
                        query_value = select_query_pack(row_valid, query_value)
                    query_destination = fx.logical_divide(
                        fx.slice(shared_query, (row, None)),
                        fx.make_layout(values_per_thread, 1),
                    )
                    query_fragment.store(query_value)
                    fx.copy(
                        shared_copy,
                        query_fragment,
                        fx.slice(query_destination, (None, chunk)),
                    )
                fx.gpu.barrier()

            def load_query_fragment(k_step):
                tile = fx.slice(query_tiles, (None, None, q_wave, k_step))
                fragment = thr_mma.make_fragment_B(tile)
                if const_expr(pipelined_kv):
                    fragment.store(Vec(query_register_packs[k_step]))
                else:
                    fx.copy(
                        shared_copy, copy_q.partition_S(tile), copy_q.retile(fragment)
                    )
                return fragment

            def reduce_lane_pair(value, maximum):
                raw = fx.Int32(
                    Vec.from_elements([value], fx.Float32).bitcast(fx.Int32)[0]
                ).ir_value()
                swapped = fx.rocdl.permlane32_swap(
                    ir.Type.parse("!llvm.struct<(i32, i32)>"), raw, raw, False, True
                )
                lhs = _f32(
                    Vec.from_elements(
                        [fx.Int32(llvm.extractvalue(fx.Int32.ir_type, swapped, [0]))],
                        fx.Int32,
                    ).bitcast(fx.Float32)[0]
                )
                rhs = _f32(
                    Vec.from_elements(
                        [fx.Int32(llvm.extractvalue(fx.Int32.ir_type, swapped, [1]))],
                        fx.Int32,
                    ).bitcast(fx.Float32)[0]
                )
                return _maximum(lhs, rhs) if maximum else lhs + rhs

            zero16 = Vec.filled(16, 0.0, fx.Float32)
            score_scale = Vec.filled(16, scale_log2, fx.Float32)

            def scale_scores(scores):
                # The backward recomputes exp2(S * scale_log2 - lse) from f32 S
                # without contraction; prescaling a rounded Q or fusing the
                # subtract makes the two exponents disagree for large logits.
                return Vec(scores) * score_scale

            output_accumulators = [zero16 for _ in fx.range_constexpr(output_chunks)]
            running_max = _f32(_NEG_BIG)
            running_sum = _f32(0.0)

            if const_expr(not paired):
                mask_row = get_mask_row()

            def stage_kv(kv_base, stage, key_tile, first, last):
                view = key_view if key_tile else value_view
                stride = k_stride if key_tile else v_stride
                coordinates = (
                    key_copy_coordinates if key_tile else value_copy_coordinates
                )
                pointers = shared_key_stages if key_tile else shared_value_stages
                destinations = (
                    key_copy_destinations if key_tile else value_copy_destinations
                )

                def pack_coordinates(load_step):
                    load_tid = lane if const_expr(split_kv) else tid
                    linear = fx.Int32(load_step * kv_load_threads) + load_tid
                    logical = fx.Int32(
                        fx.get_scalar(
                            fx.crd2idx(
                                linear * fx.Int32(values_per_thread), coordinates
                            )
                        )
                    )
                    row = logical % fx.Int32(kv_tile_rows)
                    chunk = logical // fx.Int32(kv_tile_rows * values_per_thread)
                    return linear, row, chunk

                def destination(load_step):
                    # LDS DMA adds the lane offset; keep its base wave-uniform.
                    return fx.make_view(
                        pointers[stage]
                        + (
                            fx.Int32(load_step * kv_load_threads)
                            + wave * fx.Int32(warp_size)
                        )
                        * fx.Int32(values_per_thread),
                        fx.make_layout(values_per_thread, 1),
                    )

                if const_expr(paired and qk_head_dim > v_head_dim):
                    offsets = []
                    for load_step in fx.range_constexpr(first, last):
                        linear, row, chunk = pack_coordinates(load_step)
                        if const_expr(key_tile):
                            offset = row * fx.Int32(stride[2])
                        else:
                            offset = (kv_base + row) * fx.Int32(stride[2])
                        offsets.append(offset + chunk * fx.Int32(values_per_thread))
                    if const_expr(not key_tile):
                        # AMDGPU vector constraints require a supported register tuple size.
                        padded_count = max(2, 1 << (len(offsets) - 1).bit_length())
                        offsets = offsets + [offsets[0]] * (padded_count - len(offsets))
                    packed = Vec.from_elements(offsets, fx.Int32)
                    if const_expr(not key_tile):
                        raw = packed.ir_value()
                        packed = Vec(
                            llvm.inline_asm(
                                raw.type, [raw], "", "=v,0", has_side_effects=True
                            )
                        )
                    for load_step in fx.range_constexpr(first, last):
                        source = fx.make_view(
                            fx.get_iter(view) + fx.Int32(packed[load_step - first]),
                            fx.make_layout(values_per_thread, 1),
                        )
                        copy = lds_copy
                        if const_expr(key_tile):
                            copy = copy.set_value(
                                "soffset", kv_base * fx.Int32(stride[2])
                            )
                        fx.copy(copy, source, destination(load_step))
                else:
                    columns = key_columns if key_tile else value_columns
                    width = qk_head_dim if key_tile else v_head_dim
                    for load_step in fx.range_constexpr(first, last):
                        linear, row, chunk = pack_coordinates(load_step)
                        if const_expr(columns < width):
                            # Packs past the stored columns load from a row past
                            # the buffer bound, which reads as zeros.
                            row = (
                                chunk < fx.Int32(columns // values_per_thread)
                            ).select(row, fx.Int32(seq_kv + width // values_per_thread))
                        source = fx.logical_divide(
                            fx.slice(view, (kv_base + row, None)),
                            fx.make_layout(values_per_thread, 1),
                        )
                        if const_expr(paired):
                            fx.copy(
                                lds_copy,
                                fx.slice(source, (None, chunk)),
                                destination(load_step),
                            )
                        else:
                            fx.copy(
                                lds_copy,
                                fx.slice(source, (None, chunk)),
                                fx.slice(destinations[stage], (None, linear)),
                            )

            def stage_key(kv_base, stage=0, first=0, last=key_load_iterations):
                stage_kv(kv_base, stage, True, first, last)

            def stage_value(kv_base, stage=0):
                stage_kv(kv_base, stage, False, 0, value_load_iterations)

            def stage_columns(kv_base, key_tile, column_stage):
                view = key_view if key_tile else value_view
                coordinates = (
                    key_copy_coordinates if key_tile else value_copy_coordinates
                )
                buffer = column_stage % 2
                destinations = (
                    key_copy_destinations if key_tile else value_copy_destinations
                )[buffer]
                columns = key_columns if key_tile else value_columns
                width = qk_head_dim if key_tile else v_head_dim
                stage_chunks = (
                    key_stage_columns if key_tile else value_stage_columns
                ) // values_per_thread
                for load_step in fx.range_constexpr(
                    key_stage_loads if key_tile else value_stage_loads
                ):
                    linear = fx.Int32(load_step * num_threads) + tid
                    logical = fx.Int32(
                        fx.get_scalar(
                            fx.crd2idx(
                                linear * fx.Int32(values_per_thread), coordinates
                            )
                        )
                    )
                    row = logical % fx.Int32(kv_tile_rows)
                    chunk = logical // fx.Int32(kv_tile_rows * values_per_thread)
                    if const_expr(column_stage):
                        chunk = chunk + fx.Int32(column_stage * stage_chunks)
                    if const_expr(columns < width):
                        row = (chunk < fx.Int32(columns // values_per_thread)).select(
                            row, fx.Int32(seq_kv + width // values_per_thread)
                        )
                    source = fx.logical_divide(
                        fx.slice(view, (kv_base + row, None)),
                        fx.make_layout(values_per_thread, 1),
                    )
                    fx.copy(
                        lds_copy,
                        fx.slice(source, (None, chunk)),
                        fx.slice(destinations, (None, linear)),
                    )

            def wait_stage(newer_loads):
                # Loads retire in issue order: leaving the newer stage DMAs
                # outstanding retires every older one, then the barrier
                # publishes all waves' copies.
                fx.rocdl.sched_barrier(0)
                fx.rocdl.s_waitcnt(vmcnt=newer_loads, lgkmcnt=0, expcnt=0)
                phase_fence()

            def release_stage():
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                phase_fence()

            def single_pass_scores(kv_base):
                scores_lo = zero16
                scores_hi = zero16
                for stage in fx.range_constexpr(key_stages):
                    more = stage + 1 < key_stages
                    if const_expr(stage == 0 and value_stages == 1):
                        release_stage()
                    if const_expr(more):
                        stage_columns(kv_base, True, stage + 1)
                    if const_expr(stage == 0):
                        stage_columns(kv_base, False, 0)
                    wait_stage((key_stage_loads if more else 0) + value_stage_loads)
                    for step in fx.range_constexpr(key_stage_steps):
                        query_pack = load_query_fragment(stage * key_stage_steps + step)
                        key_lo = load_key_fragment(step, False, stage % 2)
                        key_hi = load_key_fragment(step, True, stage % 2)
                        scores_lo = mfma(key_lo, query_pack, scores_lo)
                        scores_hi = mfma(key_hi, query_pack, scores_hi)
                    schedule_fwd_qk_pipeline(
                        reduction_steps=key_stage_steps, query_in_registers=True
                    )
                return scores_lo, scores_hi

            def single_pass_output(
                probability_packs, tile_output, kv_base, prefetch_base
            ):
                for stage in fx.range_constexpr(value_stages):
                    more = stage + 1 < value_stages
                    if const_expr(stage == 0 and key_stages == 1):
                        release_stage()
                    if const_expr(more):
                        stage_columns(kv_base, False, stage + 1)
                    if const_expr(stage == 0):
                        stage_columns(prefetch_base, True, 0)
                    wait_stage((value_stage_loads if more else 0) + key_stage_loads)
                    for pack in fx.range_constexpr(4):
                        probability_fragment = fx.make_fragment_like(
                            probability_coordinates, element_type
                        )
                        probability_fragment.store(Vec(probability_packs[pack]))
                        for chunk in fx.range_constexpr(shard_stage_chunks):
                            tile = fx.slice(
                                shard_value_tiles[stage % 2],
                                (None, None, chunk, pack),
                            )
                            value_pack = thr_mma.make_fragment_A(tile)
                            fx.copy(
                                transposed_lds_copy,
                                copy_v.partition_S(tile),
                                copy_v.retile(value_pack),
                            )
                            index = stage * shard_stage_chunks + chunk
                            tile_output[index] = mfma(
                                value_pack, probability_fragment, tile_output[index]
                            )
                    schedule_fwd_pv_pipeline(output_chunks=shard_stage_chunks)
                return tile_output

            def load_key_fragment(k_step, high_half, stage=0):
                tile = fx.slice(key_tiles[stage], (None, None, int(high_half), k_step))
                fragment = thr_mma.make_fragment_A(tile)
                fx.copy(shared_copy, copy_k.partition_S(tile), copy_k.retile(fragment))
                return fragment

            def load_value_fragment(probability_pack, d_chunk, stage=0):
                tile = fx.slice(
                    value_tiles[stage], (None, None, d_chunk, probability_pack)
                )
                fragment = thr_mma.make_fragment_A(tile)
                fx.copy(
                    transposed_lds_copy,
                    copy_v.partition_S(tile),
                    copy_v.retile(fragment),
                )
                return fragment

            def accumulate_probability(pack_values, pack_index, tile_output, stage):
                probability_fragment = fx.make_fragment_like(
                    probability_coordinates, element_type
                )
                probability_fragment.store(Vec(pack_values))
                for d_chunk in fx.range_constexpr(output_chunks):
                    value_pack = load_value_fragment(pack_index, d_chunk, stage)
                    tile_output[d_chunk] = mfma(
                        value_pack, probability_fragment, tile_output[d_chunk]
                    )
                return tile_output

            def apply_score_mod(raw, key_base):
                # Move the modified score to log2 units.
                if const_expr(capture_table is not None):
                    keys = [
                        key_base
                        + fx.Int32(fx.get_scalar(accumulator_coordinates[element]))
                        for element in fx.range_constexpr(16)
                    ]
                    readers = make_tile_score_mod_readers(
                        score_mod_aux, score_mod_buffers, keys
                    )
                    batch_index = fx.Int32(batch)
                    return [
                        _f32(
                            score_mod(
                                _f32(raw[element]) * _f32(scale),
                                batch_index,
                                query_head,
                                query_pos,
                                keys[element],
                                readers,
                            )
                        )
                        * _f32(_LOG2E)
                        for element in fx.range_constexpr(16)
                    ]
                return [
                    _f32(
                        score_mod(
                            _f32(raw[element]) * _f32(scale),
                            fx.Int32(batch),
                            query_head,
                            query_pos,
                            key_base
                            + fx.Int32(fx.get_scalar(accumulator_coordinates[element])),
                            score_mod_readers,
                        )
                    )
                    * _f32(_LOG2E)
                    for element in fx.range_constexpr(16)
                ]

            def process_tile_body(
                kv_chunk,
                masked,
                tile_output,
                tile_running_max,
                tile_running_sum,
                stage=0,
                tile_active=None,
                tile_exact_max=None,
                prepared_keep=None,
                tile_full=None,
                prefetch_chunk=None,
            ):
                kv_base = kv_chunk * fx.Int32(kv_tile_rows)
                if const_expr(not pipelined_kv):
                    stage_key(kv_base)
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    fx.gpu.barrier()

                if const_expr(single_pass):
                    scores_lo, scores_hi = single_pass_scores(kv_base)
                else:
                    scores_lo = zero16
                    scores_hi = zero16
                    for k_step in fx.range_constexpr(qk_reduction_steps):
                        query_pack = load_query_fragment(k_step)
                        key_lo = load_key_fragment(k_step, False, stage)
                        key_hi = load_key_fragment(k_step, True, stage)
                        scores_lo = mfma(key_lo, query_pack, scores_lo)
                        scores_hi = mfma(key_hi, query_pack, scores_hi)
                    schedule_fwd_qk_pipeline(
                        reduction_steps=qk_reduction_steps,
                        vmem_count=(key_load_iterations if pipelined_kv else 0),
                        query_in_registers=pipelined_kv,
                    )

                if const_expr(not pipelined_kv):
                    # Every wave must finish its K reads before the shared allocation
                    # is reused for V.
                    fx.gpu.barrier()
                    stage_value(kv_base)

                if const_expr(has_score_mod):
                    raw_lo = Vec(scores_lo)
                    raw_hi = Vec(scores_hi)
                else:
                    raw_lo = scale_scores(scores_lo)
                    raw_hi = scale_scores(scores_hi)
                score_values = []
                keep_values = []
                for half in fx.range_constexpr(2):
                    raw = raw_lo if half == 0 else raw_hi
                    if const_expr(has_score_mod):
                        raw = apply_score_mod(raw, kv_base + fx.Int32(32 * half))
                    half_mask_groups = []
                    if const_expr(
                        prepared_keep is None and bool(vector_mask_loads) and masked
                    ):
                        for group in fx.range_constexpr(16 // mask_load_width):
                            first_key = (
                                kv_base
                                + fx.Int32(32 * half)
                                + fx.Int32(
                                    fx.get_scalar(
                                        accumulator_coordinates[group * mask_load_width]
                                    )
                                )
                            )
                            half_mask_groups.append(load_mask_group(first_key))
                    for element in fx.range_constexpr(16):
                        key_pos = (
                            kv_base
                            + fx.Int32(32 * half)
                            + fx.Int32(fx.get_scalar(accumulator_coordinates[element]))
                        )
                        if const_expr(prepared_keep is not None):
                            keep = prepared_keep[half * 16 + element]
                        else:
                            keep = query_valid
                            if tile_active is not None:
                                keep = keep & tile_active
                            if const_expr(bool(mask_program)):
                                if masked:
                                    cached_mask = dict(query_mask_values)
                                    if const_expr(bool(vector_mask_loads)):
                                        mask_group = half_mask_groups[
                                            element // mask_load_width
                                        ]
                                        cached_mask.update(
                                            {
                                                slot: mask_group_value(
                                                    group, element % mask_load_width
                                                )
                                                for slot, group in mask_group.items()
                                            }
                                        )
                                    element_keep = evaluate_mask(
                                        query_pos, key_pos, cached_mask
                                    )
                                    if tile_full is not None:
                                        element_keep = tile_full | element_keep
                                    keep = keep & element_keep

                        keep_values.append(keep)
                        score = keep.select(_f32(raw[element]), _f32(_NEG_BIG))
                        if const_expr(kv_ragged):
                            score = (key_pos < fx.Int32(seq_kv)).select(
                                score, _f32(_NEG_PAST_END)
                            )
                        score_values.append(score)

                max_levels = score_values
                for level in fx.range_constexpr(5):
                    max_levels = [
                        _maximum(max_levels[2 * pair], max_levels[2 * pair + 1])
                        for pair in fx.range_constexpr(32 >> (level + 1))
                    ]
                local_max = max_levels[0]
                if const_expr(pipelined_kv):
                    tile_max = reduce_lane_pair(local_max, True)
                else:
                    peer_max = _f32(
                        fx.gpu.shuffle_xor(local_max, mfma_tile_size, warp_size)
                    )
                    tile_max = _maximum(local_max, peer_max)
                tile_exact_max = _maximum(tile_exact_max, tile_max)
                if const_expr(pipelined_kv):
                    rescale = fx.Uint64(
                        fx.rocdl.ballot(
                            fx.Uint64.ir_type,
                            (tile_max > tile_running_max + _f32(8.0)).ir_value(),
                        )
                    ) != fx.Uint64(0)
                    new_max = rescale.select(
                        _maximum(tile_running_max, tile_max), tile_running_max
                    )
                    correction = _f32(1.0)
                    for d in fx.range_constexpr(output_chunks):
                        raw = Vec(tile_output[d]).ir_value()
                        tile_output[d] = Vec(
                            llvm.inline_asm(
                                raw.type, [raw], "", "=v,0", has_side_effects=True
                            )
                        )
                    if rescale:
                        correction = _exp2(tile_running_max - new_max)
                        correction_vec = Vec.from_elements(
                            [correction], fx.Float32
                        ).broadcast_to(16)
                        for d in fx.range_constexpr(output_chunks):
                            tile_output[d] = Vec(tile_output[d]) * correction_vec
                    for d in fx.range_constexpr(output_chunks):
                        raw = Vec(tile_output[d]).ir_value()
                        tile_output[d] = Vec(
                            llvm.inline_asm(
                                raw.type, [raw], "", "=v,0", has_side_effects=True
                            )
                        )

                else:
                    new_max = _maximum(tile_running_max, tile_max)
                    correction = _exp2(tile_running_max - new_max)

                    correction_vec = Vec.from_elements(
                        [correction], fx.Float32
                    ).broadcast_to(16)
                    for d_chunk in fx.range_constexpr(output_chunks):
                        tile_output[d_chunk] = (
                            Vec(tile_output[d_chunk]) * correction_vec
                        )

                stream_probabilities = (
                    not paired
                    and not single_pass
                    and pipelined_kv
                    and mask_buffer_count > 0
                    and masked
                )
                shifted_scores = []
                for pack in fx.range_constexpr(4):
                    shifted = Vec.from_elements(
                        [
                            score_values[pack * 8 + i] - new_max
                            for i in fx.range_constexpr(8)
                        ],
                        fx.Float32,
                    )
                    if const_expr(pipelined_kv and not masked):
                        raw = shifted.ir_value()
                        shifted = Vec(
                            llvm.inline_asm(
                                raw.type, [raw], "", "=v,0", has_side_effects=True
                            )
                        )
                    shifted_scores.extend([shifted[i] for i in fx.range_constexpr(8)])
                sum_probabilities = []
                probability_packs = []
                tie_stream = stream_probabilities and (
                    wide_head or tie_streamed_probabilities or tie_scalar_mask_loads
                )
                for pack in fx.range_constexpr(4):
                    pack_probabilities = []
                    if const_expr(tie_stream):
                        exponents = tied_nop(
                            Vec.from_elements(
                                [
                                    _exp2(shifted_scores[pack * 8 + i])
                                    for i in fx.range_constexpr(8)
                                ],
                                fx.Float32,
                            ),
                            0,
                        )
                    for pack_element in fx.range_constexpr(8):
                        element = pack * 8 + pack_element
                        probability = keep_values[element].select(
                            _f32(exponents[pack_element])
                            if const_expr(tie_stream)
                            else _exp2(shifted_scores[element]),
                            _f32(0.0),
                        )
                        pack_probabilities.append(probability)
                        sum_probabilities.append(probability)
                    pack_values = Vec.from_elements(pack_probabilities, fx.Float32).to(
                        element_type
                    )
                    if const_expr(stream_probabilities):
                        if const_expr(tie_stream):
                            pack_values = tied_nop(pack_values, 1)
                        tile_output = accumulate_probability(
                            pack_values, pack, tile_output, stage
                        )
                    else:
                        if const_expr(has_score_mod and capture_table is None):
                            pack_values = tied_nop(
                                pack_values, 1, has_side_effects=False
                            )
                        probability_packs.append(pack_values)
                sum_levels = sum_probabilities
                for level in fx.range_constexpr(5):
                    sum_levels = [
                        sum_levels[2 * pair] + sum_levels[2 * pair + 1]
                        for pair in fx.range_constexpr(32 >> (level + 1))
                    ]
                local_sum = sum_levels[0]
                if const_expr(not stream_probabilities):
                    schedule_fwd_softmax_pipeline(
                        vmem_count=0 if single_pass else value_load_iterations
                    )
                if const_expr(pipelined_kv):
                    tile_sum = reduce_lane_pair(local_sum, False)
                else:
                    peer_sum = _f32(
                        fx.gpu.shuffle_xor(local_sum, mfma_tile_size, warp_size)
                    )
                    tile_sum = local_sum + peer_sum
                tile_running_sum = tile_running_sum * correction + tile_sum
                tile_running_max = new_max

                if const_expr(not pipelined_kv):
                    # V writes were issued before the register-only softmax.
                    # Synchronize only when the LDS data is actually consumed.
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    fx.gpu.barrier()
                if const_expr(single_pass):
                    tile_output = single_pass_output(
                        probability_packs,
                        tile_output,
                        kv_base,
                        prefetch_chunk * fx.Int32(kv_tile_rows),
                    )
                elif const_expr(not stream_probabilities):
                    for probability_pack in fx.range_constexpr(4):
                        tile_output = accumulate_probability(
                            probability_packs[probability_pack],
                            probability_pack,
                            tile_output,
                            stage,
                        )
                    schedule_fwd_pv_pipeline(output_chunks=output_chunks)

                if const_expr(not pipelined_kv):
                    # Protect V from the next tile's K staging.
                    fx.gpu.barrier()
                return tile_output, tile_running_max, tile_running_sum, tile_exact_max

            def wave_mask_bounds(kv_chunk):
                if const_expr(mask_buffer_count):
                    query_lo = query_pos
                    query_hi = query_pos
                else:
                    query_lo = q_base + wave * fx.Int32(mfma_tile_size)
                    query_hi = query_lo + fx.Int32(mfma_tile_size - 1)
                key_lo = kv_chunk * fx.Int32(kv_tile_rows)
                key_hi = key_lo + fx.Int32(kv_tile_rows - 1)
                intervals = [
                    (fx.Int32(batch), fx.Int32(batch)),
                    (fx.Int32(query_head), fx.Int32(query_head)),
                    (query_lo, query_hi),
                    (key_lo, key_hi),
                ]
                for instruction in mask_program:
                    op = instruction[0]
                    if op == "const_i32":
                        bound_value = fx.Int32(instruction[1])
                        intervals.append((bound_value, bound_value))
                    elif op == "const_bool":
                        bound_value = fx.Int32(1 if instruction[1] else 0) == fx.Int32(
                            1
                        )
                        intervals.append((bound_value, bound_value))
                    elif op == "load_i32":
                        buffer_index, index_ids = instruction[1:]
                        offset = fx.Int32(0)
                        for dimension, index in enumerate(index_ids):
                            offset = offset + intervals[index][0] * fx.Int32(
                                mask_buffer_strides[buffer_index][dimension]
                            )
                        loaded_bound = load_i32(mask_buffers[buffer_index], offset)
                        intervals.append((loaded_bound, loaded_bound))
                    else:
                        lhs_lo, lhs_hi = intervals[instruction[1]]
                        if op == "not":
                            intervals.append((~lhs_hi, ~lhs_lo))
                        else:
                            rhs_lo, rhs_hi = intervals[instruction[2]]
                            if op == "ge":
                                intervals.append((lhs_lo >= rhs_hi, lhs_hi >= rhs_lo))
                            elif op == "gt":
                                intervals.append((lhs_lo > rhs_hi, lhs_hi > rhs_lo))
                            elif op == "le":
                                intervals.append((lhs_hi <= rhs_lo, lhs_lo <= rhs_hi))
                            elif op == "lt":
                                intervals.append((lhs_hi < rhs_lo, lhs_lo < rhs_hi))
                            elif op == "eq":
                                certain = (
                                    (lhs_lo == lhs_hi)
                                    & (rhs_lo == rhs_hi)
                                    & (lhs_lo == rhs_lo)
                                )
                                possible = (lhs_lo <= rhs_hi) & (rhs_lo <= lhs_hi)
                                intervals.append((certain, possible))
                            elif op == "ne":
                                certain = (lhs_hi < rhs_lo) | (rhs_hi < lhs_lo)
                                possible = (
                                    (lhs_lo != lhs_hi)
                                    | (rhs_lo != rhs_hi)
                                    | (lhs_lo != rhs_lo)
                                )
                                intervals.append((certain, possible))
                            elif op == "and":
                                intervals.append((lhs_lo & rhs_lo, lhs_hi & rhs_hi))
                            elif op == "or":
                                intervals.append((lhs_lo | rhs_lo, lhs_hi | rhs_hi))
                return intervals[mask_program_output]

            def process_tile(
                kv_chunk,
                masked,
                tile_output,
                tile_running_max,
                tile_running_sum,
                stage=0,
                tile_active=None,
                tile_exact_max=None,
                tile_full=None,
                prefetch_chunk=None,
            ):
                if const_expr(tile_exact_max is None):
                    tile_exact_max = tile_running_max
                if const_expr(paired and mask_interval_supported and masked):
                    certain, possible = wave_mask_bounds(kv_chunk)
                    if tile_full is not None:
                        possible = possible | tile_full
                    if tile_active is not None:
                        possible = possible & tile_active
                    if const_expr(mask_buffer_count):
                        possible = fx.Uint64(
                            fx.rocdl.ballot(fx.Uint64.ir_type, possible.ir_value())
                        ) != fx.Uint64(0)
                    if possible:
                        (
                            tile_output,
                            tile_running_max,
                            tile_running_sum,
                            tile_exact_max,
                        ) = process_tile_body(
                            kv_chunk,
                            True,
                            tile_output,
                            tile_running_max,
                            tile_running_sum,
                            stage,
                            tile_active,
                            tile_exact_max,
                            tile_full=tile_full,
                        )
                    return (
                        tile_output,
                        tile_running_max,
                        tile_running_sum,
                        tile_exact_max,
                    )
                elif const_expr(
                    not paired and pipelined_kv and masked and bool(vector_mask_loads)
                ):
                    prepared_keep = []
                    kv_base = kv_chunk * fx.Int32(kv_tile_rows)
                    for half in fx.range_constexpr(2):
                        mask_groups = []
                        for group in fx.range_constexpr(16 // mask_load_width):
                            first_key = (
                                kv_base
                                + fx.Int32(32 * half)
                                + fx.Int32(
                                    fx.get_scalar(
                                        accumulator_coordinates[group * mask_load_width]
                                    )
                                )
                            )
                            mask_groups.append(load_mask_group(first_key))
                        for element in fx.range_constexpr(16):
                            key_pos = (
                                kv_base
                                + fx.Int32(32 * half)
                                + fx.Int32(
                                    fx.get_scalar(accumulator_coordinates[element])
                                )
                            )
                            cached_mask = {
                                **query_mask_values,
                                **{
                                    slot: mask_group_value(
                                        group, element % mask_load_width
                                    )
                                    for slot, group in mask_groups[
                                        element // mask_load_width
                                    ].items()
                                },
                            }
                            keep = query_valid & evaluate_mask(
                                query_pos, key_pos, cached_mask
                            )
                            if tile_active is not None:
                                keep = keep & tile_active
                            prepared_keep.append(keep)
                    if const_expr(single_pass):
                        # Every wave takes the staging barriers of every tile.
                        return process_tile_body(
                            kv_chunk,
                            masked,
                            tile_output,
                            tile_running_max,
                            tile_running_sum,
                            tile_exact_max=tile_exact_max,
                            prepared_keep=prepared_keep,
                            prefetch_chunk=prefetch_chunk,
                        )
                    levels = prepared_keep
                    for level in fx.range_constexpr(5):
                        levels = [
                            levels[2 * pair] | levels[2 * pair + 1]
                            for pair in fx.range_constexpr(32 >> (level + 1))
                        ]
                    any_work = fx.Uint64(
                        fx.rocdl.ballot(fx.Uint64.ir_type, levels[0].ir_value())
                    ) != fx.Uint64(0)
                    if any_work:
                        (
                            tile_output,
                            tile_running_max,
                            tile_running_sum,
                            tile_exact_max,
                        ) = process_tile_body(
                            kv_chunk,
                            masked,
                            tile_output,
                            tile_running_max,
                            tile_running_sum,
                            stage=stage,
                            tile_active=tile_active,
                            tile_exact_max=tile_exact_max,
                            prepared_keep=prepared_keep,
                        )
                    return (
                        tile_output,
                        tile_running_max,
                        tile_running_sum,
                        tile_exact_max,
                    )
                return process_tile_body(
                    kv_chunk,
                    masked,
                    tile_output,
                    tile_running_max,
                    tile_running_sum,
                    stage=stage,
                    tile_active=tile_active,
                    tile_exact_max=tile_exact_max,
                    tile_full=tile_full,
                    prefetch_chunk=prefetch_chunk,
                )

            def process_single_pass_run(
                block_count, block_indices, block_base, masked, run_state
            ):
                run_results = run_state
                if block_count > fx.Int32(0):
                    first_block = load_uniform_i32(block_indices, block_base)
                    stage_columns(first_block * fx.Int32(sparse_kv_block_size), True, 0)
                    results = [first_block, *run_state]
                    for block_index, iter_args in range(
                        fx.Int32(0),
                        block_count,
                        fx.Int32(1),
                        init=results,
                    ):
                        current_block = fx.Int32(iter_args[0])
                        state = [
                            [
                                iter_args[3 + d]
                                for d in fx.range_constexpr(output_chunks)
                            ],
                            _f32(iter_args[1]),
                            _f32(iter_args[2]),
                            _f32(iter_args[3 + output_chunks]),
                        ]
                        # The last tile prefetches its own block again so that
                        # every tile leaves the same DMAs outstanding.
                        next_index = fx.Int32(block_index) + fx.Int32(1)
                        next_block = load_uniform_i32(
                            block_indices,
                            block_base
                            + (next_index < block_count).select(
                                next_index, fx.Int32(block_index)
                            ),
                        )
                        first_chunk = current_block * fx.Int32(
                            kv_tiles_per_sparse_block
                        )
                        for sub_block in fx.range_constexpr(kv_tiles_per_sparse_block):
                            prefetch_chunk = (
                                next_block * fx.Int32(kv_tiles_per_sparse_block)
                                if sub_block + 1 == kv_tiles_per_sparse_block
                                else first_chunk + fx.Int32(sub_block + 1)
                            )
                            state = process_tile(
                                first_chunk + fx.Int32(sub_block),
                                masked,
                                *state[:3],
                                tile_exact_max=state[3],
                                prefetch_chunk=prefetch_chunk,
                            )
                        results = yield [
                            next_block,
                            state[1],
                            state[2],
                            *state[0],
                            state[3],
                        ]
                    # Retire the last prefetch before the LDS is reused.
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    phase_fence()
                    run_results = results[1:]
                return run_results

            def process_pipelined_run(
                block_count,
                block_indices,
                block_base,
                masked,
                run_state,
                index_cache=None,
            ):
                run_results = run_state
                if block_count > fx.Int32(0):
                    if const_expr(index_cache is None):
                        first_block = load_uniform_i32(block_indices, block_base)
                    else:
                        first_block = read_pair_cache(index_cache, fx.Int32(0))
                    first_chunk = first_block * fx.Int32(kv_tiles_per_sparse_block)
                    stage_key(first_chunk * fx.Int32(kv_tile_rows), 0)
                    stage_value(first_chunk * fx.Int32(kv_tile_rows), 0)
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    if const_expr(paired):
                        phase_fence()
                    else:
                        fx.gpu.barrier()

                    pipeline_state = [first_block, *run_state]
                    pipeline_results = pipeline_state
                    for block_index, iter_args in range(
                        fx.Int32(0),
                        block_count,
                        fx.Int32(1),
                        init=pipeline_state,
                    ):
                        current_block = fx.Int32(iter_args[0])
                        iter_max = _f32(iter_args[1])
                        iter_sum = _f32(iter_args[2])
                        iter_exact_max = _f32(iter_args[3 + output_chunks])
                        iter_output = [
                            iter_args[3 + d_chunk]
                            for d_chunk in fx.range_constexpr(output_chunks)
                        ]

                        if const_expr(paired):
                            # A union block may be absent from this query block's row.
                            own_full = fx.Uint32(0)
                            own_partial = fx.Uint32(0)
                            word_index = current_block // fx.Int32(32)
                            for word in fx.range_constexpr(len(owner_full_words)):
                                selected = word_index == fx.Int32(word)
                                own_full = own_full | selected.select(
                                    owner_full_words[word], fx.Uint32(0)
                                )
                                own_partial = own_partial | selected.select(
                                    owner_partial_words[word], fx.Uint32(0)
                                )
                            bit = fx.Uint32(1) << (
                                fx.Uint32(current_block) & fx.Uint32(31)
                            )
                            tile_full = (own_full & bit) != fx.Uint32(0)
                            tile_active = ((own_full | own_partial) & bit) != fx.Uint32(
                                0
                            )
                        else:
                            tile_active = None
                            tile_full = None

                        first_chunk = current_block * fx.Int32(
                            kv_tiles_per_sparse_block
                        )
                        second_chunk = first_chunk + fx.Int32(1)
                        stage_key(second_chunk * fx.Int32(kv_tile_rows), 1)
                        stage_value(second_chunk * fx.Int32(kv_tile_rows), 1)
                        iter_output, iter_max, iter_sum, iter_exact_max = process_tile(
                            first_chunk,
                            masked,
                            iter_output,
                            iter_max,
                            iter_sum,
                            stage=0,
                            tile_exact_max=iter_exact_max,
                            tile_active=tile_active,
                            tile_full=tile_full,
                        )
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                        if const_expr(paired):
                            phase_fence()
                        else:
                            fx.gpu.barrier()

                        next_index = fx.Int32(block_index) + fx.Int32(1)
                        next_block = current_block
                        if next_index < block_count:
                            if const_expr(index_cache is None):
                                next_block = load_uniform_i32(
                                    block_indices, block_base + next_index
                                )
                            else:
                                next_block = read_pair_cache(index_cache, next_index)
                            next_chunk = next_block * fx.Int32(
                                kv_tiles_per_sparse_block
                            )
                            stage_key(next_chunk * fx.Int32(kv_tile_rows), 0)
                            stage_value(next_chunk * fx.Int32(kv_tile_rows), 0)
                        iter_output, iter_max, iter_sum, iter_exact_max = process_tile(
                            second_chunk,
                            masked,
                            iter_output,
                            iter_max,
                            iter_sum,
                            stage=1,
                            tile_exact_max=iter_exact_max,
                            tile_active=tile_active,
                            tile_full=tile_full,
                        )
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                        if const_expr(paired):
                            phase_fence()
                        else:
                            fx.gpu.barrier()

                        pipeline_results = yield [
                            next_block,
                            iter_max,
                            iter_sum,
                            *iter_output,
                            iter_exact_max,
                        ]
                    run_results = pipeline_results[1:]
                return run_results

            def process_sparse_run(
                block_count, block_indices, block_base, masked, run_state
            ):
                run_results = run_state
                for block_index, iter_args in range(
                    fx.Int32(0),
                    block_count,
                    fx.Int32(1),
                    init=run_state,
                ):
                    iter_max = _f32(iter_args[0])
                    iter_sum = _f32(iter_args[1])
                    iter_output = [
                        iter_args[2 + d_chunk]
                        for d_chunk in fx.range_constexpr(output_chunks)
                    ]
                    sparse_block = load_uniform_i32(
                        block_indices,
                        block_base + fx.Int32(block_index),
                    )
                    for sub_block in fx.range_constexpr(kv_tiles_per_sparse_block):
                        iter_output, iter_max, iter_sum, ignored_exact_max = (
                            process_tile(
                                sparse_block * fx.Int32(kv_tiles_per_sparse_block)
                                + fx.Int32(sub_block),
                                masked,
                                iter_output,
                                iter_max,
                                iter_sum,
                            )
                        )
                    run_results = yield [iter_max, iter_sum] + iter_output

                return run_results

            def process_split_run(
                block_count,
                block_indices,
                block_base,
                masked,
                run_state,
            ):
                run_results = run_state
                split_count = (block_count + fx.Int32(num_waves - 1)) // fx.Int32(
                    num_waves
                )
                for split_index, iter_args in range(
                    fx.Int32(0),
                    split_count,
                    fx.Int32(1),
                    init=run_state,
                ):
                    iter_max = _f32(iter_args[0])
                    iter_sum = _f32(iter_args[1])
                    iter_output = [
                        iter_args[2 + d_chunk]
                        for d_chunk in fx.range_constexpr(output_chunks)
                    ]
                    block_index = fx.Int32(split_index * num_waves) + wave
                    tile_active = block_index < block_count
                    safe_index = tile_active.select(block_index, fx.Int32(0))
                    sparse_block = load_uniform_i32(
                        block_indices,
                        block_base + safe_index,
                    )
                    for sub_block in fx.range_constexpr(kv_tiles_per_sparse_block):
                        iter_output, iter_max, iter_sum, ignored_exact_max = (
                            process_tile(
                                sparse_block * fx.Int32(kv_tiles_per_sparse_block)
                                + fx.Int32(sub_block),
                                masked,
                                iter_output,
                                iter_max,
                                iter_sum,
                                tile_active=tile_active,
                            )
                        )
                    run_results = yield [iter_max, iter_sum] + iter_output
                return run_results

            def reduce_split_results(split_results):
                split_max = _f32(split_results[0])
                split_sum = _f32(split_results[1])
                split_output = [
                    split_results[2 + d_chunk]
                    for d_chunk in fx.range_constexpr(output_chunks)
                ]
                reduction = fx.slice(
                    lds.reduction_stats.view(fx.make_layout((2, 2), (2, 1))),
                    (lane_half, None),
                )
                reduction_outputs = fx.slice(
                    lds.reduction_output.view(
                        fx.make_layout(
                            (2, 16, output_chunks),
                            (output_chunks * 16, 1, 16),
                        )
                    ),
                    (lane_half, None, None),
                )
                if (wave == fx.Int32(1)) & (query_row_in_wave == fx.Int32(0)):
                    reduction[0] = split_max
                    reduction[1] = split_sum
                    for d_chunk in fx.range_constexpr(output_chunks):
                        fx.slice(reduction_outputs, (None, d_chunk)).store(
                            Vec(split_output[d_chunk])
                        )
                fx.gpu.barrier()

                other_max = _f32(reduction[0])
                other_sum = _f32(reduction[1])
                combined_max = _maximum(split_max, other_max)
                split_scale = _exp2(split_max - combined_max)
                other_scale = _exp2(other_max - combined_max)
                split_sum = split_sum * split_scale + other_sum * other_scale
                split_max = combined_max
                split_scale_vec = Vec.from_elements(
                    [split_scale], fx.Float32
                ).broadcast_to(16)
                other_scale_vec = Vec.from_elements(
                    [other_scale], fx.Float32
                ).broadcast_to(16)
                for d_chunk in fx.range_constexpr(output_chunks):
                    other_output = Vec(
                        fx.slice(reduction_outputs, (None, d_chunk)).load()
                    )
                    split_output[d_chunk] = (
                        Vec(split_output[d_chunk]) * split_scale_vec
                        + other_output * other_scale_vec
                    )
                return [split_max, split_sum] + split_output

            def phase_fence():
                fx.rocdl.sched_barrier(0)
                fx.rocdl.s_barrier()
                fx.rocdl.sched_barrier(0)

            def pin_vector(values):
                raw = Vec(values).ir_value()
                return Vec(
                    llvm.inline_asm(raw.type, [raw], "", "=v,0", has_side_effects=True)
                )

            def pin_output_chunks(values):
                elements = []
                for part in fx.range_constexpr(2):
                    chunk = pin_vector(
                        Vec.from_elements(
                            [
                                _f32(Vec(values)[part * 8 + i])
                                for i in fx.range_constexpr(8)
                            ],
                            fx.Float32,
                        )
                    )
                    elements.extend([_f32(chunk[i]) for i in fx.range_constexpr(8)])
                return Vec.from_elements(elements, fx.Float32)

            def load_phase_key(half, stage):
                return [
                    load_key_fragment(step, half, stage)
                    for step in fx.range_constexpr(qk_reduction_steps)
                ]

            def phase_qk(packs):
                scores = zero16
                for step in fx.range_constexpr(qk_reduction_steps):
                    scores = mfma(packs[step], load_query_fragment(step), scores)
                return scale_scores(scores)

            def load_phase_value(half, stage):
                return [
                    load_value_fragment(pack, d, stage)
                    for d in fx.range_constexpr(half * 2, half * 2 + 2)
                    for pack in fx.range_constexpr(4)
                ]

            def phase_pv(packs, probabilities, tile_output, half):
                for d in fx.range_constexpr(2):
                    for pack in fx.range_constexpr(4):
                        p = fx.make_fragment_like(probability_coordinates, element_type)
                        p.store(Vec(probabilities[pack]))
                        out_index = half * 2 + d
                        tile_output[out_index] = mfma(
                            packs[d * 4 + pack], p, tile_output[out_index]
                        )
                return tile_output

            def phase_exp(pending, begin, end):
                values = list(pending)
                for element in fx.range_constexpr(begin, end):
                    values[element] = _exp2(_f32(pending[element]))
                return values

            def phase_pack_sum(probabilities):
                levels = probabilities
                for level in fx.range_constexpr(5):
                    levels = [
                        levels[2 * pair] + levels[2 * pair + 1]
                        for pair in fx.range_constexpr(32 >> (level + 1))
                    ]
                local = levels[0]
                total = reduce_lane_pair(local, False)
                packs = [
                    Vec.from_elements(
                        probabilities[pack * 8 : pack * 8 + 8], fx.Float32
                    ).to(element_type)
                    for pack in fx.range_constexpr(4)
                ]
                return packs, total

            def phase_max(score_lo, score_hi):
                levels = [_f32(score_lo[i]) for i in fx.range_constexpr(16)] + [
                    _f32(score_hi[i]) for i in fx.range_constexpr(16)
                ]
                for level in fx.range_constexpr(5):
                    levels = [
                        _maximum(levels[2 * pair], levels[2 * pair + 1])
                        for pair in fx.range_constexpr(32 >> (level + 1))
                    ]
                local = levels[0]
                return reduce_lane_pair(local, True)

            def phase_shift(score_lo, score_hi, maximum, begin=0):
                shifted = []
                for part in fx.range_constexpr(begin, 4):
                    score = score_lo if part < 2 else score_hi
                    offset = (part % 2) * 8
                    values = Vec.from_elements(
                        [
                            _f32(score[offset + i]) - maximum
                            for i in fx.range_constexpr(8)
                        ],
                        fx.Float32,
                    )
                    values = pin_vector(values)
                    shifted.extend([_f32(values[i]) for i in fx.range_constexpr(8)])
                return shifted

            def finish_shift(score_lo, score_hi, maximum, leading):
                shifted = [_f32(leading[i]) for i in fx.range_constexpr(8)]
                shifted.extend(phase_shift(score_lo, score_hi, maximum, 1))
                shifted = phase_exp(shifted, 0, 16)
                materialized = pin_vector(Vec.from_elements(shifted[:16], fx.Float32))
                return [
                    _f32(materialized[i]) for i in fx.range_constexpr(16)
                ] + shifted[16:]

            def update_maximum(score_lo, score_hi, tile_maximum, exact_maximum):
                tile_max = phase_max(score_lo, score_hi)
                exact_maximum = _maximum(exact_maximum, tile_max)
                rescale = fx.Uint64(
                    fx.rocdl.ballot(
                        fx.Uint64.ir_type,
                        (tile_max > tile_maximum + _f32(8.0)).ir_value(),
                    )
                ) != fx.Uint64(0)
                new_maximum = rescale.select(
                    _maximum(tile_maximum, tile_max), tile_maximum
                )
                leading = pin_vector(
                    Vec.from_elements(
                        [
                            _f32(score_lo[i]) - new_maximum
                            for i in fx.range_constexpr(8)
                        ],
                        fx.Float32,
                    )
                )
                return exact_maximum, rescale, new_maximum, leading

            def staged_phase(
                kv_chunk,
                prefetch_chunk,
                stage,
                pending,
                tile_output,
                tile_maximum,
                tile_sum,
                exact_maximum,
            ):
                if const_expr(qk_head_dim <= v_head_dim):
                    key_lo = load_phase_key(0, stage)
                    key_hi = load_phase_key(1, stage)
                    stage_value(kv_chunk * fx.Int32(kv_tile_rows), stage)
                    fx.rocdl.s_waitcnt(
                        vmcnt=key_load_iterations + value_load_iterations,
                        lgkmcnt=0,
                        expcnt=0,
                    )
                    phase_fence()

                    score_lo = zero16
                    score_hi = zero16
                    for step in fx.range_constexpr(qk_reduction_steps):
                        query_fragment = load_query_fragment(step)
                        score_lo = Vec(mfma(key_lo[step], query_fragment, score_lo))
                        score_hi = Vec(mfma(key_hi[step], query_fragment, score_hi))
                    score_lo = scale_scores(score_lo)
                    score_hi = scale_scores(score_hi)
                    previous = phase_exp(pending, 16, 32)
                    partial_sum = Vec.from_elements(
                        previous[:first_softmax_count], fx.Float32
                    ).reduce("add", fastmath="reassoc")
                    final_sum = Vec.from_elements(
                        previous[first_softmax_count:], fx.Float32
                    ).reduce("add", fastmath="reassoc")
                    tile_sum = tile_sum + reduce_lane_pair(
                        partial_sum + final_sum, False
                    )
                    sum_raw = tile_sum.ir_value()
                    tile_sum = _f32(
                        llvm.inline_asm(
                            sum_raw.type, [sum_raw], "", "=v,0", has_side_effects=True
                        )
                    )
                    pinned = pin_vector(Vec.from_elements(previous, fx.Float32))
                    previous = [_f32(pinned[i]) for i in fx.range_constexpr(32)]
                    probabilities = [
                        pin_vector(
                            Vec.from_elements(
                                previous[part * 8 : part * 8 + 8], fx.Float32
                            ).to(element_type)
                        )
                        for part in fx.range_constexpr(4)
                    ]
                    for step in fx.range_constexpr(2 * qk_reduction_steps):
                        fx.rocdl.sched_group_barrier(0x08, 1, 1)
                        if const_expr(step < 8):
                            fx.rocdl.sched_group_barrier(0x400, 2, 1)
                        else:
                            fx.rocdl.sched_group_barrier(0x02, 5, 1)
                    phase_fence()

                else:
                    key_packs = load_phase_key(0, stage)
                    if const_expr(stagger):
                        stage_value(kv_chunk * fx.Int32(kv_tile_rows), stage)
                    fx.rocdl.s_waitcnt(vmcnt=63, lgkmcnt=0, expcnt=0)
                    phase_fence()

                    score_lo = phase_qk(key_packs)
                    previous_probabilities = phase_exp(pending, 16, 24)
                    partial_sum = Vec.from_elements(
                        previous_probabilities[:first_softmax_count], fx.Float32
                    ).reduce("add", fastmath="reassoc")
                    sum_raw = partial_sum.ir_value()
                    partial_sum = _f32(
                        llvm.inline_asm(
                            sum_raw.type, [sum_raw], "", "=v,0", has_side_effects=True
                        )
                    )
                    probabilities = []
                    for part in fx.range_constexpr(first_softmax_count // 8):
                        values = Vec.from_elements(
                            previous_probabilities[part * 8 : part * 8 + 8], fx.Float32
                        ).to(element_type)
                        probabilities.append(pin_vector(values))
                    remaining = previous_probabilities[first_softmax_count:32]
                    for step in fx.range_constexpr(qk_reduction_steps):
                        fx.rocdl.sched_group_barrier(0x08, 1, 1)
                        if const_expr(step < 4):
                            fx.rocdl.sched_group_barrier(0x400, 2, 1)
                        else:
                            fx.rocdl.sched_group_barrier(
                                0x02, first_softmax_count // 8 + 1, 1
                            )
                    phase_fence()

                    key_packs = load_phase_key(1, stage)
                    if const_expr(not stagger):
                        stage_value(kv_chunk * fx.Int32(kv_tile_rows), stage)
                    fx.rocdl.s_waitcnt(
                        vmcnt=key_load_iterations + value_load_iterations,
                        lgkmcnt=0,
                        expcnt=0,
                    )
                    phase_fence()

                    score_hi = phase_qk(key_packs)
                    final_probabilities = remaining[: 24 - first_softmax_count] + [
                        _exp2(_f32(remaining[24 - first_softmax_count + i]))
                        for i in fx.range_constexpr(8)
                    ]
                    final_sum = Vec.from_elements(
                        final_probabilities, fx.Float32
                    ).reduce("add", fastmath="reassoc")
                    local_sum = partial_sum + final_sum
                    probability_sum = reduce_lane_pair(local_sum, False)
                    for part in fx.range_constexpr((32 - first_softmax_count) // 8):
                        probabilities.append(
                            pin_vector(
                                Vec.from_elements(
                                    final_probabilities[part * 8 : part * 8 + 8],
                                    fx.Float32,
                                ).to(element_type)
                            )
                        )
                    tile_sum = tile_sum + probability_sum
                    for step in fx.range_constexpr(qk_reduction_steps):
                        fx.rocdl.sched_group_barrier(0x08, 1, 2)
                        if const_expr(step < 4):
                            fx.rocdl.sched_group_barrier(0x400, 2, 2)
                        else:
                            fx.rocdl.sched_group_barrier(
                                0x02, (32 - first_softmax_count) // 4 + 1, 2
                            )
                    phase_fence()

                if const_expr(qk_head_dim <= v_head_dim):
                    value_packs = load_phase_value(0, 1 - stage) + load_phase_value(
                        1, 1 - stage
                    )
                    stage_key(prefetch_chunk * fx.Int32(kv_tile_rows), stage)
                    fx.rocdl.s_waitcnt(
                        vmcnt=key_load_iterations + value_load_iterations,
                        lgkmcnt=0,
                        expcnt=0,
                    )
                    phase_fence()

                    probability_fragment = fx.make_fragment_like(
                        probability_coordinates, element_type
                    )
                    probability_fragment.store(Vec(probabilities[0]))
                    for d in fx.range_constexpr(output_chunks):
                        tile_output[d] = mfma(
                            value_packs[d * 4], probability_fragment, tile_output[d]
                        )
                    exact_maximum, rescale, new_maximum, leading = update_maximum(
                        score_lo, score_hi, tile_maximum, exact_maximum
                    )
                    for step in fx.range_constexpr(output_chunks):
                        fx.rocdl.sched_group_barrier(0x08, 1, 3)
                        fx.rocdl.sched_group_barrier(0x02, 6, 3)
                    fx.rocdl.sched_barrier(0)

                    for pack in fx.range_constexpr(1, 4):
                        probability_fragment = fx.make_fragment_like(
                            probability_coordinates, element_type
                        )
                        probability_fragment.store(Vec(probabilities[pack]))
                        for d in fx.range_constexpr(output_chunks):
                            tile_output[d] = mfma(
                                value_packs[d * 4 + pack],
                                probability_fragment,
                                tile_output[d],
                            )
                    shifted = finish_shift(score_lo, score_hi, new_maximum, leading)
                    for step in fx.range_constexpr(3 * output_chunks):
                        fx.rocdl.sched_group_barrier(0x08, 1, 4)
                        if const_expr(step < 8):
                            fx.rocdl.sched_group_barrier(0x02, 3, 4)
                        if const_expr(step >= 4):
                            fx.rocdl.sched_group_barrier(0x400, 2, 4)
                else:
                    value_packs = load_phase_value(0, 1 - stage)
                    if const_expr(stagger):
                        stage_key(
                            prefetch_chunk * fx.Int32(kv_tile_rows),
                            stage,
                            0,
                            key_load_iterations // 3,
                        )
                    else:
                        stage_key(prefetch_chunk * fx.Int32(kv_tile_rows), stage)
                    fx.rocdl.s_waitcnt(vmcnt=63, lgkmcnt=0, expcnt=0)
                    phase_fence()

                    tile_output = phase_pv(value_packs, probabilities, tile_output, 0)
                    exact_maximum, rescale, new_maximum, leading = update_maximum(
                        score_lo, score_hi, tile_maximum, exact_maximum
                    )
                    for step in fx.range_constexpr(8):
                        fx.rocdl.sched_group_barrier(0x08, 1, 3)
                        fx.rocdl.sched_group_barrier(0x02, 4, 3)
                    phase_fence()

                    value_packs = load_phase_value(1, 1 - stage)
                    if const_expr(stagger):
                        stage_key(
                            prefetch_chunk * fx.Int32(kv_tile_rows),
                            stage,
                            key_load_iterations // 3,
                            key_load_iterations,
                        )
                    fx.rocdl.s_waitcnt(
                        vmcnt=key_load_iterations + value_load_iterations,
                        lgkmcnt=0,
                        expcnt=0,
                    )
                    phase_fence()

                    tile_output = phase_pv(value_packs, probabilities, tile_output, 1)
                    shifted = finish_shift(score_lo, score_hi, new_maximum, leading)
                    for step in fx.range_constexpr(8):
                        fx.rocdl.sched_group_barrier(0x08, 1, 4)
                        fx.rocdl.sched_group_barrier(0x02, 3, 4)
                        fx.rocdl.sched_group_barrier(0x400, 2, 4)
                tile_output = [pin_output_chunks(out) for out in tile_output]
                if rescale:
                    correction = _exp2(tile_maximum - new_maximum)
                    correction_vec = Vec.from_elements(
                        [correction], fx.Float32
                    ).broadcast_to(16)
                    tile_output = [Vec(out) * correction_vec for out in tile_output]
                    tile_sum = tile_sum * correction
                tile_output = [pin_output_chunks(out) for out in tile_output]
                phase_fence()
                return shifted, tile_output, new_maximum, tile_sum, exact_maximum

            def process_staged_full(block_count, run_state):
                results = run_state
                if block_count > fx.Int32(0):
                    index_cache = paired_full_cache
                    first_block = read_pair_cache(index_cache, fx.Int32(0))
                    first_chunk = first_block * fx.Int32(2)
                    stage_key(first_chunk * fx.Int32(kv_tile_rows), 0)
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    phase_fence()
                    stage_key((first_chunk + fx.Int32(1)) * fx.Int32(kv_tile_rows), 1)
                    stage_value(first_chunk * fx.Int32(kv_tile_rows), 0)
                    if const_expr(stagger):
                        phase_fence()
                    key_packs = load_phase_key(0, 0)
                    fx.rocdl.s_waitcnt(vmcnt=63, lgkmcnt=0, expcnt=0)
                    phase_fence()
                    score_lo = phase_qk(key_packs)
                    phase_fence()
                    key_packs = load_phase_key(1, 0)
                    # Waves 4-7 run a barrier behind; waves 0-3 read K1 after the next one.
                    fx.rocdl.s_waitcnt(vmcnt=value_load_iterations, lgkmcnt=0, expcnt=0)
                    phase_fence()
                    score_hi = phase_qk(key_packs)
                    tile_maximum = phase_max(score_lo, score_hi)
                    exact_maximum = _maximum(
                        _f32(run_state[2 + output_chunks]), tile_maximum
                    )
                    correction = _exp2(_f32(run_state[0]) - tile_maximum)
                    correction_vec = Vec.from_elements(
                        [correction], fx.Float32
                    ).broadcast_to(16)
                    tile_output = [
                        Vec(run_state[2 + d]) * correction_vec
                        for d in fx.range_constexpr(output_chunks)
                    ]
                    tile_sum = _f32(run_state[1]) * correction
                    pending = phase_exp(
                        phase_shift(score_lo, score_hi, tile_maximum), 0, 16
                    )
                    fx.rocdl.s_waitcnt(vmcnt=value_load_iterations, lgkmcnt=0, expcnt=0)
                    phase_fence()
                    next_slot = (block_count > fx.Int32(1)).select(
                        fx.Int32(1), fx.Int32(0)
                    )
                    next_block = read_pair_cache(index_cache, next_slot)
                    stage_key(next_block * fx.Int32(sparse_kv_block_size), 0)
                    state = [
                        first_block,
                        next_block,
                        tile_maximum,
                        tile_sum,
                        exact_maximum,
                        *tile_output,
                        *pending,
                    ]
                    pipeline_results = state
                    for block_index, args in range(
                        fx.Int32(0), block_count, fx.Int32(1), init=state
                    ):
                        current_block = fx.Int32(args[0])
                        maximum = _f32(args[2])
                        total = _f32(args[3])
                        exact = _f32(args[4])
                        outputs = [
                            args[5 + d] for d in fx.range_constexpr(output_chunks)
                        ]
                        shifted = [
                            _f32(args[5 + output_chunks + i])
                            for i in fx.range_constexpr(32)
                        ]
                        next_index = fx.Int32(block_index) + fx.Int32(1)
                        following_block = fx.Int32(args[1])
                        lookahead_block = following_block
                        shifted, outputs, maximum, total, exact = staged_phase(
                            current_block * fx.Int32(2) + fx.Int32(1),
                            following_block * fx.Int32(2) + fx.Int32(1),
                            1,
                            shifted,
                            outputs,
                            maximum,
                            total,
                            exact,
                        )
                        if next_index < block_count:
                            lookahead = next_index + fx.Int32(1)
                            lookahead_safe = (lookahead < block_count).select(
                                lookahead, block_count - fx.Int32(1)
                            )
                            lookahead_block = read_pair_cache(
                                index_cache, lookahead_safe
                            )
                            shifted, outputs, maximum, total, exact = staged_phase(
                                following_block * fx.Int32(2),
                                lookahead_block * fx.Int32(2),
                                0,
                                shifted,
                                outputs,
                                maximum,
                                total,
                                exact,
                            )
                        pipeline_results = yield [
                            following_block,
                            lookahead_block,
                            maximum,
                            total,
                            exact,
                            *outputs,
                            *shifted,
                        ]
                    fx.rocdl.s_waitcnt(vmcnt=key_load_iterations, lgkmcnt=0, expcnt=0)
                    phase_fence()
                    # Waves 4-7 run a barrier behind: V1 is readable only after this one.
                    phase_fence()
                    maximum = _f32(pipeline_results[2])
                    total = _f32(pipeline_results[3])
                    exact = _f32(pipeline_results[4])
                    outputs = [
                        pipeline_results[5 + d]
                        for d in fx.range_constexpr(output_chunks)
                    ]
                    shifted = [
                        _f32(pipeline_results[5 + output_chunks + i])
                        for i in fx.range_constexpr(32)
                    ]
                    probabilities, probability_sum = phase_pack_sum(
                        phase_exp(shifted, 16, 32)
                    )
                    total = total + probability_sum
                    for half in fx.range_constexpr(2):
                        value_packs = load_phase_value(half, 1)
                        fx.rocdl.s_waitcnt(vmcnt=63, lgkmcnt=0, expcnt=0)
                        phase_fence()
                        outputs = phase_pv(value_packs, probabilities, outputs, half)
                        phase_fence()
                    if const_expr(not stagger):
                        phase_fence()
                    results = [maximum, total, *outputs, exact]
                return results

            def store_results(final_results, lse, max_scores):
                final_max = _f32(final_results[0])
                final_sum = _f32(final_results[1])
                final_output = [
                    final_results[2 + d_chunk]
                    for d_chunk in fx.range_constexpr(output_chunks)
                ]

                inverse_sum = (final_sum > _f32(0.0)).select(
                    _f32(fx.rocdl.rcp(fx.Float32.ir_type, final_sum.ir_value())),
                    _f32(0.0),
                )
                inverse_vec = Vec.from_elements([inverse_sum], fx.Float32).broadcast_to(
                    16
                )

                if const_expr(decode):
                    local_head = query_head - kv_head * fx.Int32(
                        query_heads_per_kv_head
                    )
                    output_source = fx.slice(
                        output_view,
                        (local_head, query_pos, None),
                    )
                else:
                    output_source = fx.slice(output_view, (query_pos, None))
                output_row = fx.logical_divide(
                    output_source,
                    fx.make_layout(8, 1),
                )
                store_valid = query_valid
                if const_expr(q_ragged):
                    store_valid = store_valid & (query_pos < fx.Int32(seq_q))
                if const_expr(split_kv):
                    store_valid = store_valid & (wave == fx.Int32(0))
                chunk_columns = None
                if const_expr(value_shards > 1):
                    shard_column = value_shard * fx.Int32(
                        shard_stage_chunks * mfma_tile_size
                    )
                    chunk_columns = [
                        shard_column
                        + fx.Int32(
                            chunk // shard_stage_chunks * value_stage_columns
                            + chunk % shard_stage_chunks * mfma_tile_size
                        )
                        for chunk in fx.range_constexpr(output_chunks)
                    ]
                if const_expr(kv_splits > 1):
                    if store_valid:
                        _store_partial_output_fragments(
                            final_output,
                            inverse_vec,
                            output_source,
                            accumulator_coordinates,
                            output_chunks,
                            mfma_tile_size,
                        )
                elif store_valid:
                    _store_output_fragments(
                        final_output,
                        inverse_vec,
                        output_row,
                        lane_half,
                        output_copy,
                        output_chunks,
                        mfma_tile_size,
                        element_type,
                        chunk_columns,
                        output_columns,
                    )

                stats_valid = store_valid & (lane_half == fx.Int32(0))
                if const_expr(value_shards > 1):
                    stats_valid = stats_valid & (value_shard == fx.Int32(0))
                if stats_valid:
                    has_values = final_sum > _f32(0.0)
                    lse_value = final_max + fx.math.log2(final_sum)
                    max_value = (
                        _f32(final_results[2 + output_chunks])
                        if const_expr(pipelined_kv)
                        else final_max
                    )
                    if const_expr(not output_stats_in_log2):
                        lse_value = lse_value * _f32(_LN2)
                        max_value = max_value * _f32(_LN2)
                    lse_value = has_values.select(
                        lse_value,
                        _f32(float("-inf")),
                    )
                    max_value = has_values.select(
                        max_value,
                        _f32(float("-inf")),
                    )
                    stats_offset = (
                        output_batch * fx.Int32(num_q_heads) + query_head
                    ) * fx.Int32(seq_q) + query_pos
                    lse[stats_offset] = lse_value
                    if const_expr(write_max_scores):
                        max_scores[stats_offset] = max_value

            if const_expr(paired):
                initial_state = [running_max, running_sum] + output_accumulators

                full_results = process_staged_full(
                    full_count, [*initial_state, running_max]
                )
                # Build owner membership after the full-block pipeline to limit liveness.
                owner_row = mask_row + fx.Int32(1 if stagger else 0)
                owner_full_count = load_uniform_i32(full_kv_num_blocks_view, owner_row)
                owner_partial_count = load_uniform_i32(kv_num_blocks_view, owner_row)
                owner_full_words = []
                owner_partial_words = []
                for word in fx.range_constexpr(len(partial_words)):
                    owner_full_words.append(
                        pair_word(
                            full_kv_indices_view,
                            owner_row,
                            owner_full_count,
                            max_full_blocks,
                            word,
                        )
                    )
                    owner_partial_words.append(
                        pair_word(
                            kv_indices_view,
                            owner_row,
                            owner_partial_count,
                            max_partial_blocks,
                            word,
                        )
                    )
                partial_index_cache = pair_cache(partial_words)
                final_results = process_pipelined_run(
                    partial_count,
                    kv_indices_view,
                    partial_base,
                    True,
                    full_results,
                    partial_index_cache,
                )
                store_results(final_results, logsumexp_view, max_scores_view)
            else:
                full_count = load_uniform_i32(full_kv_num_blocks_view, mask_row)
                partial_count = load_uniform_i32(kv_num_blocks_view, mask_row)
                full_base = mask_row * fx.Int32(max_full_blocks)
                partial_base = mask_row * fx.Int32(max_partial_blocks)
                if const_expr(kv_splits > 1):

                    def split_range(count, base):
                        chunk = (count + fx.Int32(kv_splits - 1)) // fx.Int32(kv_splits)
                        first = kv_split * chunk
                        first = (first < count).select(first, count)
                        rest = count - first
                        return (chunk < rest).select(chunk, rest), base + first

                    full_count, full_base = split_range(full_count, full_base)
                    partial_count, partial_base = split_range(
                        partial_count, partial_base
                    )
                initial_state = [running_max, running_sum] + output_accumulators

                if const_expr(split_kv):
                    full_results = process_split_run(
                        full_count,
                        full_kv_indices_view,
                        full_base,
                        False,
                        initial_state,
                    )
                    split_results = process_split_run(
                        partial_count,
                        kv_indices_view,
                        partial_base,
                        True,
                        full_results,
                    )
                    final_results = reduce_split_results(split_results)
                elif const_expr(single_pass):
                    full_results = process_single_pass_run(
                        full_count,
                        full_kv_indices_view,
                        full_base,
                        False,
                        [*initial_state, running_max],
                    )
                    final_results = process_single_pass_run(
                        partial_count,
                        kv_indices_view,
                        partial_base,
                        True,
                        full_results,
                    )
                elif const_expr(pipelined_kv):
                    full_results = process_pipelined_run(
                        full_count,
                        full_kv_indices_view,
                        full_base,
                        False,
                        [*initial_state, running_max],
                    )
                    final_results = process_pipelined_run(
                        partial_count,
                        kv_indices_view,
                        partial_base,
                        True,
                        full_results,
                    )
                else:
                    full_results = process_sparse_run(
                        full_count,
                        full_kv_indices_view,
                        full_base,
                        False,
                        initial_state,
                    )
                    running_max = _f32(full_results[0])
                    running_sum = _f32(full_results[1])
                    output_accumulators = [
                        full_results[2 + d_chunk]
                        for d_chunk in fx.range_constexpr(output_chunks)
                    ]
                    partial_state = [running_max, running_sum] + output_accumulators
                    final_results = process_sparse_run(
                        partial_count,
                        kv_indices_view,
                        partial_base,
                        True,
                        partial_state,
                    )
                store_results(final_results, logsumexp_view, max_scores_view)

        if const_expr(paired):
            fx.rocdl.sched_barrier(0)
            if wave >= fx.Int32(owner_waves // 2):
                run_body(True)
            fx.rocdl.sched_barrier(0)
            if wave < fx.Int32(owner_waves // 2):
                run_body(False)
        else:
            run_body(False)

    return _make_forward_launch(
        kernel,
        grid=(
            launch_grid(batch_size, seq_q)
            if None not in (batch_size, seq_q)
            else lambda query: launch_grid(
                runtime_extent(batch_size, query, 0), runtime_extent(seq_q, query, 2)
            )
        ),
        num_threads=num_threads,
        mask_buffer_count=(
            capture_slots
            if capture_table
            else mask_buffer_count + score_mod_buffer_count
        ),
        waves_per_eu=waves_per_eu,
        mod_key=mod_key,
    )
