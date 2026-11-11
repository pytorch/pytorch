# mypy: allow-untyped-defs

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import Vector as Vec

from .flex_attn_utils import (
    ceil_div,
    ELEMENT_TYPES,
    make_global_view,
    mask_capture_sizes,
)


_THREADS = 256


def build_flex_attn_split_kv_combine_module(
    *,
    kv_splits: int,
    batch_size: int,
    num_heads: int,
    seq_q: int,
    head_dim: int,
    partial_stride,
    stats_stride,
    o_stride,
    lse_stride,
    write_max_scores: bool = True,
    dtype: str = "bf16",
):
    """Return a launcher ``(partials, split_lse, split_max, logsumexp, max_scores,
    output, stream)`` merging split ``s``'s self-normalized fp32 output and log2
    statistics at ``[s, b, h, q]``; a row without keys gets zeros and -inf."""
    sizes = (kv_splits, batch_size, num_heads, seq_q, head_dim)
    if any(int(size) <= 0 for size in sizes):
        raise ValueError("FlyDSL split-KV combine requires positive sizes")
    if dtype not in ELEMENT_TYPES:
        raise ValueError(f"FlyDSL split-KV combine supports bf16/fp16, got {dtype}")
    kv_splits, batch_size, num_heads, seq_q, head_dim = map(int, sizes)
    partial_stride, stats_stride, o_stride, lse_stride = (
        tuple(map(int, stride))
        for stride in (partial_stride, stats_stride, o_stride, lse_stride)
    )
    element_type = ELEMENT_TYPES[dtype]
    rows = (batch_size, num_heads, seq_q)
    spans = mask_capture_sizes(
        ((kv_splits, *rows, head_dim), (kv_splits, *rows), (*rows, head_dim), rows),
        (partial_stride, stats_stride, o_stride, lse_stride),
    )
    if max(spans[0], spans[1], spans[3]) * 4 > 0xFFFFFFFF or spans[2] * 2 > 0xFFFFFFFF:
        raise ValueError("FlyDSL split-KV combine buffers must be below 4 GiB")
    # Four columns per thread when every row of the partials and the output
    # starts on a 16-byte and an 8-byte boundary.
    width = (
        4
        if head_dim % 4 == 0
        and partial_stride[-1] == o_stride[-1] == 1
        and all(stride % 4 == 0 for stride in (*partial_stride[:-1], *o_stride[:-1]))
        else 1
    )
    groups = head_dim // width
    items = batch_size * num_heads * seq_q * groups
    write_max_scores = bool(write_max_scores)

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def kernel(
        partials: fx.Tensor,
        split_lse: fx.Tensor,
        split_max: fx.Tensor,
        logsumexp: fx.Tensor,
        max_scores: fx.Tensor,
        output: fx.Tensor,
    ):
        item = fx.block_idx.x * fx.Int32(_THREADS) + fx.thread_idx.x
        if item < fx.Int32(items):
            column = item % fx.Int32(groups) * fx.Int32(width)
            row = item // fx.Int32(groups)
            query = row % fx.Int32(seq_q)
            head = row // fx.Int32(seq_q) % fx.Int32(num_heads)
            batch = row // fx.Int32(seq_q * num_heads)
            partial_view, lse_view, max_view = (
                make_global_view(tensor, None, span, 1)
                for tensor, span in zip(
                    (partials, split_lse, split_max), (spans[0], spans[1], spans[1])
                )
            )
            partial_copy = fx.make_copy_atom(
                getattr(fx.rocdl, f"BufferCopy{32 * width}b")(), fx.Float32
            )
            partial_source = fx.logical_divide(partial_view, fx.make_layout(width, 1))

            def offset(stride, *coordinates):
                total = fx.Int32(0)
                for coordinate, step in zip(coordinates, stride):
                    total = total + coordinate * fx.Int32(step)
                return total

            stats_base = offset(stats_stride[1:], batch, head, query)
            partial_base = offset(partial_stride[1:], batch, head, query, column)
            negative_infinity = fx.Float32(float("-inf"))
            zero = fx.Float32(0.0)
            initial = [negative_infinity, zero, negative_infinity] + [zero] * width
            results = initial
            for split, state in range(
                fx.Int32(0), fx.Int32(kv_splits), fx.Int32(1), init=initial
            ):
                running_lse = fx.Float32(state[0])
                running_sum = fx.Float32(state[1])
                running_max = fx.Float32(state[2])
                split_stats = stats_base + fx.Int32(split) * fx.Int32(stats_stride[0])
                lse = fx.Float32(lse_view[split_stats])
                merged = running_lse.maximumf(lse)
                reference = (merged == negative_infinity).select(zero, merged)
                old_scale = fx.math.exp2(running_lse - reference)
                new_scale = fx.math.exp2(lse - reference)
                running_sum = running_sum * old_scale + new_scale
                index = partial_base + fx.Int32(split) * fx.Int32(partial_stride[0])
                if const_expr(width == 1):
                    values = [fx.Float32(partial_view[index])]
                else:
                    fragment = fx.make_rmem_tensor(width, fx.Float32)
                    fx.copy(
                        partial_copy,
                        fx.slice(partial_source, (None, index // fx.Int32(width))),
                        fragment,
                    )
                    values = Vec(fragment.load())
                accumulators = []
                for element in fx.range_constexpr(width):
                    contribution = (new_scale > zero).select(
                        new_scale * fx.Float32(values[element]), zero
                    )
                    accumulators.append(
                        fx.Float32(state[3 + element]) * old_scale + contribution
                    )
                if const_expr(write_max_scores):
                    running_max = running_max.maximumf(
                        fx.Float32(max_view[split_stats])
                    )
                results = yield [merged, running_sum, running_max, *accumulators]

            total = fx.Float32(results[1])
            has_keys = total > zero
            inverse = has_keys.select(fx.Float32(1.0) / total, zero)
            merged_lse = fx.Float32(results[0])
            normalized = Vec.from_elements(
                [
                    fx.Float32(results[3 + element]) * inverse
                    for element in fx.range_constexpr(width)
                ],
                fx.Float32,
            ).to(element_type)
            output_view = make_global_view(output, None, spans[2], 1)
            out_index = offset(o_stride, batch, head, query, column)
            if const_expr(width == 1):
                output_view[out_index] = normalized[0]
            else:
                out_fragment = fx.make_rmem_tensor(width, element_type)
                out_fragment.store(normalized)
                fx.copy(
                    fx.make_copy_atom(fx.rocdl.BufferCopy64b(), element_type),
                    out_fragment,
                    fx.slice(
                        fx.logical_divide(output_view, fx.make_layout(width, 1)),
                        (None, out_index // fx.Int32(width)),
                    ),
                )
            if column == fx.Int32(0):
                stats_index = offset(lse_stride, batch, head, query)
                lse_out = make_global_view(logsumexp, None, spans[3], 1)
                lse_out[stats_index] = has_keys.select(
                    merged_lse + fx.math.log2(total), negative_infinity
                )
                if const_expr(write_max_scores):
                    max_out = make_global_view(max_scores, None, spans[3], 1)
                    max_out[stats_index] = fx.Float32(results[2])

    @flyc.jit
    def launch(
        partials: fx.Tensor,
        split_lse: fx.Tensor,
        split_max: fx.Tensor,
        logsumexp: fx.Tensor,
        max_scores: fx.Tensor,
        output: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(partials, split_lse, split_max, logsumexp, max_scores, output).launch(
            grid=(ceil_div(items, _THREADS), 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    return launch
