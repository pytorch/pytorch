# mypy: allow-untyped-defs
from __future__ import annotations

import torch

from .. import ir
from ..lowering import add, expand, iota, make_pointwise, sum_, to_dtype, view
from ..virtualized import ops, V


eq = make_pointwise(ops.eq, override_return_dtype=torch.bool)
lt = make_pointwise(ops.lt, override_return_dtype=torch.bool)


def _scan_scatter(int_matches, matches, bases, output_dtype, output_value):
    device = matches.get_device()
    experts, routes = matches.get_size()
    matches_loader = matches.make_loader()
    int_matches_loader = int_matches.make_loader()
    bases_loader = bases.make_loader()

    def combine_fn(a_tuple, b_tuple):
        (a,) = a_tuple
        (b,) = b_tuple
        return (ops.add(a, b),)

    def reindex(index, scan_index):
        return [index[0], scan_index[0]]

    def output_indexer(idx, result):
        grouped_offset = ops.add(
            bases_loader([idx[0]]),
            ops.sub(result[0], ops.constant(1, torch.int32)),
        )
        safe_grouped_offset = ops.where(
            matches_loader(idx),
            grouped_offset,
            ops.constant(0, torch.int32),
        )
        return [
            ops.indirect_indexing(
                safe_grouped_offset,
                routes,
                check=False,
                wrap_neg=False,
            )
        ]

    scan = ir.ScanScatter(
        device=device,
        dtype=output_dtype,
        inner_fn=int_matches_loader,
        ranges=[experts],
        scan_ranges=[routes],
        size=[experts, routes],
        combine_fn=combine_fn,
        reindex=reindex,
        reduction_hint=ir.ReductionHint.DEFAULT,
        output_index=0,
        dtypes=(torch.int32,),
        inner_fns=(int_matches_loader,),
        output_indexer=output_indexer,
        output_value=output_value,
        store_mask=lambda idx, result: matches_loader(idx),
    )
    buffer = ir.ComputedBuffer(
        name=None,
        layout=ir.FixedLayout(
            device,
            output_dtype,
            [routes],
            [1],
        ),
        data=scan,
    )
    buffer.name = V.graph.register_buffer(buffer)
    V.graph.register_operation(buffer)
    return ir.TensorBox.create(buffer)


def bounded_group(keys, upper_bound):
    size = keys.get_size()
    if len(size) != 1:
        raise AssertionError("bounded_group expects one-dimensional keys")
    routes = int(size[0])
    upper_bound = int(upper_bound)
    device = keys.get_device()
    if device is None:
        raise AssertionError("bounded_group keys must have a device")

    matrix_shape = [upper_bound, routes]
    expert_ids = iota(
        upper_bound,
        start=0,
        step=1,
        dtype=keys.get_dtype(),
        device=device,
        requires_grad=False,
    )
    expert_ids = expand(view(expert_ids, [upper_bound, 1]), matrix_shape)
    route_keys = expand(view(keys, [1, routes]), matrix_shape)
    matches = eq(route_keys, expert_ids)
    int_matches = to_dtype(matches, torch.int32)
    lower_keys = to_dtype(lt(route_keys, expert_ids), torch.int32)
    bases = sum_(lower_keys, axis=1, dtype=torch.int32)
    counts = sum_(int_matches, axis=1, dtype=torch.int32)
    offsets = add(bases, counts)

    sorted_keys = _scan_scatter(
        int_matches,
        matches,
        bases,
        keys.get_dtype(),
        lambda idx, result: ops.index_expr(idx[0], keys.get_dtype()),
    )
    permutation = _scan_scatter(
        int_matches,
        matches,
        bases,
        torch.int64,
        lambda idx, result: ops.index_expr(idx[1], torch.int64),
    )
    return sorted_keys, permutation, offsets
