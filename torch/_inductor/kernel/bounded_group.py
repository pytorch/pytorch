# mypy: allow-untyped-defs
from __future__ import annotations

import torch

from .. import ir
from ..lowering import expand, iota, make_pointwise, sum_, to_dtype, view
from ..virtualized import ops, V


eq = make_pointwise(ops.eq, override_return_dtype=torch.bool)
le = make_pointwise(ops.le, override_return_dtype=torch.bool)
lt = make_pointwise(ops.lt, override_return_dtype=torch.bool)


def _key_matrix(keys, bound):
    """Broadcast keys [N] against arange(bound) [E] as a pair of [E, N] views."""
    (routes,) = keys.get_size()
    shape = [bound, routes]
    groups = iota(
        bound,
        start=0,
        step=1,
        dtype=keys.get_dtype(),
        device=keys.get_device(),
        requires_grad=False,
    )
    return expand(view(keys, [1, routes]), shape), expand(
        view(groups, [bound, 1]), shape
    )


def bounded_group_offsets(keys, bound, dtype):
    """Inclusive cumsum of the histogram of keys over the bins [0, bound)."""
    route_keys, groups = _key_matrix(keys, bound)
    return sum_(le(route_keys, groups), axis=1, dtype=dtype)


def bounded_group(keys, bound):
    """Counting sort of integer keys in [0, bound): (sorted keys, permutation).

    Row e of the [bound, N] match matrix scans its matches to rank each key
    within group e; base offsets come from counting the smaller keys.
    """
    route_keys, groups = _key_matrix(keys, bound)
    matches = eq(route_keys, groups)
    bases = sum_(lt(route_keys, groups), axis=1, dtype=torch.int32)
    key_dtype = keys.get_dtype()
    sorted_keys = _scan_scatter(
        matches, bases, key_dtype, lambda idx: ops.index_expr(idx[0], key_dtype)
    )
    permutation = _scan_scatter(
        matches, bases, torch.int64, lambda idx: ops.index_expr(idx[1], torch.int64)
    )
    return sorted_keys, permutation


def _scan_scatter(matches, bases, dtype, value_fn):
    """Scatter value_fn(idx) of each match to bases[group] + rank within the group."""
    device = matches.get_device()
    groups, routes = matches.get_size()
    matches_loader = matches.make_loader()
    int_matches_loader = to_dtype(matches, torch.int32).make_loader()
    bases_loader = bases.make_loader()

    def combine_fn(a_tuple, b_tuple):
        (a,) = a_tuple
        (b,) = b_tuple
        return (ops.add(a, b),)

    def output_indexer(idx, result):
        offset = ops.add(
            bases_loader([idx[0]]), ops.sub(result[0], ops.constant(1, torch.int32))
        )
        offset = ops.where(matches_loader(idx), offset, ops.constant(0, torch.int32))
        return [ops.indirect_indexing(offset, routes, check=False, wrap_neg=False)]

    scan = ir.ScanScatter(
        device=device,
        dtype=dtype,
        inner_fn=int_matches_loader,
        ranges=[groups],
        scan_ranges=[routes],
        size=[groups, routes],
        combine_fn=combine_fn,
        reindex=lambda index, scan_index: [index[0], scan_index[0]],
        reduction_hint=ir.ReductionHint.DEFAULT,
        output_index=0,
        dtypes=(torch.int32,),
        inner_fns=(int_matches_loader,),
        output_indexer=output_indexer,
        output_value=lambda idx, result: value_fn(idx),
        store_mask=lambda idx, result: matches_loader(idx),
    )
    buffer = ir.ComputedBuffer(
        name=None,
        layout=ir.FixedLayout(device, dtype, [routes], [1]),
        data=scan,
    )
    buffer.name = V.graph.register_buffer(buffer)
    V.graph.register_operation(buffer)
    return ir.TensorBox.create(buffer)
