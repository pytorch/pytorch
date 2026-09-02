"""
Prove that the sign of a zero produced by the value output of max.dim/min.dim
can never be observed, so the value can come from a plain amax/amin instead of
the indexed reduction that orders +0.0/-0.0 ties.

Short forward walk from the value with a small allowlist: ops in _PROPAGATES
change at most the sign of a zero in their result, ops in _KILLS give identical
results for either zero, and anything else (graph outputs, mutations, unknown
ops) is an observer. The walk gives up after _MAX_VISITED nodes.
"""

from __future__ import annotations

import operator

import torch
from torch.utils._ordered_set import OrderedSet


aten = torch.ops.aten
prims = torch.ops.prims

SIGNED_ZERO_UNOBSERVABLE = "signed_zero_unobservable"
_MAX_VISITED = 64

_KILLS = OrderedSet(
    [
        aten.exp,
        aten.exp2,
        aten.abs,
        aten.cos,
        aten.cosh,
        aten.log,
        aten.log2,
        aten.log10,
        aten.sigmoid,
        aten.eq,
        aten.ne,
        aten.lt,
        aten.le,
        aten.gt,
        aten.ge,
        aten.isnan,
        aten.isinf,
        aten.isfinite,
    ]
)

_PROPAGATES = OrderedSet(
    [
        aten.add,
        aten.sub,
        aten.mul,
        aten.neg,
        aten.where,
        aten.maximum,
        aten.minimum,
        aten.amax,
        aten.amin,
        aten.max,
        aten.min,
        aten.sum,
        aten.view,
        aten.reshape,
        aten._unsafe_view,
        aten.permute,
        aten.expand,
        aten.slice,
        aten.select,
        aten.squeeze,
        aten.unsqueeze,
        aten.clone,
        aten.cat,
        prims.prepare_softmax_online,
    ]
)

_CASTS = OrderedSet([aten._to_copy, prims.convert_element_type])


def _classify(user: torch.fx.Node, node: torch.fx.Node) -> str:
    if user.op != "call_function":
        return "observe"
    if user.target is operator.getitem:
        # prepare_softmax_online returns (max, sum(exp(x - max))): the sign of
        # a zero survives in the max but not in the sum.
        if node.target is prims.prepare_softmax_online.default and user.args[1] == 1:
            return "kill"
        return "propagate"
    packet = getattr(user.target, "overloadpacket", None)
    if packet in _KILLS:
        return "kill"
    if packet in _CASTS:
        dtype = (
            user.args[1]
            if user.target is prims.convert_element_type.default
            else user.kwargs.get("dtype")
        )
        if dtype is None:
            return "propagate"
        return "propagate" if dtype.is_floating_point else "kill"
    if packet in _PROPAGATES:
        return "propagate"
    return "observe"


def signed_zero_unobservable(value: torch.fx.Node) -> bool:
    worklist = [value]
    seen: OrderedSet[torch.fx.Node] = OrderedSet([value])
    while worklist:
        node = worklist.pop()
        for user in node.users:
            kind = _classify(user, node)
            if kind == "observe":
                return False
            if kind == "propagate" and user not in seen:
                if len(seen) >= _MAX_VISITED:
                    return False
                seen.add(user)
                worklist.append(user)
    return True


def mark_arg_reductions_with_unobservable_signed_zero(graph: torch.fx.Graph) -> None:
    for node in graph.nodes:
        if node.op != "call_function" or node.target not in (
            aten.max.dim,
            aten.min.dim,
        ):
            continue
        values = [
            user
            for user in node.users
            if user.target is operator.getitem and user.args[1] == 0
        ]
        if all(signed_zero_unobservable(v) for v in values):
            node.meta[SIGNED_ZERO_UNOBSERVABLE] = True
