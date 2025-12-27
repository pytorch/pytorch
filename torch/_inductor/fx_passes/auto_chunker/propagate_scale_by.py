import functools
import logging
from collections.abc import Callable, Sequence
from typing import TypeAlias

import torch
from torch.fx import Node

from .common import CantChunk
from .core import get_chunking_meta, get_chunking_metas, log, update_chunking_meta
from .utils import (
    format_node_with_chunking_meta,
    get_args_of_node_type,
    get_scale_by_from_metas,
    get_scale_by_from_node,
)


aten = torch.ops.aten
prims = torch.ops.prims

_HandlerType: TypeAlias = Callable[[Node], bool]
propagate_rules: dict[torch._ops.OpOverload, _HandlerType] = {}


def _register_propagate_rule(
    aten_op: torch._ops.OpOverload | Sequence[torch._ops.OpOverload],
    handler: _HandlerType,
) -> _HandlerType:
    if not isinstance(aten_op, (list, tuple)):
        aten_op = [aten_op]  # type: ignore[assignment, list-item]

    assert isinstance(aten_op, (list, tuple)), f"{type(aten_op)=}"
    for op in aten_op:
        assert isinstance(op, torch._ops.OpOverload)
        propagate_rules[op] = handler
    return handler


def register_propagate_rule(
    aten_op: torch._ops.OpOverload | Sequence[torch._ops.OpOverload],
) -> Callable[[_HandlerType], _HandlerType]:
    return functools.partial(_register_propagate_rule, aten_op)


def propagate_scale_by(nodes_with_chunking_meta: Sequence[Node]) -> None:
    """
    The input is a list of nodes that have chunking metadata.
    The nodes are already sorted in topological order.
    """
    for node in nodes_with_chunking_meta:
        arg_nodes = get_args_of_node_type(node)
        arg_metas = get_chunking_metas(arg_nodes)

        if all(arg_meta is None for arg_meta in arg_metas):
            # should be graph input of the chunking subgraph
            continue

        if log.isEnabledFor(logging.DEBUG):
            print("Propagate scale_by:")
            format_node_with_chunking_meta(node, True)

        assert all(arg_meta is not None for arg_meta in arg_metas), node.format_node()

        # None of the input has scale_by set
        if all(arg_meta.scale_by is None for arg_meta in arg_metas):  # type: ignore[union-attr]
            continue

        target = node.target
        if (
            not isinstance(target, torch._ops.OpOverload)
            or target not in propagate_rules
        ):
            raise CantChunk(
                f"Missing scale_by propagation rule for target {target}: {node.format_node()}"
            )

        if not propagate_rules[target](node):
            raise CantChunk(
                f"scale_by propagate rule for {target} fail: {node.format_node()}"
            )


@register_propagate_rule(
    [
        aten.div.Tensor,
    ]
)
def propagate_div(div_node: Node) -> bool:
    lhs_node, rhs_node = div_node.args[:2]
    assert isinstance(lhs_node, Node)
    lhs_scale_by = get_scale_by_from_node(lhs_node)

    # When gradient accumulation is enabled, rhs_node can be a constant
    # representing the gradient accumulation steps
    rhs_scale_by = (
        get_scale_by_from_node(rhs_node) if isinstance(rhs_node, Node) else None
    )
    if lhs_scale_by and rhs_scale_by is None:
        update_chunking_meta(div_node, scale_by=lhs_scale_by)
        return True
    return False


@register_propagate_rule(
    [
        aten.where.self,
    ]
)
def propagate_where(where_node: Node) -> bool:
    cond_node, true_node, false_node = where_node.args
    assert isinstance(cond_node, Node)
    assert isinstance(true_node, Node)
    assert isinstance(false_node, Node)
    cond_meta, true_meta, false_meta = get_chunking_metas(
        [cond_node, true_node, false_node]
    )
    out_meta = get_chunking_meta(where_node)

    assert true_meta is not None
    assert false_meta is not None
    if true_meta.scale_by and not false_meta.scale_by:
        # the false_node must be all zero
        if false_node.target != aten.full.default:
            return False
        if false_node.args[1] != 0.0:
            return False
        assert out_meta is not None
        out_meta.scale_by = true_meta.scale_by
        return True
    return False


@register_propagate_rule(
    [
        aten.mul.Tensor,
        prims.convert_element_type.default,
        aten.sum.dim_IntList,
        aten.sum.default,  # sum to scalar
        aten.mm.default,
        aten.permute.default,
        aten.expand.default,
    ]
)
def propagate_general_copy(out_node: Node) -> bool:
    """
    A rule that holds for multiple ops: the scale_by of the output is
    set to the only scale_by of input nodes or None if no input has scale_by
    set.
    """
    args_node = get_args_of_node_type(out_node)
    args_meta = get_chunking_metas(args_node)
    out_meta = get_chunking_meta(out_node)

    scale_by = get_scale_by_from_metas(*args_meta)  # type: ignore[arg-type]
    assert out_meta is not None
    out_meta.scale_by = scale_by
    return True


@register_propagate_rule(
    [
        aten.add.Tensor,
        aten.sub.Tensor,
    ]
)
def propagate_add_sub(out_node: Node) -> bool:
    """
    The scale_by node of the two arguments must be the same.
    """
    lhs_node, rhs_node = get_args_of_node_type(out_node)
    assert isinstance(lhs_node, Node)
    assert isinstance(rhs_node, Node)
    lhs_meta, rhs_meta = get_chunking_metas([lhs_node, rhs_node])
    assert lhs_meta is not None
    assert rhs_meta is not None
    if lhs_meta.scale_by is rhs_meta.scale_by:
        update_chunking_meta(out_node, scale_by=lhs_meta.scale_by)
        return True
    return False


@register_propagate_rule(
    [
        prims.fma.default,
    ]
)
def propagate_fma(out_node: Node) -> bool:
    mul_lhs, mul_rhs, add_rhs = out_node.args[:3]
    assert isinstance(mul_lhs, Node)
    assert isinstance(mul_rhs, Node)
    assert isinstance(add_rhs, Node)
    mul_lhs_meta, mul_rhs_meta, add_rhs_meta = get_chunking_metas(
        [mul_lhs, mul_rhs, add_rhs]
    )
    assert mul_lhs_meta is not None
    assert mul_rhs_meta is not None
    add_lhs_scale_by = get_scale_by_from_metas(mul_lhs_meta, mul_rhs_meta)
    assert add_rhs_meta is not None
    if add_lhs_scale_by is add_rhs_meta.scale_by:
        update_chunking_meta(out_node, scale_by=add_lhs_scale_by)
        return True
    return False
