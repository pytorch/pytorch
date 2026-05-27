"""Batch independent same-shape mm ops into _foreach_mm.

Used by the post-scheduling foreach_mm pass to reduce kernel launch count
and enable cascading pointwise fusion in the optimizer region.
"""

import logging
import operator
from collections import defaultdict

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)

aten = torch.ops.aten

_MIN_GROUP_SIZE = 3


def _get_mm_key(node: fx.Node) -> tuple | None:
    if node.op != "call_function" or node.target is not aten.mm.default:
        return None

    args = node.args
    if len(args) != 2:
        return None

    lhs, rhs = args
    if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
        return None

    lhs_val = lhs.meta.get("val")
    rhs_val = rhs.meta.get("val")
    if not isinstance(lhs_val, torch.Tensor) or not isinstance(rhs_val, torch.Tensor):
        return None

    if any(isinstance(s, torch.SymInt) for s in lhs_val.shape) or any(
        isinstance(s, torch.SymInt) for s in rhs_val.shape
    ):
        return None

    return (
        aten.mm.default,
        tuple(lhs_val.shape),
        tuple(rhs_val.shape),
        lhs_val.dtype,
        str(lhs_val.device),
    )


def _are_independent(nodes: list[fx.Node]) -> list[fx.Node]:
    """Return the largest independent subset using bidirectional flood fill.

    For a foreach op, ALL input tensors go into a single call. So no input
    tensor can transitively depend on any other input tensor.
    """
    if len(nodes) <= 1:
        return nodes

    selected: list[fx.Node] = []
    selected_forward: OrderedSet[fx.Node] = OrderedSet()
    selected_backward: OrderedSet[fx.Node] = OrderedSet()

    def _flood(start: fx.Node, visited: OrderedSet[fx.Node], forward: bool) -> None:
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(cur.users.keys() if forward else cur.all_input_nodes)

    for node in nodes:
        if node in selected_forward or node in selected_backward:
            continue
        all_inputs = OrderedSet(node.all_input_nodes)
        if all_inputs & selected_forward or all_inputs & selected_backward:
            continue

        selected.append(node)
        _flood(node, selected_forward, forward=True)
        _flood(node, selected_backward, forward=False)
        for inp in node.all_input_nodes:
            _flood(inp, selected_forward, forward=True)
            _flood(inp, selected_backward, forward=False)

    return selected


def _fuse_mm_group(graph: fx.Graph, nodes: list[fx.Node]) -> None:
    lhs_list = [n.args[0] for n in nodes]
    rhs_list = [n.args[1] for n in nodes]

    with graph.inserting_before(nodes[-1]):
        result = graph.call_function(
            aten._foreach_mm.default, args=(lhs_list, rhs_list)
        )
        result.meta["val"] = [
            aten.mm(lhs.meta["val"], rhs.meta["val"])  # type: ignore[union-attr]
            for lhs, rhs in zip(lhs_list, rhs_list)
        ]

        for i, node in enumerate(nodes):
            with graph.inserting_after(result):
                getitem = graph.call_function(operator.getitem, args=(result, i))
                getitem.meta.update(node.meta)
            node.replace_all_uses_with(getitem)

    for node in nodes:
        graph.erase_node(node)

    log.info("foreach_mm: batched %d mm ops", len(nodes))


def foreach_mm_pass(gm: torch.fx.GraphModule) -> bool:
    """Batch independent same-shape mm calls into _foreach_mm."""
    from .. import config

    min_size = getattr(
        config.aten_distributed_optimizations,
        "foreach_mm_min_group_size",
        _MIN_GROUP_SIZE,
    )

    graph = gm.graph
    changed = False

    mm_groups: dict[tuple, list[fx.Node]] = defaultdict(list)
    for node in graph.nodes:
        key = _get_mm_key(node)
        if key is not None:
            mm_groups[key].append(node)

    for key, nodes in mm_groups.items():
        if len(nodes) < min_size:
            continue
        batch = _are_independent(nodes)
        if len(batch) < min_size:
            continue
        _fuse_mm_group(graph, batch)
        changed = True

    if changed:
        from torch._inductor.pattern_matcher import stable_topological_sort

        stable_topological_sort(graph)
        graph.eliminate_dead_code()
        graph.lint()

    return changed
