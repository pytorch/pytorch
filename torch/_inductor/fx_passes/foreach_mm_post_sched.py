"""Unwrap control_deps from optimizer mm ops, then batch via foreach_mm.

After overlap scheduling, mm ops are wrapped in control_deps for ordering.
For optimizer mm ops not involved in comm overlap, the control_deps ordering
is redundant with data flow (NS-ortho chain enforces correct order).

This pass:
1. Identifies control_deps(mm) nodes whose scheduling deps are ONLY other
   compute nodes (not comm ops like all_reduce, all_gather, wait_tensor)
2. Unwraps them to bare mm(lhs, rhs) calls
3. The existing foreach_mm_pass then batches the unwrapped mm ops
"""

import logging

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)

aten = torch.ops.aten

# Comm op targets to check against
_COMM_TARGETS: OrderedSet[torch._ops.OpOverload] = OrderedSet()


def _get_comm_targets() -> OrderedSet[torch._ops.OpOverload]:
    """Lazily build the set of comm op targets."""
    global _COMM_TARGETS
    if _COMM_TARGETS:
        return _COMM_TARGETS
    try:
        funcol = torch.ops._c10d_functional
        _COMM_TARGETS = OrderedSet(
            [
                funcol.all_reduce.default,
                funcol.all_gather_into_tensor.default,
                funcol.reduce_scatter_tensor.default,
                funcol.wait_tensor.default,
            ]
        )
        for name in [
            "all_reduce_",
            "all_gather_into_tensor_out",
            "reduce_scatter_tensor_out",
        ]:
            op = getattr(funcol, name, None)
            if op is not None:
                _COMM_TARGETS.add(op.default)
    except AttributeError:
        pass
    return _COMM_TARGETS


def _get_subgraph_inner_target(
    gm: torch.fx.GraphModule, node: fx.Node
) -> torch._ops.OpOverload | None:
    """Get the call_function target from a control_deps subgraph module."""
    if node.op != "call_function":
        return None
    if node.target is not torch.ops.higher_order.control_deps:
        return None
    if len(node.args) < 2:
        return None
    sg = node.args[1]
    if not isinstance(sg, fx.Node):
        return None
    sg_name = getattr(sg, "target", None)
    if not isinstance(sg_name, str):
        return None
    sg_mod = getattr(gm, sg_name, None)
    if sg_mod is None or not hasattr(sg_mod, "graph"):
        return None
    for n in sg_mod.graph.nodes:
        if n.op == "call_function":
            return n.target
    return None


def _is_mm_control_deps(gm: torch.fx.GraphModule, node: fx.Node) -> bool:
    """Check if node is a control_deps wrapping an mm op."""
    return _get_subgraph_inner_target(gm, node) is aten.mm.default


def _involves_comm(gm: torch.fx.GraphModule, node: fx.Node) -> bool:
    """Check if a node is a comm op by matching its target."""
    comm_targets = _get_comm_targets()
    if node.op == "call_function" and node.target in comm_targets:
        return True
    inner = _get_subgraph_inner_target(gm, node)
    if inner is not None and inner in comm_targets:
        return True
    return False


def _sched_deps_involve_comm(gm: torch.fx.GraphModule, node: fx.Node) -> bool:
    """Check if any scheduling dep of a control_deps node is a comm op."""
    sched_tuple = node.args[0]
    if not isinstance(sched_tuple, (tuple, list)):
        return False
    for dep in sched_tuple:
        if isinstance(dep, fx.Node) and _involves_comm(gm, dep):
            return True
    return False


def foreach_mm_post_scheduling_pass(gm: torch.fx.GraphModule) -> bool:
    """Unwrap control_deps(mm) nodes that don't need comm overlap."""
    graph = gm.graph
    changed = False
    unwrapped = 0

    for node in list(graph.nodes):
        if not _is_mm_control_deps(gm, node):
            continue

        if _sched_deps_involve_comm(gm, node):
            continue

        lhs = node.args[2]
        rhs = node.args[3]

        with graph.inserting_before(node):
            mm_node = graph.call_function(aten.mm.default, args=(lhs, rhs))
            mm_node.meta.update(node.meta)

        node.replace_all_uses_with(mm_node)

        subgraph_node = node.args[1]
        graph.erase_node(node)
        if isinstance(subgraph_node, fx.Node) and len(subgraph_node.users) == 0:
            sg_name = subgraph_node.target
            graph.erase_node(subgraph_node)
            if isinstance(sg_name, str) and hasattr(gm, sg_name):
                delattr(gm, sg_name)

        unwrapped += 1
        changed = True

    if changed:
        from torch._inductor.pattern_matcher import stable_topological_sort

        stable_topological_sort(graph)
        graph.eliminate_dead_code()
        graph.lint()
        log.info(
            "foreach_mm_post_sched: unwrapped %d control_deps(mm) to bare mm",
            unwrapped,
        )

    return changed
