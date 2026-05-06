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

log = logging.getLogger(__name__)

aten = torch.ops.aten

_COMM_OP_NAMES = frozenset([
    "all_reduce", "all_gather", "reduce_scatter", "wait_tensor",
    "all_gather_into_tensor", "reduce_scatter_tensor",
])


def _is_mm_control_deps(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target is not torch.ops.higher_order.control_deps:
        return False
    if len(node.args) < 4:
        return False
    subgraph = node.args[1]
    if not isinstance(subgraph, fx.Node):
        return False
    sg_name = getattr(subgraph, "target", "") or getattr(subgraph, "name", "")
    if not isinstance(sg_name, str):
        sg_name = str(sg_name)
    if not sg_name.startswith("subgraph_mm"):
        return False
    rest = sg_name[len("subgraph_mm"):]
    return rest == "" or rest[0] in ("_", " ")


def _involves_comm(node: fx.Node) -> bool:
    """Check if a node is a comm op or a control_deps wrapping a comm op."""
    name = node.name if hasattr(node, "name") else ""
    for comm in _COMM_OP_NAMES:
        if comm in name:
            return True
    if node.op == "call_function" and hasattr(node.target, "__name__"):
        tname = node.target.__name__
        for comm in _COMM_OP_NAMES:
            if comm in tname:
                return True
    return False


def _sched_deps_involve_comm(node: fx.Node) -> bool:
    """Check if any scheduling dep of a control_deps node is a comm op."""
    sched_tuple = node.args[0]
    if not isinstance(sched_tuple, (tuple, list)):
        return False
    for dep in sched_tuple:
        if isinstance(dep, fx.Node) and _involves_comm(dep):
            return True
    return False


def foreach_mm_post_scheduling_pass(gm: torch.fx.GraphModule) -> bool:
    """Unwrap control_deps(mm) nodes that don't need comm overlap."""
    graph = gm.graph
    changed = False
    unwrapped = 0

    for node in list(graph.nodes):
        if not _is_mm_control_deps(node):
            continue

        # Keep control_deps if scheduling deps involve comm ops
        if _sched_deps_involve_comm(node):
            continue

        # Unwrap: replace control_deps(sched_deps, subgraph, lhs, rhs) → mm(lhs, rhs)
        lhs = node.args[2]
        rhs = node.args[3]

        with graph.inserting_before(node):
            mm_node = graph.call_function(aten.mm.default, args=(lhs, rhs))
            mm_node.meta.update(node.meta)

        node.replace_all_uses_with(mm_node)

        subgraph_node = node.args[1]
        graph.erase_node(node)
        if isinstance(subgraph_node, fx.Node) and len(subgraph_node.users) == 0:
            graph.erase_node(subgraph_node)

        unwrapped += 1
        changed = True

    if changed:
        from torch._inductor.pattern_matcher import stable_topological_sort

        stable_topological_sort(graph)
        graph.eliminate_dead_code()
        log.info(
            "foreach_mm_post_sched: unwrapped %d control_deps(mm) to bare mm",
            unwrapped,
        )

    return changed
