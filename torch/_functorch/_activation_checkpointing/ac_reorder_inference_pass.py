"""
AC reordering pass: Duplicates checkpointed nodes for backward, then DCE removes unused forward versions.
"""

import torch
import torch.fx as fx
from torch._functorch import config
from torch._functorch.compile_utils import raise_getitems
from torch._functorch.partitioners import (
    cleanup_recompute_tags,
    force_save_bw_mutation_src,
    force_save_collectives,
    has_recomputable_ops,
    has_recomputable_rng_ops,
    is_not_collective,
    must_recompute,
)


def is_impure_node_for_dce(node):
    # Check for special collectives that should be treated as pure
    if not is_not_collective(node):
        # It's a collective (wait_tensor, all_gather_into_tensor, etc.)
        # Treat as pure - can be eliminated if unused
        return False

    # For everything else, fall back to the DEFAULT logic
    # This is what eliminate_dead_code() calls when is_impure_node=None
    impure_random = True
    if torch._guards.TracingContext.try_get():
        impure_random = torch._inductor.config.fallback_random
    return node.is_impure(impure_random)


def _is_backward_node(node: fx.Node) -> bool:
    """Check if node is in backward region via annotation"""
    return node.meta.get("custom", {}).get("backward") is not None


def rematerialize_nodes_with_ac_annotations(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Duplicate checkpointed nodes for backward use. DCE removes unused forward versions.
    """
    # Early exit if no checkpointed ops
    if not has_recomputable_ops(gm):
        return gm

    # Ban RNG ops in checkpointed regions (like partitioner does for inference mode)
    # Note: RNG functionalization is not currently supported for inference mode graphs
    if has_recomputable_rng_ops(gm):
        raise RuntimeError(
            "Activation checkpoint reordering in fullgraph does not support RNG ops in checkpointed regions. "
            "RNG state cannot be properly synchronized between "
            "forward and recomputed backward operations. Please move RNG operations outside "
            "of checkpoint regions, or use joint graph mode (where partitioner handles RNG)."
        )

    # Apply partitioner's edge case handling FIRST
    # This marks nodes that must be saved (MUST_SAVE, collectives, mutations, etc.)
    gm = cleanup_recompute_tags(gm, is_default_partition=False)

    if not config.unsafe_allow_optimization_of_collectives:
        force_save_collectives(gm)

    force_save_bw_mutation_src(gm)

    # Find backward boundary and build ordering
    first_node_in_bwd = None
    bwd_start = None
    order = {}
    for idx, node in enumerate(gm.graph.nodes):
        order[node] = idx
        if _is_backward_node(node) and first_node_in_bwd is None:
            first_node_in_bwd = node
            bwd_start = idx

    if first_node_in_bwd is None:
        return gm

    # Build reordered graph
    new_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    recomputed_nodes: dict[fx.Node, fx.Node] = {}

    def remat_input(x):
        # fx.Node can be int or float
        if not isinstance(x, fx.Node):
            return x
        return recomputed_nodes.get(x, env[x])

    # Add placeholders
    for node in gm.graph.find_nodes(op="placeholder"):
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    def gather_checkpointed_deps(node: fx.Node, visited: set) -> None:
        if node in visited or node in recomputed_nodes:
            return
        visited.add(node)
        for inp in node.all_input_nodes:
            if must_recompute(inp):
                gather_checkpointed_deps(inp, visited)

    # Insert forward nodes
    for node in list(gm.graph.nodes)[:bwd_start]:
        if node.op == "placeholder":
            continue
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    # Insert backward nodes
    for node in list(gm.graph.nodes)[bwd_start:]:
        if node.op == "output":
            continue

        # Gather ALL checkpointed deps needed by this node
        deps = set()
        for inp in node.all_input_nodes:
            if must_recompute(inp):
                gather_checkpointed_deps(inp, deps)

        # Insert deps in forward order (guaranteed disjoint from already-inserted)
        for dep in sorted(deps, key=lambda n: order[n]):
            assert dep not in recomputed_nodes, "We shouldn't have recomputed it before"
            dup = new_graph.node_copy(dep, remat_input)
            dup.name = dep.name + "_recomputed"
            recomputed_nodes[dep] = dup

        env[node] = new_graph.node_copy(node, remat_input)

    output_node = list(gm.graph.nodes)[-1]
    assert output_node.op == "output"
    new_graph.node_copy(output_node, remat_input)
    new_gm = torch.fx.GraphModule(gm, new_graph)

    # DCE with custom is_impure_node (like default_partition)
    # Treats certain collectives as pure while delegating to default impurity logic
    new_gm.graph.eliminate_dead_code(is_impure_node=is_impure_node_for_dce)

    # raise_getitems pass for better memory (like default_partition)
    new_gm = raise_getitems(new_gm)

    new_gm.recompile()

    return new_gm
