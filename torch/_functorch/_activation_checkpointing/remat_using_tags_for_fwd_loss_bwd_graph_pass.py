"""
AC rematerialize pass: Duplicates checkpointed nodes for backward, then DCE removes unused forward versions.
"""

import warnings

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
    return node.meta.get("custom", {}).get("remat_pass_tag", None) == "is_backward"


def remat_using_tags_for_fwd_loss_bwd_graph(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Duplicate checkpointed nodes for backward use. DCE removes unused forward versions. We assume that
    you already annotated your backward region with fx.traceback.annotate({"remat_pass_tag": "is_backward"})
    which helps us identify the backward region.
    """
    if not has_recomputable_ops(gm):
        return gm

    # Find backward boundary and build ordering
    bwd_start: int | None = None
    order = {}
    for idx, node in enumerate(gm.graph.nodes):
        order[node] = idx
        if _is_backward_node(node) and bwd_start is None:
            bwd_start = idx

    if bwd_start is None:
        warnings.warn(
            "remat_using_tags_for_fwd_loss_bwd_graph: Graph has recomputable ops but no backward region. "
            "This may indicate a forward-only graph (e.g., from nested compilation) or missing backward annotations. "
            "Returning graph unchanged."
        )
        return gm

    if has_recomputable_rng_ops(gm):
        raise RuntimeError(
            "Activation checkpoint rematerializing in `forward-loss-backward` graph does not support RNG ops "
            "in checkpointed regions. Please move RNG operations outside "
            "of checkpoint regions, or use joint graph mode (where partitioner handles RNG)."
        )

    # Use partitioner pass to normalize AC node tags.
    gm = cleanup_recompute_tags(gm, is_default_partition=True)

    if not config.unsafe_allow_optimization_of_collectives:
        force_save_collectives(gm)

    force_save_bw_mutation_src(gm)

    new_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    recomputed_nodes: dict[fx.Node, fx.Node] = {}

    # Insert forward nodes
    for node in list(gm.graph.nodes)[:bwd_start]:
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    def remat_input(x):
        # fx.Node can have args that are primitive types (e.g. int, float, bool)
        if not isinstance(x, fx.Node):
            return x
        return recomputed_nodes.get(x, env[x])

    def gather_checkpointed_deps(node: fx.Node, visited: set) -> None:
        if node in visited or node in recomputed_nodes:
            return
        visited.add(node)
        for inp in node.all_input_nodes:
            if must_recompute(inp):
                gather_checkpointed_deps(inp, visited)

    # Insert backward nodes
    for node in list(gm.graph.nodes)[bwd_start:]:
        # Gather all checkpointed deps needed by this node
        deps = set()
        for inp in node.all_input_nodes:
            if must_recompute(inp):
                gather_checkpointed_deps(inp, deps)

        # Insert deps in forward order (guaranteed disjoint from already-inserted)
        # This is not as inefficient as it looks, because we only add fresh dependencies
        # when they are not yet processed as recomputed nodes.
        for dep in sorted(deps, key=lambda n: order[n]):
            assert dep not in recomputed_nodes, "We shouldn't have recomputed it before"
            dup = new_graph.node_copy(dep, remat_input)
            dup.name = dep.name + "_recomputed"
            recomputed_nodes[dep] = dup

        env[node] = new_graph.node_copy(node, remat_input)

    new_gm = torch.fx.GraphModule(gm, new_graph)

    # DCE with custom is_impure_node (like default_partition)
    # Treats certain collectives as pure while delegating to default impurity logic
    new_gm.graph.eliminate_dead_code(is_impure_node=is_impure_node_for_dce)

    # raise_getitems pass for better memory (like default_partition)
    new_gm = raise_getitems(new_gm)

    new_gm.recompile()

    return new_gm
