"""
Activation Checkpointing Reordering Pass for Inference Graphs

Minimizes memory by duplicating checkpointed operations for backward use,
then using DCE to remove unused forward versions.

Strategy:
1. Insert all nodes in forward (including checkpointed operations)
2. When backward needs a checkpointed node, duplicate the entire checkpoint region
3. Run DCE to remove forward versions that have no users
"""

import torch
import torch.fx as fx
from torch._functorch.partitioners import (
    collect_deps_with_filter,
    insert_nodes_in_original_order,
    must_recompute,
)


def _is_backward_node(node: fx.Node) -> bool:
    """Check if node is in backward region via annotation"""
    return node.meta.get("custom", {}).get("backward") is not None


def _get_input_for_duplicate_checkpoint(
    x: fx.Node,
    recomputed_nodes: dict[fx.Node, fx.Node],
    env: dict[fx.Node, fx.Node],
) -> fx.Node:
    """Get input when duplicating checkpoint region - reuse checkpoint inputs from forward."""
    if x in recomputed_nodes:
        return recomputed_nodes[x]
    # Placeholders and non-checkpointed nodes can be reused from forward
    return env.get(x, x)


def _duplicate_checkpoint_region(
    node: fx.Node,
    recomputed_nodes: dict[fx.Node, fx.Node],
    env: dict[fx.Node, fx.Node],
    new_graph: fx.Graph,
    checkpointed_in_bwd: set[fx.Node],
) -> None:
    """
    Recursively duplicate a checkpointed node and its checkpointed dependencies.
    Stop at non-checkpointed nodes (checkpoint inputs) - reuse those from forward.

    This mimics partitioner behavior: recompute checkpoint region, reuse checkpoint inputs.
    """
    if node in recomputed_nodes:
        return

    # First, duplicate all checkpointed dependencies (nodes inside checkpoint region)
    for inp in node.all_input_nodes:
        # Skip placeholders (can be reused)
        if inp.op == "placeholder":
            continue
        # Only duplicate if this dependency is also checkpointed
        # (i.e., it's inside the checkpoint region)
        if must_recompute(inp) and inp not in recomputed_nodes:
            _duplicate_checkpoint_region(
                inp, recomputed_nodes, env, new_graph, checkpointed_in_bwd
            )

    # Now duplicate this node
    dup = new_graph.node_copy(
        node,
        lambda x: _get_input_for_duplicate_checkpoint(x, recomputed_nodes, env),
    )
    # Use _recomputed suffix for all duplicated nodes
    dup.name = node.name + "_recomputed"
    recomputed_nodes[node] = dup


def _get_input_for_checkpointed_duplicate(
    x: fx.Node,
    recomputed_nodes: dict[fx.Node, fx.Node],
    env: dict[fx.Node, fx.Node],
    new_graph: fx.Graph,
    checkpointed_in_bwd: set[fx.Node],
) -> fx.Node:
    """Get input when duplicating checkpointed nodes - reuse checkpoint inputs."""
    if x in recomputed_nodes:
        return recomputed_nodes[x]
    # If input is also checkpointed, duplicate it first
    if must_recompute(x) and x not in recomputed_nodes:
        _duplicate_checkpoint_region(
            x, recomputed_nodes, env, new_graph, checkpointed_in_bwd
        )
        return recomputed_nodes[x]
    # Non-checkpointed nodes (checkpoint inputs) - reuse from forward
    return env.get(x, x)


def _get_input_for_backward(
    x: fx.Node,
    recomputed_nodes: dict[fx.Node, fx.Node],
    env: dict[fx.Node, fx.Node],
    new_graph: fx.Graph,
    checkpointed_in_bwd: set[fx.Node],
) -> fx.Node:
    """Get input for regular backward nodes."""
    if x in recomputed_nodes:
        return recomputed_nodes[x]
    # If input is checkpointed and used in backward, duplicate it before using
    # This ensures checkpoint regions are recomputed with their dependencies
    if must_recompute(x):
        _duplicate_checkpoint_region(
            x, recomputed_nodes, env, new_graph, checkpointed_in_bwd
        )
        return recomputed_nodes[x]
    return env.get(x, x)


def _make_node_copy_fn(
    for_backward: bool,
    new_graph: fx.Graph,
    env: dict[fx.Node, fx.Node],
    checkpointed_in_bwd: set[fx.Node],
    recomputed_nodes: dict[fx.Node, fx.Node],
):
    """Create a node copy function for checkpoint-specific logic."""

    def node_copy_fn(n: fx.Node) -> fx.Node:
        # Forward pass: simple copy
        if not for_backward:
            return new_graph.node_copy(n, lambda x: env[x])

        # Backward pass: duplicate checkpointed nodes used in backward
        if must_recompute(n) and n in checkpointed_in_bwd:
            # Duplicate this checkpointed node and its checkpointed dependencies
            dup = new_graph.node_copy(
                n,
                lambda x: _get_input_for_checkpointed_duplicate(
                    x, recomputed_nodes, env, new_graph, checkpointed_in_bwd
                ),
            )
            dup.name = n.name + "_recomputed"
            recomputed_nodes[n] = dup
            env[n] = dup
            return dup

        # Backward pass: regular node
        return new_graph.node_copy(
            n,
            lambda x: _get_input_for_backward(
                x, recomputed_nodes, env, new_graph, checkpointed_in_bwd
            ),
        )

    return node_copy_fn


def _make_skip_condition(
    for_backward: bool,
    recomputed_nodes: dict[fx.Node, fx.Node],
    env: dict[fx.Node, fx.Node],
    checkpointed_in_bwd: set[fx.Node],
):
    """Create a skip condition function for dependency collection."""

    def skip_condition(n: fx.Node) -> bool:
        # Skip if already processed
        if n in recomputed_nodes:
            return True
        if n in env:
            # In backward, we might need to duplicate checkpointed nodes even if they're in env
            if for_backward and must_recompute(n) and n in checkpointed_in_bwd:
                return False
            return True
        return False

    return skip_condition


def _insert_node_with_deps(
    node: fx.Node,
    for_backward: bool,
    order: dict[fx.Node, int],
    recomputed_nodes: dict[fx.Node, fx.Node],
    env: dict[fx.Node, fx.Node],
    checkpointed_in_bwd: set[fx.Node],
    new_graph: fx.Graph,
) -> None:
    """Insert node and dependencies"""

    # Create skip condition
    skip_condition = _make_skip_condition(
        for_backward, recomputed_nodes, env, checkpointed_in_bwd
    )

    # Collect dependencies
    insertable_nodes = collect_deps_with_filter(node, order, skip_condition)

    # Create node copy function
    node_copy_fn = _make_node_copy_fn(
        for_backward, new_graph, env, checkpointed_in_bwd, recomputed_nodes
    )

    # Insert nodes in original order using shared utility
    insert_nodes_in_original_order(
        insertable_nodes, order, new_graph, env, node_copy_fn
    )


def _get_output_arg(
    x,
    recomputed_nodes: dict[fx.Node, fx.Node],
    env: dict[fx.Node, fx.Node],
):
    """Get output argument, preferring recomputed nodes over env."""
    if isinstance(x, fx.Node):
        if x in recomputed_nodes:
            return recomputed_nodes[x]
        return env.get(x, x)
    return x


def remap_nodes_with_ac_annotations(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Remap (rematerialize) checkpointed nodes by duplicating checkpoint regions for backward.

    This is a generic rematerialization pass that relies on:
    1. Nodes tagged with CheckpointPolicy.MUST_RECOMPUTE/PREFER_RECOMPUTE
    2. Backward boundary marked with torch.fx.traceback.annotate({"backward": 0})

    Strategy:
    1. Insert all nodes in forward (including checkpointed nodes)
    2. When backward needs a checkpointed node, duplicate the entire checkpoint region
    3. Run DCE to remove unused forward nodes

    DCE handles the cleanup: if a checkpointed node is only used in backward, its forward
    version will have no users and DCE will remove it automatically.

    Args:
        gm: Graph with forward and backward ops

    Returns:
        Graph with rematerialized checkpoint regions
    """
    # Find backward boundary
    first_node_in_bwd = None
    for node in gm.graph.nodes:
        if _is_backward_node(node):
            first_node_in_bwd = node
            break

    if first_node_in_bwd is None:
        return gm

    # Build ordering
    order = {node: idx for idx, node in enumerate(gm.graph.nodes)}
    bwd_start = order[first_node_in_bwd]

    # Track which checkpointed nodes are used in backward (so we know which to duplicate)
    checkpointed_in_bwd: set[fx.Node] = set()
    for node in gm.graph.nodes:
        if must_recompute(node):
            for user in node.users:
                if order[user] >= bwd_start:
                    checkpointed_in_bwd.add(node)
                    break

    # Build reordered graph
    new_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    recomputed_nodes: dict[fx.Node, fx.Node] = {}

    # Add placeholders
    for node in gm.graph.find_nodes(op="placeholder"):
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    # Insert forward - insert ALL nodes (no deferral)
    # DCE will remove checkpointed nodes that are only used in backward
    for node in list(gm.graph.nodes)[:bwd_start]:
        if node.op in ("placeholder", "output"):
            continue
        if node not in env:
            _insert_node_with_deps(
                node,
                False,
                order,
                recomputed_nodes,
                env,
                checkpointed_in_bwd,
                new_graph,
            )

    # Insert backward - duplicate checkpointed nodes as needed
    for node in list(gm.graph.nodes)[bwd_start:]:
        if node.op == "output":
            continue
        _insert_node_with_deps(
            node, True, order, recomputed_nodes, env, checkpointed_in_bwd, new_graph
        )

    # Handle output
    output_node = list(gm.graph.nodes)[-1]
    assert output_node.op == "output"

    new_graph.output(
        tuple(
            _get_output_arg(arg, recomputed_nodes, env) for arg in output_node.args[0]
        )
    )
    new_gm = torch.fx.GraphModule(gm, new_graph)

    # DCE removes forward checkpointed nodes that are only used in backward
    # After duplication, those forward nodes have no users
    new_gm.graph.eliminate_dead_code()
    new_gm.recompile()

    return new_gm
