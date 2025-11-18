"""
AC Reordering Pass for Inference Graphs

This pass minimizes AC node lifetimes by deferring whole AC chains to backward.

Only looks at nodes tagged with CheckpointPolicy.MUST_RECOMPUTE/PREFER_RECOMPUTE.
When an AC node is deferred/duplicated, its entire AC chain (all AC dependencies)
is also deferred/duplicated, while non-AC nodes (checkpoint inputs) are reused from forward.

This mimics partitioner behavior: recompute checkpoint regions, reuse checkpoint inputs.
"""

import torch
import torch.fx as fx
from torch._functorch.partitioners import (
    collect_deps_with_filter,
    insert_nodes_in_original_order,
)
from torch.utils.checkpoint import CheckpointPolicy


def must_recompute(node: fx.Node) -> bool:
    """Check if node is tagged for AC recomputation"""
    return node.meta.get("recompute", None) in [
        CheckpointPolicy.MUST_RECOMPUTE,
        CheckpointPolicy.PREFER_RECOMPUTE,
    ]


def _is_backward_node(node: fx.Node) -> bool:
    """Check if node is in backward region via annotation"""
    return node.meta.get("custom", {}).get("backward") is not None


def reorder_ac_nodes_for_inference(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Reorder AC nodes to minimize lifetime by deferring whole AC chains to backward.

    Only looks at nodes tagged with CheckpointPolicy.MUST_RECOMPUTE/PREFER_RECOMPUTE.
    When deferring/duplicating an AC node, we also defer/duplicate its AC dependencies
    (the whole checkpoint region), but reuse non-AC nodes (checkpoint inputs) from forward.

    This mimics partitioner behavior: recompute checkpoint region, reuse checkpoint inputs.

    Strategy:
    1. AC nodes ONLY used in backward: Defer entire AC chain to backward
    2. AC nodes used in BOTH: Duplicate AC chain in backward, keep original in forward
    3. Non-AC nodes (checkpoint inputs): Always reused from forward

    Args:
        gm: Inference graph with forward and backward ops

    Returns:
        Reordered graph with minimized AC lifetimes
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

    # Categorize AC nodes by usage
    ac_used_in_fwd = set()
    ac_used_in_bwd = set()

    for node in gm.graph.nodes:
        if must_recompute(node):
            for user in node.users:
                if order[user] < bwd_start:
                    ac_used_in_fwd.add(node)
                else:
                    ac_used_in_bwd.add(node)

    # Only need to track AC nodes used in backward or both regions
    ac_bwd_only = ac_used_in_bwd - ac_used_in_fwd
    ac_both = ac_used_in_fwd & ac_used_in_bwd

    # Build reordered graph
    new_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    ac_duplicates: dict[fx.Node, fx.Node] = {}

    # Add placeholders
    for node in gm.graph.find_nodes(op="placeholder"):
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    # Nodes to defer (don't compute in forward)
    deferred_ac = ac_bwd_only

    def insert_node_with_deps(node, for_backward=False):
        """Insert node and dependencies, deferring AC nodes as appropriate"""

        # Define skip condition based on context
        def skip_condition(n):
            # Skip deferred AC in forward
            if not for_backward and n in deferred_ac:
                return True

            if for_backward:
                # Skip if duplicate exists or already in env (and not needing dup)
                if n in ac_duplicates:
                    return True
                if n in env and n not in ac_both:
                    return True
            else:
                if n in env:
                    return True
            return False

        insertable_nodes = collect_deps_with_filter(node, order, skip_condition)

        # Helper to get inputs when duplicating AC nodes
        def get_input_for_ac_duplicate(x):
            """Get input when duplicating AC nodes - reuse checkpoint inputs."""
            if x in ac_duplicates:
                return ac_duplicates[x]
            # If input is also an AC node, duplicate it first
            if must_recompute(x) and x not in ac_duplicates:
                duplicate_ac_chain(x)
                return ac_duplicates[x]
            # Non-AC nodes (checkpoint inputs) - reuse from forward
            return env.get(x, x)

        # Helper to get inputs for regular backward nodes
        def get_input_for_backward(x):
            if x in ac_duplicates:
                return ac_duplicates[x]
            return env.get(x, x)

        # Define custom node copy function for AC-specific logic
        def node_copy_fn(n):
            # Early returns for already processed nodes
            if n in env and (not for_backward or n not in ac_both):
                return None
            if n in ac_duplicates:
                return None

            # Forward pass: simple copy
            if not for_backward:
                return new_graph.node_copy(n, lambda x: env[x])

            # Backward pass: handle AC nodes used in both regions
            if n in ac_both:
                # When duplicating AC nodes, duplicate only AC dependencies.
                # This mimics partitioner behavior: recompute ops inside checkpoint,
                # reuse checkpoint inputs from forward.
                dup = new_graph.node_copy(n, get_input_for_ac_duplicate)
                dup.name = n.name + "_recomputed"
                ac_duplicates[n] = dup
                return dup

            # Backward pass: regular node
            return new_graph.node_copy(n, get_input_for_backward)

        # Insert nodes in original order
        for n in sorted(insertable_nodes, key=lambda x: order[x]):
            new_node = node_copy_fn(n)
            if new_node is not None:
                if n not in ac_duplicates:  # Regular nodes go to env
                    env[n] = new_node

    def duplicate_ac_chain(ac_node):
        """
        Recursively duplicate an AC node and its AC dependencies.
        STOP at non-AC nodes (checkpoint inputs) - reuse those from forward.

        This mimics the partitioner's behavior where only nodes inside the
        checkpoint region are recomputed, and checkpoint inputs are reused.
        """
        if ac_node in ac_duplicates:
            return

        # First, duplicate all AC dependencies (nodes inside checkpoint region)
        for inp in ac_node.all_input_nodes:
            # Skip placeholders (can be reused)
            if inp.op == "placeholder":
                continue
            # Only duplicate if this dependency is ALSO an AC node
            # (i.e., it's inside the checkpoint region)
            if must_recompute(inp) and inp not in ac_duplicates:
                duplicate_ac_chain(inp)

        # Now duplicate this node
        def get_input(x):
            if x in ac_duplicates:
                return ac_duplicates[x]
            # Placeholders and non-AC nodes can be reused from forward
            return env.get(x, x)

        dup = new_graph.node_copy(ac_node, get_input)
        # Only add _recomputed suffix to actual AC nodes
        if ac_node in ac_both:
            dup.name = ac_node.name + "_recomputed"
        else:
            dup.name = ac_node.name + "_dup"
        ac_duplicates[ac_node] = dup

    # Insert forward (defer backward-only AC nodes)
    for node in list(gm.graph.nodes)[:bwd_start]:
        if node.op in ("placeholder", "output"):
            continue
        if node not in env and node not in deferred_ac:
            insert_node_with_deps(node, for_backward=False)

    # Insert backward (compute deferred AC nodes + create duplicates)
    for node in list(gm.graph.nodes)[bwd_start:]:
        if node.op == "output":
            continue
        insert_node_with_deps(node, for_backward=True)

    # Handle output
    output_node = list(gm.graph.nodes)[-1]
    assert output_node.op == "output"

    def get_output_arg(x):
        if isinstance(x, fx.Node):
            if x in ac_duplicates:
                return ac_duplicates[x]
            return env.get(x, x)
        return x

    new_graph.output(tuple(get_output_arg(arg) for arg in output_node.args[0]))
    new_gm = torch.fx.GraphModule(gm, new_graph)
    return new_gm
