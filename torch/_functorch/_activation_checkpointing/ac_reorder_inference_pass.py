"""
Activation Checkpointing Reordering Pass for Inference Graphs

Minimizes memory by duplicating checkpointed operations for backward use,
then using DCE to remove unused forward versions.

This closely follows the pattern from default_partition's reordering_to_mimic_autograd_engine.
"""

import torch
import torch.fx as fx
from torch._functorch.partitioners import collect_deps_with_filter, must_recompute


def _is_backward_node(node: fx.Node) -> bool:
    """Check if node is in backward region via annotation"""
    return node.meta.get("custom", {}).get("backward") is not None


def remap_nodes_with_ac_annotations(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Remap (rematerialize) checkpointed nodes by duplicating checkpoint regions for backward.

    This follows the same pattern as reordering_to_mimic_autograd_engine in partitioners.py:
    1. Walk through graph node by node
    2. For each node, pull in dependencies using collect_deps_with_filter
    3. Duplicate checkpointed nodes when backward needs them
    4. DCE cleans up unused forward copies

    Args:
        gm: Graph with forward and backward ops (inference mode joint graph)

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

    # Track which checkpointed nodes are used in backward
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

    # Add placeholders (like reordering_to_mimic_autograd_engine)
    for node in gm.graph.find_nodes(op="placeholder"):
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    def duplicate_checkpoint_chain(node: fx.Node) -> fx.Node:
        """
        Duplicate a checkpointed node and its checkpointed dependencies.
        Stops at non-checkpointed nodes (checkpoint inputs) - reuses those from forward.
        """
        if node in recomputed_nodes:
            return recomputed_nodes[node]

        # Recursively duplicate checkpointed dependencies first
        for inp in node.all_input_nodes:
            if inp.op == "placeholder":
                continue
            if must_recompute(inp) and inp not in recomputed_nodes:
                duplicate_checkpoint_chain(inp)

        # Create duplicate node
        def get_input(x: fx.Node) -> fx.Node:
            # Use recomputed version if available
            if x in recomputed_nodes:
                return recomputed_nodes[x]
            # Otherwise reuse from forward (checkpoint inputs)
            return env.get(x, x)

        dup = new_graph.node_copy(node, get_input)
        dup.name = node.name + "_recomputed"
        recomputed_nodes[node] = dup
        return dup

    def insert_node_in_graph(node: fx.Node, for_backward: bool) -> None:
        """
        Insert a node and its dependencies into the graph.
        Pattern from reordering_to_mimic_autograd_engine.
        """

        # Skip condition: already in env (but in backward, may need to duplicate AC nodes)
        def skip_condition(n: fx.Node) -> bool:
            if n in recomputed_nodes:
                return True
            if n in env:
                # In backward, we need to duplicate checkpointed nodes even if in env
                if for_backward and must_recompute(n) and n in checkpointed_in_bwd:
                    return False
                return True
            return False

        # Collect dependencies
        insertable_nodes = collect_deps_with_filter(node, order, skip_condition)

        # Insert nodes in original order
        for n in sorted(insertable_nodes, key=lambda x: order[x]):
            if n not in env:
                # Backward: duplicate checkpointed nodes
                if for_backward and must_recompute(n) and n in checkpointed_in_bwd:
                    env[n] = duplicate_checkpoint_chain(n)
                # Backward: regular nodes may need recomputed inputs
                elif for_backward:

                    def get_input_for_backward(x: fx.Node) -> fx.Node:
                        if x in recomputed_nodes:
                            return recomputed_nodes[x]
                        if must_recompute(x):
                            # Duplicate checkpoint chain for this input
                            return duplicate_checkpoint_chain(x)
                        return env.get(x, x)

                    env[n] = new_graph.node_copy(n, get_input_for_backward)
                # Forward: simple copy
                else:
                    env[n] = new_graph.node_copy(n, lambda x: env[x])

    # Insert forward nodes
    for node in list(gm.graph.nodes)[:bwd_start]:
        if node.op in ("placeholder", "output"):
            continue
        insert_node_in_graph(node, for_backward=False)

    # Insert backward nodes
    for node in list(gm.graph.nodes)[bwd_start:]:
        if node.op == "output":
            continue
        insert_node_in_graph(node, for_backward=True)

    # Handle output
    output_node = list(gm.graph.nodes)[-1]
    assert output_node.op == "output"

    def get_output_arg(x):
        if isinstance(x, fx.Node):
            if x in recomputed_nodes:
                return recomputed_nodes[x]
            return env.get(x, x)
        return x

    new_graph.output(tuple(get_output_arg(arg) for arg in output_node.args[0]))
    new_gm = torch.fx.GraphModule(gm, new_graph)

    # DCE removes forward checkpointed nodes that are only used in backward
    new_gm.graph.eliminate_dead_code()
    new_gm.recompile()

    return new_gm
