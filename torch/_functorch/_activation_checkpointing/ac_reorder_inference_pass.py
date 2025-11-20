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
    if has_recomputable_ops(gm):
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

    # Add placeholders
    for node in gm.graph.find_nodes(op="placeholder"):
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    def duplicate_checkpoint_chain(node: fx.Node) -> fx.Node:
        """
        Duplicate a checkpointed node and its checkpointed dependencies.
        Stops at non-checkpointed nodes (checkpoint inputs) - reuses those from forward.
        """
        if node in recomputed_nodes:
            return recomputed_nodes[node]

        # Recursively duplicate checkpointed dependencies first, in original forward order
        checkpointed_inputs = [
            inp
            for inp in node.all_input_nodes
            if inp.op != "placeholder" and must_recompute(inp)
        ]
        # Sort by original order to preserve forward ordering
        for inp in sorted(checkpointed_inputs, key=lambda n: order[n]):
            if inp not in recomputed_nodes:
                duplicate_checkpoint_chain(inp)

        # Create duplicate node
        def get_input(x: fx.Node) -> fx.Node:
            # Use recomputed version if available
            if x in recomputed_nodes:
                return recomputed_nodes[x]
            # Otherwise reuse from forward (checkpoint inputs)
            assert x in env
            return env[x]

        dup = new_graph.node_copy(node, get_input)
        dup.name = node.name + "_recomputed"
        recomputed_nodes[node] = dup
        return dup

    def insert_bwd_node(node: fx.Node) -> None:
        """Insert a backward node - duplicates checkpointed inputs on demand."""
        if node in env or node in recomputed_nodes:
            return

        def remat_input(x: fx.Node) -> fx.Node:
            if must_recompute(x):
                return duplicate_checkpoint_chain(x)
            assert x in env
            return env[x]

        env[node] = new_graph.node_copy(node, remat_input)

    # Insert forward nodes
    for node in list(gm.graph.nodes)[:bwd_start]:
        if node.op in ("placeholder", "output"):
            continue
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    # Insert backward nodes
    for node in list(gm.graph.nodes)[bwd_start:]:
        if node.op == "output":
            continue
        insert_bwd_node(node)

    output_node = list(gm.graph.nodes)[-1]
    assert output_node.op == "output"

    def get_output_arg(x):
        if not isinstance(x, fx.Node):
            return x
        if x in recomputed_nodes:
            return recomputed_nodes[x]
        assert x in env
        return env[x]

    new_graph.output(tuple(get_output_arg(arg) for arg in output_node.args[0]))
    new_gm = torch.fx.GraphModule(gm, new_graph)

    # DCE with custom is_impure_node (like default_partition)
    # Override to treat certain collectives as pure for DCE purposes
    new_gm.graph.eliminate_dead_code(is_impure_node=is_not_collective)

    # raise_getitems pass for better memory (like default_partition)
    new_gm = raise_getitems(new_gm)

    new_gm.recompile()

    return new_gm
