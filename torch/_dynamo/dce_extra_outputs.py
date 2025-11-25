"""
Dead Code Elimination pass for unused extra outputs in invoke_subgraph calls.

When enable_side_effects_with_extra_outputs is True, HOPs like invoke_subgraph
return all intermediate tensors/symints as extra outputs to support side effects.
However, many of these extra outputs may not actually be used in the parent graph.

This pass removes unused extra outputs by:
1. Identifying which outputs of invoke_subgraph calls are actually used
2. Removing unused outputs from the subgraph's output node
3. Updating the invoke_subgraph call to reflect the new output arity
4. Updating getitem indices to account for removed outputs
"""

import operator
import torch
from torch import fx
from typing import Dict, List, Set, Tuple


def dce_invoke_subgraph_extra_outputs(gm: fx.GraphModule) -> bool:
    """
    Remove unused extra outputs from invoke_subgraph calls.

    Args:
        gm: The GraphModule to optimize

    Returns:
        True if any modifications were made, False otherwise
    """
    modified = False
    graph = gm.graph

    # Find all invoke_subgraph calls
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.higher_order.invoke_subgraph
        ):
            if _dce_single_invoke_subgraph(gm, node):
                modified = True

    if modified:
        graph.lint()
        gm.recompile()

    return modified


def _dce_single_invoke_subgraph(gm: fx.GraphModule, invoke_node: fx.Node) -> bool:
    """
    DCE unused outputs for a single invoke_subgraph node.

    Args:
        gm: The parent GraphModule
        invoke_node: The invoke_subgraph call node

    Returns:
        True if modifications were made, False otherwise
    """
    # Get the subgraph module
    subgraph_attr = invoke_node.args[0]
    if not isinstance(subgraph_attr, fx.Node) or subgraph_attr.op != "get_attr":
        return False

    subgraph_name = subgraph_attr.target
    subgraph = getattr(gm, subgraph_name)

    if not isinstance(subgraph, fx.GraphModule):
        return False

    # Find which outputs are used via getitem
    used_indices: Set[int] = set()
    getitem_nodes: List[fx.Node] = []

    for user in invoke_node.users:
        if user.op == "call_function" and user.target == operator.getitem:
            idx = user.args[1]
            if isinstance(idx, int):
                used_indices.add(idx)
                getitem_nodes.append(user)

    # Get the output node from subgraph
    output_nodes = [n for n in subgraph.graph.nodes if n.op == "output"]
    if len(output_nodes) != 1:
        return False

    output_node = output_nodes[0]
    if not output_node.args or not isinstance(output_node.args[0], (tuple, list)):
        return False

    old_outputs = list(output_node.args[0])
    num_outputs = len(old_outputs)

    # If all outputs are used, nothing to DCE
    if len(used_indices) == num_outputs:
        return False

    # Build mapping from old indices to new indices
    old_to_new: Dict[int, int] = {}
    new_outputs = []
    new_idx = 0

    for old_idx in range(num_outputs):
        if old_idx in used_indices:
            old_to_new[old_idx] = new_idx
            new_outputs.append(old_outputs[old_idx])
            new_idx += 1

    # Update subgraph output node
    output_node.args = (tuple(new_outputs),)

    # Update getitem nodes to use new indices
    for getitem_node in getitem_nodes:
        old_idx = getitem_node.args[1]
        if old_idx in old_to_new:
            new_idx = old_to_new[old_idx]
            getitem_node.args = (getitem_node.args[0], new_idx)

    # Update example_value metadata on invoke_node
    if "example_value" in invoke_node.meta:
        old_example = invoke_node.meta["example_value"]
        if isinstance(old_example, (tuple, list)):
            new_example = tuple(
                old_example[old_idx]
                for old_idx in range(num_outputs)
                if old_idx in used_indices
            )
            invoke_node.meta["example_value"] = new_example

    subgraph.graph.lint()
    subgraph.recompile()

    return True
