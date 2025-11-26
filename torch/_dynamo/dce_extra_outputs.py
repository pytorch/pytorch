"""
DCE pass for unused extra outputs in HOP subgraphs.

When enable_side_effects_with_extra_outputs is True, HOPs like invoke_subgraph,
checkpoint (tag_activation_checkpoint), and autograd.Function (autograd_function_apply)
return all intermediate tensors/symints as extra outputs to support side effects.
However, many of these extra outputs may not actually be used in the parent graph.

Special handling for autograd_function_apply:
- The forward subgraph MUST return (output, saved_values, ...) where indices 0 and 1
  are always required by the runtime
- Only indices 2+ (extra intermediates) can be removed by DCE

This pass removes unused extra outputs by:
1. Identifying which outputs of HOP calls are actually used
2. Removing unused outputs from the subgraph's output node
3. Updating the HOP call to reflect the new output arity
4. Updating getitem indices to account for removed outputs
"""

import collections
import operator

import torch


# HOPs that may have extra outputs that can be DCE'd
_HOPS_WITH_EXTRA_OUTPUTS = {
    torch.ops.higher_order.invoke_subgraph,
    torch.ops.higher_order.tag_activation_checkpoint,
    torch.ops.higher_order.autograd_function_apply,
}


def dce_hop_extra_outputs(gm: torch.fx.GraphModule) -> bool:
    """
    Remove unused extra outputs from HOP calls.

    Args:
        gm: The GraphModule to optimize

    Returns:
        True if any modifications were made, False otherwise
    """
    modified = False
    graph = gm.graph

    # Group HOP nodes by subgraph name
    # Multiple invocations may share the same subgraph, so we need to check
    # which indices are used across ALL invocations before removing any
    subgraph_to_nodes: dict[str, list[torch.fx.Node]] = collections.defaultdict(list)

    for node in graph.nodes:
        if node.op == "call_function" and node.target in _HOPS_WITH_EXTRA_OUTPUTS:
            subgraph_attr = node.args[0]
            if (
                isinstance(subgraph_attr, torch.fx.Node)
                and subgraph_attr.op == "get_attr"
            ):
                subgraph_name = subgraph_attr.target
                assert isinstance(subgraph_name, str)
                subgraph_to_nodes[subgraph_name].append(node)

    # Process each unique subgraph
    for subgraph_name, invoke_nodes in subgraph_to_nodes.items():
        if _dce_subgraph(gm, subgraph_name, invoke_nodes):
            modified = True

    if modified:
        graph.lint()
        gm.recompile()

    return modified


def _dce_subgraph(
    gm: torch.fx.GraphModule, subgraph_name: str, invoke_nodes: list[torch.fx.Node]
) -> bool:
    subgraph = getattr(gm, subgraph_name)

    if not isinstance(subgraph, torch.fx.GraphModule):
        return False

    # Collect used indices across ALL invocations of this subgraph
    used_indices: set[int] = set()

    # Check if this is the forward subgraph of autograd_function_apply
    # For autograd_function_apply, the fwd subgraph must return (output, saved_values, ...)
    # where indices 0 and 1 are ALWAYS required by the runtime
    is_autograd_fwd = any(
        node.target == torch.ops.higher_order.autograd_function_apply
        for node in invoke_nodes
    )

    for invoke_node in invoke_nodes:
        for user in list(invoke_node.users):
            if user.op == "call_function" and user.target == operator.getitem:
                if len(list(user.users)) > 0:
                    idx = user.args[1]
                    assert isinstance(idx, int)
                    used_indices.add(idx)

    output_node = next(n for n in subgraph.graph.nodes if n.op == "output")
    old_outputs = list(output_node.args[0])

    # For autograd_function_apply forward subgraph, indices 0 (output) and 1 (saved_values)
    # are ALWAYS used by the runtime, even if not explicitly accessed via getitem
    if is_autograd_fwd and len(old_outputs) >= 2:
        used_indices.add(0)  # output
        used_indices.add(1)  # saved_values

    if len(used_indices) == len(old_outputs):
        return False

    # can still have side effect inside invoke subgraph, so we shouldn't kill it
    if len(used_indices) == 0:
        return False

    # Build mapping from old indices to new indices
    old_to_new: dict[int, int] = {}
    new_outputs = []
    new_idx = 0

    for old_idx in range(len(old_outputs)):
        if old_idx in used_indices:
            old_to_new[old_idx] = new_idx
            new_outputs.append(old_outputs[old_idx])
            new_idx += 1

    # Update subgraph output node
    output_node.args = (tuple(new_outputs),)

    for invoke_node in invoke_nodes:
        # Update getitem nodes to use new indices
        for user in list(invoke_node.users):
            if user.op == "call_function" and user.target == operator.getitem:
                old_idx = user.args[1]
                assert isinstance(old_idx, int)
                if old_idx in old_to_new:
                    new_idx = old_to_new[old_idx]
                    user.args = (user.args[0], new_idx)

        # Update example_value metadata on invoke_node
        if "example_value" in invoke_node.meta:
            old_example = invoke_node.meta["example_value"]
            assert isinstance(old_example, (tuple, list))
            new_example = tuple(
                old_example[old_idx]
                for old_idx in range(len(old_outputs))
                if old_idx in used_indices
            )
            invoke_node.meta["example_value"] = new_example

    subgraph.graph.lint()
    subgraph.recompile()

    return True
