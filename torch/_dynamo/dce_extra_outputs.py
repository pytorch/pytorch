"""
DCE pass for unused extra outputs in HOP subgraphs.

When enable_side_effects_with_extra_outputs is True, HOPs like invoke_subgraph and
checkpoint (tag_activation_checkpoint)
return all intermediate tensors/symints as extra outputs to support side effects.
However, many of these extra outputs may not actually be used in the parent graph.

This pass removes unused extra outputs by:
1. Collecting all callers for each subgraph
2. Checking if each output is used by all callers
3. Removing unused outputs from the subgraph's output node
4. Updating the HOP call and getitem indices in all call sites

"""

import collections
import operator

import torch


# HOPs that may have extra outputs that can be DCE'd
_HOPS_WITH_EXTRA_OUTPUTS = {
    torch.ops.higher_order.invoke_subgraph,
    torch.ops.higher_order.tag_activation_checkpoint,
    # torch.ops.higher_order.autograd_function_apply,
}


def dce_hop_extra_outputs(gm: torch.fx.GraphModule) -> bool:
    """
    Remove unused extra outputs from HOP calls in all submodules.

    For each subgraph output, check if any caller has a getitem for that index
    with users. If no caller uses it, remove the output.
    If the user in caller is an output node, to simply the algorithm, we do not recursively check
    if the caller's output is used further up in the call chain.

    Args:
        gm: The GraphModule to optimize

    Returns:
        True if any modifications were made, False otherwise
    """
    # Collect all subgraph usages: subgraph_id -> list of (parent_gm, subgraph_name, hop_node)
    subgraph_id_to_callers: dict[
        int, list[tuple[torch.fx.GraphModule, str, torch.fx.Node]]
    ] = collections.defaultdict(list)
    _collect_all_subgraph_usages(gm, subgraph_id_to_callers)

    if not subgraph_id_to_callers:
        return False

    modified = False

    for callers in subgraph_id_to_callers.values():
        parent_gm, subgraph_name, _ = callers[0]
        subgraph = getattr(parent_gm, subgraph_name)

        if not isinstance(subgraph, torch.fx.GraphModule):
            continue

        output_node = next(n for n in subgraph.graph.nodes if n.op == "output")
        output_args = output_node.args[0]
        if not isinstance(output_args, (tuple, list)):
            continue

        num_outputs = len(output_args)
        used_indices: set[int] = set()

        # Check which outputs are used by any caller
        for idx in range(num_outputs):
            if _is_output_used(idx, callers):
                used_indices.add(idx)

        # DCE if some outputs are unused
        if 0 < len(used_indices) < num_outputs:
            if _dce_subgraph(subgraph, callers, used_indices):
                modified = True

    return modified


def _collect_all_subgraph_usages(
    gm: torch.fx.GraphModule,
    subgraph_id_to_callers: dict[
        int, list[tuple[torch.fx.GraphModule, str, torch.fx.Node]]
    ],
) -> None:
    """Recursively collect all HOP usages across the graph tree."""
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in _HOPS_WITH_EXTRA_OUTPUTS:
            subgraph_attr = node.args[0]
            if (
                isinstance(subgraph_attr, torch.fx.Node)
                and subgraph_attr.op == "get_attr"
            ):
                subgraph_name = subgraph_attr.target
                assert isinstance(subgraph_name, str)
                subgraph = getattr(gm, subgraph_name, None)
                if isinstance(subgraph, torch.fx.GraphModule):
                    subgraph_id = id(subgraph)
                    subgraph_id_to_callers[subgraph_id].append(
                        (gm, subgraph_name, node)
                    )
                    _collect_all_subgraph_usages(subgraph, subgraph_id_to_callers)


def _is_output_used(
    output_idx: int,
    callers: list[tuple[torch.fx.GraphModule, str, torch.fx.Node]],
) -> bool:
    """Check if output_idx is used by ANY caller (has a getitem with users)."""
    for _parent_gm, _subgraph_name, hop_node in callers:
        for user in hop_node.users:
            if user.op == "call_function" and user.target == operator.getitem:
                if user.args[1] == output_idx and len(user.users) > 0:
                    return True
    return False


def _dce_subgraph(
    subgraph: torch.fx.GraphModule,
    callers: list[tuple[torch.fx.GraphModule, str, torch.fx.Node]],
    used_indices: set[int],
) -> bool:
    """
    DCE a subgraph by removing unused output indices.

    Updates the subgraph's output node, all getitem nodes in callers,
    and example_value metadata on HOP nodes.
    """
    output_node = next(n for n in subgraph.graph.nodes if n.op == "output")
    old_outputs = list(output_node.args[0])

    # Check if this is the forward subgraph of autograd_function_apply
    # For autograd_function_apply, the fwd subgraph must return (output, saved_values, ...)
    # where indices 0 and 1 are ALWAYS required by the runtime
    # is_autograd_fwd = any(
    #     node.target == torch.ops.higher_order.autograd_function_apply
    #     for node in hop_nodes
    # )
    is_autograd_fwd = False

    # For autograd_function_apply forward subgraph, indices 0 (output) and 1 (saved_values)
    # are ALWAYS used by the runtime, even if not explicitly accessed via getitem
    if is_autograd_fwd and len(old_outputs) >= 2:
        used_indices.add(0)  # output
        used_indices.add(1)  # saved_values

    # Nothing to DCE if all outputs are used or no outputs are used
    if len(used_indices) >= len(old_outputs) or len(used_indices) == 0:
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
    # Create a new output node with the filtered outputs
    with subgraph.graph.inserting_before(output_node):
        new_output_node = subgraph.graph.output(tuple(new_outputs))
    output_node.replace_all_uses_with(new_output_node)
    subgraph.graph.erase_node(output_node)

    for parent_gm, _, hop_node in callers:
        # Update getitem nodes to use new indices
        for user in list(hop_node.users):
            if user.op == "call_function" and user.target == operator.getitem:
                old_idx = user.args[1]
                assert isinstance(old_idx, int)

                if old_idx not in old_to_new:
                    assert len(list(user.users)) == 0
                    parent_gm.graph.erase_node(user)
                    continue

                new_idx = old_to_new[old_idx]
                # Create a new getitem node with the new index
                with parent_gm.graph.inserting_before(user):
                    new_getitem = parent_gm.graph.call_function(
                        operator.getitem, args=(user.args[0], new_idx)
                    )
                    # Copy metadata from old node
                    new_getitem.meta = user.meta.copy()
                user.replace_all_uses_with(new_getitem)
                parent_gm.graph.erase_node(user)

        # Update example_value metadata on hop_node
        if "example_value" in hop_node.meta:
            old_example = hop_node.meta["example_value"]
            assert isinstance(old_example, (tuple, list))
            new_example = tuple(
                old_example[old_idx]
                for old_idx in range(len(old_outputs))
                if old_idx in used_indices
            )
            hop_node.meta["example_value"] = new_example

    # Recompile subgraph and all modified parent graphs
    subgraph.graph.lint()
    subgraph.recompile()

    for parent_gm in {caller[0] for caller in callers}:
        parent_gm.graph.lint()
        parent_gm.recompile()

    return True
