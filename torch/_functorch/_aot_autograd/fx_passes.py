from collections import defaultdict
from typing import Dict, List

import torch
from torch import fx
from torch.utils import _pytree as pytree


def is_primal(node: fx.Node) -> bool:
    return node.op == "placeholder" and "tangents" not in str(node.target)


def move_resize_zero_to_end_of_graph(graph: fx.Graph) -> None:
    node_list = list(graph.nodes)
    return_node = node_list[-1]
    assert return_node.target == "output"
    resize_nodes_to_insert_at_end = []
    node_to_usage_map = defaultdict(list)
    for i, n in enumerate(node_list):
        if n.op == "call_function":
            if (
                n.target is torch.ops.inductor.resize_storage_bytes_.default
                and n.args[1] == 0
            ):
                resize_nodes_to_insert_at_end.append((n, i))
            args_flat = pytree.arg_tree_leaves(*n.args, **n.kwargs)
            for arg in args_flat:
                node_to_usage_map[arg].append(i)
    # Assumptions: the size-0 tensor X can then be used only in the following ops:
    # 1. `aten.set_(X, ...)`
    # 2. `aten.set_(..., X)`
    # 3. `inductor.resize_storage_bytes_(X, 0)`
    # 4. graph output
    for resize_node, resize_node_idx in reversed(resize_nodes_to_insert_at_end):
        usage_idx_list = node_to_usage_map[resize_node.args[0]]
        pos_in_usage_idx_list = usage_idx_list.index(resize_node_idx)
        for i in range(pos_in_usage_idx_list + 1, len(usage_idx_list)):
            node = node_list[usage_idx_list[i]]
            if node.target is torch.ops.aten.set_.source_Tensor and (
                node.args[0] == resize_node.args[0]
                or node.args[1] == resize_node.args[0]
            ):
                continue
            if (
                node.target is torch.ops.inductor.resize_storage_bytes_.default
                and node.args[0] == resize_node.args[0]
                and node.args[1] == 0
            ):
                continue
            if node.target == "output":
                continue
            raise RuntimeError(
                f"Input to .resize_storage_bytes_(X, 0) is used after the resize-to-0 op by `{node}`. "
                f"Violating graph: {graph}"
            )
        if pos_in_usage_idx_list == len(usage_idx_list) - 1:
            # If `.resize_(X, 0)` node is the last usage of X, we move the resize node to the end of the graph.
            next_valid_usage_of_resized_tensor = return_node
        else:
            next_valid_usage_of_resized_tensor = node_list[
                usage_idx_list[pos_in_usage_idx_list + 1]
            ]
        with graph.inserting_before(next_valid_usage_of_resized_tensor):
            new_resize_node = graph.call_function(
                torch.ops.inductor.resize_storage_bytes_.default, resize_node.args
            )
        graph.erase_node(resize_node)


def refunctionalize_set(graph: fx.Graph) -> None:
    node_list = list(graph.nodes)
    return_node = node_list[-1]
    assert return_node.target == "output"
    primal_inputs = [*filter(is_primal, node_list)]
    primal_input_to_set_idx = defaultdict(list)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    set_nodes_to_be_deleted = set()

    # Step 1: Clean up to enforce `.set_(primal_X, ...)` usage.
    # We replace:
    # ```
    # set_1 = aten.set_(primal_X, Y1)
    # set_2 = aten.set_(set_1, Y2)
    # ```
    # to:
    # ```
    # set_1 = aten.set_(primal_X, Y1)
    # set_2 = aten.set_(primal_X, Y2)
    # ```
    # Note that this process is iterative and will handle any number (>=2) of nested `.set_()`.
    for i, n in enumerate(node_list):
        if n.op == "call_function" and n.target is torch.ops.aten.set_.source_Tensor:
            if (
                n.args[0].target is torch.ops.aten.set_.source_Tensor
                and n.args[0].args[0] in primal_inputs
            ):
                n.args = (n.args[0].args[0], n.args[1])
            assert n.args[0] in primal_inputs, (
                "Violated assumption: every `.set_` node should be setting into the primal input. "
                f"Please report a bug to PyTorch. Violating graph: {graph}"
            )

    # Step 2: Re-functionalizing `.set_(primal_X, ...)`.
    # Replacement conditions:
    # 1. `.set_(primal_X, Y)` exists in graph.
    # 2. `.set_(Y, ...)` is not called between primal_X's this `.set_(primal_X, Y)` and its next `.set_(primal_X, ...)` (or end of graph)  # noqa: B950
    #    - This ensures primal_X and Y are semantically always meant to be the same tensor within this section of the graph.
    # If the above two conditions are met, then within this section of the graph we will replace
    # usage of output of this `.set_(primal_X, Y)` node with Y, and delete this `.set_(primal_X, Y)` node.
    # For any primal input, if we have deleted the last `.set_(primal_X, ...)`, we will re-insert
    # `.set_(primal_X, Y_last)` at the end of the graph.
    for i, n in enumerate(node_list):
        # For aten.set_(X, Y), X must be primal input of graph.
        if (
            n.op == "call_function"
            and n.target is torch.ops.aten.set_.source_Tensor
            and n.args[0] in primal_inputs
        ):
            primal_input_to_set_idx[n.args[0]].append(i)
    for primal_input, set_idx_list in primal_input_to_set_idx.items():
        for i in range(len(set_idx_list)):
            set_node_idx = set_idx_list[i]
            if i < len(set_idx_list) - 1:
                next_set_node_idx = set_idx_list[i + 1]
            else:
                # The last section always ends with the graph output node.
                next_set_node_idx = len(node_list) - 1
            set_node = node_list[set_node_idx]
            Y_input = set_node.args[1]
            # Between primal_X's this `.set_(primal_X, Y)` and its next `.set_(primal_X, ...)` (or end of graph),
            # if `.set_(Y, ...)` is not called, we will replace usage of output of this `.set_(primal_X, Y)` node with Y,
            # and delete the `.set_(primal_X, Y)` node.
            # (NOTE: Y may or may not be a primal input. If Y is not a primal and `.set_(Y, ...)` exists, we error out.)
            set_Y_exists = False
            for idx in range(set_node_idx + 1, next_set_node_idx):
                if (
                    node_list[idx].target is torch.ops.aten.set_.source_Tensor
                    and node_list[idx].args[0] == Y_input
                ):
                    set_Y_exists = True
                    assert (
                        Y_input in primal_inputs
                    ), f".set_(non_primal, ...) is not supported. Violating graph: {graph}"
            if not set_Y_exists:
                set_node.replace_all_uses_with(
                    Y_input,
                    delete_user_cb=lambda n: node_to_idx[n] > set_node_idx
                    and node_to_idx[n] < next_set_node_idx,
                )
                set_nodes_to_be_deleted.add(set_node)
        # For any primal input, if we have deleted the last `.set_(primal_X, ...)`,
        # we will re-insert `.set_(primal_X, Y_last)` at the end of the graph.
        last_set_node = node_list[primal_input_to_set_idx[primal_input][-1]]
        if last_set_node in set_nodes_to_be_deleted:
            with graph.inserting_before(return_node):
                new_last_set_node = graph.call_function(
                    last_set_node.target, last_set_node.args
                )
    # Replace set_ nodes in graph output with the corresponding primal input.
    new_return_node_args = []
    for arg in return_node.args:
        if arg in set_nodes_to_be_deleted:
            arg = arg.args[0]
            assert arg in primal_inputs
        new_return_node_args.append(arg)
    return_node.args = tuple(new_return_node_args)
    # Finally, delete the old set_ nodes.
    for set_node in set_nodes_to_be_deleted:
        graph.erase_node(set_node)


def collect_graph_epilogue_mutable_ops(graph: fx.Graph) -> List[fx.Node]:
    epilogue_mutable_ops = []
    node_list = list(graph.nodes)
    for node in reversed(node_list):
        if node.op == "output":
            continue
        elif node.op == "call_function" and (
            isinstance(node.target, torch._ops.OpOverload)
            and (
                node.target._schema.is_mutable
                or node.target is torch.ops.inductor.resize_storage_bytes_.default
            )
        ):
            epilogue_mutable_ops.append(node)
        else:
            break
    return reversed(epilogue_mutable_ops)  # type: ignore[return-value]


def collect_nodes_set_into_primal_in_graph_epilogue(
    graph: fx.Graph,
) -> Dict[fx.Node, fx.Node]:
    primal_inputs = [*filter(is_primal, graph.nodes)]
    epilogue_mutable_ops = collect_graph_epilogue_mutable_ops(graph)
    node_to_primal_map = {
        node.args[1]: node.args[0]
        for node in epilogue_mutable_ops
        if node.target is torch.ops.aten.set_.source_Tensor
        and node.args[0] in primal_inputs
    }
    return node_to_primal_map  # type: ignore[return-value]
