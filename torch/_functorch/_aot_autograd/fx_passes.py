from collections import defaultdict
from typing import Dict

import torch
from torch import fx
from .functional_utils import collect_graph_epilogue_mutable_ops


def is_primal(node: fx.Node) -> bool:
    return node.op == "placeholder" and "tangents" not in str(node.target)


def move_resize_zero_to_end_of_graph(graph: fx.Graph) -> None:
    node_list = list(graph.nodes)
    return_node = node_list[-1]
    assert return_node.target == "output"
    resize_nodes_to_insert_at_end = []
    resized_tensors = set()
    nodes_seen = set()
    for i, n in enumerate(node_list):
        nodes_seen.add(n)
        if (
            n.op == "call_function"
            and n.target is torch.ops.inductor.resize_storage_bytes_.default
        ):
            resize_node = n
            resized_tensor = resize_node.args[0]
            assert (
                resize_node.args[1] == 0
            ), f"NYI: resizing a tensor `{resized_tensor}` to non-zero size. Violating graph: {graph}"
            assert (
                resized_tensor not in resized_tensors
            ), f"NYI: resizing the same tensor `{resized_tensor}` multiple times. Violating graph: {graph}"
            assert (
                len(set(resize_node.users.keys()) - nodes_seen) == 0
            ), f"NYI: output of resize node `{resize_node}` should not be used in downstream ops. Violating graph: {graph}"
            assert (
                len(set(resized_tensor.users.keys()) - nodes_seen) == 0
            ), f"NYI: size-0 tensor `{resized_tensor}` should not be used in downstream ops. Violating graph: {graph}"
            resize_nodes_to_insert_at_end.append((resize_node, i))
            resized_tensors.add(resized_tensor)
    for resize_node, resize_node_idx in reversed(resize_nodes_to_insert_at_end):
        with graph.inserting_before(return_node):
            new_resize_node = graph.call_function(
                torch.ops.inductor.resize_storage_bytes_.default, resize_node.args
            )
        graph.erase_node(resize_node)


def refunctionalize_set(graph: fx.Graph) -> None:
    node_list = list(graph.nodes)
    return_node = node_list[-1]
    assert return_node.target == "output"
    primal_inputs = [*filter(is_primal, node_list)]
    primal_input_to_set_node_idx = defaultdict(list)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    set_nodes_to_delete = set()

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
    # 1. `set__Z = .set_(primal_X, Y)` exists in graph.
    # 2. `.set_(Y, ...)` is not called between primal_X's this `.set_(primal_X, Y)` and its next `.set_(primal_X, ...)` (or end of graph)  # noqa: B950
    #    - This ensures primal_X and Y are semantically always meant to be the same tensor within this section of the graph.
    # If the above two conditions are met, then within this section of the graph we will replace
    # downstream usage of set__Z with Y, and delete this `set__Z = .set_(primal_X, Y)` node.
    # Example:
    #     ```
    #     def f(primal_X):
    #         ...
    #         set__Z = .set_(primal_X, Y)
    #         out = torch.matmul(set__Z, set__Z)
    #         ...
    #     ```
    # will be transformed to:
    #     ```
    #     def f(primal_X):
    #         ...
    #         out = torch.matmul(Y, Y)
    #         ...
    #     ```
    # For any primal input, if we have deleted the last `.set_(primal_X, ...)`, we will re-insert
    # `.set_(primal_X, Y_last)` at the end of the graph.
    for i, n in enumerate(node_list):
        # For aten.set_(X, Y), X must be primal input of graph.
        if n.op == "call_function" and n.target is torch.ops.aten.set_.source_Tensor:
            assert (
                n.args[0] in primal_inputs
            ), f"NYI: Calling `.set_(X, Y)` but X is not primal input of graph. Violating graph: {graph}"
            primal_input_to_set_node_idx[n.args[0]].append(i)
    for primal_input, set_idx_list in primal_input_to_set_node_idx.items():
        for i in range(len(set_idx_list)):
            set_node_idx = set_idx_list[i]
            if i < len(set_idx_list) - 1:
                next_set_node_idx = set_idx_list[i + 1]
            else:
                # The last section always ends with the graph output node.
                next_set_node_idx = len(node_list) - 1
            set_node = node_list[set_node_idx]
            Y_input = set_node.args[1]
            # Between primal_X's this `set__Z = .set_(primal_X, Y)` and its next `.set_(primal_X, ...)` (or end of graph):
            # 1. We assert that `.set_(Y, ...)` is never called.
            # 2. Then, we will replace downstream usage of set__Z with Y, and delete the `set__Z = .set_(primal_X, Y)` node.
            for idx in range(set_node_idx + 1, next_set_node_idx):
                assert not (
                    node_list[idx].target is torch.ops.aten.set_.source_Tensor
                    and Y_input in node_list[idx].args
                ), f"NYI: `.set_(Y, ...)` after `.set_(primal_X, Y)`. Violating graph: {graph}"
            set_node.replace_all_uses_with(
                Y_input,
                delete_user_cb=lambda n: node_to_idx[n] > set_node_idx
                and node_to_idx[n] < next_set_node_idx,
            )
            set_nodes_to_delete.add(set_node)
        # For any primal input, if we have deleted the last `.set_(primal_X, ...)`,
        # we will re-insert `.set_(primal_X, Y_last)` at the end of the graph.
        last_set_node = node_list[primal_input_to_set_node_idx[primal_input][-1]]
        if last_set_node in set_nodes_to_delete:
            with graph.inserting_before(return_node):
                new_last_set_node = graph.call_function(
                    last_set_node.target, last_set_node.args
                )
    # Replace set_ nodes in graph output with the corresponding primal input.
    new_return_node_args = []
    for arg in return_node.args:
        if arg in set_nodes_to_delete:
            arg = arg.args[0]
            assert arg in primal_inputs
        new_return_node_args.append(arg)
    return_node.args = tuple(new_return_node_args)
    # Finally, delete the old set_ nodes.
    for set_node in set_nodes_to_delete:
        graph.erase_node(set_node)


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
