# mypy: allow-untyped-defs

import copy

from typing import Callable

import torch

from torch._ops import HigherOrderOperator

from ..utils import node_replace_


def _replace_with_hop_helper(
    node: torch.fx.Node,
    enter_block_node: torch.fx.Node,
    node_filter: Callable,
    wrap_hoo: HigherOrderOperator,
):
    graph: torch.fx.Graph = node.graph
    gm: torch.fx.GraphModule = graph.owning_module
    assert isinstance(node.target, str)
    sub_gm = getattr(gm, node.target)

    with graph.inserting_before(node):
        get_attr_node = graph.get_attr(node.target)
        get_attr_node.meta["nn_module_stack"] = copy.copy(
            enter_block_node.meta.get("nn_module_stack", {})
        )
        output_node = next(iter(reversed(sub_gm.graph.nodes)), None)
        # Split_module pass intentially doesn't add output node
        # if the graph doesn't return anything.
        # TODO (tmanlaibaatar) Figure out if this is right behaviour
        # for split_module
        if isinstance(output_node, torch.fx.Node) and output_node.op != "output":
            output_node = None
        if output_node is not None:
            assert len(output_node.args) == 1
            output_args = output_node.args[0]
            enter_block_node_args = enter_block_node.args
            if isinstance(output_args, (tuple, list)):
                call_func_node = graph.call_function(
                    wrap_hoo,
                    (*enter_block_node_args, get_attr_node, *node.args),
                    {},
                )
                # Create the metadata
                call_func_node.meta["val"] = tuple(
                    arg.meta["val"] for arg in output_args
                )
                call_func_node.meta["nn_module_stack"] = copy.copy(
                    enter_block_node.meta.get("nn_module_stack", {})
                )
                call_func_node.meta["torch_fn"] = (
                    f"{wrap_hoo.__name__}",
                    f"{wrap_hoo.__class__.__name__}.{wrap_hoo.__name__}",
                )
                node_replace_(node, call_func_node, delete_old=True)

                # Rename the name of getitem nodes to the actual name of its contents
                # for passing verifier and better readability, also propagate metadata
                for get_item_node in call_func_node.users.keys():
                    idx: int = get_item_node.args[1]
                    output_node = output_args[idx]
                    get_item_node._rename(output_node.name)
                    get_item_node.meta = output_node.meta
                    pass

            elif isinstance(output_args, torch.fx.Node):
                call_func_node = graph.create_node(
                    "call_function",
                    wrap_hoo,
                    (*enter_block_node_args, get_attr_node, *node.args),
                    {},
                    output_args.name,
                )
                call_func_node.meta = output_args.meta
                node_replace_(node, call_func_node, delete_old=True)
            else:
                raise NotImplementedError(
                    f"repalce_set_grad_with_hop_pass doesnt' support output type {type(output_args)}"
                )
        else:
            # TODO (shangdiy): remove this line, since the export graph can be non-functional
            node.graph.erase_node(node)
