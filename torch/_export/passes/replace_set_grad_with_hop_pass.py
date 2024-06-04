# mypy: allow-untyped-defs
import contextlib
import copy

import torch
from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled

from ..utils import (
    node_inline_,
    node_replace_,
    nodes_filter,
    nodes_first,
    nodes_map,
    sequential_split,
)


def _is_set_grad_enabled_node(node: torch.fx.Node):
    return (
        node
        and node.op == "call_function"
        and node.target == torch._C._set_grad_enabled
    )


def _is_set_grad_enabled_sub_mod(node: torch.fx.Node, omit_if_same_with_ambient=False):
    if node.op == "call_module":
        assert isinstance(node.target, str)
        subgm = getattr(node.graph.owning_module, node.target)
        first_non_ph = nodes_first(
            subgm.graph.nodes, lambda node: node.op != "placeholder"
        )
        if (
            first_non_ph
            and first_non_ph.op == "call_function"
            and first_non_ph.target == torch._C._set_grad_enabled
        ):
            return (
                first_non_ph.args[0] != torch.is_grad_enabled()
                if omit_if_same_with_ambient
                else True
            )
    return False


def _replace_with_hop(node: torch.fx.Node):
    assert node.op == "call_module"
    graph: torch.fx.Graph = node.graph
    gm: torch.fx.GraphModule = graph.owning_module
    assert isinstance(node.target, str)
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    set_grad_nodes = nodes_filter(sub_graph.nodes, _is_set_grad_enabled_node)
    if len(set_grad_nodes) > 0:
        assert len(set_grad_nodes) == 1
        set_grad_node = set_grad_nodes[0]
        enable_grad_val = set_grad_node.args[0]
        with graph.inserting_before(node):
            get_attr_node = graph.get_attr(node.target)
            get_attr_node.meta["nn_module_stack"] = copy.copy(
                set_grad_node.meta.get("nn_module_stack", {})
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
                if isinstance(output_args, (tuple, list)):
                    call_func_node = graph.call_function(
                        wrap_with_set_grad_enabled,
                        (enable_grad_val, get_attr_node, *node.args),
                        {},
                    )
                    # Create the metadata
                    call_func_node.meta["val"] = tuple(
                        arg.meta["val"] for arg in output_args
                    )
                    call_func_node.meta["nn_module_stack"] = copy.copy(
                        set_grad_node.meta.get("nn_module_stack", {})
                    )
                    call_func_node.meta["torch_fn"] = (
                        f"{wrap_with_set_grad_enabled.__name__}",
                        f"{wrap_with_set_grad_enabled.__class__.__name__}.{wrap_with_set_grad_enabled.__name__}",
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
                        wrap_with_set_grad_enabled,
                        (enable_grad_val, get_attr_node, *node.args),
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
                node.graph.erase_node(node)
        sub_graph.erase_node(set_grad_node)


def _remove_set_grad_and_inline(node: torch.fx.Node):
    assert node.op == "call_module"
    graph: torch.fx.Graph = node.graph
    gm: torch.fx.GraphModule = graph.owning_module
    assert isinstance(node.target, str)
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    nodes_map(
        sub_graph.nodes,
        lambda n: sub_graph.erase_node(n) if _is_set_grad_enabled_node(n) else n,
    )
    node_inline_(node)


def _sequential_split_and_maybe_inline_subgraphs(
    gm: torch.fx.GraphModule, graph_signature
):
    """
    Helper function for replace_set_grad_with_hop_pass().
    Split the graph module into multiple subgraphs based on the set_grad_enabled nodes.
    For each subgraph, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module.
    """
    # If there is no set_grad_enabled node, return the original graph module
    need_replacing = False
    for node in gm.graph.nodes:
        if _is_set_grad_enabled_node(node):
            need_replacing = True

    if need_replacing:
        new_gm = sequential_split(gm, _is_set_grad_enabled_node)

        replace_ctx = contextlib.nullcontext()
        if graph_signature is not None:
            replace_ctx = new_gm._set_replace_hook(graph_signature.get_replace_hook())  # type: ignore[assignment]

        with replace_ctx:

            def _maybe_inline_or_replace_with_hop(node: torch.fx.Node):
                if _is_set_grad_enabled_sub_mod(node, omit_if_same_with_ambient=True):
                    _replace_with_hop(node)
                else:
                    _remove_set_grad_and_inline(node)

            nodes_map(
                list(new_gm.graph.nodes),
                lambda node: (
                    _maybe_inline_or_replace_with_hop(node)
                    if node.op == "call_module"
                    else node
                ),
            )
        new_gm.recompile()
        return new_gm

    return gm


def replace_set_grad_with_hop_pass(gm: torch.fx.GraphModule, graph_signature):
    new_gm = _sequential_split_and_maybe_inline_subgraphs(gm, graph_signature)
    # recursively call
    for node in new_gm.graph.nodes:
        if node.op == "get_attr":
            subgm = getattr(new_gm, node.target)
            if not isinstance(subgm, torch.fx.GraphModule):
                continue
            new_subgm = replace_set_grad_with_hop_pass(subgm, None)
            setattr(new_gm, node.target, new_subgm)

    new_gm.recompile()
    new_gm.graph.lint()
    return new_gm
