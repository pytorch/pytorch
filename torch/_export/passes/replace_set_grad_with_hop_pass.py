import functools

import torch
from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled

from ..utils import (
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
            return True
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
        enable_grad = set_grad_node.args[0]
        with graph.inserting_before(node):
            get_attr_node = graph.get_attr(node.target)
            call_func_node = graph.call_function(
                wrap_with_set_grad_enabled, (enable_grad, get_attr_node, *node.args)
            )
        node_replace_(node, call_func_node, delete_old=True)


def replace_set_grad_with_hop_pass(gm: torch.fx.GraphModule):
    new_gm = sequential_split(gm, _is_set_grad_enabled_node)
    call_module_nodes = nodes_filter(
        list(new_gm.graph.nodes),
        functools.partial(_is_set_grad_enabled_sub_mod, omit_if_same_with_ambient=True),
    )
    nodes_map(call_module_nodes, _replace_with_hop)
    return new_gm
