import torch
from .utils import flatten_arg_list, is_node_used_by_ops

def remove_noop_consecutive_transposes(mod):
    """
    We support detecting two kinds of no-op transposes:

    1. .permute([0, 1]) followed by .permute([0, 1])
    2. .t() followed by .t()
    """
    node_list = list(mod.graph.nodes)
    node_to_idx = {}
    for i, n in enumerate(node_list):
        node_to_idx[n] = i
    first_node = None
    second_node = None
    for i, n in enumerate(node_list):
        if (
            n.target is torch.ops.aten.permute.default
            and n.args[1] == [0, 1]
            and n.args[0].target is torch.ops.aten.permute.default
            and n.args[0].args[1] == [0, 1]
            # Only do the removal if the first permute output has no other use before second permute
            and not is_node_used_by_ops(n.args[0], node_list[(node_to_idx[n.args[0]]+1):(node_to_idx[n])])
        ):
            first_node = n.args[0]
            second_node = n
        elif (
            n.target is torch.ops.aten.t.default
            and n.args[0].target is torch.ops.aten.t.default
            # Only do the removal if the first permute output has no other use before second permute
            and not is_node_used_by_ops(n.args[0], node_list[(node_to_idx[n.args[0]]+1):(node_to_idx[n])])
        ):
            first_node = n.args[0]
            second_node = n
        if first_node is not None and second_node is not None:
            second_node.replace_all_uses_with(first_node.args[0])
            mod.graph.erase_node(second_node)
            mod.graph.erase_node(first_node)
    mod.graph.lint()
    mod.recompile()


def remove_noop_views(mod):
    """
    X: "f32[512, 1024]" = ...
    view_1: "f32[512, 1024]" = torch.ops.aten.view.default(X, [512, 1024])
    (... uses view_1)

    ->

    (... uses X)
    """
    node_list = list(mod.graph.nodes)
    for i, n in enumerate(node_list):
        if n.target is torch.ops.aten.view.default:
            view_node = n
            view_input = view_node.args[0]
            if list(view_input.meta.get("tensor_meta").shape) == view_node.args[1]:
                view_node.replace_all_uses_with(view_input)
                mod.graph.erase_node(view_node)
    mod.graph.lint()
    mod.recompile()
