# mypy: allow-untyped-defs
import contextlib
import copy

import torch
from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled

from ..utils import node_inline_, nodes_filter, nodes_first, nodes_map, sequential_split
from .replace_with_hop_pass_util import _replace_with_hop_helper


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
        _replace_with_hop_helper(
            node, set_grad_node, _is_set_grad_enabled_node, wrap_with_set_grad_enabled
        )
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
    need_replacing = any(_is_set_grad_enabled_node(node) for node in gm.graph.nodes)
    if not need_replacing:
        return gm, graph_signature

    # sequential_split returns a new graph module that could have different output
    # args names. We need to fix the graph signature.
    new_gm = sequential_split(gm, _is_set_grad_enabled_node)

    replace_ctx = contextlib.nullcontext()
    new_signature = None
    if graph_signature is not None:
        # Cannot deep copy a real ScriptObject, which is referenced
        # in the FakeScriptObject. Copy should be good enough to guard
        # against accidental mutation to original graph_signature.
        new_signature = copy.copy(graph_signature)
        new_gm_out_node = next(reversed(new_gm.graph.find_nodes(op="output")))
        assert new_gm_out_node.op == "output" and len(new_gm_out_node.args[0]) == len(
            new_signature.output_specs
        )
        for arg_node, out_spec in zip(
            new_gm_out_node.args[0], new_signature.output_specs
        ):
            if arg_node is None:
                assert out_spec.arg.value is None
            elif out_spec.arg.name != arg_node.name:
                out_spec.arg.name = arg_node.name

        replace_ctx = new_gm._set_replace_hook(new_signature.get_replace_hook())  # type: ignore[assignment]

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
    return new_gm, new_signature


def replace_set_grad_with_hop_pass(gm: torch.fx.GraphModule, graph_signature):
    new_gm, new_signature = _sequential_split_and_maybe_inline_subgraphs(
        gm, graph_signature
    )
    # recursively call
    for node in new_gm.graph.nodes:
        if node.op == "get_attr":
            subgm = getattr(new_gm, node.target)
            if not isinstance(subgm, torch.fx.GraphModule):
                continue
            new_subgm, _ = replace_set_grad_with_hop_pass(subgm, None)
            setattr(new_gm, node.target, new_subgm)

    new_gm.recompile()
    new_gm.graph.lint()
    return new_gm, new_signature
