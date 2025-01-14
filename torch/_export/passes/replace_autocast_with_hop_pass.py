# mypy: allow-untyped-defs
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING, Union

import torch
from torch._higher_order_ops.wrap import wrap_with_autocast

from ..utils import node_inline_, nodes_filter, nodes_first, sequential_split
from .replace_with_hop_pass_util import (
    _replace_with_hop_helper,
    _replace_with_hop_pass_helper,
    _sequential_split_and_maybe_inline_subgraphs_helper,
)


if TYPE_CHECKING:
    from torch.export.graph_signature import ExportGraphSignature


def _is_autocast_node(node: torch.fx.Node) -> Union[torch.fx.Node, bool]:
    return (
        node
        and node.op == "call_function"
        and node.target
        in [
            torch.amp.autocast_mode._enter_autocast,
            torch.amp.autocast_mode._exit_autocast,
        ]
    )


def _is_enter_autocast_node(node: torch.fx.Node) -> Union[torch.fx.Node, bool]:
    return (
        node
        and node.op == "call_function"
        and node.target == torch.amp.autocast_mode._enter_autocast
    )


def _is_exit_autocast_node(node: torch.fx.Node) -> Union[torch.fx.Node, bool]:
    return (
        node
        and node.op == "call_function"
        and node.target == torch.amp.autocast_mode._exit_autocast
    )


def _is_autocast_sub_mod(node: torch.fx.Node) -> bool:
    """
    Check if the first non-placeholder node is `torch.amp.autocast_mode._enter_autocast`.
    """
    if node.op == "call_module":
        assert isinstance(node.target, str)
        subgm = getattr(node.graph.owning_module, node.target)
        first_non_ph = nodes_first(
            subgm.graph.nodes, lambda node: node.op != "placeholder"
        )
        if (
            first_non_ph
            and first_non_ph.op == "call_function"
            and first_non_ph.target == torch.amp.autocast_mode._enter_autocast
        ):
            # TODO: check if current auto-cast type is the same as the args of
            # _enter_autocast. If so, return False, i.e. do not create a submodule.
            return True
    return False


def _check_valid_autocast_block(
    enter_autocast_node: torch.fx.Node, exit_autocast_node: torch.fx.Node
) -> None:
    assert _is_enter_autocast_node(enter_autocast_node)
    assert _is_exit_autocast_node(exit_autocast_node)
    assert exit_autocast_node.args[0] == enter_autocast_node


def _replace_with_hop(node: torch.fx.Node) -> None:
    assert node.op == "call_module"
    graph: torch.fx.Graph = node.graph
    assert graph.owning_module is not None
    gm: torch.fx.GraphModule = graph.owning_module
    assert isinstance(node.target, str)
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    autocast_nodes = nodes_filter(sub_graph.nodes, _is_autocast_node)
    if len(autocast_nodes) > 0:
        assert len(autocast_nodes) > 1  # need at least an enter node and an exist node
        enter_autocast_node = autocast_nodes[0]
        exit_autocast_node = autocast_nodes[-1]
        _check_valid_autocast_block(enter_autocast_node, exit_autocast_node)

        _replace_with_hop_helper(node, enter_autocast_node, wrap_with_autocast)
        sub_graph.erase_node(exit_autocast_node)
        sub_graph.erase_node(enter_autocast_node)


def _split_autocast(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    split_autocast creates a new graph module that splits the input graph module into multiple submodules
    based on the `_enter_autocast` and `_exit_autocast` nodes. It doesn't mutate the input graph module.

    Nodes between the **outer-most** `_enter_autocast` and `_exit_autocast(_enter_autocast)` are splitted
    into a submodule. Nested autocast regions are not splitted.
    `_enter_autocast` and `_exit_autocast(_enter_autocast)` nodes are in the submodule as well.

    Below is an example of splitting. A, B, C, D, E are blocks of non-autocast nodes in the original graph
    module. Nodes marked with the same number are grouped into the same submodule.
    A               # 0
    enter_autocast  # 1
    B               # 1
    exit_autocast   # 1
    C               # 2
    enter_autocast  # 3
    D               # 3
    exit_autocast   # 3
    E               # 4
    """
    enter_autocast_node_stack: List[torch.fx.Node] = []
    first_node_after_outer_most_exit: bool = False

    def node_call_back(node: torch.fx.Node) -> bool:
        nonlocal enter_autocast_node_stack, first_node_after_outer_most_exit
        increment_id = False
        if first_node_after_outer_most_exit or (
            len(enter_autocast_node_stack) == 0 and _is_enter_autocast_node(node)
        ):
            assert len(enter_autocast_node_stack) == 0
            first_node_after_outer_most_exit = False
            increment_id = True
        if _is_enter_autocast_node(node):
            enter_autocast_node_stack.append(node)
        elif _is_exit_autocast_node(node):
            assert len(enter_autocast_node_stack) > 0
            last_enter_autocast_node = enter_autocast_node_stack.pop()
            assert node.args[0] == last_enter_autocast_node
            if len(enter_autocast_node_stack) == 0:
                # next node should be in the next submodule since
                # autocast block ends
                first_node_after_outer_most_exit = True
        return increment_id

    return sequential_split(gm, node_call_back)


def _sequential_split_and_maybe_inline_subgraphs(
    gm: torch.fx.GraphModule, graph_signature: Optional[ExportGraphSignature]
) -> tuple[torch.fx.GraphModule, Optional[ExportGraphSignature]]:
    """
    Helper function for replace_autocast_with_hop_pass().
    Split the graph module into multiple subgraphs based on the autocast nodes.
    For each subgraph, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module.
    Nodes between `_enter_autocast` and `_exit_autocast(_enter_autocast)` are considered
    as a subgraph.
    """
    need_replacing = any(_is_autocast_node(node) for node in gm.graph.nodes)
    if not need_replacing:
        return gm, graph_signature

    # split_autocast returns a new graph module that could have different output
    # args names. We need to fix the graph signature in `_sequential_split_and_maybe_inline_subgraphs_helper`.
    new_gm = _split_autocast(gm)

    def _maybe_inline_or_replace_with_hop(node: torch.fx.Node) -> None:
        if _is_autocast_sub_mod(node):
            _replace_with_hop(node)
        else:
            assert node.op == "call_module"
            assert isinstance(node.target, str)
            node_inline_(node)

    return _sequential_split_and_maybe_inline_subgraphs_helper(
        new_gm, graph_signature, _maybe_inline_or_replace_with_hop
    )


def replace_autocast_with_hop_pass(
    gm: torch.fx.GraphModule, graph_signature: Optional[ExportGraphSignature]
) -> tuple[torch.fx.GraphModule, Optional[ExportGraphSignature]]:
    """
    Split gm into sub-graph-modules using `sequential_split_and_maybe_inline_subgraphs`, and
    then recursively call itself on each of the submodules.
    """
    return _replace_with_hop_pass_helper(
        gm,
        graph_signature,
        _sequential_split_and_maybe_inline_subgraphs,
    )
