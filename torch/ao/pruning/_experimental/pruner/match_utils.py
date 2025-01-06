"""
Contains utility functions to check if a pattern is in the graph and return the matching nodes
"""
from typing import Any, Optional, Union

import torch
from torch import nn
from torch.ao.quantization.utils import MatchAllNode
from torch.fx import Node
from torch.nn.utils import parametrize


def _match(
    modules: dict[str, nn.ModuleDict],
    node: Node,
    current: Union[nn.Module, Any],
) -> bool:
    r"""
    checks to see if a single node of a pattern matches
    """
    if isinstance(current, type) and issubclass(current, MatchAllNode):
        return True
    if not isinstance(node, Node):
        return False
    if isinstance(current, type) and issubclass(current, torch.nn.Module):
        return (
            node.op == "call_module"
            and parametrize.type_before_parametrizations(modules[node.target])  # type: ignore[index]
            == current
        )
    elif callable(current):
        return node.op == "call_function" and node.target is current
    elif isinstance(current, str):
        return node.target == current
    return False


def apply_match(
    modules: dict[str, nn.ModuleDict],
    pattern: Union[tuple[Any], Any],
    node: Node,
    matched_node_pattern: list[Node],
) -> Optional[list[Node]]:
    r"""
    This function will return the matched nodes if the pattern matches the node given
    If there is no match, it will return None
    """
    if isinstance(pattern, tuple):
        if len(pattern) == 1:
            if _match(modules, node, pattern[0]):
                return matched_node_pattern + [node]

        first, *rest = pattern
        if _match(modules, node, first):
            if rest is None:
                return matched_node_pattern + [node]

            for user in node.users:
                return apply_match(
                    modules, tuple(rest), user, matched_node_pattern + [node]
                )
    elif _match(modules, node, pattern):
        return [node]
    return None
