import torch
from torch import nn
from torch.ao.quantization.fx.match_utils import (
    MatchAllNode,
)
from torch.fx import Node
from torch.nn.utils import parametrize
from typing import Any, Dict, List, Optional, Tuple, Union

def match(modules: Dict[str, nn.ModuleDict], node: Node, current: nn.Module) -> bool:
    if isinstance(current, type) and issubclass(current, MatchAllNode):
        return True
    if not isinstance(node, Node):
        return False
    if isinstance(current, type) and issubclass(current, torch.nn.Module):
        return (
            node.op == "call_module"
            and parametrize.type_before_parametrizations(modules[node.target])
            == current
        )
    elif callable(current):
        return node.op == "call_function" and node.target is current
    elif isinstance(current, str):
        return node.target == current
    return False

def apply_match(
    modules: Dict[str, nn.ModuleDict],
    pattern: Union[Tuple[Any], Any],
    node: Node,
    matched_node_pattern: List[Node],
) -> Optional[List[Node]]:
    if isinstance(pattern, tuple):
        if len(pattern) == 1:
            if match(modules, node, pattern[0]):
                return matched_node_pattern + [node]

        first, *rest = pattern
        if match(modules, node, first):
            if rest is None:
                return matched_node_pattern + [node]

            for user in node.users:
                return apply_match(
                    modules, tuple(rest), user, matched_node_pattern + [node]
                )
    elif match(modules, node, pattern):
        return [node]
    return None
