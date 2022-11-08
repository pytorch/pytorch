import torch
from torch import nn
from torch.ao.quantization.fx.match_utils import (
    MatchAllNode,
)
from torch.fx import Node
from torch.nn.utils import parametrize
from typing import Any, Dict, List, Callable, Optional, Tuple, Type, Union, Set

def match(modules, node: Node, current: nn.Module) -> bool:
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
    else:
        return False

def apply_match(
    modules,
    pattern: Union[Tuple[nn.Module], nn.Module],
    node: Node,
    matched_node_pattern: List[Node],
) -> Optional[List[Node]]:
    if isinstance(pattern, tuple):
        if len(pattern) == 1:
            if match(modules, node, pattern[0]):
                return matched_node_pattern + [node]

        s, *args = pattern
        if match(modules, node, s):
            if args is None:
                return matched_node_pattern + [node]

            for user in node.users:
                return apply_match(
                    modules, tuple(args), user, matched_node_pattern + [node]
                )
    elif match(modules, node, pattern):
        return [node]
    else:
        return None
