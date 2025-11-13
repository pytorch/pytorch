from typing import Optional, TypeAlias

import torch.fx
import torch.fx.traceback
from torch._dynamo.graph_utils import _get_flat_args


Node: TypeAlias = torch.fx.Node


def is_gradient_acc(node: Node) -> bool:
    return node.meta.get("is_gradient_acc", False)


def get_device(node: Node) -> torch.device:
    return node.meta["val"].device


def get_stream(node: Node) -> Optional[int]:
    maybe_annotation = node.meta.get("custom", None)
    if maybe_annotation is not None:
        return node.meta["custom"].get("stream", None)
    else:
        return None


def set_stream(node: Node, ind: int) -> None:
    if "custom" in node.meta:
        node.meta["custom"].update({"stream": ind})
    else:
        node.meta["custom"] = {"stream": ind}


def assign_backward_streams(gm: torch.fx.GraphModule) -> None:
    """Assigns backward streams to gradient accumulation nodes"""

    for node in gm.graph.nodes:
        if is_gradient_acc(node):
            # Accumulation stream selection. Follow the rules from top to bottom to determine the accumulation stream:
            # 1. Match first stream assignment encountered in the args from left to right
            # 2. Match first stream assignment of the first user
            gradients = _get_flat_args(node, {})
            users = list(node.users.keys())

            # All gradients will be on same device, they will be coerced if they were not with a .to() node
            for neighbor in gradients + users:
                ind = get_stream(neighbor)
                if ind is not None:
                    set_stream(node, ind)
                    break
