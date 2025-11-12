from typing import Optional, TypeAlias

import torch.fx
import torch.fx.traceback
from torch._dynamo.graph_utils import _get_flat_args
from torch._dynamo.variables.streams import get_current_stream


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
    annotations = node.meta.get("custom", {})
    annotations.update({"stream": ind})
    torch.fx.traceback.annotate(annotations)


def assign_backward_streams(gm: torch.fx.GraphModule) -> None:
    """Assigns backward streams to gradient accumulation nodes"""

    for node in gm.graph.nodes:
        if is_gradient_acc(node):
            # Accumulation stream selection. Follow the rules from top to bottom to determine the accumulation stream:
            # 1. If the device of the gradient is the same as the device of the consumer,
            # then the accumulation stream is the consumer node's stream.
            # 2. If the device of the gradient matches the device of the producer,
            # then accumulation stream is the producer node's stream.
            # 3. If neither is true, pick the current stream of the device of the gradient.
            # Accumulation stream synchronization:
            # Prior to accumulation, have the accumulation stream wait for producer stream
            # and the stashed event (recorded on the previous producer stream).
            gradients = _get_flat_args(node, {})
            users = list(node.users.keys())
            assert len(users) == 1, (
                "There should only be one user of the accumulated gradients"
            )
            user = users[0]
            consumer_device = get_device(user)
            # TODO mlazos: is the assumption that all gradients are on same device correct?
            # they are about to be added together after all ..
            gradient_device = get_device(gradients[0])

            if consumer_device == gradient_device:
                stream_ind = get_stream(user)
                if not stream_ind:
                    stream_ind = get_current_stream(consumer_device)
                set_stream(node, stream_ind)
            # TODO: not sure how to get "producer device"
            else:
                pass
