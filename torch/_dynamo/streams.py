from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
from torch._library.custom_ops import custom_op
from torch.fx import Node
from torch.utils._ordered_set import OrderedSet


Tensor = torch.Tensor

# Avoid circular dependency for the dataclass
TensorVariable = Any


# Stream state consists of the fork stream node
# and the external to the stream that are accessed from within the
# stream
@dataclass
class StreamState:
    # the fork node that initiated the creation of this stream state
    # we will finalize it once the stream state is popped
    fork_node: Node
    # Nodes not created within the stream
    external_nodes: OrderedSet[Node]
    # Nodes created within the stream
    internal_nodes: OrderedSet[Node]


@custom_op("streams::fork", mutates_args={"args"})
def fork_stream_(
    index: int, device: torch.device, device_index: int, args: list[Tensor]
) -> None:
    pass


@fork_stream_.register_fake
def _(index: int, device: torch.device, device_index: int, args: list[Tensor]) -> None:
    pass


@custom_op("streams::join", mutates_args={"args"})
def join_stream_(
    index: int, device: torch.device, device_index: int, args: list[Tensor]
) -> None:
    pass


@join_stream_.register_fake
def _(index: int, device: torch.device, device_index: int, args: list[Tensor]) -> None:
    pass


class StreamStateManager:
    def __init__(self) -> None:
        self.state_stack: deque[StreamState] = deque()

    def in_stream_context(self) -> bool:
        return bool(self.state_stack)

    def track_internal_node(self, node: Node) -> None:
        # if we are in a stream context, all created nodes are internal
        if self.in_stream_context():
            # if we have seen the node before, it is an internal
            self._cur_state().internal_nodes.add(node)

    def track_node(self, node: Node) -> None:
        # If we are in a stream context, args of ops may be external
        if self.in_stream_context() and node not in self._internal_nodes():
            self._external_nodes().add(node)

    def push_stream_state(self, node: Node) -> None:
        self.state_stack.append(StreamState(node, OrderedSet(), OrderedSet()))

    def pop_stream_state(self) -> StreamState:
        assert self.state_stack, "No stream state to pop"
        return self.state_stack.pop()

    def _cur_state(self) -> StreamState:
        assert self.state_stack, "No stream state to pop"
        return self.state_stack[-1]

    def _internal_nodes(self) -> OrderedSet[Node]:
        return self._cur_state().internal_nodes

    def _external_nodes(self) -> OrderedSet[Node]:
        return self._cur_state().external_nodes


stream_state_mgr = StreamStateManager()
