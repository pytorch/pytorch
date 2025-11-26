from typing import Optional, TypeAlias

import torch.fx
import torch.fx.traceback
from torch._dynamo.graph_utils import _get_flat_args
from torch._dynamo.variables.streams import get_current_stream, new_event


Node: TypeAlias = torch.fx.Node
Graph: TypeAlias = torch.fx.Graph


def is_gradient_acc(node: Node) -> bool:
    return node.meta.get("is_gradient_acc", False)


def is_bwd_node(node: Node) -> bool:
    return node.meta.get("partitioner_tag") == "is_backward"


def get_device(node: Node) -> torch.device:
    return node.meta["val"].device


def get_stream(node: Node) -> Optional[int]:
    maybe_annotation = node.meta.get("custom", None)
    if maybe_annotation is not None:
        return node.meta["custom"].get("stream", None)
    else:
        return None


def get_stream_or_current_stream(node: Node) -> int:
    ind = get_stream(node)
    if ind is None:
        ind = get_current_stream(get_device(node))
    return ind


def set_stream(node: Node, ind: int) -> None:
    if "custom" in node.meta:
        node.meta["custom"].update({"stream": ind})
    else:
        node.meta["custom"] = {"stream": ind}


def insert_record_event_after_node(graph: Graph, node: Node, event_ind: int) -> None:
    with graph.inserting_after(node):
        node = graph.call_function(
            torch.ops.streams.record_event.default,
            (
                event_ind,
                get_stream_or_current_stream(node),
            ),
        )
        node.meta["partitioner_tag"] = "must_be_in_backward"


def insert_wait_event_before_node(graph: Graph, node: Node, event_ind: int) -> None:
    with graph.inserting_before(node):
        node = graph.call_function(
            torch.ops.streams.wait_event.default,
            (
                event_ind,
                get_stream_or_current_stream(node),
            ),
        )
        node.meta["partitioner_tag"] = "must_be_in_backward"


def insert_sync(
    graph: Graph,
    consumer: Node,
    producer: Node,
    node_to_wait_event_ind: dict[Node, int],
) -> None:
    if producer not in node_to_wait_event_ind:
        node_to_wait_event_ind[producer] = new_event()

        insert_record_event_after_node(
            graph, producer, node_to_wait_event_ind[producer]
        )
        insert_wait_event_before_node(graph, consumer, node_to_wait_event_ind[producer])


def assign_backward_streams(gm: torch.fx.GraphModule) -> None:
    """Assigns backward streams to gradient accumulation nodes"""

    # NB: iterate in reverse order to more closely match eager
    # the user node stream will be populated first
    for node in reversed(list(gm.graph.nodes)):
        if is_gradient_acc(node):
            # Accumulation stream selection. Follow the rules from top to bottom to determine the accumulation stream:
            # 1. Match first stream assignment of the first user with a stream
            # 2. Match first stream assignment encountered in the args from left to right
            # This differs from eager in some cases:
            # Specifically the eager code uses the autograd node to determine the stream,
            # crucially this does not necessarily correspond to the FX graph node. For example,
            # in the backward for an add node with a constant we will passthrough and during backward tracing,
            # no op will be added to the FX graph, so our stream assignment will differ in this case.
            gradients = _get_flat_args(node, {})
            users = list(node.users.keys())

            # All gradients will be on same device, they will be coerced if they were not with a .to() node
            for neighbor in users + gradients:
                ind = get_stream(neighbor)
                if ind is not None:
                    set_stream(node, ind)
                    break


def insert_backward_syncs(gm: torch.fx.GraphModule) -> None:
    """Inserts stream syncs for backward nodes if consumer and producer are on different streams"""
    node_to_wait_event_ind = {}
    for node in gm.graph.nodes:
        if is_bwd_node(node):
            flat_args = _get_flat_args(node, {})
            cur_node_stream = get_stream(node)

            for arg in flat_args:
                if is_bwd_node(arg):
                    arg_stream = get_stream(arg)
                    if arg_stream != cur_node_stream and get_device(arg).type != "cpu":
                        insert_sync(gm.graph, node, arg, node_to_wait_event_ind)
