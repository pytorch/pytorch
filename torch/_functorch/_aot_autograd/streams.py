import torch.fx
from torch._dynamo.graph_utils import _get_flat_args

from .utils import _is_backward_node_with_seq_nr, _is_forward_node_with_seq_nr


Node = torch.fx.Node


def seq_number(node: Node) -> int:
    assert "seq_nr" in node.meta, "No seq nr in seq_number call"
    return node.meta.get("seq_nr")  # type: ignore[return-type]


def assign_backward_streams(gm: torch.fx.GraphModule) -> None:
    """Assigns backward streams to gradient accumulation nodes"""

    max_fw_seq_nr = -1
    max_bw_seq_nr = -1
    bw_nodes = []
    for node in gm.graph.nodes:
        if _is_forward_node_with_seq_nr(node):
            max_fw_seq_nr = max(max_fw_seq_nr, seq_number(node))
        elif _is_backward_node_with_seq_nr(node):
            bw_nodes.append(node)
            max_bw_seq_nr = max(max_bw_seq_nr, seq_number(node))

    if max_bw_seq_nr > max_fw_seq_nr:
        # in this case, there are some gradient accumulation nodes
        # these nodes will need stream assignments
        for node in bw_nodes:
            if seq_number(node) == max_bw_seq_nr:
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


def insert_sync(producer, consumer) -> None:
    pass
