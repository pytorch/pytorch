from enum import Enum
from typing import List, Optional, Set, Tuple, Union

import torch.fx as fx
from torch.fx.passes.shape_prop import TensorMetadata


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class CommType(str, Enum):
    ALLREDUCE = "allreduce_"
    ALLGATHER = "allgather_"
    BROADCAST = "broadcast_"
    REDUCESCATTER = "reduce_scatter_"
    SCATTER = "scatter_"


comm_block_op_sequence: Tuple[Union[str, Set[CommType]], ...] = (
    "clone",
    "_tensor_constant",
    "_tensor_constant",
    # The supported communication type.
    {CommType.ALLREDUCE},
    "comm_result",
    "getitem",
    "getitem",
    "wait_comm",
)


def get_comm_block_nodes(
    wait_node: fx.Node, comm_type: CommType
) -> Tuple[int, List[fx.Node]]:
    """
    Given a wait_comm node, find out all the nodes belong to this communcation.

    Args:
        wait_node(fx.Node): The target wait_comm node.
        comm_type(CommType): The communication type of this communication block.
            Currently, only allreduce is supported. An exception will be raised
            if other values are passed.
    Returns:
        comm_idx(int): The index to the communication node in the return list.
        node_list(List[fx.Node]): The list that contain the nodes in the order
           of inserting to the graph.
    """
    if not wait_node.name.startswith("wait_comm"):
        raise ValueError(
            "Passing a wait_node that name does not start with ``wait_comm``. "
            f"Name is {wait_node.name}, OP is {wait_node.op}."
        )
    node = wait_node
    node_list = []
    for i, prefix in enumerate(reversed(comm_block_op_sequence)):
        node_list.append(node)
        if isinstance(prefix, set):
            if comm_type not in prefix:
                raise ValueError(f"Not supported CommType {comm_type}")
            prefix = comm_type
            comm_idx = i
        assert node.name.startswith(
            prefix
        ), f"Comm block op sequence mismatches, {node.op} {node.name} {i} {prefix}."
        node = node.prev

    comm_idx = len(node_list) - comm_idx - 1
    node_list.reverse()

    return comm_idx, node_list


def get_node_tensor_metadata(node: fx.Node, is_required: bool = True) -> TensorMetadata:
    metadata = node.meta.get("tensor_meta", None)
    if is_required and metadata is None:
        raise RuntimeError(
            f"Callsite expects that ``tensor_meta`` exists in ``{node.name}``, "
            f"but got None instead. Node: {node.op} {node.name} {node.target}"
        )
    return metadata


def get_output_node(gm: fx.GraphModule) -> Optional[fx.Node]:
    """
    Take a graphmodule and returns the graph output node. We traverse in reverse
    to expedite it, with the idea that last node should be output
    """
    if gm.graph is None:
        raise ValueError("Missing graph from graph module.")

    for node in reversed(gm.graph.nodes):
        if node.op == OP.OUTPUT:
            return node
    return None


def rebuild_graph(gm: fx.GraphModule, remove_dead_code: bool = True) -> None:
    """
    Runs the required steps to ensure production-ready graph.
    note - per the fx docs, eliminate dead code is not very precise.
    Hence, the flag to make this step optional.
    """

    gm.graph.lint()
    if remove_dead_code:
        gm.graph.eliminate_dead_code()
    gm.recompile()
