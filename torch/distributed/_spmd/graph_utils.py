from enum import Enum
from typing import Dict, List, Set, Tuple, Union

import torch.fx as fx
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils._pytree import tree_flatten, tree_unflatten


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
    Given a wait_comm node, find out all the nodes belong to this communication.

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


def get_output(graph: fx.Graph) -> fx.Node:
    """
    Take a graphmodule and returns the graph output node. We traverse in reverse
    to expedite it, with the idea that last node should be output
    """
    for node in reversed(graph.nodes):
        if node.op == OP.OUTPUT:
            return node
    raise RuntimeError(f"Cannot find the output node in {graph}")


def is_leaf_subgraph(graph: fx.Graph, subgraph: List[fx.Node]) -> bool:
    """
    This function ensures nodes in ``subgraph`` satisfy one of the rules:
    1. The user of the node is in ``subgraph``.
    2. The user of the node is output.
    3. There are no users -- the node is a side-effect node.
    """
    all_nodes: Set[fx.Node] = set(subgraph)
    output = get_output(graph)
    for node in subgraph:
        for user in node.users:
            if not isinstance(user, fx.Node):
                continue
            if user not in all_nodes and user != output:
                return False
    return True


def clone_subgraph(
    graph: fx.Graph, subgraph: List[fx.Node], target: fx.Node
) -> List[fx.Node]:
    """
    Clone the given subgraph and insert it before ``target``.
    This API currently does not support inserting after ``target``.
    """

    all_nodes = set(subgraph)
    mapping: Dict[fx.Node, fx.Node] = dict()
    cloned_subgraph = []
    with graph.inserting_before(target):
        for node in subgraph:
            cloned_node = graph.call_function(
                node.target, node.args, node.kwargs, node.type
            )
            # TODO: there are many flatten/unflatten in IterGraph that
            # can be simplified with tree_map. Will simplify this in
            # a follow-up PR.
            original_input, _ = tree_flatten((node.args, node.kwargs))
            cloned_input, spec = tree_flatten((cloned_node.args, cloned_node.kwargs))
            mapped_cloned_input = []
            for original_input_node, cloned_input_node in zip(
                original_input, cloned_input
            ):
                if original_input_node in all_nodes:
                    assert original_input_node in mapping
                    mapped_cloned_input.append(mapping[original_input_node])
                else:
                    mapped_cloned_input.append(cloned_input_node)
            cloned_node.args, cloned_node.kwargs = tree_unflatten(
                mapped_cloned_input, spec
            )
            mapping[node] = cloned_node
            cloned_subgraph.append(cloned_node)

    return cloned_subgraph


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
