import logging
import os
import tempfile
from enum import Enum
from typing import Callable, cast, Dict, Iterable, List, Set

import torch.fx as fx
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten


logger: logging.Logger = logging.getLogger("graph_utils")


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


def get_node_tensor_metadata(node: fx.Node, is_required: bool = True) -> TensorMetadata:
    metadata = node.meta.get("tensor_meta", None)
    if is_required and metadata is None:
        raise RuntimeError(
            f"Callsite expects that ``tensor_meta`` exists in ``{node.name}``, "
            f"but got None instead. Node: {node.op} {node.name} {node.target}"
        )
    return metadata


def get_output(graph: fx.Graph) -> fx.Node:
    """Take a graphmodule and return the graph output node.

    We traverse in reverse to expedite it, with the idea that last node should be output
    """
    for node in reversed(graph.nodes):
        if node.op == OP.OUTPUT:
            return node
    raise RuntimeError(f"Cannot find the output node in {graph}")


def find_node(
    graph: fx.Graph, predicate: Callable, reverse_order: bool = False
) -> List[fx.Node]:
    """Take a predicate and return all the nodes in the `graph` where the predicate holds."""
    nodes = cast(Iterable[fx.Node], graph.nodes)
    if reverse_order:
        nodes = cast(Iterable[fx.Node], iter(reversed(nodes)))  # type: ignore[call-overload]
    return [node for node in nodes if predicate(node)]


def is_leaf_subgraph(graph: fx.Graph, subgraph: List[fx.Node]) -> bool:
    """Ensure nodes in ``subgraph`` satisfy one of the following rules.

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
    """Clone the given subgraph and insert it before ``target``.

    This API currently does not support inserting after ``target``.
    """
    all_nodes = set(subgraph)
    mapping: Dict[fx.Node, fx.Node] = {}
    cloned_subgraph = []
    with graph.inserting_before(target):
        for node in subgraph:
            cloned_node = graph.call_function(
                node.target, node.args, node.kwargs, node.type  # type: ignore[arg-type]
            )
            # TODO: there are many flatten/unflatten in IterGraph that
            # can be simplified with tree_map. Will simplify this in
            # a follow-up PR.
            original_input = pytree.arg_tree_leaves(*node.args, **node.kwargs)
            cloned_input, spec = tree_flatten((cloned_node.args, cloned_node.kwargs))
            mapped_cloned_input = []
            for original_input_node, cloned_input_node in zip(
                original_input, cloned_input
            ):
                if (
                    isinstance(original_input_node, fx.Node)
                    and original_input_node in all_nodes
                ):
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
    """Run the required steps to ensure production-ready graph.

    Note - per the fx docs, elimination of dead code is not very precise.
    Hence, the flag to make this step optional.
    """
    gm.graph.lint()
    if remove_dead_code:
        gm.graph.eliminate_dead_code()
    gm.recompile()


def dump_graphs_to_files(graphs: Dict[str, fx.GraphModule], folder: str = "") -> str:
    if not folder:
        folder = tempfile.mkdtemp()

    for prefix, gm in graphs.items():
        with open(os.path.join(folder, f"{prefix}.graph"), "w") as fp:
            fp.write(str(gm))

    logger.warning("Dump graphs to %s", folder)

    return folder
