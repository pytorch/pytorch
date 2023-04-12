from dataclasses import dataclass, field
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx._compatibility import compatibility
from typing import Dict, List, Any, Type
import logging
import os


__all__ = ['get_module_partitions', 'check_subgraphs_connected']

# Set`PYTORCH_MATCHER_LOGLEVEL=INFO` to see debug logs
def _init_logger():
    logger = logging.getLogger(__name__)

    level = os.environ.get('PYTORCH_MATCHER_LOGLEVEL', 'WARNING').upper()
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(filename)s > %(message)s")
    console.setFormatter(formatter)
    console.setLevel(level)
    # add the handlers to the logger
    logger.addHandler(console)
    logger.propagate = False
    return logger

logger = _init_logger()


@compatibility(is_backward_compatible=False)
@dataclass
class ModulePartition():
    # Nodes in a particular partition
    nodes: List[Node]

    # Module type
    module_type: Type

    # Nodes in the graph that are needed as inputs to the partition
    input_nodes: List[Node] = field(default_factory=list)

    # Nodes in the partition that are being used by nodes outside of the
    # partition
    output_nodes: List[Node] = field(default_factory=list)

    # Parameters that are being used
    params: List[str] = field(default_factory=list)


@compatibility(is_backward_compatible=False)
def get_module_partitions(
    graph: Graph,
    wanted_module_types: List[Type]
) -> Dict[Type, List[ModulePartition]]:
    """
    Args:
        graph: The graph we want to partition
        wanted_module_types: List of module types of nodes that we want to find
            in the graph. Note that these types have to the types of a leaf
            module

    Returns:
        Dictionary mapping type of module (ex. torch.nn.modules.linear.Linear)
        to a list of ModulePartitions that correspond to the list of nodes that
        were flattened from a module of that type
    """
    modules: Dict[Type, Dict[str, List[Node]]] = {}

    for node in graph.nodes:
        if (nn_module_stack := node.meta.get("nn_module_stack", None)) is None:
            continue

        # First value in the nn_module_stack contains the leaf module trace
        module_call, (_, module_type) = list(nn_module_stack.items())[0]

        if module_type not in wanted_module_types:
            continue

        diff_modules = modules.setdefault(module_type, {})
        partition = diff_modules.setdefault(module_call, [])
        partition.append(node)

    def make_partition(nodes: List[Node], module_type: Type) -> ModulePartition:
        input_nodes = set()
        output_nodes = set()
        params = set()
        for node in nodes:
            for arg in node.args:
                if isinstance(arg, Node) and arg not in nodes:
                    input_nodes.add(arg)
            
            if node.op == "get_attr":
                params.add(node.target)

            for user in node.users.keys():
                if user not in nodes:
                    output_nodes.add(node)

        return ModulePartition(nodes, module_type, list(input_nodes), list(output_nodes), list(params))

    ret: Dict[Type[Any], List[ModulePartition]] = {}
    for k, v in modules.items():
        ret[k] = [make_partition(partition, k) for partition in v.values()]

    return ret


@compatibility(is_backward_compatible=False)
def check_subgraphs_connected(subgraph1: ModulePartition, subgraph2: ModulePartition) -> bool:
    """
    Given two subgraphs A and B (in the form of a list of nodes), checks if
    A has nodes connecting to at least one node in B -- aka there exists a node
    in B that uses a node in A (not the other way around).
    """

    for node in reversed(subgraph1.nodes):
        for user in node.users.keys():
            if user in subgraph2.nodes:
                return True
    return False
