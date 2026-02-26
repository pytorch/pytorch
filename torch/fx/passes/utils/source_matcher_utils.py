import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.node import Node


__all__ = ["get_source_partitions", "check_subgraphs_connected", "SourcePartition"]


# Set`PYTORCH_MATCHER_LOGLEVEL=INFO` to see debug logs
def _init_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)

    level = os.environ.get("PYTORCH_MATCHER_LOGLEVEL", "WARNING").upper()
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
class SourcePartition:
    # Nodes in a particular partition
    nodes: list[Node]

    # The source these nodes decomposed from
    source: Any

    # Nodes in the graph that are needed as inputs to the partition
    # These do not include the params of the partition
    input_nodes: list[Node] = field(default_factory=list)

    # Nodes in the partition that are being used by nodes outside of the
    # partition
    output_nodes: list[Node] = field(default_factory=list)

    # Parameters that are being used
    params: list[Node] = field(default_factory=list)


@compatibility(is_backward_compatible=False)  # type: ignore[misc]
def get_source_partitions(
    graph: Graph,
    wanted_sources: list[Any],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> dict[Any, list[SourcePartition]]:
    """
    Args:
        graph: The graph we want to partition
        wanted_sources: List of sources of nodes that were decomposed from this
            source. This can be a function (ex. torch.nn.functional.linear) or a
            leaf module type (ex. torch.nn.Linear).

    Returns:
        Dictionary mapping sources that were given to a list of SourcePartitions
        that correspond to the list of nodes that were decomposed from the given
        source.
    """
    modules: dict[type, dict[str, list[Node]]] = {}

    def add_to_partition(src: Any, fqn: str, node: Node) -> None:
        diff_modules = modules.setdefault(src, {})
        partition = diff_modules.setdefault(fqn, [])
        partition.append(node)

    for node in graph.nodes:
        # The metadata source_fn should contain a tuple of a unique name for the
        # source, and the source function if the node is decomposed from a
        # function, or the type of module if the node is decomposed from a leaf
        # module

        # TODO: Bypass "torch_fn" when "source_fn_stack" because now "torch_fn" can
        # be different from "source_fn_stack", for example for the add_ node
        # decomposed from batch norm. We should remove the check on "source_fn_stack"
        # after we fix "torch_fn". T199561090
        source_fn_st = node.meta.get("source_fn_stack", None)
        if source_fn_st is None:
            matched = False
            torch_fn = node.meta.get("torch_fn", None)
            if torch_fn is not None:
                node_fqn, source_fn = torch_fn
                source_fn_name = source_fn.split(".")[1]
                if source_fn_name in wanted_sources:
                    add_to_partition(source_fn_name, node_fqn, node)
                    matched = True
            # Fallback: when source_fn_stack is not populated (e.g. strict=False export),
            # use nn_module_stack to resolve the originating module type.
            # Only apply to call_function nodes to avoid incorrectly including
            # placeholder, get_attr, or output nodes in partitions.
            if not matched and node.op == "call_function":
                nn_module_stack = node.meta.get("nn_module_stack", None)
                if nn_module_stack is not None:
                    # Get the innermost module (last entry in the ordered dict)
                    innermost_fqn, innermost_cls = list(nn_module_stack.values())[-1]
                    for src in wanted_sources:
                        if isinstance(src, type):
                            if isinstance(innermost_cls, type) and issubclass(
                                innermost_cls, src
                            ):
                                add_to_partition(src, innermost_fqn, node)
                                break
                            elif isinstance(innermost_cls, str):
                                src_str = src.__module__ + "." + src.__qualname__
                                if innermost_cls == src_str:
                                    add_to_partition(src, innermost_fqn, node)
                                    break
                        elif innermost_cls == src:
                            add_to_partition(src, innermost_fqn, node)
                            break

        if source_fn_st is not None:
            source_fn = source_fn_st[-1]
            if source_fn[1] in wanted_sources:
                add_to_partition(source_fn[1], source_fn[0], node)

    def make_partition(nodes: list[Node], module_type: type) -> SourcePartition:
        input_nodes = set()
        output_nodes = set()
        params = set()
        for node in nodes:
            for arg in node.args:
                if isinstance(arg, Node) and arg not in nodes and arg.op != "get_attr":
                    input_nodes.add(arg)

            if node.op == "get_attr":
                params.add(node)
                # get_attr nodes won't be output nodes
                continue

            for user in node.users:
                if user not in nodes:
                    output_nodes.add(node)

        return SourcePartition(
            nodes,
            module_type,
            list(input_nodes),
            list(output_nodes),
            list(params),  # type: ignore[arg-type]
        )

    ret: dict[type[Any], list[SourcePartition]] = {}

    if filter_fn:
        # for each partition, we apply filter_fn to filter out all partitions that doesn't satisfy the
        # filter condition
        filtered_modules = {}
        for tp, name_to_partition in modules.items():
            filtered_name_to_partition = {
                name: partition
                for name, partition in name_to_partition.items()
                if all(map(filter_fn, partition))
            }
            filtered_modules[tp] = filtered_name_to_partition
        modules = filtered_modules

    for k, v in modules.items():
        ret[k] = [make_partition(partition, k) for partition in v.values()]

    return ret


@compatibility(is_backward_compatible=False)  # type: ignore[misc]
def check_subgraphs_connected(
    subgraph1: SourcePartition, subgraph2: SourcePartition
) -> bool:
    """
    Given two subgraphs A and B (in the form of a list of nodes), checks if
    A has nodes connecting to at least one node in B -- aka there exists a node
    in B that uses a node in A (not the other way around).
    """

    for node in reversed(subgraph1.nodes):
        for user in node.users:
            if user in subgraph2.nodes:
                return True
    return False
