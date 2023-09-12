import itertools
import logging
import operator
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Set, Type

import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.node import Node


__all__ = ['get_source_partitions', 'check_subgraphs_connected', 'SourcePartition']

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
class SourcePartition:
    # Nodes in a particular partition
    nodes: List[Node]

    # The source these nodes decomposed from
    source: Any

    # Nodes in the graph that are needed as inputs to the partition
    input_nodes: List[Node] = field(default_factory=list)

    # Nodes in the partition that are being used by nodes outside of the
    # partition
    output_nodes: List[Node] = field(default_factory=list)

    # Parameters that are being used
    params: List[Node] = field(default_factory=list)


@compatibility(is_backward_compatible=False)
def get_source_partitions(
    graph: Graph,
    wanted_sources: List[Any],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Dict[Any, List[SourcePartition]]:
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
    modules: Dict[Type, Dict[str, List[Node]]] = {}

    for node in graph.nodes:
        # The metadata source_fn should contain a tuple of a unique name for the
        # source, and the source function if the node is decomposed from a
        # function, or the type of module if the node is decomposed from a leaf
        # module

        if (source_fn := node.meta.get("source_fn", None)) is None:
            continue

        if source_fn[1] not in wanted_sources:
            continue

        diff_modules = modules.setdefault(source_fn[1], {})
        partition = diff_modules.setdefault(source_fn[0], [])
        partition.append(node)

    def make_partition(nodes: List[Node], module_type: Type) -> SourcePartition:
        input_nodes = set()
        output_nodes = set()
        params = set()
        for node in nodes:
            for arg in node.args:
                if isinstance(arg, Node) and arg not in nodes:
                    input_nodes.add(arg)

            if node.op == "get_attr":
                params.add(node)

            for user in node.users.keys():
                if user not in nodes:
                    output_nodes.add(node)

        return SourcePartition(
            nodes,
            module_type,
            list(input_nodes),
            list(output_nodes),
            list(params),  # type: ignore[arg-type]
        )

    ret: Dict[Type[Any], List[SourcePartition]] = {}

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


@compatibility(is_backward_compatible=False)
def check_subgraphs_connected(subgraph1: SourcePartition, subgraph2: SourcePartition) -> bool:
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

_EQUIVALENT_TYPES: List[Set] = [
    {torch.nn.Conv2d, torch.nn.functional.conv2d},
    {torch.nn.AdaptiveAvgPool2d, torch.nn.functional.adaptive_avg_pool2d},
    {torch.nn.ReLU, torch.nn.functional.relu, torch.nn.functional.relu_},
    {torch.nn.BatchNorm2d, torch.nn.functional.batch_norm},
    {torch.nn.Hardtanh, torch.nn.functional.hardtanh, torch.nn.functional.hardtanh_},
    {torch.add, operator.add, operator.iadd, "add", "add_"},
    {torch.mul, operator.mul, operator.imul},
]


@compatibility(is_backward_compatible=False)
def _create_equivalent_types_dict():
    _DICT = {}
    for values in _EQUIVALENT_TYPES:
        for v in values:
            _DICT[v] = list(values)
    return _DICT


_EQUIVALENT_TYPES_DICT = _create_equivalent_types_dict()

@compatibility(is_backward_compatible=False)
def get_equivalent_types() -> List[Set]:
    return _EQUIVALENT_TYPES

@compatibility(is_backward_compatible=False)
def update_equivalent_types_dict(customized_equivalent_types=None):
    """Help function for user who wants to customize the _EQUIVALENT_TYPES and _EQUIVALENT_TYPES_DICT.
    When customized_equivalent_types passes in,
    re-generate _EQUIVALENT_TYPES and _EQUIVALENT_TYPES_DICT.
    """
    if customized_equivalent_types is None:
        raise ValueError("customized_equivalent_types should not be None")
    global _EQUIVALENT_TYPES
    global _EQUIVALENT_TYPES_DICT
    _EQUIVALENT_TYPES = customized_equivalent_types
    _EQUIVALENT_TYPES_DICT = _create_equivalent_types_dict()

@compatibility(is_backward_compatible=False)
def _partitions_sequential(partitions: List[SourcePartition]) -> bool:
    prev_partition = None
    for partition in partitions:
        if prev_partition is not None and not check_subgraphs_connected(
            prev_partition, partition
        ):
            return False
        prev_partition = partition
    return True


@compatibility(is_backward_compatible=False)
def _get_matching_types(partition_type) -> List[Any]:
    matching_types = [partition_type]
    if partition_type in _EQUIVALENT_TYPES_DICT:
        matching_types.extend(_EQUIVALENT_TYPES_DICT[partition_type])
    return matching_types


@compatibility(is_backward_compatible=False)
def _valid_type_sequence(partition_types: List[Any]) -> bool:
    partition_types_set = set()  # type: ignore[var-annotated]
    for partition_type in partition_types:
        matching_types = _get_matching_types(partition_type)
        matching_types_set = set(matching_types)
        if len(partition_types_set & matching_types_set) > 0:
            return False
        partition_types_set |= matching_types_set
    return True


@compatibility(is_backward_compatible=False)
def find_sequential_partitions(
    gm: torch.fx.GraphModule,
    partition_types: List[Any],
    include_functional_equivalent=True,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[SourcePartition]:
    if not _valid_type_sequence(partition_types):
        raise ValueError(
            f"Invalid partition types: {partition_types}. Each type in the sequence must be unique"
        )

    typed_partitions: OrderedDict[Any, List[SourcePartition]] = OrderedDict()
    for partition_type in partition_types:
        types_to_match = _get_matching_types(partition_type)
        partitions = get_source_partitions(gm.graph, types_to_match, filter_fn)
        typed_partitions[partition_type] = list(itertools.chain(*partitions.values()))

    typed_partitions_list = list(typed_partitions.values())
    fusion_candidates = itertools.product(*typed_partitions_list)
    fused_partitions = []
    for candidate in fusion_candidates:
        if _partitions_sequential(candidate):  # type: ignore[arg-type]
            fused_partitions.append(candidate)
    return fused_partitions
