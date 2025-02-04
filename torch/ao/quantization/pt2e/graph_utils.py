# mypy: allow-untyped-defs
import itertools
import operator
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import torch
from torch.export import ExportedProgram
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import (
    check_subgraphs_connected,
    get_source_partitions,
    SourcePartition,
)


__all__ = [
    "find_sequential_partitions",
    "get_equivalent_types",
    "update_equivalent_types_dict",
    "bfs_trace_with_node_process",
]

_EQUIVALENT_TYPES: list[set] = [
    {torch.nn.Conv1d, torch.nn.functional.conv1d},
    {torch.nn.Conv2d, torch.nn.functional.conv2d},
    {torch.nn.AdaptiveAvgPool2d, torch.nn.functional.adaptive_avg_pool2d},
    {torch.nn.ReLU, torch.nn.functional.relu, torch.nn.functional.relu_},
    {torch.nn.BatchNorm2d, torch.nn.functional.batch_norm},
    {torch.nn.Hardtanh, torch.nn.functional.hardtanh, torch.nn.functional.hardtanh_},
    {torch.add, operator.add, operator.iadd, "add", "add_"},
    {torch.mul, operator.mul, operator.imul, "mul", "mul_"},
]


def _create_equivalent_types_dict():
    _DICT = {}
    for values in _EQUIVALENT_TYPES:
        for v in values:
            _DICT[v] = list(values)
    return _DICT


_EQUIVALENT_TYPES_DICT = _create_equivalent_types_dict()


def get_equivalent_types() -> list[set]:
    return _EQUIVALENT_TYPES


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


def _partitions_sequential(partitions: Sequence[SourcePartition]):
    prev_partition = None
    for partition in partitions:
        if prev_partition is not None and not check_subgraphs_connected(
            prev_partition, partition
        ):
            return False
        prev_partition = partition
    return True


def _get_matching_types(partition_type):
    matching_types = [partition_type]
    if partition_type in _EQUIVALENT_TYPES_DICT:
        matching_types.extend(_EQUIVALENT_TYPES_DICT[partition_type])
    return matching_types


def _valid_type_sequence(partition_types: list[Any]):
    partition_types_set = set()  # type: ignore[var-annotated]
    for partition_type in partition_types:
        matching_types = _get_matching_types(partition_type)
        matching_types_set = set(matching_types)
        if len(partition_types_set & matching_types_set) > 0:
            return False
        partition_types_set |= matching_types_set
    return True


def find_sequential_partitions(
    gm: torch.fx.GraphModule,
    partition_types: list[Any],
    include_functional_equivalent=True,
    filter_fn: Optional[Callable[[Node], bool]] = None,
):
    if not _valid_type_sequence(partition_types):
        raise ValueError(
            f"Invalid partition types: {partition_types}. Each type in the sequence must be unique"
        )

    typed_partitions: OrderedDict[Any, list[SourcePartition]] = OrderedDict()
    for partition_type in partition_types:
        types_to_match = _get_matching_types(partition_type)
        partitions = get_source_partitions(gm.graph, types_to_match, filter_fn)
        typed_partitions[partition_type] = list(
            itertools.chain.from_iterable(partitions.values())
        )

    typed_partitions_list = list(typed_partitions.values())
    fusion_candidates = itertools.product(*typed_partitions_list)
    fused_partitions = [
        candidate
        for candidate in fusion_candidates
        if _partitions_sequential(candidate)
    ]
    return fused_partitions


def _get_submodule(
    graph_module: torch.fx.GraphModule, node: torch.fx.Node, arg_index: int
) -> tuple[str, torch.nn.Module, torch.fx.Node]:
    submod_node = node.args[arg_index]
    assert isinstance(submod_node, torch.fx.Node)
    assert submod_node.op == "get_attr"
    assert isinstance(submod_node.target, str)
    submodule = graph_module.get_submodule(submod_node.target)
    # pyre-ignore
    return submod_node.target, submodule, node


def _get_control_flow_submodules(
    graph_module: torch.fx.GraphModule,
) -> list[tuple[str, torch.nn.Module, torch.fx.Node]]:
    """
    Returns a list of submodules used for control flow operations
    (torch.ops.higher_order.cond/map) that are in the given toplevel graph (does not look
    into submodules). Specifically, the returned value is a list containing a
    tuple of (name of the submodule that's stored in the graph module, the
    submodule itself, and the fx node that uses this submodule).
    """
    control_flow_submodules = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue

        if node.target is torch.ops.higher_order.cond:
            control_flow_submodules.append(_get_submodule(graph_module, node, 1))
            control_flow_submodules.append(_get_submodule(graph_module, node, 2))
        if node.target is torch.ops.higher_order.map_impl:
            control_flow_submodules.append(_get_submodule(graph_module, node, 0))

    return control_flow_submodules


def bfs_trace_with_node_process(
    model: Union[ExportedProgram, torch.fx.GraphModule], node_op: Callable
) -> None:
    """Traverse the graph module and apply node_op to each node."""

    assert isinstance(
        model, (ExportedProgram, torch.fx.GraphModule)
    ), f"Expected GraphModule or ExportedProgram, got {type(model)}"
    gm = model.graph_module if isinstance(model, ExportedProgram) else model
    queue = [gm]
    while queue:
        current_graph_module = queue.pop(0)
        for node in current_graph_module.graph.nodes:
            if node.op in ["output", "placeholder"]:
                continue

            node_op(node)

        control_flow_submodules = [
            submodule
            for _, submodule, _ in _get_control_flow_submodules(current_graph_module)
        ]
        queue.extend(control_flow_submodules)
