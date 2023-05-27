import itertools
from typing import Any, List, OrderedDict, Set
import operator

import torch

from torch.fx.passes.utils.source_matcher_utils import (
    check_subgraphs_connected,
    get_source_partitions,
    SourcePartition,
)

_EQUIVALENT_TYPES: List[Set] = [
    {torch.nn.Conv2d, torch.nn.functional.conv2d},
    {torch.nn.ReLU, torch.nn.functional.relu, torch.nn.functional.relu_},
    {torch.nn.BatchNorm2d, torch.nn.functional.batch_norm},
    {torch.add, operator.add},
]


def _create_equivalent_types_dict():
    _DICT = {}
    for values in _EQUIVALENT_TYPES:
        for v in values:
            _DICT[v] = list(values)
    return _DICT


_EQUIVALET_TYPES_DICT = _create_equivalent_types_dict()


def _partitions_sequential(partitions: List[SourcePartition]):
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
    if partition_type in _EQUIVALET_TYPES_DICT:
        matching_types.extend(_EQUIVALET_TYPES_DICT[partition_type])
    return matching_types


def _valid_type_sequence(partition_types: List[Any]):
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
    partition_types: List[Any],
    include_functional_equivalent=True,
):
    if not _valid_type_sequence(partition_types):
        raise ValueError(
            f"Invalid partition types: {partition_types}. Each type in the sequence must be unique"
        )

    typed_partitions: OrderedDict[Any, List[SourcePartition]] = OrderedDict()
    for partition_type in partition_types:
        types_to_match = _get_matching_types(partition_type)
        partitions = get_source_partitions(gm.graph, types_to_match)
        typed_partitions[partition_type] = list(itertools.chain(*partitions.values()))

    typed_partitions_list = list(typed_partitions.values())
    fusion_candidates = itertools.product(*typed_partitions_list)
    fused_partitions = []
    for candidate in fusion_candidates:
        if _partitions_sequential(candidate):  # type: ignore[arg-type]
            fused_partitions.append(candidate)
    return fused_partitions
