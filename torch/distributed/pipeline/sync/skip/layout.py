# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Static skip connection layout of ``@skippable`` modules."""
from typing import Dict, Iterable, List, Tuple

from torch import nn

from .namespace import Namespace

__all__: List[str] = []


class SkipLayout:
    """Represents a skip connection layout across partitions."""

    # Skip routes indexed by 'ns, name': {(ns, name): (prev_j, next_j), ...}
    by_ns_name: Dict[Tuple[Namespace, str], Tuple[int, int]]

    # Skip routes indexed by partition number 'j': [[next_j]: [(prev_j, ns, name), ...], ...]
    by_partition: List[List[Tuple[int, Namespace, str]]]

    def __init__(self, num_partitions: int, skip_routes: Dict[Tuple[Namespace, str], Tuple[int, int]],) -> None:
        # The skip routes are already indexed by 'ns, name'.
        self.by_ns_name = skip_routes

        # Index skip routes by partition number 'j'.
        self.by_partition = [[] for _ in range(num_partitions)]

        for (ns, name), (prev_j, next_j) in skip_routes.items():
            self.by_partition[next_j].append((prev_j, ns, name))

        for p in self.by_partition:
            p.sort()

    def copy_policy(self, next_j: int) -> Iterable[Tuple[int, Namespace, str]]:
        """Generates skip routes for the given destination partition number.
        The skip routes are sorted by source partition number in ascending
        order.

        Yields:
            Each tuple of (source partition number, namespace, name).

        """
        for prev_j, ns, name in self.by_partition[next_j]:
            if prev_j == next_j:
                # This skip tensor will be popped at the same partition where
                # it is stashed. In this case, copy is not required.
                continue

            yield (prev_j, ns, name)

    def requires_copy(self, ns: Namespace, name: str) -> bool:
        """Whether the given namespace and name requires partition-to-partition
        copy or not.
        """
        prev_j, next_j = self.by_ns_name.get((ns, name), (-1, -1))
        return prev_j != next_j


def inspect_skip_layout(partitions: List[nn.Sequential]) -> SkipLayout:
    """Inspects the skip connection layout in the given partitions."""
    # NOTE(sublee): Hide circular import inside this subroutine. Circular
    # import is not ideal but placing this logic near to SkipLayout may
    # increase cohesion of code.
    from .skippable import Skippable

    skip_routes: Dict[Tuple[Namespace, str], Tuple[int, int]] = {}
    stashed_at: Dict[Tuple[Namespace, str], int] = {}

    for j, partition in enumerate(partitions):
        def inspect_layer(layer):
            if not isinstance(layer, Skippable):
                return

            for ns, name in layer.stashable():
                stashed_at[(ns, name)] = j

            for ns, name in layer.poppable():
                prev_j = stashed_at.pop((ns, name))
                skip_routes[(ns, name)] = (prev_j, j)

        if isinstance(partition, nn.Sequential):
            for layer in partition:
                inspect_layer(layer)
        else:
            inspect_layer(partition)

    return SkipLayout(len(partitions), skip_routes)
