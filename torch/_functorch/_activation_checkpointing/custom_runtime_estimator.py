# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom runtime estimator and knapsack solver interfaces for AOTAutograd partitioner.

This module provides abstract base classes that enable cache-friendly callable support
for the activation_memory_budget_runtime_estimator and activation_memory_budget_solver
configs in AOTAutograd partitioner, following the CustomGraphPass pattern from Inductor.
"""

import hashlib
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, BinaryIO, Optional, TYPE_CHECKING, TypeAlias, Union

import torch.fx as fx


if TYPE_CHECKING:
    from ..partitioners import NodeInfo


class CustomRuntimeEstimator(ABC):
    """
    Implement this interface for custom runtime estimators:

    1) The __call__() method contains the implementation of the runtime estimation.

    2) The uuid() method enables AOTAutograd to cache compiled graphs when your custom
    runtime estimator is used. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom runtime estimators would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom runtime estimator
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.

    ** IMPORTANT ** If your custom runtime estimator's behavior depends on some external state,
    then you'll need to implement something more complicated (or disable caching).

    EXAMPLE:

    class MyCustomRuntimeEstimator(CustomRuntimeEstimator):
        def __call__(self, node: fx.Node) -> float:
            # my custom runtime estimation logic
            #     ...
            return estimated_runtime

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """

    @abstractmethod
    def __call__(self, node: fx.Node) -> float:
        """
        Implementation of the custom runtime estimator.

        Args:
            node: An fx.Node object whose runtime is to be estimated.

        Returns:
            float: The estimated runtime for the node.
        """

    @abstractmethod
    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom runtime estimator implementation.
        Return None to skip AOTAutograd caching entirely.
        """


class CustomKnapsackSolver(ABC):
    """
    Implement this interface for custom knapsack solvers:

    1) The __call__() method contains the implementation of the knapsack solver.

    2) The uuid() method enables AOTAutograd to cache compiled graphs when your custom
    knapsack solver is used. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom knapsack solvers would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom knapsack solver
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.

    ** IMPORTANT ** If your custom knapsack solver's behavior depends on some external state,
    then you'll need to implement something more complicated (or disable caching).

    EXAMPLE:

    class MyCustomKnapsackSolver(CustomKnapsackSolver):
        def __call__(
            self,
            memory: list[float],
            joint_graph: fx.Graph,
            max_memory: float,
            node_info: NodeInfo,
            all_recomputable_banned_nodes: list[fx.Node],
        ) -> tuple[list[int], list[int]]:
            # my custom knapsack solver logic
            #     ...
            return saved_node_idx, recomp_node_idx

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """

    @abstractmethod
    def __call__(
        self,
        memory: list[float],
        joint_graph: fx.Graph,
        max_memory: float,
        node_info: "NodeInfo",
        all_recomputable_banned_nodes: list[fx.Node],
    ) -> tuple[list[int], list[int]]:
        """
        Implementation of the custom knapsack solver.

        Args:
            memory: List of memory costs for each node.
            joint_graph: The fx.Graph representing the joint forward-backward graph.
            max_memory: Maximum memory budget.
            node_info: NodeInfo object containing node classification information.
            all_recomputable_banned_nodes: List of nodes that were banned from recomputation.

        Returns:
            tuple[list[int], list[int]]: A tuple of (saved_node_idx, recomp_node_idx)
                where saved_node_idx contains indices of nodes to save and
                recomp_node_idx contains indices of nodes to recompute.
        """

    @abstractmethod
    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom knapsack solver implementation.
        Return None to skip AOTAutograd caching entirely.
        """


# Type aliases for config values
CustomRuntimeEstimatorType: TypeAlias = Optional[Union[CustomRuntimeEstimator, str]]

CustomKnapsackSolverType: TypeAlias = Optional[Union[CustomKnapsackSolver, str]]


def _hash_file(f: BinaryIO, hasher: "hashlib._Hash") -> None:
    """
    Update a hasher with the contents of a file.

    Note: We don't use hashlib.file_digest here because it creates its own hasher
    internally and returns a digest, which doesn't support our use case of
    accumulating multiple files into a single hasher.
    """
    hasher.update(f.read())


@lru_cache(1)
def get_hash_for_files(paths: tuple[str, ...], extra: str = "") -> bytes:
    """
    Helper to compute a unique string by hashing the contents of a list of files.
    This is useful for generating stable cache keys based on source file contents.
    Any change to the files will result in a new hash value.
    Args:
        paths: Tuple of file paths to hash.
        extra: Optional extra string to include in the hash.
    Returns:
        bytes: SHA256 hash digest of the file contents.
    """
    hasher = hashlib.sha256()
    hasher.update(extra.encode("utf-8"))
    for path in paths:
        with open(path, "rb") as f:
            _hash_file(f, hasher)
    return hasher.digest()
