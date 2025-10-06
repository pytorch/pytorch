from collections import defaultdict
from typing import Optional

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet


class AugmentedGraphHelper:
    """
    Graph helper that augments the original graph with additional
    dependencies and uses, plus tracks node equivalences for coalescing.
    TODO: if this becomes too large of compile time, consider binding
    graphcycles.cc
    """

    def __init__(
        self, graph: fx.Graph, node_to_idx: Optional[dict[fx.Node, int]] = None
    ):
        # Each node starts in its own singleton set
        self.graph = graph
        self.merge_sets = {node: OrderedSet([node]) for node in graph.nodes}
        # Extra dependencies: node depends on dep (dep must come before node)
        self.extra_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        # Optional node to index mapping for bounded searches
        self.node_to_idx = node_to_idx

    def add_extra_dep(self, *, n: fx.Node, dep: fx.Node) -> None:
        """Add extra dependency: node depends on dep."""
        self.extra_deps[n].add(dep)

    def merge_to_set(self, existing_node: fx.Node, new_node: fx.Node) -> None:
        """
        Merge new_node into existing_node's set. The new node must be a singleton set.
        """
        existing_set = self.merge_sets[existing_node]
        new_set = self.merge_sets[new_node]
        assert len(new_set) == 1
        # Add all nodes from new_set to existing_set
        existing_set.update(new_set)
        # Update all nodes from new_set to point to existing_set
        for node in new_set:
            self.merge_sets[node] = existing_set

    def unmerge_node(self, node: fx.Node) -> None:
        """Remove a node from its merge set, making it singleton."""
        old_set = self.merge_sets[node]
        # If already singleton, nothing to do
        if len(old_set) == 1:
            return
        # Remove from old set
        old_set.remove(node)
        # Make node singleton
        self.merge_sets[node] = OrderedSet([node])

    def get_merged_deps(self, node: fx.Node) -> OrderedSet[fx.Node]:
        """
        Get all dependencies of a node considering merges and extra deps.
        Combines:
        1. Direct deps (all_input_nodes) of node and its merge equivalents
        2. Extra deps of node and its merge equivalents
        """
        deps: OrderedSet[fx.Node] = OrderedSet()
        # For each node in the merge set
        for merged_node in self.merge_sets[node]:
            # Add direct dependencies from all_input_nodes
            deps.update(merged_node.all_input_nodes)
            # Add extra dependencies
            deps.update(self.extra_deps[merged_node])
        return deps

    def has_cycle(self) -> bool:
        merged_deps = {n: self.get_merged_deps(n) for n in self.graph.nodes}
        return torch._dynamo.graph_deduplication._has_cycle(self.graph, merged_deps)

    def has_path(
        self,
        source: fx.Node,
        target: fx.Node,
        bounded_search_range: Optional[tuple[int, int]] = None,
    ) -> bool:
        """
        Check if there's a path from source to target.

        If bounds are provided, only searches nodes within the idx of these ranges.
        """
        # we should not be checking path from node to itself
        assert self.merge_sets[source] is not self.merge_sets[target]

        # search backwards from target to source
        visited: OrderedSet[fx.Node] = OrderedSet()
        queue = [target]
        visited.add(target)

        while queue:
            current = queue.pop()

            # Get all dependencies
            for dep in self.get_merged_deps(current):
                # If using bounds, skip nodes outside the range
                if bounded_search_range is not None and self.node_to_idx is not None:
                    min_idx, max_idx = bounded_search_range
                    dep_idx = self.node_to_idx.get(dep)
                    assert dep_idx is not None
                    if dep_idx < min_idx or dep_idx > max_idx:
                        continue

                # Check if we reached source or its equivalent
                if dep in self.merge_sets[source]:
                    return True

                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)

        return False
