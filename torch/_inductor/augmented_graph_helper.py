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
        self,
        graph: fx.Graph,
        node_ancestors: Optional[dict[fx.Node, OrderedSet[fx.Node]]] = None,
    ):
        # Each node starts in its own singleton set
        self.graph = graph
        self.merge_sets = {node: OrderedSet([node]) for node in graph.nodes}

        # Extra dependencies: node depends on dep (dep must come before node)
        self.extra_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        # Extra uses: reverse of extra_deps (node is used by user)
        self.extra_uses: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        # Note: only reflect original ancestors, not maintained through additional deps
        # or merge sets
        self.node_ancestors = node_ancestors

    def add_extra_dep(self, *, n: fx.Node, dep: fx.Node) -> None:
        """Add extra dependency: node depends on dep."""
        self.extra_deps[n].add(dep)
        self.extra_uses[dep].add(n)

    def remove_extra_dep(self, *, n: fx.Node, dep: fx.Node) -> None:
        if dep in self.extra_deps[n]:
            self.extra_deps[n].discard(dep)
            self.extra_uses[dep].discard(n)

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

    def has_path(self, source: fx.Node, target: fx.Node) -> bool:
        """Check if there's a path from source to target."""
        # we should not be checking path from node to itself
        assert self.merge_sets[source] is not self.merge_sets[target]

        # search backwards from target to source
        visited: OrderedSet[fx.Node] = OrderedSet()
        queue = [target]
        visited.add(target)

        while queue:
            current = queue.pop()

            for dep in self.get_merged_deps(current):
                # Check if we reached source or its equivalent
                if dep in self.merge_sets[source]:
                    return True

                if dep in visited:
                    continue

                # We are searching from target, so this node is necessarily an ancestor
                # of target.
                # If dep is an ancestor of source, any path through dep to source would imply a cycle
                if self.node_ancestors:
                    source_set = self.merge_sets[source]
                    is_ancestor_of_source = any(
                        dep in self.node_ancestors[s] for s in source_set
                    )
                    # Add to visited to avoid recomputing this check if we see dep again
                    if is_ancestor_of_source:
                        visited.add(dep)
                        continue

                visited.add(dep)
                queue.append(dep)

        return False

    def transfer_erased_node_deps(self, erased_to_new: dict[fx.Node, fx.Node]) -> None:
        """
        Transfer all extra dependencies from erased nodes to their replacements, handling
        cross-dependencies between erased nodes correctly.
        """
        erased_merge_sets: dict[fx.Node, fx.Node] = {}

        for replaced, new in erased_to_new.items():
            for equiv in self.merge_sets[replaced]:
                erased_merge_sets[equiv] = new

        # Transfer dependencies
        for old_node, new_node in erased_merge_sets.items():
            # Transfer dependencies FROM old_node (what old_node depended on)
            for extra_dep in self.extra_deps[old_node]:
                # Redirect if dep is also being erased
                updated_dep = erased_merge_sets.get(extra_dep, extra_dep)
                self.extra_deps[new_node].add(updated_dep)
                self.extra_uses[updated_dep].discard(old_node)
                self.extra_uses[updated_dep].add(new_node)

            # Transfer dependencies TO old_node (what depended on old_node)
            for extra_use in self.extra_uses[old_node]:
                # Redirect if this user is also being erased
                updated_use = erased_merge_sets.get(extra_use, extra_use)

                # Update the user's deps to point to new_node
                self.extra_deps[updated_use].discard(old_node)
                self.extra_deps[updated_use].add(new_node)
                self.extra_uses[new_node].add(updated_use)

        # Clean up erased nodes
        for old_node in erased_merge_sets:
            self.extra_deps[old_node].clear()
            self.extra_uses[old_node].clear()
            del self.merge_sets[old_node]

    def get_all_extra_deps(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """
        Get all extra dependencies in a format suitable for topological sort.
        Returns a copy to avoid external modifications.
        """
        return {
            node: OrderedSet(deps)
            for node, deps in self.extra_deps.items()
            if deps  # Only include nodes with non-empty deps
        }
