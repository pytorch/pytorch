"""Pattern Safety Checker: validates mutation and aliasing safety in pattern transformations."""

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional

import torch
import torch.fx
from torch._inductor.fx_passes.control_dependencies import control_deps
from torch._inductor.fx_passes.memory_estimator import GraphAliasTracker
from torch.utils._ordered_set import OrderedSet


def detect_escaping_tensors(graph: torch.fx.Graph) -> OrderedSet[torch.fx.Node]:
    """
    Find all tensors observable to caller (inputs + outputs).

    Escaping tensors are visible to the caller, so we must preserve their
    values and aliasing relationships.

    Returns:
        Set of nodes that escape (inputs and outputs)
    """
    escaping: OrderedSet[torch.fx.Node] = OrderedSet()

    for node in graph.nodes:
        if node.op == "placeholder":
            escaping.add(node)
        elif node.op == "output":
            if node.args:
                output_vals = node.args[0]
                if isinstance(output_vals, (tuple, list)):
                    for val in output_vals:
                        if isinstance(val, torch.fx.Node):
                            escaping.add(val)
                elif isinstance(output_vals, torch.fx.Node):
                    escaping.add(output_vals)

    return escaping


@dataclass
class MutationInfo:
    """Normalized mutation representation for all node types"""

    mutation_node: torch.fx.Node
    mutated_targets: list[torch.fx.Node]
    mutated_storages: OrderedSet[Any]


class MutationDetector:
    def __init__(self, alias_tracker: GraphAliasTracker) -> None:
        self.alias_tracker = alias_tracker
        self.detectors = [self._detect_mutation, self._detect_control_deps]

    def detect_all(
        self,
        nodes: list[torch.fx.Node],
        node_map: Optional[dict[torch.fx.Node, torch.fx.Node]] = None,
    ) -> list[MutationInfo]:
        mutations = []
        for node in nodes:
            for detector in self.detectors:
                info = detector(node, node_map)
                if info:
                    mutations.append(info)
                    break
        return mutations

    def _detect_mutation(
        self,
        node: torch.fx.Node,
        node_map: Optional[dict[torch.fx.Node, torch.fx.Node]] = None,
    ) -> Optional[MutationInfo]:
        from torch._inductor.pattern_matcher import get_mutated_args

        targets = get_mutated_args(node)
        if not targets:
            return None
        if node_map:
            targets = [node_map[t] for t in targets if t in node_map]

        storages = OrderedSet(
            [s for t in targets for s in self.alias_tracker.node_to_output_storages[t]]
        )

        return MutationInfo(node, targets, storages)

    def _detect_control_deps(
        self,
        node: torch.fx.Node,
        node_map: Optional[dict[torch.fx.Node, torch.fx.Node]] = None,
    ) -> Optional[MutationInfo]:
        if node.target != control_deps:
            return None

        num_mutated = node.meta.get("num_mutated_args", 0)
        if num_mutated == 0:
            return None

        deps_tuple = node.args[0]
        if not isinstance(deps_tuple, (list, tuple)):
            return None

        targets = list(deps_tuple[:num_mutated])

        if node_map:
            targets = [node_map[t] for t in targets if t in node_map]

        storages = OrderedSet(
            [s for t in targets for s in self.alias_tracker.node_to_output_storages[t]]
        )
        return MutationInfo(node, targets, storages)


class NodeExecutionTracker:
    """Tracks execution order of nodes in the graph."""

    def __init__(self, graph: torch.fx.Graph):
        self._node_to_order = {node: i for i, node in enumerate(graph.nodes)}

    def comes_after(self, node1: torch.fx.Node, node2: torch.fx.Node) -> bool:
        """Returns True if node1 comes after node2 in execution order."""
        return self._node_to_order.get(node1, -1) > self._node_to_order.get(node2, -1)

    def get_last_node(self, nodes: list[torch.fx.Node]) -> torch.fx.Node:
        """Returns the node that executes last among the given nodes."""
        return max(nodes, key=lambda n: self._node_to_order.get(n, -1))


class PatternSafetyChecker:
    """
    Validates safety of pattern replacements in FX graphs.
    """

    def __init__(self, graph: torch.fx.Graph) -> None:
        self.graph = graph
        self.alias_tracker: Optional[GraphAliasTracker] = None
        self.escaping_nodes: Optional[OrderedSet[torch.fx.Node]] = None
        self.execution_tracker: Optional[NodeExecutionTracker] = None
        self.mutation_detector: Optional[MutationDetector] = None
        self.full_graph_mutations: Optional[list[MutationInfo]] = None
        self._escaping_storages: OrderedSet[Any] = OrderedSet()
        self._before_mut_escaping: OrderedSet[Any] = OrderedSet()
        self._tracker_invalid = True

    def _build_tracker(self) -> None:
        self.alias_tracker = GraphAliasTracker(list(self.graph.nodes))
        self.escaping_nodes = detect_escaping_tensors(self.graph)
        self.execution_tracker = NodeExecutionTracker(self.graph)
        self.mutation_detector = MutationDetector(self.alias_tracker)
        self.full_graph_mutations = self.mutation_detector.detect_all(
            list(self.graph.nodes)
        )
        self._all_mutation_storages = OrderedSet(
            [s for m in self.full_graph_mutations for s in m.mutated_storages]
        )
        self._escaping_storages = OrderedSet(
            [
                s
                for n in self.escaping_nodes
                for s in self.alias_tracker.node_to_output_storages[n]
            ]
        )
        self._before_mut_escaping = (
            self._all_mutation_storages & self._escaping_storages
        )
        self._tracker_invalid = False

    def _rebuild_if_needed(self) -> None:
        if self._tracker_invalid:
            self._build_tracker()

    def invalidate(self) -> None:
        self._tracker_invalid = True

    def check_escaping_mutation_invariance(
        self,
        match_nodes: list[torch.fx.Node],
        replacement_graph: torch.fx.GraphModule,
        args: list[torch.fx.Node],
        escaping_nodes: OrderedSet[torch.fx.Node],
        replacement_mutations: list[MutationInfo],
    ) -> bool:
        """
        Check that replacement doesn't change mutation state of escaping nodes.
        """
        assert self.full_graph_mutations is not None
        match_set = OrderedSet(match_nodes)
        non_pattern_mutations = [
            m for m in self.full_graph_mutations if m.mutation_node not in match_set
        ]

        after_mut_storages = OrderedSet(
            [s for m in non_pattern_mutations for s in m.mutated_storages]
        )
        after_mut_storages.update(
            s for m in replacement_mutations for s in m.mutated_storages
        )
        after_mut_escaping = after_mut_storages & self._escaping_storages

        if (after_mut_escaping - self._before_mut_escaping) or (
            self._before_mut_escaping - after_mut_escaping
        ):
            return False
        return True

    def check_subsequent_use_safety(
        self,
        replacement_mutations: list[MutationInfo],
        match_nodes: list[torch.fx.Node],
    ) -> bool:
        """Check if replacement mutates tensors with subsequent uses"""
        assert self.execution_tracker is not None
        assert self.alias_tracker is not None
        assert self.full_graph_mutations is not None

        last_pattern_node = self.execution_tracker.get_last_node(match_nodes)

        target_storage_pairs = [
            (target, storage)
            for mut_info in replacement_mutations
            for target in mut_info.mutated_targets
            for storage in self.alias_tracker.node_to_output_storages[target]
        ]

        for target, storage in target_storage_pairs:
            users = self.alias_tracker.storage_to_uses[storage]  # type: ignore[union-attr]
            # Filter users that come after last_pattern_node
            subsequent_users = []
            for u in users:
                if self.execution_tracker.comes_after(  # type: ignore[union-attr]
                    u, last_pattern_node
                ):
                    subsequent_users.append(u)

            for user in subsequent_users:
                # check if use already expects mutated value in original code
                target_storages = self.alias_tracker.node_to_output_storages[target]  # type: ignore[union-attr]
                use_already_mutated = False

                for mut_info in self.full_graph_mutations:  # type: ignore[union-attr]
                    if self.execution_tracker.comes_after(user, mut_info.mutation_node):  # type: ignore[union-attr]
                        if mut_info.mutated_storages & target_storages:
                            use_already_mutated = True
                            break

                if not use_already_mutated:
                    return False
        return True

    def check_escaping_aliasing_invariance(
        self,
        match_nodes: list[torch.fx.Node],
        replacement_graph: torch.fx.GraphModule,
        args: list[torch.fx.Node],
    ) -> bool:
        """
        Check if aliasing changes are safe.
        """
        assert self.alias_tracker is not None
        assert self.escaping_nodes is not None
        assert self.full_graph_mutations is not None

        observable = OrderedSet(args)
        match_set = OrderedSet(match_nodes)
        for node in match_nodes:
            if any(user not in match_set for user in node.users):
                observable.add(node)
        pattern_escaping = observable

        if len(pattern_escaping) < 2:
            return True

        pattern_aliasing = self._build_aliasing_map(
            pattern_escaping, self.alias_tracker
        )
        replacement_tracker = GraphAliasTracker(list(replacement_graph.graph.nodes))

        replacement_nodes, pattern_to_repl = self._map_pattern_to_replacement(
            replacement_graph, args, pattern_escaping
        )
        replacement_aliasing = self._build_aliasing_map(
            replacement_nodes, replacement_tracker
        )

        # Find nodes with external mutations (outside pattern)
        non_pattern_mutations = [
            m for m in self.full_graph_mutations if m.mutation_node not in match_set
        ]
        external_mutation_targets: OrderedSet[torch.fx.Node] = OrderedSet()
        for mut_info in non_pattern_mutations:
            external_mutation_targets.update(mut_info.mutated_targets)

        # Check each pair
        for t1, t2 in combinations(pattern_escaping, 2):
            # Only check pairs where at least one is globally escaping
            t1_is_escaping = t1 in self.escaping_nodes
            t2_is_escaping = t2 in self.escaping_nodes

            if not t1_is_escaping and not t2_is_escaping:
                continue

            repl_t1 = pattern_to_repl.get(t1)
            repl_t2 = pattern_to_repl.get(t2)

            if repl_t1 is None or repl_t2 is None:
                continue

            pattern_aliased = t2 in pattern_aliasing[t1]
            repl_aliased = repl_t2 in replacement_aliasing[repl_t1]

            # Rule 1: always block removed aliasing
            if pattern_aliased and not repl_aliased:
                return False

            # Rule 2 & 3: only compute if new aliasing detected
            if not pattern_aliased and repl_aliased:
                # Rule 2: block new aliasing between two globally escaping nodes
                if t1_is_escaping and t2_is_escaping:
                    return False

                # Rule 3: block new aliasing if it interacts with external mutations
                if t1 in external_mutation_targets or t2 in external_mutation_targets:
                    return False

        return True

    def _build_aliasing_map(
        self, nodes: OrderedSet[torch.fx.Node], alias_tracker: GraphAliasTracker
    ) -> dict[torch.fx.Node, OrderedSet[torch.fx.Node]]:
        """Build map showing which nodes share storage (alias each other)"""

        node_to_storages = {
            node: alias_tracker.node_to_output_storages[node] for node in nodes
        }

        aliasing_map = {
            node: OrderedSet(
                [
                    other
                    for other in nodes
                    if node_to_storages[node] & node_to_storages[other]
                ]
            )
            for node in nodes
        }

        # Special handling for control_deps nodes:
        for node in nodes:
            if node.op == "call_function" and node.target == control_deps:
                mutated_tuple = node.args[0]
                if isinstance(mutated_tuple, (list, tuple)):
                    for mutated_tensor in mutated_tuple:
                        if (
                            isinstance(mutated_tensor, torch.fx.Node)
                            and mutated_tensor in nodes
                        ):
                            aliasing_map[node].add(mutated_tensor)
                            aliasing_map[mutated_tensor].add(node)

        return aliasing_map

    def _map_pattern_to_replacement(
        self,
        replacement_graph: torch.fx.GraphModule,
        args: list[torch.fx.Node],
        pattern_escaping: OrderedSet[torch.fx.Node],
    ) -> tuple[OrderedSet[torch.fx.Node], dict[torch.fx.Node, torch.fx.Node]]:
        """Map pattern nodes to their replacement equivalents"""
        placeholders = [
            n for n in replacement_graph.graph.nodes if n.op == "placeholder"
        ]

        replacement_nodes: OrderedSet[torch.fx.Node] = OrderedSet()
        pattern_to_repl: dict[torch.fx.Node, torch.fx.Node] = {}

        # map pattern inputs to replacement placeholders
        for node in pattern_escaping:
            if node in args:
                idx = args.index(node)
                if idx < len(placeholders):
                    repl_node = placeholders[idx]
                    replacement_nodes.add(repl_node)
                    pattern_to_repl[node] = repl_node

        # get replacement outputs
        for repl_node in replacement_graph.graph.nodes:
            if repl_node.op == "output" and repl_node.args:
                output_arg = repl_node.args[0]

                if isinstance(output_arg, (list, tuple)):
                    repl_outputs = [
                        o for o in output_arg if isinstance(o, torch.fx.Node)
                    ]
                elif isinstance(output_arg, torch.fx.Node):
                    repl_outputs = [output_arg]
                else:
                    repl_outputs = []

                # get pattern outputs (escaping nodes that are not inputs)
                pattern_outputs = [n for n in pattern_escaping if n not in args]

                # map each pattern output to corresponding replacement output
                for pattern_out in pattern_outputs:
                    # find which arg(s) this pattern output mutates/aliases
                    assoc_indices = self._find_all_associated_arg_indices(
                        pattern_out, args
                    )

                    if assoc_indices:
                        # use the first associated index to map to replacement output
                        assoc_idx = assoc_indices[0]
                        if assoc_idx < len(repl_outputs):
                            replacement_nodes.add(repl_outputs[assoc_idx])
                            pattern_to_repl[pattern_out] = repl_outputs[assoc_idx]
                    else:
                        # positional mapping
                        idx = pattern_outputs.index(pattern_out)
                        if idx < len(repl_outputs):
                            replacement_nodes.add(repl_outputs[idx])
                            pattern_to_repl[pattern_out] = repl_outputs[idx]

        return replacement_nodes, pattern_to_repl

    def _find_all_associated_arg_indices(
        self, node: torch.fx.Node, args: list[torch.fx.Node]
    ) -> list[int]:
        """
        Find all argument indices a node is associated with.
        """
        assert self.alias_tracker is not None
        result_indices = []
        for idx, arg in enumerate(args):
            arg_storages = self.alias_tracker.node_to_output_storages[arg]
            if self.alias_tracker.node_to_output_storages[node] & arg_storages:
                result_indices.append(idx)
        if result_indices:
            return result_indices

        # For control_deps nodes, extract all mutated tensors from args[0]
        if (
            node.op == "call_function"
            and node.target == control_deps
            and len(node.args) >= 1
        ):
            mutated_tuple = node.args[0]
            if isinstance(mutated_tuple, (list, tuple)):
                for mutated_tensor in mutated_tuple:
                    if (
                        isinstance(mutated_tensor, torch.fx.Node)
                        and mutated_tensor in args
                    ):
                        idx = args.index(mutated_tensor)
                        if idx not in result_indices:
                            result_indices.append(idx)

                if result_indices:
                    return result_indices

        return []

    def _build_placeholder_map(
        self, replacement_graph: torch.fx.GraphModule, args: list[torch.fx.Node]
    ) -> dict[torch.fx.Node, torch.fx.Node]:
        placeholders = [
            n for n in replacement_graph.graph.nodes if n.op == "placeholder"
        ]
        return {
            placeholders[i]: args[i]
            for i in range(min(len(placeholders), len(args)))
            if isinstance(args[i], torch.fx.Node)
        }

    def check_pattern_replacement_safety(
        self,
        match_nodes: list[torch.fx.Node],
        replacement_graph: torch.fx.GraphModule,
        args: list[torch.fx.Node],
    ) -> bool:
        self._rebuild_if_needed()
        assert self.escaping_nodes is not None
        assert self.mutation_detector is not None

        placeholder_map = self._build_placeholder_map(replacement_graph, args)
        replacement_mutations = self.mutation_detector.detect_all(
            list(replacement_graph.graph.nodes), node_map=placeholder_map
        )

        if not self.check_escaping_mutation_invariance(
            match_nodes,
            replacement_graph,
            args,
            self.escaping_nodes,
            replacement_mutations,
        ):
            return False

        if not self.check_subsequent_use_safety(replacement_mutations, match_nodes):
            return False

        if not self.check_escaping_aliasing_invariance(
            match_nodes, replacement_graph, args
        ):
            return False

        return True
