from collections import deque
from typing import Callable, Dict, List, Set, Tuple

import networkx as nx
import numpy as np

from torch._functorch._activation_checkpointing.graph_info_provider import (
    GraphInfoProvider,
)


class KnapsackEvaluator:
    """
    This class evaluates the theoretical runtime and peak memory usage of a given checkpointing strategy.
    It takes in a graph and a list of nodes that are saved and recomputed, and then simulates the
    backward pass to calculate the peak memory usage.
    """

    def __init__(
        self,
        graph_info_provider: GraphInfoProvider,
    ) -> None:
        self._graph_info_provider = graph_info_provider

    def _get_backward_memory_from_topologically_sorted_graph(
        self,
        node_graph: nx.DiGraph,
        node_memories: Dict[str, float],
        saved_nodes_set: Set[str],
        peak_memory_after_forward_pass: float,
    ) -> List[Tuple[float, str]]:
        """
        Simulates the backward pass and keeps track of the peak memory usage.

        High Level Steps:
            1. Set Initial Peak/Current Memory
                Allows you to set the peak memory after the forward pass, but typically this is
                the sum of the estimated memory of the saved nodes.
            2. Perform a reverse topological sort of the node_graph.
                If full graph is defined then will sort the full graph and only process the subset
                of nodes in the node_graph.
            3. Iterate through the sorted graph nodes.
                If the node is saved then just drop it's memory from current memory.
                If the node is not saved then add it's memory to current memory and then traverse it's
                predecessors to simulate recomuptation chain. Will check if new peak memory after all
                predecessors are processed.

        Args:
            node_graph (nx.DiGraph): A directed graph representing the recomputable forward nodes.
            saved_nodes_set (Set[str]): A set of node names that are saved.
            peak_memory_after_forward_pass (float): The peak memory usage after the forward pass.
        """
        current_memory = [
            (peak_memory_after_forward_pass, "Initial Peak/Current Memory")
        ]
        already_computed = set()
        sorted_nodes = list(reversed(list(nx.topological_sort(node_graph))))
        dependencies_computed = set()

        for node in sorted_nodes:
            if node in saved_nodes_set or node in already_computed:
                current_memory.append(
                    (
                        current_memory[-1][0] - node_memories[node],
                        f"Dropping Node(already saved): {node}",
                    )
                )
                continue

            already_computed.add(node)
            current_memory.append(
                (
                    current_memory[-1][0] + node_memories[node],
                    f"Recomputing Node: {node}",
                )
            )
            # Create a queue of dependencies required for recomputation
            predecessor_queue = deque(
                [
                    dependency
                    for dependency, v in node_graph.in_edges(node)
                    if dependency not in already_computed
                ]
            )
            while predecessor_queue:
                dep = predecessor_queue.popleft()
                already_computed.add(dep)
                dependencies_computed.add(dep)
                current_memory.append(
                    (
                        current_memory[-1][0] + node_memories[dep],
                        f"Recomputing Predecessor of {node}: {dep}",
                    )
                )
                # Add predecessors of the predecessor to the queue if they haven't been recomputed yet
                for dependency_of_dependency, _ in node_graph.in_edges(dep):
                    if (
                        dependency_of_dependency in already_computed
                        or dependency_of_dependency in saved_nodes_set
                        or dependency_of_dependency in predecessor_queue
                    ):
                        continue
                    predecessor_queue.append(dependency_of_dependency)
            dependencies_computed.clear()
            current_memory.append(
                (current_memory[-1][0] - node_memories[node], f"Dropping Node: {node}")
            )
        return current_memory

    def _validate_all_indexes_accounted_for_in_provided_output(
        self, saved_nodes_idxs: List[int], recomputable_node_idxs: List[int]
    ) -> None:
        """
        Validate that all indexes are accounted for in the provided output.
        This function checks that the union of saved nodes and recomputable nodes
        covers all candidate nodes without any overlaps.
        """
        recomputable_node_idxs_set = set(recomputable_node_idxs)
        saved_nodes_idxs_set = set(saved_nodes_idxs)
        all_candidate_nodes_idxs = set(
            range(len(self._graph_info_provider.all_recomputable_banned_nodes))
        )
        # Check that there are no overlaps between saved nodes and recomputable nodes
        assert (
            len(recomputable_node_idxs_set.intersection(saved_nodes_idxs_set)) == 0
        ), "Saved nodes and recomputable nodes cannot have any overlaps"
        # Check that all candidate nodes are accounted for
        assert (
            recomputable_node_idxs_set.union(saved_nodes_idxs_set)
            == all_candidate_nodes_idxs
        ), "All candidate nodes must be accounted for in the provided output"

    def evaluate_knapsack_output(
        self,
        saved_nodes_idxs: List[int],
        recomputable_node_idxs: List[int],
        account_for_backward_pass: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate the theoretical runtime and peak memory usage of a given checkpointing strategy.
        Args:
        - saved_nodes_idxs (List[int]): The indices of nodes that are saved.
        - recomputable_node_idxs (List[int]): The indices of nodes that need to be recomputed.
        """
        self._validate_all_indexes_accounted_for_in_provided_output(
            saved_nodes_idxs, recomputable_node_idxs
        )
        recomputation_runtime = sum(
            self._graph_info_provider.all_node_runtimes[
                self._graph_info_provider.all_recomputable_banned_nodes[node]
            ]
            for node in recomputable_node_idxs
        )
        if account_for_backward_pass:
            memory_list = self._get_backward_memory_from_topologically_sorted_graph(
                node_graph=self._graph_info_provider.recomputable_node_only_graph_with_larger_graph_context,
                saved_nodes_set={
                    self._graph_info_provider.all_recomputable_banned_nodes[i]
                    for i in saved_nodes_idxs
                },
                node_memories=self._graph_info_provider.all_node_memories,
                peak_memory_after_forward_pass=sum(
                    self._graph_info_provider.all_node_memories[
                        self._graph_info_provider.all_recomputable_banned_nodes[i]
                    ]
                    for i in saved_nodes_idxs
                ),
            )
            peak_memory = max(memory_list, key=lambda x: x[0])[0]
        else:
            peak_memory = sum(
                self._graph_info_provider.all_node_memories[
                    self._graph_info_provider.all_recomputable_banned_nodes[node]
                ]
                for node in saved_nodes_idxs
            )
        return {
            "peak_memory": peak_memory,
            "recomputation_runtime": recomputation_runtime,
            "non_ac_peak_memory": self._graph_info_provider.get_non_ac_peak_memory(),
            "theoretical_max_runtime": self._graph_info_provider.get_theoretical_max_runtime(),
            "percentage_of_theoretical_peak_memory": peak_memory
            / self._graph_info_provider.get_non_ac_peak_memory(),
            "percentage_of_theoretical_peak_runtime": recomputation_runtime
            / self._graph_info_provider.get_theoretical_max_runtime(),
        }

    def evaluate_distribution_of_results_for_knapsack_algo(
        self,
        knapsack_algo: Callable[
            [List[float], List[float], float], Tuple[float, List[int], List[int]]
        ],
        memory_budget_values: List[float],
    ) -> List[Dict[str, float]]:
        """
        Evaluates the distribution of results for a given knapsack algorithm.
        Args:
            knapsack_algo (Callable): The knapsack algorithm to use for evaluation.
            memory_budget_values (List[float]): A list of memory budgets to evaluate.
        """
        results = list()
        for memory_budget in memory_budget_values:
            _, saved_nodes, recomputed_nodes = knapsack_algo(
                self._graph_info_provider.get_knapsack_memory_input(),
                self._graph_info_provider.get_knapsack_runtime_input(),
                memory_budget,
            )
            result = self.evaluate_knapsack_output(
                saved_nodes_idxs=saved_nodes,
                recomputable_node_idxs=recomputed_nodes,
            )
            result["memory_budget"] = memory_budget
            results.append(result)
        return results

    def get_knee_point_memory_budget(
        self,
        knapsack_algo: Callable[
            [List[float], List[float], float], Tuple[float, List[int], List[int]]
        ],
        max_mem_budget: float = 0.1,
        min_mem_budget: float = 0.001,
        iterations: int = 100,
    ) -> float:
        """
        Finds the memory budget at the knee point in the Pareto frontier.
        The knee point is defined as the point where the trade-off between
        runtime and memory usage is optimal.
        Args:
            knapsack_algo (callable): Knapsack algorithm to use for evaluation.
            max_mem_budget (float, optional): Maximum memory budget. Defaults to 0.1.
            min_mem_budget (float, optional): Minimum memory budget. Defaults to 0.001.
            iterations (int, optional): Number of memory budgets to evaluate. Defaults to 100.
        Returns:
            float: Memory budget at the knee point.
        """
        results = self.evaluate_distribution_of_results_for_knapsack_algo(
            knapsack_algo=knapsack_algo,
            memory_budget_values=np.linspace(
                min_mem_budget, max_mem_budget, iterations
            ).tolist(),
        )
        runtime_norm = [
            (
                result["percentage_of_theoretical_peak_runtime"]
                - min(r["percentage_of_theoretical_peak_runtime"] for r in results)
            )
            / (
                max(r["percentage_of_theoretical_peak_runtime"] for r in results)
                - min(r["percentage_of_theoretical_peak_runtime"] for r in results)
            )
            for result in results
        ]
        memory_norm = [
            (
                result["percentage_of_theoretical_peak_memory"]
                - min(r["percentage_of_theoretical_peak_memory"] for r in results)
            )
            / (
                max(r["percentage_of_theoretical_peak_memory"] for r in results)
                - min(r["percentage_of_theoretical_peak_memory"] for r in results)
            )
            for result in results
        ]
        distances = [
            np.sqrt(runtime**2 + memory**2)
            for runtime, memory in zip(runtime_norm, memory_norm)
        ]
        knee_index = np.argmin(distances)
        return results[knee_index]["memory_budget"]
