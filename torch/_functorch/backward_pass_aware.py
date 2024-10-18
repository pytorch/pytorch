#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import functools
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import networkx as nx
import torch.fx as fx
import xpress as xp
from pulp import (
    lpDot,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpStatus,
    lpSum,
    LpVariable,
    PULP_CBC_CMD,
    value,
)


@dataclass
class NodeInfo:
    # Be careful about iterating over these explicitly, as their order may not
    # be deterministic
    inputs: List[fx.Node]
    _required_fw_nodes: Set[fx.Node]
    required_bw_nodes: Set[fx.Node]
    unclaimed_nodes: Set[fx.Node]
    fw_order: Dict[fx.Node, int]

    @functools.cached_property
    def required_fw_nodes(self) -> List[fx.Node]:
        return sorted(
            (n for n in self._required_fw_nodes), key=lambda n: self.fw_order[n]
        )

    def is_required_fw(self, n: fx.Node) -> bool:
        return n in self._required_fw_nodes

    def is_required_bw(self, n: fx.Node) -> bool:
        return n in self.required_bw_nodes

    def is_unclaimed(self, n: fx.Node) -> bool:
        return n in self.unclaimed_nodes

    def get_fw_order(self, n: fx.Node) -> int:
        assert n in self._required_fw_nodes, f"Node {n} not in fw nodes!"
        return self.fw_order[n]


class LongRecomputationChains:
    """
    this class is intended to make AC backward pass aware. The code will be invoked after knapsack-based "foward-pass" AC (so it assumes that subset of nodes is already marked for recomputation).
    The main idea is as follows.
    The algorithm will decide to flip certain targeted nodes tagged as "recompute" and flip them to be "saved". Doing so will avoid rematerializing "long" or "memory-heavy" chains and thus avoid backward pass memory spikes.
    """

    def __init__(
        self,
        memory: List[float],
        joint_graph: fx.Graph,
        bw_memory_fraction_target: float,
        node_info: NodeInfo,
        all_recomputable_banned_nodes: List[fx.Node],
    ) -> None:
        """
        data: a dictionary of node_id -> node_data
        saved_nodes: a set of node_ids that are "saved" i.e. marked for recomputation by the knapsack solver
        my_digraph: a networkx digraph object representing the computation graph
        """
        print("====== initializing LongRecomputationChains =======")
        self.prob = xp.problem()
        self.node_info = node_info

        # print(node_info)
        self.graph = self._compute_digraph(
            node_info, joint_graph, all_recomputable_banned_nodes
        )

        # Discretization level
        S = 100000
        self.memory: Dict[fx.Node, int] = {
            node: int(normalized_memory * S)
            for node, normalized_memory in zip(all_recomputable_banned_nodes, memory)
        }
        print(
            "=========GRAPH SIZE:{graphsize} nodes".format(
                graphsize=len(self.graph.nodes)
            )
        )

        self.all_nodes = list(joint_graph.nodes)
        self.bw_mem_target = bw_memory_fraction_target

        self.data = self._extract_node_info()

        self.formulate_ILP(self.bw_mem_target)

    def _compute_digraph(
        self,
        node_info: NodeInfo,
        joint_graph: fx.Graph,
        all_recomputable_banned_nodes: Set[fx.Node],
    ) -> nx.DiGraph:
        """
        from the joint graph, we construct a subgraph corresponding to all the forward nodes that are recomputable.
        """
        nx_graph = nx.DiGraph()
        for node in joint_graph.nodes:
            if (
                node in node_info.required_fw_nodes
                and node in all_recomputable_banned_nodes
            ):
                for inp in node.all_input_nodes:
                    nx_graph.add_edge(inp.name, node.name)
        return nx_graph

    def _extract_node_info(self) -> Dict[fx.Node, Dict[str, float]]:
        data = {}
        for i in self.graph.nodes:
            data[i] = {}
            if i in self.memory:
                data[i]["mem"] = self.memory[
                    i
                ]  # for each node, store the normalized memory size of the node.
            else:  # if node not found in memory dict, then it is because it is not recomputable?
                data[i][
                    "mem"
                ] = 0.0  # setting a high weight will ensure that this module does not get selected
        return data

    def formulate_ILP(self, recomp_budget) -> None:
        """
        solve the ILP problem to find the set of nodes to save.
        """
        prob = LpProblem("BWPA", LpMinimize)
        # Define variables
        X = LpVariable.dicts("X", list(self.graph.nodes), 0, None, LpBinary)
        Z = LpVariable.dicts("Z", list(self.graph.nodes), 0, None, LpContinuous)
        y = LpVariable("y", 0, None, LpContinuous)
        Q = LpVariable.dicts(
            "Q",
            [(i, j) for i in self.graph.nodes for j in self.graph.nodes],
            0,
            None,
            LpContinuous,
        )
        recomp_budget = 2
        # Define problem
        prob = LpProblem("ILP", LpMinimize)
        # Add variables to problem

        # Define constraints
        M = 20 * sum([self.data[i]["mem"] for i in graph.nodes])

        for i in list(self.graph.nodes):
            prec = list(self.graph.predecessors(i))
            if i in prec:
                prec.remove(i)

            # Constraint 1: cumulative memory
            prob += (
                Z[i]
                == -self.data[i]["mem"] * X[i]
                + sum(Z[t] - Q[t, t] for t in prec)
                + self.data[i]["mem"]
            )

            # Constraint 2: peak memory dummy constraint
            prob += Z[i] <= y

            for t in prec:
                # Linearization constraints
                prob += Q[t, t] <= M * X[t]
                prob += Q[t, t] >= 0
                prob += Z[t] - Q[t, t] >= 0
                prob += Z[i] - Q[t, t] <= M * (1 - X[t])

        # Add budget constraint
        prob += (
            sum(self.data[i]["mem"] * X[i] for i in list(self.graph.nodes))
            <= recomp_budget
        )
        # Set Objeictive
        prob += lpSum(y)
        self.prob = prob

    def solve_ILP(self) -> Tuple[List[int], List[int]]:
        solver = PULP_CBC_CMD()
        status = self.prob.solve(solver)  # Use the default solver
        if status != 1:
            logger.error("Solver failed to find a solution: %s", LpStatus[status])
        else:
            print("Solver found a solution")

        saved_values = [i for i in X.keys() if X[i].varValue > 0.9]
        recomp_values = [i for i in X.keys() if X[i].varValue <= 0.9]
        return saved_values, recomp_values


def get_saved_recomp_nodes_from_BWPA(
    memory: List[float],
    joint_graph: fx.Graph,
    max_memory: float,
    node_info: NodeInfo,
    all_recomputable_banned_nodes: List[fx.Node],
) -> Tuple[List[int], List[int]]:
    lrc = LongRecomputationChains(
        memory, joint_graph, max_memory, node_info, all_recomputable_banned_nodes
    )
    saved_node_idx, recomp_node_idx = lrc.solve_ILP()
    return saved_node_idx, recomp_node_idx
