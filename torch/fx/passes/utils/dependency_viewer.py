from collections import defaultdict
from typing import Dict, Set, Iterable
from enum import Enum

from torch.fx.node import Node
from torch.fx.graph import Graph

class DependencyType(Enum):
    """Enum for the type of dependency between two nodes/partitions, a and b, in a graph"""

    INDEPENDENT = 1         # A and B are independent
    A_DEPENDS_ON_B = 2      # A depends on B, i.e. A is a downstream consumer of B's output
    B_DEPENDS_ON_A = 3      # B depends on A, i.e. B is a downstream consumer of A's output
    INTER_DEPENDENT = 4     # A and B are dependent on each other, i.e. there is loop between A and B


class DependencyViewer:
    def __init__(self, graph: Graph):
        self.graph = graph

        # map of node to it's upstream dependency nodes
        # if A is found in upstream_nodes[B], then B depends on A
        self.upstream_nodes = self.__build_dependency_map()

    def __build_dependency_map(self) -> Dict[Node, Set[Node]]:
        upstream_nodes = defaultdict(set)

        # assumptions: nodes in graph are sorted in topological order
        for node in self.graph.nodes:
            for input_node in node.all_input_nodes:
                # add input_node and input_node's upstream dependency
                upstream_nodes[node].add(input_node)
                upstream_nodes[node].update(upstream_nodes[input_node])

        return upstream_nodes

    def node_depends_on(self, a: Node, b: Node) -> DependencyType:
        """
        Returns the dependency type between node a and b
        """
        b_depends_on_a = a in self.upstream_nodes[b]
        a_depends_on_b = b in self.upstream_nodes[a]

        if a_depends_on_b and b_depends_on_a:
            return DependencyType.INTER_DEPENDENT
        elif b_depends_on_a:
            return DependencyType.B_DEPENDS_ON_A
        elif a_depends_on_b:
            return DependencyType.A_DEPENDS_ON_B
        else:
            return DependencyType.INDEPENDENT

    def partition_depends_on(self, partition_a: Iterable[Node], partition_b: Iterable[Node]) -> DependencyType:
        """
        Returns the dependency type between partition_a and partition_b
        """

        dependency_found = DependencyType.INDEPENDENT

        for node_a in partition_a:
            for node_b in partition_b:
                dependency = self.node_depends_on(node_a, node_b)

                if dependency == DependencyType.INTER_DEPENDENT:
                    return dependency
                elif dependency == DependencyType.INDEPENDENT:
                    continue
                else:
                    if dependency_found == DependencyType.INDEPENDENT:
                        dependency_found = dependency
                    elif dependency_found != dependency:
                        return DependencyType.INTER_DEPENDENT

        return dependency_found
