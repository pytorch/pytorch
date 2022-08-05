from collections import defaultdict
from typing import Dict, Set, Iterable

from torch.fx.node import Node
from torch.fx.graph import Graph


class GraphViewer:
    def __init__(self, graph: Graph):
        self.graph = graph

        # map of node to it's upstream dependency nodes
        # if A is found in dependency_map[B], then B depends on A (or a is an upstream depedency of b)
        self.dependency_map = self.__build_dependency_map()

    def __build_dependency_map(self) -> Dict[Node, Set[Node]]:
        dependency_map = defaultdict(set)

        # assumptions: nodes in graph are sorted in topological order
        for node in self.graph.nodes:
            for input_node in node.all_input_nodes:
                # add input_node and input_node's upstream dependency
                dependency_map[node].add(input_node)
                dependency_map[node].update(dependency_map[input_node])

        return dependency_map

    def node_depends_on(self, a: Node, b: Node) -> int:
        """
        Returns
            1 if b depends on a (,or equivalently a is an upstream depedency of b)
            -1 if a depends on b (,or equivalently b is an upstream depedency of a)
            0 if a and b doesn't have dependency between each other
        """

        if a in self.dependency_map[b]:
            return 1
        elif b in self.dependency_map[a]:
            return -1
        else:
            return 0

    def partition_depends_on(self, partition_a: Iterable[Node], partition_b: Iterable[Node]) -> int:
        """
        Returns
            1 if b depends on a (,or equivalently a is an upstream depedency of b)
            -1 if a depends on b (,or equivalently b is an upstream depedency of a)
            0 if a and b doesn't have dependency between each other
        """

        # TODO: build a cache here to speedup the query

        for node_a in partition_a:
            for node_b in partition_b:
                dependency = self.node_depends_on(node_a, node_b)
                if dependency != 0:
                    return dependency
        return 0
