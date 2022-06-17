from typing import Dict, List, Set, Iterable

from torch.fx.passes.fuser_utils import fuse_by_partitions
from torch.fx.passes.tools_common import NodeList

from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.passes.operator_support import OperatorSupportBase

from itertools import groupby
from collections import defaultdict

class Partition:
    def __init__(self, id: int = None, nodes: Iterable[Node] = set()):
        self.id = id
        self.nodes: Set[Node] = set(nodes)

    def __repr__(self) -> str:
        return str(self.nodes)

    def add_node(self, node: Node):
        self.nodes.add(node)

    def size(self):
        return len(self.nodes)

class CapabilityBasedPartitioner:

    def __init__(self,
                 graph_module: GraphModule,
                 operator_support: OperatorSupportBase) -> None:
        self.graph_module = graph_module
        self.operator_support = operator_support

        # map of node to it's upstream dependency nodes
        # if A is found in dependency_map[B], then B depends on A (or a is an upstream depedency of b)
        self.dependency_map = self.__build_dependency_map()

    def __build_dependency_map(self) -> Dict[Node, Set[Node]]:
        dependency_map = defaultdict(set)

        # assumptions: nodes in graph are sorted in topological order
        for node in self.graph_module.graph.nodes:
            for input_node in node.all_input_nodes:
                # add input_node and input_node's upstream dependency
                dependency_map[node].add(input_node)
                dependency_map[node].update(dependency_map[input_node])

        return dependency_map

    def __node_depends_on(self, a: Node, b: Node) -> int:
        # Returns
        # 1 if b depends on a (,or equivalently a is an upstream depedency of b)
        # -1 if a depends on b (,or equivalently b is an upstream depedency of a)
        # 0 if a and b doesn't have dependency between each other

        if a in self.dependency_map[b]:
            return 1
        elif b in self.dependency_map[a]:
            return -1
        else:
            return 0

    def __partition_depends_on(self, partition_a: Partition, partition_b: Partition) -> int:
        # Returns
        # 1 if b depends on a (,or equivalently a is an upstream depedency of b)
        # -1 if a depends on b (,or equivalently b is an upstream depedency of a)
        # 0 if a and b doesn't have dependency between each other

        # TODO: build a cache here to speedup the query

        for node_a in partition_a.nodes:
            for node_b in partition_b.nodes:
                dependency = self.__node_depends_on(node_a, node_b)
                if dependency != 0:
                    return dependency
        return 0

    def get_candidates(self) -> NodeList:
        candidates = []
        for node in self.graph_module.graph.nodes:
            if self.operator_support.is_node_supported(self.graph_module.named_modules(), node):
                candidates.append(node)
        return candidates

    def partition(self, candidates: NodeList) -> List[Partition]:
        # assumptions: nodes in candidate list is sorted in topological order
        assignment: Dict[Node, int] = {}   # maping from node to partition_id
        partitions_by_id: Dict[int, Partition] = {}  # mapping from partition_id to partition

        def assign(node, id):
            assignment[node] = id

            if id not in partitions_by_id:
                partitions_by_id[id] = Partition(id=id, nodes=[node])
            else:
                partitions_by_id[id].add_node(node)

        def all_equal(iterable):
            g = groupby(iterable)
            return next(g, True) and not next(g, False)

        # visit candidates in reversed topological order
        for node in reversed(candidates):

            user_partitions: Set[Partition] = set()
            for user_node in node.users:
                if user_node in assignment:
                    id = assignment[user_node]
                    user_partitions.add(partitions_by_id[id])
                else:
                    user_partitions.add(Partition(nodes=[user_node]))

            # print(node)
            # print('user_partitions', user_partitions)

            # Filter out all the partitions that has dependency on other users
            # TODO: find a better way to do this, rather than pair-wise comparision
            user_partitions_list = list(user_partitions)
            for i in range(len(user_partitions_list)):
                for j in range(i + 1, len(user_partitions_list)):
                    pi = user_partitions_list[i]
                    pj = user_partitions_list[j]
                    dependency = self.__partition_depends_on(pi, pj)
                    if dependency == 1 and pj in user_partitions:
                        user_partitions.remove(pj)
                    elif dependency == -1 and pi in user_partitions:
                        user_partitions.remove(pi)

            # print("user_partitions after filtering", user_partitions)

            # We use the following rules for partition assignment:
            # 1. If none of the candidates has been assigned to a partition, create a new partition
            # 2. If all of the candidate has been assigned to the same partition, assign to the same partition
            # 3. If candidates has been assigned to more then one paritions, assign to the largest partition (by node count)
            # 4. If none of rule above can break the tie, randomly assign to one of the largest partition among candidates
            assigned_candidate_partition_ids = [partition.id for partition in user_partitions if partition.id is not None]

            if len(assigned_candidate_partition_ids) == 0:
                # create a new partition
                id = len(partitions_by_id)
                assign(node, id)

            elif all_equal(assigned_candidate_partition_ids):
                id = assigned_candidate_partition_ids[0]
                assign(node, id)

            else:
                partitions_size_by_id = [[partitions_by_id[id].size(), id] for id in assigned_candidate_partition_ids]
                partitions_size_by_id = sorted(partitions_size_by_id, reverse=True)

                id = partitions_size_by_id[0][1]
                assign(node, id)

        return list(partitions_by_id.values())

    def fuse_partitions(self, partitions: List[Partition]) -> GraphModule:
        # fuse_by_partitions expects partitions in List[List[Node]]: [ [node0, node1], [node2, node3] ]
        return fuse_by_partitions(self.graph_module, [list(partition.nodes) for partition in partitions])
