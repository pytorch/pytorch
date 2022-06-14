import operator
from typing import Dict, List, Set, NamedTuple, Tuple, Iterable
from functools import cmp_to_key

from torch.fx.passes.fuser_utils import fuse_by_partitions

from torch.fx.passes.tools_common import NodeList, NodeSet, legalize_graph

import torch
# from torch.fx.experimental.partitioner_utils import (
#     Partition,
#     Device,
#     PartitionerConfig,
#     get_partition_to_latency_mapping,
#     get_latency_of_partitioned_graph,
#     NodeLatency,
#     get_extra_size_of,
#     PartitionMode,
# )
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.operator_support import (
    get_node_target,
    OperatorSupportBase,
)

from itertools import groupby
from collections import defaultdict

class Partition:
    def __init__(self, id=None, nodes: List[Node]=list()):
        self.id = id
        self.nodes: List[Node] = nodes

    def __repr__(self) -> str:
        return str(self.nodes)

    def add_node(self, node: Node):
        self.nodes.append(node)

    def size(self):
        return len(self.nodes)

class CapabilityBasedPartitioner:

    def __init__(self,
                 module: torch.fx.GraphModule,
                 operator_support: OperatorSupportBase) -> None:
        self.partitions = None

        self.module = module
        self.operator_support = operator_support
        self.kernel_registry = [operator.add]
        self.executors = None

        # map of node to it's upstream dependency nodes
        # if A is found in dependency_map[B], then B depends on A (or a is an upstream depedency of b)
        self.dependency_map  = self.__build_dependency_map()

    def __build_dependency_map(self) -> Dict[Node, Set[Node]]:
        dependency_map = defaultdict(set)

        # assumptions: nodes in graph are sorted in topological order
        for node in self.module.graph.nodes:
            for input_node in node.all_input_nodes:
                # add input_node and input_node's upstream dependency
                dependency_map[node].add(input_node)
                dependency_map[node].update(dependency_map[input_node])

        print(dependency_map)
        return dependency_map

    def __node_depends_on(self, a: Node, b: Node) -> bool:
        # True if b depends on a (,or equivalently a is an upstream depedency of b)
        return a in self.dependency_map[b]

    def __partition_depends_on(self, partition_a: Partition, partition_b: Partition) -> bool:
        # True if b depends on a (,or equivalently a is an upstream depedency of b)

        # TODO: build a cache here to speedup the query

        # return True if any of node in partition_a is an upstream node of any node in partition_b
        for node_a in partition_a.nodes:
            for node_b in partition_b.nodes:
                if self.__node_depends_on(node_a, node_b):
                    return True

        return False

    def get_candidates(self):
        candidates = []

        # TODO: replace following with self.operator_support.is_node_supported()
        for node in self.module.graph.nodes:
            if node.op == "call_function":
                if node.target in self.kernel_registry:
                    candidates.append(node)
        return candidates

    def partition(self, candidates: NodeList) -> NodeList:
        # assumptions: nodes in candidate list is sorted in topological order
        assignment = {}
        partition_id = 0
        partitions_by_id = dict()

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

            partition_candidates: List[Partition] = list()
            partition_candidates_id : Set[int] = set()
            for user_node in node.users:
                if user_node in assignment:
                    if assignment[user_node] not in partition_candidates_id:
                        id = assignment[user_node]
                        partition_candidates_id.add(id)
                        partition_candidates.append(partitions_by_id[id])

                else:
                    partition_candidates.append(Partition(nodes=[user_node]))


            # TODO: simple sort is probably not enough, need to do strict topo sort
            # After sorting: partitions_sorted[0] <= partitions_sorted[1] <=... partitions_sorted[n]
            partitions_sorted = sorted(partition_candidates, key=cmp_to_key(self.__partition_depends_on))

            print(node)
            print("partitions_sorted", partitions_sorted)

            # find all the parallel partitions
            # After filtering: partition_candidates[0] == partition_candidates[1] ==... partition_candidates[n]
            partition_candidates = [ partitions_sorted[0] ]
            for partition in partitions_sorted[1:]:
                if self.__partition_depends_on(partition_candidates[-1], partition):
                    # partition depends on partition_candidates[-1]
                    break
                else:
                    partition_candidates.append(partition)

            # We use the following rules for partition assignment:
            # 1. If none of the candidates has been assigned to a partition, create a new partition
            # 2. If all of the candidate has been assigned to the same partition, assign to the same partition
            # 3. If candidates has been assigned to more then one paritions, assign to the largest partition (by node count)
            # 4. If none of rule above can break the tie, randomly assign to one of the largest partition among candidates

            print("partition_candidates", partition_candidates)

            candidate_partition_ids = []
            for partition in partition_candidates:
                if partition.id is not None:
                    candidate_partition_ids.append(partition.id)

            if len(candidate_partition_ids) == 0:
                # create a new partition
                assign(node, partition_id)
                partition_id += 1

            elif all_equal(candidate_partition_ids):
                id = candidate_partition_ids[0]
                assign(node, id)

            else:
                partitions_size_by_id = [ [partitions_by_id[id].size(), id] for id in candidate_partition_ids]
                partitions_size_by_id = sorted(partitions_size_by_id, reverse=True)

                id = partitions_size_by_id[0][1]
                assign(node, id)

        print("assignment", assignment)

        return [partition.nodes for partition in partitions_by_id.values()]

    def fuse_partitions(self, partitions):
        # partitions: [ [node0, node1], [node2, node3] ]
        return fuse_by_partitions(self.module, partitions)