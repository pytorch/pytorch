import operator
from typing import Dict, List, Set, NamedTuple, Tuple

from numpy import partition

import torch
# from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
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
# from torch.fx.passes.split_module import split_module

from torch.fx.passes.split_utils import Component

from torch.fx.passes.operator_support import (
    get_node_target,
    OperatorSupportBase,
)

from itertools import groupby
from collections import defaultdict


class CapabilityBasedPartitioner:

    def __init__(self,
                 module: torch.fx.GraphModule,
                 operator_support: OperatorSupportBase) -> None:
        self.partitions = None

        self.module = module
        self.operator_support = operator_support
        self.kernel_registry = [operator.add]
        self.executors = None

    def get_candidates(self):
        candidates = []

        # TODO: replace following with self.operator_support.is_node_supported()
        for node in self.module.graph.nodes:
            if node.op == "call_function":
                if node.target in self.kernel_registry:
                    candidates.append(node)
        return candidates

    def partition(self, candidates):
        assignment = {}
        partition_id = 0

        def all_equal(iterable):
            g = groupby(iterable)
            return next(g, True) and not next(g, False)

        # visit candidates in reversed topological order
        for node in reversed(candidates):
            # all the users of `node` has been assigned, or is not in the candidate list
            users_assigned = [user in assignment for user in node.users]

            # assign a node to a partition if all of its users has been assigned to the same partition
            if all(users_assigned):
                users_assignment = [assignment[user] for user in node.users]

                if all_equal(users_assignment):
                    assignment[node] = users_assignment[0]
            else:
                # create a new partition for all other cases:
                # user1         user2
                # not_supported not_supported
                # partition1    not_supported
                # partition1    partition2
                assignment[node] = partition_id
                partition_id += 1

        partitions_by_id = defaultdict(list)
        # current assigment contains nodes from bottom to the top
        # reverser the list, so that in each partitions, nodes are sorted from top to bottom
        for node, partition_id in reversed(assignment.items()):
            partitions_by_id[partition_id].append(node)

        return partitions_by_id.values()

    def fuse_partitions(self, partitions):
        # partitions: [ [node0, node1], [node2, node3] ]

        all_components: List[Component] = []

        # Mapping from tag to the corresponding component.
        tag_to_component: Dict[str, Component] = {}

        # Mapping from node in original module or created submodules to
        # corresponding component.
        node_to_component: Dict[torch.fx.Node, Component] = {}

        for partition_id, nodes in partitions.items():
            tag = str(partition_id)
            comp = Component(torch.fx.Graph(), len(all_components), str(partition_id))
            all_components.append(comp)
            tag_to_component[tag] = comp

            for node in nodes:
                node_to_component[node] = comp

            comp.graph.node_copy(node, )
