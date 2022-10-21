from typing import Dict, List, Set, Iterable, Optional

from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
from torch.fx.passes.tools_common import NodeList

from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupportBase

from collections import defaultdict
import logging
import itertools

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class Partition:
    def __init__(self, id: int = None, nodes: Iterable[Node] = None):
        self.id = id
        self.nodes: Set[Node] = set(nodes) if nodes is not None else set()

    def __repr__(self) -> str:
        return str(self.nodes)

    def add_node(self, node: Node):
        self.nodes.add(node)

    def remove_node(self, node: Node):
        self.nodes.remove(node)

    def size(self):
        return len(self.nodes)

class CapabilityBasedPartitioner:

    def __init__(self,
                 graph_module: GraphModule,
                 operator_support: OperatorSupportBase,
                 allows_single_node_partition: bool = False
                 ) -> None:
        self.graph_module = graph_module
        self.operator_support = operator_support
        self.allows_single_node_partition = allows_single_node_partition

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

    def __get_supported_nodes(self) -> NodeList:
        logging.debug("Collecting supported nodes...")
        supported_nodes = []
        for node in self.graph_module.graph.nodes:
            if self.operator_support.is_node_supported(dict(self.graph_module.named_modules()), node):
                supported_nodes.append(node)
        return supported_nodes

    def propose_partitions(self) -> List[Partition]:
        candidates: NodeList = self.__get_supported_nodes()

        # assumptions: nodes in candidate list is sorted in topological order
        assignment: Dict[Node, int] = {}   # maping from node to partition_id
        partitions_by_id: Dict[int, Partition] = {}  # mapping from partition_id to partition
        new_partition_id = itertools.count()

        def assign(node: Node, id: Optional[int] = None):
            # If id is None, remove the node from original assigment

            # node has been assigned before, clean up and re-assign
            if node in assignment:
                original_id = assignment[node]
                del assignment[node]
                partitions_by_id[original_id].remove_node(node)
                if partitions_by_id[original_id].size() == 0:
                    del partitions_by_id[original_id]

            if id is not None:
                assignment[node] = id
                if id not in partitions_by_id:
                    partitions_by_id[id] = Partition(id=id, nodes=[node])
                else:
                    partitions_by_id[id].add_node(node)

        logger.debug("Proposing partitions...")

        # visit candidates in reversed topological order
        for node in reversed(candidates):
            # use Dict as an ordered set to ensure deterministic partitioning result, don't care value
            user_partitions: Dict[Partition, None] = {}
            for user_node in node.users:
                if user_node in assignment:
                    id = assignment[user_node]
                    user_partitions[partitions_by_id[id]] = None
                else:
                    user_partitions[Partition(nodes=[user_node])] = None

            # Filter out all the partitions that has dependency on other users
            # TODO: find a better way to do this, rather than pair-wise comparision
            user_partitions_list = list(user_partitions.keys())
            for i in range(len(user_partitions_list)):
                for j in range(i + 1, len(user_partitions_list)):
                    pi = user_partitions_list[i]
                    pj = user_partitions_list[j]
                    dependency = self.__partition_depends_on(pi, pj)
                    if dependency == 1 and pj in user_partitions:
                        del user_partitions[pj]
                    elif dependency == -1 and pi in user_partitions:
                        del user_partitions[pi]

            # We use the following rules for partition assignment:
            # 1. If none of the candidates has been assigned to a partition, create a new partition
            # 2. If there is one partition candidate, assign to the partition
            # 3. If there are more than one partition candidates, assign current node to the first partition and
            #    merge the other partitions with first partition, since user_partitions doesn't have depedency between
            #    each other.

            assigned_candidate_partition_ids = [partition.id for partition in user_partitions if partition.id is not None]

            if len(assigned_candidate_partition_ids) == 0:
                # create a new partition
                assign(node, next(new_partition_id))
            elif len(assigned_candidate_partition_ids) == 1:
                id = assigned_candidate_partition_ids[0]
                assign(node, id)
            else:
                # users are assigned to more than one partition, since user_partitions doesn't have
                # dependency on each other, they can be fused into a single partition
                id = assigned_candidate_partition_ids[0]
                assign(node, id)

                reassignment: Dict[Node, int] = {}
                for other_id in assigned_candidate_partition_ids[1:]:
                    for other_node in partitions_by_id[other_id].nodes:
                        reassignment[other_node] = id
                for other_node in reassignment:
                    assign(other_node, id)

        # post processing to re-assign "getitem" nodes into upstream partition
        logger.debug("Reassigning getitem nodes to its producer node's partition...")
        nodes_reassignment: Dict[Node, int] = {}
        for node in self.graph_module.graph.nodes:
            is_tuple_output = True
            for user in node.users:
                if user.op != "call_function" or \
                   _get_qualified_name(user.target) != "_operator.getitem":     # type: ignore[arg-type]
                    is_tuple_output = False
                    break

            # node has tuple outputs, re-assign all following getitem node into node's partition
            if is_tuple_output:
                id = assignment.get(node, None)     # type: ignore[arg-type]
                for user in node.users:
                    if assignment.get(user, None) != id:    # type: ignore[arg-type]
                        nodes_reassignment[user] = id
        for node, id in nodes_reassignment.items():
            assign(node, id)

        # filter out single node partitions
        if not self.allows_single_node_partition:
            logger.debug("Filtering out single node partitions...")
            non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
            partitions_to_remove: List[int] = []
            for id, partition in partitions_by_id.items():
                compute_node_count = 0
                for node in partition.nodes:
                    if node.op == "call_function" and \
                       _get_qualified_name(node.target) not in non_compute_ops:  # type: ignore[arg-type]
                        compute_node_count += 1
                if compute_node_count <= 1:
                    partitions_to_remove.append(id)
            for id in partitions_to_remove:
                del partitions_by_id[id]

        logger.debug("Partitions proposed:")
        for id, partition in partitions_by_id.items():
            logger.debug(f"partition #{id}", [node.name for node in partition.nodes])

        return list(partitions_by_id.values())

    def fuse_partitions(self, partitions: List[Partition]) -> GraphModule:
        logger.debug("Fusing partitions...")
        # fuse_by_partitions expects partitions in List[List[Node]]: [ [node0, node1], [node2, node3] ]
        return fuse_by_partitions(self.graph_module, [list(partition.nodes) for partition in partitions])

    def partition_and_fuse(self) -> GraphModule:
        partitions = self.propose_partitions()
        fused_gm = self.fuse_partitions(partitions)
        return fused_gm
