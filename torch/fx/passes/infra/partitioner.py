from typing import Dict, List, Set, Iterable, Optional

from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
from torch.fx.passes.tools_common import NodeList

from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupportBase

from collections import defaultdict
import logging
import itertools

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# TODO: partition here could use a refactor to improvement efficiency on merge and traversal
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

    def __get_supported_nodes(self) -> NodeList:
        logging.debug("Collecting supported nodes...")
        supported_nodes = []
        for node in self.graph_module.graph.nodes:
            if self.operator_support.is_node_supported(dict(self.graph_module.named_modules()), node):
                supported_nodes.append(node)
        return supported_nodes

    def __is_node_supported(self, node: Node) -> bool:
        # TODO: reject 'getitem' node since they are special cased in partitioning.
        return self.operator_support.is_node_supported(dict(self.graph_module.named_modules()), node)

    def propose_partitions(self) -> List[Partition]:
        # candidates: NodeList = self.__get_supported_nodes()

        # assumptions: nodes in candidate list is sorted in topological order
        assignment: Dict[Node, int] = {}   # maping from node to partition_id
        partitions_by_id: Dict[int, Partition] = {}  # mapping from partition_id to partition
        new_partition_id = itertools.count()

        def maybe_merge_partition(self_id: int, other_id: int):
            # merged nodes
            merged_nodes = copy(partitions_by_id[self_id].nodes).update(partitions_by_id[other_id].nodes)

            # def merge_breaks_dagpartitions: List[Partition]):
            visited: NodeSet = set()

            def dfs_find_cycle(node):
                if node in visited:
                    return False
                if node in merged_nodes:
                    return True  # found cycle, return

                # branching on partition or not
                visited.add(node)
                if node in assigment:
                    for p_node in partitions_by_id[assignment[node]].nodes:
                        for user_node in p_node.users:
                            if dfs_find_cycle(user_node):
                                return True
                else:
                    for user_node in node.users:
                        if dfs_find_cycle(user_node):
                            return True
                return False
                
            # check if merge would create cyclic dependency.
            for node in merged_nodes:
                for user_node in node.users:
                    if user_node not in merged_nodes and dfs_find_cycle(user_node):
                        # return false indicating no fusion happening.
                        return False

            # no cyclic dependency, let's move forward with the merge
            # updating partition nodes
            partitions_by_id[self_id].nodes = merged_nodes
            # updating node map
            for node in partitions_by_id[other_id].nodes:
                assignment[node] = self_id
            # delete other partition
            del partitions_by_id[other_id]

            return True

        def merge_single_node(node: Node, id: int):
            assert node not in assignment

            assignment[node] = id
            if id not in partitions_by_id:
                partitions_by_id[id] = Partition(id=id, nodes=[node])
            else:
                partitions_by_id[id].add_node(node)

        logging.debug("Proposing partitions...")

        def mergeAcyclicUserPartitions(node, merge_self=True):
            # use Dict as an ordered set to ensure deterministic partitioning result, don't care value
            merge_candidates: Dict[int, None] = {}

            if self.__is_node_supported(node) and node not in assignment:
                partition_id = next(new_partition_id)
                merge_single_node(node, partition_id)
                merge_candidates[partition_id] = None

            for user_node in node.users:
                if user_node in assignment:
                    merge_candidates[assignment[user_node].id] = None

            # Filter out all the partitions that has dependency on other users
            # TODO: find a better way to do this, rather than pair-wise comparision
            merge_candidates_list = list(merge_candidates.keys())
            if len(merge_candidates_list) > 1:
                self_id = merge_candidates_list[0]
                for other_id in merge_candidates_list[1:]:
                    maybe_merge_partition(self_id, other_id)

        # visit candidates in reversed topological order
        # for node in reversed(candidates):
        #     mergeAcyclicUserPartitions(node, True)

        # not very efficient, this handles sibling fusion of partitions that share inputs.
        for node in self.graph_module.graph.nodes:
            mergeAcyclicUserPartitions(node)
            #if node not in assignment:
            #    mergeAcyclicUserPartitions(node, False)

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
                        nodes_reassignment[user] = id  # type: ignore[assignment]
        for node, id in nodes_reassignment.items():
            merge_single_node(node, id)

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

        logging.debug("Partitions proposed:")
        for id, partition in partitions_by_id.items():
            logging.debug(f"partition #{id}", [node.name for node in partition.nodes])

        return list(partitions_by_id.values())

    def fuse_partitions(self, partitions: List[Partition]) -> GraphModule:
        logging.debug("Fusing partitions...")
        # fuse_by_partitions expects partitions in List[List[Node]]: [ [node0, node1], [node2, node3] ]
        return fuse_by_partitions(self.graph_module, [list(partition.nodes) for partition in partitions])

    def partition_and_fuse(self) -> GraphModule:
        partitions = self.propose_partitions()
        fused_gm = self.fuse_partitions(partitions)
        return fused_gm
