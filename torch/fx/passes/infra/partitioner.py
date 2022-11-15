from typing import Dict, List, Set, Iterable, Sequence, Optional

from torch.fx.passes.utils.fuser_utils import fuse_by_partitions

from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupportBase

import logging
import itertools
from copy import copy

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
                 allows_single_node_partition: bool = False,
                 non_compute_ops: Optional[Sequence[str]] = None,
                 allowed_single_node_partition_ops: Optional[Sequence[str]] = None,
                 ) -> None:
        self.graph_module = graph_module
        self.operator_support = operator_support
        self.allows_single_node_partition = allows_single_node_partition
        self.non_compute_ops = non_compute_ops if non_compute_ops is not None else []
        self.allowed_single_node_partition_ops = (
            allowed_single_node_partition_ops
            if allowed_single_node_partition_ops is not None
            else []
        )

    def __is_node_supported(self, node: Node) -> bool:
        return (
            self.operator_support.is_node_supported(dict(self.graph_module.named_modules()), node)
        )

    def propose_partitions(self) -> List[Partition]:
        # assumptions: nodes in candidate list is sorted in topological order
        assignment: Dict[Node, int] = {}   # maping from node to partition_id
        partitions_by_id: Dict[int, Partition] = {}  # mapping from partition_id to partition
        new_partition_id = itertools.count()

        # try to merge partition other_id into partition self_id
        # merge only happens if the end graph doesn't contain cyclic dependency
        # returns `True` when merge happens, `False` otherwise.
        def maybe_merge_partition(self_id: int, other_id: int):
            # merged_nodes is the union of nodes in two partition to-be-merged
            merged_nodes = copy(partitions_by_id[self_id].nodes)
            merged_nodes.update(partitions_by_id[other_id].nodes)

            visited: Set[Node] = set()

            def dfs_find_cycle(node):
                if node in visited:
                    return False
                if node in merged_nodes:
                    return True  # found cycle, return

                visited.add(node)
                # branching on hitting partition or not
                if node in assignment:
                    # Since partition is not merged in the graph yet, when we
                    # hit a node in a partition through DFS, we need to
                    # traverse all nodes in the partition to properly reflect
                    # dependencies after the fusion
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
                        # return false indicating cyclic dependency found and
                        # merge is aborted
                        return False

            # no cyclic dependency found, move forward with the merge
            # updating partition nodes
            partitions_by_id[self_id].nodes = merged_nodes
            # updating assignment map
            for node in partitions_by_id[other_id].nodes:
                assignment[node] = self_id
            # delete other partition
            del partitions_by_id[other_id]

            return True

        def merge_single_node(node: Node, id: Optional[int]):
            if node in assignment:
                partitions_by_id[assignment[node]].remove_node(node)

            if id is None:
                assignment.pop(node)
            elif id not in partitions_by_id:
                assignment[node] = id
                partitions_by_id[id] = Partition(id=id, nodes=[node])
            else:
                assignment[node] = id
                partitions_by_id[id].add_node(node)

        logger.debug("Proposing partitions...")

        for node in reversed(self.graph_module.graph.nodes):
            # use Dict as an ordered set to ensure deterministic partitioning result, don't care value
            merge_candidates: Dict[int, None] = {}

            # Note a limited horizontal fusion is enabled:
            #   when `node` is not supported, the code below attempts to fuse consumer of `node`.
            #
            # I don't see a need to add a knob to disable horizontal fusion yet, we can short-cut
            # the fusion by adding an `else` block here to skip horizontal fusion.
            if self.__is_node_supported(node) and node not in assignment:
                partition_id = next(new_partition_id)
                merge_single_node(node, partition_id)
                merge_candidates[partition_id] = None

            for user_node in node.users:
                if user_node in assignment:
                    merge_candidates[assignment[user_node]] = None

            merge_candidates_list = list(merge_candidates.keys())
            if len(merge_candidates_list) > 1:
                self_id = merge_candidates_list[0]
                for other_id in merge_candidates_list[1:]:
                    # note: merge partition `other_id` into partition `self_id` if
                    # it doesn't create cyclic depenency in the graph, otherwise,
                    # this is a no-op
                    maybe_merge_partition(self_id, other_id)

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
            default_non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
            non_compute_ops = default_non_compute_ops.union(set(self.non_compute_ops))
            partitions_to_remove: List[int] = []
            for id, partition in partitions_by_id.items():
                compute_node_count = 0
                for node in partition.nodes:
                    if node.op == "call_function" and \
                       _get_qualified_name(node.target) not in non_compute_ops:  # type: ignore[arg-type]
                        compute_node_count += 1
                    if node.op == "call_function" and \
                       _get_qualified_name(node.target) in self.allowed_single_node_partition_ops:
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
