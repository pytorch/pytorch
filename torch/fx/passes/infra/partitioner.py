from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
import collections
import itertools
import logging

from copy import copy
from typing import Dict, Iterable, List, Optional, Sequence, Set

from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupportBase


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class Partition:
    def __init__(self, id: Optional[int] = None, nodes: Optional[Iterable[Node]] = None):
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

class _DependencyViewer:
    def __init__(self, graph_module: GraphModule):
        self.upstreams = collections.defaultdict(set)
        self.downstreams = collections.defaultdict(set)

        for node in graph_module.graph.nodes:
            for input_node in node.all_input_nodes:
                # add input_node and input_node's upstream dependency
                self.upstreams[node].add(input_node)
                self.upstreams[node].update(self.upstreams[input_node])

        for node in reversed(graph_module.graph.nodes):
            for output_node in node.users:
                # add output_node and output_node's downstream dependency
                self.downstreams[node].add(output_node)
                self.downstreams[node].update(self.downstreams[output_node])

    def downstreams_of(self, node: Node) -> Set[Node]:
        return self.downstreams[node]

    def upstreams_of(self, node: Node) -> Set[Node]:
        return self.upstreams[node]

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
        self.dependency_viewer = _DependencyViewer(graph_module)

    def __is_node_supported(self, node: Node) -> bool:
        return (
            self.operator_support.is_node_supported(dict(self.graph_module.named_modules()), node)
        )

    def propose_partitions(self) -> List[Partition]:
        # partition_map is a mapping from partition id to a set of partition id's.
        # The value set contains all the partition ids that can be reached by doing a
        # DFS starting from the partition id in the key.
        partition_map : Dict[int, Set] = collections.defaultdict(set)

        # assumptions: nodes in candidate list is sorted in topological order
        assignment: Dict[Node, int] = {}   # mapping from node to partition_id
        partitions_by_id: Dict[int, Partition] = {}  # mapping from partition_id to partition
        new_partition_id = itertools.count()

        # try to merge partition other_id into partition self_id
        # merge only happens if the end graph doesn't contain cyclic dependency
        # returns `True` when merge happens, `False` otherwise.
        def maybe_merge_partition(self_id: int, other_id: int):
            # merged_nodes is the union of nodes in two partition to-be-merged
            merged_nodes = copy(partitions_by_id[self_id].nodes)
            merged_nodes.update(partitions_by_id[other_id].nodes)

            def dfs_iter_find_cycle(all_user_nodes: Set[Node]):
                for user_node in all_user_nodes:
                    visited_partition_ids = set()

                    for path_node in self.dependency_viewer.downstreams_of(user_node):
                        # If any of the nodes in the dfs path of this node are in the merged_nodes
                        # list then there is a cycle in the graph.
                        if path_node in merged_nodes:
                            return True

                        # If any of the nodes in the dfs path of this node are in the assignment
                        # map then we have to make sure that the partitions that these nodes belong
                        # to do not form a cycle with the current partitions being merged. This means
                        # iterating through all the nodes in all the parititons that are traversed in
                        # the dfs path and checking if they are in the merged_nodes list.
                        if path_node in assignment:
                            partition_id = assignment[path_node]
                            # If the partition id has already been visited then we know that it doesn't
                            # form a cycle with the current partitions being merged.
                            if partition_id in visited_partition_ids:
                                continue
                            p_map = partition_map[partition_id]
                            if self_id in p_map or other_id in p_map:
                                return True

                            visited_partition_ids.add(partition_id)

                return False

            # check if merge would create cyclic dependency.
            all_user_nodes = set()
            for node in merged_nodes:
                for user_node in node.users:
                    if user_node not in merged_nodes:
                        all_user_nodes.add(user_node)

            if dfs_iter_find_cycle(all_user_nodes):
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

            partition_map[self_id] = partition_map[self_id].union(partition_map[other_id])
            del partition_map[other_id]

            return True

        def merge_single_node(node: Node, id: Optional[int]):
            def _update_partition_map(node: Node, id: int):
                # Iterate through all the downstream nodes of this node and update the partition map
                # to indicate that there is a path from the partition id of this node to the target
                # partition id.
                downstream_nodes = self.dependency_viewer.downstreams_of(node)
                for curr_node in downstream_nodes:
                    target_id = assignment.get(curr_node, None)
                    if target_id is not None:
                        partition_map[id].add(target_id)

                # Iterate through all the upstream nodes of this node and update the partition map
                # to indicate that there is a path from the partition id of the upstream node to the
                # current node's partition id.
                upstream_nodes = self.dependency_viewer.upstreams_of(node)
                for curr_node in upstream_nodes:
                    source_id = assignment.get(curr_node, None)
                    if source_id is not None:
                        partition_map[source_id].add(id)

            if node in assignment:
                partitions_by_id[assignment[node]].remove_node(node)

            if id is None:
                assignment.pop(node)
            elif id not in partitions_by_id:
                assignment[node] = id
                partitions_by_id[id] = Partition(id=id, nodes=[node])
                _update_partition_map(node, id)
            else:
                assignment[node] = id
                partitions_by_id[id].add_node(node)
                _update_partition_map(node, id)

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

            # merge all possible partitions
            for node in assignment:
                merge_candidates[assignment[node]] = None

            merge_candidates_list = list(merge_candidates.keys())
            if len(merge_candidates_list) > 1:
                self_id = merge_candidates_list[0]
                for other_id in merge_candidates_list[1:]:
                    # note: merge partition `other_id` into partition `self_id` if
                    # it doesn't create cyclic dependency in the graph, otherwise,
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
                    if node.op == "call_function":
                        assert callable(node.target)
                        if _get_qualified_name(node.target) not in non_compute_ops:
                            compute_node_count += 1
                        if _get_qualified_name(node.target) in self.allowed_single_node_partition_ops:
                            compute_node_count += 1
                if compute_node_count <= 1:
                    partitions_to_remove.append(id)
            for id in partitions_to_remove:
                del partitions_by_id[id]

        logger.debug("Partitions proposed:")
        for id, partition in partitions_by_id.items():
            logger.debug("partition #%s: %s", id, [node.name for node in partition.nodes])

        return [partition for partition in partitions_by_id.values() if partition.size() > 0]

    def fuse_partitions(self, partitions: List[Partition], prefix: str = "fused_") -> GraphModule:
        logger.debug("Fusing partitions...")
        # fuse_by_partitions expects partitions in List[List[Node]]: [ [node0, node1], [node2, node3] ]
        return fuse_by_partitions(
            self.graph_module,
            [list(partition.nodes) for partition in partitions],
            prefix=prefix,
        )

    # remove non-compute-ops that sits at the boundary of a partition.
    def remove_bookend_non_compute_ops(self, partitions: List[Partition]):
        non_compute_ops = set(self.non_compute_ops)

        def is_non_compute_node(node: Node):
            return node.op == "call_function" and \
                _get_qualified_name(node.target) in non_compute_ops  # type: ignore[arg-type]

        # cache transparent nodes
        transparent_input_nodes: Dict[Node, bool] = {}
        transparent_output_nodes: Dict[Node, bool] = {}

        def is_transparent_input_node(node: Node, partition: Set[Node], removed_nodes: Set[Node]):
            if node.op == "placeholder" or (node not in partition) or (node in removed_nodes):
                return True
            if node in transparent_input_nodes:
                return transparent_input_nodes[node]
            if is_non_compute_node(node):
                for input_n in node.all_input_nodes:
                    if not is_transparent_input_node(input_n, partition, removed_nodes):
                        transparent_input_nodes[node] = False
                        return False
                transparent_input_nodes[node] = True
                return True
            transparent_input_nodes[node] = False
            return False

        def is_transparent_output_node(node: Node, partition: Set[Node], removed_nodes: Set[Node]):
            if node.op == "placeholder" or (node not in partition) or (node in removed_nodes):
                return True
            if node in transparent_output_nodes:
                return transparent_output_nodes[node]
            if is_non_compute_node(node):
                for output_n in node.users:
                    if not is_transparent_output_node(output_n, partition, removed_nodes):
                        transparent_output_nodes[node] = False
                        return False
                transparent_output_nodes[node] = True
                return True
            transparent_output_nodes[node] = False
            return False

        for partition in partitions:
            # Note it's ok to use `set` here, since we are only query if a node
            # has been removed. We are NEVER going to iterate on nodes inside
            # the set.
            remove_node: Set[Node] = set()
            for node in partition.nodes:
                if is_non_compute_node(node) and \
                    (is_transparent_input_node(node, partition.nodes, remove_node) or
                     is_transparent_output_node(node, partition.nodes, remove_node)):
                    remove_node.add(node)

            if len(remove_node) != 0:
                partition.nodes = partition.nodes - remove_node

    def partition_and_fuse(self, prefix: str = "fused_") -> GraphModule:
        partitions = self.propose_partitions()
        fused_gm = self.fuse_partitions(partitions, prefix=prefix)
        return fused_gm
