import collections
import itertools
import logging
import operator
from collections.abc import Iterable, Sequence

from torch.fx.graph_module import GraphModule
from torch.fx.node import _get_qualified_name, Node
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Partition:
    def __init__(
        self,
        id: int | None = None,
        nodes: Iterable[Node] | None = None,
        node_orders: Iterable[int] | None = None,
    ) -> None:
        self.id = id
        self.nodes: dict[Node, int | None] = {}
        if nodes is not None:
            if node_orders is None:
                self.nodes = dict.fromkeys(nodes, None)
            else:
                nodes_list = list(nodes)
                node_orders_list = list(node_orders)
                if len(nodes_list) != len(node_orders_list):
                    raise AssertionError(
                        "nodes and node_orders must have the same length"
                    )
                self.nodes = dict(zip(nodes_list, node_orders_list))

    def __repr__(self) -> str:
        return str(self.nodes)

    def add_node(self, node: Node, node_order: int | None = None) -> None:
        self.nodes.update({node: node_order})

    def remove_node(self, node: Node) -> None:
        del self.nodes[node]

    def size(self) -> int:
        return len(self.nodes)


class _DependencyViewer:
    """Lightweight, on-demand graph traversal helpers.

    We intentionally avoid caching full transitive closures here to keep memory
    bounded on large graphs; see `propose_partitions` for the overall
    complexity trade-offs.
    """

    @staticmethod
    def downstreams_of(node: Node) -> set[Node]:
        visited: set[Node] = set()
        stack: list[Node] = list(node.users)

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(current.users)

        return visited

    @staticmethod
    def upstreams_of(node: Node) -> set[Node]:
        visited: set[Node] = set()
        stack: list[Node] = list(node.all_input_nodes)

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(current.all_input_nodes)

        return visited


class CapabilityBasedPartitioner:
    def __init__(
        self,
        graph_module: GraphModule,
        operator_support: OperatorSupportBase,
        allows_single_node_partition: bool = False,
        non_compute_ops: Sequence[str] | None = None,
        allowed_single_node_partition_ops: Sequence[str] | None = None,
        skip_horizontal_fusion: bool = False,
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
        self.skip_horizontal_fusion = skip_horizontal_fusion

    def _is_node_supported(self, node: Node) -> bool:
        return self.operator_support.is_node_supported(
            dict(self.graph_module.named_modules()), node
        )

    def _propose_partitions_skip_horizontal_fusion(self) -> dict[int, Partition]:
        # partition_map is a mapping from partition id to a set of partition id's.
        # The value set contains all the partition ids that can be reached by doing a
        # DFS starting from the partition id in the key.
        partition_map: dict[int, set[int]] = collections.defaultdict(set)

        assignment: dict[Node, int] = {}  # mapping from node to partition_id
        partitions_by_id: dict[
            int, Partition
        ] = {}  # mapping from partition_id to partition
        partitions_order: dict[
            int, int
        ] = {}  # mapping from partition_id to minimum topo order of nodes in partition
        partition_users: dict[
            int, set[Node]
        ] = {}  # mapping from partition_id to partition users
        new_partition_id = itertools.count()

        def downstream_partitions(
            user_nodes: Iterable[Node], source_ids: Iterable[int]
        ) -> set[int]:
            source_ids_set = set(source_ids)
            downstream_partition_ids: set[int] = set()
            for user_node in user_nodes:
                for path_node in itertools.chain(
                    (user_node,), _DependencyViewer.downstreams_of(user_node)
                ):
                    target_id = assignment.get(path_node)
                    if target_id is None:
                        continue
                    downstream_partition_ids.add(target_id)
                    downstream_partition_ids.update(partition_map[target_id])

            downstream_partition_ids.difference_update(source_ids_set)
            return downstream_partition_ids

        # try to merge partition other_id into partition self_id
        # merge only happens if the end graph doesn't contain cyclic dependency
        # returns `True` when merge happens, `False` otherwise.
        def maybe_merge_partition(self_id: int, other_id: int) -> tuple[int, bool]:
            # merged_nodes is the union of nodes in two partition to-be-merged
            self_nodes = partitions_by_id[self_id].nodes
            other_nodes = partitions_by_id[other_id].nodes

            def dfs_iter_find_cycle(all_user_nodes: set[Node]) -> bool:
                for user_node in all_user_nodes:
                    visited_partition_ids = set()

                    for path_node in itertools.chain(
                        (user_node,), _DependencyViewer.downstreams_of(user_node)
                    ):
                        # If any of the nodes in the dfs path of this node are in the merged_nodes
                        # list then there is a cycle in the graph.
                        if path_node in self_nodes or path_node in other_nodes:
                            return True

                        # If any of the nodes in the dfs path of this node are in the assignment
                        # map then we have to make sure that the partitions that these nodes belong
                        # to do not form a cycle with the current partitions being merged. This means
                        # iterating through all the nodes in all the partitions that are traversed in
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

            # find new partition users if merge.
            all_user_nodes = partition_users[self_id] | partition_users[other_id]
            all_user_nodes.difference_update(other_nodes, self_nodes)

            # check if merge would create cyclic dependency.
            if dfs_iter_find_cycle(all_user_nodes):
                # return false indicating cyclic dependency found and
                # merge is aborted
                return self_id, False

            # merge the smaller partition into the larger.
            merge_id, removed_id = self_id, other_id
            if len(self_nodes) < len(other_nodes):
                merge_id, removed_id = removed_id, merge_id
            # no cyclic dependency found, move forward with the merge
            # updating partition nodes
            partitions_by_id[merge_id].nodes.update(partitions_by_id[removed_id].nodes)
            # updating assignment map
            for node in partitions_by_id[removed_id].nodes:
                assignment[node] = merge_id
            # delete other partition
            del partitions_by_id[removed_id]

            partitions_order[merge_id] = min(
                partitions_order[merge_id], partitions_order[removed_id]
            )
            del partitions_order[removed_id]

            partition_map[merge_id] = partition_map[merge_id].union(
                partition_map[removed_id]
            )
            del partition_map[removed_id]

            partition_users[merge_id] = all_user_nodes
            del partition_users[removed_id]

            partition_map[merge_id].update(
                downstream_partitions(all_user_nodes, (merge_id, removed_id))
            )
            partition_map[merge_id].discard(merge_id)
            partition_map[merge_id].discard(removed_id)

            return merge_id, True

        def merge_single_node(
            node: Node, node_order: int | None, id: int | None
        ) -> None:
            def _update_partition_map(node: Node, id: int) -> None:
                # Update reachability through both assigned and unsupported users.
                partition_map[id].update(downstream_partitions(node.users, (id,)))

            if node in assignment:
                partitions_by_id[assignment[node]].remove_node(node)

            if id is None:
                assignment.pop(node, None)
            elif id not in partitions_by_id:
                assignment[node] = id
                if node_order is None:
                    raise AssertionError("node_order is required for new partitions")
                partitions_by_id[id] = Partition(
                    id=id, nodes=[node], node_orders=[node_order]
                )
                partition_users[id] = set(node.users)
                _update_partition_map(node, id)
            else:
                assignment[node] = id
                partitions_by_id[id].add_node(node, node_order)

        logger.debug("Proposing partitions with horizontal fusion disabled...")

        for node_order, node in enumerate(reversed(self.graph_module.graph.nodes)):
            # use Dict as an ordered set to ensure deterministic partitioning result, don't care value
            merge_candidates: dict[int, None] = {}
            created_partition = False

            # Note a limited horizontal fusion is enabled:
            #   when `node` is not supported, the code below attempts to fuse consumer of `node`.
            #
            # When skip_horizontal_fusion is True, only merge the newly-created
            # partition for this supported node with partitions for its direct users.
            # This preserves data-dependent vertical fusion while avoiding fusion of
            # independent consumer partitions around an unsupported node.
            if self._is_node_supported(node) and node not in assignment:
                partition_id = next(new_partition_id)
                partitions_order[partition_id] = partition_id
                merge_single_node(node, node_order, partition_id)
                merge_candidates[partition_id] = None
                created_partition = True

            if created_partition:
                for user in node.users:
                    if user in assignment:
                        merge_candidates[assignment[user]] = None

            merge_candidates_list = list(merge_candidates.keys())
            if len(merge_candidates_list) > 1:
                self_id = merge_candidates_list[0]
                for other_id in merge_candidates_list[1:]:
                    # note: merge partitions if it doesn't create cyclic dependency
                    # in the graph, otherwise, this is a no-op
                    self_id, _ = maybe_merge_partition(self_id, other_id)

        # sort partition nodes based on descending node order
        for partition in partitions_by_id.values():
            partition.nodes = dict(
                sorted(
                    partition.nodes.items(), key=operator.itemgetter(1), reverse=True
                )
            )

        # post processing to re-assign "getitem" nodes into upstream partition
        # Run iteratively until no more changes, to handle nested getitem chains
        # (e.g., getitem_619 = getitem_618[0] where getitem_618 = with_effects_167[1])
        logger.debug("Reassigning getitem nodes to its producer node's partition...")
        while True:
            nodes_reassignment: dict[Node, int | None] = {}
            for node in self.graph_module.graph.nodes:
                is_tuple_output = True
                for user in node.users:
                    if (
                        user.op != "call_function"
                        or _get_qualified_name(user.target) != "_operator.getitem"
                    ):  # type: ignore[arg-type]
                        is_tuple_output = False
                        break

                # node has tuple outputs, re-assign all following getitem node into node's partition
                if is_tuple_output:
                    id = assignment.get(node)  # type: ignore[arg-type]
                    for user in node.users:
                        if assignment.get(user) != id:  # type: ignore[arg-type]
                            nodes_reassignment[user] = id  # type: ignore[assignment]

            # no more re-assignments
            if not nodes_reassignment:
                break

            for node, id in nodes_reassignment.items():
                merge_single_node(node, None, id)

        return partitions_by_id

    def _finalize_partitions(
        self, partitions_by_id: dict[int, Partition]
    ) -> list[Partition]:
        # filter out single node partitions
        if not self.allows_single_node_partition:
            logger.debug("Filtering out single node partitions...")
            default_non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
            non_compute_ops = default_non_compute_ops.union(set(self.non_compute_ops))
            partitions_to_remove: list[int] = []
            for id, partition in partitions_by_id.items():
                compute_node_count = 0
                for node in partition.nodes:
                    if node.op == "call_function":
                        if not callable(node.target):
                            raise AssertionError(
                                f"Expected callable target, got {type(node.target)}"
                            )
                        if _get_qualified_name(node.target) not in non_compute_ops:
                            compute_node_count += 1
                        if (
                            _get_qualified_name(node.target)
                            in self.allowed_single_node_partition_ops
                        ):
                            compute_node_count += 1
                if compute_node_count <= 1:
                    partitions_to_remove.append(id)
            for id in partitions_to_remove:
                del partitions_by_id[id]

        logger.debug("Partitions proposed:")
        for id, partition in partitions_by_id.items():
            logger.debug(
                "partition #%s: %s", id, [node.name for node in partition.nodes]
            )

        return [
            partition for partition in partitions_by_id.values() if partition.size() > 0
        ]

    def propose_partitions(self) -> list[Partition]:
        """Group supported nodes into partitions while avoiding cycles.

        Two paths exist:
        1) Fast path (no potentially cyclic unsupported nodes): one pass builds
           a single partition of all supported nodes.
        2) General path: a greedy, depth-bounded scan forms multiple partitions
           while skipping unsupported nodes that would create cycles.

        Assumptions:
        - `graph.nodes` iteration is in topological order (FX invariant).
        - Traversals use reversed order to walk from outputs toward inputs.

        Complexity:
        - Fast path: O(|V|) time, O(|V|) space.
        - General path: O(|V|*|U|) time, O(|V|) space. (|U| unsupported nodes, assuming |E|≈2·|V|)
          Runtime stays below O(|V|^2) because:
          * depth-window breaking halts scans once we move too far upstream,
          * already-assigned nodes are skipped, limiting revisits,
          * blocklists prune entire upstreams of unsupported nodes
          * on-demand DFS runs only for encountered unsupported nodes.
        """
        if self.skip_horizontal_fusion:
            return self._finalize_partitions(
                self._propose_partitions_skip_horizontal_fusion()
            )

        assignment: dict[Node, int] = {}  # mapping from node to partition_id
        # mapping from partition_id to partition
        partitions_by_id: dict[int, Partition] = {}
        supported_map: dict[Node, bool] = {}
        nodes: list[Node] = list(self.graph_module.graph.nodes)

        needs_cycle_detection = False
        for node in nodes:
            is_supported = self._is_node_supported(node)
            supported_map[node] = is_supported
            if not needs_cycle_detection and not is_supported:
                needs_cycle_detection = (
                    len(node.all_input_nodes) > 0 and len(node.users) > 0
                )

        if not needs_cycle_detection:
            logger.debug("Proposing partitions with fast path (no cycles possible)...")
            nodes_in_partition: list[Node] = []
            node_orders: list[int] = []
            for node_order, node in enumerate(reversed(nodes)):
                if supported_map[node]:
                    nodes_in_partition.append(node)
                    node_orders.append(node_order)

            partitions_by_id = {
                0: Partition(id=0, nodes=nodes_in_partition, node_orders=node_orders)
            }

        else:

            def compute_depths(
                nodes: list[Node],
            ) -> tuple[dict[Node, int], dict[Node, int]]:
                """Return depth maps from outputs (bottom-up) and inputs (top-down).

                - Output depth grows as we walk upstream from graph outputs.
                - Input depth grows as we walk downstream from graph inputs.
                """
                output_depth: dict[Node, int] = {}
                for node in reversed(nodes):
                    if not node.users:
                        output_depth[node] = 0
                    else:
                        output_depth[node] = 1 + min(
                            output_depth[user] for user in node.users
                        )

                input_depth: dict[Node, int] = {}
                for node in nodes:
                    if not node.all_input_nodes:
                        input_depth[node] = 0
                    else:
                        input_depth[node] = 1 + min(
                            input_depth[input_n] for input_n in node.all_input_nodes
                        )

                return output_depth, input_depth

            def greedy_partition(partition_id: int, start_index: int) -> None:
                """Greedily pull supported upstream nodes while steering around
                unsupported ops that already feed the current partition.

                Blocklist grows when an unsupported node would flow into the
                current partition; its upstreams are skipped for this build.
                Scanning also halts if we step beyond the depth window relative
                to the last node already placed in the partition.
                """
                blocklist: set[Node] = set()
                last_added_output_depth = -1
                last_added_input_depth = -1
                current_partition = Partition(id=partition_id)
                partitions_by_id[partition_id] = current_partition

                def depth_window_exceeded() -> bool:
                    """Return True only if both depth windows are exceeded.

                    - Output depth increases as we walk upstream.
                    - Input depth decreases as we walk upstream.
                    """
                    output_exceeded = (
                        output_depth[candidate_node] > last_added_output_depth + 1
                    )
                    input_exceeded = (
                        input_depth[candidate_node] < last_added_input_depth - 1
                    )
                    return output_exceeded and input_exceeded

                for idx in range(start_index, len(nodes)):
                    node_idx = len(nodes) - 1 - idx
                    candidate_node = nodes[node_idx]
                    if candidate_node in assignment:
                        if depth_window_exceeded():
                            break
                        continue
                    if candidate_node in blocklist:
                        if depth_window_exceeded():
                            break
                        continue
                    if not supported_map[candidate_node]:
                        if (
                            _DependencyViewer.downstreams_of(candidate_node)
                            & current_partition.nodes.keys()
                        ):
                            blocklist.update(
                                _DependencyViewer.upstreams_of(candidate_node)
                            )
                        continue
                    assignment[candidate_node] = partition_id
                    current_partition.add_node(candidate_node, idx)
                    last_added_output_depth = output_depth[candidate_node]
                    last_added_input_depth = input_depth[candidate_node]

            def reassign_node_to_partition(node: Node, target_id: int | None) -> None:
                """Reassign `node` to `target_id`, preserving stored order.

                - If `target_id` is None: unassign and remove the node.
                - If already in `target_id`: no-op.
                - Cleans up empty partitions after moving.
                """
                if target_id is None:
                    if node in assignment:
                        partitions_by_id[assignment[node]].remove_node(node)
                        assignment.pop(node)
                    return

                current_id = assignment.get(node)
                if current_id == target_id:
                    return

                node_order = None
                if current_id is not None:
                    partition = partitions_by_id.get(current_id)
                    if partition is not None and node in partition.nodes:
                        node_order = partition.nodes.pop(node)
                        if partition.size() == 0:
                            partitions_by_id.pop(current_id, None)

                if target_id not in partitions_by_id:
                    partitions_by_id[target_id] = Partition(id=target_id)
                partitions_by_id[target_id].add_node(node, node_order)
                assignment[node] = target_id

            logger.debug(
                "Proposing partitions with general path (cycle detection enabled)..."
            )

            output_depth, input_depth = compute_depths(nodes)

            partition_id = 0
            for i, node in enumerate(reversed(nodes)):
                if node in assignment:
                    continue
                if not supported_map[node]:
                    continue
                partition_id += 1
                greedy_partition(partition_id, i)

            # post processing to re-assign "getitem" nodes into upstream partition
            # Run iteratively until no more changes, to handle nested getitem chains
            # (e.g., getitem_619 = getitem_618[0] where getitem_618 = with_effects_167[1])
            logger.debug(
                "Reassigning getitem nodes to its producer node's partition..."
            )
            while True:
                nodes_reassignment: dict[Node, int | None] = {}
                for node in self.graph_module.graph.nodes:
                    is_tuple_output = True
                    for user in node.users:
                        if (
                            user.op != "call_function"
                            or _get_qualified_name(user.target) != "_operator.getitem"
                        ):  # type: ignore[arg-type]
                            is_tuple_output = False
                            break

                    # node has tuple outputs, re-assign all following getitem node into node's partition
                    if is_tuple_output:
                        id = assignment.get(node)  # type: ignore[arg-type]
                        for user in node.users:
                            if assignment.get(user) != id:  # type: ignore[arg-type]
                                nodes_reassignment[user] = id  # type: ignore[assignment]

                # no more re-assignments
                if not nodes_reassignment:
                    break

                for node, id in nodes_reassignment.items():
                    reassign_node_to_partition(node, id)

            # sort partition nodes based on descending node order
            for partition in partitions_by_id.values():
                partition.nodes = dict(
                    sorted(
                        partition.nodes.items(),
                        key=operator.itemgetter(1),
                        reverse=True,
                    )
                )

        # filter out single node partitions
        if not self.allows_single_node_partition:
            logger.debug("Filtering out single node partitions...")
            default_non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
            non_compute_ops = default_non_compute_ops.union(set(self.non_compute_ops))
            partitions_to_remove: list[int] = []
            for id, partition in partitions_by_id.items():
                compute_node_count = 0
                for node in partition.nodes:
                    if node.op == "call_function":
                        if not callable(node.target):
                            raise AssertionError(
                                f"Expected callable target, got {type(node.target)}"
                            )
                        if _get_qualified_name(node.target) not in non_compute_ops:
                            compute_node_count += 1
                        if (
                            _get_qualified_name(node.target)
                            in self.allowed_single_node_partition_ops
                        ):
                            compute_node_count += 1
                if compute_node_count <= 1:
                    partitions_to_remove.append(id)
            for id in partitions_to_remove:
                del partitions_by_id[id]

        logger.debug("Partitions proposed:")
        for id, partition in partitions_by_id.items():
            logger.debug(
                "partition #%s: %s", id, [node.name for node in partition.nodes]
            )

        return [
            partition for partition in partitions_by_id.values() if partition.size() > 0
        ]

    def fuse_partitions(
        self, partitions: list[Partition], prefix: str = "fused_"
    ) -> GraphModule:
        logger.debug("Fusing partitions...")
        # fuse_by_partitions expects partitions in List[Dict[Node, None]]: [ {node0 : None}, {node1 : None} ]
        return fuse_by_partitions(
            self.graph_module,
            [partition.nodes for partition in partitions],
            prefix=prefix,
        )

    # remove non-compute-ops that sits at the boundary of a partition.
    def remove_bookend_non_compute_ops(self, partitions: list[Partition]) -> None:
        non_compute_ops = set(self.non_compute_ops)

        def is_non_compute_node(node: Node) -> bool:
            return (
                node.op == "call_function"
                and _get_qualified_name(node.target) in non_compute_ops  # type: ignore[arg-type]
            )

        # cache transparent nodes
        transparent_input_nodes: dict[Node, bool] = {}
        transparent_output_nodes: dict[Node, bool] = {}

        def is_transparent_input_node(
            node: Node, partition: set[Node], removed_nodes: set[Node]
        ) -> bool:
            if (
                node.op == "placeholder"
                or (node not in partition)
                or (node in removed_nodes)
            ):
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

        def is_transparent_output_node(
            node: Node, partition: set[Node], removed_nodes: set[Node]
        ) -> bool:
            if (
                node.op == "placeholder"
                or (node not in partition)
                or (node in removed_nodes)
            ):
                return True
            if node in transparent_output_nodes:
                return transparent_output_nodes[node]
            if is_non_compute_node(node):
                for output_n in node.users:
                    if not is_transparent_output_node(
                        output_n, partition, removed_nodes
                    ):
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
            remove_node: set[Node] = set()
            for node in partition.nodes:
                if is_non_compute_node(node) and (
                    is_transparent_input_node(node, set(partition.nodes), remove_node)
                    or is_transparent_output_node(
                        node, set(partition.nodes), remove_node
                    )
                ):
                    remove_node.add(node)

            if len(remove_node) != 0:
                for node in remove_node:
                    partition.nodes.pop(node, None)

    def partition_and_fuse(self, prefix: str = "fused_") -> GraphModule:
        partitions = self.propose_partitions()
        fused_gm = self.fuse_partitions(partitions, prefix=prefix)
        return fused_gm
