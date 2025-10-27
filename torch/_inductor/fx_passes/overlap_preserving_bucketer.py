from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional

import torch.fx as fx
from torch._inductor.augmented_graph_helper import AugmentedGraphHelper
from torch._inductor.fx_passes.bucketing import (
    bucket_key,
    is_all_gather_into_tensor as is_all_gather,
    is_reduce_scatter_tensor as is_reduce_scatter,
    is_wait_tensor,
)
from torch._inductor.fx_passes.overlap_scheduling import (
    CollBucket,
    CollectiveInfo,
    get_group_name,
    is_compute_node,
)
from torch.utils._ordered_set import OrderedSet


@dataclass
class Event:
    """Represents a point in the timeline with a representative operation."""

    node: fx.Node  # Single representative node
    event_type: Literal["compute", "starts", "waits"]
    position: int
    prev: Optional["Event"] = None
    next: Optional["Event"] = None

    @property
    def is_start(self) -> bool:
        return self.event_type == "starts"

    @property
    def is_wait(self) -> bool:
        return self.event_type == "waits"

    @property
    def is_compute(self) -> bool:
        return self.event_type == "compute"

    def unlink(self) -> tuple[Optional["Event"], Optional["Event"]]:
        """Remove this event from the linked list, return (prev, next)."""
        prev_event, next_event = self.prev, self.next
        if self.prev:
            self.prev.next = self.next
        if self.next:
            self.next.prev = self.prev
        self.prev = None
        self.next = None
        return prev_event, next_event

    def insert_between(
        self, prev_event: Optional["Event"], next_event: Optional["Event"]
    ) -> None:
        """Insert this event between prev_event and next_event in the linked list."""
        if prev_event:
            prev_event.next = self
        self.prev = prev_event

        if next_event:
            next_event.prev = self
        self.next = next_event


class OverlapPreservingBucketer:
    """
    Buckets collective operations while preserving compute-collective overlap relationships.
    Uses an augmented graph to track dependencies between compute and collective operations.
    """

    def __init__(
        self,
        graph: fx.Graph,
        collective_info: dict[fx.Node, CollectiveInfo],
        node_ancestors: dict[fx.Node, OrderedSet[fx.Node]],
        scheduled: OrderedSet[fx.Node],
        max_bucket_memory_gb: float = 1.0,
        max_coll_distance: int = 1000,
        insert_overlap_deps: bool = False,
    ):
        self.graph = graph
        self.collective_info = collective_info
        self.node_ancestors = node_ancestors
        self.scheduled = scheduled
        self.max_bucket_memory_gb = max_bucket_memory_gb
        self.node_idx = {n: i for i, n in enumerate(scheduled)}
        self.aug_graph = AugmentedGraphHelper(self.graph, self.node_ancestors)
        self.max_coll_distance = max_coll_distance
        self.insert_overlap_deps = insert_overlap_deps
        self.node_to_event: dict[fx.Node, Event] = {}
        self.pg_to_timeline: dict[str, Optional[Event]] = self.build_timelines()

        self._add_hiding_interval_constraints()

    def build_timelines(self) -> dict[str, Optional[Event]]:
        all_pgs = OrderedSet()
        for start in self.collective_info:
            pg = get_group_name(start)
            all_pgs.add(pg)

        pg_timeline: dict[str, Optional[Event]] = {}
        for pg in all_pgs:
            pg_timeline[pg] = self.build_timeline(pg)

        return pg_timeline

    def build_timeline(self, pg: str) -> Optional[Event]:
        head = None
        prev_event = None
        position = 0

        for node in self.scheduled:
            node_type = None

            # Determine if this node is relevant for this PG
            if node in self.collective_info and get_group_name(node) == pg:
                node_type = "starts"
            elif is_wait_tensor(node) and get_group_name(node.args[0]) == pg:
                node_type = "waits"
            elif is_compute_node(node):
                node_type = "compute"

            if node_type is None:
                continue

            event = Event(node=node, event_type=node_type, position=position)

            # Link to previous event using insert_between
            event.insert_between(prev_event, None)

            # Add sequential dependency to augmented graph
            if prev_event:
                self.aug_graph.add_extra_dep(n=event.node, dep=prev_event.node)
            else:
                head = event

            prev_event = event
            position += 1

        return head

    def _populate_node_to_event(self, pg: str) -> None:
        """Populate node_to_event mapping for a specific PG's timeline."""
        self.node_to_event.clear()
        head = self.pg_to_timeline[pg]
        curr = head
        while curr is not None:
            self.node_to_event[curr.node] = curr
            curr = curr.next

    def _add_hiding_interval_constraints(self) -> None:
        """
        Add hiding interval constraints: start -> compute -> wait.
        """
        for start, info in self.collective_info.items():
            if info.hiding_node and not info.is_exposed:
                # Enforce: start -> compute -> wait
                self.aug_graph.add_extra_dep(n=info.hiding_node, dep=start)
                self.aug_graph.add_extra_dep(n=info.wait_node, dep=info.hiding_node)

    def bucket_collectives(self) -> None:
        """Main entry point for bucketing collectives."""

        # Group collectives by PG first
        pg_collectives: dict[str, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for start in self.collective_info:
            pg = get_group_name(start)
            pg_collectives[pg].add(start)

        all_buckets: list[CollBucket] = []
        for pg, collectives in pg_collectives.items():
            # Populate node_to_event for this PG's timeline
            self._populate_node_to_event(pg)

            # Group by bucket key within this PG
            grouped_collectives: dict[object, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
            for start in collectives:
                key = bucket_key(start)
                if key is not None:
                    grouped_collectives[key].add(start)

            # Find buckets for this PG
            for collective_group in grouped_collectives.values():
                buckets = self._find_buckets(collective_group)
                all_buckets.extend(buckets)

        # Apply bucketing transformations
        # Dependencies are tracked in aug_graph.extra_deps during bucketing
        for coll_bucket in all_buckets:
            if len(coll_bucket.collectives) <= 1:
                continue

            self._apply_bucket(coll_bucket)

        # Extract all dependencies from augmented graph
        additional_deps = self.aug_graph.get_all_extra_deps()

        # Apply topological sort with all dependencies
        from torch._dynamo.graph_deduplication import _stable_topological_sort

        _stable_topological_sort(self.graph, additional_deps)

        # After topological sort, preserve dependencies using effect tokens
        if self.insert_overlap_deps:
            self._preserve_dependencies_with_tokens(additional_deps)

        self.graph.lint()

    def _find_buckets(
        self,
        collective_group: OrderedSet[fx.Node],
    ) -> list[CollBucket]:
        """Find valid buckets within a group of similar collectives."""

        max_bucket_bytes = int(self.max_bucket_memory_gb * 1024 * 1024 * 1024)
        buckets = []
        processed: OrderedSet[fx.Node] = OrderedSet()

        for start_node in collective_group:
            if start_node in processed:
                continue

            # Initialize bucket with first collective
            bucket_info = CollBucket(
                collectives=[start_node],
                total_bytes=self.collective_info[start_node].size_bytes,
            )
            processed.add(start_node)
            start_node_idx = self.node_idx[start_node]

            # TODO - limit within range
            for candidate in collective_group:
                if candidate in processed:
                    continue

                candidate_idx = self.node_idx[candidate]
                # Check if candidate is within max distance from the bucket start
                if abs(candidate_idx - start_node_idx) > self.max_coll_distance:
                    continue

                candidate_bytes = self.collective_info[candidate].size_bytes
                if bucket_info.total_bytes + candidate_bytes > max_bucket_bytes:
                    continue

                if self._can_add_to_bucket(bucket_info, candidate):
                    bucket_info.collectives.append(candidate)
                    bucket_info.total_bytes += candidate_bytes
                    processed.add(candidate)

            if len(bucket_info.collectives) > 1:
                buckets.append(bucket_info)

        return buckets

    def _ancestor_dep(self, n1: fx.Node, n2: fx.Node) -> bool:
        """Check if there's an ancestor relationship between two nodes."""
        return n1 in self.node_ancestors[n2] or n2 in self.node_ancestors[n1]

    def _get_intervals(
        self, event: Event
    ) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
        """Get (execution_interval, hiding_interval) if event is a start.

        Returns:
            (execution_interval, hiding_interval) where:
            - execution_interval is (start_pos, wait_pos) or None
            - hiding_interval is (start_pos, compute_pos) or None if no hiding node
        """
        if not event.is_start:
            return None, None

        coll = event.node
        if coll not in self.collective_info:
            return None, None

        info = self.collective_info[coll]
        execution_interval = (event.position, self.node_to_event[info.wait_node].position)

        hiding_interval = None
        if info.hiding_node:
            hiding_interval = (event.position, self.node_to_event[info.hiding_node].position)

        return execution_interval, hiding_interval

    def _preserves_hiding_intervals(
        self,
        bucket_info: CollBucket,
        candidate: fx.Node,
        start_pos: fx.Node,
        wait_pos: fx.Node,
    ) -> bool:
        """
        Check that (start_pos, wait_pos) doesn't violate any hiding intervals or collectives.

        Collects all execution and hiding intervals in the affected timeline regions,
        then checks:
        1. All bucket hiding compute stays between new start/wait
        2. No other collective's compute interval is enclosed by bucket execution interval
        3. No other collective's execution interval encloses bucket compute intervals
        """
        # Collect all collectives being bucketed
        all_bucketed_colls = [candidate] + list(bucket_info.collectives)
        all_bucketed_waits = [
            self.collective_info[coll].wait_node for coll in all_bucketed_colls
        ]

        # Collect hiding compute positions for the bucket
        bucket_hiding_compute_positions = []
        for coll in all_bucketed_colls:
            if hiding_node := self.collective_info[coll].hiding_node:
                bucket_hiding_compute_positions.append(
                    self.node_to_event[hiding_node].position
                )

        # Get new positions
        new_start_event = self.node_to_event[start_pos]
        new_wait_event = self.node_to_event[wait_pos]

        # Check 1: All bucket hiding compute must be between new start and wait
        for compute_pos in bucket_hiding_compute_positions:
            if not (new_start_event.position < compute_pos < new_wait_event.position):
                return False

        def get_wait(n: fx.Node) -> fx.Node:
            return self.collective_info[n].wait_node

        def get_pos(n: fx.Node) -> int:
            return self.node_to_event[n].position

        latest_start_pos = max(get_pos(candidate), get_pos(bucket_info.collectives[0]))
        earliest_wait_pos = min(get_pos(get_wait(candidate)), get_pos(get_wait(bucket_info.collectives[0])))

        # Bucket execution interval
        bucket_execution_interval = (new_start_event.position, new_wait_event.position)

        # Because collectives on the same PG operate under LIFO semantics, 
        # it's only possible for us to force an early realization of an unrelated collective
        # by delaying a start or raising a wait.
        # We search in the interval from old_start -> new_start, to see if would be 
        # forcing another collective to be realized prior to its hiding nodes.
        # Similarly, we search from old_wait -> new_wait, in the reverse direction, 
        # to check the same thing.

        execution_intervals = [bucket_execution_interval]
        hiding_intervals = [
            (bucket_execution_interval[0], pos)
            for pos in bucket_hiding_compute_positions
        ]

        curr_event = new_start_event.next
        while curr_event is not None and curr_event.position < latest_start_pos:
            if (
                curr_event.node not in all_bucketed_colls
                and curr_event.node not in all_bucketed_waits
            ):
                exec_interval, hiding_interval = self._get_intervals(curr_event)
                if exec_interval:
                    execution_intervals.append(exec_interval)
                if hiding_interval:
                    hiding_intervals.append(hiding_interval)
            curr_event = curr_event.next

        curr_event = new_wait_event.prev
        while curr_event is not None and curr_event.position > earliest_wait_pos:
            if (
                curr_event.node not in all_bucketed_colls
                and curr_event.node not in all_bucketed_waits
            ):
                exec_interval, hiding_interval = self._get_intervals(curr_event)
                if exec_interval:
                    execution_intervals.append(exec_interval)
                if hiding_interval:
                    hiding_intervals.append(hiding_interval)
            curr_event = curr_event.prev

        # Check: no hiding interval should be enclosed by any execution interval
        def enclosed_interval(inner: tuple[int, int], outer: tuple[int, int]) -> bool:
            return outer[0] < inner[0] and inner[1] < outer[1]

        for hiding_interval in hiding_intervals:
            for execution_interval in execution_intervals:
                if enclosed_interval(hiding_interval, execution_interval):
                    return False

        return True

    def remove_from_event(
        self, node: fx.Node
    ) -> tuple[Optional[Event], Optional[Event]]:
        """Remove node from timeline and return (prev_event, next_event)."""
        event = self.node_to_event[node]
        assert not event.is_compute, "Cannot remove compute events from timeline"

        prev_event, next_event = event.unlink()

        # Remove augmented graph dependency
        if prev_event:
            self.aug_graph.remove_extra_dep(n=node, dep=prev_event.node)
        if next_event:
            self.aug_graph.remove_extra_dep(n=next_event.node, dep=node)

        # Add bypass dependency
        if prev_event and next_event:
            self.aug_graph.add_extra_dep(n=next_event.node, dep=prev_event.node)

        return prev_event, next_event

    def restore_to_event(
        self, node: fx.Node, prev_event: Optional[Event], next_event: Optional[Event]
    ) -> None:
        """Restore node to timeline after failed merge attempt."""
        event = self.node_to_event[node]

        # Reinsert into linked list
        event.insert_between(prev_event, next_event)
        if prev_event:
            self.aug_graph.add_extra_dep(n=node, dep=prev_event.node)
        if next_event and not prev_event:
            self.aug_graph.add_extra_dep(n=next_event.node, dep=node)

        # Remove bypass dependency
        if prev_event and next_event:
            self.aug_graph.remove_extra_dep(n=next_event.node, dep=prev_event.node)

    def _try_rail_position(
        self,
        bucket_info: CollBucket,
        candidate: fx.Node,
        start_pos: fx.Node,
        wait_pos: fx.Node,
    ) -> bool:
        """
        Try a specific rail position for the candidate.
        Returns True if valid and merges are successful.
        """
        candidate_info = self.collective_info[candidate]
        candidate_start = candidate
        candidate_wait = candidate_info.wait_node

        # Quick check: does this violate hiding intervals?
        if not self._preserves_hiding_intervals(
            bucket_info, candidate, start_pos, wait_pos
        ):
            return False

        # Determine which start needs to move
        existing_coll = bucket_info.collectives[0]
        if start_pos == existing_coll:
            start_to_move = candidate
        else:
            assert start_pos == candidate
            start_to_move = existing_coll

        # Remove start from timeline
        start_prev, start_next = self.remove_from_event(start_to_move)

        # Check if starts can be merged
        if self.aug_graph.has_path(existing_coll, candidate) or self.aug_graph.has_path(
            candidate, existing_coll
        ):
            # Restore start constraints
            self.restore_to_event(start_to_move, start_prev, start_next)
            return False

        # Merge starts
        self.aug_graph.merge_to_set(existing_coll, candidate)

        # Determine which wait needs to move
        existing_wait = self.collective_info[existing_coll].wait_node
        candidate_wait = self.collective_info[candidate].wait_node

        if wait_pos == existing_wait:
            wait_to_move = candidate_wait
        else:
            wait_to_move = existing_wait

        # Remove wait from timeline
        wait_prev, wait_next = self.remove_from_event(wait_to_move)

        # Check if waits can be merged
        if self.aug_graph.has_path(
            existing_wait, candidate_wait
        ) or self.aug_graph.has_path(candidate_wait, existing_wait):
            # Restore wait constraints
            self.restore_to_event(wait_to_move, wait_prev, wait_next)
            # Unmerge the start we just merged
            self.aug_graph.unmerge_node(candidate)
            # Restore start constraints
            self.restore_to_event(start_to_move, start_prev, start_next)
            return False

        # Merge waits - success!
        self.aug_graph.merge_to_set(existing_wait, candidate_wait)

        # Update node_to_event for moved nodes
        target_start_event = self.node_to_event[start_pos]
        target_wait_event = self.node_to_event[wait_pos]

        self.node_to_event[candidate] = target_start_event
        self.node_to_event[candidate_wait] = target_wait_event
        self.node_to_event[existing_coll] = target_start_event
        self.node_to_event[existing_wait] = target_wait_event

        return True

    def _has_ancestor_conflicts(
        self, bucket_info: CollBucket, candidate: fx.Node
    ) -> bool:
        """
        Check if candidate has ancestor conflicts with bucket collectives.
        Returns True if there are conflicts.
        """
        candidate_info = self.collective_info[candidate]
        candidate_wait = candidate_info.wait_node

        for coll in bucket_info.collectives:
            # Check if collectives are ancestors of each other
            if self._ancestor_dep(coll, candidate):
                return True

            # Check if waits are ancestors of each other
            coll_wait = self.collective_info[coll].wait_node
            if self._ancestor_dep(candidate_wait, coll_wait):
                return True

            # Check if existing hiding node conflicts with candidate wait
            if hiding_node := self.collective_info[coll].hiding_node:
                if self._ancestor_dep(hiding_node, candidate_wait):
                    return True

            # Check if candidate hiding node conflicts with existing wait
            if new_hiding_node := candidate_info.hiding_node:
                if self._ancestor_dep(new_hiding_node, coll_wait):
                    return True

        return False

    def _can_add_to_bucket(
        self,
        bucket_info: CollBucket,
        candidate: fx.Node,
    ) -> bool:
        """
        Check if candidate can be added to bucket without interfering
        with comm/compute overlap.
        """
        candidate_info = self.collective_info[candidate]

        # Step 1: Quick check using precomputed ancestors
        # These ancestors are computed prior to adding augmented dependencies and not updated,
        # so if any of these checks fail then the merge will not be topologically valid
        # even ignoring comm/compute overlap
        if self._has_ancestor_conflicts(bucket_info, candidate):
            return False

        # Step 2: Try different rail positions
        existing_coll = bucket_info.collectives[0]
        existing_wait = self.collective_info[existing_coll].wait_node

        candidate_start = candidate
        candidate_wait = candidate_info.wait_node

        # Try combinations in order of likelihood to succeed
        # (early start, later wait is most likely to work)
        combinations = [
            (
                existing_coll,
                candidate_wait,
            ),  # Move candidate start early, keep wait late
            (
                existing_coll,
                existing_wait,
            ),  # Move candidate start early, move wait early
            (candidate_start, candidate_wait),  # Keep both in place
            (candidate_start, existing_wait),  # Keep start in place, move wait early
        ]

        for start_pos, wait_pos in combinations:
            if self._try_rail_position(bucket_info, candidate, start_pos, wait_pos):
                return True

        return False

    def _apply_bucket(self, bucket_info: CollBucket):
        """
        Apply bucketing transformation.

        Dependencies are added to aug_graph.extra_deps and transferred from old nodes.
        Returns (new_start, new_wait).
        """

        from torch._inductor.fx_passes.bucketing import (
            merge_all_gather_bucket,
            merge_reduce_scatter_bucket,
        )

        bucket = bucket_info.collectives

        # Collect old nodes BEFORE they're erased
        old_starts = list(bucket)
        old_waits = [self.collective_info[n].wait_node for n in bucket]

        # Find where to place the bucketed operations
        next_node = bucket[0]
        while next_node in bucket:
            next_node = next_node.next

        # Don't use wait_insertion_point - let merge functions place waits naturally
        # The wait_insertion_point feature tries to move waits to a specific location,
        # but this can cause issues when that location is one of the nodes being erased
        # Create bucketed collective (this will erase old nodes)
        if is_all_gather(bucket[0]):
            new_nodes, replacements = merge_all_gather_bucket(
                self.graph,
                bucket,
                insert_before=next_node,
                mode="custom_ops",
            )
        else:
            assert is_reduce_scatter(bucket[0])
            new_nodes, replacements = merge_reduce_scatter_bucket(
                self.graph,
                bucket,
                insert_before=next_node,
                mode="custom_ops",
            )

        # Get new nodes
        new_waits = [n for n in new_nodes if is_wait_tensor(n)]
        assert len(new_waits) == 1

        new_wait = new_waits[0]
        new_start = new_wait.args[0]
        assert isinstance(new_start, fx.Node)

        # Create mapping of all erased nodes to their replacements
        erased_to_new = {}
        for old_start in old_starts:
            erased_to_new[old_start] = new_start
        for old_wait in old_waits:
            erased_to_new[old_wait] = new_wait

        # Transfer all dependencies from old nodes to new nodes
        self.aug_graph.transfer_erased_node_deps(erased_to_new)

    def _preserve_dependencies_with_tokens(
        self, additional_deps: dict[fx.Node, OrderedSet[fx.Node]]
    ) -> None:
        """
        Preserve dependencies using effect tokens and with_effects higher-order op.

        Uses the standalone token_dependencies utility for consistent behavior
        across different overlap scheduling approaches.
        """
        from torch._inductor.fx_passes.control_dependencies import (
            preserve_node_ordering,
        )

        preserve_node_ordering(self.graph, additional_deps)
