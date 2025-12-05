import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor.augmented_graph_helper import AugmentedGraphHelper
from torch._inductor.fx_passes.bucketing import (
    _schedulable_wait_node,
    bucket_key,
    BucketMode,
    has_mergeable_all_gather_convert_dtype,
    is_all_gather_into_tensor as is_all_gather,
    is_reduce_scatter_tensor as is_reduce_scatter,
)
from torch._inductor.fx_passes.overlap_scheduling import (
    CollBucket,
    CollectiveInfo,
    get_group_name,
    is_compute_node,
)
from torch.utils._ordered_set import OrderedSet


bucket_log = logging.getLogger(__name__)


@dataclass
class WhyNoBucket:
    name1: str
    name2: str
    reason: str
    args: tuple[Any, ...]

    def __init__(self, node1: fx.Node, node2: fx.Node) -> None:
        self.name1 = node1.name
        self.name2 = node2.name
        self.reason = ""
        self.args = ()

    def __call__(self, reason: str, *args: Any) -> None:
        if bucket_log.isEnabledFor(logging.DEBUG):
            bucket_log.debug(
                "cannot bucket %s with %s: " + reason,  # noqa: G003
                self.name1,
                self.name2,
                *args,
            )


def is_collective_or_wait(n: fx.Node) -> bool:
    """Check if node is a collective start or wait."""
    if _schedulable_wait_node(n):
        return True
    # Collective starts have exactly one use: the wait_tensor
    if len(n.users) == 1:
        user = next(iter(n.users.keys()))
        if _schedulable_wait_node(user):
            return True
    return False


@dataclass
class PGEvent:
    """
    Represents an important event in a process group timeline. Either
    a collective start, wait, or hiding compute. Each node is linked
    to its prev and next and these dependencies are reflected
    in the augmented graph.

    We want to enforce a sequential ordering of collective starts and waits
    because NCCL collectives on the same process group execute on the same CUDA
    stream, creating implicit dependencies between all operations on that PG.

    A wait of a particular collective will implicitly force realization of all collectives
    enqueued prior to that collective.
    """

    node: fx.Node
    event_type: Literal["compute", "starts", "waits"]
    position: int
    prev: Optional["PGEvent"] = None
    next: Optional["PGEvent"] = None

    @property
    def is_start(self) -> bool:
        return self.event_type == "starts"

    @property
    def is_wait(self) -> bool:
        return self.event_type == "waits"

    @property
    def is_compute(self) -> bool:
        return self.event_type == "compute"

    def unlink(self) -> tuple[Optional["PGEvent"], Optional["PGEvent"]]:
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
        self, prev_event: Optional["PGEvent"], next_event: Optional["PGEvent"]
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
        gm: fx.GraphModule,
        collective_info: dict[fx.Node, CollectiveInfo],
        scheduled: OrderedSet[fx.Node],
        max_bucket_memory_gb: float = 1.0,
        max_coll_distance: int = 1000,
        insert_overlap_deps: bool = False,
        bucket_mode: BucketMode = "custom_ops_multidtype",
        region_of: dict[fx.Node, any] = None,
        replaced: dict[fx.Node, fx.Node] = None,
    ):
        self.gm = gm
        self.graph = gm.graph
        self.collective_info = collective_info
        self.max_bucket_memory_gb = max_bucket_memory_gb
        self.max_coll_distance = max_coll_distance
        self.insert_overlap_deps = insert_overlap_deps
        self.bucket_mode = bucket_mode
        self.region_of = region_of or {}
        self.replaced = replaced or {}  # Tracks collapsed nodes for later expansion
        self.node_to_event: dict[fx.Node, PGEvent] = {}
        self.all_hiding_nodes: OrderedSet[fx.Node] = OrderedSet()

        # Update scheduled to reflect collapsed nodes
        # Nodes that were collapsed are replaced by their subgraph representatives
        self.scheduled = self._update_scheduled_for_collapsed(scheduled)
        self.node_idx = {n: i for i, n in enumerate(self.scheduled)}

        # Build fusion region helpers
        self._region_representatives = self._build_region_representatives()

        # Compute ancestors including original graph edges and hiding interval dependencies
        self.node_ancestors = self._compute_node_ancestors()
        self.aug_graph = AugmentedGraphHelper(self.graph, self.node_ancestors)

        # Build timelines and add constraints to aug_graph
        self.pg_to_timeline_head: dict[str, Optional[PGEvent]] = self.build_timelines()
        self._add_hiding_interval_constraints()

    def _update_scheduled_for_collapsed(
        self, scheduled: OrderedSet[fx.Node]
    ) -> OrderedSet[fx.Node]:
        """
        Update scheduled to reflect collapsed fusion regions.

        Nodes that were collapsed into subgraphs are removed and replaced
        by the subgraph node at their first occurrence position.
        """
        if not self.replaced:
            return scheduled

        # Build set of valid graph nodes for quick lookup
        valid_nodes = set(self.graph.nodes)

        # Track which replacement nodes we've already added
        added_replacements: set[fx.Node] = set()

        new_scheduled: OrderedSet[fx.Node] = OrderedSet()
        for node in scheduled:
            if node in valid_nodes:
                # Node is still valid in the graph
                new_scheduled.add(node)
            elif node in self.replaced:
                # Node was collapsed - add its replacement if not already added
                replacement = self.replaced[node]
                if replacement not in added_replacements:
                    added_replacements.add(replacement)
                    new_scheduled.add(replacement)
            # else: node was erased (e.g., internal fusion region node), skip it

        # Also update hiding_nodes in collective_info to use replacement nodes
        self._update_collective_info_for_collapsed(valid_nodes)

        return new_scheduled

    def _update_collective_info_for_collapsed(
        self, valid_nodes: set[fx.Node]
    ) -> None:
        """
        Update hiding_nodes in collective_info to use replacement nodes.

        Nodes that were collapsed into subgraphs need to be replaced with
        their subgraph representative.
        """
        for info in self.collective_info.values():
            updated_hiding_nodes: OrderedSet[fx.Node] = OrderedSet()
            for hn in info.hiding_nodes:
                if hn in valid_nodes:
                    updated_hiding_nodes.add(hn)
                elif hn in self.replaced:
                    updated_hiding_nodes.add(self.replaced[hn])
                # else: node was erased and not in replaced, skip
            info.hiding_nodes = updated_hiding_nodes

    def _build_region_representatives(self) -> dict[int, fx.Node]:
        """Build a mapping from region ID to representative node (last node in region)."""
        region_representatives = {}
        if self.region_of:
            # Use id() to get unique regions since FusionRegion is not hashable
            seen_region_ids: set[int] = set()
            for region in self.region_of.values():
                region_id = id(region)
                if region_id not in seen_region_ids:
                    seen_region_ids.add(region_id)
                    # Use the last node (anchor) as the representative
                    region_representatives[region_id] = region.end
        return region_representatives

    def _get_region_for_node(self, node: fx.Node):
        """Get the fusion region for a node, if any."""
        return self.region_of.get(node)

    def _get_region_cost(self, node: fx.Node) -> float:
        """Get the cost of the fusion region containing this node, or 0 if no region."""
        region = self._get_region_for_node(node)
        return region.cost_ms if region else 0.0

    def _nodes_in_same_region(self, node1: fx.Node, node2: fx.Node) -> bool:
        """Check if two nodes belong to the same fusion region."""
        region1 = self._get_region_for_node(node1)
        region2 = self._get_region_for_node(node2)
        return region1 is not None and region1 is region2

    def _get_fusion_group_representative(self, node: fx.Node) -> fx.Node:
        """Get the fusion group output/representative for a node, or the node itself."""
        region = self._get_region_for_node(node)
        if region is not None:
            # Return the last node (output) of the fusion region
            return region.end
        return node

    def _redirect_deps_to_fusion_outputs(self, deps_map: dict[fx.Node, OrderedSet[fx.Node]]) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """Redirect dependencies to target fusion group outputs instead of internal nodes."""
        if not self.region_of:
            return deps_map

        redirected_deps: dict[fx.Node, OrderedSet[fx.Node]] = {}

        for node, deps in deps_map.items():
            # Get representative for the target node
            rep_node = self._get_fusion_group_representative(node)

            # Get representatives for all dependency nodes
            rep_deps = OrderedSet()
            for dep in deps:
                rep_dep = self._get_fusion_group_representative(dep)
                # Only add if not self-dependency after redirection
                if rep_dep != rep_node:
                    rep_deps.add(rep_dep)

            # Only add if we have actual dependencies
            if rep_deps:
                if rep_node in redirected_deps:
                    redirected_deps[rep_node].update(rep_deps)
                else:
                    redirected_deps[rep_node] = rep_deps

        return redirected_deps

    def _compute_node_ancestors(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """
        Compute ancestor sets for all nodes including:
        1. Original graph edges
        2. Hiding interval deps: collective_start -> hiding_node -> wait
        """
        augmented_inputs: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for start, info in self.collective_info.items():
            if info.is_exposed:
                continue
            for hiding_node in info.hiding_nodes:
                augmented_inputs[hiding_node].add(start)
                augmented_inputs[info.wait_node].add(hiding_node)

        node_ancestors: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in self.scheduled:
            for input_node in itertools.chain(
                augmented_inputs[node], node.all_input_nodes
            ):
                node_ancestors[node].add(input_node)
                node_ancestors[node] |= node_ancestors[input_node]

        return node_ancestors

    def build_timelines(self) -> dict[str, Optional[PGEvent]]:
        "Construct each process groups ordered series of event"
        all_pgs: OrderedSet[str] = OrderedSet()
        for start in self.collective_info:
            pg = get_group_name(start)
            all_pgs.add(pg)

        pg_timeline: dict[str, Optional[PGEvent]] = {}
        for pg in all_pgs:
            pg_timeline[pg] = self.build_timeline(pg)

        return pg_timeline

    def build_timeline(self, pg: str) -> Optional[PGEvent]:
        """
        Build a timeline of important events (starts, waits, hiding compute) for this process group
        and constrain this ordering in the augmented graph.

        Sequential dependencies are added between all events because NCCL collectives on the same
        process group execute on the same CUDA stream, enforcing LIFO semantics where later-issued
        collectives must complete before earlier ones can finish.
        """

        head = None
        prev_event = None
        position = 0
        hiding_nodes = OrderedSet()

        for node in self.scheduled:
            node_type = None

            # Determine if this node is relevant for this PG
            if node in self.collective_info and get_group_name(node) == pg:
                node_type = "starts"
                hiding_nodes |= self.collective_info[node].hiding_nodes
            elif _schedulable_wait_node(node):
                wait_input = node.args[0]
                if isinstance(wait_input, fx.Node) and get_group_name(wait_input) == pg:
                    node_type = "waits"
            elif is_compute_node(node) or node in hiding_nodes:
                node_type = "compute"

            if node_type is None:
                continue

            event = PGEvent(node=node, event_type=node_type, position=position)  # type: ignore[arg-type]

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
        head = self.pg_to_timeline_head[pg]
        curr = head
        while curr is not None:
            self.node_to_event[curr.node] = curr
            curr = curr.next

    def _add_hiding_interval_constraints(self) -> None:
        """
        Add hiding interval constraints: start -> compute -> wait.
        """
        for start, info in self.collective_info.items():
            if info.is_exposed:
                continue
            for hn in info.hiding_nodes:
                # Enforce: start -> compute -> wait
                self.aug_graph.add_extra_dep(n=hn, dep=start)
                self.aug_graph.add_extra_dep(n=info.wait_node, dep=hn)

            self.all_hiding_nodes |= info.hiding_nodes

    def bucket_collectives(self) -> None:
        # Build region cost map for fusion-aware bucketing
        region_costs = {}
        if self.region_of:
            # Use id() to get unique regions since FusionRegion is not hashable
            seen_region_ids: set[int] = set()
            for region in self.region_of.values():
                region_id = id(region)
                if region_id not in seen_region_ids:
                    seen_region_ids.add(region_id)
                    region_costs[region_id] = region.cost_ms

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
            grouped_collectives: dict[object, OrderedSet[fx.Node]] = defaultdict(
                OrderedSet
            )
            for start in collectives:
                key = bucket_key(start, self.bucket_mode)
                if key is not None:
                    grouped_collectives[key].add(start)

            # Find buckets for this PG
            for key, collective_group in grouped_collectives.items():
                bucket_log.debug(
                    "bucketing collective group with key %s: %s",
                    key,
                    [n.name for n in collective_group],
                )
                buckets = self._find_buckets(collective_group)
                all_buckets.extend(buckets)

        # Apply bucketing transformations
        # Dependencies are tracked in aug_graph.extra_deps during bucketing
        for coll_bucket in all_buckets:
            if len(coll_bucket.collectives) <= 1:
                continue

            counters["inductor"]["collective_buckets"] += 1
            self._apply_bucket(coll_bucket)

        # Expand collapsed fusion regions back to original nodes
        if self.region_of and self.replaced:
            from torch._inductor.fx_passes.fusion_regions import expand_fusion_regions

            self.replaced = expand_fusion_regions(
                self.gm, self.region_of, self.replaced
            )
            # Fixup aug_graph deps to use the new node identities
            self.aug_graph.fixup_replaced_nodes(self.replaced)
            # Clear region_of since the regions are now inlined
            # and the mapping is stale (points to erased module nodes)
            self.region_of = {}

        # Extract all dependencies from augmented graph
        # This includes:
        # - Sequential timeline deps (added during build_timeline)
        # - Hiding interval deps (added during _add_hiding_interval_constraints)
        # - All transferred deps from bucketing (transferred during _apply_bucket)
        additional_deps = self.aug_graph.get_all_extra_deps()

        # Filter to only include valid graph nodes
        valid_nodes = set(self.graph.nodes)
        filtered_additional_deps: dict[fx.Node, OrderedSet[fx.Node]] = {}
        for n, deps in additional_deps.items():
            if n not in valid_nodes:
                continue
            valid_deps = OrderedSet(d for d in deps if d in valid_nodes)
            if valid_deps:
                filtered_additional_deps[n] = valid_deps
        additional_deps = filtered_additional_deps

        # Redirect dependencies to fusion region outputs BEFORE topological sort
        # This ensures the sort and effect tokens use consistent node references
        if self.region_of:
            additional_deps = self._redirect_deps_to_fusion_outputs(additional_deps)

        # Apply topological sort with all dependencies (now fusion-aware)
        from torch._dynamo.graph_deduplication import _stable_topological_sort

        for n, deps in additional_deps.items():
            torch._check(
                not n._erased, lambda: f"Erased node deps not transferred: {n}"
            )
            for d in deps:
                torch._check(
                    not d._erased, lambda: f"Erased node deps not transferred: {d}"
                )

        _stable_topological_sort(self.graph, additional_deps)

        # After topological sort, preserve dependencies using effect tokens
        # Only preserve edges where NOT both nodes are collective starts or waits
        if self.insert_overlap_deps:
            filtered_deps: dict[fx.Node, OrderedSet[fx.Node]] = {}
            for node, deps in additional_deps.items():
                filtered_node_deps: OrderedSet[fx.Node] = OrderedSet()

                # only preserve comm-compute overlap for now, although we could more
                # generally constrain
                for dep in deps:
                    if not (is_collective_or_wait(node) and is_collective_or_wait(dep)):
                        filtered_node_deps.add(dep)

                if filtered_node_deps:
                    filtered_deps[node] = filtered_node_deps

            # filtered_deps is already fusion-aware since additional_deps was redirected
            self._preserve_dependencies_with_tokens(filtered_deps)

        self.graph.lint()

    def _find_buckets(
        self,
        collective_group: OrderedSet[fx.Node],
    ) -> list[CollBucket]:
        """Find valid buckets within a group of similar collectives."""
        max_bucket_bytes = int(self.max_bucket_memory_gb * 1024 * 1024 * 1024)
        buckets = []
        processed: OrderedSet[fx.Node] = OrderedSet()

        # Sort collectives by node index for efficient distance checking
        sorted_collectives = sorted(collective_group, key=lambda n: self.node_idx[n])

        for i, start_node in enumerate(sorted_collectives):
            if start_node in processed:
                continue

            if (
                start_node in self.all_hiding_nodes
                or self.collective_info[start_node].wait_node in self.all_hiding_nodes
            ):
                continue

            # Initialize bucket with first collective
            bucket_info = CollBucket(
                collectives=[start_node],
                total_bytes=self.collective_info[start_node].size_bytes,
            )
            processed.add(start_node)

            # Greedy optimization: stop after consecutive failures
            consecutive_failures = 0
            max_consecutive_failures = 20

            # Check candidates in sorted order, break when beyond max distance
            for candidate in sorted_collectives[i + 1 : i + 1 + self.max_coll_distance]:
                candidate_bytes = self.collective_info[candidate].size_bytes
                # proxy on memory use, if we see a too large bucket,
                # dont look for another, later bucket
                if bucket_info.total_bytes + candidate_bytes > max_bucket_bytes:
                    break

                if candidate in processed:
                    continue

                if self._can_add_to_bucket(bucket_info, candidate):
                    bucket_info.collectives.append(candidate)
                    bucket_info.total_bytes += candidate_bytes
                    processed.add(candidate)
                    consecutive_failures = 0  # Reset on success
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        break

            if len(bucket_info.collectives) > 1:
                buckets.append(bucket_info)

        return buckets

    def _ancestor_dep(self, n1: fx.Node, n2: fx.Node) -> bool:
        """Check if there's an ancestor relationship between two nodes."""
        return n1 in self.node_ancestors[n2] or n2 in self.node_ancestors[n1]

    def _get_intervals(
        self, event: PGEvent
    ) -> tuple[Optional[tuple[int, int]], list[tuple[int, int]]]:
        """Get (execution_interval, hiding_intervals) for a collective event."""
        # For start events, directly use the node
        if event.is_start:
            coll = event.node
        # For wait events, look up the start node from the event's args
        elif event.is_wait:
            wait_input = event.node.args[0]
            if not isinstance(wait_input, fx.Node):
                return None, []
            coll = wait_input
        else:
            return None, []

        if coll not in self.collective_info:
            return None, []

        info = self.collective_info[coll]
        start_event = self.node_to_event[coll]
        wait_event = self.node_to_event[info.wait_node]

        execution_interval = (start_event.position, wait_event.position)

        hiding_intervals = []
        if info.hiding_nodes:
            for hiding_node in info.hiding_nodes:
                if hiding_node in self.node_to_event:
                    hiding_intervals.append(
                        (
                            start_event.position,
                            self.node_to_event[hiding_node].position,
                        )
                    )

        return execution_interval, hiding_intervals

    def _preserves_hiding_intervals(
        self,
        bucket_info: CollBucket,
        candidate: fx.Node,
        start_pos: fx.Node,
        wait_pos: fx.Node,
        why: WhyNoBucket,
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
            for coll_hiding_node in self.collective_info[coll].hiding_nodes:
                if coll_hiding_node in self.node_to_event:
                    bucket_hiding_compute_positions.append(
                        self.node_to_event[coll_hiding_node].position
                    )

        # Get new positions
        new_start_event = self.node_to_event[start_pos]
        new_wait_event = self.node_to_event[wait_pos]

        # Check 1: All bucket hiding compute must be between new start and wait
        for compute_pos in bucket_hiding_compute_positions:
            if not (new_start_event.position < compute_pos < new_wait_event.position):
                why(
                    "hiding compute at pos %d not between start %d and wait %d",
                    compute_pos,
                    new_start_event.position,
                    new_wait_event.position,
                )
                return False

        def get_wait(n: fx.Node) -> fx.Node:
            return self.collective_info[n].wait_node

        def get_pos(n: fx.Node) -> int:
            return self.node_to_event[n].position

        latest_start_pos = max(get_pos(candidate), get_pos(bucket_info.collectives[0]))
        earliest_wait_pos = min(
            get_pos(get_wait(candidate)), get_pos(get_wait(bucket_info.collectives[0]))
        )

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
                exec_interval, hiding_interval_list = self._get_intervals(curr_event)
                if exec_interval:
                    execution_intervals.append(exec_interval)
                hiding_intervals.extend(hiding_interval_list)
            curr_event = curr_event.next

        curr_event = new_wait_event.prev
        while curr_event is not None and curr_event.position > earliest_wait_pos:
            if (
                curr_event.node not in all_bucketed_colls
                and curr_event.node not in all_bucketed_waits
            ):
                exec_interval, hiding_interval_list = self._get_intervals(curr_event)
                if exec_interval:
                    execution_intervals.append(exec_interval)
                hiding_intervals.extend(hiding_interval_list)
            curr_event = curr_event.prev

        # Check: no hiding interval should be enclosed by any execution interval
        def enclosed_interval(inner: tuple[int, int], outer: tuple[int, int]) -> bool:
            return outer[0] < inner[0] and inner[1] < outer[1]

        for hiding_interval in hiding_intervals:
            for execution_interval in execution_intervals:
                if enclosed_interval(hiding_interval, execution_interval):
                    why(
                        "hiding interval %s enclosed by execution interval %s",
                        hiding_interval,
                        execution_interval,
                    )
                    return False

        return True

    def remove_from_event(
        self, node: fx.Node
    ) -> tuple[Optional[PGEvent], Optional[PGEvent]]:
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
        self,
        node: fx.Node,
        prev_event: Optional[PGEvent],
        next_event: Optional[PGEvent],
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

    def _try_timeline_position(
        self,
        bucket_info: CollBucket,
        candidate: fx.Node,
        start_pos: fx.Node,
        wait_pos: fx.Node,
        why: WhyNoBucket,
    ) -> bool:
        """
        Try a specific timeline position for the candidate.
        Returns True if valid and merges are successful.
        """
        candidate_info = self.collective_info[candidate]
        candidate_wait = candidate_info.wait_node

        # Quick check: does this violate hiding intervals?
        if not self._preserves_hiding_intervals(
            bucket_info, candidate, start_pos, wait_pos, why
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
            why("path exists between starts")
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
            why("path exists between waits")
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
            if (
                coll in self.node_ancestors[candidate]
                or candidate in self.node_ancestors[coll]
            ):
                return True

            # Check if waits are ancestors of each other
            coll_wait = self.collective_info[coll].wait_node
            if (
                coll_wait in self.node_ancestors[candidate_wait]
                or candidate_wait in self.node_ancestors[coll_wait]
            ):
                return True

            # Check if existing hiding node conflicts with candidate wait
            for old_hiding_node in self.collective_info[coll].hiding_nodes:
                if candidate_wait in self.node_ancestors[old_hiding_node]:
                    return True

            # Check if candidate hiding node conflicts with existing wait
            for new_hiding_node in candidate_info.hiding_nodes:
                if coll_wait in self.node_ancestors[new_hiding_node]:
                    return True

        return False

    def _can_add_to_bucket(
        self,
        bucket_info: CollBucket,
        candidate: fx.Node,
    ) -> bool:
        """
        Check if candidate can be added to bucket without breaking comm/compute overlap.

        Strategy: Try all timeline positions - combinations of [existing_start, candidate_start]
        x [existing_wait, candidate_wait]. For each position, verify:
        1. Hiding intervals preserved - for any (start, hiding_compute, wait) interval, no other
           collective's (start, wait) pair falls between start and hiding_compute, which would
           force realization and break overlap due to LIFO semantics
        2. Topologically valid (no dependency cycles)

        Return True if any timeline position satisfies both constraints.
        """
        existing_coll = bucket_info.collectives[0]
        why = WhyNoBucket(existing_coll, candidate)

        candidate_info = self.collective_info[candidate]

        if (
            candidate in self.all_hiding_nodes
            or candidate_info.wait_node in self.all_hiding_nodes
        ):
            why("nyi: bucketing collective used for overlap")
            return False

        # Step 1: Quick check using precomputed ancestors
        # These ancestors are computed prior to adding augmented dependencies and not updated,
        # so if any of these checks fail then the merge will not be topologically valid
        # even ignoring comm/compute overlap
        if self._has_ancestor_conflicts(bucket_info, candidate):
            why("has ancestor conflicts")
            return False

        # Step 2: Try different rail positions
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

        for i, (start_pos, wait_pos) in enumerate(combinations):
            if self._try_timeline_position(
                bucket_info, candidate, start_pos, wait_pos, why
            ):
                bucket_log.debug(
                    "bucketed %s with %s using timeline position %d: (start=%s, wait=%s)",
                    candidate.name,
                    existing_coll.name,
                    i + 1,
                    start_pos.name,
                    wait_pos.name,
                )
                return True

        why("all timeline positions failed")
        return False

    def _apply_bucket(self, bucket_info: CollBucket) -> None:
        """
        Apply bucketing transformation.

        Dependencies are added to aug_graph.extra_deps and transferred from old nodes.
        """

        from torch._inductor.fx_passes.bucketing import (
            is_all_reduce_tensor,
            merge_all_gather_bucket,
            merge_all_reduce_bucket,
            merge_reduce_scatter_bucket,
        )

        bucket = bucket_info.collectives

        # Collect old nodes BEFORE they're erased
        old_starts = list(bucket)
        old_waits = [self.collective_info[n].wait_node for n in bucket]

        fused_convert_dtypes = []
        for n in old_starts:
            if has_mergeable_all_gather_convert_dtype(n):
                fused_convert_dtypes.append(n.args[0])

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
        elif is_all_reduce_tensor(bucket[0]):
            new_nodes, replacements = merge_all_reduce_bucket(
                self.graph,
                bucket,
                mode="custom_ops",
                insert_before=next_node,
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
        new_waits = [n for n in new_nodes if _schedulable_wait_node(n)]
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

        # Handle convert_element_type nodes that were fused and erased
        # The bucketed operation may have a _pre_bucket op that handles dtype conversion
        if fused_convert_dtypes:
            # all gather bucketing may fuse in dtype conversion into the bucketing
            # if so, we need to transfer hiding deps from the old dtype conversion
            # to the new bucketing node
            new_convert_dtypes_node = new_start.kwargs["out"]
            assert isinstance(new_convert_dtypes_node, fx.Node)
            assert (
                new_convert_dtypes_node.target
                == torch.ops.bucketing._pre_bucket_all_gather.default
            )

            for n in fused_convert_dtypes:
                erased_to_new[n] = new_convert_dtypes_node

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
