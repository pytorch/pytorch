from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.fx as fx
from torch._inductor.augmented_graph_helper import AugmentedGraphHelper
from torch._inductor.fx_passes.bucketing import (
    _ag_group_key,
    _rs_group_key,
    is_all_gather_into_tensor as is_all_gather,
    is_reduce_scatter_tensor as is_reduce_scatter,
    is_wait_tensor,
)
from torch._inductor.fx_passes.overlap_scheduling import CollectiveInfo
from torch.utils._ordered_set import OrderedSet


@dataclass(slots=True)
class CollBucket:
    """Track information about a bucket of collectives."""

    collectives: list[fx.Node] = field(
        default_factory=list
    )  # Original collective starts
    total_bytes: int = 0
    min_start_idx: Optional[int] = None  # Minimum index of collective starts
    max_wait_idx: Optional[int] = None  # Maximum index of collective waits

    bucketed_start: Optional[fx.Node] = None  # After bucketing
    bucketed_wait: Optional[fx.Node] = None  # After bucketing

    def add_collective(
        self,
        coll_info: CollectiveInfo,
        node_idx: dict[fx.Node, int],
    ) -> None:
        """
        Add a collective to this bucket and update bucket metadata.

        This handles all updates needed when adding a collective:
        - Appends to collectives list
        - Updates total bytes
        - Updates min_start_idx and max_wait_idx
        """
        collective = coll_info.start_node

        # Add to bucket
        self.collectives.append(collective)
        self.total_bytes += coll_info.size_bytes

        # Update min start index
        start_idx = node_idx[collective]
        if self.min_start_idx is None:
            self.min_start_idx = start_idx
        else:
            self.min_start_idx = min(self.min_start_idx, start_idx)

        # Update max wait index
        wait_idx = node_idx[coll_info.wait_node]
        if self.max_wait_idx is None:
            self.max_wait_idx = wait_idx
        else:
            self.max_wait_idx = max(self.max_wait_idx, wait_idx)


def bucket_key(node: torch.fx.Node) -> Optional[object]:
    if is_all_gather(node):
        return _ag_group_key(node)
    elif is_reduce_scatter(node):
        return _rs_group_key(node)
    else:
        return None


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
    ):
        self.graph = graph
        self.collective_info = collective_info
        self.node_ancestors = node_ancestors
        self.scheduled = scheduled
        self.max_bucket_memory_gb = max_bucket_memory_gb
        self.node_idx = {n: i for i, n in enumerate(scheduled)}
        self.aug_graph = AugmentedGraphHelper(self.graph, node_to_idx=self.node_idx)

    def bucket_collectives(self) -> None:
        """Main entry point for bucketing collectives."""

        # Add extra dependencies for hidden collectives
        # For each hidden collective, add: compute -> start and wait -> compute
        for start_node, info in self.collective_info.items():
            if info.hiding_node and not info.is_exposed:
                # Add edge: hiding_compute depends on start (start must come before compute)
                self.aug_graph.add_extra_dep(n=info.hiding_node, dep=start_node)
                # Add edge: wait depends on hiding_compute (compute must come before wait)
                self.aug_graph.add_extra_dep(n=info.wait_node, dep=info.hiding_node)

        # Group collectives by bucket key (type, group, etc.)
        grouped_collectives: dict[object, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for start in self.collective_info:
            key = bucket_key(start)
            if key is not None:
                grouped_collectives[key].add(start)

        all_buckets: list[CollBucket] = []
        for collective_group in grouped_collectives.values():
            buckets = self._find_buckets(collective_group)
            all_buckets.extend(buckets)

        # Collect all extra dependencies to preserve after bucketing
        additional_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)

        # Apply bucketing transformations
        for coll_bucket in all_buckets:
            if len(coll_bucket.collectives) <= 1:
                continue

            bucket_deps = self._apply_bucket(coll_bucket)
            additional_deps.update(bucket_deps)

        # Apply topological sort with all the collected dependencies
        from torch._dynamo.graph_deduplication import _stable_topological_sort

        _stable_topological_sort(self.graph, additional_deps)

        # After topological sort, preserve dependencies using effect tokens
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
            bucket_info = CollBucket()
            bucket_info.add_collective(self.collective_info[start_node], self.node_idx)
            processed.add(start_node)

            for candidate in collective_group:
                if candidate in processed:
                    continue

                candidate_bytes = self.collective_info[candidate].size_bytes
                if bucket_info.total_bytes + candidate_bytes > max_bucket_bytes:
                    continue

                if self._can_add_to_bucket(bucket_info, candidate):
                    bucket_info.add_collective(
                        self.collective_info[candidate], self.node_idx
                    )
                    processed.add(candidate)

            if len(bucket_info.collectives) > 1:
                buckets.append(bucket_info)

        return buckets

    def _ancestor_dep(self, n1: fx.Node, n2: fx.Node) -> bool:
        """Check if there's an ancestor relationship between two nodes."""
        return n1 in self.node_ancestors[n2] or n2 in self.node_ancestors[n1]

    def _has_path(
        self,
        source: fx.Node,
        source_bounds: tuple[int, int],
        target: fx.Node,
        target_bounds: tuple[int, int],
    ) -> bool:
        """Check if there's a path from source to target with bounded search."""

        search_range = (
            min(source_bounds[0], target_bounds[0]),
            max(source_bounds[1], target_bounds[1]),
        )

        return self.aug_graph.has_path(
            source,
            target,
            bounded_search_range=search_range,
        )

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
        candidate_wait = candidate_info.wait_node

        # Step 1: Quick check using precomputed ancestors
        # This will not be fully up to date because bucketing changes ancestors,
        # however any ancestor at the start of bucketing will remain an ancestor.
        for coll in bucket_info.collectives:
            if self._ancestor_dep(coll, candidate):
                return False

            coll_wait = self.collective_info[coll].wait_node
            if self._ancestor_dep(candidate_wait, coll_wait):
                return False

            if hiding_node := self.collective_info[coll].hiding_node:
                if self._ancestor_dep(hiding_node, candidate_wait):
                    return False

            if new_hiding_node := candidate_info.hiding_node:
                if self._ancestor_dep(new_hiding_node, coll_wait):
                    return False

        # Step 2: Check and merge starts
        # Check if there's a path between any existing start and candidate start.
        # Because the collectives have already been merged, we can just start from one
        # of them.
        existing_coll = bucket_info.collectives[0]

        # Calculate bounds for path search
        candidate_idx = self.node_idx[candidate]
        candidate_wait_idx = self.node_idx[candidate_wait]

        bucket_min_idx = bucket_info.min_start_idx
        bucket_max_idx = bucket_info.max_wait_idx
        assert bucket_min_idx is not None and bucket_max_idx is not None
        existing_bounds = (bucket_min_idx, bucket_max_idx)
        candidate_bounds = (candidate_idx, candidate_wait_idx)

        if self._has_path(existing_coll, existing_bounds, candidate, candidate_bounds):
            return False
        if self._has_path(candidate, candidate_bounds, existing_coll, existing_bounds):
            return False

        # Safe to merge starts - do the merge
        self.aug_graph.merge_to_set(existing_coll, candidate)

        # Step 3: Check and merge waits
        existing_wait = self.collective_info[existing_coll].wait_node

        if self._has_path(
            existing_wait, existing_bounds, candidate_wait, candidate_bounds
        ) or self._has_path(
            candidate_wait, candidate_bounds, existing_wait, existing_bounds
        ):
            # Unmerge the start we just merged
            self.aug_graph.unmerge_node(candidate)
            return False

        self.aug_graph.merge_to_set(existing_wait, candidate_wait)
        return True

    def _apply_bucket(
        self, bucket_info: CollBucket
    ) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """Apply bucketing transformation and return dependencies to preserve."""

        from torch._inductor.fx_passes.bucketing import (
            merge_all_gather_bucket,
            merge_reduce_scatter_bucket,
        )

        bucket = bucket_info.collectives

        # Find where to place the bucketed operations
        next_node = bucket[0]
        while next_node in bucket:
            next_node = next_node.next
        waits = [self.collective_info[n].wait_node for n in bucket]
        first_wait = min(waits, key=lambda w: self.node_idx[w])

        # Create bucketed collective
        if is_all_gather(bucket[0]):
            new_nodes, replacements = merge_all_gather_bucket(
                self.graph,
                bucket,
                wait_insertion_point=first_wait,
                insert_before=next_node,
                mode="custom_ops",
            )
        else:
            assert is_reduce_scatter(bucket[0])
            new_nodes, replacements = merge_reduce_scatter_bucket(
                self.graph,
                bucket,
                wait_insertion_point=first_wait,
                insert_before=next_node,
                mode="custom_ops",
            )

        # Build dependencies to preserve overlap
        # replacements maps old_start -> new_start, old_wait -> new_wait
        new_waits = [n for n in new_nodes if is_wait_tensor(n)]
        assert len(new_waits) == 1

        new_wait = new_waits[0]
        new_start = new_wait.args[0]
        assert isinstance(new_start, fx.Node)

        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)

        # Create dependencies to preserve overlap
        for coll in bucket:
            info = self.collective_info[coll]
            if info.hiding_node and not info.is_exposed:
                # Compute depends on collective start
                overlap_deps[info.hiding_node].add(new_start)
                # Wait depends on compute
                overlap_deps[new_wait].add(info.hiding_node)

        return overlap_deps

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

        if torch._inductor.config.test_configs.aten_fx_overlap_insert_overlap_deps:
            preserve_node_ordering(self.graph, additional_deps)
