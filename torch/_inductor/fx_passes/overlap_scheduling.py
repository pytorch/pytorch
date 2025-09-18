import collections
import dataclasses
import functools
import heapq
import itertools
import math
import operator
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Counter as CounterType,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Union,
)

import torch
import torch.fx as fx
from torch import fx
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor import utils
from torch.utils._mode_utils import no_dispatch
from torch.utils._ordered_set import OrderedSet

from ..pattern_matcher import stable_topological_sort


def bucket_key(node: torch.fx.Node) -> Optional[object]:
    from torch._inductor.fx_passes.bucketing import _ag_group_key, _rs_group_key

    if node.target == torch.ops._c10d_functional.all_gather_into_tensor.default:
        return _ag_group_key(node)
    elif node.target == torch.ops._c10d_functional.reduce_scatter_tensor.default:
        return _rs_group_key(node)
    else:
        return None


def is_all_gather_into_tensor(node: torch.fx.Node) -> bool:  # type: ignore[arg-type]
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
    )


def is_reduce_scatter_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.reduce_scatter_tensor.default
    )


def is_wait_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


def is_wait_tensor_from_all_gather_into_tensor(node: torch.fx.Node) -> bool:
    return is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0])  # type: ignore[arg-type]


def estimate_collective_time(n):
    return torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
        n
    )


def estimate_fx_collective_size(fx_node: torch.fx.Node) -> int:
    size = 0
    for node in fx_node.all_input_nodes:
        if (t := node.meta.get("val")) is not None:
            # todo - symbolic
            size += t.numel() * t.element_size()

    return size


def is_compute_node(n):
    return (
        getattr(n.target, "overloadpacket", None)
        in torch.utils.flop_counter.flop_registry
    )


def get_hint(x: Union[int, torch.SymInt]) -> Optional[int]:
    if isinstance(x, int):
        return x
    assert utils.is_symbolic(x)
    if not x.node.has_hint():
        return None
    return x.node.hint


def get_collective_do_bench() -> Callable[[Callable[[], Any]], float]:
    with dynamo_timed("collective_compute_do_bench"):
        return functools.partial(
            torch._inductor.runtime.benchmarking.benchmarker.benchmark_gpu,
            warmup=5,
        )


def benchmark_node(n):
    from torch._dynamo.testing import rand_strided

    # todo - skip unbacked, symbolic
    success, args, kwargs = torch._inductor.fx_utils.get_fake_args_kwargs(n)

    if not success:
        return 0

    unbacked_tensor = False

    key = f"{str(n.target)}: "

    def to_real(t: torch.Tensor) -> Optional[torch.Tensor]:
        shape = [get_hint(dim) for dim in t.shape]
        stride = [get_hint(s) for s in t.stride()]

        if any(s is None for s in itertools.chain(shape, stride)):
            nonlocal unbacked_tensor
            unbacked_tensor = True
            return None

        nonlocal key
        key += f"T: {shape, stride, t.dtype} "
        return rand_strided(shape, stride, device=t.device, dtype=t.dtype)

    with no_dispatch():
        args, kwargs = torch.utils._pytree.tree_map_only(
            torch.Tensor,
            lambda t: to_real(t),
            (args, kwargs),
        )

        if val := get_cached_node_time(key):
            return val

        if unbacked_tensor:
            return 0

        bench = get_collective_do_bench()
        out = bench(lambda: n.target(*args, **kwargs))
        set_cached_node_time(key, out)
        return out


@functools.cache
def get_pad_cache() -> torch._inductor.codecache.LocalCache:
    return torch._inductor.codecache.LocalCache()


def get_cached_node_time(key: str) -> float:
    return get_pad_cache().lookup(key)  # type: ignore[return-value]


def set_cached_node_time(key: str, value: float) -> None:
    return get_pad_cache().set_value(key, value=value)


@dataclass
class CollectiveInfo:
    """Track info about a collective operation"""

    start_node: fx.Node
    wait_node: fx.Node
    size_bytes: int
    estimated_time_ms: float
    exposed_time_ms: float  # How much of this collective is still exposed
    hiding_node: Optional[fx.Node] = None  # Node that hides this collective
    in_flight_with: OrderedSet[fx.Node] = dataclasses.field(
        default_factory=OrderedSet
    )  # Nodes that are in flight with this collective

    @property
    def is_exposed(self):
        return self.exposed_time_ms != 0


class OverlapScheduler:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        compute_overlap_multipler: float = 1.,
        max_node_distance: int = 1000,
        max_in_flight_gb: float = 2,
    ):
        self.gm = gm
        self.graph = gm.graph
        self.compute_overlap_multipler = compute_overlap_multipler
        self.max_node_distance = max_node_distance
        self.max_in_flight_bytes: int = int(max_in_flight_gb * 1e9)

        # Build structures
        stable_topological_sort(self.graph)
        self.nodes = list(self.graph.nodes)
        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        self.node_ancestors = self._collect_node_ancestors()

        # Identify collectives and compute nodes
        self.collective_info: Dict[fx.Node, CollectiveInfo] = {}
        self.unscheduled_collectives: OrderedSet[fx.Node] = OrderedSet()

        self.wait_to_start: Dict[fx.Node, fx.Node] = {}
        self._identify_collectives()

        self.compute_depth = self._calculate_compute_node_depth()
        self.compute_nodes = [n for n in self.nodes if is_compute_node(n)]

        # Scheduling state
        self.potentially_hidden_collectives = (
            self.compute_potential_hidden_collectives()
        )
        self.potentially_hidden_waits = self.compute_potential_hidden_waits()
        self.in_degree = Counter(user for node in self.nodes for user in node.users)
        self.ready = []

        for node in self.nodes:
            if self.in_degree[node] == 0:
                heapq.heappush(self.ready, (self._compute_score(node), node))

        self.in_flight: Dict[fx.Node, CollectiveInfo] = {}  # start -> info
        self.in_flight_bytes = 0
        self.scheduled = OrderedSet()

    def _collect_node_ancestors(self) -> Dict[fx.Node, OrderedSet[fx.Node]]:
        """Collect all ancestors for each node."""
        ancestors = defaultdict(OrderedSet)
        for node in self.nodes:
            for input_node in node.all_input_nodes:
                ancestors[node].add(input_node)
                ancestors[node] |= ancestors[input_node]

        return ancestors

    def _identify_collectives(self):
        """Identify all collective operations."""
        for node in self.nodes:
            if is_wait_tensor(node):
                start = node.args[0]
                coll_time_us = estimate_collective_time(start)
                coll_time_ms = coll_time_us / 1000000

                info = CollectiveInfo(
                    start_node=start,
                    wait_node=node,
                    size_bytes=estimate_fx_collective_size(start),
                    estimated_time_ms=coll_time_ms,
                    exposed_time_ms=coll_time_ms,  # Initially fully exposed
                )
                self.collective_info[start] = info
                self.wait_to_start[node] = start
                self.unscheduled_collectives.add(start)

    def _calculate_compute_node_depth(self):
        """Compute forward depth and minimum dominance depth (infinity if blocks no compute)."""

        # First pass: forward compute depth
        in_degree: dict[fx.Node, int] = {}
        compute_depth: dict[fx.Node, int] = {}
        queue: list[fx.Node] = []

        for node in self.graph.nodes:
            num_inputs = len(node.all_input_nodes)
            if num_inputs == 0:
                queue.append(node)
            else:
                in_degree[node] = num_inputs

        while queue:
            node = queue.pop()

            max_input_depth = max(
                (compute_depth[inp] for inp in node.all_input_nodes), default=0
            )
            compute_depth[node] = max_input_depth + is_compute_node(node)

            for use in node.users:
                in_degree[use] -= 1
                if in_degree[use] == 0:
                    queue.append(use)

        # Second pass: minimum dominance (what's the earliest compute this blocks)
        compute_depth_dominance = {}

        for node in reversed(self.graph.nodes):
            if is_compute_node(node):
                # consider compute nodes to be at their own depth
                dominance = compute_depth[node]
            else:
                # For non-compute nodes, find minimum compute they block
                dominance = min(
                    (compute_depth_dominance[succ] for succ in node.users),
                    default=sys.maxsize,
                )

            compute_depth_dominance[node] = dominance

        return compute_depth_dominance

    def run(self) -> torch.fx.GraphModule:
        """Run the scheduling algorithm."""

        while self.ready:
            if self._should_force_wait_for_memory():
                self._force_oldest_wait()
                continue

            _, node = heapq.heappop(self.ready)

            # we don't always remove nodes from the heap when we schedule them
            if node in self.scheduled:
                continue

            if is_compute_node(node):
                self._handle_compute(node)
            elif node in self.collective_info:
                self._handle_collective_start(node)
            elif is_wait_tensor(node):
                self._handle_wait(node)
            else:
                self._handle_other(node)

        self._reorder_graph()
        return self.gm

    def _handle_other(self, node: fx.Node) -> None:
        self._schedule(node)

    def _schedule(self, node: fx.Node) -> None:
        """Schedule a node."""
        assert node not in self.scheduled
        assert all(n in self.scheduled for n in node.all_input_nodes)
        self.scheduled.add(node)

        for user in node.users:
            self.in_degree[user] -= 1
            if self.in_degree[user] == 0:
                heapq.heappush(self.ready, (self._compute_score(user), user))

    def _compute_score(self, node: fx.Node) -> tuple:
        """Compute priority score for a node"""

        if is_wait_tensor(node):
            info = self.collective_info[self.wait_to_start[node]]
            overlappable = info.is_exposed and node in self.potentially_hidden_waits
        else:
            overlappable = 1 if self.in_overlappable_collective_unary_chain(node) else 0

        return (
            self.compute_depth[node],  # what depth compute it blocks
            overlappable,  # Defer hideable collective ops
            self.node_idx[node],  # Original order for stability
        )

    @staticmethod
    def is_cheap_fn(node: fx.Node) -> bool:
        return getattr(node.target, "is_view", False) or torch.Tag.pointwise in getattr(
            node.target, "tags", ()
        )

    def in_overlappable_collective_unary_chain(self, curr: fx.Node) -> bool:
        while True:
            if len(curr.users) != 1:
                return False

            user = next(iter(curr.users))
            if len(user.all_input_nodes) != 1:
                return False

            if user in self.unscheduled_collectives:
                return user in self.potentially_hidden_collectives

            if not self.is_cheap_fn(user):
                return False

            curr = user

        return False

    def _should_force_wait_for_memory(self) -> bool:
        """Check if we need to force a wait due to memory pressure"""
        return self.in_flight_bytes >= self.max_in_flight_bytes

    def _force_oldest_wait(self) -> None:
        """Schedule the oldest in flight wait"""
        self._handle_wait(self._get_oldest_wait())

    def _handle_collective_start(self, node: fx.Node):
        """Handle scheduling a collective start."""
        info = self.collective_info[node]
        self.in_flight[node] = info
        self.in_flight_bytes += info.size_bytes
        self.unscheduled_collectives.discard(node)
        self._schedule(node)

    def _handle_wait(self, node: fx.Node):
        """Handle scheduling a wait."""
        assert node in self.wait_to_start
        coll_start = self.wait_to_start[node]

        assert coll_start in self.in_flight
        self.in_flight_bytes -= self.in_flight[coll_start].size_bytes
        del self.in_flight[coll_start]
        self._schedule(node)

    def _handle_compute(self, node: fx.Node):
        """Handle scheduling compute and finding overlaps."""

        compute_time = benchmark_node(node)
        available_compute = compute_time * self.compute_overlap_multipler

        # First reduce exposed time of in-flight collectives
        for info in self.in_flight.values():
            if info.exposed_time_ms == 0:
                continue
            overlap_amount = min(info.exposed_time_ms, available_compute)
            info.exposed_time_ms -= overlap_amount
            available_compute -= overlap_amount
            if info.exposed_time_ms == 0:
                info.hiding_node = node
            elif available_compute == 0:
                break

        # Then, look for unscheduled collectives we can overlap
        if available_compute:
            self._schedule_collectives_for_overlap(node, available_compute)

        self._schedule(node)

    def _schedule_collectives_for_overlap(
        self, compute_node: fx.Node, available_compute_time: float
    ) -> None:
        """Opportunistically schedule collectives that can be hidden by compute."""
        compute_ancestors = self.node_ancestors[compute_node]

        # copy unscheduled_collectives to local because we modify it during iteration
        possible_collectives = []
        for collective in self.unscheduled_collectives:
            distance = abs(self.node_idx[compute_node] - self.node_idx[collective])
            if distance > self.max_node_distance:
                break

            possible_collectives.append(collective)

        for collective in possible_collectives:
            if available_compute_time == 0:
                break

            info = self.collective_info[collective]

            # Skip if compute depends on collective or vice versa
            if (
                collective in compute_ancestors
                or compute_node in self.node_ancestors[collective]
            ):
                continue

            while (
                self.in_flight
                and (self.max_in_flight_bytes - self.in_flight_bytes) < info.size_bytes
                and self._wait_is_hidden(self._get_oldest_wait(), compute_node)
            ):
                self._force_oldest_wait()

            if (self.max_in_flight_bytes - self.in_flight_bytes) < info.size_bytes:
                continue

            # Check if we can reach this collective without scheduling compute, other collectives, or waits
            path = self._find_schedulable_path(collective, compute_node)
            if path is None:
                continue

            # Schedule path to this collective
            self._schedule_path_to_collective(path, compute_node)
            # Update the exposed time for this newly scheduled collective
            overlap_amount = min(info.estimated_time_ms, available_compute_time)
            info.exposed_time_ms -= overlap_amount
            if info.exposed_time_ms == 0:
                info.hiding_node = compute_node
            available_compute_time -= overlap_amount
            self._handle_collective_start(collective)

    def _find_schedulable_path(
        self, target: fx.Node, compute_node
    ) -> Optional[OrderedSet[fx.Node]]:
        """Find path to target by collecting unscheduled dependencies."""

        # TODO - following path faster than doing set difference here
        unscheduled_ancestors = self.node_ancestors[target] - self.scheduled

        # only schedule non distributed, non compute nodes
        for node in unscheduled_ancestors:
            if is_compute_node(node):
                return None

            if node in self.unscheduled_collectives:
                return None

            # if we schedule a wait tensor whose start collective is hidden by the
            # current compute node we are scheduling, then we are effectively exposing it.
            # similarly, dont schedule a wait of a collective that could be otherwise hidden,
            # thus forcing it to be exposed.
            # however, if it is already hidden or it cannot be possible hidden,
            # it's fine to schedule it
            if is_wait_tensor(node):
                info = self.collective_info[self.wait_to_start[node]]
                if info.hiding_node and info.hiding_node != compute_node:
                    continue
                elif node not in self.potentially_hidden_waits:
                    continue

                return None

        return unscheduled_ancestors

    def _get_oldest_wait(self) -> fx.Node:
        oldest_start = next(iter(self.in_flight))
        return self.collective_info[oldest_start].wait_node

    def _wait_is_hidden(self, wait_node, compute_node: Optional[fx.Node] = None):
        assert is_wait_tensor(wait_node)
        info = self.collective_info[self.wait_to_start[wait_node]]
        return not info.is_exposed and info.hiding_node != compute_node

    def _schedule_path_to_collective(self, path: OrderedSet[fx.Node], compute_node):
        """Schedule all nodes needed to reach a collective."""
        for node in sorted(path, key=lambda n: self.node_idx[n]):
            assert not (is_compute_node(node) or node in self.unscheduled_collectives)

            if is_wait_tensor(node):
                info = self.collective_info[self.wait_to_start[node]]
                assert not info.hiding_node == compute_node
                self._handle_wait(node)
                continue

            self._schedule(node)

    def reorder_graph(self):
        output_node = self.graph.output_node()
        for node in self.scheduled:
            output_node.prepend(node)

    def _reorder_graph(self):
        """Reorder graph based on schedule."""
        exposed = [c for c in self.collective_info.values() if c.exposed_time_ms != 0]
        nonexposed = [
            c for c in self.collective_info.values() if c.exposed_time_ms == 0
        ]

        potentially_hidden_collectives = self.compute_potential_hidden_collectives(
            limit_coll_per_compute=True
        )
        bad_exposed = [
            c for c in exposed if c.start_node in potentially_hidden_collectives
        ]
        print(
            "Total exposed",
            len(exposed),
            "Total bad exposed",
            len(bad_exposed),
            "total potential",
            len(potentially_hidden_collectives),
        )

        # this is not actually required
        # we could just bucket
        self.reorder_graph()
        # TODO :add 
        # self._bucket_collectives()

    def compute_potential_hidden_nodes(
        self, nodes_to_check: Iterable[fx.Node], limit_coll_per_compute=False
    ) -> dict[fx.Node, fx.Node]:
        """
        Returns a dict containing a mapping of nodes which could potentially be hidden to their hiding node
        """

        used_compute_nodes = OrderedSet()

        def could_be_hidden(start) -> Optional[fx.Node]:
            for compute_node in self.compute_nodes:
                if limit_coll_per_compute and compute_node in used_compute_nodes:
                    continue
                if (
                    start not in self.node_ancestors[compute_node]
                    and compute_node not in self.node_ancestors[start]
                ):
                    if limit_coll_per_compute:
                        used_compute_nodes.add(compute_node)
                    return compute_node

            return None

        # TODO: We could potentially limit compute nodes per overlap time,
        # today, this is optimistic, and just serves to avoid deferring
        # collectives/waits that have no possible overlap as well as for analysis of how
        # successfully we hid compute
        potentially_hidden = {}
        for node in nodes_to_check:
            if mm := could_be_hidden(node):
                potentially_hidden[node] = mm

        return potentially_hidden

    def compute_potential_hidden_collectives(self, limit_coll_per_compute=False):
        """Compute which collective operations could be hidden by matrix multiplications."""
        return self.compute_potential_hidden_nodes(
            self.collective_info.keys(), limit_coll_per_compute
        )

    def compute_potential_hidden_waits(self, limit_coll_per_compute=False):
        """Compute which wait operations could be hidden by matrix multiplications."""
        wait_nodes = [info.wait_node for info in self.collective_info.values()]
        return self.compute_potential_hidden_nodes(wait_nodes, limit_coll_per_compute)

    def _bucket_collectives(self):
        from torch._inductor.fx_passes.bucketing import (
            merge_all_gather_bucket,
            merge_reduce_scatter_bucket,
        )

        """Bucket collectives based on scheduling results."""

        node_idx = {n: i for i, n in enumerate(self.scheduled)}

        # Group collectives by type and exposure
        grouped_exposed_collectives: dict[Any, OrderedSet] = defaultdict(OrderedSet)
        grouped_hidden_collectives: dict[Any, OrderedSet] = defaultdict(OrderedSet)

        for start, info in self.scheduled_collectives.items():
            key = bucket_key(start)
            if key is None:
                continue

            if info.is_exposed:
                grouped_exposed_collectives[key].add(start)
            else:
                grouped_hidden_collectives[key].add(start)

        all_buckets = []
        for hidden_bucket_group in grouped_hidden_collectives.values():
            all_buckets.extend(self._bucket_group(hidden_bucket_group, node_idx, False))

        for exposed_bucket_group in grouped_exposed_collectives.values():
            all_buckets.extend(self._bucket_group(exposed_bucket_group, node_idx, True))

        for bucket in all_buckets:
            if len(bucket) <= 1:
                continue

            # put wait after first wait of bucekt
            waits = [next(iter(n.users)) for n in bucket]
            first_wait = min(waits, key=lambda w: node_idx[w])

            if is_all_gather_into_tensor(bucket[0]):
                merge_all_gather_bucket(
                    self.graph, bucket, wait_insertion_point=first_wait
                )
            elif is_reduce_scatter_tensor(bucket[0]):
                info = self.scheduled_collectives[bucket[0]]
                if not info.is_exposed:
                    f = open("my_graph_old.txt", "a")
                    f.write(str(self.graph))
                    f.close()
                    merge_reduce_scatter_bucket(
                        self.graph, bucket, wait_insertion_point=first_wait
                    )
                    f = open("my_graph_new.txt", "a")
                    f.write(str(self.graph))
                    f.close()
                    breakpoint()
                    pass

        stable_topological_sort(self.graph)
        self.graph.lint()
        breakpoint()

        pass

    def _bucket_group(
        self,
        grouped_collectives: OrderedSet[fx.Node],
        node_idx: dict[fx.Node, int],
        exposed_group: bool,
    ):
        """Bucket hidden collectives based on in-flight relationships."""
        buckets: list[list[fx.Node]] = []
        processed = OrderedSet()

        for start in grouped_collectives:
            if start in processed:
                continue

            bucket = [start]
            bucket_descendants = self.node_descendants[start].copy()
            processed.add(start)
            info = self.scheduled_collectives[start]
            first_wait = info.wait_node
            first_hiding_node = info.hiding_node

            # Only look at collectives that were in-flight with this one
            for candidate_start in info.in_flight_with:
                if (
                    candidate_start in processed
                    or candidate_start not in grouped_collectives
                ):
                    continue

                # if they were in flight together, there should be no way for them
                # to depend on each other. nonetheless we can be extra safe initially.
                if candidate_start in bucket_descendants:
                    continue

                # collective overlap node should be subsequent to the earliest wait in the bucket
                # or we will force exposure
                if not exposed_group:
                    candidate_info = self.scheduled_collectives[candidate_start]
                    hiding_node = candidate_info.hiding_node
                    assert first_hiding_node is not None
                    assert hiding_node is not None
                    if node_idx[first_wait] > node_idx[hiding_node]:
                        continue

                    if node_idx[candidate_info.wait_node] > node_idx[first_hiding_node]:
                        continue

                    # since we're iterating in order the first hiding node is the earliest
                    first_wait = min(
                        first_wait, candidate_info.wait_node, key=lambda n: node_idx[n]
                    )

                bucket.append(candidate_start)
                bucket_descendants |= self.node_descendants[candidate_start]
                processed.add(candidate_start)

            if len(bucket) > 1:
                buckets.append(bucket)

        return buckets


def schedule_overlap_bucketing(
    gm: torch.fx.GraphModule, **kwargs
) -> torch.fx.GraphModule:
    """Schedule nodes to maximize compute-collective overlap, then bucket."""
    return OverlapScheduler(gm, **kwargs).run()
