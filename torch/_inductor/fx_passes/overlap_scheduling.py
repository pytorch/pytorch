import functools
import heapq
import itertools
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.fx as fx
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.fx_passes.bucketing import is_wait_tensor
from torch._inductor.fx_passes.memory_estimator import (
    _is_releasable,
    build_memory_profile,
    MemoryTracker,
)
from torch.fx.operator_schemas import normalize_function
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import _disable_current_modes


log = logging.getLogger(__name__)

from torch._inductor.fx_passes.bucketing import bucket_key

from ..pattern_matcher import stable_topological_sort


def get_group_name(n: fx.Node) -> str:
    """Extract the group name from a collective operation node."""
    opt_args_kwargs = normalize_function(
        n.target,  # type: ignore[arg-type]
        args=n.args,
        kwargs=n.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    assert opt_args_kwargs is not None
    _, kwargs = opt_args_kwargs
    return kwargs["group_name"]


def get_custom_estimation(
    n: fx.Node,
    custom_runtime_estimation: Callable[[fx.Node], float | None] | None = None,
) -> float | None:
    if custom_runtime_estimation is None:
        return None

    return custom_runtime_estimation(n)


def estimate_collective_time(
    n: fx.Node,
    override_size: int | None = None,
    custom_runtime_estimation: Callable[[fx.Node], float | None] | None = None,
) -> float:
    """Estimate the runtime of a collective operation, optionally with an overridden size."""
    if (est := get_custom_estimation(n, custom_runtime_estimation)) is not None:
        return est

    # Use analytical model (benchmarking is handled separately in alignment)
    return torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
        n, override_size
    )


def estimate_fx_collective_size(fx_node: torch.fx.Node) -> int:
    size = 0
    for node in fx_node.all_input_nodes:
        if (t := node.meta.get("val")) is not None:
            # todo - symbolic
            size += t.numel() * t.element_size()

    return size


def is_compute_node(n: fx.Node) -> bool:
    """
    Should we consider this node computationally expensive ?
    Currently uses flop registration, but we could expand more generally.
    """
    return (
        getattr(n.target, "overloadpacket", None)
        in torch.utils.flop_counter.flop_registry
    )


def get_hint(x: int | torch.SymInt) -> int | None:
    if isinstance(x, int):
        return x
    assert isinstance(x, torch.SymInt)
    if not x.node.has_hint():
        return None
    return x.node.hint


def get_collective_do_bench() -> Callable[[Callable[[], Any]], float]:
    with dynamo_timed("collective_compute_do_bench"):
        return functools.partial(
            # pyrefly: ignore [bad-argument-type]
            torch._inductor.runtime.benchmarking.benchmarker.benchmark_gpu,
            warmup=5,
        )


def benchmark_node_with_cache_key(
    n: fx.Node,
    custom_runtime_estimation: Callable[[fx.Node], float | None] | None = None,
) -> tuple[float, str | None]:
    """Benchmark a compute node and return (runtime, cache_key)."""
    assert is_compute_node(n)

    from torch._dynamo.testing import rand_strided

    # todo - skip unbacked, symbolic
    success, args, kwargs = torch._inductor.fx_utils.get_fake_args_kwargs(n)

    if not success:
        return 0, None

    unbacked_tensor = False

    key = f"{str(n.target)}: "

    def to_real(t: torch.Tensor) -> torch.Tensor | None:
        shape = [get_hint(dim) for dim in t.shape]
        stride = [get_hint(s) for s in t.stride()]

        if any(s is None for s in itertools.chain(shape, stride)):
            nonlocal unbacked_tensor
            unbacked_tensor = True
            return None

        nonlocal key
        key += f"T: {shape, stride, t.dtype} "
        return rand_strided(shape, stride, device=t.device, dtype=t.dtype)  # type: ignore[arg-type]

    with _disable_current_modes():
        args, kwargs = torch.utils._pytree.tree_map_only(
            torch.Tensor,
            lambda t: to_real(t),
            (args, kwargs),
        )

        if val := get_cached_node_time(key):
            return val, key

        if unbacked_tensor:
            return 0, key

        if (est := get_custom_estimation(n, custom_runtime_estimation)) is not None:
            set_cached_node_time(key, est)
            return est, key

        bench = get_collective_do_bench()
        out = bench(lambda: n.target(*args, **kwargs))  # type: ignore[operator]
        set_cached_node_time(key, out)
        return out, key


def benchmark_node(
    n: fx.Node,
    custom_runtime_estimation: Callable[[fx.Node], float | None] | None = None,
) -> float:
    return benchmark_node_with_cache_key(n, custom_runtime_estimation)[0]


@functools.cache
def get_benchmark_cache() -> torch._inductor.codecache.LocalCache:
    return torch._inductor.codecache.LocalCache()


def get_cached_node_time(key: str) -> float:
    return get_benchmark_cache().lookup(key)  # type: ignore[return-value]


def set_cached_node_time(key: str, value: float) -> None:
    return get_benchmark_cache().set_value(key, value=value)


@dataclass
class CollectiveInfo:
    """Track info about a collective operation"""

    start_node: fx.Node
    wait_node: fx.Node
    size_bytes: int
    estimated_time_ms: float
    exposed_time_ms: float  # How much of this collective is still exposed
    hiding_node: fx.Node | None = None  # Node that hides this collective

    @property
    def is_exposed(self) -> bool:
        return self.exposed_time_ms != 0


@dataclass
class CollBucket:
    """Track information about a bucket of collectives."""

    collectives: list[fx.Node]  # Original collective starts
    bucketed_start: fx.Node | None = None  # After bucketing
    bucketed_wait: fx.Node | None = None  # After bucketing
    total_bytes: int = 0


def gb_to_bytes(gb: float) -> int:
    """Convert gigabytes to bytes."""
    return int(gb * 1024 * 1024 * 1024)


class OverlapScheduler:
    """
    Scheduler that reorders operations to maximize compute-collective overlap.

    The reordering is done as a scheduling pass. We maintain a priority queue of
    schedulable nodes. The nodes are ranked by:

    1) the compute node index they dominate. this allows reordering locally, such as with
    parallel mms, and also allows overlapping reduce scatter nodes outputs in the backward
    with compute by deferring their waits.

    2) whether the current node is a collective or wait that is currently exposed but has a compute
    node which it could be overlapped with.

    3) original order in the graph for stability.

    When we schedule compute nodes, we first overlap exposed in-flight collectives, then look for unscheduled
    collectives that can be scheduled concurrently.

    TODO:
        - experiment with other priority scores / allow other mechanisms of reorder / more strict adherence to original graph
        - memory limit for deferred scheduling of reduce_scatter nodes.
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        max_in_flight_gb: float,
        max_compute_pre_fetch: int,
        collective_bucketing: bool,
        insert_overlap_deps: bool,
        compute_overlap_multipler: float,
        max_coll_distance: int,
        custom_runtime_estimation: Callable[[fx.Node], float | None] | None,
        collective_estimator: Literal["analytical", "benchmark"],
    ):
        self.gm = gm
        self.graph = gm.graph
        self.compute_overlap_multipler = compute_overlap_multipler
        self.max_node_distance = max_coll_distance
        self.max_in_flight_bytes: int = gb_to_bytes(max_in_flight_gb)
        self.custom_runtime_estimation = custom_runtime_estimation
        self.collective_bucketing = collective_bucketing
        self.insert_overlap_deps = insert_overlap_deps
        self.max_compute_pre_fetch = max_compute_pre_fetch
        self.collective_estimator = collective_estimator

        # Build structures
        stable_topological_sort(self.graph)
        self.nodes = list(self.graph.nodes)
        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        self.node_ancestors: dict[fx.Node, OrderedSet[fx.Node]] = (
            self._collect_node_ancestors()
        )

        # Identify collectives and compute nodes
        self.collective_info: dict[fx.Node, CollectiveInfo] = {}
        self.unscheduled_collectives: OrderedSet[fx.Node] = OrderedSet()

        # Memory tracking using abstracted MemoryTracker
        self.original_peak_memory = max(
            build_memory_profile(self.graph, _is_releasable)
        )
        self.memory_tracker = MemoryTracker(self.graph)

        self.wait_to_start: dict[fx.Node, fx.Node] = {}
        self._identify_collectives()

        self.compute_index_domination = self._calculate_compute_node_domination_index()
        self.compute_nodes = [n for n in self.nodes if is_compute_node(n)]
        self.current_compute_index = 0

        # Scheduling state
        self.potentially_hidden_collectives = (
            self.compute_potential_hidden_collectives()
        )
        self.potentially_hidden_waits = self.compute_potential_hidden_waits()
        self.in_degree = Counter(user for node in self.nodes for user in node.users)
        self.ready: list[tuple[object, fx.Node]] = []

        for node in self.nodes:
            if self.in_degree[node] == 0:
                heapq.heappush(self.ready, (self._compute_score(node), node))

        self.in_flight: dict[fx.Node, CollectiveInfo] = {}  # start -> info
        self.in_flight_bytes = 0
        self.scheduled: OrderedSet[fx.Node] = OrderedSet()
        self.max_compute_pre_fetch = max_compute_pre_fetch

    def _collect_node_ancestors(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """Collect all ancestors for each node."""
        ancestors: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in self.nodes:
            for input_node in node.all_input_nodes:
                ancestors[node].add(input_node)
                ancestors[node] |= ancestors[input_node]

        return ancestors

    def off_compute_path(self, n: fx.Node) -> bool:
        """Check if a node is off the compute path (doesn't block any compute)."""
        return self.compute_index_domination[n] == sys.maxsize

    def _identify_collectives(self) -> None:
        """Identify all collective operations."""
        for node in self.nodes:
            if is_wait_tensor(node):
                start = node.args[0]
                coll_time_ms = estimate_collective_time(
                    start, custom_runtime_estimation=self.custom_runtime_estimation
                )

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

    def _calculate_compute_node_domination_index(self) -> dict[fx.Node, int]:
        """
        Compute the topological index of the earliest compute node each node dominates.

        Compute nodes are assigned indices based on their topological order (0, 1, 2, ...).
        For each node, returns the minimum index of compute nodes it blocks/dominates.
        Returns sys.maxsize if the node doesn't block any compute nodes.
        """
        compute_node_index: dict[fx.Node, int] = {}
        for node in self.graph.nodes:
            if is_compute_node(node):
                compute_node_index[node] = len(compute_node_index)

        domination_index: dict[fx.Node, int] = {}
        for node in reversed(self.graph.nodes):
            if node in compute_node_index:
                # Compute nodes dominate themselves (return their own index)
                domination_index[node] = compute_node_index[node]
            else:
                domination_index[node] = min(
                    (domination_index[succ] for succ in node.users), default=sys.maxsize
                )

        return domination_index

    def _log_collective_benchmarks(
        self,
        collective_nodes: list[fx.Node],
        collective_keys: list[str],
        benchmarked_medians: list[float],
        world_size: int,
    ) -> None:
        """Log collective benchmarks with analytical comparisons for tlparse."""
        collective_benchmarks = {}
        for key, benchmarked_ms, coll_node in zip(
            collective_keys, benchmarked_medians, collective_nodes
        ):
            # NCCL estimator (deterministic, no need to align)
            nccl_ms = torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
                coll_node, None, use_nccl_estimator=True
            )

            # Inductor analytical (deterministic, no need to align)
            inductor_ms = torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
                coll_node, None, use_nccl_estimator=False
            )

            collective_benchmarks[key] = {
                "benchmarked_ms": benchmarked_ms,
                "analytical_nccl_ms": nccl_ms,
                "analytical_inductor_ms": inductor_ms,
            }

        # Emit tlparse artifact
        from torch._logging import trace_structured

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "node_runtime_estimation",
                "encoding": "json",
            },
            payload_fn=lambda: {
                "world_size": world_size,
                "collective_benchmarks": collective_benchmarks,
            },
        )

    def _align_compute_nodes_runtime_estimations_across_all_distributed_ranks(
        self,
    ) -> None:
        """Align runtime estimations across ranks (compute + collectives)."""
        log.info(
            "Overlap scheduling: Aligning runtime estimations across all distributed ranks"
        )

        # Benchmark compute nodes
        runtime_estimations_keys: list[str | None] = []
        runtime_estimations: list[float] = []
        compute_key_count = 0

        for n in self.compute_nodes:
            val, key = benchmark_node_with_cache_key(n, self.custom_runtime_estimation)
            runtime_estimations.append(val)
            runtime_estimations_keys.append(key)
            compute_key_count += 1

        # Benchmark collectives if enabled (only CUDA events - others are deterministic)
        # Skip if custom estimation is provided for collectives
        collective_nodes: list[fx.Node] = []
        benchmarked_collective_nodes: list[
            fx.Node
        ] = []  # Track which were actually benchmarked
        if self.collective_estimator == "benchmark":
            from torch._inductor.fx_passes.node_runtime_estimation import (
                benchmark_collective_with_cuda_events,
            )

            collective_nodes = [
                info.start_node for info in self.collective_info.values()
            ]

            # Benchmark CUDA events (non-deterministic, needs alignment)
            # Skip collectives with custom estimation
            for n in collective_nodes:
                if get_custom_estimation(n, self.custom_runtime_estimation) is not None:
                    continue

                # Benchmark actual size
                cuda_val, cuda_key = benchmark_collective_with_cuda_events(n, nruns=2)
                if cuda_val is not None:
                    runtime_estimations.append(cuda_val)
                    runtime_estimations_keys.append(cuda_key)
                    benchmarked_collective_nodes.append(n)

        # Single all_gather and compute medians
        import torch.distributed as dist
        from torch._subclasses.fake_tensor import unset_fake_temporarily
        from torch.distributed.distributed_c10d import _get_default_group

        world_size = dist.get_world_size()
        pg = _get_default_group()

        with unset_fake_temporarily():
            gathered_runtime_estimations: list[list[float]] = [
                [] for _ in range(world_size)
            ]
            dist.all_gather_object(
                gathered_runtime_estimations, runtime_estimations, pg
            )
            median_runtime_estimations = torch.median(
                torch.tensor(gathered_runtime_estimations), dim=0
            ).values.tolist()

        # Cache medians
        collective_keys = []
        collective_medians = []
        for idx, (key, median_runtime_estimation) in enumerate(
            zip(runtime_estimations_keys, median_runtime_estimations)
        ):
            if key is None:
                continue
            if idx < compute_key_count:
                # Compute node
                set_cached_node_time(key, median_runtime_estimation)
            else:
                # Collective CUDA event benchmark
                from torch._inductor.fx_passes.node_runtime_estimation import (
                    set_cached_runtime,
                )

                set_cached_runtime(key, median_runtime_estimation)

                # Update CollectiveInfo with aligned benchmark
                coll_idx = idx - compute_key_count
                coll_node = benchmarked_collective_nodes[coll_idx]
                info = self.collective_info[coll_node]
                info.estimated_time_ms = median_runtime_estimation
                info.exposed_time_ms = median_runtime_estimation

                collective_keys.append(key)
                collective_medians.append(median_runtime_estimation)

        # Log benchmarks with analytical comparisons
        if collective_keys:
            self._log_collective_benchmarks(
                benchmarked_collective_nodes,
                collective_keys,
                collective_medians,
                world_size,
            )

        log.info("Overlap scheduling: Runtime estimations aligned")

    def run(self) -> torch.fx.GraphModule:
        """Run the scheduling algorithm."""
        # All ranks must make identical decisions on overlap reordering,
        # Thus we must have identical runtime estimations across ranks.
        # For now we do benchmarking only for compute nodes.
        self._align_compute_nodes_runtime_estimations_across_all_distributed_ranks()

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

        if self.collective_bucketing:
            self._bucket_collectives()
        elif self.insert_overlap_deps:
            # If not bucketing, add effect tokens to preserve hiding dependencies
            self._add_effect_tokens_for_overlap()

        return self.gm

    def _add_effect_tokens_for_overlap(self) -> None:
        """
        Add effect tokens to preserve hiding dependency relationships when not bucketing.

        This ensures that communication-compute overlap is preserved through effect tokens
        when overlap preserving bucketing is not enabled.
        """
        from torch._inductor.fx_passes.control_dependencies import (
            preserve_node_ordering,
        )

        # Collect hiding dependencies: hiding_node -> collective_start, wait -> hiding_node
        additional_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)

        for start_node, info in self.collective_info.items():
            if info.hiding_node and not info.is_exposed:
                # Compute depends on collective start (compute must wait for collective to start)
                additional_deps[info.hiding_node].add(start_node)
                # Wait depends on compute (wait must wait for compute to finish)
                additional_deps[info.wait_node].add(info.hiding_node)

        # Apply effect tokens to preserve these dependencies
        if additional_deps:
            preserve_node_ordering(self.graph, additional_deps)

    def _handle_other(self, node: fx.Node) -> None:
        self._schedule(node)

    def _schedule(self, node: fx.Node) -> None:
        """Schedule a node."""
        assert node not in self.scheduled
        assert all(n in self.scheduled for n in node.all_input_nodes)
        self.scheduled.add(node)
        self.memory_tracker.schedule_node(node)

        log.debug(
            "Scheduled node %s: current_memory=%d bytes, total_scheduled=%d",
            node.name,
            self.memory_tracker.get_current_memory_bytes(),
            len(self.scheduled),
        )

        for user in node.users:
            self.in_degree[user] -= 1
            if self.in_degree[user] == 0:
                heapq.heappush(self.ready, (self._compute_score(user), user))

    def _compute_score(self, node: fx.Node) -> object:
        """Compute priority score for a node"""

        if is_wait_tensor(node):
            info = self.collective_info[self.wait_to_start[node]]
            # defer waits locally if they are exposed.
            compute_local_priority = int(info.is_exposed)
        else:
            # if we're scheduling this collective via its queue, then it was not
            # pre-fetched. we might as well maximize overlap for the
            # local, non-mm nodes prior to the next compute node.
            if self.in_overlappable_collective_unary_chain(node):
                compute_local_priority = -1
            else:
                compute_local_priority = 0

        return (
            self.compute_index_domination[node],  # what index compute it blocks
            compute_local_priority,  # collective_start=-1, wait=1, or neither=0
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
                return True

            if not self.is_cheap_fn(user):
                return False

            curr = user

        return False

    def _should_force_wait_for_memory(self) -> bool:
        """Check if we need to force a wait due to memory pressure"""
        if not self.in_flight:
            return False
        return self.in_flight_bytes >= self.max_in_flight_bytes or (
            self.memory_tracker.current_memory_bytes - self.original_peak_memory
        ) > gb_to_bytes(1.0)

    def _force_oldest_wait(self) -> None:
        """Schedule the oldest in flight wait"""
        self._handle_wait(self._get_oldest_wait())

    def _handle_collective_start(self, node: fx.Node) -> None:
        """Handle scheduling a collective start."""
        info = self.collective_info[node]

        if self.should_assume_bucketed(node):
            latency = estimate_collective_time(
                node, 0, custom_runtime_estimation=self.custom_runtime_estimation
            )
            assert latency <= info.exposed_time_ms
            info.exposed_time_ms = info.exposed_time_ms - latency

        self.in_flight[node] = info
        self.in_flight_bytes += info.size_bytes
        self.unscheduled_collectives.discard(node)
        self._schedule(node)

    def _handle_wait(self, node: fx.Node) -> None:
        """Handle scheduling a wait."""
        assert node in self.wait_to_start
        coll_start = self.wait_to_start[node]
        assert coll_start in self.in_flight

        # Scheduling a wait of a collective also forces the wait
        # of every node enqueued prior to the collective on the
        # same process group
        group_name = get_group_name(coll_start)
        to_schedule: list[fx.Node] = []
        for in_flight_coll in self.in_flight:
            if in_flight_coll == coll_start:
                break
            if get_group_name(in_flight_coll) == group_name:
                to_schedule.append(in_flight_coll)

        for coll_to_schedule in to_schedule:
            self._handle_wait(self.collective_info[coll_to_schedule].wait_node)

        self.in_flight_bytes -= self.in_flight[coll_start].size_bytes
        del self.in_flight[coll_start]
        self._schedule(node)

    def _handle_compute(self, node: fx.Node) -> None:
        """Handle scheduling compute and finding overlaps."""

        compute_time = benchmark_node(node, self.custom_runtime_estimation)
        available_compute = compute_time * self.compute_overlap_multipler

        # TODO: separate overlap time per process group
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
        self.current_compute_index += 1

    def _schedule_collectives_for_overlap(
        self, compute_node: fx.Node, available_compute_time: float
    ) -> None:
        """Opportunistically schedule collectives that can be hidden by compute."""
        compute_ancestors = self.node_ancestors[compute_node]

        # Filter collectives by distance and compute index domination
        possible_collectives = []
        for collective in self.unscheduled_collectives:
            distance = abs(self.node_idx[compute_node] - self.node_idx[collective])
            if distance > self.max_node_distance:
                break

            # Skip collectives that are too far ahead in compute index, but allow scheduling
            # collectives which are off compute path (which typically release memory)
            # TODO: we could potentially be more strict about limiting the amount of
            # pre-fetched memory before memory peak, and adjust allowed collective mem.
            if not self.off_compute_path(collective):
                if (
                    self.compute_index_domination[collective]
                    - self.current_compute_index
                ) > self.max_compute_pre_fetch:
                    continue

            possible_collectives.append(collective)

        possible_collectives = sorted(
            possible_collectives,
            key=lambda n: (self.compute_index_domination[n], self.node_idx[n]),
        )

        log.debug(
            "Scheduling collectives for overlap: compute_node=%s, available_time=%.2f ms, candidates=%d, current_memory=%d bytes",
            compute_node.name,
            available_compute_time,
            len(possible_collectives),
            self.memory_tracker.current_memory_bytes,
        )

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

            log.debug(
                "Overlapping collective %s with compute %s: coll_domination=%d, current_depth=%d",
                collective.name,
                compute_node.name,
                self.compute_index_domination[collective],
                self.current_compute_index,
            )

            # Schedule path to this collective
            self._schedule_path_to_collective(path, compute_node)
            self._handle_collective_start(collective)

            # Update the exposed time for this newly scheduled collective
            # after scheduling, which will account for latency reduction of bucketing
            overlap_amount = min(available_compute_time, info.exposed_time_ms)
            info.exposed_time_ms -= overlap_amount
            if info.exposed_time_ms == 0:
                info.hiding_node = compute_node
            available_compute_time -= overlap_amount

    def _find_schedulable_path(
        self, target: fx.Node, curr_compute_node: fx.Node | None
    ) -> OrderedSet[fx.Node] | None:
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
                if info.hiding_node and info.hiding_node != curr_compute_node:
                    continue
                elif node not in self.potentially_hidden_waits:
                    continue

                return None

        return unscheduled_ancestors

    def should_assume_bucketed(self, node: fx.Node) -> bool:
        """
        Check if there's an in-flight collective that can be bucketed with the given node. If so, assume they will bucket.
        This is a optimistic heuristic to account for latency reduction with bucketing. The two nodes may not get bucketed.
        """
        if not torch._inductor.config.test_configs.assume_bucketing_reduces_latency:
            return False

        key = bucket_key(node, mode="custom_ops_multidtype")
        if key is None:
            return False

        for in_flight_coll in self.in_flight:
            if bucket_key(in_flight_coll, mode="custom_ops_multidtype") == key:
                return True

        return False

    def _get_oldest_wait(self) -> fx.Node:
        oldest_start = next(iter(self.in_flight))
        return self.collective_info[oldest_start].wait_node

    def _wait_is_hidden(
        self, wait_node: fx.Node, compute_node: fx.Node | None = None
    ) -> bool:
        assert is_wait_tensor(wait_node)
        info = self.collective_info[self.wait_to_start[wait_node]]
        return not info.is_exposed and info.hiding_node != compute_node

    def _schedule_path_to_collective(
        self, path: OrderedSet[fx.Node], curr_compute_node: fx.Node
    ) -> None:
        """Schedule all nodes needed to reach a collective."""

        assert all(n not in self.scheduled for n in path)
        for node in sorted(path, key=lambda n: self.node_idx[n]):
            assert not (is_compute_node(node) or node in self.unscheduled_collectives)
            if is_wait_tensor(node):
                # When we schedule wait tensors, we also force realization of all
                # collectives enqueued prior to their corresponding collective.
                # It's possible the scheduling of one wait tensor here has forced
                # another in the path. If so, skip scheduling it.
                if node in self.scheduled:
                    continue

                info = self.collective_info[self.wait_to_start[node]]
                assert info.hiding_node != curr_compute_node
                self._handle_wait(node)
                continue

            self._schedule(node)

    def reorder_graph(self) -> None:
        output_node = self.graph.output_node()
        for node in self.scheduled:
            if node.op == "placeholder":
                continue
            output_node.prepend(node)
        self.graph.lint()

    def _reorder_graph(self) -> None:
        """Reorder graph based on schedule."""
        exposed = [
            c
            for c in self.collective_info.values()
            if c.exposed_time_ms == c.estimated_time_ms
        ]

        potentially_hidden_collectives = self.compute_potential_hidden_collectives(
            limit_coll_per_compute=True
        )
        bad_exposed = [
            c for c in exposed if c.start_node in potentially_hidden_collectives
        ]

        counters["inductor"]["overlap_scheduling_exposed"] += len(exposed)
        counters["inductor"]["overlap_scheduling_bad_exposed"] += len(bad_exposed)
        counters["inductor"]["overlap_scheduling_potentially_hidden"] += len(
            potentially_hidden_collectives
        )
        counters["inductor"]["overlap_original_mem"] = self.original_peak_memory
        counters["inductor"]["rescheduled_mem"] = self.memory_tracker.peak_memory

        log.info(
            "Overlap scheduling results: exposed=%d, bad_exposed=%d, potentially_hidden=%d, "
            "original_peak_memory=%d bytes, rescheduled_peak_memory=%d bytes",
            len(exposed),
            len(bad_exposed),
            len(potentially_hidden_collectives),
            self.original_peak_memory,
            self.memory_tracker.peak_memory,
        )

        self.reorder_graph()

    def _bucket_collectives(self) -> None:
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            node_ancestors=self.node_ancestors,
            scheduled=self.scheduled,
            max_bucket_memory_gb=1.0,  # Could make this configurable
            max_coll_distance=self.max_node_distance,
            insert_overlap_deps=self.insert_overlap_deps,
        )
        bucketer.bucket_collectives()

    def compute_potential_hidden_nodes(
        self, nodes_to_check: Iterable[fx.Node], limit_coll_per_compute: bool = False
    ) -> dict[fx.Node, fx.Node]:
        """
        Returns a dict containing a mapping of nodes which could potentially be hidden to their hiding node
        """

        used_compute_nodes: OrderedSet[fx.Node] = OrderedSet()

        def could_be_hidden(start: fx.Node) -> fx.Node | None:
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

    def compute_potential_hidden_collectives(
        self, limit_coll_per_compute: bool = False
    ) -> dict[fx.Node, fx.Node]:
        """Compute which collective operations could be hidden by compute."""
        return self.compute_potential_hidden_nodes(
            self.collective_info.keys(), limit_coll_per_compute
        )

    def compute_potential_hidden_waits(
        self, limit_coll_per_compute: bool = False
    ) -> dict[fx.Node, fx.Node]:
        """Compute which wait operations could be hidden by compte."""
        wait_nodes = [info.wait_node for info in self.collective_info.values()]
        return self.compute_potential_hidden_nodes(wait_nodes, limit_coll_per_compute)


def schedule_overlap_bucketing(
    gm: torch.fx.GraphModule,
    max_in_flight_gb: float = 2.0,
    max_compute_pre_fetch: int = 5,
    collective_bucketing: bool = False,
    insert_overlap_deps: bool = False,
    compute_overlap_multipler: float = 1.0,
    max_coll_distance: int = 1000,
    custom_runtime_estimation: Callable[[fx.Node], float | None] | None = None,
    collective_estimator: Literal["analytical", "benchmark"] = "analytical",
) -> torch.fx.GraphModule:
    """Schedule nodes to maximize compute-collective overlap.

    Args:
        gm: Input graph module to optimize.
        max_in_flight_gb: Maximum GB of concurrent collective data. Too much in flight memory
            can cause memory fragmentation within the CUDA Caching Allocator.
        max_compute_pre_fetch: Maximum compute node prefetch distance.
        collective_bucketing: Enable overlap-preserving collective bucketing.
        insert_overlap_deps: Insert overlap dependencies using control deps operator. This should only be used if
            compiling with inductor, or for subsequent passes before removing the ops prior to execution.
        compute_overlap_multipler: Scale factor for compute time used to hide collectives. This can be used
            to address over or under aggressive overlapping.
        max_coll_distance: Maximum node distance for overlap or bucketing. Mostly intended to reduce compile time.
        custom_runtime_estimation: Custom runtime estimation function that estimates runtime in ms for an fx node.
            If None, uses default estimations. This is currently limited to collectives and compute nodes.
        collective_estimator: Method for estimating collective runtime. "analytical" uses bandwidth formulas,
            "benchmark" uses CUDA events with power-of-2 rounding and interpolation.
    """

    return OverlapScheduler(
        gm,
        compute_overlap_multipler=compute_overlap_multipler,
        max_in_flight_gb=max_in_flight_gb,
        max_coll_distance=max_coll_distance,
        max_compute_pre_fetch=max_compute_pre_fetch,
        custom_runtime_estimation=custom_runtime_estimation,
        collective_bucketing=collective_bucketing,
        insert_overlap_deps=insert_overlap_deps,
        collective_estimator=collective_estimator,
    ).run()
