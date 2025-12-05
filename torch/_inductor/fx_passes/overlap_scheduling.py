import functools
import heapq
import itertools
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.fx as fx
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.comm_analysis import estimate_fx_collective_memory_footprint
from torch._inductor.fx_passes.bucketing import _schedulable_wait_node, is_wait_tensor
from torch._inductor.fx_passes.memory_estimator import MemoryTracker
from torch.fx.operator_schemas import normalize_function
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import _disable_current_modes


log = logging.getLogger(__name__)

from torch._inductor.fx_passes.bucketing import bucket_key

from ..pattern_matcher import stable_topological_sort


@dataclass
class WhyNoOverlap:
    """Track reasons why a collective cannot overlap with compute."""

    compute_name: str
    collective_name: str

    def __init__(self, compute_node: fx.Node, collective_node: fx.Node) -> None:
        self.compute_name = compute_node.name
        self.collective_name = collective_node.name

    def __call__(self, reason: str, *args: Any) -> None:
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "cannot overlap %s with %s: " + reason,  # noqa: G003
                self.collective_name,
                self.compute_name,
                *args,
            )


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
    custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
    | None = None,
    override_size: int | None = None,
) -> float | None:
    if custom_runtime_estimation is None:
        return None

    return custom_runtime_estimation(n, override_size)


def estimate_collective_time(
    n: fx.Node,
    override_size: int | None = None,
    custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
    | None = None,
) -> float:
    """Estimate the runtime of a collective operation, optionally with an overridden size."""
    if (
        est := get_custom_estimation(n, custom_runtime_estimation, override_size)
    ) is not None:
        return est

    # Use analytical model (benchmarking is handled separately in alignment)
    return torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
        n, override_size
    )


def is_compute_node(n: fx.Node) -> bool:
    """
    Should we consider this node computationally expensive ?
    Currently uses flop registration, but we could expand more generally.
    """
    return (
        getattr(n.target, "overloadpacket", None)
        in torch.utils.flop_counter.flop_registry
    )


def is_reduce_scatter(n: fx.Node) -> bool:
    """Check if node is a reduce_scatter collective."""
    return "reduce_scatter" in str(n.target).lower()


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
    custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
    | None = None,
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

        if (
            est := get_custom_estimation(n, custom_runtime_estimation, None)
        ) is not None:
            set_cached_node_time(key, est)
            return est, key

        bench = get_collective_do_bench()
        out = bench(lambda: n.target(*args, **kwargs))  # type: ignore[operator]
        set_cached_node_time(key, out)
        return out, key


def benchmark_node(
    n: fx.Node,
    custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
    | None = None,
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
    hiding_nodes: OrderedSet[fx.Node] = field(default_factory=OrderedSet)

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
        custom_runtime_estimation: Callable[[fx.Node, int | None], float | None] | None,
        collective_estimator: Literal["analytical", "benchmark"],
        max_memory_increase_gb: float | None = 1.0,
        max_memory_increase_ratio: float | None = 0.05,
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

        # Identify compute nodes early (needed for baseline memory computation)
        self.compute_nodes = [n for n in self.nodes if is_compute_node(n)]
        self.current_compute_index = 0

        # Compute baseline memory profile from original schedule
        self.original_mem_before_compute_index: list[int] = []
        self.original_peak_memory = self._compute_baseline_memory()

        # Maximum allowed peak memory = baseline + max(absolute, ratio * baseline)
        # When both limits are specified, use the more permissive one
        memory_increase_bytes = None
        if max_memory_increase_gb is not None:
            memory_increase_bytes = gb_to_bytes(max_memory_increase_gb)
        if max_memory_increase_ratio is not None:
            ratio_increase = int(self.original_peak_memory * max_memory_increase_ratio)
            memory_increase_bytes = (
                max(memory_increase_bytes, ratio_increase)
                if memory_increase_bytes is not None
                else ratio_increase
            )
        if memory_increase_bytes is None:
            memory_increase_bytes = 0

        self.allowed_peak_memory_bytes = (
            self.original_peak_memory + memory_increase_bytes
        )

        # Track cumulative prefetch memory at each compute index
        # When we prefetch a collective at compute index i that will be used at index j,
        # it adds memory from i to j, so we need to track this cumulative effect
        self.cumulative_prefetch_mem_by_compute_index: list[int] = [
            0 for _ in range(len(self.compute_nodes))
        ]

        self.memory_tracker = MemoryTracker(self.graph)

        self.wait_to_start: dict[fx.Node, fx.Node] = {}
        self._identify_collectives()
        self.wasted_compute = 0.0

        self.compute_index_domination = self._calculate_compute_node_domination_index()

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

        # Track deferred memory freeing from waits
        # When we defer a wait, we also defer the memory freeing from its downstream nodes
        self.wait_freeing_potential: dict[fx.Node, int] = self._compute_wait_freeing_potential()

    def _collect_node_ancestors(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """Collect all ancestors for each node."""
        ancestors: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in self.nodes:
            for input_node in node.all_input_nodes:
                ancestors[node].add(input_node)
                ancestors[node] |= ancestors[input_node]

        return ancestors

    def _compute_wait_freeing_potential(self) -> dict[fx.Node, int]:
        """
        Compute how much memory would be freed by scheduling each wait node.

        When a wait is scheduled, its direct dependents can become ready.
        If those dependents are the last users of some storages, they free memory.
        This estimates that freeing potential.
        """
        wait_freeing: dict[fx.Node, int] = {}
        alias_tracker = self.memory_tracker.alias_tracker

        for wait_node in self.wait_to_start:
            freeing = 0

            # Look at immediate dependents of the wait
            for user in wait_node.users:
                # Check if this user would free any storages when scheduled
                input_storages = alias_tracker.get_storage_uses(user)
                for storage_key in input_storages:
                    if not self.memory_tracker.device_filter(storage_key.device):
                        continue

                    allocator = alias_tracker.storage_to_allocator[storage_key]
                    if not self.memory_tracker.is_releasable(allocator):
                        continue

                    # Check if user is the last user of this storage
                    last_user = alias_tracker.storage_to_last_user.get(storage_key)
                    if last_user == user:
                        freeing += self.memory_tracker._get_storage_size(storage_key)

            wait_freeing[wait_node] = freeing

        total_freeing = sum(wait_freeing.values())
        log.debug(
            "Computed wait freeing potential: %d waits, total %d MB",
            len(wait_freeing),
            total_freeing // (1024 * 1024),
        )

        return wait_freeing

    def _compute_baseline_memory(self) -> int:
        """
        Simulate the original schedule to compute baseline memory profile.
        Returns the peak memory observed during simulation.
        """
        baseline_tracker = MemoryTracker(self.graph)

        last_compute_max_memory = 0
        peak_memory = 0

        # Track where original peak occurs
        self._original_peak_index = 0
        self._original_peak_node = None

        for idx, node in enumerate(self.nodes):
            baseline_tracker.schedule_node(node)
            current_mem = baseline_tracker.current_memory_bytes

            # Track peak
            if current_mem > peak_memory:
                peak_memory = current_mem
                self._original_peak_index = idx
                self._original_peak_node = node.name

            # Record the max memory between this and previous compute node
            last_compute_max_memory = max(last_compute_max_memory, current_mem)

            if is_compute_node(node):
                self.original_mem_before_compute_index.append(last_compute_max_memory)
                last_compute_max_memory = current_mem

        return peak_memory

    def _prefetch_would_exceed_memory_budget(self, start_node: fx.Node) -> bool:
        """
        Check if prefetching this collective would exceed memory budget at ANY compute node
        between now and when it's used.
        """
        info = self.collective_info[start_node]
        size = info.size_bytes

        domination_index = self.compute_index_domination[start_node]

        # For off-path collectives (don't block compute), check current memory budget.
        # reduce_scatter is memory-beneficial (consumes large input, produces small output)
        # so we allow it even when tight on memory. Other off-path collectives are blocked
        # if they would exceed the budget.
        if domination_index == sys.maxsize:
            if not hasattr(self, '_off_path_count'):
                self._off_path_count = 0
                self._off_path_rs_count = 0

            if is_reduce_scatter(start_node):
                self._off_path_rs_count += 1
                # reduce_scatter helps memory - always allow
                return False

            # Other off-path collectives (like all_gather) use memory
            self._off_path_count += 1
            current_mem = self.memory_tracker.current_memory_bytes
            # Check against budget
            if current_mem + size > self.allowed_peak_memory_bytes:
                return True
            return False

        # Track on-path prefetch attempts
        if not hasattr(self, '_on_path_prefetch_count'):
            self._on_path_prefetch_count = 0
            self._on_path_prefetch_blocked = 0
        self._on_path_prefetch_count += 1

        # check current mem
        current_mem = self.memory_tracker.current_memory_bytes
        if current_mem + size > self.allowed_peak_memory_bytes:
            self._on_path_prefetch_blocked += 1
            return True

        start_index = self.current_compute_index

        # then, check future mem
        for compute_idx in range(start_index, domination_index):
            cumulative_prefetch = self.cumulative_prefetch_mem_by_compute_index[
                compute_idx
            ]

            # Check 1: Would cumulative prefetch exceed in-flight limit?
            if (cumulative_prefetch + size) > self.max_in_flight_bytes:
                self._on_path_prefetch_blocked += 1
                return True

            # Check 2: Would total memory (baseline + cumulative prefetch) exceed budget?
            baseline_mem = self.original_mem_before_compute_index[compute_idx]
            projected = baseline_mem + cumulative_prefetch + size

            if projected > self.allowed_peak_memory_bytes:
                self._on_path_prefetch_blocked += 1
                return True

        return False

    def _update_cumulative_prefetch_memory(
        self, collective: fx.Node, info: CollectiveInfo, path_memory_cost: int = 0
    ) -> None:
        """
        Update cumulative prefetch memory for all compute indices this collective will be live.

        Args:
            collective: The collective node being prefetched
            info: CollectiveInfo for the collective
            path_memory_cost: Memory cost from scheduling path nodes to reach the collective
        """
        domination_index = self.compute_index_domination[collective]
        if domination_index == sys.maxsize:
            return

        # Track both collective size and path memory cost
        total_memory_cost = info.size_bytes + path_memory_cost

        for compute_idx in range(self.current_compute_index, domination_index):
            self.cumulative_prefetch_mem_by_compute_index[compute_idx] += (
                total_memory_cost
            )

    def off_compute_path(self, n: fx.Node) -> bool:
        """Check if a node is off the compute path (doesn't block any compute)."""
        return self.compute_index_domination[n] == sys.maxsize

    def _identify_collectives(self) -> None:
        """Identify all collective operations and process groups."""
        self.all_pgs: OrderedSet[str] = OrderedSet()

        for node in self.nodes:
            if _schedulable_wait_node(node):
                start = node.args[0]
                coll_time_ms = estimate_collective_time(
                    start, custom_runtime_estimation=self.custom_runtime_estimation
                )

                info = CollectiveInfo(
                    start_node=start,
                    wait_node=node,
                    size_bytes=estimate_fx_collective_memory_footprint(start),
                    estimated_time_ms=coll_time_ms,
                    exposed_time_ms=coll_time_ms,  # Initially fully exposed
                )
                self.collective_info[start] = info
                self.wait_to_start[node] = start
                self.unscheduled_collectives.add(start)
                self.all_pgs.add(get_group_name(start))

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
                if (
                    get_custom_estimation(n, self.custom_runtime_estimation, None)
                    is not None
                ):
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

            # Proactive memory management: force high-freeing-potential waits
            # before memory builds up too much
            if self._should_force_wait_proactively():
                self._force_highest_freeing_wait()
                continue

            _, node = heapq.heappop(self.ready)

            # we don't always remove nodes from the heap when we schedule them
            if node in self.scheduled:
                continue

            # Check if we should schedule a wait instead of this node to free memory
            # This is a dynamic check that happens AFTER popping, not at push time
            if not _schedulable_wait_node(node) and self._should_schedule_wait_instead(node):
                # Push the current node back and schedule a wait instead
                heapq.heappush(self.ready, (self._compute_score(node), node))
                self._force_highest_freeing_wait()
                continue

            if node.op == "placeholder":
                self._schedule(node)
            elif node in self.collective_info:
                self._handle_collective_start(node)
            elif _schedulable_wait_node(node):
                self._handle_wait(node)
            else:
                self._handle_compute_or_other(node)

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
            if info.is_exposed:
                continue
            for hn in info.hiding_nodes:
                # Compute depends on collective start (compute must wait for collective to start)
                additional_deps[hn].add(start_node)
                # Wait depends on compute (wait must wait for compute to finish)
                additional_deps[info.wait_node].add(hn)

        # Apply effect tokens to preserve these dependencies
        if additional_deps:
            preserve_node_ordering(self.graph, additional_deps)

    def get_non_collective_runtime_estimate(self, node: fx.Node) -> float | None:
        """Get runtime estimation for a node in ms. Returns None if no estimation is available."""

        # TODO: non custom estimation of aten nodes, potentially requires notion of fusion group
        if is_compute_node(node):
            return benchmark_node(node, self.custom_runtime_estimation)

        if self.custom_runtime_estimation is None:
            return None

        return self.custom_runtime_estimation(node, None)

    def _reduce_exposed_time_of_in_flight_collectives(
        self,
        node: fx.Node,
        available_compute: float,
        exclude_pg: str | None = None,
    ) -> dict[str, float]:
        """
        Reduce exposed time of in-flight collectives using available compute time.

        Collectives on different process groups can overlap simultaneously with the same
        compute, so we track remaining time separately per PG.
        """
        # Initialize all PGs with full available compute (except excluded)
        remaining_time_per_pg: dict[str, float] = {
            pg: available_compute for pg in self.all_pgs if pg != exclude_pg
        }

        for start_node, info in self.in_flight.items():
            if info.exposed_time_ms == 0:
                continue

            pg_name = get_group_name(start_node)
            if pg_name == exclude_pg:
                continue

            pg_remaining = remaining_time_per_pg[pg_name]
            if pg_remaining <= 0:
                continue

            overlap_amount = min(info.exposed_time_ms, pg_remaining)
            info.exposed_time_ms -= overlap_amount
            remaining_time_per_pg[pg_name] -= overlap_amount
            info.hiding_nodes.add(node)

        return remaining_time_per_pg

    def _handle_compute_or_other(self, node: fx.Node) -> None:
        """Handle scheduling compute or other nodes and attempt to overlap with collectives."""
        runtime_estimate = self.get_non_collective_runtime_estimate(node)

        # TODO: we could consider skipping overlapping for overlapable, unary chains to collectives.
        # using these nodes for overlap prevents bucketing. potentially if chain time < latency
        if runtime_estimate is None:
            assert not is_compute_node(node), "should have estimate for compute nodes"
            self._schedule(node)
            return

        available_compute = runtime_estimate * self.compute_overlap_multipler

        # First, reduce exposed time of in-flight collectives (per PG)
        remaining_time_per_pg = self._reduce_exposed_time_of_in_flight_collectives(
            node, available_compute
        )
        # Then, schedule new collectives for overlap
        self._schedule_collectives_for_overlap(node, remaining_time_per_pg)
        self._schedule(node)

        if is_compute_node(node):
            self.current_compute_index += 1

    def _schedule(self, node: fx.Node) -> None:
        """Schedule a node."""
        assert node not in self.scheduled
        assert all(n in self.scheduled for n in node.all_input_nodes)
        self.scheduled.add(node)

        mem_before = self.memory_tracker.current_memory_bytes
        self.memory_tracker.schedule_node(node)
        mem_after = self.memory_tracker.current_memory_bytes

        # Track when we hit peak memory
        if mem_after == self.memory_tracker.peak_memory and mem_after > mem_before:
            self._peak_memory_node = node.name
            self._peak_memory_index = len(self.scheduled)
            self._peak_memory_bytes = mem_after

        # Track memory freed by nodes that depend on waits
        if not hasattr(self, '_wait_user_mem_freed'):
            self._wait_user_mem_freed = 0
            self._wait_user_count = 0

        # Check if this node depends on a wait (has a wait in its inputs)
        for inp in node.all_input_nodes:
            if _schedulable_wait_node(inp):
                mem_freed = mem_before - mem_after
                if mem_freed > 0:
                    self._wait_user_mem_freed += mem_freed
                    self._wait_user_count += 1
                break

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

        if _schedulable_wait_node(node):
            info = self.collective_info[self.wait_to_start[node]]
            # Defer waits locally if they are exposed, BUT only if:
            # 1. Memory is not under pressure
            # 2. Deferring this wait won't cause us to exceed budget (considering its freeing potential)
            current_mem = self.memory_tracker.current_memory_bytes
            memory_under_pressure = current_mem > self.allowed_peak_memory_bytes

            # Get the freeing potential of this wait
            freeing_potential = self.wait_freeing_potential.get(node, 0)

            # Check if deferring this wait would risk exceeding memory budget
            # Use original profile as reference - allow only small increase above original
            if self.current_compute_index < len(self.original_mem_before_compute_index):
                original_target = self.original_mem_before_compute_index[
                    self.current_compute_index
                ]
            else:
                original_target = self.original_peak_memory

            # Allow 1GB headroom above original profile
            headroom = original_target + gb_to_bytes(1.0) - current_mem
            defer_would_risk_budget = headroom < freeing_potential

            if not hasattr(self, '_wait_score_count'):
                self._wait_score_count = 0
                self._wait_score_under_pressure = 0
                self._wait_score_risk_budget = 0
            self._wait_score_count += 1
            if memory_under_pressure:
                self._wait_score_under_pressure += 1
                # Don't defer waits when memory is high
                compute_local_priority = 0
            elif defer_would_risk_budget and freeing_potential > 0:
                self._wait_score_risk_budget += 1
                # Don't defer waits that have high freeing potential and little headroom
                compute_local_priority = 0
            else:
                # Normal behavior: defer exposed waits
                compute_local_priority = int(info.is_exposed)
        else:
            # if we're scheduling this collective via its queue, then it was not
            # pre-fetched. we might as well maximize overlap for the
            # local, non-mm nodes prior to the next compute node.
            if self.in_overlappable_collective_unary_chain(node):
                compute_local_priority = -1
            else:
                compute_local_priority = 0

        # Compute a memory-aware score that limits deviation from original order
        # when current memory is high relative to original profile
        current_mem = self.memory_tracker.current_memory_bytes
        if self.current_compute_index < len(self.original_mem_before_compute_index):
            original_target = self.original_mem_before_compute_index[
                self.current_compute_index
            ]
        else:
            original_target = self.original_peak_memory

        # If we're over budget relative to original profile, prioritize original order
        # This limits reordering when it would cause memory increases
        memory_tight = current_mem > original_target + gb_to_bytes(1.0)

        if memory_tight:
            # When memory is tight, use original order as primary sort key
            # This prevents further reordering that could increase memory
            return (
                0,  # Don't deprioritize based on domination
                0,  # Don't deprioritize based on wait/collective
                self.node_idx[node],  # Stick to original order
            )

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
        """Check if we need to force a wait due to memory pressure.

        Force a wait if either:
        1. In-flight bytes exceed limit (existing check)
        2. Actual tracked memory exceeds allowed peak budget
        """
        if not self.in_flight:
            return False

        # Check 1: in-flight memory limit
        if self.in_flight_bytes >= self.max_in_flight_bytes:
            return True

        # Check 2: actual memory exceeds budget - force a wait to try to free memory
        current_mem = self.memory_tracker.current_memory_bytes
        if current_mem > self.allowed_peak_memory_bytes:
            if not hasattr(self, '_force_wait_for_memory_count'):
                self._force_wait_for_memory_count = 0
            self._force_wait_for_memory_count += 1
            return True

        return False

    def _force_oldest_wait(self) -> None:
        """Schedule the oldest in flight wait"""
        self._handle_wait(self._get_oldest_wait())

    def _should_force_wait_proactively(self) -> bool:
        """Check if we should proactively force a high-freeing-potential wait.

        Returns True if:
        1. There are in-flight collectives (waits are available)
        2. Memory exceeds the ORIGINAL baseline at this point in scheduling
        3. A high-freeing-potential wait is ready

        This is more aggressive than the budget check - we try to stay close to
        the original memory profile, not just within the budget.
        """
        if not self.in_flight:
            return False

        # Get the original memory target at the current compute index
        # This is the baseline we want to stay close to
        if self.current_compute_index < len(self.original_mem_before_compute_index):
            original_target = self.original_mem_before_compute_index[
                self.current_compute_index
            ]
        else:
            original_target = self.original_peak_memory

        current_mem = self.memory_tracker.current_memory_bytes

        # Force waits if we're exceeding the original profile significantly
        # Allow some headroom (configured memory increase) before forcing
        max_allowed = original_target + gb_to_bytes(1.0)  # 1 GB headroom above original

        if current_mem <= max_allowed:
            return False

        # Check if there's a high-freeing-potential wait that's ready
        best_freeing = self._get_best_ready_wait_freeing()
        if best_freeing is None:
            return False

        wait_node, freeing_potential = best_freeing
        # Force if the wait would help (any significant freeing potential)
        if freeing_potential > gb_to_bytes(0.5):  # At least 0.5 GB freeing
            if not hasattr(self, '_proactive_force_count'):
                self._proactive_force_count = 0
            self._proactive_force_count += 1
            return True

        return False

    def _should_schedule_wait_instead(self, popped_node: fx.Node) -> bool:
        """Check if we should schedule a wait instead of the popped node.

        This is a dynamic check that happens AFTER popping from the heap,
        allowing us to react to current memory state rather than stale scores.
        """
        if not self.in_flight:
            return False

        # Get current memory and compare to original profile
        current_mem = self.memory_tracker.current_memory_bytes
        if self.current_compute_index < len(self.original_mem_before_compute_index):
            original_target = self.original_mem_before_compute_index[
                self.current_compute_index
            ]
        else:
            original_target = self.original_peak_memory

        # Only intervene if we're significantly above original profile (1 GB threshold)
        if current_mem <= original_target + gb_to_bytes(1.0):
            return False

        # Check if there's a ready wait with good freeing potential
        best_freeing = self._get_best_ready_wait_freeing()
        if best_freeing is None:
            return False

        wait_node, freeing_potential = best_freeing

        # Schedule wait if it would help reduce memory significantly
        if freeing_potential > gb_to_bytes(0.5):
            if not hasattr(self, '_swap_for_wait_count'):
                self._swap_for_wait_count = 0
            self._swap_for_wait_count += 1
            return True

        return False

    def _get_best_ready_wait_freeing(self) -> tuple[fx.Node, int] | None:
        """Find the ready wait with highest freeing potential."""
        best_wait = None
        best_freeing = 0

        # Look through in-flight collectives for their waits
        for start_node in self.in_flight:
            wait_node = self.collective_info[start_node].wait_node

            # Check if the wait is in the ready queue (all deps satisfied)
            if self.in_degree[wait_node] > 0:
                continue

            freeing = self.wait_freeing_potential.get(wait_node, 0)
            if freeing > best_freeing:
                best_freeing = freeing
                best_wait = wait_node

        if best_wait is None:
            return None
        return best_wait, best_freeing

    def _force_highest_freeing_wait(self) -> None:
        """Force the ready wait with highest freeing potential.

        Also schedules the wait's direct users that would free memory,
        since scheduling just the wait doesn't directly free memory.
        """
        result = self._get_best_ready_wait_freeing()
        if result is None:
            # Fallback to oldest wait
            self._force_oldest_wait()
            return

        wait_node, _ = result
        self._handle_wait(wait_node)

        # Now also schedule the wait's users that would free memory
        # These users are what actually free memory, not the wait itself
        self._schedule_memory_freeing_users(wait_node)

    def _schedule_memory_freeing_users(self, wait_node: fx.Node) -> None:
        """Schedule users of a wait that would free significant memory."""
        alias_tracker = self.memory_tracker.alias_tracker

        # Debug: track why users aren't being scheduled
        if not hasattr(self, '_user_skip_reasons'):
            self._user_skip_reasons = {'scheduled': 0, 'not_ready': 0, 'special_node': 0, 'low_freeing': 0}

        for user in wait_node.users:
            # Skip if already scheduled
            if user in self.scheduled:
                self._user_skip_reasons['scheduled'] += 1
                continue

            # Skip if not ready (has unscheduled dependencies)
            if self.in_degree[user] > 0:
                self._user_skip_reasons['not_ready'] += 1
                continue

            # Skip waits, collectives, and compute nodes - let them go through normal scheduling
            if _schedulable_wait_node(user) or user in self.collective_info or is_compute_node(user):
                self._user_skip_reasons['special_node'] += 1
                continue

            # Check how much memory this user would free
            input_storages = alias_tracker.get_storage_uses(user)
            freeing = 0
            for storage_key in input_storages:
                if not self.memory_tracker.device_filter(storage_key.device):
                    continue
                allocator = alias_tracker.storage_to_allocator[storage_key]
                if not self.memory_tracker.is_releasable(allocator):
                    continue

                # Check if all other uses are scheduled
                all_uses = alias_tracker.storage_to_uses[storage_key]
                if all(u in self.scheduled or u == user for u in all_uses):
                    freeing += self.memory_tracker._get_storage_size(storage_key)

            # Schedule if it would free significant memory
            if freeing > gb_to_bytes(0.1):  # At least 0.1 GB
                self._schedule(user)
                if not hasattr(self, '_forced_user_schedule_count'):
                    self._forced_user_schedule_count = 0
                self._forced_user_schedule_count += 1
            else:
                self._user_skip_reasons['low_freeing'] += 1

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

        # If we are waiting on an exposed collective, use this time to
        # overlap on other PGs.
        info = self.collective_info[coll_start]
        if info.exposed_time_ms > 0:
            exposed_time = info.exposed_time_ms
            exclude_pg = group_name

            remaining_time_per_pg = self._reduce_exposed_time_of_in_flight_collectives(
                node, exposed_time, exclude_pg=exclude_pg
            )
            self._schedule_collectives_for_overlap(
                node, remaining_time_per_pg, exclude_pg=exclude_pg
            )

        self.in_flight_bytes -= self.in_flight[coll_start].size_bytes
        del self.in_flight[coll_start]
        self._schedule(node)

    def _schedule_collectives_for_overlap(
        self,
        overlap_node: fx.Node,
        remaining_time_per_pg: dict[str, float],
        exclude_pg: str | None = None,
    ) -> None:
        """Opportunistically schedule collectives that can be hidden by available overlap time."""
        if not remaining_time_per_pg or all(
            t <= 0 for t in remaining_time_per_pg.values()
        ):
            return

        overlap_node_ancestors = self.node_ancestors[overlap_node]

        # Compile candidates - limit by distance to bound compile time
        candidates = []
        for i, collective in enumerate(self.unscheduled_collectives):
            if i > self.max_node_distance:
                break

            pg_name = get_group_name(collective)
            if pg_name == exclude_pg:
                continue

            if (
                not self.off_compute_path(collective)
                and self.compute_index_domination[collective]
                - self.current_compute_index
                > self.max_compute_pre_fetch
            ):
                continue

            candidates.append(collective)

        # Sort candidates prioritizing:
        # 1. reduce_scatter operations (reduce memory pressure)
        # 2. Earlier domination index
        # 3. Original order for stability
        candidates.sort(
            key=lambda n: (
                not is_reduce_scatter(n),  # reduce_scatter first
                self.compute_index_domination[n],
                self.node_idx[n],
            ),
        )

        for collective in candidates:
            pg_name = get_group_name(collective)
            pg_available_time = remaining_time_per_pg[pg_name]

            if pg_available_time <= 0:
                continue

            why = WhyNoOverlap(overlap_node, collective)
            info = self.collective_info[collective]

            if (
                collective in overlap_node_ancestors
                or overlap_node in self.node_ancestors[collective]
            ):
                why("dependency conflict")
                continue

            # Check if prefetching would exceed memory budget
            if self._prefetch_would_exceed_memory_budget(collective):
                why("prefetch would exceed memory budget")
                continue

            # Try to free memory by forcing hidden waits
            while (
                self.in_flight
                and (self.max_in_flight_bytes - self.in_flight_bytes) < info.size_bytes
                and self._wait_is_hidden(self._get_oldest_wait(), overlap_node)
            ):
                self._force_oldest_wait()

            if (self.max_in_flight_bytes - self.in_flight_bytes) < info.size_bytes:
                why("in-flight memory limit")
                continue

            # Check if we can reach this collective without scheduling compute, other collectives, or waits
            path = self._find_schedulable_path(collective, overlap_node, why)
            if path is None:
                continue

            # Check if path + collective memory would exceed budget
            # Skip this check for reduce_scatter since it's memory-beneficial
            # (consumes large input, produces small output)
            if not is_reduce_scatter(collective):
                path_memory_cost = self._estimate_path_memory_cost(path)
                total_prefetch_cost = info.size_bytes + path_memory_cost
                current_mem = self.memory_tracker.current_memory_bytes
                if current_mem + total_prefetch_cost > self.allowed_peak_memory_bytes:
                    why(
                        "path memory would exceed budget (current=%d MB, path=%d MB, coll=%d MB, budget=%d MB)",
                        current_mem // (1024 * 1024),
                        path_memory_cost // (1024 * 1024),
                        info.size_bytes // (1024 * 1024),
                        self.allowed_peak_memory_bytes // (1024 * 1024),
                    )
                    if not hasattr(self, '_path_memory_blocked'):
                        self._path_memory_blocked = 0
                    self._path_memory_blocked += 1
                    continue

            log.debug(
                "Overlapping collective %s with node %s: coll_domination=%d, current_depth=%d",
                collective.name,
                overlap_node.name,
                self.compute_index_domination[collective],
                self.current_compute_index,
            )

            # TODO: We previously tracked path compute time and added it back to available
            # overlap time. With per-PG tracking this is complex: if there were in-flight
            # collectives on one PG but not another, we can't add path time back to the PG
            # that wasn't in-flight

            # Schedule path and collective
            path_memory_cost = self._estimate_path_memory_cost(path)

            # Track estimated vs actual path memory for debugging
            mem_before = self.memory_tracker.current_memory_bytes
            self._schedule_path_to_collective(path, overlap_node)
            mem_after_path = self.memory_tracker.current_memory_bytes
            actual_path_cost = mem_after_path - mem_before

            if not hasattr(self, '_path_estimate_vs_actual'):
                self._path_estimate_vs_actual = []
            self._path_estimate_vs_actual.append((path_memory_cost, actual_path_cost))

            self._handle_collective_start(collective)
            # Track both collective and path memory in cumulative prefetch
            self._update_cumulative_prefetch_memory(collective, info, path_memory_cost)

            # Update exposed time for this collective
            overlap_amount = min(pg_available_time, info.exposed_time_ms)
            info.exposed_time_ms -= overlap_amount
            info.hiding_nodes.add(overlap_node)

            # Update available time for this PG
            remaining_time_per_pg[pg_name] -= overlap_amount

            if sum(remaining_time_per_pg.values()) == 0:
                break

        if remaining_time_per_pg:
            self.wasted_compute += min(remaining_time_per_pg.values())

    def _estimate_path_memory_cost(self, path: OrderedSet[fx.Node]) -> int:
        """
        Estimate memory cost of scheduling a path to a collective.

        This accounts for fresh allocations from path nodes. Nodes with users
        outside the path will have their outputs stay live longer, but we
        conservatively estimate based on fresh allocations only.
        """
        total_cost = 0
        alias_tracker = self.memory_tracker.alias_tracker

        for node in path:
            # Skip waits - they don't allocate new memory
            if _schedulable_wait_node(node):
                continue

            # Get fresh allocations for this node
            fresh_allocations = alias_tracker.get_fresh_allocations(node)
            for storage_key in fresh_allocations:
                if self.memory_tracker.device_filter(storage_key.device):
                    total_cost += self.memory_tracker._get_storage_size(storage_key)

        return total_cost

    def _find_schedulable_path(
        self, target: fx.Node, curr_overlap_node: fx.Node | None, why: WhyNoOverlap
    ) -> OrderedSet[fx.Node] | None:
        """Find path to target by collecting unscheduled dependencies."""
        # Get unscheduled ancestors
        unscheduled_ancestors = self.node_ancestors[target] - self.scheduled

        # only schedule non distributed, non compute nodes
        for node in unscheduled_ancestors:
            if is_compute_node(node):
                why("path blocked by compute node %s", node.name)
                return None

            if node in self.unscheduled_collectives:
                why("path blocked by unscheduled collective %s", node.name)
                return None

            # if we schedule a wait tensor whose start collective is hidden by the
            # current compute node we are scheduling, then we are effectively exposing it.
            # similarly, dont schedule a wait of a collective that could be otherwise hidden,
            # thus forcing it to be exposed.
            # however, if it is already hidden it's fine to schedule it
            if _schedulable_wait_node(node):
                info = self.collective_info[self.wait_to_start[node]]
                # Allow if fully hidden by other nodes
                if not info.is_exposed and curr_overlap_node not in info.hiding_nodes:
                    continue

                why(
                    "path blocked by wait node %s (exposed=%s, hiding_nodes=%s)",
                    node.name,
                    info.is_exposed,
                    curr_overlap_node in info.hiding_nodes,
                )

            # Skip c10 ops and dtensor shard ops - they should be scheduled via main loop
            target_str = str(node.target)
            if "c10" in target_str or "_dtensor" in target_str:
                log.debug(
                    "Skipping c10/dtensor op %s in path to collective",
                    node.name,
                )
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
        self, wait_node: fx.Node, overlap_node: fx.Node | None = None
    ) -> bool:
        assert is_wait_tensor(wait_node)
        info = self.collective_info[self.wait_to_start[wait_node]]
        return not info.is_exposed and overlap_node not in info.hiding_nodes

    def _schedule_path_to_collective(
        self, path: OrderedSet[fx.Node], curr_overlap_node: fx.Node
    ) -> None:
        """Schedule all nodes needed to reach a collective."""

        # Track nesting level for prefetches
        if not hasattr(self, '_prefetch_nesting_level'):
            self._prefetch_nesting_level = 0
            self._nested_prefetch_count = 0
        self._prefetch_nesting_level += 1
        if self._prefetch_nesting_level > 1:
            self._nested_prefetch_count += 1

        # Track memory impact of scheduling path nodes
        mem_before_path = self.memory_tracker.current_memory_bytes

        assert all(n not in self.scheduled for n in path)
        for node in sorted(path, key=lambda n: self.node_idx[n]):
            assert not (is_compute_node(node) or node in self.unscheduled_collectives)
            if _schedulable_wait_node(node):
                # When we schedule wait tensors, we also force realization of all
                # collectives enqueued prior to their corresponding collective.
                # It's possible the scheduling of one wait tensor here has forced
                # another in the path. If so, skip scheduling it.
                if node in self.scheduled:
                    continue

                info = self.collective_info[self.wait_to_start[node]]
                assert curr_overlap_node not in info.hiding_nodes
                self._handle_wait(node)
                continue

            self._schedule(node)

        # Track path memory impact
        mem_after_path = self.memory_tracker.current_memory_bytes
        path_mem_delta = mem_after_path - mem_before_path
        if not hasattr(self, '_path_mem_increases'):
            self._path_mem_increases = []
        self._path_mem_increases.append(path_mem_delta)

        # Decrement nesting level
        self._prefetch_nesting_level -= 1

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

        potentially_hidden_collectives = self.compute_potential_hidden_collectives()
        bad_exposed = [
            c for c in exposed if c.start_node in potentially_hidden_collectives
        ]

        # Compute total exposed and potential exposed time
        total_exposed = sum(c.exposed_time_ms for c in self.collective_info.values())
        hideable_exposed_ms = sum(
            self.collective_info[c].exposed_time_ms
            for c in potentially_hidden_collectives
        )
        total_potential_exposed = sum(
            c.estimated_time_ms for c in self.collective_info.values()
        )

        counters["inductor"]["overlap_scheduling_exposed"] += len(exposed)
        counters["inductor"]["overlap_scheduling_bad_exposed"] += len(bad_exposed)
        counters["inductor"]["overlap_scheduling_potentially_hidden"] += len(
            potentially_hidden_collectives
        )
        counters["inductor"]["overlap_original_mem"] = self.original_peak_memory
        counters["inductor"]["rescheduled_mem"] = self.memory_tracker.peak_memory

        # Debug: print off-path prefetch stats
        off_path_count = getattr(self, '_off_path_count', 0)
        off_path_rs_count = getattr(self, '_off_path_rs_count', 0)
        print(f"DEBUG off-path prefetch checks: non-reduce_scatter={off_path_count}, reduce_scatter={off_path_rs_count}")

        # Count on-path vs off-path collectives
        on_path = sum(1 for c in self.collective_info if self.compute_index_domination[c] != sys.maxsize)
        off_path = sum(1 for c in self.collective_info if self.compute_index_domination[c] == sys.maxsize)
        print(f"DEBUG collective counts: on_path={on_path}, off_path={off_path}, total={len(self.collective_info)}")

        # Breakdown by type
        on_path_rs = sum(1 for c in self.collective_info if self.compute_index_domination[c] != sys.maxsize and is_reduce_scatter(c))
        off_path_rs = sum(1 for c in self.collective_info if self.compute_index_domination[c] == sys.maxsize and is_reduce_scatter(c))
        on_path_ag = on_path - on_path_rs
        off_path_ag = off_path - off_path_rs
        print(f"DEBUG on_path: reduce_scatter={on_path_rs}, other={on_path_ag}")
        print(f"DEBUG off_path: reduce_scatter={off_path_rs}, other={off_path_ag}")

        # On-path prefetch stats
        on_path_count = getattr(self, '_on_path_prefetch_count', 0)
        on_path_blocked = getattr(self, '_on_path_prefetch_blocked', 0)
        print(f"DEBUG on_path prefetch: attempts={on_path_count}, blocked={on_path_blocked}, allowed={on_path_count - on_path_blocked}")
        force_wait_count = getattr(self, '_force_wait_for_memory_count', 0)
        proactive_force_count = getattr(self, '_proactive_force_count', 0)
        swap_for_wait_count = getattr(self, '_swap_for_wait_count', 0)
        forced_user_count = getattr(self, '_forced_user_schedule_count', 0)
        user_skip = getattr(self, '_user_skip_reasons', {})
        print(f"DEBUG force waits for memory: {force_wait_count}, proactive: {proactive_force_count}, swap: {swap_for_wait_count}, forced_users: {forced_user_count}")
        print(f"DEBUG user skip reasons: {user_skip}")
        wait_score_count = getattr(self, '_wait_score_count', 0)
        wait_score_under_pressure = getattr(self, '_wait_score_under_pressure', 0)
        wait_score_risk_budget = getattr(self, '_wait_score_risk_budget', 0)
        print(f"DEBUG wait scoring: total={wait_score_count}, under_pressure={wait_score_under_pressure}, risk_budget={wait_score_risk_budget}")
        total_wait_freeing = sum(self.wait_freeing_potential.values())
        print(f"DEBUG wait freeing potential: total={total_wait_freeing/1024**3:.2f} GB, avg={total_wait_freeing/len(self.wait_freeing_potential)/1024**3:.4f} GB per wait")
        peak_node = getattr(self, '_peak_memory_node', 'unknown')
        peak_idx = getattr(self, '_peak_memory_index', -1)
        peak_bytes = getattr(self, '_peak_memory_bytes', 0)
        print(f"DEBUG peak memory at: node={peak_node}, scheduled_index={peak_idx}/{len(self.scheduled)}, bytes={peak_bytes/1024**3:.2f} GB")

        # Path memory stats
        path_increases = getattr(self, '_path_mem_increases', [])
        if path_increases:
            total_path_increase = sum(max(0, x) for x in path_increases)
            max_path_increase = max(path_increases) if path_increases else 0
            positive_paths = sum(1 for x in path_increases if x > 0)
            print(f"DEBUG path mem: num_paths={len(path_increases)}, positive_increase_paths={positive_paths}, total_increase={total_path_increase/1024**3:.2f} GB, max_single={max_path_increase/1024**3:.2f} GB")
        path_blocked = getattr(self, '_path_memory_blocked', 0)
        print(f"DEBUG prefetches blocked by path memory: {path_blocked}")

        # Estimated vs actual path memory
        estimate_vs_actual = getattr(self, '_path_estimate_vs_actual', [])
        if estimate_vs_actual:
            total_estimated = sum(e for e, a in estimate_vs_actual)
            total_actual = sum(a for e, a in estimate_vs_actual)
            underestimates = sum(1 for e, a in estimate_vs_actual if e < a)
            print(f"DEBUG path estimate vs actual: paths={len(estimate_vs_actual)}, estimated={total_estimated/1024**3:.2f} GB, actual={total_actual/1024**3:.2f} GB, underestimates={underestimates}")

        # Nested prefetch count
        nested_count = getattr(self, '_nested_prefetch_count', 0)
        print(f"DEBUG nested prefetches (from paths with waits): {nested_count}")

        # Memory freed by wait users
        wait_user_freed = getattr(self, '_wait_user_mem_freed', 0)
        wait_user_count = getattr(self, '_wait_user_count', 0)
        print(f"DEBUG nodes after waits that free memory: count={wait_user_count}, total_freed={wait_user_freed/1024**3:.2f} GB")

        # Original vs rescheduled peak location
        orig_peak_idx = getattr(self, '_original_peak_index', -1)
        orig_peak_node = getattr(self, '_original_peak_node', 'unknown')
        resched_peak_idx = getattr(self, '_peak_memory_index', -1)
        resched_peak_node = getattr(self, '_peak_memory_node', 'unknown')
        print(f"DEBUG original peak: index={orig_peak_idx}/{len(self.nodes)}, node={orig_peak_node}")
        print(f"DEBUG rescheduled peak: index={resched_peak_idx}/{len(self.scheduled)}, node={resched_peak_node}")

        log.info(
            "Overlap scheduling results: exposed=%d, bad_exposed=%d, potentially_hidden=%d, "
            "original_peak_memory=%d bytes, rescheduled_peak_memory=%d bytes, "
            "total_exposed_ms=%.2f, hideable_exposed_ms=%.2f, total_potential_exposed_ms=%.2f, "
            "wasted_compute_ms=%.2f",
            len(exposed),
            len(bad_exposed),
            len(potentially_hidden_collectives),
            self.original_peak_memory,
            self.memory_tracker.peak_memory,
            total_exposed,
            hideable_exposed_ms,
            total_potential_exposed,
            self.wasted_compute,
        )

        self.reorder_graph()

    def _bucket_collectives(self) -> None:
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            scheduled=self.scheduled,
            max_bucket_memory_gb=2.0,  # Could make this configurable
            max_coll_distance=self.max_node_distance,
            insert_overlap_deps=self.insert_overlap_deps,
        )
        bucketer.bucket_collectives()

    def compute_potential_hidden_nodes(
        self, nodes_to_check: Iterable[fx.Node]
    ) -> dict[fx.Node, fx.Node]:
        """
        Returns a dict containing a mapping of nodes which could potentially be hidden to their hiding node
        """

        def could_be_hidden(start: fx.Node) -> fx.Node | None:
            for compute_node in self.compute_nodes:
                if (
                    start not in self.node_ancestors[compute_node]
                    and compute_node not in self.node_ancestors[start]
                ):
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

    def compute_potential_hidden_collectives(self) -> dict[fx.Node, fx.Node]:
        """Compute which collective operations could be hidden by compute."""
        return self.compute_potential_hidden_nodes(self.collective_info.keys())

    def compute_potential_hidden_waits(self) -> dict[fx.Node, fx.Node]:
        """Compute which wait operations could be hidden by compte."""
        wait_nodes = [info.wait_node for info in self.collective_info.values()]
        return self.compute_potential_hidden_nodes(wait_nodes)


def schedule_overlap_bucketing(
    gm: torch.fx.GraphModule,
    max_in_flight_gb: float = 5,
    max_compute_pre_fetch: int = 200,
    collective_bucketing: bool = False,
    insert_overlap_deps: bool = False,
    compute_overlap_multipler: float = 1.0,
    max_coll_distance: int = 200,
    custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
    | None = None,
    collective_estimator: Literal["analytical", "benchmark"] = "analytical",
    max_memory_increase_gb: float | None = 1.0,
    max_memory_increase_ratio: float | None = 0.05,
) -> torch.fx.GraphModule:
    """Schedule nodes to maximize compute-collective overlap.

    Args:
        gm: Input graph module to optimize.
        max_in_flight_gb: Maximum GB of concurrent collective data. Too much in flight memory
            can cause memory fragmentation within the CUDA Caching Allocator.
        max_compute_pre_fetch: Maximum mm nodes to pre fetch. Note: should already be limited by max_in_flight_gb and
            max_memory_increase_gb
        collective_bucketing: Enable overlap-preserving collective bucketing.
        insert_overlap_deps: Insert overlap dependencies using control deps operator. This should only be used if
            compiling with inductor, or for subsequent passes before removing the ops prior to execution.
        compute_overlap_multipler: Scale factor for compute time used to hide collectives. This can be used
            to address over or under aggressive overlapping.
        max_coll_distance: Maximum pre fetch or bucketing candidates. Mainly intended for compile time
        custom_runtime_estimation: Custom runtime estimation function that estimates runtime in ms for an fx node.
            If None, uses default estimations. This is currently limited to collectives and compute nodes.
        collective_estimator: Method for estimating collective runtime. "analytical" uses bandwidth formulas,
            "benchmark" uses CUDA events with power-of-2 rounding and interpolation.
        max_memory_increase_gb: Maximum GB increase above baseline memory (absolute cap). If None, no absolute limit.
        max_memory_increase_ratio: Maximum increase as ratio of baseline peak memory. If None, no ratio limit.
            Uses minimum of absolute and ratio limits when both are specified.
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
        max_memory_increase_gb=max_memory_increase_gb,
        max_memory_increase_ratio=max_memory_increase_ratio,
    ).run()
