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
from torch._logging import trace_structured
from torch.fx.operator_schemas import normalize_function
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import _disable_current_modes


log = logging.getLogger(__name__)

from torch._inductor.fx_passes.bucketing import bucket_key

from ..pattern_matcher import stable_topological_sort


def estimate_runtime_analytical(n: torch.fx.Node) -> float:
    """Estimate runtime using analytical roofline model for mm operations."""
    if n.target != torch.ops.aten.mm.default:
        return 0.0
    import torch.utils._pytree as pytree
    from torch.distributed._tools import RuntimeEstimator

    def _val(node: Any) -> Any:
        if not isinstance(node, torch.fx.Node):
            return node
        return node.meta["val"]

    args = pytree.tree_map(_val, n.args)
    kwargs = pytree.tree_map(_val, n.kwargs)
    _, ms = RuntimeEstimator._roofline_estimate(n.target, args, kwargs)
    return ms


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


def estimate_mem_bound_runtime_ms(node: fx.Node) -> float:
    """Estimate runtime for a memory-bound node based on input/output bytes.

    Returns 0 for view nodes (no memory cost).
    """
    from torch._inductor.fx_passes.fusion_regions import is_view_node
    from torch.utils._pytree import tree_flatten
    from torch.utils._runtime_estimation import get_transfer_time

    if is_view_node(node):
        return 0.0

    input_vals = [inp.meta.get("val") for inp in node.all_input_nodes]
    output_vals = [node.meta.get("val")]
    flat_inputs, _ = tree_flatten(input_vals)
    flat_outputs, _ = tree_flatten(output_vals)

    transfer_time_ns = get_transfer_time(flat_inputs, flat_outputs)
    return transfer_time_ns / 1e6


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
        log_final_collectives_estimations: bool = False,
        bucket_exposed_first: bool = True,
        enable_fusion_regions: bool = False,
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
        self.log_final_collectives_estimations = log_final_collectives_estimations
        self.bucket_exposed_first = bucket_exposed_first

        # Build and collapse fusion regions FIRST so all subsequent operations
        # work on the collapsed graph where fused ops are atomic units
        self.region_of: dict[fx.Node, Any] = {}
        if enable_fusion_regions:
            from torch._inductor.fx_passes.fusion_regions import (
                build_fusion_regions,
                collapse_fusion_regions,
            )

            self.region_of = build_fusion_regions(self.gm)
            if self.region_of:
                self.region_of = collapse_fusion_regions(self.gm, self.region_of)
                # fuse_by_partitions replaces gm.graph, so we need to update our reference
                self.graph = gm.graph

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

        # Calculate domination indices for both compute and reduce_scatter nodes
        self.reduce_scatter_nodes = self.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.reduce_scatter_tensor.default,
        )
        self.compute_index_domination = self._calculate_domination_index(
            self.compute_nodes
        )
        self.reduce_scatter_domination = self._calculate_domination_index(
            self.reduce_scatter_nodes
        )

        # Scheduling state
        self.potentially_hidden_collectives = (
            self.compute_potential_hidden_collectives()
        )
        self.potentially_hidden_waits = self.compute_potential_hidden_waits()
        self.in_degree = Counter(user for node in self.nodes for user in node.users)

        # Two separate queues: on-path (domination-based) and off-path (node_idx-based)
        self.on_path_ready: list[tuple[object, fx.Node]] = []
        self.off_path_ready: list[tuple[object, fx.Node]] = []

        for node in self.nodes:
            if self.in_degree[node] == 0:
                self._add_to_ready_queue(node)

        self.in_flight: dict[fx.Node, CollectiveInfo] = {}  # start -> info
        self.in_flight_bytes = 0
        self.scheduled: OrderedSet[fx.Node] = OrderedSet()
        self.max_compute_pre_fetch = max_compute_pre_fetch

        self.last_on_path_node_idx = -1

    def _add_to_ready_queue(self, node: fx.Node) -> None:
        if self.off_compute_path(node):
            score = self._compute_off_path_score(node)
            heapq.heappush(self.off_path_ready, (score, node))
        else:
            score = self._compute_on_path_score(node)
            heapq.heappush(self.on_path_ready, (score, node))

    def _collect_node_ancestors(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """Collect all ancestors for each node."""
        ancestors: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in self.nodes:
            for input_node in node.all_input_nodes:
                ancestors[node].add(input_node)
                ancestors[node] |= ancestors[input_node]

        return ancestors

    def _compute_baseline_memory(self) -> int:
        """
        Simulate the original schedule to compute baseline memory profile.
        Returns the peak memory observed during simulation.
        """
        baseline_tracker = MemoryTracker(self.graph)

        last_compute_max_memory = 0
        peak_memory = 0

        for node in self.nodes:
            baseline_tracker.schedule_node(node)
            current_mem = baseline_tracker.current_memory_bytes

            # Record the max memory between this and previous compute node
            last_compute_max_memory = max(last_compute_max_memory, current_mem)

            if is_compute_node(node):
                self.original_mem_before_compute_index.append(last_compute_max_memory)
                last_compute_max_memory = current_mem

            peak_memory = max(peak_memory, current_mem)

        return peak_memory

    def _prefetch_would_exceed_memory_budget(self, start_node: fx.Node) -> bool:
        """
        Check if prefetching this collective would exceed memory budget at ANY compute node
        between now and when it's used.
        """
        info = self.collective_info[start_node]
        size = info.size_bytes

        domination_index = self.compute_index_domination[start_node]

        # If off-path, assume it doesn't increase memory
        if domination_index == sys.maxsize:
            return False

        # check current mem
        if (
            self.memory_tracker.current_memory_bytes + size
            > self.allowed_peak_memory_bytes
        ):
            return True

        start_index = self.current_compute_index

        # then, check future mem
        for compute_idx in range(start_index, domination_index):
            cumulative_prefetch = self.cumulative_prefetch_mem_by_compute_index[
                compute_idx
            ]

            # Check 1: Would cumulative prefetch exceed in-flight limit?
            if (cumulative_prefetch + size) > self.max_in_flight_bytes:
                return True

            # Check 2: Would total memory (baseline + cumulative prefetch) exceed budget?
            baseline_mem = self.original_mem_before_compute_index[compute_idx]
            projected = baseline_mem + cumulative_prefetch + size

            if projected > self.allowed_peak_memory_bytes:
                return True

        return False

    def _update_cumulative_prefetch_memory(
        self, collective: fx.Node, info: CollectiveInfo
    ) -> None:
        """
        Update cumulative prefetch memory for all compute indices this collective will be live.
        """
        domination_index = self.compute_index_domination[collective]
        if domination_index == sys.maxsize:
            return

        for compute_idx in range(self.current_compute_index, domination_index):
            self.cumulative_prefetch_mem_by_compute_index[compute_idx] += (
                info.size_bytes
            )

    def off_compute_path(self, n: fx.Node) -> bool:
        """Check if a node is off the compute path (doesn't block any compute)."""
        return self.compute_index_domination[n] == sys.maxsize

    def dominates_reduce_scatter(self, n: fx.Node) -> bool:
        """Check if a node dominates (blocks) any reduce_scatter."""
        return self.reduce_scatter_domination[n] != sys.maxsize

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

    def _calculate_domination_index(
        self, target_nodes: list[fx.Node]
    ) -> dict[fx.Node, int]:
        """
        Calculate the topological index of the earliest target node each node dominates.

        target_nodes are assigned indices based on their topological order (0, 1, 2, ...).
        For each node, returns the minimum index of target nodes it blocks/dominates.
        Returns sys.maxsize if the node doesn't block any target nodes.
        """
        target_node_index: dict[fx.Node, int] = {}
        for node in self.graph.nodes:
            if node in target_nodes:
                target_node_index[node] = len(target_node_index)

        domination_index: dict[fx.Node, int] = {}
        for node in reversed(self.graph.nodes):
            if node in target_node_index:
                domination_index[node] = target_node_index[node]
            else:
                domination_index[node] = min(
                    (domination_index[succ] for succ in node.users), default=sys.maxsize
                )

        return domination_index

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

        # Also collect analytical estimations for logging
        runtime_estimations_analytical: list[float] = []

        for n in self.compute_nodes:
            val, key = benchmark_node_with_cache_key(n, self.custom_runtime_estimation)

            # Analytical estimations
            val_analytical = estimate_runtime_analytical(n)
            runtime_estimations_analytical.append(val_analytical)

            runtime_estimations.append(val)
            runtime_estimations_keys.append(key)
            compute_key_count += 1

        # Log compute estimations
        from torch._inductor.fx_passes.node_runtime_estimation import (
            _log_compute_estimations,
        )

        _log_compute_estimations(
            self.compute_nodes,
            runtime_estimations,
            runtime_estimations_analytical,
        )

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
                cuda_val, cuda_key = benchmark_collective_with_cuda_events(n, nruns=5)
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
            from torch._inductor.fx_passes.node_runtime_estimation import (
                _log_collective_benchmarks,
            )

            _log_collective_benchmarks(
                benchmarked_collective_nodes,
                collective_keys,
                collective_medians,
                world_size,
                "fx_collectives_node_runtime_estimation",
            )
        else:
            # No benchmarking - log analytical estimations for all collectives
            from torch._inductor.fx_passes.node_runtime_estimation import (
                _log_collective_benchmarks,
            )

            all_collective_nodes = [
                info.start_node for info in self.collective_info.values()
            ]
            if all_collective_nodes:
                _log_collective_benchmarks(
                    all_collective_nodes,
                    artifact_name="fx_collectives_analytical_estimation",
                )

        log.info("Overlap scheduling: Runtime estimations aligned")

    def _get_next_node(self) -> fx.Node:
        """Get next node: off-path nodes scheduled near original position, exposed waits deferred."""
        if self.off_path_ready:
            _, node = self.off_path_ready[0]

            should_schedule = False
            if not self.on_path_ready or node in self.scheduled:
                should_schedule = True
            elif _schedulable_wait_node(node):
                # Defer exposed waits until hidden or over memory budget
                info = self.collective_info[self.wait_to_start[node]]
                over_budget = (
                    self.memory_tracker.current_memory_bytes
                    > self.allowed_peak_memory_bytes
                )
                should_schedule = not info.is_exposed or over_budget
            elif self.dominates_reduce_scatter(node):
                # Only schedule off-path nodes that dominate reduce_scatters after original position
                should_schedule = self.node_idx[node] <= self.last_on_path_node_idx

            if should_schedule:
                heapq.heappop(self.off_path_ready)
                return node

        return heapq.heappop(self.on_path_ready)[1]

    def run(self) -> torch.fx.GraphModule:
        """Run the scheduling algorithm."""
        # All ranks must make identical decisions on overlap reordering,
        # Thus we must have identical runtime estimations across ranks.
        # For now we do benchmarking only for compute nodes.
        self._align_compute_nodes_runtime_estimations_across_all_distributed_ranks()

        while self.on_path_ready or self.off_path_ready:
            if self._should_force_wait_for_memory():
                self._force_oldest_wait()
                continue

            node = self._get_next_node()

            # we don't always remove nodes from the heap when we schedule them
            if node in self.scheduled:
                continue

            if node.op == "placeholder":
                self._schedule(node)
            elif node in self.collective_info:
                self._handle_collective_start(node)
            elif _schedulable_wait_node(node):
                self._handle_wait(node)
            else:
                self._handle_compute_or_other(node)

            # Track progress for off-path scheduling - only for nodes from main queue
            if not self.off_compute_path(node):
                self.last_on_path_node_idx = max(
                    self.last_on_path_node_idx, self.node_idx[node]
                )

        self._reorder_graph()

        # Finalize: bucket collectives (if enabled), inline fusions, apply deps
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            finalize_overlap_scheduling,
        )

        finalize_overlap_scheduling(
            gm=self.gm,
            collective_info=self.collective_info,
            scheduled=self.scheduled,
            collective_bucketing=self.collective_bucketing,
            insert_overlap_deps=self.insert_overlap_deps,
            max_bucket_memory_gb=2.0,
            max_coll_distance=self.max_node_distance,
            region_of=self.region_of,
            bucket_exposed_first=self.bucket_exposed_first,
        )

        if self.log_final_collectives_estimations:
            from torch._inductor.fx_passes.node_runtime_estimation import (
                _log_graph_collective_benchmarks,
            )

            _log_graph_collective_benchmarks(
                self.gm, "fx_collectives_estimations_after_overlap_bucketing"
            )

        return self.gm

    def get_non_collective_runtime_estimate(self, node: fx.Node) -> float | None:
        """Get runtime estimation for a node in ms. Returns None if no estimation is available."""
        if is_compute_node(node):
            return benchmark_node(node, self.custom_runtime_estimation)

        # Use precomputed cost for fusion region call_module nodes
        # This takes priority even over custom estimation since fusion regions
        # have already computed their cost based on their contents
        if node in self.region_of:
            return self.region_of[node].cost_ms

        if self.custom_runtime_estimation is not None:
            if (est := self.custom_runtime_estimation(node, None)) is not None:
                return est
            # Custom estimation provided but returned None - don't fall through to fusible estimation
            return None

        # assume any node without flop counter is mem bound
        if node.op == "call_function":
            return estimate_mem_bound_runtime_ms(node)

        return None

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
                self._add_to_ready_queue(user)

    def _compute_on_path_score(self, node: fx.Node) -> object:
        """Compute priority score for on-path nodes (domination-based)."""
        if _schedulable_wait_node(node):
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

    def _compute_off_path_score(self, node: fx.Node) -> object:
        """
        Off-path priority scoring.

        Nodes that dominate reduce_scatters are prioritized (lower score = higher priority)
        to ensure they get scheduled eagerly for potential overlap.
        """
        dominates_rs = 0 if self.dominates_reduce_scatter(node) else 1
        return (dominates_rs, self.node_idx[node])

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

        return self.in_flight_bytes >= self.max_in_flight_bytes

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

        def get_priority(n: fx.Node) -> int:
            dominates_next_compute = (
                self.compute_index_domination[n] == self.current_compute_index + 1
            )
            if dominates_next_compute:
                return 0  # Dominates next compute layer - most urgent
            elif self.off_compute_path(n) and self.dominates_reduce_scatter(n):
                return 1  # Off-path but blocks reduce_scatter
            elif not self.off_compute_path(n):
                return 2  # On-path but not immediate
            else:
                return 3  # Off-path, doesn't block reduce_scatter

        candidates.sort(
            key=lambda n: (
                get_priority(n),
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
            self._schedule_path_to_collective(path, overlap_node)
            self._handle_collective_start(collective)
            self._update_cumulative_prefetch_memory(collective, info)

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
                if (not info.is_exposed) and (
                    curr_overlap_node not in info.hiding_nodes
                ):
                    continue

                why(
                    "path blocked by wait node %s (exposed=%s, hidden_by_curr_overlap=%s)",
                    node.name,
                    info.is_exposed,
                    curr_overlap_node in info.hiding_nodes,
                )
                return None

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
            bucket_exposed_first=self.bucket_exposed_first,
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
    log_final_collectives_estimations: bool = False,
    bucket_exposed_first: bool = True,
    enable_fusion_regions: bool = False,
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
        enable_fusion_regions: Enable fusion region detection and cost estimation for fusible ops.
    """
    if not any(is_wait_tensor(n) for n in gm.graph.nodes):
        return gm

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "overlap_scheduling_graph_before",
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(False),
    )
    ret = OverlapScheduler(
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
        log_final_collectives_estimations=log_final_collectives_estimations,
        bucket_exposed_first=bucket_exposed_first,
        enable_fusion_regions=enable_fusion_regions,
    ).run()
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "overlap_scheduling_graph_after",
            "encoding": "string",
        },
        payload_fn=lambda: ret.print_readable(False),
    )
    return ret


def schedule_overlap_bucketing_from_inductor_configs(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Schedule nodes to maximize compute-collective overlap using inductor configs.

    Reads configuration from torch._inductor.config.aten_distributed_optimizations
    and calls schedule_overlap_bucketing with those settings.
    """
    if not any(is_wait_tensor(n) for n in gm.graph.nodes):
        return gm

    from torch._inductor import config

    dist_opts = config.aten_distributed_optimizations

    kwargs: dict[str, object] = {}

    config_keys = (
        "collective_bucketing",
        "max_compute_pre_fetch",
        "custom_runtime_estimation",
        "insert_overlap_deps",
        "collective_estimator",
        "max_memory_increase_gb",
        "max_memory_increase_ratio",
        "compute_overlap_multipler",
        "max_in_flight_gb",
        "max_coll_distance",
        "log_final_collectives_estimations",
        "bucket_exposed_first",
        "enable_fusion_regions",
    )
    for key in config_keys:
        if (val := getattr(dist_opts, key, None)) is not None:
            kwargs[key] = val

    return schedule_overlap_bucketing(gm, **kwargs)  # type: ignore[arg-type]
