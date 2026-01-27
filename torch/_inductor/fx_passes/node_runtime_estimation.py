"""
Collective runtime estimation using CUDA events and power-of-2 rounding.

This module provides:
1. Collective benchmarking with CUDA events
2. Caching of benchmark results
3. Logging of benchmarks to tlparse artifacts
4. Loading of benchmarks from files as estimators with linear interpolation
"""

from __future__ import annotations

import atexit
import functools
import itertools
import json
import logging
import operator
import os
from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache
from typing import Any, Callable, Optional

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import _schedulable_wait_node
from torch._inductor.utils import clear_on_fresh_cache, tabulate_2d
from torch._logging import getArtifactLogger, trace_structured
from torch.fx.operator_schemas import normalize_function


# Setup logger for artifact logging
log = getArtifactLogger(__name__, "node_runtime_estimation")
_module_log = logging.getLogger(__name__)


# TODO: Consider using a distributed-aware cache or rank-local disk cache
# not using local cache because different ranks might write to it concurrently.
# solvable in future, potentially with workflow to seed cache
@clear_on_fresh_cache
@lru_cache
def _get_collective_cache() -> dict[str, float]:
    """Get process-local cache for collective benchmarks."""
    return {}


def get_cached_runtime(key: str) -> Optional[float]:
    """Get cached runtime from process-local cache."""
    return _get_collective_cache().get(key)


def set_cached_runtime(key: str, value: float) -> None:
    """Set cached runtime in process-local cache."""
    _get_collective_cache()[key] = value


def get_hint(x: int | torch.SymInt) -> Optional[int]:
    if isinstance(x, int):
        return x
    assert isinstance(x, torch.SymInt)
    return x.node.hint if x.node.has_hint() else None


def can_benchmark_collective() -> bool:
    """Check if we can benchmark collectives (not fake process group)."""
    import torch.distributed as c10d

    if not c10d.is_initialized():
        return False

    pg = c10d.distributed_c10d._get_default_group()
    if torch.distributed.distributed_c10d.get_backend(pg) == "fake":
        return False

    return True


def _median(lst):
    assert len(lst) > 0
    return torch.median(torch.tensor(lst)).item()


def _benchmark_collective_with_cuda_events_impl(
    n: torch.fx.Node,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    nruns: int,
) -> float | None:
    """
    Core benchmarking logic using CUDA events and barriers.
    Returns runtime in ms or None on failure.
    """
    from torch._dynamo.testing import rand_strided

    # Convert FakeTensors to real tensors before benchmarking
    def to_real(t: torch.Tensor) -> torch.Tensor:
        shape = [get_hint(dim) for dim in t.shape]
        stride = [get_hint(s) for s in t.stride()]

        if any(s is None for s in itertools.chain(shape, stride)):
            # This should not happen, as can_benhcmark_collective checks for unbacked
            raise ValueError("Cannot convert tensor with symbolic dimensions")

        return rand_strided(shape, stride, device=t.device, dtype=t.dtype)  # type: ignore[arg-type]

    args, kwargs = torch.utils._pytree.tree_map_only(
        torch.Tensor,
        to_real,
        (args, kwargs),
    )

    # Warmup: call collective once and wait
    torch.cuda.synchronize()
    result = n.target(*args, **kwargs)  # type: ignore[operator]
    torch.ops._c10d_functional.wait_tensor(result)
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    comm_times = []
    for _ in range(nruns):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        result = n.target(*args, **kwargs)  # type: ignore[operator]
        torch.ops._c10d_functional.wait_tensor(result)
        end_evt.record()
        end_evt.synchronize()

        comm_times.append(start_evt.elapsed_time(end_evt))

    return _median(comm_times)


def benchmark_collective_with_cuda_events(
    n: torch.fx.Node,
    nruns: int = 2,
) -> tuple[float | None, str]:
    """
    Benchmark collective with CUDA events. Returns (runtime_ms, cache_key) or (None, "") on failure.
    """
    # context manager not allowed with profiler.
    with torch.utils._python_dispatch._disable_current_modes():
        return benchmark_collective_with_cuda_events_impl(n, nruns)


def benchmark_collective_with_cuda_events_impl(
    n: torch.fx.Node,
    nruns: int = 3,
) -> tuple[float | None, str]:
    """
    Benchmark collective with CUDA events. Returns (runtime_ms, cache_key) or (None, "") on failure.
    """
    from torch._inductor import fx_utils
    from torch.distributed.distributed_c10d import _get_group_size_by_name

    # Early check: can we actually run collectives?
    if not can_benchmark_collective():
        return None, ""

    success, args, kwargs = fx_utils.get_fake_args_kwargs(n)

    opt_args_kwargs = normalize_function(
        n.target,  # type: ignore[arg-type]
        args=n.args,
        kwargs=n.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    assert opt_args_kwargs is not None
    group_name = opt_args_kwargs[1]["group_name"]
    group_size = _get_group_size_by_name(group_name)

    if not success:
        return None, ""

    # Extract actual input size in BYTES (first tensor argument)
    actual_bytes: Optional[int] = None

    def extract_tensor_info(t: torch.Tensor) -> torch.Tensor:
        nonlocal actual_bytes
        if actual_bytes is None:
            shape = [get_hint(dim) for dim in t.shape]
            if any(s is None for s in shape):
                return t

            total_elems = 1
            for dim in shape:
                assert dim is not None
                total_elems *= dim

            actual_bytes = total_elems * t.dtype.itemsize
        else:
            raise RuntimeError(f"should only be one input tensor to collective {n}")
        return t

    torch.utils._pytree.tree_map_only(torch.Tensor, extract_tensor_info, (args, kwargs))

    if actual_bytes is None:
        return None, ""

    # Cache key by BYTES (dtype-agnostic)
    key = f"{n.target}: ({group_size} group size, {actual_bytes} bytes)"

    # Check cache
    if (cached := get_cached_runtime(key)) is not None:
        return cached, key

    # Benchmark using CUDA events with actual args/kwargs
    runtime = _benchmark_collective_with_cuda_events_impl(n, args, kwargs, nruns)

    if runtime is None:
        return None, key

    # Cache the result
    set_cached_runtime(key, runtime)

    # Also record in global benchmark storage for logging
    record_benchmark_result(
        key=key,
        runtime_ms=runtime,
        category="collectives",
        bytes_count=actual_bytes,
        group_size=group_size,
    )

    return runtime, key


def _log_compute_estimations(
    compute_nodes: list[fx.Node],
    benchmarked_estimations: list[float],
    analytical_estimations: list[float],
) -> None:
    """Log compute node runtime estimations comparing benchmarked vs analytical."""
    import torch.utils._pytree as pytree
    from torch._inductor.fx_utils import count_flops_fx
    from torch.utils._dtype_abbrs import dtype_abbrs

    def _node_summary(n: fx.Node) -> str:
        ret = str(n)
        for arg in pytree.arg_tree_leaves(n.args, n.kwargs):
            if not isinstance(arg, torch.fx.Node):
                continue
            if "val" in arg.meta:
                t = arg.meta["val"]
                ret += f" {dtype_abbrs[t.dtype]}{tuple(t.shape)}"
        return ret

    headers = [
        "Node",
        "Benchmarked Est(us)",
        "Analytical Est(us)",
        "Diff(ratio)",
        "Diff(us)",
        "Flops",
    ]

    rows = [
        [
            _node_summary(node),
            est_b * 1e3,
            est_a * 1e3,
            (est_a / est_b) if est_b > 0 else 0,
            (est_a - est_b) * 1e3,
            count_flops_fx(node),
        ]
        for node, est_b, est_a in zip(
            compute_nodes, benchmarked_estimations, analytical_estimations
        )
    ]

    log_str = tabulate_2d(rows, headers)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "fx_compute_nodes_runtime_estimation",
            "encoding": "string",
        },
        payload_fn=lambda: log_str,
    )


def _log_graph_collective_benchmarks(gm: fx.GraphModule, artifact_name: str) -> None:
    from torch._inductor import fx_utils

    collective_nodes = []
    collective_keys = []
    benchmarked = []

    for node in gm.graph.nodes:
        if _schedulable_wait_node(node):
            start = node.args[0]
            collective_nodes.append(start)
            opt_args_kwargs = normalize_function(
                start.target,  # type: ignore[arg-type]
                args=start.args,
                kwargs=start.kwargs,
                normalize_to_only_use_kwargs=True,
            )
            assert opt_args_kwargs is not None
            _, kwargs = opt_args_kwargs
            group_name = kwargs.get("group_name", None)
            group_size = kwargs.get("group_size", None)

            # Extract first tensor input size in bytes
            tensor_bytes: Optional[int] = None
            success, args, kw = fx_utils.get_fake_args_kwargs(start)
            if success:

                def extract_first_tensor_bytes(t: torch.Tensor) -> torch.Tensor:
                    nonlocal tensor_bytes
                    if tensor_bytes is None:
                        shape = [get_hint(dim) for dim in t.shape]
                        if all(s is not None for s in shape):
                            numel = functools.reduce(operator.mul, shape, 1)
                            tensor_bytes = numel * t.dtype.itemsize
                    return t

                torch.utils._pytree.tree_map_only(
                    torch.Tensor, extract_first_tensor_bytes, (args, kw)
                )

            collective_keys.append(
                f"{start.target} group_size:{group_size} group_name:{group_name} input_bytes:{tensor_bytes}"
            )
            benchmarked_ms, _ = benchmark_collective_with_cuda_events(start, nruns=5)
            benchmarked.append(benchmarked_ms if benchmarked_ms else 0.0)

    if collective_nodes:
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        _log_collective_benchmarks(
            collective_nodes,
            collective_keys,
            benchmarked,
            world_size,
            artifact_name,
        )


def _log_collective_benchmarks(
    collective_nodes: list[fx.Node],
    collective_keys: list[str],
    benchmarked_medians: list[float],
    world_size: int,
    artifact_name: str,
) -> None:
    """Log collective benchmarks with analytical comparisons for tlparse."""
    headers = [
        "Collective Key",
        "Benchmarked(ms)",
        "NCCL Est(ms)",
        "Inductor Est(ms)",
        "NCCL Diff(ratio)",
        "Inductor Diff(ratio)",
    ]

    rows = []
    collective_benchmarks = {}
    for key, benchmarked_ms, coll_node in zip(
        collective_keys, benchmarked_medians, collective_nodes
    ):
        # NCCL estimator (deterministic, no need to align)
        nccl_ms = (
            torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
                coll_node, None, use_nccl_estimator=True
            )
        )

        # Inductor analytical (deterministic, no need to align)
        inductor_ms = (
            torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
                coll_node, None, use_nccl_estimator=False
            )
        )

        collective_benchmarks[key] = {
            "benchmarked_ms": benchmarked_ms,
            "analytical_nccl_ms": nccl_ms,
            "analytical_inductor_ms": inductor_ms,
        }

        # Compute percentage differences
        nccl_diff_pct = (nccl_ms / benchmarked_ms) if benchmarked_ms > 0 else 0
        inductor_diff_pct = (inductor_ms / benchmarked_ms) if benchmarked_ms > 0 else 0

        rows.append(
            [
                key,
                f"{benchmarked_ms:.4f}",
                f"{nccl_ms:.4f}",
                f"{inductor_ms:.4f}",
                f"{nccl_diff_pct:.2f}",
                f"{inductor_diff_pct:.2f}",
            ]
        )

    log_str = f"World size: {world_size}\n"
    log_str += tabulate_2d(rows, headers)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": artifact_name,
            "encoding": "string",
        },
        payload_fn=lambda: log_str,
    )


# ==============================================================================
# Goal 2: Benchmark logging and estimator loading
# ==============================================================================

# Global storage for all benchmarked results
_BENCHMARK_RESULTS: dict[str, dict[str, Any]] = {
    "compute": {},  # key -> {bytes: int, runtime_ms: float, flops: int}
    "collectives": {},  # key -> {bytes: int, runtime_ms: float, group_size: int}
}

_LOG_ON_EXIT_ENABLED: bool = False


def _parse_benchmark_key(key: str) -> dict[str, Any]:
    """
    Parse a benchmark key into its components.
    Example: "torch.ops.aten.mm.default: T: ([1024, 1024], [1024, 1024], torch.float32)"
    """
    parts = key.split(": ", 1)
    result = {"target": parts[0], "raw_key": key}

    if len(parts) > 1:
        # Try to extract bytes if present
        if "bytes" in parts[1]:
            import re

            match = re.search(r"(\d+)\s*bytes", parts[1])
            if match:
                result["bytes"] = int(match.group(1))
        # Try to extract group size if present
        if "group size" in parts[1]:
            import re

            match = re.search(r"(\d+)\s*group size", parts[1])
            if match:
                result["group_size"] = int(match.group(1))

    return result


def record_benchmark_result(
    key: str,
    runtime_ms: float,
    category: str = "compute",
    bytes_count: Optional[int] = None,
    flops: Optional[int] = None,
    group_size: Optional[int] = None,
) -> None:
    """
    Record a benchmark result for later logging.

    Args:
        key: The unique key for this benchmark (e.g., from benchmark_node_with_cache_key)
        runtime_ms: The benchmarked runtime in milliseconds
        category: "compute" or "collectives"
        bytes_count: Optional byte count for interpolation
        flops: Optional flop count for compute nodes
        group_size: Optional group size for collectives
    """
    result: dict[str, Any] = {
        "runtime_ms": runtime_ms,
    }
    if bytes_count is not None:
        result["bytes"] = bytes_count
    if flops is not None:
        result["flops"] = flops
    if group_size is not None:
        result["group_size"] = group_size

    _BENCHMARK_RESULTS[category][key] = result


def get_all_benchmark_results() -> dict[str, dict[str, Any]]:
    """Get all recorded benchmark results."""
    return _BENCHMARK_RESULTS


def log_all_benchmarks_to_tlparse() -> None:
    """
    Log all collected benchmarks to tlparse as a JSON artifact.
    This can be called manually or registered to run at program exit.
    """
    # Combine compute and collective caches
    all_benchmarks = {
        "compute": dict(_BENCHMARK_RESULTS["compute"]),
        "collectives": dict(_BENCHMARK_RESULTS["collectives"]),
    }

    # Also include any results from the existing caches
    compute_cache = get_benchmark_cache()._cache if hasattr(get_benchmark_cache(), "_cache") else {}
    collective_cache = _get_collective_cache()

    for key, value in collective_cache.items():
        if key not in all_benchmarks["collectives"]:
            parsed = _parse_benchmark_key(key)
            all_benchmarks["collectives"][key] = {
                "runtime_ms": value,
                **{k: v for k, v in parsed.items() if k != "raw_key"},
            }

    if not all_benchmarks["compute"] and not all_benchmarks["collectives"]:
        _module_log.debug("No benchmarks to log")
        return

    _module_log.info(
        "Logging %d compute and %d collective benchmarks to tlparse",
        len(all_benchmarks["compute"]),
        len(all_benchmarks["collectives"]),
    )

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "all_benchmarked_runtimes",
            "encoding": "json",
        },
        payload_fn=lambda: all_benchmarks,
    )


def enable_benchmark_logging_on_exit() -> None:
    """
    Enable automatic logging of all benchmarks to tlparse on program exit.
    """
    global _LOG_ON_EXIT_ENABLED
    if not _LOG_ON_EXIT_ENABLED:
        atexit.register(log_all_benchmarks_to_tlparse)
        _LOG_ON_EXIT_ENABLED = True
        _module_log.info("Registered benchmark logging on program exit")


def save_benchmarks_to_file(filepath: str) -> None:
    """
    Save all benchmarks to a JSON file that can be loaded later as an estimator.

    Args:
        filepath: Path to save the JSON file
    """
    all_benchmarks = get_all_benchmark_results()

    # Also include any results from existing caches
    collective_cache = _get_collective_cache()
    for key, value in collective_cache.items():
        if key not in all_benchmarks["collectives"]:
            parsed = _parse_benchmark_key(key)
            all_benchmarks["collectives"][key] = {
                "runtime_ms": value,
                **{k: v for k, v in parsed.items() if k != "raw_key"},
            }

    with open(filepath, "w") as f:
        json.dump(all_benchmarks, f, indent=2)

    _module_log.info("Saved benchmarks to %s", filepath)


class BenchmarkEstimator:
    """
    An estimator that loads benchmarks from a file and provides runtime estimates
    with linear interpolation for missing entries.

    Usage:
        estimator = BenchmarkEstimator.from_file("benchmarks.json")

        # Use as a custom_runtime_estimation function
        schedule_overlap_bucketing(
            gm,
            custom_runtime_estimation=estimator.estimate_runtime,
        )
    """

    def __init__(self, benchmarks: dict[str, dict[str, Any]]) -> None:
        """
        Initialize the estimator with benchmark data.

        Args:
            benchmarks: Dictionary with "compute" and "collectives" keys,
                       each containing key -> {runtime_ms, bytes, ...} mappings.
        """
        self.benchmarks = benchmarks
        self._compute_by_target: dict[str, list[tuple[int, float]]] = {}
        self._collectives_by_target: dict[str, dict[int, list[tuple[int, float]]]] = {}

        # Index benchmarks by target operation and bytes for interpolation
        self._build_indices()

    def _build_indices(self) -> None:
        """Build indices for fast lookup and interpolation."""
        # Index compute benchmarks by target
        for key, data in self.benchmarks.get("compute", {}).items():
            parsed = _parse_benchmark_key(key)
            target = parsed.get("target", key)
            bytes_count = data.get("bytes", parsed.get("bytes", 0))
            runtime_ms = data.get("runtime_ms", 0)

            if target not in self._compute_by_target:
                self._compute_by_target[target] = []
            self._compute_by_target[target].append((bytes_count, runtime_ms))

        # Sort by bytes for binary search
        for target in self._compute_by_target:
            self._compute_by_target[target].sort(key=lambda x: x[0])

        # Index collective benchmarks by target and group_size
        for key, data in self.benchmarks.get("collectives", {}).items():
            parsed = _parse_benchmark_key(key)
            target = parsed.get("target", key)
            bytes_count = data.get("bytes", parsed.get("bytes", 0))
            runtime_ms = data.get("runtime_ms", 0)
            group_size = data.get("group_size", parsed.get("group_size", 0))

            if target not in self._collectives_by_target:
                self._collectives_by_target[target] = {}
            if group_size not in self._collectives_by_target[target]:
                self._collectives_by_target[target][group_size] = []
            self._collectives_by_target[target][group_size].append(
                (bytes_count, runtime_ms)
            )

        # Sort by bytes for binary search
        for target in self._collectives_by_target:
            for group_size in self._collectives_by_target[target]:
                self._collectives_by_target[target][group_size].sort(
                    key=lambda x: x[0]
                )

    @staticmethod
    def _linear_interpolate(
        points: list[tuple[int, float]], query_bytes: int
    ) -> Optional[float]:
        """
        Perform linear interpolation on sorted (bytes, runtime) points.

        Args:
            points: Sorted list of (bytes, runtime_ms) tuples
            query_bytes: The byte count to estimate runtime for

        Returns:
            Interpolated runtime in ms, or None if no data available
        """
        if not points:
            return None

        if len(points) == 1:
            # Single point - linear scaling from origin
            bytes_0, runtime_0 = points[0]
            if bytes_0 == 0:
                return runtime_0
            return runtime_0 * query_bytes / bytes_0

        # Find insertion point
        bytes_values = [p[0] for p in points]
        idx = bisect_left(bytes_values, query_bytes)

        if idx == 0:
            # Query is smaller than all points - extrapolate from first two
            b1, r1 = points[0]
            b2, r2 = points[1]
            if b2 == b1:
                return r1
            slope = (r2 - r1) / (b2 - b1)
            return max(0, r1 + slope * (query_bytes - b1))

        if idx >= len(points):
            # Query is larger than all points - extrapolate from last two
            b1, r1 = points[-2]
            b2, r2 = points[-1]
            if b2 == b1:
                return r2
            slope = (r2 - r1) / (b2 - b1)
            return max(0, r2 + slope * (query_bytes - b2))

        # Interpolate between two surrounding points
        b1, r1 = points[idx - 1]
        b2, r2 = points[idx]
        if b2 == b1:
            return r1
        t = (query_bytes - b1) / (b2 - b1)
        return r1 + t * (r2 - r1)

    def estimate_runtime(
        self, node: fx.Node, override_size: Optional[int] = None
    ) -> Optional[float]:
        """
        Estimate the runtime of a node using loaded benchmarks with interpolation.

        Args:
            node: The FX node to estimate
            override_size: Optional size override in bytes

        Returns:
            Estimated runtime in ms, or None if no estimate available
        """
        target = str(node.target)

        # Determine if this is a collective operation
        is_collective = False
        if hasattr(node.target, "namespace"):
            is_collective = node.target.namespace in {
                "_c10d_functional",
                "c10d_functional",
            }

        # Get size in bytes
        bytes_count = override_size
        if bytes_count is None:
            bytes_count = self._extract_node_bytes(node)

        if bytes_count is None:
            return None

        if is_collective:
            # Get group size for collectives
            group_size = self._extract_group_size(node)
            if target in self._collectives_by_target:
                # Try exact group_size match first
                if group_size in self._collectives_by_target[target]:
                    points = self._collectives_by_target[target][group_size]
                    result = self._linear_interpolate(points, bytes_count)
                    if result is not None:
                        return result

                # Fall back to any available group_size
                for gs, points in self._collectives_by_target[target].items():
                    result = self._linear_interpolate(points, bytes_count)
                    if result is not None:
                        # Scale by group size ratio (approximate)
                        if gs != 0 and group_size != 0:
                            result *= group_size / gs
                        return result
        else:
            if target in self._compute_by_target:
                points = self._compute_by_target[target]
                return self._linear_interpolate(points, bytes_count)

        return None

    def _extract_node_bytes(self, node: fx.Node) -> Optional[int]:
        """Extract byte count from a node's inputs."""
        from torch._inductor import fx_utils

        success, args, kwargs = fx_utils.get_fake_args_kwargs(node)
        if not success:
            return None

        total_bytes = 0
        found_tensor = False

        def count_bytes(t: torch.Tensor) -> torch.Tensor:
            nonlocal total_bytes, found_tensor
            shape = [get_hint(dim) for dim in t.shape]
            if all(s is not None for s in shape):
                numel = functools.reduce(operator.mul, shape, 1)
                total_bytes += numel * t.dtype.itemsize
                found_tensor = True
            return t

        torch.utils._pytree.tree_map_only(torch.Tensor, count_bytes, (args, kwargs))
        return total_bytes if found_tensor else None

    def _extract_group_size(self, node: fx.Node) -> int:
        """Extract group size from a collective node."""
        try:
            opt_args_kwargs = normalize_function(
                node.target,
                args=node.args,
                kwargs=node.kwargs,
                normalize_to_only_use_kwargs=True,
            )
            if opt_args_kwargs is not None:
                _, kwargs = opt_args_kwargs
                group_name = kwargs.get("group_name", "default")
                if torch.distributed.is_initialized():
                    from torch.distributed.distributed_c10d import (
                        _get_group_size_by_name,
                    )

                    return _get_group_size_by_name(group_name)
        except Exception:
            pass
        return 0

    @classmethod
    def from_file(cls, filepath: str) -> "BenchmarkEstimator":
        """
        Load an estimator from a JSON file.

        Args:
            filepath: Path to the JSON file saved by save_benchmarks_to_file

        Returns:
            BenchmarkEstimator instance
        """
        with open(filepath) as f:
            benchmarks = json.load(f)
        _module_log.info("Loaded benchmark estimator from %s", filepath)
        return cls(benchmarks)

    @classmethod
    def from_tlparse_artifact(cls, artifact_path: str) -> "BenchmarkEstimator":
        """
        Load an estimator from a tlparse artifact JSON file.

        Args:
            artifact_path: Path to the tlparse artifact file

        Returns:
            BenchmarkEstimator instance
        """
        return cls.from_file(artifact_path)


def create_estimator_from_benchmarks() -> Callable[[fx.Node, Optional[int]], Optional[float]]:
    """
    Create an estimator function from the currently collected benchmarks.
    This can be used as the custom_runtime_estimation parameter.

    Returns:
        A function that estimates runtime for a node
    """
    estimator = BenchmarkEstimator(get_all_benchmark_results())
    return estimator.estimate_runtime


# Reference to existing cache function for compatibility
def get_benchmark_cache() -> Any:
    """Get the existing benchmark cache from overlap_scheduling."""
    from torch._inductor.fx_passes.overlap_scheduling import get_benchmark_cache
    return get_benchmark_cache()


# ==============================================================================
# Goal 3: PyTorch Profiler Trace Estimator
# ==============================================================================


class ProfilerTraceEstimator:
    """
    An estimator that loads runtime estimates from a PyTorch profiler trace file
    (Chrome trace format) and provides runtime estimates with linear interpolation.

    Usage:
        # From a profiler trace file
        estimator = ProfilerTraceEstimator.from_chrome_trace("trace.json")

        # Use as a custom_runtime_estimation function
        schedule_overlap_bucketing(
            gm,
            custom_runtime_estimation=estimator.estimate_runtime,
        )
    """

    def __init__(self, benchmarks: dict[str, dict[str, Any]]) -> None:
        """
        Initialize the estimator with benchmark data extracted from profiler trace.

        Args:
            benchmarks: Dictionary with "compute" and "collectives" keys,
                       each containing key -> {runtime_ms, bytes, ...} mappings.
        """
        # Reuse BenchmarkEstimator's infrastructure
        self._estimator = BenchmarkEstimator(benchmarks)

    def estimate_runtime(
        self, node: fx.Node, override_size: Optional[int] = None
    ) -> Optional[float]:
        """
        Estimate the runtime of a node using loaded profiler data with interpolation.

        Args:
            node: The FX node to estimate
            override_size: Optional size override in bytes

        Returns:
            Estimated runtime in ms, or None if no estimate available
        """
        return self._estimator.estimate_runtime(node, override_size)

    @classmethod
    def from_chrome_trace(cls, filepath: str) -> "ProfilerTraceEstimator":
        """
        Load an estimator from a Chrome trace JSON file.

        The Chrome trace format is the standard output of PyTorch profiler's
        export_chrome_trace() method.

        Args:
            filepath: Path to the Chrome trace JSON file

        Returns:
            ProfilerTraceEstimator instance
        """
        with open(filepath) as f:
            trace_data = json.load(f)

        benchmarks = cls._parse_chrome_trace(trace_data)
        _module_log.info(
            "Loaded profiler trace estimator from %s with %d compute ops and %d collective ops",
            filepath,
            len(benchmarks.get("compute", {})),
            len(benchmarks.get("collectives", {})),
        )
        return cls(benchmarks)

    @staticmethod
    def _parse_chrome_trace(trace_data: Any) -> dict[str, dict[str, Any]]:
        """
        Parse Chrome trace format into benchmark data.

        Chrome trace format has events like:
        {
            "name": "aten::mm",
            "ph": "X",  # Complete event
            "ts": 1234567,  # Timestamp in microseconds
            "dur": 123,  # Duration in microseconds
            "cat": "cpu_op",
            "args": {...}
        }
        """
        benchmarks: dict[str, dict[str, Any]] = {
            "compute": {},
            "collectives": {},
        }

        # Handle both list format and dict format (with "traceEvents" key)
        events = trace_data
        if isinstance(trace_data, dict):
            events = trace_data.get("traceEvents", [])

        # Aggregate events by name to compute average runtime
        event_stats: dict[str, list[tuple[float, Optional[int]]]] = defaultdict(list)

        for event in events:
            if not isinstance(event, dict):
                continue

            name = event.get("name", "")
            dur = event.get("dur")  # Duration in microseconds
            ph = event.get("ph", "")  # Phase (X = complete event)

            # Only process complete events with duration
            if ph not in ("X", "x") or dur is None:
                continue

            # Skip metadata and system events
            if name.startswith("[") or name in ("", "ProfilerStep", "Iteration"):
                continue

            runtime_ms = dur / 1000.0  # Convert from microseconds to milliseconds

            # Try to extract size information from args
            args = event.get("args", {})
            input_bytes = None

            # Try to extract input dimensions/shapes
            shapes = args.get("Input Dims", args.get("input_dims", None))
            dtype_str = args.get("Input type", args.get("dtype", None))

            if shapes and dtype_str:
                input_bytes = ProfilerTraceEstimator._estimate_bytes_from_shapes(
                    shapes, dtype_str
                )

            event_stats[name].append((runtime_ms, input_bytes))

        # Convert aggregated stats to benchmark format
        for name, stats in event_stats.items():
            # Group by similar sizes for interpolation
            size_groups: dict[int, list[float]] = defaultdict(list)

            for runtime_ms, input_bytes in stats:
                key_bytes = input_bytes if input_bytes is not None else 0
                size_groups[key_bytes].append(runtime_ms)

            # Create benchmark entries for each size group
            name_lower = name.lower().replace("_", "")
            is_collective = any(
                coll in name_lower
                for coll in ["allreduce", "allgather", "reducescatter", "broadcast", "c10d", "nccl"]
            )
            category = "collectives" if is_collective else "compute"

            for size_bytes, runtimes in size_groups.items():
                # Use median runtime
                median_runtime = sorted(runtimes)[len(runtimes) // 2]

                key = f"{name}: ({size_bytes} bytes)"
                benchmarks[category][key] = {
                    "runtime_ms": median_runtime,
                    "bytes": size_bytes,
                }

        return benchmarks

    @staticmethod
    def _estimate_bytes_from_shapes(
        shapes: Any, dtype_str: Optional[str]
    ) -> Optional[int]:
        """
        Estimate total bytes from shape information and dtype string.

        Args:
            shapes: Shape information (could be string, list, etc.)
            dtype_str: Data type string like "float32", "Float", etc.

        Returns:
            Estimated total bytes, or None if cannot determine
        """
        try:
            # Parse shapes - could be [[1024, 1024]] or "[[1024, 1024]]" etc.
            if isinstance(shapes, str):
                import ast
                shapes = ast.literal_eval(shapes)

            if not shapes:
                return None

            # Calculate total elements
            total_elements = 0
            for shape in shapes if isinstance(shapes, list) else [shapes]:
                if isinstance(shape, (list, tuple)):
                    numel = 1
                    for dim in shape:
                        if isinstance(dim, (int, float)):
                            numel *= int(dim)
                    total_elements += numel

            if total_elements == 0:
                return None

            # Determine element size from dtype
            element_size = 4  # Default to float32
            if dtype_str:
                dtype_lower = dtype_str.lower()
                if "float64" in dtype_lower or "double" in dtype_lower:
                    element_size = 8
                elif "float16" in dtype_lower or "half" in dtype_lower:
                    element_size = 2
                elif "bfloat16" in dtype_lower:
                    element_size = 2
                elif "int64" in dtype_lower or "long" in dtype_lower:
                    element_size = 8
                elif "int32" in dtype_lower or "int" in dtype_lower:
                    element_size = 4
                elif "int16" in dtype_lower or "short" in dtype_lower:
                    element_size = 2
                elif "int8" in dtype_lower or "byte" in dtype_lower:
                    element_size = 1
                elif "bool" in dtype_lower:
                    element_size = 1

            return total_elements * element_size

        except Exception:
            return None

    @classmethod
    def from_profiler_events(cls, events: Any) -> "ProfilerTraceEstimator":
        """
        Load an estimator directly from PyTorch profiler events.

        Args:
            events: EventList from profiler.events() or profiler.key_averages()

        Returns:
            ProfilerTraceEstimator instance
        """
        benchmarks: dict[str, dict[str, Any]] = {
            "compute": {},
            "collectives": {},
        }

        for event in events:
            name = getattr(event, "key", getattr(event, "name", str(event)))

            # Get runtime in ms (event times are in us)
            cpu_time_ms = getattr(event, "cpu_time_total", 0) / 1000.0
            cuda_time_ms = getattr(event, "cuda_time_total", 0) / 1000.0

            # Prefer CUDA time if available
            runtime_ms = cuda_time_ms if cuda_time_ms > 0 else cpu_time_ms

            if runtime_ms <= 0:
                continue

            # Try to get input shapes
            input_shapes = getattr(event, "input_shapes", None)
            input_bytes = None
            if input_shapes:
                input_bytes = cls._estimate_bytes_from_shapes(input_shapes, None)

            name_lower = name.lower().replace("_", "")
            is_collective = any(
                coll in name_lower
                for coll in ["allreduce", "allgather", "reducescatter", "broadcast", "c10d", "nccl"]
            )
            category = "collectives" if is_collective else "compute"

            key = f"{name}: ({input_bytes or 0} bytes)"
            benchmarks[category][key] = {
                "runtime_ms": runtime_ms,
                "bytes": input_bytes or 0,
            }

        _module_log.info(
            "Created profiler trace estimator with %d compute ops and %d collective ops",
            len(benchmarks["compute"]),
            len(benchmarks["collectives"]),
        )
        return cls(benchmarks)
