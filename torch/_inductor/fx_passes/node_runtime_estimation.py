"""
Collective runtime estimation using CUDA events and power-of-2 rounding.
"""

from __future__ import annotations

import functools
import itertools
import operator
from functools import lru_cache
from typing import Any, Optional

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import _schedulable_wait_node
from torch._inductor.utils import clear_on_fresh_cache
from torch._logging import getArtifactLogger, trace_structured
from torch.fx.operator_schemas import normalize_function


def _format_csv(headers: list[str], rows: list[list[str]]) -> str:
    """Format data as CSV. Human-readable and easily parseable."""
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    return "\n".join(lines)


def _get_collective_key(coll_node: fx.Node) -> str:
    """Extract a unique key for a collective node including group info and tensor size."""
    from torch._inductor import fx_utils

    opt_args_kwargs = normalize_function(
        coll_node.target,  # type: ignore[arg-type]
        args=coll_node.args,
        kwargs=coll_node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    assert opt_args_kwargs is not None
    _, kwargs = opt_args_kwargs
    group_name = kwargs.get("group_name", None)
    group_size = kwargs.get("group_size", None)

    tensor_bytes: Optional[int] = None
    success, args, kw = fx_utils.get_fake_args_kwargs(coll_node)
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

    return f"{coll_node.target} group_size:{group_size} group_name:{group_name} input_bytes:{tensor_bytes}"


def _get_collective_estimations(coll_node: fx.Node) -> tuple[float, float]:
    """Get NCCL and Inductor analytical estimations for a collective node.

    Returns: (nccl_ms, inductor_ms)
    """
    nccl_ms = (
        torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
            coll_node, None, use_nccl_estimator=True
        )
    )
    inductor_ms = (
        torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
            coll_node, None, use_nccl_estimator=False
        )
    )
    return nccl_ms, inductor_ms


# Setup logger for artifact logging
log = getArtifactLogger(__name__, "node_runtime_estimation")


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
            f"{est_b * 1e3:.4f}",
            f"{est_a * 1e3:.4f}",
            f"{(est_a / est_b) if est_b > 0 else 0:.4f}",
            f"{(est_a - est_b) * 1e3:.4f}",
            str(count_flops_fx(node)),
        ]
        for node, est_b, est_a in zip(
            compute_nodes, benchmarked_estimations, analytical_estimations
        )
    ]

    log_str = _format_csv(headers, rows)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "fx_compute_nodes_runtime_estimation",
            "encoding": "string",
        },
        payload_fn=lambda: log_str,
    )


def _log_graph_collective_benchmarks(gm: fx.GraphModule, artifact_name: str) -> None:
    collective_nodes = []
    collective_keys = []
    benchmarked = []

    for node in gm.graph.nodes:
        if _schedulable_wait_node(node):
            start = node.args[0]
            collective_nodes.append(start)
            collective_keys.append(_get_collective_key(start))
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
    collective_keys: Optional[list[str]] = None,
    benchmarked_medians: Optional[list[float]] = None,
    world_size: Optional[int] = None,
    artifact_name: str = "fx_collectives_analytical_estimation",
) -> None:
    """Log collective estimations for tlparse. Includes benchmarks if provided."""
    if world_size is None:
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )

    has_benchmarks = benchmarked_medians is not None

    if has_benchmarks:
        headers = [
            "Collective Key",
            "Benchmarked(ms)",
            "NCCL Est(ms)",
            "Inductor Est(ms)",
            "NCCL Diff(ratio)",
            "Inductor Diff(ratio)",
        ]
    else:
        headers = [
            "Collective Key",
            "NCCL Est(ms)",
            "Inductor Est(ms)",
        ]

    rows = []
    for i, coll_node in enumerate(collective_nodes):
        key = collective_keys[i] if collective_keys else _get_collective_key(coll_node)
        nccl_ms, inductor_ms = _get_collective_estimations(coll_node)

        if benchmarked_medians is not None:
            benchmarked_ms = benchmarked_medians[i]
            nccl_diff_pct = (nccl_ms / benchmarked_ms) if benchmarked_ms > 0 else 0
            inductor_diff_pct = (
                (inductor_ms / benchmarked_ms) if benchmarked_ms > 0 else 0
            )
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
        else:
            rows.append(
                [
                    key,
                    f"{nccl_ms:.4f}",
                    f"{inductor_ms:.4f}",
                ]
            )

    log_str = f"# World size: {world_size}\n"
    log_str += _format_csv(headers, rows)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": artifact_name,
            "encoding": "string",
        },
        payload_fn=lambda: log_str,
    )
