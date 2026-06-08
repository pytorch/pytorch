# mypy: allow-untyped-defs
"""
Partitioned scatter optimization for high-contention index_put operations.

Algorithm:
  1. Assign each write operation a partition: partition_id = op_id & (P - 1)
  2. Scatter into an expanded buffer of size P * dim_size along scatter_dim
  3. Reshape to [..., P, dim_size, ...] and sum across partitions
  4. Add result to the original input
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.fx_passes.memory_estimator import MemoryTracker
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._logging import getArtifactLogger
from torch.fx.experimental.symbolic_shapes import optimization_hint


log = logging.getLogger(__name__)
_artifact_log = getArtifactLogger(__name__, "partitioned_scatter")
aten = torch.ops.aten
prims = torch.ops.prims

partitioned_scatter_patterns = PatternMatcherPass(
    pass_name="partitioned_scatter_optimization"
)

# Set in partitioned_scatter_optimization_pass, read in validate_match / create_replacement.
_current_pass_state: "ScatterMemoryState | None" = None


@dataclass
class ScatterMemoryState:
    # Live GPU bytes at each compute node in the original (un-transformed) graph.
    baseline_mem_by_node: list[int]

    # Extra bytes from earlier scatter transforms still live at each node.
    cumulative_overhead_by_node: list[int]

    node_to_idx: dict[fx.Node, int]

    # total_gpu_memory - non_model_floor_bytes
    allowed_peak_bytes: int

    total_gpu_bytes: int
    non_model_floor_bytes: int
    n_candidates: int = 0
    n_applied: int = 0
    skip_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    applied_partitions: list[int] = field(default_factory=list)


def _build_scatter_memory_state(graph: fx.Graph) -> "ScatterMemoryState | None":
    """
    Simulate the original graph with MemoryTracker to build per-pass memory state.
    Returns None when CUDA is unavailable; the pass runs unconstrained in that case.
    """
    if not torch.cuda.is_available():
        return None

    try:
        _, total_gpu = torch.cuda.mem_get_info()
    except Exception:
        _artifact_log.debug(
            "partitioned_scatter: mem_get_info failed, disabling memory gating"
        )
        return None

    floor_bytes: int = config.partitioned_scatter_non_model_floor_bytes
    allowed_peak = max(0, total_gpu - floor_bytes)

    def is_releasable(n: fx.Node) -> bool:
        return not n.name.startswith("primals")

    try:
        tracker = MemoryTracker(graph, is_releasable=is_releasable)
    except Exception as e:
        _artifact_log.debug(
            "partitioned_scatter: MemoryTracker init failed (%s), "
            "running without memory gating",
            e,
        )
        return None

    compute_nodes = [
        n for n in graph.nodes if n.op not in ("placeholder", "get_attr", "output")
    ]
    node_to_idx: dict[fx.Node, int] = {n: i for i, n in enumerate(compute_nodes)}
    baseline_mem: list[int] = []

    try:
        for node in compute_nodes:
            tracker.schedule_node(node)
            baseline_mem.append(tracker.get_current_memory_bytes())
    except Exception as e:
        _artifact_log.debug(
            "partitioned_scatter: MemoryTracker simulation failed (%s), "
            "running without memory gating",
            e,
        )
        return None

    _artifact_log.debug(
        "partitioned_scatter: memory state built — "
        "graph_nodes=%d compute_nodes=%d "
        "total_gpu=%d MB floor=%d MB allowed_peak=%d MB",
        sum(1 for _ in graph.nodes),
        len(compute_nodes),
        total_gpu // 1_000_000,
        floor_bytes // 1_000_000,
        allowed_peak // 1_000_000,
    )

    return ScatterMemoryState(
        baseline_mem_by_node=baseline_mem,
        cumulative_overhead_by_node=[0] * len(compute_nodes),
        node_to_idx=node_to_idx,
        allowed_peak_bytes=allowed_peak,
        total_gpu_bytes=total_gpu,
        non_model_floor_bytes=floor_bytes,
    )


def _compute_num_partitions(
    available_bytes: int,
    output_size: int,
    element_bytes: int,
    min_p: int,
    max_p: int,
    index_size: int = 0,
    scatter_dim_size: int = 0,
    force: bool = False,
) -> int:
    """
    Return the largest power-of-2 P in [min_p, max_p] satisfying:
      1. Memory: output_size * element_bytes * (P - 1) <= available_bytes
      2. Diminishing-returns cap (skipped when force=True):
         P * output_size <= 4 * index_size, using scatter_dim_size for tighter bound.

    Returns 0 if min_p doesn't fit. Power-of-2 is required by the bitwise-AND
    partition assignment.
    """
    if available_bytes <= 0 or output_size == 0 or element_bytes == 0:
        return 0

    max_raw = available_bytes / (output_size * element_bytes) + 1
    if max_raw < min_p:
        return 0

    p = 2 ** int(math.log2(max_raw))
    p = min(p, max_p)

    if not force and index_size > 0 and scatter_dim_size > 0:
        writes_per_slot = index_size / scatter_dim_size
        contention_cap = max(min_p, 2 ** int(math.log2(max(1, 4 * writes_per_slot))))
        p = min(p, contention_cap)

    return p


def _check_memory(
    state: ScatterMemoryState,
    output_node: fx.Node,
    output_size: int,
    element_bytes: int,
    index_size: int = 0,
    scatter_dim_size: int = 0,
    force: bool = False,
) -> int:
    """
    Compute num_partitions for this node given current memory state.
    Returns num_partitions >= min_p, or 0 if the memory constraint cannot be met.
    """
    min_p = config.partitioned_scatter_min_partitions
    max_p = config.partitioned_scatter_max_partitions

    idx = state.node_to_idx.get(output_node)
    if idx is None:
        return max_p

    baseline = state.baseline_mem_by_node[idx]
    cumulative = state.cumulative_overhead_by_node[idx]
    available = state.allowed_peak_bytes - baseline - cumulative

    num_partitions = _compute_num_partitions(
        available,
        output_size,
        element_bytes,
        min_p,
        max_p,
        index_size=index_size,
        scatter_dim_size=scatter_dim_size,
        force=force,
    )

    if _artifact_log.isEnabledFor(logging.DEBUG):
        overhead = output_size * element_bytes * max(0, num_partitions - 1)
        _artifact_log.debug(
            "partitioned_scatter: memory check node=%s "
            "baseline=%d MB cumulative=%d MB available=%d MB "
            "expanded_buffer_cost=%d MB num_partitions=%d "
            "total_gpu=%d MB floor=%d MB allowed_peak=%d MB",
            output_node.name,
            baseline // 1_000_000,
            cumulative // 1_000_000,
            available // 1_000_000,
            overhead // 1_000_000,
            num_partitions,
            state.total_gpu_bytes // 1_000_000,
            state.non_model_floor_bytes // 1_000_000,
            state.allowed_peak_bytes // 1_000_000,
        )

    return num_partitions


def _commit_scatter_overhead(
    state: ScatterMemoryState,
    scatter_node: fx.Node,
    reduce_node: fx.Node,
    overhead_bytes: int,
) -> None:
    """Charge overhead_bytes to every compute node in [scatter_idx, reduce_idx]."""
    scatter_idx = state.node_to_idx.get(scatter_node, 0)
    reduce_idx = state.node_to_idx.get(reduce_node, scatter_idx)
    n = len(state.cumulative_overhead_by_node)

    for i in range(scatter_idx, min(reduce_idx + 1, n)):
        state.cumulative_overhead_by_node[i] += overhead_bytes


def _resolve_numel(numel: Any) -> int | None:
    """Resolve numel to a concrete int, handling SymInt via optimization_hint."""
    if isinstance(numel, torch.SymInt):
        hint = optimization_hint(numel)
        if hint is None:
            return None
        return hint * 2  # 2× safety margin for dynamic shapes
    return int(numel)


def partitioned_scatter_optimization_pass(graph: fx.Graph) -> fx.Graph:
    """
    Apply partitioned scatter optimization to high-contention index_put operations.
    Controlled by config.partitioned_scatter_enabled.
    """
    global _current_pass_state

    if not config.partitioned_scatter_enabled:
        return graph

    _current_pass_state = _build_scatter_memory_state(graph)

    try:
        num_matches = partitioned_scatter_patterns.apply(graph)
    finally:
        state = _current_pass_state
        _current_pass_state = None

    if state is not None and state.n_candidates > 0:
        log.info(
            "partitioned_scatter: candidates=%d applied=%d skipped=%d "
            "partitions_per_op=%s "
            "skip_breakdown=%s "
            "total_gpu=%d MB floor=%d MB allowed_peak=%d MB",
            state.n_candidates,
            state.n_applied,
            state.n_candidates - state.n_applied,
            state.applied_partitions,
            dict(state.skip_reasons),
            state.total_gpu_bytes // 1_000_000,
            state.non_model_floor_bytes // 1_000_000,
            state.allowed_peak_bytes // 1_000_000,
        )
    elif num_matches > 0:
        log.info(
            "partitioned_scatter: applied=%d (no memory state, CUDA unavailable)",
            num_matches,
        )

    if num_matches > 0:
        graph.lint()

    return graph


def _record_skip(
    state: "ScatterMemoryState | None", reason: str, node_name: str, *args: Any
) -> None:
    if state is not None:
        state.skip_reasons[reason] += 1
    counters["inductor"][f"partitioned_scatter_skipped_{reason}"] += 1
    if _artifact_log.isEnabledFor(logging.DEBUG):
        fmt = f"partitioned_scatter: SKIP node=%s reason={reason}"
        if args:
            fmt += " " + " ".join(str(a) for a in args)
        _artifact_log.debug(fmt, node_name)


def validate_match(match: Match) -> bool:
    """
    Gates (in order):
      1. accumulate=True only
      2. Single non-None index (multi-axis not supported)
      3. Valid tensor metadata
      4. Non-bool dtype
      5. scatter_dim in bounds
      6. index_size >= min_index_size
      7. contention_ratio >= threshold  (uses scatter_dim_size, not output_numel)
      8. Memory budget
    """
    state = _current_pass_state
    output_node = match.output_node()

    if not output_node or not hasattr(output_node, "args") or len(output_node.args) < 4:
        return False

    node_name = output_node.name

    if state is not None:
        state.n_candidates += 1

    if output_node.args[3] is not True:
        _record_skip(state, "accumulate_false", node_name)
        return False

    input_node = output_node.args[0]
    indices_arg = output_node.args[1]

    if not isinstance(input_node, fx.Node):
        _record_skip(state, "input_not_node", node_name)
        return False

    scatter_dim, index_node = _extract_scatter_dim_and_index(indices_arg)
    if scatter_dim is None or index_node is None:
        _record_skip(state, "multi_index", node_name)
        return False

    input_meta = _get_tensor_meta(input_node)
    index_meta = _get_tensor_meta(index_node)
    if not input_meta or not index_meta:
        _record_skip(state, "no_meta", node_name)
        return False

    if input_meta["dtype"] == torch.bool or index_meta["dtype"] == torch.bool:
        _record_skip(state, "bool_dtype", node_name)
        return False

    if scatter_dim >= len(input_meta["shape"]):
        _record_skip(state, "dim_out_of_bounds", node_name)
        return False

    output_size = _resolve_numel(input_meta["numel"])
    index_size = _resolve_numel(index_meta["numel"])

    if output_size is None or index_size is None:
        _record_skip(state, "dynamic_no_hint", node_name)
        return False

    if output_size == 0 or index_size == 0:
        _record_skip(state, "zero_size", node_name)
        return False

    # Contention is per scatter-dim slot, so use scatter_dim_size as denominator.
    # For a [vocab, dim] output, ratio = N/vocab, not N/(vocab*dim).
    scatter_dim_size = _resolve_numel(input_meta["shape"][scatter_dim])
    if scatter_dim_size is None or scatter_dim_size == 0:
        _record_skip(state, "zero_size", node_name)
        return False

    force: bool = config.partitioned_scatter_force

    min_index_size: int = config.partitioned_scatter_min_index_size
    if not force and index_size < min_index_size:
        _record_skip(
            state,
            "index_too_small",
            node_name,
            f"index_size={index_size} min={min_index_size}",
        )
        return False

    contention_ratio = index_size / scatter_dim_size
    min_contention: float = config.partitioned_scatter_min_contention_ratio
    if not force and contention_ratio < min_contention:
        _record_skip(
            state,
            "low_contention",
            node_name,
            f"contention_ratio={contention_ratio:.3f} threshold={min_contention:.3f}",
        )
        return False

    element_bytes: int = input_meta["dtype"].itemsize
    min_p: int = config.partitioned_scatter_min_partitions
    max_p: int = config.partitioned_scatter_max_partitions

    if state is not None:
        num_partitions = _check_memory(
            state,
            output_node,
            output_size,
            element_bytes,
            index_size=index_size,
            scatter_dim_size=scatter_dim_size,
            force=force,
        )
    else:
        num_partitions = _compute_num_partitions(
            available_bytes=2**62,
            output_size=output_size,
            element_bytes=element_bytes,
            min_p=min_p,
            max_p=max_p,
            index_size=index_size,
            scatter_dim_size=scatter_dim_size,
            force=force,
        )

    if num_partitions < min_p:
        _record_skip(
            state,
            "memory_budget",
            node_name,
            f"num_partitions={num_partitions} min={min_p}",
        )
        return False

    match._num_partitions = num_partitions  # type: ignore[attr-defined]
    match._scatter_dim = scatter_dim  # type: ignore[attr-defined]
    match._index_node = index_node  # type: ignore[attr-defined]
    match._output_size = output_size  # type: ignore[attr-defined]
    match._element_bytes = element_bytes  # type: ignore[attr-defined]

    if _artifact_log.isEnabledFor(logging.DEBUG):
        _artifact_log.debug(
            "partitioned_scatter: APPLY node=%s "
            "num_partitions=%d scatter_dim=%d "
            "contention_ratio=%.1f (index_size=%d / scatter_dim_size=%d) "
            "output_size=%d dtype=%s%s",
            node_name,
            num_partitions,
            scatter_dim,
            contention_ratio,
            index_size,
            scatter_dim_size,
            output_size,
            input_meta["dtype"],
            " [force]" if force else "",
        )

    return True


@register_graph_pattern(
    CallFunction(aten.index_put.default, Arg(), Arg(), Arg(), True),
    pass_dict=partitioned_scatter_patterns,  # type: ignore[arg-type]
    extra_check=validate_match,
)
@register_graph_pattern(
    CallFunction(aten.index_put_.default, Arg(), Arg(), Arg(), True),
    pass_dict=partitioned_scatter_patterns,  # type: ignore[arg-type]
    extra_check=validate_match,
)
def create_replacement(match: Match, input_tensor, indices, values) -> None:
    """Replace high-contention index_put with partitioned scatter."""
    state = _current_pass_state

    num_partitions: int = match._num_partitions  # type: ignore[attr-defined]
    scatter_dim: int = match._scatter_dim  # type: ignore[attr-defined]
    index_node = match._index_node  # type: ignore[attr-defined]
    output_size: int = match._output_size  # type: ignore[attr-defined]
    element_bytes: int = match._element_bytes  # type: ignore[attr-defined]

    def repl(input_tensor, index_node, values):
        dim_size = input_tensor.shape[scatter_dim]
        num_operations = index_node.numel()

        # Flatten multi-dimensional indices to 1-D
        if len(index_node.shape) > 1:
            flat_index = index_node.reshape(num_operations)
            values_ndim = len(index_node.shape)
            flat_values = values.reshape(
                [num_operations] + list(values.shape[values_ndim:])
            )
        else:
            flat_index = index_node
            flat_values = values

        # partition_id = op_id & (num_partitions - 1), requires power-of-2
        operation_ids = torch.ops.prims.iota.default(
            num_operations,
            start=0,
            step=1,
            dtype=flat_index.dtype,
            device=flat_index.device,
            requires_grad=False,
        )
        partition_ids = torch.ops.aten.bitwise_and.Scalar(
            operation_ids, num_partitions - 1
        )

        # Expanded buffer: one copy per partition along scatter_dim
        expanded_shape = list(input_tensor.shape)
        expanded_shape[scatter_dim] *= num_partitions
        expanded_buffer = torch.ops.aten.full.default(
            expanded_shape,
            0,
            dtype=flat_values.dtype,
            layout=torch.strided,
            device=flat_values.device,
            pin_memory=False,
        )

        # Shift each write into its partition's slice
        partition_offsets = partition_ids * dim_size
        adjusted_index = flat_index + partition_offsets

        if isinstance(indices, (list, tuple)):
            adjusted_indices = [
                adjusted_index if i == scatter_dim else idx
                for i, idx in enumerate(indices)
            ]
        else:
            adjusted_indices = [adjusted_index]

        scattered_buffer = torch.ops.aten.index_put.default(
            expanded_buffer, adjusted_indices, flat_values, True
        )

        # Reshape to [..., num_partitions, dim_size, ...] then sum partitions
        reduce_shape = list(expanded_shape)
        reduce_shape[scatter_dim] = num_partitions
        reduce_shape.insert(scatter_dim + 1, dim_size)
        reshaped = torch.ops.aten.view.default(scattered_buffer, reduce_shape)

        # Preserve dtype for integer types that don't promote during sum
        if flat_values.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            reduced = torch.ops.aten.sum.dim_IntList(
                reshaped, [scatter_dim], dtype=flat_values.dtype
            )
        else:
            reduced = torch.ops.aten.sum.dim_IntList(reshaped, [scatter_dim])

        return input_tensor + reduced

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, [input_tensor, index_node, values])

    # Charge the expanded buffer cost to nodes in the scatter→sum window.
    # Conservative: we attribute it entirely to the scatter node since the
    # exact sum node position requires a graph re-walk after replacement.
    if state is not None:
        overhead_bytes = output_size * element_bytes * (num_partitions - 1)
        _commit_scatter_overhead(
            state, match.output_node(), match.output_node(), overhead_bytes
        )
        state.n_applied += 1
        state.applied_partitions.append(num_partitions)

    counters["inductor"]["partitioned_scatter_applied"] += 1


def _extract_scatter_dim_and_index(
    indices_arg: Any,
) -> tuple[int | None, fx.Node | None]:
    """Extract scatter dimension and index node from indices argument."""
    if not isinstance(indices_arg, (list, tuple)):
        return 0, indices_arg

    index_node = None
    scatter_dim = None

    for dim, idx in enumerate(indices_arg):
        if idx is not None:
            if index_node is not None:
                return None, None
            index_node = idx
            scatter_dim = dim

    return scatter_dim, index_node


def _get_tensor_meta(node: fx.Node) -> dict[str, Any] | None:
    """Extract tensor metadata from an FX node's meta['val'] FakeTensor."""
    if not hasattr(node, "meta") or "val" not in node.meta:
        return None

    val = node.meta["val"]
    if not hasattr(val, "shape") or not hasattr(val, "dtype"):
        return None

    return {
        "shape": tuple(val.shape),
        "dtype": val.dtype,
        "device": val.device,
        "numel": val.numel(),
    }


__all__ = ["partitioned_scatter_optimization_pass"]
