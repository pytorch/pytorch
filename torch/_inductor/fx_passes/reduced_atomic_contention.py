# mypy: allow-untyped-defs
"""
Partitioned Scatter Optimization for Reduced Atomic Contention.

This pass transforms high-contention index_put operations by distributing
writes across multiple partitions, reducing atomic contention.

Algorithm:
  1. Enumerate scatter operations: operation_id = [0, 1, ..., N-1]
  2. Assign to partitions: partition_id = operation_id & (num_partitions - 1)
     (bitwise AND requires num_partitions to be a power of 2)
  3. Create expanded buffer along scatter_dim: size = num_partitions * dim_size
  4. Adjust indices: adjusted_idx = original_idx + (partition_id * dim_size)
  5. Scatter into expanded buffer with reduced contention
  6. Reshape and sum across partitions: result = sum(partitioned_view, dim=scatter_dim)
  7. Add to original input: output = input + result

Memory accounting (modelled on overlap_scheduling.py):
  - Build a per-node baseline memory profile from the original graph using
    MemoryTracker before any transforms run.
  - Track cumulative overhead from earlier transforms in the same pass via
    a per-node array (same pattern as cumulative_prefetch_mem_by_compute_index).
  - Gate each transform on: baseline[node] + cumulative[node] + overhead <= ceiling
  - Ceiling = total_gpu_memory - non_model_floor_bytes, where the floor covers
    GPU memory invisible to the FX profile: CUDA driver context, PyTorch caching
    allocator pool, and kernel workspace buffers (cuBLAS/Triton scratch).
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
from torch.fx.experimental.symbolic_shapes import optimization_hint


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

partitioned_scatter_patterns = PatternMatcherPass(
    pass_name="partitioned_scatter_optimization"
)

# Module-level state for the current pass invocation.
# Set in partitioned_scatter_optimization_pass, read in validate_match
# and create_replacement. Reset to None after the pass completes.
_current_pass_state: "ScatterMemoryState | None" = None


# ---------------------------------------------------------------------------
# Memory state — one instance per pass invocation
# ---------------------------------------------------------------------------


@dataclass
class ScatterMemoryState:
    """
    Per-pass memory accounting state.

    Modelled on OverlapScheduler in overlap_scheduling.py:
      - baseline_mem_by_node   ↔  original_mem_before_compute_index
      - cumulative_overhead    ↔  cumulative_prefetch_mem_by_compute_index
      - allowed_peak_bytes     ↔  allowed_peak_memory_bytes
    """

    # Live GPU bytes at each compute node from the original (un-transformed) graph.
    # Indexed by position in the compute node list (same order as graph iteration).
    baseline_mem_by_node: list[int]

    # Extra bytes committed by earlier scatter transforms still live at each node.
    # Updated via _commit_scatter_overhead after each transform fires.
    cumulative_overhead_by_node: list[int]

    # Map from FX node → its index in the compute node list.
    node_to_idx: dict[fx.Node, int]

    # Hard memory ceiling: total_gpu_memory - non_model_floor_bytes.
    # The floor accounts for GPU memory invisible to the FX profile:
    # CUDA driver context, caching allocator pool, and kernel workspaces.
    allowed_peak_bytes: int

    # For logging / summary
    total_gpu_bytes: int
    non_model_floor_bytes: int
    n_candidates: int = 0
    n_applied: int = 0
    skip_reasons: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # num_partitions chosen for each applied op, in graph order
    applied_partitions: list[int] = field(default_factory=list)


def _build_scatter_memory_state(graph: fx.Graph) -> "ScatterMemoryState | None":
    """
    Build per-pass memory state by simulating the original graph with MemoryTracker.

    Returns None when CUDA is unavailable (CPU-only environment); the pass will
    still run in that case but without memory gating.
    """
    if not torch.cuda.is_available():
        return None

    try:
        _, total_gpu = torch.cuda.mem_get_info()
    except Exception:
        log.debug("partitioned_scatter: mem_get_info failed, disabling memory gating")
        return None

    floor_bytes: int = config.partitioned_scatter_non_model_floor_bytes
    allowed_peak = max(0, total_gpu - floor_bytes)

    # primals are live for the full forward pass and must not be marked releasable.
    is_releasable = lambda n: not n.name.startswith("primals")
    tracker = MemoryTracker(graph, is_releasable=is_releasable)

    compute_nodes = [
        n for n in graph.nodes
        if n.op not in ("placeholder", "get_attr", "output")
    ]
    node_to_idx: dict[fx.Node, int] = {n: i for i, n in enumerate(compute_nodes)}
    baseline_mem: list[int] = []

    for node in compute_nodes:
        tracker.schedule_node(node)
        baseline_mem.append(tracker.get_current_memory_bytes())

    log.debug(
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
    Return the largest power-of-2 num_partitions P such that:
      1. Memory constraint: output_size * element_bytes * (P - 1) <= available_bytes
      2. Diminishing-returns cap (skipped when force=True):
         reduction work (P * output_size elements) does not exceed the index
         scatter work (index_size elements) by more than 4×.

    The second constraint prevents over-partitioning on large-memory GPUs where
    constraint (1) is trivially satisfied but adding more partitions buys negligible
    atomic savings while inflating the sum-reduction kernel.

    Returns 0 if even min_p doesn't fit.
    Power-of-2 is required because partition assignment uses bitwise AND.
    """
    if available_bytes <= 0 or output_size == 0 or element_bytes == 0:
        return 0

    # Constraint 1: expanded buffer fits in available headroom.
    # overhead = output_size * element_bytes * (P - 1)  →  P <= available / overhead_per_p + 1
    max_raw = available_bytes / (output_size * element_bytes) + 1
    if max_raw < min_p:
        return 0

    # Floor to power of 2 (largest that fits, not smallest that exceeds).
    p = 2 ** int(math.log2(max_raw))
    p = min(p, max_p)

    # Constraint 2: diminishing-returns cap.
    # The sum-reduction processes P * output_size elements; the scatter writes
    # index_size elements.  When the reduction dominates by more than 4× over the
    # scatter, adding more partitions costs more than it saves.
    # Cap: P * output_size <= 4 * index_size  →  P <= 4 * index_size / output_size
    # Use scatter_dim_size for a tighter bound when the output is multi-dimensional:
    # contention is per scatter-dim slot (scatter_dim_size), not per total element.
    # Bypassed in force mode so the caller can always obtain max_p partitions.
    if not force and index_size > 0 and scatter_dim_size > 0:
        # Estimate: each extra partition reduces atomic serialization by ~1/P, and
        # adds output_size / scatter_dim_size reduction elements per scatter slot.
        # Cap at 4× the "writes per scatter slot" so reduction never drowns savings.
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

    Mirrors _prefetch_would_exceed_memory_budget from overlap_scheduling.py:
      projected = baseline[node] + cumulative[node] + expanded_bytes <= allowed_peak

    Returns num_partitions (>= min_p) or 0 if memory constraint cannot be met.
    """
    min_p = config.partitioned_scatter_min_partitions
    max_p = config.partitioned_scatter_max_partitions

    idx = state.node_to_idx.get(output_node)
    if idx is None:
        # Node not in profile (shouldn't happen); fall back to unconstrained max.
        return max_p

    baseline = state.baseline_mem_by_node[idx]
    cumulative = state.cumulative_overhead_by_node[idx]
    available = state.allowed_peak_bytes - baseline - cumulative

    num_partitions = _compute_num_partitions(
        available, output_size, element_bytes, min_p, max_p,
        index_size=index_size, scatter_dim_size=scatter_dim_size,
        force=force,
    )

    if log.isEnabledFor(logging.DEBUG):
        overhead = output_size * element_bytes * max(0, num_partitions - 1)
        log.debug(
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
    """
    Charge overhead_bytes to every compute node in [scatter_idx, reduce_idx].

    Mirrors _update_cumulative_prefetch_memory from overlap_scheduling.py:
    the expanded buffer is live from the scatter node until the sum (reduce)
    node consumes it, so it must be counted against all nodes in that window.
    """
    scatter_idx = state.node_to_idx.get(scatter_node, 0)
    reduce_idx = state.node_to_idx.get(reduce_node, scatter_idx)
    n = len(state.cumulative_overhead_by_node)

    for i in range(scatter_idx, min(reduce_idx + 1, n)):
        state.cumulative_overhead_by_node[i] += overhead_bytes


def _resolve_numel(numel: Any) -> int | None:
    """
    Resolve numel to a concrete integer, handling SymInt for dynamic shapes.

    For SymInt, uses optimization_hint to get the hinted concrete value and
    applies a 2× safety multiplier so we don't over-allocate if the tensor
    grows at runtime.  Returns None if no hint is available.
    """
    if isinstance(numel, torch.SymInt):
        hint = optimization_hint(numel)
        if hint is None:
            return None
        # 2× safety factor: if the real size doubles at runtime, we still fit
        return hint * 2
    return int(numel)


# ---------------------------------------------------------------------------
# Pass entry point
# ---------------------------------------------------------------------------


def partitioned_scatter_optimization_pass(graph: fx.Graph) -> fx.Graph:
    """
    Apply partitioned scatter optimization to high-contention index_put operations.

    Reduces atomic contention by distributing writes across multiple partitions,
    at the cost of temporary memory for the expanded buffer.

    Controlled by: config.partitioned_scatter_enabled
    Enable via: TORCHINDUCTOR_PARTITIONED_SCATTER_ENABLED=1
    """
    global _current_pass_state

    if not config.partitioned_scatter_enabled:
        return graph

    # Build memory state from the original graph before any transforms.
    # When CUDA is unavailable this returns None; the pass runs unconstrained.
    _current_pass_state = _build_scatter_memory_state(graph)

    try:
        num_matches = partitioned_scatter_patterns.apply(graph)
    finally:
        state = _current_pass_state
        _current_pass_state = None

    if state is not None:
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


# ---------------------------------------------------------------------------
# Pattern matching and validation
# ---------------------------------------------------------------------------


def _record_skip(state: "ScatterMemoryState | None", reason: str, node_name: str, *args: Any) -> None:
    """Record a skip decision with structured logging and counter increment."""
    if state is not None:
        state.skip_reasons[reason] += 1
    counters["inductor"][f"partitioned_scatter_skipped_{reason}"] += 1
    if log.isEnabledFor(logging.DEBUG):
        fmt = f"partitioned_scatter: SKIP node=%s reason={reason}"
        if args:
            fmt += " " + " ".join(str(a) for a in args)
        log.debug(fmt, node_name)


def validate_match(match: Match) -> bool:
    """
    Check if a matched index_put should be transformed.

    Gates (in order):
      1. accumulate=True                 (correctness — only scatter-adds benefit)
      2. Single non-None index           (multi-axis indexing not supported)
      3. Valid tensor metadata           (need shapes/dtype to compute budget)
      4. Non-bool dtype                  (bool scatter-adds are typically tiny)
      5. scatter_dim in bounds           (safety)
      6. index_size >= min_index_size    (small scatters have no meaningful contention)
      7. contention_ratio >= threshold   (low write density → low atomic pressure)
      8. Memory budget via _check_memory (maximise partitions within headroom)
    """
    state = _current_pass_state
    output_node = match.output_node()

    if not output_node or not hasattr(output_node, "args") or len(output_node.args) < 4:
        return False

    node_name = output_node.name

    if state is not None:
        state.n_candidates += 1

    # Gate 1: accumulate=True only
    if output_node.args[3] is not True:
        _record_skip(state, "accumulate_false", node_name)
        return False

    input_node = output_node.args[0]
    indices_arg = output_node.args[1]

    if not isinstance(input_node, fx.Node):
        _record_skip(state, "input_not_node", node_name)
        return False

    # Gate 2: single non-None index
    scatter_dim, index_node = _extract_scatter_dim_and_index(indices_arg)
    if scatter_dim is None or index_node is None:
        _record_skip(state, "multi_index", node_name)
        return False

    # Gate 3: tensor metadata
    input_meta = _get_tensor_meta(input_node)
    index_meta = _get_tensor_meta(index_node)
    if not input_meta or not index_meta:
        _record_skip(state, "no_meta", node_name)
        return False

    # Gate 4: bool dtype not supported (accumulate semantics differ)
    if input_meta["dtype"] == torch.bool or index_meta["dtype"] == torch.bool:
        _record_skip(state, "bool_dtype", node_name)
        return False

    # Gate 5: scatter_dim in bounds
    if scatter_dim >= len(input_meta["shape"]):
        _record_skip(state, "dim_out_of_bounds", node_name)
        return False

    # Resolve numel — handles both static int and SymInt (dynamic shapes)
    output_size = _resolve_numel(input_meta["numel"])
    index_size = _resolve_numel(index_meta["numel"])

    if output_size is None or index_size is None:
        _record_skip(state, "dynamic_no_hint", node_name)
        return False

    if output_size == 0 or index_size == 0:
        _record_skip(state, "zero_size", node_name)
        return False

    # scatter_dim_size: number of unique slots along the scatter dimension.
    # For [n, D] output scattered along dim 0, this is n (not n*D).
    # Atomic contention is per slot: index_size / scatter_dim_size writes compete
    # for the same memory location, independent of the trailing D dimensions.
    scatter_dim_size = _resolve_numel(input_meta["shape"][scatter_dim])
    if scatter_dim_size is None or scatter_dim_size == 0:
        _record_skip(state, "zero_size", node_name)
        return False

    force: bool = config.partitioned_scatter_force

    # Gate 6: minimum index size (small scatters have negligible atomic contention).
    # Skipped in force mode — useful for synthetic/small benchmarks or when the
    # caller knows index values are highly skewed despite small index_size.
    min_index_size: int = config.partitioned_scatter_min_index_size
    if not force and index_size < min_index_size:
        _record_skip(
            state, "index_too_small", node_name,
            f"index_size={index_size} min={min_index_size}",
        )
        return False

    # Gate 7: contention ratio gate — skip if write density is too low.
    # Use scatter_dim_size (not output_numel) as the denominator: contention is
    # per scatter-dimension slot.  For a [vocab=50000, dim=768] embedding with
    # N=65536 indices, the correct ratio is 65536/50000 = 1.31, not
    # 65536/(50000*768) = 0.0017 which would incorrectly skip the optimisation.
    # Skipped in force mode — useful when static heuristics under-estimate
    # contention (e.g. skewed/adversarial index distributions).
    contention_ratio = index_size / scatter_dim_size
    min_contention: float = config.partitioned_scatter_min_contention_ratio
    if not force and contention_ratio < min_contention:
        _record_skip(
            state, "low_contention", node_name,
            f"contention_ratio={contention_ratio:.3f} threshold={min_contention:.3f}",
        )
        return False

    # Gate 8: memory budget — select num_partitions within available headroom.
    element_bytes: int = input_meta["dtype"].itemsize
    min_p: int = config.partitioned_scatter_min_partitions
    max_p: int = config.partitioned_scatter_max_partitions

    if state is not None:
        num_partitions = _check_memory(
            state, output_node, output_size, element_bytes,
            index_size=index_size, scatter_dim_size=scatter_dim_size,
            force=force,
        )
    else:
        # No memory state (CUDA unavailable) — apply diminishing-returns cap only
        # (unless force mode, which skips the cap and goes straight to max_p).
        num_partitions = _compute_num_partitions(
            available_bytes=2**62,  # effectively unlimited
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
            state, "memory_budget", node_name,
            f"num_partitions={num_partitions} min={min_p}",
        )
        return False

    # Store parameters for create_replacement (stashed on match object)
    match._num_partitions = num_partitions  # type: ignore[attr-defined]
    match._scatter_dim = scatter_dim  # type: ignore[attr-defined]
    match._index_node = index_node  # type: ignore[attr-defined]
    match._output_size = output_size  # type: ignore[attr-defined]
    match._element_bytes = element_bytes  # type: ignore[attr-defined]

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
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
        """Partitioned scatter implementation traced by replace_by_example."""
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

        # Assign each write operation to a partition via bitwise AND.
        # Requires num_partitions to be a power of 2.
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

        # Expanded buffer: num_partitions copies of the output along scatter_dim.
        # Each partition writes into its own slice, eliminating inter-partition
        # atomic contention.
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

        # Scatter into expanded buffer — reduced contention because threads in
        # different partitions write to addresses separated by dim_size elements
        scattered_buffer = torch.ops.aten.index_put.default(
            expanded_buffer, adjusted_indices, flat_values, True
        )

        # Reshape to [... num_partitions, dim_size, ...] then sum partitions
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

    # Update memory state: charge expanded buffer cost to every compute node
    # in the window [scatter_node, sum_node], mirroring the overlap_scheduling
    # _update_cumulative_prefetch_memory pattern.
    if state is not None:
        overhead_bytes = output_size * element_bytes * (num_partitions - 1)
        scatter_node = match.output_node()

        # The sum node is the last node inserted — find it by walking newly added nodes.
        # replace_by_example inserts nodes before the original; the reduce (sum) is
        # the penultimate of the inserted nodes (before the final add).
        # We conservatively charge to the scatter node itself as the entire window,
        # since after replacement the exact reduce node position requires graph re-walk.
        # This is conservative: it charges the overhead to all nodes from the scatter
        # onwards rather than just up to the reduce, which is safe (underestimates
        # headroom for later nodes rather than overestimating it).
        _commit_scatter_overhead(state, scatter_node, scatter_node, overhead_bytes)

        state.n_applied += 1
        state.applied_partitions.append(num_partitions)

    counters["inductor"]["partitioned_scatter_applied"] += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_scatter_dim_and_index(
    indices_arg: Any,
) -> tuple[int | None, fx.Node | None]:
    """Extract scatter dimension and index node from indices argument."""
    # Case 1: bare tensor index → dim 0
    if not isinstance(indices_arg, (list, tuple)):
        return 0, indices_arg

    # Case 2: list[index | None] — position of the single non-None entry is the dim
    index_node = None
    scatter_dim = None

    for dim, idx in enumerate(indices_arg):
        if idx is not None:
            if index_node is not None:
                # Multiple non-None indices → multi-axis indexing, not supported
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
