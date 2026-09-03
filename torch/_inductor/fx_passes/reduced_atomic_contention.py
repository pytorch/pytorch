# mypy: allow-untyped-defs
"""
Partitioned scatter optimization for high-contention scatter operations.

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
from torch._inductor.fx_passes.memory_estimator import build_memory_profile
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Ignored,
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

_INDEX_PUT_TARGETS = (aten.index_put.default, aten.index_put_.default)

# embedding_dense_backward lands here rather than on index_put: its decomposition
# is in decomps_to_exclude, so Inductor lowers this op directly at IR level.
_MASKED_INDEX_PUT_TARGET = aten._unsafe_masked_index_put_accumulate.default

# target -> (indices arg position, mask arg position or None)
#   index_put(self, indices, values, accumulate)
#   _unsafe_masked_index_put_accumulate(self, mask, indices, values)
_SCATTER_ARGS = {
    aten.index_put.default: (1, None),
    aten.index_put_.default: (1, None),
    _MASKED_INDEX_PUT_TARGET: (2, 1),
}

# index_add only decomposes into index_put for some dtypes; below fp32 it reaches
# the post-grad graph intact, so match it directly.
#   index_add(self, dim, index, source, *, alpha=1)
_INDEX_ADD_TARGET = aten.index_add.default

# Same atomic_add store via the scatter_reduce_ lowering, but with an explicit
# dim and a values-shaped index.
#   scatter_add(self, dim, index, src)
#   scatter_reduce(self, dim, index, src, reduce, *, include_self)
#   scatter(self, dim, index, src, *, reduce)
_SCATTER_REDUCE_TARGETS = (
    aten.scatter_add.default,
    aten.scatter_reduce.two,
    aten.scatter.reduce,
)


def _is_summing_scatter(node: fx.Node) -> bool:
    """Only reduce="sum" lowers to atomic_add, and include_self=False leaves
    untouched slots at their original value, which scatter-into-zeros cannot."""
    if node.kwargs.get("include_self", True) is not True:
        return False
    if node.target is aten.scatter_add.default:
        return True
    reduce = node.kwargs.get("reduce")
    if reduce is None and len(node.args) > 4:
        reduce = node.args[4]
    return reduce in ("sum", "add")


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """Dtype the partial sums are accumulated in, widened for narrow floats when
    partitioned_scatter_fp32_accumulation is set."""
    if (
        config.partitioned_scatter_fp32_accumulation
        and dtype.is_floating_point
        and dtype.itemsize < 4
    ):
        return torch.float32
    return dtype


@dataclass
class ScatterCandidate:
    """A scatter op that passed the cheap (non-memory) gates."""

    output_node: fx.Node
    index_node: fx.Node
    scatter_dim: int
    output_size: int
    index_size: int
    scatter_dim_size: int
    # Upper bound on the scattered values' numel, exact for the index_put family.
    values_numel: int
    contention_ratio: float
    dtype: torch.dtype
    acc_dtype: torch.dtype

    # Only set for _unsafe_masked_index_put_accumulate.
    mask_node: "fx.Node | None" = None


@dataclass
class ScatterMemoryState:
    # Peak live GPU bytes at each compute node in the original (un-transformed)
    # graph, taken at the allocation phase (before this node's last-use inputs
    # are freed) so the expanded scatter buffer is charged against the true peak.
    peak_mem_by_node: list[int]

    node_to_idx: dict[fx.Node, int]

    # total_gpu_memory - non_model_floor_bytes
    allowed_peak_bytes: int

    total_gpu_bytes: int
    non_model_floor_bytes: int

    # Expanded-buffer bytes already granted to scatters transformed earlier in this
    # invocation. peak_mem_by_node profiles the *original* graph, so without this every
    # candidate is sized against the same baseline and each claims the whole budget.
    # A running sum rather than a per-live-range charge because scatters sharing an
    # input fuse into one kernel, which needs all their expanded buffers live at once.
    committed_overhead_bytes: int = 0


@dataclass
class ScatterPassContext:
    """Per-invocation state, passed to the pattern callbacks via closures."""

    # Scatter nodes that survived the cheap pre-scan, keyed by output node.
    candidates: dict[fx.Node, ScatterCandidate] = field(default_factory=dict)

    # Built lazily, only when at least one candidate survives the pre-scan.
    memory: "ScatterMemoryState | None" = None

    n_candidates: int = 0
    n_applied: int = 0
    skip_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    applied_partitions: list[int] = field(default_factory=list)


def _record_skip(
    ctx: ScatterPassContext, reason: str, node_name: str, *args: Any
) -> None:
    ctx.skip_reasons[reason] += 1
    counters["inductor"][f"partitioned_scatter_skipped_{reason}"] += 1
    if _artifact_log.isEnabledFor(logging.DEBUG):
        fmt = f"partitioned_scatter: SKIP node=%s reason={reason}"
        if args:
            fmt += " " + " ".join(str(a) for a in args)
        _artifact_log.debug(fmt, node_name)


def _evaluate_candidate(
    output_node: fx.Node, force: bool, ctx: ScatterPassContext
) -> "ScatterCandidate | None":
    """Every gate that does not need the memory profile: supported shape, dtype
    and device, then the min-index-size and contention-ratio thresholds. Each
    rejection is recorded under its own skip reason."""
    node_name = output_node.name
    is_scatter_reduce = output_node.target in _SCATTER_REDUCE_TARGETS
    is_index_add = output_node.target is _INDEX_ADD_TARGET

    input_node = output_node.args[0]
    if not isinstance(input_node, fx.Node):
        _record_skip(ctx, "input_not_node", node_name)
        return None

    if is_scatter_reduce or is_index_add:
        scatter_dim, index_node = output_node.args[1], output_node.args[2]
        mask_node = None
        if not isinstance(scatter_dim, int) or not isinstance(index_node, fx.Node):
            _record_skip(ctx, "no_meta", node_name)
            return None
    else:
        # pyrefly: ignore [bad-index]
        indices_pos, mask_pos = _SCATTER_ARGS[output_node.target]
        mask_node = output_node.args[mask_pos] if mask_pos is not None else None
        scatter_dim, index_node = _extract_scatter_dim_and_index(
            output_node.args[indices_pos]
        )
        if scatter_dim is None or index_node is None:
            _record_skip(ctx, "multi_index", node_name)
            return None

    input_meta = _get_tensor_meta(input_node)
    index_meta = _get_tensor_meta(index_node)
    if not input_meta or not index_meta:
        _record_skip(ctx, "no_meta", node_name)
        return None

    # Only rewrite accelerator ops: the expanded-buffer trade-off targets GPU
    # atomic contention and is a pessimization on CPU. Matches the non-CPU
    # device_filter that the memory estimator uses.
    if input_meta["device"].type == "cpu":
        _record_skip(ctx, "cpu_device", node_name)
        return None

    if input_meta["dtype"] == torch.bool or index_meta["dtype"] == torch.bool:
        _record_skip(ctx, "bool_dtype", node_name)
        return None

    ndim = len(input_meta["shape"])
    if scatter_dim < 0:
        scatter_dim += ndim
    if not 0 <= scatter_dim < ndim:
        _record_skip(ctx, "dim_out_of_bounds", node_name)
        return None

    output_size = _resolve_numel(input_meta["numel"])
    # index_put writes one row per index element; a values-shaped scatter index
    # only writes a slot once per position along scatter_dim.
    if is_scatter_reduce:
        if scatter_dim >= len(index_meta["shape"]):
            _record_skip(ctx, "dim_out_of_bounds", node_name)
            return None
        index_size = _resolve_numel(index_meta["shape"][scatter_dim])
    else:
        index_size = _resolve_numel(index_meta["numel"])

    if output_size is None or index_size is None:
        _record_skip(ctx, "dynamic_no_hint", node_name)
        return None

    if output_size == 0 or index_size == 0:
        _record_skip(ctx, "zero_size", node_name)
        return None

    # Contention is per scatter-dim slot, so use scatter_dim_size as denominator.
    # For a [vocab, dim] output, ratio = N/vocab, not N/(vocab*dim).
    scatter_dim_size = _resolve_numel(input_meta["shape"][scatter_dim])
    if scatter_dim_size is None or scatter_dim_size == 0:
        _record_skip(ctx, "zero_size", node_name)
        return None

    min_index_size: int = config.partitioned_scatter_min_index_size
    if not force and index_size < min_index_size:
        _record_skip(
            ctx,
            "index_too_small",
            node_name,
            f"index_size={index_size} min={min_index_size}",
        )
        return None

    contention_ratio = index_size / scatter_dim_size
    min_contention: float = config.partitioned_scatter_min_contention_ratio
    if not force and contention_ratio < min_contention:
        _record_skip(
            ctx,
            "low_contention",
            node_name,
            f"contention_ratio={contention_ratio:.3f} threshold={min_contention:.3f}",
        )
        return None

    acc_dtype = _accumulation_dtype(input_meta["dtype"])

    return ScatterCandidate(
        output_node=output_node,
        index_node=index_node,
        scatter_dim=scatter_dim,
        output_size=output_size,
        index_size=index_size,
        scatter_dim_size=scatter_dim_size,
        values_numel=index_size * (output_size // scatter_dim_size),
        contention_ratio=contention_ratio,
        dtype=input_meta["dtype"],
        acc_dtype=acc_dtype,
        # pyrefly: ignore [bad-argument-type]
        mask_node=mask_node,
    )


def _scan_candidates(graph: fx.Graph, ctx: ScatterPassContext) -> None:
    """
    Cheap pre-scan for accumulating scatter ops that could be rewritten.

    Applies every gate that does not require the memory profile (device, dtype,
    shape, min index size, contention ratio) so we can skip building the
    memory profile entirely when no op would qualify.
    """
    force: bool = config.partitioned_scatter_force

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target in _SCATTER_ARGS:
            # index_put takes accumulate as a trailing arg; the masked op is always
            # accumulating.
            if node.target in _INDEX_PUT_TARGETS and (
                len(node.args) < 4 or node.args[3] is not True
            ):
                continue
        elif node.target in _SCATTER_REDUCE_TARGETS:
            if not _is_summing_scatter(node):
                continue
        elif node.target is _INDEX_ADD_TARGET:
            # alpha scales the source; the rewrite adds it unscaled.
            if node.kwargs.get("alpha", 1) != 1:
                continue
        else:
            continue

        ctx.n_candidates += 1
        candidate = _evaluate_candidate(node, force, ctx)
        if candidate is not None:
            ctx.candidates[node] = candidate


def _build_scatter_memory_state(graph: fx.Graph) -> "ScatterMemoryState | None":
    """
    Build a per-node peak-memory profile of the original graph.
    Returns None when CUDA is unavailable or the profile can't be built; the
    pass then runs unconstrained by the memory budget.
    """
    if not torch.cuda.is_available():
        return None

    _, total_gpu = torch.cuda.mem_get_info()

    floor_bytes: int = config.partitioned_scatter_non_model_floor_bytes
    allowed_peak = max(0, total_gpu - floor_bytes)

    def is_releasable(n: fx.Node) -> bool:
        return not n.name.startswith("primals")

    # build_memory_profile emits two entries per compute node: the first after
    # its allocations (the node-local peak, while last-use inputs are still
    # live) and the second after its deallocations. We key on the allocation
    # entry so the expanded scatter buffer is charged against the real peak.
    # Unbacked symbolic sizes have no optimization_hint, so sizing the profile
    # can raise. Memory gating is a heuristic, so fall back to running the pass
    # unconstrained rather than failing the compile.
    try:
        profile = build_memory_profile(graph, is_releasable)
    except Exception as e:
        _artifact_log.debug(
            "partitioned_scatter: build_memory_profile failed (%s), "
            "running without memory gating",
            e,
        )
        return None

    compute_nodes = [
        n for n in graph.nodes if n.op not in ("placeholder", "get_attr", "output")
    ]
    node_to_idx: dict[fx.Node, int] = {}
    peak_mem: list[int] = []
    for i, node in enumerate(compute_nodes):
        alloc_idx = 1 + 2 * i
        if alloc_idx >= len(profile):
            break
        node_to_idx[node] = i
        peak_mem.append(profile[alloc_idx])

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
        peak_mem_by_node=peak_mem,
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
      2. Traffic cap (skipped when force=True): P <= writes_per_slot, where
         writes_per_slot = index_size / scatter_dim_size. The expanded buffer
         costs P * output_bytes to zero-fill plus the same to reduce, against the
         scatter's own index_size * row_bytes, so the overhead ratio is exactly
         P / writes_per_slot. Capping it at 1x also lands near the point where
         extra partitions stop being written at all.

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
        traffic_cap = max(min_p, 2 ** int(math.log2(max(1, writes_per_slot))))
        p = min(p, traffic_cap)

    return p


def _widen_bytes(candidate: ScatterCandidate) -> int:
    """Bytes the widened partials add outside the expanded buffer: the values copy
    in acc_dtype, plus the width the peak's own output copy cannot credit."""
    if candidate.acc_dtype == candidate.dtype:
        return 0

    values_bytes = candidate.values_numel * candidate.acc_dtype.itemsize
    uncredited = candidate.acc_dtype.itemsize - candidate.dtype.itemsize
    return values_bytes + candidate.output_size * uncredited


def _overhead_bytes(candidate: ScatterCandidate, num_partitions: int) -> int:
    """Bytes the rewrite adds on top of the output the peak already counts."""
    copies = max(0, num_partitions - 1)
    expanded = candidate.output_size * candidate.acc_dtype.itemsize * copies
    return expanded + _widen_bytes(candidate)


def _check_memory(
    state: ScatterMemoryState,
    candidate: ScatterCandidate,
    force: bool = False,
) -> int:
    """
    Compute num_partitions for this candidate given the peak-memory profile.
    Returns num_partitions >= min_p, or 0 if the memory constraint cannot be met.
    """
    min_p = config.partitioned_scatter_min_partitions
    max_p = config.partitioned_scatter_max_partitions

    idx = state.node_to_idx.get(candidate.output_node)
    if idx is None:
        # No profile entry means we cannot bound this scatter's peak, so fail
        # closed: granting max_p here would allocate max_p-1 extra copies of the
        # output with no memory check at all.
        return 0

    baseline = state.peak_mem_by_node[idx]
    # The widening is a fixed cost, so it comes off the budget before sizing P.
    charged = state.committed_overhead_bytes + _widen_bytes(candidate)
    available = state.allowed_peak_bytes - baseline - charged

    num_partitions = _compute_num_partitions(
        available,
        candidate.output_size,
        candidate.acc_dtype.itemsize,
        min_p,
        max_p,
        index_size=candidate.index_size,
        scatter_dim_size=candidate.scatter_dim_size,
        force=force,
    )

    if _artifact_log.isEnabledFor(logging.DEBUG):
        overhead = _overhead_bytes(candidate, num_partitions)
        _artifact_log.debug(
            "partitioned_scatter: memory check node=%s "
            "baseline_peak=%d MB committed=%d MB available=%d MB "
            "expanded_buffer_cost=%d MB num_partitions=%d "
            "total_gpu=%d MB floor=%d MB allowed_peak=%d MB",
            candidate.output_node.name,
            baseline // 1_000_000,
            state.committed_overhead_bytes // 1_000_000,
            available // 1_000_000,
            overhead // 1_000_000,
            num_partitions,
            state.total_gpu_bytes // 1_000_000,
            state.non_model_floor_bytes // 1_000_000,
            state.allowed_peak_bytes // 1_000_000,
        )

    return num_partitions


def _resolve_numel(numel: Any) -> int | None:
    """Resolve numel to a concrete int, handling SymInt via optimization_hint."""
    if isinstance(numel, torch.SymInt):
        hint = optimization_hint(numel)
        if hint is None:
            return None
        return hint * 2  # 2× safety margin for dynamic shapes
    return int(numel)


def _validate_memory(match: Match, ctx: ScatterPassContext, force: bool) -> bool:
    """
    Second-stage gate (the memory budget) for a matched scatter node.

    The cheap gates already ran in the pre-scan; here we only look up the
    surviving candidate and size the partition count against the memory budget.
    """
    output_node = match.output_node()
    candidate = ctx.candidates.get(output_node)
    if candidate is None:
        # Node failed a cheap gate in the pre-scan (already recorded).
        return False

    min_p: int = config.partitioned_scatter_min_partitions
    max_p: int = config.partitioned_scatter_max_partitions

    if ctx.memory is not None:
        num_partitions = _check_memory(ctx.memory, candidate, force=force)
    else:
        num_partitions = _compute_num_partitions(
            available_bytes=2**62,
            output_size=candidate.output_size,
            element_bytes=candidate.acc_dtype.itemsize,
            min_p=min_p,
            max_p=max_p,
            index_size=candidate.index_size,
            scatter_dim_size=candidate.scatter_dim_size,
            force=force,
        )

    if num_partitions < min_p:
        _record_skip(
            ctx,
            "memory_budget",
            output_node.name,
            f"num_partitions={num_partitions} min={min_p}",
        )
        return False

    match._num_partitions = num_partitions  # type: ignore[attr-defined]
    match._scatter_dim = candidate.scatter_dim  # type: ignore[attr-defined]
    match._index_node = candidate.index_node  # type: ignore[attr-defined]
    match._mask_node = candidate.mask_node  # type: ignore[attr-defined]
    match._overhead_bytes = _overhead_bytes(candidate, num_partitions)  # type: ignore[attr-defined]

    if _artifact_log.isEnabledFor(logging.DEBUG):
        _artifact_log.debug(
            "partitioned_scatter: APPLY node=%s "
            "num_partitions=%d scatter_dim=%d "
            "contention_ratio=%.1f (index_size=%d / scatter_dim_size=%d) "
            "output_size=%d dtype=%s%s",
            output_node.name,
            num_partitions,
            candidate.scatter_dim,
            candidate.contention_ratio,
            candidate.index_size,
            candidate.scatter_dim_size,
            candidate.output_size,
            candidate.dtype,
            " [force]" if force else "",
        )

    return True


def _as_dtype(tensor, dtype: torch.dtype):
    """Cast only when the dtype differs, to keep identity converts out of the graph."""
    if tensor.dtype == dtype:
        return tensor
    return torch.ops.prims.convert_element_type.default(tensor, dtype)


def _expanded_zeros(
    input_tensor, scatter_dim: int, num_partitions: int, acc_dtype: torch.dtype
):
    """Zero buffer holding one copy of the output per partition along scatter_dim."""
    expanded_shape = list(input_tensor.shape)
    expanded_shape[scatter_dim] *= num_partitions
    buffer = torch.ops.aten.full.default(
        expanded_shape,
        0,
        dtype=acc_dtype,
        layout=torch.strided,
        device=input_tensor.device,
        pin_memory=False,
    )
    return expanded_shape, buffer


def _sum_partitions(
    input_tensor,
    scattered_buffer,
    expanded_shape: list,
    scatter_dim: int,
    num_partitions: int,
    dim_size,
    dtype,
):
    """Split scatter_dim into [num_partitions, dim_size], sum it away, accumulate."""
    reduce_shape = list(expanded_shape)
    reduce_shape[scatter_dim] = num_partitions
    reduce_shape.insert(scatter_dim + 1, dim_size)
    reshaped = torch.ops.aten.view.default(scattered_buffer, reduce_shape)

    # Preserve dtype for integer types that don't promote during sum
    if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
        reduced = torch.ops.aten.sum.dim_IntList(reshaped, [scatter_dim], dtype=dtype)
    else:
        reduced = torch.ops.aten.sum.dim_IntList(reshaped, [scatter_dim])

    # Partials are wider than the output, so round only after summing them.
    if reduced.dtype != dtype:
        widened = _as_dtype(input_tensor, reduced.dtype)
        return _as_dtype(widened + reduced, dtype)

    return input_tensor + reduced


def _commit(match: Match, ctx: ScatterPassContext, num_partitions: int) -> None:
    """Charge this scatter's overhead so later candidates in the same invocation
    see a correspondingly smaller budget."""
    if ctx.memory is not None:
        ctx.memory.committed_overhead_bytes += match._overhead_bytes  # type: ignore[attr-defined]

    ctx.n_applied += 1
    ctx.applied_partitions.append(num_partitions)
    counters["inductor"]["partitioned_scatter_applied"] += 1


def _create_replacement(
    match: Match, ctx: ScatterPassContext, input_tensor, indices, values, mask=None
) -> None:
    """Replace high-contention index_put with partitioned scatter."""
    num_partitions: int = match._num_partitions  # type: ignore[attr-defined]
    scatter_dim: int = match._scatter_dim  # type: ignore[attr-defined]
    index_node = match._index_node  # type: ignore[attr-defined]

    def scatter(input_tensor, index_node, values, mask):
        dim_size = input_tensor.shape[scatter_dim]
        num_operations = index_node.numel()

        # Flatten multi-dimensional indices to 1-D
        if len(index_node.shape) > 1:
            flat_index = index_node.reshape(num_operations)
            values_ndim = len(index_node.shape)
            flat_values = values.reshape(
                [num_operations] + list(values.shape[values_ndim:])
            )
            if mask is not None:
                # Mask shares the index's leading dims, so flatten it the same way.
                mask = mask.reshape([num_operations] + list(mask.shape[values_ndim:]))
        else:
            flat_index = index_node
            flat_values = values

        # index_add takes an int32 index, which the offset below can push out of
        # range, and the iota that builds that offset follows this dtype.
        flat_index = _as_dtype(flat_index, torch.int64)

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

        acc_dtype = _accumulation_dtype(input_tensor.dtype)
        expanded_shape, expanded_buffer = _expanded_zeros(
            input_tensor, scatter_dim, num_partitions, acc_dtype
        )
        flat_values = _as_dtype(flat_values, acc_dtype)

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

        if mask is None:
            scattered_buffer = torch.ops.aten.index_put.default(
                expanded_buffer, adjusted_indices, flat_values, True
            )
        else:
            # Keep the masked op rather than folding the mask into the values: it
            # clamps the out-of-range indices that masked-off lanes may carry.
            scattered_buffer = _MASKED_INDEX_PUT_TARGET(
                expanded_buffer, mask, adjusted_indices, flat_values
            )

        return _sum_partitions(
            input_tensor,
            scattered_buffer,
            expanded_shape,
            scatter_dim,
            num_partitions,
            dim_size,
            input_tensor.dtype,
        )

    mask_node = match._mask_node  # type: ignore[attr-defined]
    if mask_node is None:
        # replace_by_example traces by arity, so the mask cannot just default.
        def repl(input_tensor, index_node, values):  # type: ignore[misc]
            return scatter(input_tensor, index_node, values, None)

        example_args = [input_tensor, index_node, values]
    else:
        repl = scatter  # type: ignore[assignment]
        example_args = [input_tensor, index_node, values, mask_node]

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, example_args)
    _commit(match, ctx, num_partitions)


def _create_scatter_reduce_replacement(
    match: Match, ctx: ScatterPassContext, input_tensor, values
) -> None:
    """Replace a high-contention scatter_add / scatter_reduce(sum)."""
    num_partitions: int = match._num_partitions  # type: ignore[attr-defined]
    scatter_dim: int = match._scatter_dim  # type: ignore[attr-defined]
    index_node = match._index_node  # type: ignore[attr-defined]

    def repl(input_tensor, index, values):
        dim_size = input_tensor.shape[scatter_dim]
        # scatter_add takes an int32 index too; same widening as the index_put path.
        index = _as_dtype(index, torch.int64)

        # Writes only collide if they differ along scatter_dim, so partition on
        # that position; a contiguous slice of values then stays in one partition.
        num_operations = index.shape[scatter_dim]
        operation_ids = torch.ops.prims.iota.default(
            num_operations,
            start=0,
            step=1,
            dtype=index.dtype,
            device=index.device,
            requires_grad=False,
        )
        partition_ids = torch.ops.aten.bitwise_and.Scalar(
            operation_ids, num_partitions - 1
        )
        broadcast_shape = [1] * len(index.shape)
        broadcast_shape[scatter_dim] = num_operations
        partition_offsets = (
            torch.ops.aten.view.default(partition_ids, broadcast_shape) * dim_size
        )
        adjusted_index = index + partition_offsets

        acc_dtype = _accumulation_dtype(input_tensor.dtype)
        expanded_shape, expanded_buffer = _expanded_zeros(
            input_tensor, scatter_dim, num_partitions, acc_dtype
        )
        values = _as_dtype(values, acc_dtype)
        # Summing onto a zero buffer, so scatter_add serves all three source ops.
        scattered_buffer = torch.ops.aten.scatter_add.default(
            expanded_buffer, scatter_dim, adjusted_index, values
        )

        return _sum_partitions(
            input_tensor,
            scattered_buffer,
            expanded_shape,
            scatter_dim,
            num_partitions,
            dim_size,
            input_tensor.dtype,
        )

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, [input_tensor, index_node, values])
    _commit(match, ctx, num_partitions)


def _build_pattern_pass(ctx: ScatterPassContext) -> PatternMatcherPass:
    """
    Construct a per-invocation PatternMatcherPass whose callbacks close over
    `ctx`, avoiding module-level global state (which is unsafe under concurrent
    compilation).
    """
    patterns = PatternMatcherPass(pass_name="partitioned_scatter_optimization")
    force: bool = config.partitioned_scatter_force

    def extra_check(match: Match) -> bool:
        return _validate_memory(match, ctx, force)

    def replacement(match: Match, input_tensor, indices, values) -> None:
        _create_replacement(match, ctx, input_tensor, indices, values)

    def masked_replacement(match: Match, input_tensor, mask, indices, values) -> None:
        _create_replacement(match, ctx, input_tensor, indices, values, mask=mask)

    for target in _INDEX_PUT_TARGETS:
        register_graph_pattern(
            CallFunction(target, Arg(), Arg(), Arg(), True),
            extra_check=extra_check,
            pass_dict=patterns,  # type: ignore[arg-type]
        )(replacement)

    # No trailing True to match: this op is always accumulating.
    register_graph_pattern(
        CallFunction(_MASKED_INDEX_PUT_TARGET, Arg(), Arg(), Arg(), Arg()),
        extra_check=extra_check,
        pass_dict=patterns,  # type: ignore[arg-type]
    )(masked_replacement)

    def index_add_replacement(match: Match, input_tensor, values) -> None:
        # index_add is index_put accumulate with the 1-D index at dim.
        scatter_dim: int = match._scatter_dim  # type: ignore[attr-defined]
        index_node = match._index_node  # type: ignore[attr-defined]
        indices = [None] * scatter_dim + [index_node]
        _create_replacement(match, ctx, input_tensor, indices, values)

    register_graph_pattern(
        CallFunction(_INDEX_ADD_TARGET, Arg(), Ignored(), Ignored(), Arg()),
        extra_check=extra_check,
        pass_dict=patterns,  # type: ignore[arg-type]
    )(index_add_replacement)

    def scatter_reduce_replacement(match: Match, input_tensor, values) -> None:
        _create_scatter_reduce_replacement(match, ctx, input_tensor, values)

    # dim is Ignored() because the replacement needs the normalized dim off the
    # candidate; reduce/include_self were already vetted by _is_summing_scatter.
    for scatter_pattern in (
        CallFunction(aten.scatter_add.default, Arg(), Ignored(), Ignored(), Arg()),
        CallFunction(
            aten.scatter_reduce.two, Arg(), Ignored(), Ignored(), Arg(), Ignored()
        ),
        CallFunction(
            aten.scatter.reduce, Arg(), Ignored(), Ignored(), Arg(), reduce=Ignored()
        ),
    ):
        register_graph_pattern(
            scatter_pattern,
            extra_check=extra_check,
            pass_dict=patterns,  # type: ignore[arg-type]
        )(scatter_reduce_replacement)

    return patterns


def _log_summary(ctx: ScatterPassContext, num_matches: int) -> None:
    if ctx.n_candidates == 0:
        return

    if ctx.memory is not None:
        log.info(
            "partitioned_scatter: candidates=%d applied=%d skipped=%d "
            "partitions_per_op=%s "
            "skip_breakdown=%s "
            "total_gpu=%d MB floor=%d MB allowed_peak=%d MB",
            ctx.n_candidates,
            ctx.n_applied,
            ctx.n_candidates - ctx.n_applied,
            ctx.applied_partitions,
            dict(ctx.skip_reasons),
            ctx.memory.total_gpu_bytes // 1_000_000,
            ctx.memory.non_model_floor_bytes // 1_000_000,
            ctx.memory.allowed_peak_bytes // 1_000_000,
        )
    else:
        log.info(
            "partitioned_scatter: candidates=%d applied=%d skipped=%d "
            "partitions_per_op=%s skip_breakdown=%s (no memory state)",
            ctx.n_candidates,
            ctx.n_applied,
            ctx.n_candidates - ctx.n_applied,
            ctx.applied_partitions,
            dict(ctx.skip_reasons),
        )


def partitioned_scatter_optimization_pass(graph: fx.Graph) -> fx.Graph:
    """
    Apply partitioned scatter optimization to high-contention index_put operations.
    Controlled by config.partitioned_scatter_enabled.
    """
    if not config.partitioned_scatter_enabled:
        return graph

    ctx = ScatterPassContext()

    # Stage 1: cheap pre-scan. If nothing qualifies we avoid building the
    # (relatively expensive) memory profile entirely.
    _scan_candidates(graph, ctx)
    if not ctx.candidates:
        _log_summary(ctx, 0)
        return graph

    # Stage 2: build the memory profile and run the pattern matcher.
    ctx.memory = _build_scatter_memory_state(graph)
    patterns = _build_pattern_pass(ctx)
    num_matches = patterns.apply(graph)

    _log_summary(ctx, num_matches)

    if num_matches > 0:
        graph.lint()

    return graph


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
