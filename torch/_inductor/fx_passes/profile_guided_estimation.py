"""
Profile-Guided Estimation (PGE) for overlap scheduling.

Parses a Chrome Trace JSON (from torch.profiler) and builds lookup tables
for kernel runtimes (collectives, matmuls, attention, custom ops, etc.).
Used as a custom_runtime_estimation hook in the overlap scheduler.

When the same profile is loaded on all ranks, estimates are deterministic
and no cross-rank synchronization is needed.
"""

from __future__ import annotations

import functools
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx as fx
from torch._inductor.analysis.profile_analysis import (
    _create_extern_mapping,
    _dtype_map,
    _get_size_from_string,
    ParseException,
)
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor,
    is_all_reduce_tensor,
    is_all_to_all_tensor,
    is_reduce_scatter_tensor,
)
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def _rank_stride(ranks: tuple[int, ...]) -> int | None:
    """Compute the stride of a sorted rank tuple, or None if non-uniform.

    Examples:
        (0, 2, 4, 6) → stride 2
        (0, 1)       → stride 1
        (1, 3, 5, 7) → stride 2
        (0, 1, 4, 5) → None (non-uniform)
    """
    if len(ranks) <= 1:
        return None
    stride = ranks[1] - ranks[0]
    if stride <= 0:
        return None
    for i in range(2, len(ranks)):
        if ranks[i] - ranks[i - 1] != stride:
            return None
    return stride


@dataclass
class CollectiveRecord:
    """A single collective kernel observation from the profile."""

    collective_name: str  # "all_gather_into_tensor", "reduce_scatter_tensor", etc.
    pg_ranks: tuple[int, ...]  # sorted rank tuple
    group_size: int
    in_nelems: int  # "In msg nelems" from profile
    out_nelems: int  # "Out msg nelems" from profile
    dtype: str  # "Float", "BFloat16", etc.
    duration_us: float


@dataclass
class OpRecord:
    """A single op observation from the profile (any CPU op with GPU kernels)."""

    op_name: str  # normalized name, e.g. "aten::mm", "mylib::my_custom_op"
    input_shapes: tuple[tuple[int, ...], ...]
    input_strides: tuple[tuple[int, ...], ...]
    dtype: torch.dtype | None
    duration_us: float  # sum of all GPU kernels for this CPU op


def _to_nested_tuple(x: Any) -> Any:
    """Recursively convert nested lists to tuples for hashability."""
    if isinstance(x, (list, tuple)):
        return tuple(_to_nested_tuple(i) for i in x)
    return x


@dataclass
class ProfileData:
    """Parse Chrome Trace JSON and build lookup tables for kernel runtimes."""

    collectives: list[CollectiveRecord] = field(default_factory=list)
    ops: list[OpRecord] = field(default_factory=list)
    pg_configs: dict[str, tuple[int, ...]] = field(default_factory=dict)

    # Lookup indices built after loading
    _collective_index: dict[
        tuple[str, tuple[int, ...], str], list[tuple[int, float]]
    ] = field(default_factory=dict)
    # Fallback index by mesh dimension (name, stride, group_size, dtype).
    # Matches PGs belonging to the same mesh dimension regardless of specific ranks.
    # E.g. (0,2,4,6) and (1,3,5,7) both have stride=2, size=4 → same mesh dim.
    _collective_index_by_mesh_dim: dict[
        tuple[str, int, int, str], list[tuple[int, float]]
    ] = field(default_factory=dict)
    # Count of distinct PGs per mesh dimension (stride, group_size) — used for
    # ambiguity check (skip fallback if multiple PGs share the same mesh dim).
    _pg_count_by_mesh_dim: dict[tuple[int, int], int] = field(default_factory=dict)
    # Generic op index: (op_name, input_shapes, input_strides, dtype) -> avg_dur_us
    _op_index: dict[
        tuple[
            str,
            tuple[tuple[int, ...], ...],
            tuple[tuple[int, ...], ...],
            torch.dtype | None,
        ],
        float,
    ] = field(default_factory=dict)
    # Peak observed bandwidth per PG (GB/s), computed from largest messages
    _pg_peak_bw: dict[tuple[int, ...], float] = field(default_factory=dict)
    # Mesh-dimension fallback: (stride, group_size) -> peak BW (GB/s)
    _mesh_dim_peak_bw: dict[tuple[int, int], float] = field(default_factory=dict)

    def load(self, trace_path: str) -> None:
        """Load and parse a Chrome Trace JSON file."""
        import os

        if not os.path.isfile(trace_path):
            raise FileNotFoundError(
                f"PGE trace file not found: {trace_path}. "
                f"Check config.aten_distributed_optimizations.profile_guided_estimations_profile_path"
            )
        with open(trace_path) as f:
            data = json.load(f)

        self._parse_pg_configs(data)
        self._parse_events(data)
        self._build_indices()

        log.info(
            "PGE loaded: %d collectives, %d op records (%d distinct shapes), %d PGs",
            len(self.collectives),
            len(self.ops),
            len(self._op_index),
            len(self.pg_configs),
        )

    def _parse_pg_configs(self, data: dict[str, Any]) -> None:
        dist_info = data.get("distributedInfo", {})
        pg_config = dist_info.get("pg_config", {})
        # pg_config can be a list of dicts or a dict of dicts
        if isinstance(pg_config, list):
            for pg_info in pg_config:
                pg_name = str(pg_info.get("pg_name", ""))
                ranks = pg_info.get("ranks", [])
                if ranks:
                    self.pg_configs[pg_name] = tuple(sorted(ranks))
        elif isinstance(pg_config, dict):
            for pg_name, pg_info in pg_config.items():
                ranks = pg_info.get("ranks", [])
                if ranks:
                    self.pg_configs[pg_name] = tuple(sorted(ranks))

    def _parse_events(self, data: dict[str, Any]) -> None:
        events = data.get("traceEvents", [])
        # Reuse profile_analysis's External id -> CPU op mapping
        try:
            extern_mapping = _create_extern_mapping(data)
        except (ParseException, KeyError):
            # Malformed trace (e.g. duplicate External ids, missing traceEvents)
            extern_mapping = defaultdict(list)
            for ev in events:
                if (
                    isinstance(ev, dict)
                    and ev.get("cat") == "cpu_op"
                    and "args" in ev
                    and "External id" in ev["args"]
                ):
                    extern_mapping[ev["args"]["External id"]].append(ev)

        # Build External id -> total GPU kernel duration
        gpu_dur: dict[int, float] = defaultdict(float)
        for ev in events:
            if not isinstance(ev, dict) or ev.get("cat") != "kernel":
                continue
            args = ev.get("args", {})
            eid = args.get("External id")
            dur = ev.get("dur", 0.0)
            if eid is not None and dur > 0:
                gpu_dur[eid] += dur

        # Parse collectives from GPU kernel events directly
        # (NCCL kernels carry collective metadata in args)
        for ev in events:
            if not isinstance(ev, dict) or ev.get("cat") != "kernel":
                continue
            args = ev.get("args", {})
            coll_name = args.get("Collective name")
            if coll_name is None:
                continue
            pg_name = args.get("Process Group Name", "")
            pg_ranks_str = args.get("Process Group Ranks", "")
            group_size = args.get("Group size", 0)
            in_nelems = args.get("In msg nelems", 0)
            out_nelems = args.get("Out msg nelems", 0)
            dtype = args.get("dtype", "")
            dur = ev.get("dur", 0.0)
            if dur <= 0:
                continue

            pg_ranks = self._parse_ranks(pg_ranks_str, pg_name)

            self.collectives.append(
                CollectiveRecord(
                    collective_name=coll_name,
                    pg_ranks=pg_ranks,
                    group_size=group_size,
                    in_nelems=in_nelems,
                    out_nelems=out_nelems,
                    dtype=dtype,
                    duration_us=dur,
                )
            )

        # Parse all CPU ops that have associated GPU kernels
        for eid, cpu_evs in extern_mapping.items():
            if not cpu_evs:
                continue
            total_dur = gpu_dur.get(eid, 0.0)
            if total_dur <= 0:
                continue
            cpu_ev = cpu_evs[0]
            self._parse_op(cpu_ev.get("name", ""), cpu_ev.get("args", {}), total_dur)

    def _parse_ranks(self, ranks_str: str, pg_name: str) -> tuple[int, ...]:
        """Parse rank list from profile string or fall back to pg_configs."""
        if isinstance(ranks_str, str) and ranks_str.startswith("["):
            try:
                ranks = json.loads(ranks_str)
                return tuple(sorted(ranks))
            except (json.JSONDecodeError, TypeError):
                pass
        # Fall back to pg_configs
        if pg_name in self.pg_configs:
            return self.pg_configs[pg_name]
        return ()

    def _parse_op(self, name: str, args: dict[str, Any], total_dur: float) -> None:
        """Parse any CPU op into a generic OpRecord."""
        input_dims = args.get("Input Dims", [])
        input_strides = args.get("Input Strides", [])
        input_types = args.get("Input type", [])
        if not input_dims:
            return
        dtype_str = input_types[0] if input_types else ""
        dtype = _dtype_map.get(dtype_str)
        # Skip empty entries (non-tensor args like scalars/None) so the
        # tuples match what _get_node_input_shapes/strides extract from FX nodes.
        shapes = tuple(
            _to_nested_tuple(d)
            for d in input_dims
            if isinstance(d, (list, tuple)) and d
        )
        strides = tuple(
            _to_nested_tuple(d)
            for d in input_strides
            if isinstance(d, (list, tuple)) and d
        )
        if not shapes:
            return
        self.ops.append(
            OpRecord(
                op_name=name,
                input_shapes=shapes,
                input_strides=strides,
                dtype=dtype,
                duration_us=total_dur,
            )
        )

    def _build_indices(self) -> None:
        """Build lookup indices from parsed records."""
        coll_idx: dict[tuple[str, tuple[int, ...], str], list[tuple[int, float]]] = (
            defaultdict(list)
        )
        coll_idx_by_mesh_dim: dict[
            tuple[str, int, int, str], list[tuple[int, float]]
        ] = defaultdict(list)
        # Track distinct PG rank sets per mesh dimension for ambiguity check
        pg_sets_by_mesh_dim: dict[tuple[int, int], OrderedSet[tuple[int, ...]]] = (
            defaultdict(OrderedSet)
        )
        for rec in self.collectives:
            norm_name = self._normalize_collective_name(rec.collective_name)
            gs = len(rec.pg_ranks) if rec.pg_ranks else rec.group_size
            coll_idx[(norm_name, rec.pg_ranks, rec.dtype)].append(
                (rec.out_nelems, rec.duration_us)
            )
            stride = _rank_stride(rec.pg_ranks)
            if stride is not None:
                coll_idx_by_mesh_dim[(norm_name, stride, gs, rec.dtype)].append(
                    (rec.out_nelems, rec.duration_us)
                )
                pg_sets_by_mesh_dim[(stride, gs)].add(rec.pg_ranks)
        # Sort by nelems for interpolation
        self._collective_index = {
            k: sorted(v, key=lambda x: x[0]) for k, v in coll_idx.items()
        }
        self._collective_index_by_mesh_dim = {
            k: sorted(v, key=lambda x: x[0]) for k, v in coll_idx_by_mesh_dim.items()
        }
        self._pg_count_by_mesh_dim = {
            k: len(pgs) for k, pgs in pg_sets_by_mesh_dim.items()
        }

        op_groups: defaultdict[
            tuple[
                str,
                tuple[tuple[int, ...], ...],
                tuple[tuple[int, ...], ...],
                torch.dtype | None,
            ],
            list[float],
        ] = defaultdict(list)
        for rec in self.ops:
            key = (rec.op_name, rec.input_shapes, rec.input_strides, rec.dtype)
            op_groups[key].append(rec.duration_us)
        self._op_index = {k: sum(v) / len(v) for k, v in op_groups.items()}

        # Per-PG peak bandwidth: compute bytes/us for each collective observation,
        # then take the max from the top-N largest messages per PG (where bandwidth
        # is most representative of hardware speed, not dominated by startup latency).
        # Uses output-convention bytes (matching _estimate_with_pg_bandwidth).
        _TOP_N = 5  # consider top N largest messages for peak BW
        pg_bw_samples: dict[tuple[int, ...], list[tuple[int, float]]] = defaultdict(
            list
        )
        mesh_dim_bw_samples: dict[tuple[int, int], list[tuple[int, float]]] = (
            defaultdict(list)
        )
        for rec in self.collectives:
            if rec.out_nelems <= 0 or rec.duration_us <= 0:
                continue
            gs = len(rec.pg_ranks) if rec.pg_ranks else rec.group_size
            elem_bytes = self._dtype_elem_bytes(rec.dtype)
            total_bytes = rec.out_nelems * elem_bytes
            bw_gbps = total_bytes / (rec.duration_us * 1e-6) / 1e9  # GB/s
            pg_bw_samples[rec.pg_ranks].append((total_bytes, bw_gbps))
            stride = _rank_stride(rec.pg_ranks)
            if stride is not None:
                mesh_dim_bw_samples[(stride, gs)].append((total_bytes, bw_gbps))

        def _peak_bw_from_samples(
            samples: list[tuple[int, float]],
        ) -> float:
            """Get peak BW from the top-N largest messages."""
            # Sort by message size descending, take top N, return max BW
            sorted_samples = sorted(samples, key=lambda x: x[0], reverse=True)
            top = sorted_samples[:_TOP_N]
            return max(bw for _, bw in top) if top else 0.0

        self._pg_peak_bw = {
            pg: _peak_bw_from_samples(samples)
            for pg, samples in pg_bw_samples.items()
            if samples
        }
        self._mesh_dim_peak_bw = {
            key: _peak_bw_from_samples(samples)
            for key, samples in mesh_dim_bw_samples.items()
            if samples
        }

    def get_collective_keys(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Return the collective index keys: (name, pg_ranks, dtype)."""
        return list(self._collective_index.keys())

    @property
    def op_count(self) -> int:
        """Number of distinct op shapes in the index."""
        return len(self._op_index)

    def get_op_names(self) -> list[str]:
        """Return distinct op names in the op index."""
        return list(OrderedSet(name for name, _, _, _ in self._op_index))

    @staticmethod
    def _dtype_elem_bytes(dtype: str) -> int:
        """Return bytes per element for a dtype string (NCCL CamelCase or TypeMeta)."""
        return _get_size_from_string(dtype.lower())

    @staticmethod
    def _normalize_collective_name(name: str) -> str:
        """Normalize collective name between profile and FX conventions.

        Profile uses: _allgather_base, allreduce, reduce_scatter_tensor_coalesced
        FX uses: all_gather_into_tensor, all_reduce, reduce_scatter_tensor
        """
        n = name.lower()
        if "allgather" in n or "all_gather" in n:
            return "all_gather"
        if "reduce_scatter" in n:
            return "reduce_scatter"
        if "allreduce" in n or "all_reduce" in n:
            return "all_reduce"
        if "all_to_all" in n or "alltoall" in n:
            return "all_to_all"
        return name

    # Maximum ratio of target_nelems / max_observed before switching from
    # log-log extrapolation to bandwidth-based estimation.
    EXTRAPOLATION_CAP = 2.0

    def _estimate_with_pg_bandwidth(
        self,
        pg_ranks: tuple[int, ...],
        nelems: int,
        dtype: str,
    ) -> float | None:
        """Estimate collective duration using peak observed bandwidth for this PG.

        Used when the target size exceeds the extrapolation cap. Returns ms or None.
        """
        bw_gbps = self._pg_peak_bw.get(pg_ranks)
        if bw_gbps is None or bw_gbps <= 0:
            # Try mesh-dimension fallback
            stride = _rank_stride(pg_ranks)
            gs = len(pg_ranks)
            if stride is not None:
                bw_gbps = self._mesh_dim_peak_bw.get((stride, gs))
        if bw_gbps is None or bw_gbps <= 0:
            return None  # fall through to analytical
        elem_bytes = self._dtype_elem_bytes(dtype)
        total_bytes = nelems * elem_bytes
        dur_ms = total_bytes / (bw_gbps * 1e6)  # GB/s → bytes/ms = 1e6
        return dur_ms

    def lookup_collective(
        self,
        collective_name: str,
        pg_ranks: tuple[int, ...],
        nelems: int,
        dtype: str,
    ) -> tuple[float, str] | None:
        """Look up collective duration in ms. Returns (duration_ms, source) or None.

        ``source`` is ``"profile"`` for exact/interpolated matches, or
        ``"pg_bandwidth"`` when bandwidth-based extrapolation was used.

        Tries exact rank match first, then falls back to mesh-dimension match
        (e.g. (0,2,4,6) and (1,3,5,7) both have stride=2, size=4 → same mesh dim).

        When the target size exceeds EXTRAPOLATION_CAP * max_observed, uses
        bandwidth-based estimation from peak observed bandwidth instead of
        linear extrapolation (which overestimates for large messages).
        """
        norm_name = self._normalize_collective_name(collective_name)
        # Try exact rank match first
        key = (norm_name, pg_ranks, dtype)
        entries = self._collective_index.get(key)
        if not entries:
            # Fall back to mesh-dimension match
            gs = len(pg_ranks)
            stride = _rank_stride(pg_ranks)
            if (
                stride is not None
                and self._pg_count_by_mesh_dim.get((stride, gs), 0) == 1
            ):
                mesh_dim_key = (norm_name, stride, gs, dtype)
                entries = self._collective_index_by_mesh_dim.get(mesh_dim_key)
            if not entries:
                return None

        # Exact match
        for n, dur in entries:
            if n == nelems:
                return (dur / 1e3, "profile")  # us -> ms

        # Check extrapolation distance: if target is far beyond observed range,
        # use bandwidth-based model instead of log-log extrapolation
        max_observed = max((n for n, _ in entries if n > 0), default=0)
        if max_observed > 0 and nelems > max_observed * self.EXTRAPOLATION_CAP:
            est = self._estimate_with_pg_bandwidth(pg_ranks, nelems, dtype)
            if est is not None:
                return (est, "pg_bandwidth")
            # Fall through to log-log if no BW data available

        # Interpolation in log-log space
        result = self._interpolate_log_log(entries, nelems)
        if result is not None:
            return (result, "profile")
        return None

    def _interpolate_log_log(
        self, entries: list[tuple[int, float]], target_nelems: int
    ) -> float | None:
        """Interpolate duration in log-log space (log(nelems) vs log(dur))."""
        if not entries or target_nelems <= 0:
            return None

        log_target = math.log(target_nelems)

        # Find bracketing entries
        lower: tuple[int, float] | None = None
        upper: tuple[int, float] | None = None
        for n, dur in entries:
            if n <= 0 or dur <= 0:
                continue
            if n <= target_nelems:
                lower = (n, dur)
            if n >= target_nelems and upper is None:
                upper = (n, dur)

        if lower is not None and upper is not None:
            log_n0, log_d0 = math.log(lower[0]), math.log(lower[1])
            log_n1, log_d1 = math.log(upper[0]), math.log(upper[1])
            if log_n1 == log_n0:
                return lower[1] / 1e3
            t = (log_target - log_n0) / (log_n1 - log_n0)
            log_dur = log_d0 + t * (log_d1 - log_d0)
            return math.exp(log_dur) / 1e3  # us -> ms
        elif lower is not None:
            # Linear extrapolation (not log-log) from nearest lower;
            # EXTRAPOLATION_CAP in lookup_collective limits how far this reaches.
            return (lower[1] * target_nelems / lower[0]) / 1e3
        elif upper is not None:
            # Linear extrapolation from nearest upper
            return (upper[1] * target_nelems / upper[0]) / 1e3

        return None

    def lookup_op(
        self,
        op_name: str,
        input_shapes: tuple[tuple[int, ...], ...],
        input_strides: tuple[tuple[int, ...], ...],
        dtype: torch.dtype | None,
    ) -> float | None:
        """Look up op duration in ms by exact shape+stride match. Returns None on miss."""
        key = (op_name, input_shapes, input_strides, dtype)
        dur_us = self._op_index.get(key)
        if dur_us is not None:
            return dur_us / 1e3  # us -> ms
        return None


@functools.cache
def _dtype_to_nccl_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to NCCL/ScalarType name (for collective matching).

    Derives the name from torch.Tensor.type() which returns e.g.
    "torch.BFloat16Tensor" -> "BFloat16".
    """
    return (
        torch.tensor([], dtype=dtype)
        .type()
        .removeprefix("torch.")
        .removesuffix("Tensor")
    )


def _get_node_dtype(node: fx.Node) -> torch.dtype | None:
    """Extract dtype from FX node metadata."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return val.dtype
    if isinstance(val, (list, tuple)) and val:
        first = val[0]
        if isinstance(first, torch.Tensor):
            return first.dtype
    return None


def _fx_target_to_profile_name(node: fx.Node) -> str | None:
    """Convert FX node target to the profile op name format.

    FX: torch.ops.aten.mm.default → "aten::mm"
    FX: torch.ops.deepep.dispatch.default → "deepep::dispatch"
    """
    target = node.target
    if isinstance(target, torch._ops.OpOverload):
        # e.g. "aten::mm" from torch.ops.aten.mm.default
        ns = target.namespace
        op_name = target._schema.name.split("::")[-1]
        return f"{ns}::{op_name}"
    if hasattr(target, "__name__"):
        return target.__name__
    return None


def _get_node_input_shapes_and_strides(
    node: fx.Node,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]] | None:
    """Extract input shapes and strides from FX node tensor args.

    Returns (shapes, strides) or None if no tensor args or symbolic dims.
    """
    from torch._inductor.fx_passes.node_runtime_estimation import get_hint

    shapes: list[tuple[int, ...]] = []
    strides: list[tuple[int, ...]] = []
    for arg in node.args:
        if not isinstance(arg, fx.Node):
            continue
        val = arg.meta.get("val")
        if isinstance(val, torch.Tensor):
            resolved_shape = []
            for s in val.shape:
                h = get_hint(s)
                if h is None:
                    return None
                resolved_shape.append(h)
            resolved_stride = []
            for s in val.stride():
                h = get_hint(s)
                if h is None:
                    return None
                resolved_stride.append(h)
            shapes.append(tuple(resolved_shape))
            strides.append(tuple(resolved_stride))
    if not shapes:
        return None
    return tuple(shapes), tuple(strides)


def _is_collective_node(node: fx.Node) -> bool:
    """Check if node is a collective communication op."""
    return (
        is_all_gather_into_tensor(node)
        or is_reduce_scatter_tensor(node)
        or is_all_reduce_tensor(node)
        or is_all_to_all_tensor(node)
    )


def _get_collective_info(
    node: fx.Node,
) -> tuple[str, tuple[int, ...], int, str] | None:
    """Extract (collective_name, pg_ranks, nelems, dtype) from collective node."""
    import torch.distributed as c10d
    from torch.fx.operator_schemas import normalize_function

    if not c10d.is_initialized():
        return None

    target = node.target
    if not isinstance(target, torch._ops.OpOverload):
        return None
    collective_name = target.name().split("::")[-1].split(".")[0]

    opt = normalize_function(
        target,
        args=node.args,
        kwargs=node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if opt is None:
        return None
    _, kwargs = opt
    group_name = kwargs.get("group_name", "")

    try:
        from torch.distributed.distributed_c10d import (
            _resolve_process_group,
            get_process_group_ranks,
        )

        pg = _resolve_process_group(group_name)
        pg_ranks = tuple(sorted(get_process_group_ranks(pg)))
    except (RuntimeError, KeyError, ValueError):
        log.debug(
            "PGE: failed to resolve process group for %s", node.name, exc_info=True
        )
        return None

    # Get nelems from input tensor
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        nelems = 1
        for s in val.shape:
            nelems *= int(s)
        dtype = _dtype_to_nccl_str(val.dtype)
    else:
        # Try first arg
        if node.args and isinstance(node.args[0], fx.Node):
            inp_val = node.args[0].meta.get("val")
            if isinstance(inp_val, torch.Tensor):
                nelems = 1
                for s in inp_val.shape:
                    nelems *= int(s)
                dtype = _dtype_to_nccl_str(inp_val.dtype)
            else:
                return None
        else:
            return None

    return (collective_name, pg_ranks, nelems, dtype)


class ProfileGuidedEstimator:
    """Profile-guided runtime estimator for FX nodes.

    Implements the ``custom_runtime_estimation`` interface:
    ``(fx.Node, int | None) -> float | None`` (returns ms or None for fallback).

    Handles collectives via interpolation (latency + bandwidth model) and all
    other ops (matmul, SDPA, custom ops, etc.) via exact shape match from the
    profile trace.
    """

    def __init__(
        self,
        trace_path: str,
        diagnostics_gm: torch.fx.GraphModule | None = None,
    ) -> None:
        self.profile = ProfileData()
        self.profile.load(trace_path)
        self._log_profile_vs_analytical_comparison(diagnostics_gm)

    def _log_profile_vs_analytical_comparison(
        self, diagnostics_gm: torch.fx.GraphModule | None
    ) -> None:
        """Log profile data and PGE vs analytical comparison to trace_structured.

        Logs all profile entries (collectives, ops with durations).
        If diagnostics_gm is provided, walks the graph and compares PGE
        estimates with analytical (roofline / NCCL) for each matched node.
        """
        profile = self.profile
        op_entries = [
            {
                "op": op_name,
                "shapes": [list(s) for s in shapes],
                "strides": [list(s) for s in strides],
                "dtype": str(dtype) if dtype is not None else "",
                "profile_ms": dur_us / 1e3,
            }
            for (op_name, shapes, strides, dtype), dur_us in profile._op_index.items()
        ]

        diagnostics: list[dict[str, Any]] = []
        if diagnostics_gm is not None:
            from torch._inductor.fx_passes.overlap_scheduling import (
                estimate_roofline_runtime_ms,
            )

            for node in diagnostics_gm.graph.nodes:
                pge_est = self(node)
                if pge_est is None:
                    continue
                entry: dict[str, Any] = {
                    "node": node.name,
                    "op": str(node.target),
                    "pge_ms": pge_est,
                }
                if _is_collective_node(node):
                    try:
                        entry["analytical_ms"] = (
                            torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
                                node
                            )
                        )
                    except (RuntimeError, ValueError, TypeError):
                        pass
                else:
                    analytical = estimate_roofline_runtime_ms(node)
                    if analytical is not None and analytical > 0:
                        entry["analytical_ms"] = analytical
                diagnostics.append(entry)

        payload: dict[str, Any] = {
            "collective_count": len(profile.collectives),
            "op_count": profile.op_count,
            "op_entries": op_entries,
        }
        if diagnostics:
            payload["diagnostics"] = diagnostics

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "pge_profile_vs_analytical",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(payload),
        )

    def __call__(self, node: fx.Node, override_size: int | None = None) -> float | None:
        if _is_collective_node(node):
            return self._estimate_collective(node, override_size)
        return self._estimate_op(node)

    def _estimate_collective(
        self, node: fx.Node, override_size: int | None
    ) -> float | None:
        info = _get_collective_info(node)
        if info is None:
            return None
        coll_name, pg_ranks, nelems, dtype = info
        val = node.meta.get("val")
        if override_size is not None:
            if override_size == 0:
                return None
            if isinstance(val, torch.Tensor):
                elem_size = val.element_size()
                if elem_size > 0:
                    nelems = override_size // elem_size
        result = self.profile.lookup_collective(coll_name, pg_ranks, nelems, dtype)
        if result is not None:
            return result[0]
        return None

    def _estimate_op(self, node: fx.Node) -> float | None:
        """Estimate any non-collective op via exact shape+stride match in profile."""
        profile_name = _fx_target_to_profile_name(node)
        if profile_name is None:
            return None
        result = _get_node_input_shapes_and_strides(node)
        if result is None:
            return None
        input_shapes, input_strides = result
        dtype = _get_node_dtype(node)
        return self.profile.lookup_op(profile_name, input_shapes, input_strides, dtype)
