"""
Profile-Guided Estimation (PGE) for overlap scheduling.

Parses one or more Chrome Trace JSON files (from torch.profiler) and builds
lookup tables for kernel runtimes (collectives, matmuls, attention, custom
ops, etc.). Multiple same-capture rank profiles remove collective arrival
skew before aggregation.

The same profile set supplies every rank. The overlap scheduler still aligns
runtime-bearing nodes because live tensor shapes can differ across ranks.
"""

from __future__ import annotations

import functools
import gzip
import json
import logging
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.fx as fx
from torch._inductor.analysis.profile_analysis import (
    _create_extern_mapping,
    _dtype_map,
    _get_size_from_string,
    ParseException,
)
from torch._inductor.comm_analysis import (
    get_collective_type_from_kernel_name,
    NCCL_COLL,
)
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)

_OpInputMetadata = tuple[tuple[object, ...], ...]


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
    pg_name: str
    pg_desc: str
    pg_ranks: tuple[int, ...]  # sorted rank tuple
    group_size: int
    in_nelems: int  # "In msg nelems" from profile
    out_nelems: int  # "Out msg nelems" from profile
    dtype: str  # "Float", "BFloat16", etc.
    duration_us: float
    start_us: float | None = None
    rank: int | None = None
    sequence: int | None = None
    comms_id: str | None = None
    kernel_name: str = ""
    clock_domain: str | None = None


@dataclass
class OpRecord:
    """A single op observation from the profile (any CPU op with GPU kernels)."""

    op_name: str  # normalized name, e.g. "aten::mm", "mylib::my_custom_op"
    input_shapes: _OpInputMetadata
    input_strides: _OpInputMetadata
    dtype: torch.dtype | None
    duration_us: float  # sum of all GPU kernels for this CPU op


def _to_nested_tuple(x: object) -> object:
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
    pg_descriptions: dict[str, str] = field(default_factory=dict)

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
    # Index by logical process-group description. DeviceMesh gives every
    # parallel subgroup for one mesh axis the same description.
    _collective_index_by_pg_desc: dict[
        tuple[str, str, int, str], list[tuple[int, float]]
    ] = field(default_factory=dict)
    _all_to_allv_index: dict[tuple[tuple[int, ...], str, int, int], float] = field(
        default_factory=dict
    )
    _all_to_allv_index_by_mesh_dim: dict[tuple[int, int, str, int, int], float] = field(
        default_factory=dict
    )
    _all_to_allv_index_by_pg_desc: dict[tuple[str, int, str, int, int], float] = field(
        default_factory=dict
    )
    # Count of distinct PGs per mesh dimension (stride, group_size) — used for
    # ambiguity check (skip fallback if multiple PGs share the same mesh dim).
    _pg_count_by_mesh_dim: dict[tuple[int, int], int] = field(default_factory=dict)
    # Generic op index: (op_name, input_shapes, input_strides, dtype) -> avg_dur_us
    _op_index: dict[
        tuple[
            str,
            _OpInputMetadata,
            _OpInputMetadata,
            torch.dtype | None,
        ],
        float,
    ] = field(default_factory=dict)
    # Peak observed bandwidth per PG (GB/s), computed from largest messages
    _pg_peak_bw: dict[tuple[str, tuple[int, ...], str], float] = field(
        default_factory=dict
    )
    _mesh_dim_peak_bw: dict[tuple[str, int, int, str], float] = field(
        default_factory=dict
    )
    _pg_desc_peak_bw: dict[tuple[str, str, int, str], float] = field(
        default_factory=dict
    )
    profile_count: int = 0
    arrival_corrected_collectives: int = 0
    arrival_partial_collectives: int = 0
    arrival_clock_fallback_collectives: int = 0
    arrival_incomplete_collectives: int = 0
    collective_record_count: int = 0
    op_record_count: int = 0

    def load(self, trace_path: str | list[str]) -> None:
        """Load and parse one or more Chrome Trace JSON files."""
        paths = [trace_path] if isinstance(trace_path, str) else list(trace_path)
        trace_paths = [os.path.realpath(path) for path in paths]
        if not trace_paths:
            raise ValueError("PGE requires at least one trace file")
        if len(OrderedSet(trace_paths)) != len(trace_paths):
            raise ValueError("PGE trace file list contains duplicate paths")

        capture_ids: list[tuple[str, str | None, str | None] | None] = []
        world_sizes: list[int | None] = []
        profile_ranks: list[int | None] = []
        for index, path in enumerate(trace_paths):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"PGE trace file not found: {path}. "
                    f"Check config.aten_distributed_optimizations.profile_guided_estimations_profile_path"
                )
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt") as f:
                data = json.load(f)

            capture_ids.append(self._capture_id(data))
            dist_info = data.get("distributedInfo", {})
            rank = dist_info.get("rank")
            profile_ranks.append(rank if isinstance(rank, int) else None)
            world_size = dist_info.get("world_size")
            world_sizes.append(world_size if isinstance(world_size, int) else None)
            self._parse_pg_configs(data)
            self._parse_events(data, parse_ops=index == 0)

        self.profile_count = len(trace_paths)
        if self.profile_count > 1:
            known_capture_ids = [item for item in capture_ids if item is not None]
            if known_capture_ids and (
                len(known_capture_ids) != self.profile_count
                or len(OrderedSet(known_capture_ids)) != 1
            ):
                raise ValueError(
                    "PGE rank profiles must identify the same profiler capture"
                )
            known_world_sizes = [size for size in world_sizes if size is not None]
            if len(OrderedSet(known_world_sizes)) > 1:
                raise ValueError("PGE rank profiles must have the same world size")
            if None in profile_ranks or len(OrderedSet(profile_ranks)) != len(
                profile_ranks
            ):
                raise ValueError(
                    "PGE rank profiles must have distinct distributed ranks"
                )
            self._correct_collective_arrival_delays()
        self.collective_record_count = len(self.collectives)
        self.op_record_count = len(self.ops)
        self._build_indices()
        self.collectives.clear()
        self.ops.clear()

        log.info(
            "PGE loaded %d profiles: %d collectives, %d op records "
            "(%d distinct shapes), %d PGs; corrected=%d, clock_fallback=%d, "
            "partial=%d, incomplete=%d",
            self.profile_count,
            self.collective_record_count,
            self.op_record_count,
            len(self._op_index),
            len(self.pg_configs),
            self.arrival_corrected_collectives,
            self.arrival_clock_fallback_collectives,
            self.arrival_partial_collectives,
            self.arrival_incomplete_collectives,
        )

    @staticmethod
    def _capture_id(data: dict[str, Any]) -> tuple[str, str | None, str | None] | None:
        """Return job capture metadata when the profiler exported it."""
        job_name = data.get("PT_PROFILER_JOB_NAME", data.get("mast_job_name"))
        if job_name is None:
            return None
        job_version = data.get("PT_PROFILER_JOB_VERSION", data.get("mast_job_version"))
        attempt = data.get(
            "PT_PROFILER_JOB_ATTEMPT_INDEX", data.get("mast_job_attempt")
        )
        return (
            str(job_name),
            str(job_version) if job_version is not None else None,
            str(attempt) if attempt is not None else None,
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
                pg_desc = pg_info.get("pg_desc")
                if isinstance(pg_desc, str):
                    self.pg_descriptions[pg_name] = pg_desc
        elif isinstance(pg_config, dict):
            for pg_name, pg_info in pg_config.items():
                ranks = pg_info.get("ranks", [])
                if ranks:
                    self.pg_configs[pg_name] = tuple(sorted(ranks))
                pg_desc = pg_info.get("pg_desc")
                if isinstance(pg_desc, str):
                    self.pg_descriptions[pg_name] = pg_desc

    def _parse_events(self, data: dict[str, Any], parse_ops: bool = True) -> None:
        events = data.get("traceEvents", [])
        rank = data.get("distributedInfo", {}).get("rank")
        if not isinstance(rank, int):
            rank = None
        host_name = data.get("host_name")
        base_time_ns = data.get("baseTimeNanoseconds")
        has_time_base = isinstance(base_time_ns, int | float)
        base_time_us = float(base_time_ns) / 1e3 if has_time_base else 0.0
        clock_domain = (
            str(host_name)
            if has_time_base and isinstance(host_name, str) and host_name
            else None
        )
        extern_mapping = {}
        gpu_dur: dict[int, float] = defaultdict(float)
        if parse_ops:
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
            for ev in events:
                if not isinstance(ev, dict) or ev.get("cat") != "kernel":
                    continue
                args = ev.get("args", {})
                eid = args.get("External id")
                dur = ev.get("dur", 0.0)
                if eid is not None and dur > 0 and args.get("Collective name") is None:
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
            start_us = ev.get("ts")
            if not isinstance(start_us, int | float):
                start_us = None
            else:
                start_us += base_time_us
            sequence = args.get("Seq")
            if not isinstance(sequence, int):
                sequence = None
            comms_id = args.get("Comms Id")
            if comms_id is not None:
                comms_id = str(comms_id)

            pg_ranks = self._parse_ranks(pg_ranks_str, pg_name)
            pg_desc = self.pg_descriptions.get(str(pg_name))
            if pg_desc is None:
                event_pg_desc = args.get("Process Group Description", "")
                pg_desc = str(event_pg_desc) if event_pg_desc is not None else ""

            self.collectives.append(
                CollectiveRecord(
                    collective_name=coll_name,
                    pg_name=str(pg_name),
                    pg_desc=pg_desc,
                    pg_ranks=pg_ranks,
                    group_size=group_size,
                    in_nelems=in_nelems,
                    out_nelems=out_nelems,
                    dtype=dtype,
                    duration_us=dur,
                    start_us=start_us,
                    rank=rank,
                    sequence=sequence,
                    comms_id=comms_id,
                    kernel_name=ev.get("name", ""),
                    clock_domain=clock_domain,
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

    def _correct_collective_arrival_delays(self) -> None:
        """Replace cross-rank observations with arrival-corrected durations."""
        grouped: defaultdict[tuple[Any, ...], list[CollectiveRecord]] = defaultdict(
            list
        )
        comms_id_records: defaultdict[
            tuple[str, tuple[int, ...], tuple[Any, ...]], list[CollectiveRecord]
        ] = defaultdict(list)
        ungrouped: list[CollectiveRecord] = []
        for rec in self.collectives:
            if rec.sequence is not None and rec.pg_ranks:
                key = ("sequence", rec.pg_name, rec.pg_ranks, rec.sequence)
                grouped[key].append(rec)
            elif rec.comms_id is not None:
                normalized_name = self._normalize_collective_name(rec.collective_name)
                signature = (
                    normalized_name,
                    rec.pg_name,
                    rec.pg_ranks,
                    rec.group_size,
                    rec.dtype,
                    rec.kernel_name,
                )
                if normalized_name != "all_to_allv":
                    signature += (rec.in_nelems, rec.out_nelems)
                comms_id_records[(rec.comms_id, rec.pg_ranks, signature)].append(rec)
            else:
                ungrouped.append(rec)

        for (comms_id, pg_ranks, signature), records in comms_id_records.items():
            records_by_rank: defaultdict[int, list[CollectiveRecord]] = defaultdict(
                list
            )
            if any(rec.rank is None or rec.start_us is None for rec in records):
                ungrouped.extend(records)
                continue
            for rec in records:
                rank = rec.rank
                if rank is None:
                    raise AssertionError("validated collective record has no rank")
                records_by_rank[rank].append(rec)
            occurrence_counts = OrderedSet(
                len(rank_records) for rank_records in records_by_rank.values()
            )
            if len(occurrence_counts) != 1:
                ungrouped.extend(records)
                continue
            for rank_records in records_by_rank.values():
                rank_records.sort(
                    key=lambda rec: (
                        rec.start_us if rec.start_us is not None else math.inf
                    )
                )
            for occurrence, rank_records in enumerate(zip(*records_by_rank.values())):
                key = (
                    "comms_id",
                    comms_id,
                    pg_ranks,
                    signature,
                    occurrence,
                )
                grouped[key].extend(rank_records)

        corrected = list(ungrouped)
        for records in grouped.values():
            expected_ranks = OrderedSet(records[0].pg_ranks)
            observed_ranks = OrderedSet(
                [rec.rank for rec in records if rec.rank is not None]
            )
            signatures = OrderedSet(
                [
                    (
                        self._normalize_collective_name(rec.collective_name),
                        rec.pg_name,
                        rec.pg_ranks,
                        rec.group_size,
                        rec.in_nelems,
                        rec.out_nelems,
                        rec.dtype,
                        rec.kernel_name,
                    )
                    for rec in records
                ]
            )
            all_to_allv = all(
                self._normalize_collective_name(rec.collective_name) == "all_to_allv"
                for rec in records
            )
            invariant_signatures = OrderedSet(
                [
                    (
                        rec.pg_name,
                        rec.pg_ranks,
                        rec.group_size,
                        rec.dtype,
                        rec.kernel_name,
                    )
                    for rec in records
                ]
            )
            if (
                not expected_ranks
                or (len(signatures) != 1 and not all_to_allv)
                or len(invariant_signatures) != 1
                or any(rec.start_us is None for rec in records)
            ):
                self.arrival_incomplete_collectives += 1
                corrected.extend(records)
                continue

            intervals_by_rank: defaultdict[int, list[tuple[float, float]]] = (
                defaultdict(list)
            )
            for rec in records:
                rank = rec.rank
                start_us = rec.start_us
                if rank is None or start_us is None:
                    continue
                intervals_by_rank[rank].append((start_us, start_us + rec.duration_us))
            intervals = [
                (
                    min(start for start, _ in rank_intervals),
                    max(end for _, end in rank_intervals),
                )
                for rank_intervals in intervals_by_rank.values()
            ]
            if observed_ranks != expected_ranks:
                self.arrival_incomplete_collectives += 1
                if len(intervals) == 1:
                    corrected.extend(records)
                    continue
                # Even a partial set of ranks can remove some arrival skew. The
                # shortest observed span is clock-independent and remains an
                # upper bound when an unobserved rank arrived later.
                duration_us = min(end - start for start, end in intervals)
                self.arrival_partial_collectives += 1
                source_records = records if all_to_allv else records[:1]
                corrected.extend(
                    replace(rec, duration_us=duration_us, rank=None)
                    for rec in source_records
                )
                continue

            last_arrival = max(start for start, _ in intervals)
            first_completion = min(end for _, end in intervals)
            clock_domains = OrderedSet([rec.clock_domain for rec in records])
            if (
                len(clock_domains) == 1
                and None not in clock_domains
                and last_arrival <= first_completion
            ):
                # A collective cannot complete before its final rank starts. When
                # clocks agree, measure from that arrival to the final completion.
                duration_us = max(end for _, end in intervals) - last_arrival
                self.arrival_corrected_collectives += 1
            else:
                # Cross-host clocks are not sufficiently aligned. The shortest
                # rank duration is the clock-independent approximation.
                duration_us = min(end - start for start, end in intervals)
                self.arrival_clock_fallback_collectives += 1

            source_records = records if all_to_allv else records[:1]
            corrected.extend(
                replace(
                    rec,
                    duration_us=duration_us,
                    start_us=last_arrival,
                    rank=None,
                )
                for rec in source_records
            )

        self.collectives = corrected

    def _parse_ranks(self, ranks_str: str | list[int], pg_name: str) -> tuple[int, ...]:
        """Parse rank list from profile string or fall back to pg_configs."""
        if isinstance(ranks_str, list):
            return tuple(sorted(ranks_str))
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
        tensor_indices = [
            index
            for index, dtype_str in enumerate(input_types)
            if dtype_str in _dtype_map
            and index < len(input_dims)
            and index < len(input_strides)
            and isinstance(input_dims[index], (list, tuple))
            and isinstance(input_strides[index], (list, tuple))
        ]
        if not tensor_indices:
            return
        dtype = _dtype_map[input_types[tensor_indices[0]]]
        shapes = tuple(
            tuple(_to_nested_tuple(item) for item in input_dims[index])
            for index in tensor_indices
        )
        strides = tuple(
            tuple(_to_nested_tuple(item) for item in input_strides[index])
            for index in tensor_indices
        )
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
        coll_idx_by_pg_desc: dict[
            tuple[str, str, int, str], list[tuple[int, float]]
        ] = defaultdict(list)
        all_to_allv_idx: defaultdict[
            tuple[tuple[int, ...], str, int, int], list[float]
        ] = defaultdict(list)
        all_to_allv_idx_by_mesh_dim: defaultdict[
            tuple[int, int, str, int, int], list[float]
        ] = defaultdict(list)
        all_to_allv_idx_by_pg_desc: defaultdict[
            tuple[str, int, str, int, int], list[float]
        ] = defaultdict(list)
        # Track distinct PG rank sets per mesh dimension for ambiguity check
        pg_sets_by_mesh_dim: dict[tuple[int, int], OrderedSet[tuple[int, ...]]] = (
            defaultdict(OrderedSet)
        )
        for rec in self.collectives:
            norm_name = self._normalize_collective_name(rec.collective_name)
            gs = len(rec.pg_ranks) if rec.pg_ranks else rec.group_size
            stride = _rank_stride(rec.pg_ranks)
            if stride is not None:
                pg_sets_by_mesh_dim[(stride, gs)].add(rec.pg_ranks)
            if norm_name == "all_to_allv":
                all_to_allv_idx[
                    (rec.pg_ranks, rec.dtype, rec.in_nelems, rec.out_nelems)
                ].append(rec.duration_us)
                if rec.pg_desc and rec.pg_desc != "undefined":
                    all_to_allv_idx_by_pg_desc[
                        (
                            rec.pg_desc,
                            gs,
                            rec.dtype,
                            rec.in_nelems,
                            rec.out_nelems,
                        )
                    ].append(rec.duration_us)
                if stride is not None:
                    all_to_allv_idx_by_mesh_dim[
                        (stride, gs, rec.dtype, rec.in_nelems, rec.out_nelems)
                    ].append(rec.duration_us)
                continue
            coll_idx[(norm_name, rec.pg_ranks, rec.dtype)].append(
                (rec.out_nelems, rec.duration_us)
            )
            if rec.pg_desc and rec.pg_desc != "undefined":
                coll_idx_by_pg_desc[(norm_name, rec.pg_desc, gs, rec.dtype)].append(
                    (rec.out_nelems, rec.duration_us)
                )
            if stride is not None:
                coll_idx_by_mesh_dim[(norm_name, stride, gs, rec.dtype)].append(
                    (rec.out_nelems, rec.duration_us)
                )
        # Aggregate repeated observations so lookup is not determined by whichever
        # iteration happened to occur first in the trace.
        self._collective_index = {
            k: self._aggregate_collective_samples(v) for k, v in coll_idx.items()
        }
        self._collective_index_by_mesh_dim = {
            k: self._aggregate_collective_samples(v)
            for k, v in coll_idx_by_mesh_dim.items()
        }
        self._collective_index_by_pg_desc = {
            k: self._aggregate_collective_samples(v)
            for k, v in coll_idx_by_pg_desc.items()
        }
        self._all_to_allv_index = {
            key: statistics.median(samples) for key, samples in all_to_allv_idx.items()
        }
        self._all_to_allv_index_by_mesh_dim = {
            key: statistics.median(samples)
            for key, samples in all_to_allv_idx_by_mesh_dim.items()
        }
        self._all_to_allv_index_by_pg_desc = {
            key: statistics.median(samples)
            for key, samples in all_to_allv_idx_by_pg_desc.items()
        }
        self._pg_count_by_mesh_dim = {
            k: len(pgs) for k, pgs in pg_sets_by_mesh_dim.items()
        }

        op_groups: defaultdict[
            tuple[
                str,
                _OpInputMetadata,
                _OpInputMetadata,
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
        pg_bw_samples: dict[
            tuple[str, tuple[int, ...], str], list[tuple[int, float]]
        ] = defaultdict(list)
        mesh_dim_bw_samples: dict[
            tuple[str, int, int, str], list[tuple[int, float]]
        ] = defaultdict(list)
        pg_desc_bw_samples: dict[tuple[str, str, int, str], list[tuple[int, float]]] = (
            defaultdict(list)
        )
        for rec in self.collectives:
            if rec.out_nelems <= 0 or rec.duration_us <= 0:
                continue
            norm_name = self._normalize_collective_name(rec.collective_name)
            if norm_name == "all_to_allv":
                continue
            gs = len(rec.pg_ranks) if rec.pg_ranks else rec.group_size
            elem_bytes = self._dtype_elem_bytes(rec.dtype)
            total_bytes = rec.out_nelems * elem_bytes
            bw_gbps = total_bytes / (rec.duration_us * 1e-6) / 1e9  # GB/s
            pg_bw_samples[(norm_name, rec.pg_ranks, rec.dtype)].append(
                (total_bytes, bw_gbps)
            )
            if rec.pg_desc and rec.pg_desc != "undefined":
                pg_desc_bw_samples[(norm_name, rec.pg_desc, gs, rec.dtype)].append(
                    (total_bytes, bw_gbps)
                )
            stride = _rank_stride(rec.pg_ranks)
            if stride is not None:
                mesh_dim_bw_samples[(norm_name, stride, gs, rec.dtype)].append(
                    (total_bytes, bw_gbps)
                )

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
        self._pg_desc_peak_bw = {
            key: _peak_bw_from_samples(samples)
            for key, samples in pg_desc_bw_samples.items()
            if samples
        }

    @staticmethod
    def _aggregate_collective_samples(
        entries: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        samples_by_size: defaultdict[int, list[float]] = defaultdict(list)
        for nelems, duration_us in entries:
            samples_by_size[nelems].append(duration_us)

        aggregated = []
        for nelems, samples in samples_by_size.items():
            aggregated.append((nelems, statistics.median(samples)))
        return sorted(aggregated, key=lambda x: x[0])

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
        if "all_to_allv" in n or "alltoallv" in n:
            return "all_to_allv"
        if "all_to_all" in n or "alltoall" in n:
            return "all_to_all"
        return name

    # Maximum ratio of target_nelems / max_observed before switching from
    # log-log extrapolation to bandwidth-based estimation.
    EXTRAPOLATION_CAP = 2.0

    def _estimate_with_pg_bandwidth(
        self,
        collective_name: str,
        pg_ranks: tuple[int, ...],
        nelems: int,
        dtype: str,
        *,
        pg_desc: str | None = None,
    ) -> float | None:
        """Estimate collective duration using peak observed bandwidth for this PG.

        Used when the target size exceeds the extrapolation cap. Returns ms or None.
        """
        gs = len(pg_ranks)
        bw_gbps: float | None = None
        if pg_desc and pg_desc != "undefined":
            bw_gbps = self._pg_desc_peak_bw.get((collective_name, pg_desc, gs, dtype))
        if (bw_gbps is None or bw_gbps <= 0) and self.profile_count > 1:
            # Rank-specific samples from multiple profiles can produce different
            # schedules. Use a shared mesh estimate only when it is unambiguous.
            stride = _rank_stride(pg_ranks)
            if (
                stride is not None
                and self._pg_count_by_mesh_dim.get((stride, gs), 0) == 1
            ):
                bw_gbps = self._mesh_dim_peak_bw.get(
                    (collective_name, stride, gs, dtype)
                )
            else:
                bw_gbps = None
        elif bw_gbps is None or bw_gbps <= 0:
            bw_gbps = self._pg_peak_bw.get((collective_name, pg_ranks, dtype))
            if bw_gbps is None or bw_gbps <= 0:
                stride = _rank_stride(pg_ranks)
                if stride is not None:
                    bw_gbps = self._mesh_dim_peak_bw.get(
                        (collective_name, stride, gs, dtype)
                    )
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
        *,
        pg_desc: str | None = None,
        input_nelems: int | None = None,
    ) -> tuple[float, str] | None:
        """Look up collective duration in ms. Returns (duration_ms, source) or None.

        ``source`` is ``"profile"`` for exact/interpolated matches, or
        ``"pg_bandwidth"`` when bandwidth-based extrapolation was used.

        A process-group description identifies equivalent DeviceMesh subgroups.
        Multiple profiles use only description- or mesh-based aggregate estimates,
        because rank-specific estimates can make distributed schedules diverge.

        When the target size exceeds EXTRAPOLATION_CAP * max_observed, uses
        bandwidth-based estimation from peak observed bandwidth instead of
        linear extrapolation (which overestimates for large messages).
        """
        norm_name = self._normalize_collective_name(collective_name)
        gs = len(pg_ranks)
        if norm_name == "all_to_allv":
            if input_nelems is None:
                return None
            duration_us = None
            if pg_desc and pg_desc != "undefined":
                duration_us = self._all_to_allv_index_by_pg_desc.get(
                    (pg_desc, gs, dtype, input_nelems, nelems)
                )
            if duration_us is None and self.profile_count > 1:
                stride = _rank_stride(pg_ranks)
                if (
                    stride is not None
                    and self._pg_count_by_mesh_dim.get((stride, gs), 0) == 1
                ):
                    duration_us = self._all_to_allv_index_by_mesh_dim.get(
                        (stride, gs, dtype, input_nelems, nelems)
                    )
            elif duration_us is None:
                duration_us = self._all_to_allv_index.get(
                    (pg_ranks, dtype, input_nelems, nelems)
                )
            return (duration_us / 1e3, "profile") if duration_us is not None else None
        entries: list[tuple[int, float]] | None = None
        if pg_desc and pg_desc != "undefined":
            desc_key = (norm_name, pg_desc, gs, dtype)
            entries = self._collective_index_by_pg_desc.get(desc_key)
        if not entries and self.profile_count > 1:
            # Never choose an exact-rank estimate from a multi-profile input.
            # Only an unambiguous shared estimate can preserve rank invariance.
            stride = _rank_stride(pg_ranks)
            if (
                stride is not None
                and self._pg_count_by_mesh_dim.get((stride, gs), 0) == 1
            ):
                mesh_dim_key = (norm_name, stride, gs, dtype)
                entries = self._collective_index_by_mesh_dim.get(mesh_dim_key)
            else:
                entries = None
        elif not entries:
            key = (norm_name, pg_ranks, dtype)
            entries = self._collective_index.get(key)
            if not entries:
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
            est = self._estimate_with_pg_bandwidth(
                norm_name, pg_ranks, nelems, dtype, pg_desc=pg_desc
            )
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
            # Preserve the observed launch-latency floor when extrapolating
            # below the smallest profiled message.
            latency_floor = min(dur for _, dur in entries if dur > 0)
            duration_us = upper[1] * target_nelems / upper[0]
            return max(duration_us, latency_floor) / 1e3

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


def _get_node_input_dtype(node: fx.Node) -> torch.dtype | None:
    """Extract the first tensor input dtype, matching profiler input metadata."""
    for arg in node.args:
        if not isinstance(arg, fx.Node):
            continue
        val = arg.meta.get("val")
        if isinstance(val, torch.Tensor):
            return val.dtype
        if isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, torch.Tensor):
                    return item.dtype
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


def _get_node_output_shapes_and_strides(
    node: fx.Node,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]] | None:
    """Extract tensor output shapes and strides for profiler out-variant matching."""
    from torch._inductor.fx_passes.node_runtime_estimation import get_hint
    from torch.utils._pytree import tree_leaves

    shapes: list[tuple[int, ...]] = []
    strides: list[tuple[int, ...]] = []
    for value in tree_leaves(node.meta.get("val")):
        if not isinstance(value, torch.Tensor):
            continue
        resolved_shape = []
        for dim in value.shape:
            hint = get_hint(dim)
            if hint is None:
                return None
            resolved_shape.append(hint)
        resolved_stride = []
        for dim in value.stride():
            hint = get_hint(dim)
            if hint is None:
                return None
            resolved_stride.append(hint)
        shapes.append(tuple(resolved_shape))
        strides.append(tuple(resolved_stride))
    if not shapes:
        return None
    return tuple(shapes), tuple(strides)


@functools.lru_cache(maxsize=8)
def _load_profile_data(
    profile_files: tuple[tuple[str, int, int], ...],
) -> ProfileData:
    trace_paths = tuple(path for path, _, _ in profile_files)
    profile = ProfileData()
    profile.load(trace_paths[0] if len(trace_paths) == 1 else list(trace_paths))
    return profile


def _is_collective_node(node: fx.Node) -> bool:
    """Check if node is a collective communication op."""
    if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
        return False
    collective_type = get_collective_type_from_kernel_name(node.target.name())
    return collective_type not in (NCCL_COLL.UNSUPPORTED, NCCL_COLL.P2P)


def _get_collective_info(
    node: fx.Node,
) -> tuple[str, tuple[int, ...], str, int, int, str] | None:
    """Extract collective identity, size, and dtype from a collective node."""
    import torch.distributed as c10d
    from torch.fx.operator_schemas import normalize_function
    from torch.utils._pytree import tree_leaves

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
    if "all_to_all" in collective_name:
        output_splits = kwargs.get("output_split_sizes")
        input_splits = kwargs.get("input_split_sizes")
        split_values = (output_splits, input_splits)
        if any(
            splits is not None
            and (not isinstance(splits, (list, tuple)) or len(splits) > 0)
            for splits in split_values
        ):
            collective_name = "all_to_allv"

    try:
        from torch.distributed.distributed_c10d import (
            _resolve_process_group,
            get_process_group_ranks,
        )

        pg = _resolve_process_group(group_name)
        pg_ranks = tuple(sorted(get_process_group_ranks(pg)))
        pg_desc = pg.group_desc
    except (RuntimeError, KeyError, ValueError):
        log.debug(
            "PGE: failed to resolve process group for %s", node.name, exc_info=True
        )
        return None

    output_tensors = [
        value
        for value in tree_leaves(node.meta.get("val"))
        if isinstance(value, torch.Tensor)
    ]
    input_tensors: list[torch.Tensor] = []
    if node.args:
        for arg in tree_leaves(node.args[0]):
            if isinstance(arg, fx.Node):
                input_tensors.extend(
                    value
                    for value in tree_leaves(arg.meta.get("val"))
                    if isinstance(value, torch.Tensor)
                )
    tensors = output_tensors or input_tensors
    dtypes = OrderedSet(tensor.dtype for tensor in tensors + input_tensors)
    if not tensors or len(dtypes) != 1:
        return None
    out_nelems = sum(int(tensor.numel()) for tensor in tensors)
    in_nelems = sum(int(tensor.numel()) for tensor in input_tensors)
    if not input_tensors:
        in_nelems = out_nelems
    dtype = _dtype_to_nccl_str(tensors[0].dtype)

    return (collective_name, pg_ranks, pg_desc, in_nelems, out_nelems, dtype)


class ProfileGuidedEstimator:
    """Profile-guided runtime estimator for FX nodes.

    Implements the ``custom_runtime_estimation`` interface:
    ``(fx.Node, int | None) -> float | None`` (returns ms or None for fallback).

    Handles collectives via interpolation (latency + bandwidth model) and all
    other ops (matmul, SDPA, custom ops, etc.) via exact shape match from the
    profile trace. When given multiple same-capture rank profiles, collective
    durations are corrected for rank-arrival skew. The first profile supplies
    non-collective measurements.
    """

    def __init__(
        self,
        trace_path: str | list[str],
        diagnostics_gm: torch.fx.GraphModule | None = None,
    ) -> None:
        if isinstance(trace_path, str):
            trace_paths = (trace_path,)
        else:
            trace_paths = tuple(trace_path)
        canonical_paths = tuple(os.path.realpath(path) for path in trace_paths)
        profile_files = []
        for path in canonical_paths:
            stat = os.stat(path)
            profile_files.append((path, stat.st_mtime_ns, stat.st_size))
        self.profile = _load_profile_data(tuple(profile_files))
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
            "collective_count": profile.collective_record_count,
            "op_record_count": profile.op_record_count,
            "op_count": profile.op_count,
            "op_entries": op_entries,
            "profile_count": profile.profile_count,
            "arrival_correction": {
                "corrected_collectives": profile.arrival_corrected_collectives,
                "clock_fallback_collectives": (
                    profile.arrival_clock_fallback_collectives
                ),
                "partial_collectives": profile.arrival_partial_collectives,
                "incomplete_collectives": profile.arrival_incomplete_collectives,
            },
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
        coll_name, pg_ranks, pg_desc, input_nelems, output_nelems, dtype = info
        if override_size is not None:
            if override_size == 0:
                return None
            elem_size = self.profile._dtype_elem_bytes(dtype)
            if elem_size > 0:
                output_nelems = override_size // elem_size
        result = self.profile.lookup_collective(
            coll_name,
            pg_ranks,
            output_nelems,
            dtype,
            pg_desc=pg_desc,
            input_nelems=input_nelems,
        )
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
        dtype = _get_node_input_dtype(node)
        output = _get_node_output_shapes_and_strides(node)
        if output is not None:
            output_shapes, output_strides = output
            estimate = self.profile.lookup_op(
                profile_name,
                input_shapes + output_shapes,
                input_strides + output_strides,
                dtype,
            )
            if estimate is not None:
                return estimate
        return self.profile.lookup_op(profile_name, input_shapes, input_strides, dtype)
