# mypy: allow-untyped-defs
"""``FlightRecorderPlugin`` -- serializes ``CommsObserver`` records into the OSS
FlightRecorder schema (the ``{version, pg_config, pg_status, entries}`` shape that
``torch.distributed.flight_recorder`` / ``fr_trace`` consumes), so the CUPTI monitor
can be the single event/timing engine behind the existing analyzer.
"""

from __future__ import annotations

import collections
import json
import threading
from collections.abc import Callable  # noqa: TC003
from typing import Any

from torch.profiler._cupti.comms.plugin import CommRecordPlugin
from torch.profiler._cupti.observers.comms import CommRecord  # noqa: TC001


# NCCL profiler func name -> torch.distributed.flight_recorder op name. The
# analyzer requires ``profiling_name`` == "nccl:<op>" with <op> in its
# COLLECTIVES/P2P sets; an unmapped func falls back to its lowercased name (a
# backend can override per record via a ``profiling_name`` metadata field).
_FR_OP_NAMES = {
    "AllReduce": "all_reduce",
    "Broadcast": "broadcast",
    "Reduce": "reduce",
    "AllGather": "all_gather",
    "ReduceScatter": "reduce_scatter",
    "AllToAll": "all_to_all",
    "AllToAllv": "all_to_all",
    "Send": "send",
    "Recv": "recv",
    "Gather": "gather",
    "Scatter": "scatter",
}

_FR_VERSION = "2.10"


def _fr_profiling_name(metadata: dict[str, Any]) -> str:
    func = metadata.get("func", "")
    op = _FR_OP_NAMES.get(func) or (func.lower() if func else "unknown")
    return f"nccl:{op}"


def _fr_entry(
    coll_id: int,
    metadata: dict[str, Any],
    record: CommRecord | None,
    clock_converter: Callable[[int], int] | None = None,
) -> dict[str, Any]:
    """One OSS-FlightRecorder entry. Maps the fields we know from the record +
    standard NCCL metadata; everything a target deployment needs but CUPTI can't see
    (process_group identity + ranks, full input/output sizes, timeout, stack frames)
    is read straight from the metadata bag -- the comms hook supplies most of it
    (sizes/dtypes/pg identity/ranks) and a backend can add the rest under the OSS key
    name. ``record`` is None for a still-queued (scheduled) collective.

    ``clock_converter`` (CUPTI native ns -> unix-epoch ns) converts the three timestamp
    fields for cross-rank correlation / tlparse timelines; identity (native ns) when
    None. ``duration_ms`` is a delta and stays in native units (no double-convert)."""
    m = metadata
    count = m.get("count")
    sizes = [[count]] if count is not None else []
    dtypes = [m["datatype"]] if m.get("datatype") is not None else []
    ranks = m.get("process_group_ranks")
    to_unix = clock_converter if clock_converter is not None else (lambda ns: ns)
    return {
        "record_id": m.get("record_id", coll_id),
        "pg_id": m.get("pg_id", 0),
        "process_group": m.get("process_group", ["0", ""]),
        "process_group_ranks": list(ranks) if ranks is not None else [],
        "collective_seq_id": m.get("collective_seq_id", m.get("seq", 0)),
        "p2p_seq_id": m.get("p2p_seq_id", 0),
        "op_id": m.get("op_id", coll_id),
        "profiling_name": m.get("profiling_name") or _fr_profiling_name(m),
        "time_created_ns": m.get(
            "time_created_ns", to_unix(record.start_ns) if record else 0
        ),
        "input_sizes": m.get("input_sizes", sizes),
        "output_sizes": m.get("output_sizes", sizes),
        "input_dtypes": m.get("input_dtypes", dtypes),
        "output_dtypes": m.get("output_dtypes", dtypes),
        "state": "completed" if record else "scheduled",
        "time_discovered_started_ns": to_unix(record.start_ns) if record else None,
        "time_discovered_completed_ns": to_unix(record.end_ns) if record else None,
        "duration_ms": (record.latency_ns / 1e6) if record else None,
        "retired": record is not None,
        "timeout_ms": m.get("timeout_ms", 0),
        "is_p2p": m.get("is_p2p", False),
        "frames": m.get("frames", []),
    }


class FlightRecorderPlugin(CommRecordPlugin):
    """Serializes ``CommsObserver``'s records into the **OSS FlightRecorder schema**
    -- the ``{version, pg_config, pg_status, entries}`` shape that
    ``torch.distributed.flight_recorder`` (``fr_trace``) consumes -- so the CUPTI
    monitor can be the single event/timing engine behind the existing analyzer
    instead of a parallel one.

    Each collective is one entry: ``scheduled`` while in flight (from ``on_schedule``),
    ``started`` once its kernel begins (from ``on_start``, if the start-event path is
    wired -- otherwise skipped), and ``completed`` once its kernel is timed (from
    ``on_end``), with the device timing filled in. Keeping the in-flight collectives as
    ``scheduled`` entries is
    what lets a dump during a hang feed the analyzer's cross-rank STATE-mismatch
    check (one rank completed, another still scheduled -> the culprit). Fields CUPTI
    can't observe -- process-group identity/ranks, full tensor sizes, timeout, stack
    frames -- are read from each collective's metadata, so a backend supplies them
    through the metadata store (see :func:`_fr_entry`).

    ``clock_converter`` (CUPTI native ns -> unix-epoch ns) converts the entries'
    timestamp fields so they line up cross-rank and in tlparse timelines; pass the
    observer's ``convert_time`` (``FlightRecorderPlugin(clock_converter=
    observer.convert_time)``). None keeps the raw native ns (backward-compatible).

    ``max_entries`` bounds the ring (oldest evicted). :meth:`dump` / :meth:`dump_json`
    are safe to call from another thread (e.g. the watchdog's ``on_timeout`` or a
    dump signal)."""

    def __init__(
        self,
        max_entries: int = 2000,
        *,
        comm_lib_version: str = "",
        clock_converter: Callable[[int], int] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._max = max_entries
        self._comm_lib_version = comm_lib_version
        self._clock_converter = clock_converter
        # coll_id -> OSS entry, in issue order. A scheduled entry is overwritten in
        # place by its completed entry (assignment keeps the OrderedDict position).
        self._entries: collections.OrderedDict[int, dict[str, Any]] = (
            collections.OrderedDict()
        )
        self._last_enqueued = -1
        self._last_completed = -1

    def on_schedule(self, coll_id: int, metadata: dict[str, Any]) -> None:
        with self._lock:
            if coll_id not in self._entries:
                entry = _fr_entry(coll_id, metadata, None, self._clock_converter)
                self._entries[coll_id] = entry
                self._last_enqueued = max(
                    self._last_enqueued, int(entry["collective_seq_id"])
                )
                self._evict()

    def on_start(self, coll_id: int, metadata: dict[str, Any]) -> None:
        # The device-start signal: flip the scheduled entry to "started". A late start
        # (after on_end, undefined record order) or a graph replay's repeat start lands
        # on a non-"scheduled" entry and no-ops -- correct in both cases.
        with self._lock:
            entry = self._entries.get(coll_id)
            if entry is not None and entry["state"] == "scheduled":
                entry["state"] = "started"
                entry["time_discovered_started_ns"] = entry.get("time_created_ns")

    def on_end(self, record: CommRecord) -> None:
        with self._lock:
            entry = _fr_entry(
                record.coll_id, record.metadata, record, self._clock_converter
            )
            self._entries[record.coll_id] = entry  # overwrites the scheduled entry
            self._last_completed = max(
                self._last_completed, int(entry["collective_seq_id"])
            )
            self._evict()

    def _evict(self) -> None:
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def _pg_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for e in self._entries.values():
            pg = e["process_group"]
            name = str(pg[0])
            desc = str(pg[1]) if len(pg) > 1 else ""
            config.setdefault(
                name,
                {"name": name, "desc": desc, "ranks": repr(e["process_group_ranks"])},
            )
        return config or {"0": {"name": "0", "desc": "", "ranks": "[]"}}

    def dump(self) -> dict[str, Any]:
        """The OSS FlightRecorder dump as a dict (``fr_trace``-compatible)."""
        with self._lock:
            return {
                "version": _FR_VERSION,
                "comm_lib_version": self._comm_lib_version,
                "pg_config": self._pg_config(),
                "pg_status": {
                    "0": {
                        "last_enqueued_collective": self._last_enqueued,
                        "last_started_collective": self._last_completed,
                        "last_completed_collective": self._last_completed,
                    }
                },
                "entries": [dict(e) for e in self._entries.values()],
            }

    def dump_json(self) -> str:
        """The OSS FlightRecorder dump serialized to a JSON string."""
        return json.dumps(self.dump())

    def write_dump(self, rank: int | None = None, path: str | None = None) -> str:
        """Pickle the dump to the per-rank file ``fr_trace`` / ``tlparse`` read, and
        return the path. ``fr_trace`` reads a directory of ``<prefix><rank>`` pickles,
        so wire this into the hang path -- e.g. ``HangDetectorPlugin(on_timeout=lambda
        *_: plugin.write_dump())`` -- to get an automatic on-timeout dump.

        ``rank`` defaults to ``$RANK``. ``path`` is the full file path; if omitted it
        is ``<base><rank>`` where base is ``$TORCH_FR_DUMP_TEMP_FILE`` /
        ``$TORCH_NCCL_DEBUG_INFO_TEMP_FILE`` or
        ``$XDG_CACHE_HOME/torch/comm_lib_trace_rank_`` -- the same convention c10d's
        FlightRecorder uses."""
        import os
        import pickle

        if rank is None:
            rank = int(os.environ.get("RANK", "0"))
        if path is None:
            base = os.environ.get("TORCH_FR_DUMP_TEMP_FILE") or os.environ.get(
                "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
            )
            if not base:
                cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(
                    os.path.expanduser("~"), ".cache"
                )
                base = os.path.join(cache, "torch", "comm_lib_trace_rank_")
            path = f"{base}{rank}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.dump(), f)
        return path
