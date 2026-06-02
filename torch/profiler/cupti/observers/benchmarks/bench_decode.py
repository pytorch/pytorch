# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Decoder throughput microbenchmark for the CUPTI mux dispatcher.

Measures how fast ``CuptiActivityMux._parse_and_demux`` (the dispatcher)
turns a raw CUPTI user-defined-record buffer into demuxed column arrays and
feeds them to an observer's ``_on_records`` -- isolated from the GPU/CUPTI.
We synthesize the record buffer and the per-buffer record-layout structs by
hand, build a headless mux + observer (no CUDA, no real subscriber), and
time the parse over many iterations.

This exercises both dispatch paths:
  - homogeneous (single enabled kind) -> the vectorized stride fast-path,
  - heterogeneous (multiple kinds)     -> the per-record KIND-walk.

Per observer type (NodeTimerObserver, ProfilerObserver) it reports records/s
and MB/s, so you can see the dispatcher's parse rate and the per-observer
``_on_records`` cost. Pure CPU; no special env needed.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import threading
import time
from typing import Any

import numpy as np

from torch.profiler.cupti.mux import (
    _UDCompleteInfo,
    _UDFieldLayoutEntry,
    _UDRecordLayout,
    CuptiActivityMux,
    Observer,
)
from torch.profiler.cupti.observers.node_timer import _TIMED_FIELDS, NodeTimerObserver
from torch.profiler.cupti.observers.profiler import _DEFAULT_WANTS, ProfilerObserver
from torch.profiler.cupti.types import ActivityKind


# Field byte widths for the synthetic layout. KIND is the 4-byte id at
# offset 0 (required first); every other field is given 8 bytes (valid gather
# width; the values are irrelevant to throughput).
_KIND_FIELD = 0
_KIND_SIZE = 4
_FIELD_SIZE = 8


class _Layout:
    """Holds the ctypes objects for one kind's record layout, kept alive for
    the lifetime of the benchmark (the mux reads them by pointer)."""

    def __init__(self, field_ids: "set[int]") -> None:
        ids = [_KIND_FIELD] + sorted(f for f in field_ids if f != _KIND_FIELD)
        self._entries = (_UDFieldLayoutEntry * len(ids))()
        offset = 0
        for i, fid in enumerate(ids):
            size = _KIND_SIZE if fid == _KIND_FIELD else _FIELD_SIZE
            self._entries[i].structSize = ctypes.sizeof(_UDFieldLayoutEntry)
            self._entries[i].fieldId = fid
            self._entries[i].offset = offset
            self._entries[i].size = size
            self._entries[i].alignment = size
            offset += size
        self.record_size = offset
        self.layout = _UDRecordLayout(
            structSize=ctypes.sizeof(_UDRecordLayout),
            pEntries=ctypes.cast(self._entries, ctypes.POINTER(_UDFieldLayoutEntry)),
            numFields=len(ids),
            recordSize=offset,
        )


def _build_info(layouts: "dict[int, _Layout]") -> "tuple[Any, list]":
    """Build a `_UDCompleteInfo` whose ppRecordLayouts[kind] points at each
    kind's layout (null for kinds not present). Returns (info, keepalive)."""
    n = max(layouts) + 1
    ptr_arr = (ctypes.POINTER(_UDRecordLayout) * n)()
    keepalive: list = [ptr_arr]
    for kind, lay in layouts.items():
        holder = (_UDRecordLayout * 1)(lay.layout)
        keepalive.append(holder)
        ptr_arr[kind] = ctypes.cast(holder, ctypes.POINTER(_UDRecordLayout))
    info = _UDCompleteInfo(
        structSize=ctypes.sizeof(_UDCompleteInfo),
        threadId=0,
        ppRecordLayouts=ctypes.cast(
            ptr_arr, ctypes.POINTER(ctypes.POINTER(_UDRecordLayout))
        ),
        numRecordLayouts=n,
    )
    keepalive.append(info)
    return info, keepalive


def _build_buffer(
    kinds_cycle: "list[int]", layouts: "dict[int, _Layout]", n_records: int
) -> "tuple[Any, int, list]":
    """Build a raw record buffer of ``n_records`` records, cycling through
    ``kinds_cycle`` (one element => homogeneous). Writes each record's KIND
    id at its start; other fields stay zero. Returns (buffer, valid_size,
    keepalive)."""
    starts: list[int] = []
    kind_seq: list[int] = []
    off = 0
    for i in range(n_records):
        kind = kinds_cycle[i % len(kinds_cycle)]
        starts.append(off)
        kind_seq.append(kind)
        off += layouts[kind].record_size
    valid_size = off
    buf = (ctypes.c_uint8 * valid_size)()
    npbuf = np.frombuffer(buf, dtype=np.uint8)
    starts_arr = np.array(starts, dtype=np.int64)
    kinds_arr = np.array(kind_seq, dtype=np.uint32)
    # Write the 4-byte KIND id at each record start (vectorized per byte).
    for b in range(4):
        npbuf[starts_arr + b] = (kinds_arr >> (8 * b)) & 0xFF
    return buf, valid_size, [buf, npbuf]


def _headless_mux(observer: object, wants: "dict[int, set[int]]") -> CuptiActivityMux:
    """A mux with just enough state for `_parse_and_demux` (no CUPTI): the
    lock and one registered Observer wrapping the given observer's callback."""
    mux = CuptiActivityMux.__new__(CuptiActivityMux)
    mux._lock = threading.Lock()
    mux._fsize_seen = {}  # _parse_and_demux records discovered widths here
    resolved = {k: frozenset(v) for k, v in wants.items()}
    mux._observers = [Observer(resolved, observer._on_records)]  # type: ignore[attr-defined]
    return mux


def _headless_node_timer(kinds: "list[int]") -> "tuple[object, dict[int, set[int]]]":
    obs = NodeTimerObserver.__new__(NodeTimerObserver)
    obs._lock = threading.Lock()  # type: ignore[attr-defined]
    obs._dur_ns = {}  # type: ignore[attr-defined]
    obs._count = {}  # type: ignore[attr-defined]
    wants = {k: set(_TIMED_FIELDS[k]) for k in kinds}
    return obs, wants


def _headless_profiler() -> "tuple[object, dict[int, set[int]]]":
    obs = ProfilerObserver.__new__(ProfilerObserver)
    obs._lock = threading.Lock()  # type: ignore[attr-defined]
    obs._chunks = {}  # type: ignore[attr-defined]
    wants = {k: set(v) for k, v in _DEFAULT_WANTS.items()}
    return obs, wants


def _padded_layout_fields(
    wants: "dict[int, set[int]]", kinds: "list[int]"
) -> "dict[int, set[int]]":
    """Simulate uniform-padding: grow each kind's field set with synthetic 8B
    filler ids until all kinds have the same non-KIND field count (=> same
    record size in this bench's flat-8 model). Mirrors what the mux's
    pad_to_uniform would do if the registry exposed spare fields."""
    nonkind = {k: {f for f in wants[k] if f != _KIND_FIELD} for k in kinds}
    target_n = max(len(s) for s in nonkind.values())
    out: dict[int, set[int]] = {}
    filler = 90  # synthetic ids, distinct from real field ids
    for k in kinds:
        s = set(nonkind[k])
        while len(s) < target_n:
            s.add(filler)
            filler += 1
        out[k] = s
    return out


def _bench_one(
    label: str,
    observer: object,
    wants: "dict[int, set[int]]",
    kinds_cycle: "list[int]",
    n_records: int,
    iters: int,
    layout_fields: "dict[int, set[int]] | None" = None,
) -> dict:
    # Layout (record contents/size) may be padded; the observer still gathers
    # only its real `wants`, so filler fields are present but never read.
    lf = layout_fields if layout_fields is not None else wants
    layouts = {k: _Layout(lf[k]) for k in set(kinds_cycle)}
    info, _ka_info = _build_info(layouts)
    buf, valid_size, _ka_buf = _build_buffer(kinds_cycle, layouts, n_records)
    mux = _headless_mux(observer, wants)

    # Mirror the mux's dispatch decision for labeling: one kind -> stride;
    # multi-kind but one record size -> stride + vectorized KIND dispatch;
    # otherwise the sequential walk.
    distinct_sizes = {layouts[k].record_size for k in set(kinds_cycle)}
    if len(set(kinds_cycle)) == 1:
        path = "stride"
    elif len(distinct_sizes) == 1:
        path = "stride+dispatch"
    else:
        path = "walk"
    # Warm.
    for _ in range(3):
        mux._parse_and_demux(buf, valid_size, info)
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mux._parse_and_demux(buf, valid_size, info)
        samples.append(time.perf_counter() - t0)
        # Reset observer accumulation OUTSIDE the timed region so the
        # ProfilerObserver's per-chunk stash doesn't grow unbounded.
        for attr in ("_chunks", "_dur_ns", "_count"):
            acc = getattr(observer, attr, None)
            if acc is not None:
                acc.clear()
    med = statistics.median(samples)
    # Per-kind filler fields added by padding ({field id: bytes}) -- the delta
    # between the layout fields and the observer's requested fields. Empty for
    # unpadded modes.
    filler = {
        int(k): {int(f): _FIELD_SIZE for f in sorted(set(lf[k]) - set(wants[k]))}
        for k in set(kinds_cycle)
        if set(lf[k]) - set(wants[k])
    }
    return {
        "label": label,
        "path": path,
        "kinds": [int(k) for k in dict.fromkeys(kinds_cycle)],
        "records": n_records,
        # Per-kind record size (KIND=4B + 8B per selected field; the 8B width
        # is a placeholder -- real CUPTI field widths vary). Buffer size
        # tracks the field selection, not the kind count.
        "record_bytes": {int(k): layouts[k].record_size for k in layouts},
        # Per-kind filler fields {field id: bytes} added to reach uniform size.
        "filler_fields": filler,
        "buffer_mb": round(valid_size / 1e6, 3),
        "median_ms": round(med * 1e3, 4),
        "records_per_s_M": round((n_records / med) / 1e6, 2),
        "throughput_MBps": round((valid_size / med) / 1e6, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=50_000)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    k = ActivityKind
    nt_kernel, nt_kernel_wants = _headless_node_timer([k.CONCURRENT_KERNEL])
    nt_multi, nt_multi_wants = _headless_node_timer(
        [k.CONCURRENT_KERNEL, k.MEMCPY, k.MEMSET]
    )
    prof, prof_wants = _headless_profiler()

    results = [
        # NodeTimerObserver, kernel-only: single-kind stride fast-path.
        _bench_one(
            "node_timer:kernel",
            nt_kernel,
            nt_kernel_wants,
            [k.CONCURRENT_KERNEL],
            args.records,
            args.iters,
        ),
        # NodeTimerObserver, all kinds: same 3 fields per kind => uniform
        # record size => stride + vectorized KIND dispatch (no walk).
        _bench_one(
            "node_timer:multi",
            nt_multi,
            nt_multi_wants,
            [k.CONCURRENT_KERNEL, k.MEMCPY, k.MEMSET],
            args.records,
            args.iters,
        ),
        # ProfilerObserver, all kinds: wider/uneven field selection =>
        # variable record size => sequential KIND-walk.
        _bench_one(
            "profiler:multi",
            prof,
            prof_wants,
            [k.CONCURRENT_KERNEL, k.MEMCPY, k.MEMSET],
            args.records,
            args.iters,
        ),
        # Same, but kernel records padded to memcpy/memset size => uniform =>
        # stride+dispatch. Shows what mux.enable_uniform_padding() would buy
        # (the real registry can't pad kernels yet -- needs extra field ids).
        _bench_one(
            "profiler:multi_padded",
            prof,
            prof_wants,
            [k.CONCURRENT_KERNEL, k.MEMCPY, k.MEMSET],
            args.records,
            args.iters,
            layout_fields=_padded_layout_fields(
                prof_wants, [k.CONCURRENT_KERNEL, k.MEMCPY, k.MEMSET]
            ),
        ),
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
