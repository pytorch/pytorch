
"""Always-on per-graph-node activity-duration observer.

The smallest real consumer of the mux: registers for one or more activity
kinds with just the compact fields needed for timing (START, END,
GRAPH_NODE_ID) and buffers the raw per-kernel spans the mux poll thread
delivers. ``drain()`` returns them as flat numpy columns ``(graph_node_id,
start_ns, end_ns)`` -- consumers aggregate or bucket as they need (e.g.
total duration per node, or kernels bucketed into training steps by start
time), staying vectorized.

By default only CONCURRENT_KERNEL is timed -- the common case. Opt into
MEMCPY / MEMSET via ``kinds`` to time those nodes too. This keeps the mux's
vectorized bulk-parse: what the mux vectorizes over is record *size*, not
kind. A single kind is pure stride; multiple kinds whose selected fields
yield the same record size (as here -- every kind times the same 3 fields:
START, END, GRAPH_NODE_ID, so all records are one size) still parse via
stride + a vectorized kind-dispatch. The mux only falls back to a per-record
kind-walk when the enabled kinds have *different* record sizes. (This
observer just buffers the raw columns -- the cost is in the mux's decode,
not here.)

Durations are keyed by graph_node_id alone, kind-agnostic: each CUDA-graph
node is a single op, so its kind is unambiguous. Eager (non-graph)
activities report ``graph_node_id == 0`` and collapse into one node-0
bucket -- with multiple kinds enabled, that bucket mixes their durations.

This is framework-agnostic. A caller that knows how graph_node_ids map to
named regions (e.g. an annotation table) attributes the per-node totals
itself.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

from torch.profiler.cupti.observers.base import MuxObserver
from torch.profiler.cupti.types import (
    ActivityKind,
    KernelField,
    MemcpyField,
    MemsetField,
)


# kind -> (start_field, end_field, graph_node_field) for duration timing.
_TIMED_FIELDS: dict[int, tuple[int, int, int]] = {
    ActivityKind.CONCURRENT_KERNEL: (
        KernelField.START,
        KernelField.END,
        KernelField.GRAPH_NODE_ID,
    ),
    ActivityKind.MEMCPY: (
        MemcpyField.START,
        MemcpyField.END,
        MemcpyField.GRAPH_NODE_ID,
    ),
    ActivityKind.MEMSET: (
        MemsetField.START,
        MemsetField.END,
        MemsetField.GRAPH_NODE_ID,
    ),
}


class NodeTimerObserver(MuxObserver):
    def __init__(self, kinds: Iterable[int] | None = None) -> None:
        self._kinds: tuple[int, ...] = (
            tuple(kinds) if kinds is not None else (ActivityKind.CONCURRENT_KERNEL,)
        )
        unknown = [k for k in self._kinds if k not in _TIMED_FIELDS]
        if unknown:
            raise ValueError(
                f"NodeTimerObserver: unsupported activity kind(s) {unknown}; "
                f"timing supports {sorted(_TIMED_FIELDS)}"
            )
        self._lock = threading.Lock()
        # Raw (graph_node_id, start_ns, end_ns) column chunks as the poll thread
        # delivers them; grouped into per-node spans in drain(). Kept raw (not
        # pre-aggregated to a per-node total) so consumers that need per-kernel
        # timing -- e.g. bucketing kernels into steps by start time -- can build
        # on this observer.
        self._chunks: list[tuple[Any, Any, Any]] = []
        # Register last (base __init__) so the buffer is ready before the mux
        # poll thread can deliver records.
        super().__init__({k: set(_TIMED_FIELDS[k]) for k in self._kinds})

    def _on_records(self, kind: int, fields: dict[int, Any]) -> None:
        # Mux poll thread: just stash the columns (cheap append); grouping by
        # node happens in drain(), off this hot path.
        spec = _TIMED_FIELDS.get(kind)
        if spec is None:
            return
        sf, ef, gf = spec
        start = fields.get(sf)
        end = fields.get(ef)
        gnode = fields.get(gf)
        if start is None or end is None or gnode is None:
            return
        with self._lock:
            self._chunks.append((gnode, start, end))

    def drain(self, flush: bool = False) -> "tuple[Any, Any, Any]":
        """Return the raw per-kernel spans delivered since the last call as
        three parallel numpy columns ``(graph_node_id, start_ns, end_ns)``
        (dtypes ``<u8, <i8, <i8``), and reset. Empty -> three length-0 arrays.

        Flat and vectorized on purpose: consumers bucket/aggregate with numpy
        (e.g. ``searchsorted``/``bincount``) without a Python loop over the
        (~50k) kernels, so this can run on a poller thread without holding the
        GIL. (A node's duration is ``end - start``; eager activities share
        ``graph_node_id == 0``.)

        Cheap by default: reads only what the mux poll thread has already
        delivered -- this is the always-on path and is called on a rolling
        basis, so it must NOT force a synchronous ``cuptiActivityFlushAll`` on
        every call (records still buffered are picked up on the next drain).
        Pass ``flush=True`` only to close a bounded measurement window where a
        complete snapshot is required, accepting the cost of a forced flush."""
        import numpy as np

        if flush:
            # Flush CUPTI so the tail lands first. Must happen before taking
            # _lock -- the flush delivers via _on_records.
            self._mux.force_drain()
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return (
                np.empty(0, dtype="<u8"),
                np.empty(0, dtype="<i8"),
                np.empty(0, dtype="<i8"),
            )
        gnode = np.concatenate([c[0] for c in chunks]).astype("<u8", copy=False)
        start = np.concatenate([c[1] for c in chunks]).astype("<i8", copy=False)
        end = np.concatenate([c[2] for c in chunks]).astype("<i8", copy=False)
        return gnode, start, end
