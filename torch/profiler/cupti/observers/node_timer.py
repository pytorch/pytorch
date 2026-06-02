# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Always-on per-graph-node activity-duration observer.

The smallest real consumer of the mux: registers for one or more activity
kinds with just the compact fields needed for timing (START, END,
GRAPH_NODE_ID) and accumulates total duration + count per graph node. The
mux poll thread feeds records via the callback; consumers pull the running
aggregate with ``drain()``.

By default only CONCURRENT_KERNEL is timed -- the common case, and the one
where the mux keeps its homogeneous (single-kind) vectorized bulk-parse
fast-path. Opt into MEMCPY / MEMSET via ``kinds`` to time those nodes too,
at the cost of that fast-path: mixing kinds makes the mux buffers
heterogeneous, so its decode falls back to a per-record kind-walk. (This
observer's own aggregation stays vectorized either way -- the cost is in
the mux's decode, not here.)

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
        self._dur_ns: dict[int, int] = {}
        self._count: dict[int, int] = {}
        # Register last (base __init__) so aggregation state is ready before
        # the mux poll thread can deliver records.
        super().__init__({k: set(_TIMED_FIELDS[k]) for k in self._kinds})

    def _on_records(self, kind: int, fields: dict[int, Any]) -> None:
        # Runs on the mux poll thread; keep it to vectorized numpy + a short
        # locked merge of the few distinct graph nodes.
        import numpy as np

        spec = _TIMED_FIELDS.get(kind)
        if spec is None:
            return
        sf, ef, gf = spec
        start = fields.get(sf)
        end = fields.get(ef)
        gnode = fields.get(gf)
        if start is None or end is None or gnode is None:
            return
        dur = (end.astype("<i8") - start.astype("<i8")).clip(min=0)
        uniq, inv = np.unique(gnode, return_inverse=True)
        sums = np.bincount(inv, weights=dur, minlength=uniq.size)
        counts = np.bincount(inv, minlength=uniq.size)
        with self._lock:
            for i, node in enumerate(uniq.tolist()):
                self._dur_ns[node] = self._dur_ns.get(node, 0) + int(sums[i])
                self._count[node] = self._count.get(node, 0) + int(counts[i])

    def drain(self) -> dict[int, tuple[int, int]]:
        """Return ``{graph_node_id: (total_duration_ns, count)}`` accumulated
        since the last call and reset."""
        # Flush CUPTI so the tail of the interval is aggregated first. Must
        # happen before taking _lock -- the flush delivers via _on_records.
        self._mux.force_drain()
        with self._lock:
            out = {n: (self._dur_ns[n], self._count[n]) for n in self._dur_ns}
            self._dur_ns = {}
            self._count = {}
        return out
