# mypy: allow-untyped-defs
"""Always-on per-graph-node activity-duration observer.

The smallest real consumer of the monitor: registers for one or more activity
kinds with just the compact fields needed for timing (START, END,
GRAPH_NODE_ID, STREAM_ID) and buffers the raw per-kernel spans the monitor's
worker thread delivers. :meth:`NodeTimerObserver.drain` returns them as flat numpy
columns ``(graph_node_id, start_ns, end_ns, stream_id)`` -- consumers aggregate or
bucket as they need (e.g. total duration per node, or kernels bucketed into training
steps by start time, or split per stream), staying vectorized.

By default only CONCURRENT_KERNEL is timed -- the common case. Opt into MEMCPY /
MEMCPY2 (peer-to-peer) / MEMSET via ``kinds`` to time those nodes too. Requesting
MEMCPY2 also enables MEMCPY implicitly (CUPTI emits no MEMCPY2 records unless
MEMCPY is on too). Every kind here times the same 4
fields (START, END, GRAPH_NODE_ID, STREAM_ID), so all records are one size and the
monitor decodes them via its vectorized stride + kind-dispatch path; this observer
just buffers the raw columns (the cost is in the monitor's decode, not here).

Durations are keyed by graph_node_id alone, kind-agnostic: each CUDA-graph node
is a single op, so its kind is unambiguous. Eager (non-graph) activities report
``graph_node_id == 0`` and collapse into one node-0 bucket.

**Named regions.** Pass ``annotations=ObserverAnnotationSettings(...)`` (from the
base); :meth:`drain_annotated` then returns ``{name: [(start_ns, end_ns), ...]}``,
resolving each span graph-first then eager-fallback:

  * **Graph** (``graph_annotation_resolver``) -- ``graph_node_id -> name`` via the resolver
    (a custom one or the default registry). Needs no extra record kinds, so it stays on the
    monitor's **vectorized** decode path. On by default; ``None`` disables it.
  * **Eager** (``support_eager_annotations``) -- ``record_function``-style regions bracketed
    with :meth:`push_annotation`/:meth:`annotate` (inherited from the base) become
    external-correlation ids, joined ``correlation_id -> external_id -> name``. This
    folds in the EXTERNAL_CORRELATION + RUNTIME record kinds (CUPTI only emits the
    former when the latter is enabled), whose differing record sizes drop the decode
    onto the slower per-record walk -- off by default for that cost. External ids don't
    survive CUDA-graph capture, so eager covers only eager activity.

With no ``annotations`` at all, nothing is named (everything in ``""``).
"""

from __future__ import annotations

import threading
from typing import Any, TYPE_CHECKING

from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

from torch.profiler._cupti.observers.base import (
    CuptiMonitorObserver,
    ObserverAnnotationSettings,
)
from torch.profiler._cupti.records import (
    CORRELATION_FIELD,
    ExternalCorrelation,
    Kernel,
    Memcpy,
    Memcpy2,
    Memset,
)


if TYPE_CHECKING:
    from collections.abc import Iterable


# kind -> (start_field_id, end_field_id, graph_node_field_id, stream_field_id) for
# duration timing. stream lets consumers attribute spans to their CUDA stream.
_TIMED_FIELDS: dict[int, tuple[int, int, int, int]] = {
    int(ActivityKind.CONCURRENT_KERNEL): (
        int(Kernel.START),
        int(Kernel.END),
        int(Kernel.GRAPH_NODE_ID),
        int(Kernel.STREAM_ID),
    ),
    int(ActivityKind.MEMCPY): (
        int(Memcpy.START),
        int(Memcpy.END),
        int(Memcpy.GRAPH_NODE_ID),
        int(Memcpy.STREAM_ID),
    ),
    # Peer-to-peer / cross-device copies. Field ids differ from MEMCPY (src/dst
    # device fields shift correlation/graph ids), but the timed span is the same.
    int(ActivityKind.MEMCPY2): (
        int(Memcpy2.START),
        int(Memcpy2.END),
        int(Memcpy2.GRAPH_NODE_ID),
        int(Memcpy2.STREAM_ID),
    ),
    int(ActivityKind.MEMSET): (
        int(Memset.START),
        int(Memset.END),
        int(Memset.GRAPH_NODE_ID),
        int(Memset.STREAM_ID),
    ),
}

_EXTERNAL = int(ActivityKind.EXTERNAL_CORRELATION)


class NodeTimerObserver(CuptiMonitorObserver):
    """Buffers raw per-activity ``(graph_node_id, start_ns, end_ns, stream_id)`` spans the
    monitor delivers; :meth:`drain` returns them as flat numpy columns. Construct with
    ``annotations=ObserverAnnotationSettings(...)`` and :meth:`drain_annotated` returns
    ``{name: [(start_ns, end_ns), ...]}``. See the module docstring."""

    def __init__(
        self,
        kinds: Iterable[int] | None = None,
        *,
        annotations: ObserverAnnotationSettings | None = None,
    ) -> None:
        self._kinds: tuple[int, ...] = (
            tuple(kinds) if kinds is not None else (ActivityKind.CONCURRENT_KERNEL,)
        )
        unknown = [k for k in self._kinds if int(k) not in _TIMED_FIELDS]
        if unknown:
            raise ValueError(
                f"NodeTimerObserver: unsupported activity kind(s) {unknown}; "
                f"timing supports {sorted(_TIMED_FIELDS)}"
            )
        # CUPTI only emits MEMCPY2 (peer-to-peer) records when MEMCPY is also enabled;
        # MEMCPY2 alone silently yields nothing. Enable MEMCPY alongside it implicitly.
        kind_ints = {int(k) for k in self._kinds}
        memcpy, memcpy2 = int(ActivityKind.MEMCPY), int(ActivityKind.MEMCPY2)
        if memcpy2 in kind_ints and memcpy not in kind_ints:
            self._kinds += (ActivityKind.MEMCPY,)
        self._lock = threading.Lock()
        # Raw span column chunks as the worker thread delivers them: each is
        # (graph_node_id, start, end, stream, correlation_id|None -- the id only when
        # eager). Kept raw (not pre-aggregated) so consumers build per-kernel timing on top.
        self._chunks: list[tuple[Any, Any, Any, Any, Any]] = []
        # EXTERNAL_CORRELATION chunks (external_id, correlation_id) for the eager name
        # join; only populated when eager naming is on.
        self._ext_chunks: list[tuple[Any, Any]] = []
        # Base timing selection per kind. The base folds in the eager fields
        # (CORRELATION_ID + EXTERNAL_CORRELATION + RUNTIME) and sets up the resolver
        # from `annotations`. Register last so the buffers are ready before delivery.
        fields = {int(k): set(_TIMED_FIELDS[int(k)]) for k in self._kinds}
        super().__init__(fields, annotations=annotations)

    def _on_activities(self, columns: dict[Any, dict[int, Any]]) -> None:
        # Worker thread: just stash the columns (cheap append); grouping/joining
        # happens after drain, off this hot path.
        spans: list[tuple[Any, Any, Any, Any, Any]] = []
        exts: list[tuple[Any, Any]] = []
        for kind, cols in columns.items():
            k = int(kind)
            if k == _EXTERNAL:
                eid = cols.get(int(ExternalCorrelation.EXTERNAL_ID))
                corr = cols.get(int(ExternalCorrelation.CORRELATION_ID))
                if eid is not None and corr is not None:
                    if self._eager:
                        eid = self._resolve_named_ancestor(eid)
                    exts.append((eid, corr))
                continue
            spec = _TIMED_FIELDS.get(k)
            if spec is None:
                continue
            sf, ef, gf, stf = spec
            start, end, gnode, stream = (
                cols.get(sf),
                cols.get(ef),
                cols.get(gf),
                cols.get(stf),
            )
            if start is None or end is None or gnode is None or stream is None:
                continue
            corr = cols.get(CORRELATION_FIELD[k]) if self._eager else None
            spans.append((gnode, start, end, stream, corr))
        if spans or exts:
            with self._lock:
                self._chunks.extend(spans)
                self._ext_chunks.extend(exts)

    def _resolve_named_ancestor(self, eid_col):
        """Map each (innermost) external id to the innermost ENCLOSING id that names a
        region, via the monitor's active-id chain -- so a kernel nested below a named
        region (e.g. a collective inside it) still resolves to that region. Resolved
        at dispatch, while the chain is live (it is dropped shortly after the pop). The
        innermost id is returned as-is when no named region encloses it (incl. the
        common non-nested case) or no monitor is available."""
        names = set(self.annotation_names())
        if not names or self._monitor is None:
            return eid_col
        import numpy as np

        chain = self._monitor.external_id_chain
        return np.asarray(
            [
                next((c for c in reversed(chain(int(e))) if c in names), int(e))
                for e in eid_col.tolist()
            ],
            dtype=eid_col.dtype,
        )

    def _take(self) -> tuple[list, list]:
        with self._lock:
            chunks, self._chunks = self._chunks, []
            ext_chunks, self._ext_chunks = self._ext_chunks, []
        return chunks, ext_chunks

    def drain(self, flush: bool = False) -> tuple[Any, Any, Any, Any]:
        """Return the raw per-activity spans delivered since the last call as four
        parallel numpy columns ``(graph_node_id, start_ns, end_ns, stream_id)`` (dtypes
        ``<u8, <i8, <i8, <u8``), and reset. Empty -> four length-0 arrays.

        Flat and vectorized on purpose: consumers bucket/aggregate with numpy
        (e.g. ``bincount``/``searchsorted``) without a Python loop over the (~50k)
        kernels. A node's duration is ``end - start``; eager activities share
        ``graph_node_id == 0``.

        Cheap by default: reads only what the worker thread has already delivered,
        so it must NOT force a synchronous flush on every call (records still
        buffered are picked up on the next drain). Pass ``flush=True`` to nudge
        CUPTI to hand over completed buffers first -- a plain ``cuptiActivityFlushAll``,
        not a sync fence (the timer is a rolling consumer; stragglers are
        picked up on the next drain)."""
        import numpy as np

        if flush and self._monitor is not None:
            self._monitor.flush()
        chunks, _ = self._take()
        if not chunks:
            return (
                np.empty(0, dtype="<u8"),
                np.empty(0, dtype="<i8"),
                np.empty(0, dtype="<i8"),
                np.empty(0, dtype="<u8"),
            )
        gnode = np.concatenate([c[0] for c in chunks]).astype("<u8", copy=False)
        start = np.concatenate([c[1] for c in chunks]).astype("<i8", copy=False)
        end = np.concatenate([c[2] for c in chunks]).astype("<i8", copy=False)
        stream = np.concatenate([c[3] for c in chunks]).astype("<u8", copy=False)
        return gnode, start, end, stream

    def drain_annotated(self, flush: bool = False) -> dict[str, list[tuple[int, int]]]:
        """Resolve the spans delivered since the last call to region names and
        return ``{name: [(start_ns, end_ns), ...]}`` (and reset). Each span is named
        graph-first, eager-fallback: the graph resolver on ``graph_node_id`` for
        captured nodes, else the ``correlation_id -> external_id -> name`` join for
        eager regions. Spans that resolve to no name (incl. when no naming is enabled)
        go into the ``""`` bucket. ``flush`` behaves as in :meth:`drain`."""
        if flush and self._monitor is not None:
            self._monitor.flush()
        chunks, ext_chunks = self._take()
        names = self.annotation_names(reset=True)  # external_id -> name (this window)
        if not chunks:
            return {}
        import numpy as np

        gnode = np.concatenate([c[0] for c in chunks]).astype("<u8", copy=False)
        start = np.concatenate([c[1] for c in chunks]).astype("<i8", copy=False)
        end = np.concatenate([c[2] for c in chunks]).astype("<i8", copy=False)

        # Resolve a name per span ("" == unnamed), vectorized: name only the *unique*
        # graph nodes / correlation ids (a small set), then gather back to per-span.
        span_names = np.empty(len(gnode), dtype=object)
        span_names[:] = ""

        resolver = self._annotation_resolver
        if resolver is not None:
            uniq_g, inv_g = np.unique(gnode, return_inverse=True)
            g_names = np.array(
                [(resolver(int(g)) or "") if g else "" for g in uniq_g.tolist()],
                dtype=object,
            )
            span_names = g_names[inv_g]

        # Eager fallback for spans the graph resolver didn't name: correlation_id ->
        # external_id -> name. The external ids were resolved at dispatch through the
        # monitor's active-id chain to the innermost ENCLOSING named region (so a
        # collective nested in a region maps to the region, not its own untracked id);
        # here we just keep those that name a region (in `names`) and map their
        # correlation ids.
        if self._eager and names:
            corr = np.concatenate([c[4] for c in chunks]).astype("<u8", copy=False)
            corr_to_ext: dict[int, int] = {}
            for eid_col, corr_col in ext_chunks:
                for eid, c in zip(eid_col.tolist(), corr_col.tolist()):
                    if int(eid) in names:
                        corr_to_ext[int(c)] = int(eid)
            if corr_to_ext:
                uniq_c, inv_c = np.unique(corr, return_inverse=True)
                c_names = np.array(
                    [
                        names.get(corr_to_ext[ci], "") if ci in corr_to_ext else ""
                        for ci in (int(c) for c in uniq_c.tolist())
                    ],
                    dtype=object,
                )
                eager = c_names[inv_c]
                missing = span_names == ""
                span_names[missing] = eager[missing]

        # Bucket (start, end) by name -- group over the unique names (a small set),
        # selecting each name's spans with a numpy mask. Spans with no graph/eager
        # name fall into the "" bucket (nothing is dropped).
        out: dict[str, list[tuple[int, int]]] = {}
        all_names = span_names.astype(str)
        uniq_names, inv = np.unique(all_names, return_inverse=True)
        for i, nm in enumerate(uniq_names.tolist()):
            mask = inv == i
            out[nm] = list(zip(start[mask].tolist(), end[mask].tolist()))
        return out
