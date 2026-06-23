# mypy: allow-untyped-defs
"""Per-collective communication records from the CUPTI monitor.

``CommsObserver`` captures the collective kernels' GPU timing, correlates each to
its collective, and assembles a :class:`CommRecord` per collective by joining that
timing with the per-collective metadata the NCCL profiler plugin records.

**Timing correlation.** ``CONCURRENT_KERNEL`` spans are attributed to a collective
by the external-correlation id the comms hook pushes around the call
(``CommMonitorHook``, in ``torch.profiler._cupti.comms``): CUPTI emits an
``EXTERNAL_CORRELATION`` record linking that id to the kernel's ``correlation_id``.
All subsystems push on one external-correlation kind, so CUPTI tags the kernel with
the *innermost* active id, which need not be the collective's (a nested tracer/op
region may be pushed inside the collective before its kernel launches). The monitor
snapshots each id's full push chain (``external_id_chain``), so the observer maps a
kernel's innermost id through it and attributes the kernel to the collective id active
when that id was pushed (no kind filtering, no stream/timestamp pairing). External ids
do not survive CUDA-graph capture/replay, so
eager collectives are keyed by external id and CUDA-graph collectives by
``graph_node_id`` (the anchor registers each collective's replay-kernel node, named
downstream by a graph metadata resolver). To make CUPTI emit the
``EXTERNAL_CORRELATION`` records, both
``RUNTIME`` and ``DRIVER`` must be enabled as carriers (NCCL launches the collective
kernel via the DRIVER API).

**Metadata join + records.** A collective's metadata is recorded synchronously when
the call is issued (the plugin's ``startEvent``), before its kernel completes, so
metadata generally precedes timing. :meth:`CommsObserver.poll` absorbs newly-issued
metadata into an in-flight set, drains completed timing, and pairs each into a
``CommRecord``; collectives still in flight (issued, kernel not yet timed) stay in
the in-flight set (eager by external id, graph-replay by resolved start vs kernel
completion) -- an entry that never clears is a stuck collective. Completed records are
kept in a bounded ring and the per-collective lifecycle (schedule / start / end /
wait) is driven on any registered :class:`CommRecordPlugin` (:meth:`add_plugin`) --
the extension point for dump / slow-collective reporting (the consumers live in
``torch.profiler._cupti.comms``).

**Stall heartbeat.** When ``quiescence_timeout_s`` is set the observer runs a
background thread that drains on a cadence and, if no lifecycle callback has fired for
the timeout while collectives are still in flight, publishes the in-flight set via
``on_progress`` (after a plain suspicion flush to rule out an undelivered completion).
Any real callback resets the deadline; the :class:`HangDetectorPlugin` consumes this
heartbeat to flag hangs. ``on_progress`` is NOT fired by :meth:`poll`.

``CUDA_EVENT`` (the per-collective *start* signal) is captured when an
``event_resolver`` or ``start_events`` asks for it. At :meth:`poll` it drives
``on_start``: eager start events map to their external id by ``correlation_id``; graph
ones resolve via ``event_resolver`` (``event_id -> (coll_id, {graph_node_id},
metadata)``) and feed the graph in-flight accounting. This module buffers on the
worker thread and does the join at poll, off the hot path.
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
from collections.abc import Callable  # noqa: TC003
from dataclasses import dataclass, field
from typing import Any

from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

from torch.profiler._cupti.observers.base import CuptiMonitorObserver
from torch.profiler._cupti.records import Api, CudaEvent, ExternalCorrelation, Kernel


logger = logging.getLogger(__name__)

_KERNEL = int(ActivityKind.CONCURRENT_KERNEL)
_EXTERNAL = int(ActivityKind.EXTERNAL_CORRELATION)
_CUDA_EVENT = int(ActivityKind.CUDA_EVENT)
# Cap on the persistent correlation_id -> external_id map (oldest evicted).
_CORR_TO_EXT_MAX = 16384
# Cap on the recent-metadata cache that serves a late CPU wait (oldest evicted).
_WAIT_META_MAX = 16384


@dataclass
class CommRecord:
    """One collective's joined record: the NCCL plugin metadata (parsed) plus GPU
    timing. ``coll_id`` is the external-correlation id for eager collectives, or the
    ``graph_node_id`` for CUDA-graph-captured ones (kept separately too)."""

    coll_id: int
    name: str
    start_ns: int
    end_ns: int
    graph_node_id: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ns(self) -> int:
        return self.end_ns - self.start_ns


def _parse(blob: str | None) -> dict[str, Any]:
    """Parse a metadata blob (the plugin emits a JSON object) to a dict; ``{}`` for
    missing or malformed input -- the record stays usable as timing-only."""
    if not blob:
        return {}
    try:
        obj = json.loads(blob)
    except (ValueError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _join_record(
    timed: dict[str, Any],
    in_flight: dict[int, dict[str, Any]],
    resolver: Callable[[int], str | None] | None,
) -> CommRecord:
    """Build a :class:`CommRecord` from one timing record, attaching its metadata:
    eager by READING the in-flight set on ``external_id``; graph (no external id) by
    the ``graph_node_id`` resolver. Empty metadata when neither resolves (timing-only
    record). Does NOT pop ``in_flight`` -- :meth:`CommsObserver.poll` drops the entry
    after this read, keyed by the completing kernel's ``external_id``."""
    eid = int(timed["external_id"])
    gnode = int(timed["graph_node_id"])
    meta = in_flight.get(eid) if eid else None
    if meta is None and gnode and resolver is not None:
        meta = _parse(resolver(gnode))
    return CommRecord(
        coll_id=eid or gnode,
        name=timed["name"],
        start_ns=int(timed["start_ns"]),
        end_ns=int(timed["end_ns"]),
        graph_node_id=gnode,
        metadata=meta or {},
    )


class CommsObserver(CuptiMonitorObserver):
    """Builds per-collective :class:`CommRecord`\\ s from the monitor: buffers
    collective-kernel spans + external-correlation records on the worker thread, and
    at :meth:`poll` attributes each kernel to its collective, joins the NCCL metadata,
    and tracks in-flight (issued, not yet timed) and recently-completed collectives.
    See the module docstring.

    A kernel is kept only when the instrumentation marked it a collective: an external-
    correlation id carrying collective metadata (eager -- the comm hook or a
    :class:`CollectiveMeasurer` / ``SymmMemDispatchMode`` pushes one around the launch)
    or a registered graph node (graph capture, via ``_GraphCommAnchor``). There is no
    kernel-name heuristic: the multiplexer pushes a correlation id around every
    collective in scope, so the mark is authoritative and arbitrary other kernels are
    ignored. ``metadata_resolver``
    (``graph_node_id -> blob``) supplies metadata for CUDA-graph collectives.
    ``max_records`` bounds the completed-record ring. ``CUDA_EVENT`` (the per-collective
    device-start signal feeding ``on_start``) is captured when ``start_events`` or an
    ``event_resolver`` is set; ``event_resolver`` (``event_id -> (coll_id,
    {graph_node_id}, metadata)``, also settable via :meth:`set_event_resolver`) resolves
    graph start events and drives the graph in-flight accounting.

    ``quiescence_timeout_s`` (with ``quiescence_interval_s``) enables the stall-heartbeat
    thread (see the module docstring): set it for hang detection, leave it ``None`` for a
    purely poll-driven observer.

    Single metadata consumer: :meth:`poll` drains the process-global metadata store
    (``take_external_metadata``), so a ``CommsObserver`` and a chrome-trace
    ``ProfilerObserver`` cannot both consume the eager metadata in one process."""

    def __init__(
        self,
        *,
        metadata_resolver: Callable[[int], str | None] | None = None,
        max_records: int = 1024,
        start_events: bool = False,
        event_resolver: Callable[
            [int], tuple[int, frozenset[int], dict[str, Any]] | None
        ]
        | None = None,
        quiescence_timeout_s: float | None = None,
        quiescence_interval_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._meta_resolver = metadata_resolver
        self._event_resolver = event_resolver
        # Quiescence (stall) detection: when a timeout is set the observer runs its own
        # background thread that drains on a cadence and -- if no lifecycle callback has
        # fired for the timeout while collectives are still in flight -- publishes the
        # in-flight set via on_progress (the stall heartbeat a hang detector consumes).
        # Any real callback resets the deadline; see _quiescence_tick.
        self._quiescence_timeout_s = quiescence_timeout_s
        self._quiescence_interval_s = quiescence_interval_s
        self._clock = clock
        self._last_activity = clock()
        # A suspected stall has been plain-flushed once and is awaiting the next tick's
        # verdict (flush delivery is async). Reset whenever progress resumes.
        self._suspected = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Capture CUDA_EVENT (the per-collective start signal) when any consumer wants
        # it: an explicit opt-in or a graph start-event resolver (graph mode).
        self._capture_events = start_events or event_resolver is not None
        self._lock = threading.Lock()
        # kernel = (start, end, correlation_id, graph_node, name);
        # ext = (external_id, correlation_id).
        self._kernels: list[tuple[Any, Any, Any, Any, Any]] = []
        self._ext: list[tuple[Any, Any]] = []
        # CUDA_EVENT (start) records delivered: (event_id col, correlation_id col). The
        # observer is their sole consumer -- poll() fires on_start off them (eager by
        # correlation_id, graph by event_resolver) and clears them.
        self._cuda_events: list[tuple[Any, Any]] = []
        # Persistent, bounded external_id-by-correlation_id map, fed every drain from
        # the EXTERNAL_CORRELATION records, so both the kernel correlation and the
        # eager CUDA_EVENT on_start path can read it. Capped to bound memory.
        self._corr_to_ext: collections.OrderedDict[int, int] = collections.OrderedDict()
        # external_id -> parsed metadata for collectives whose kernel hasn't been timed
        # yet (in flight): the plugin records metadata at the call, before the kernel
        # completes, so it is buffered here until the matching timing arrives.
        self._in_flight: dict[int, dict[str, Any]] = {}
        # Graph-replay in-flight accounting (external ids don't survive capture, so a
        # graph collective is tracked by its resolved start events vs its kernels'
        # completions, keyed by graph_node_id). A collective is in flight while any of
        # its kernel nodes has more starts than completions; counting (not timestamps)
        # tolerates start/kernel records arriving in either order. Populated only in
        # graph mode (an event_resolver is set).
        self._graph_start_count: dict[int, int] = {}
        self._graph_coll_nodes: dict[int, frozenset[int]] = {}
        self._graph_complete_count: dict[int, int] = {}
        self._graph_meta: dict[int, dict[str, Any]] = {}
        # external_ids that have already received on_schedule, so each fires exactly
        # once and always before its on_end.
        self._scheduled: set[int] = set()
        # Optional source of CPU-waited external ids (the comms hook's drain_waits);
        # set via set_wait_source. poll() drains it and fires on_wait for each.
        self._wait_source: Callable[[], list[int]] | None = None
        # Bounded recent-metadata cache keyed by external_id, so a CPU wait arriving
        # after the collective has completed -- i.e. after _in_flight dropped it --
        # still has metadata for on_wait. Seeded on every on_schedule / on_end fire;
        # oldest evicted past the cap.
        self._wait_meta: collections.OrderedDict[int, dict[str, Any]] = (
            collections.OrderedDict()
        )
        self._past: collections.deque[CommRecord] = collections.deque(
            maxlen=max_records
        )
        # Plugins are duck-typed (the CommRecordPlugin lifecycle hooks); the base lives
        # in the comms package, so the observer (the producer) doesn't import it.
        self._plugins: list[Any] = []
        fields: dict[Any, set[Any]] = {
            ActivityKind.CONCURRENT_KERNEL: {
                Kernel.START,
                Kernel.END,
                Kernel.CORRELATION_ID,
                Kernel.GRAPH_NODE_ID,
                Kernel.NAME,
            },
            ActivityKind.EXTERNAL_CORRELATION: {
                ExternalCorrelation.EXTERNAL_ID,
                ExternalCorrelation.CORRELATION_ID,
            },
            # Carriers so CUPTI emits EXTERNAL_CORRELATION for the kernel's launch;
            # NCCL launches collective kernels via the DRIVER API, so DRIVER is required.
            ActivityKind.RUNTIME: {Api.CORRELATION_ID},
            ActivityKind.DRIVER: {Api.CORRELATION_ID},
        }
        if self._capture_events:
            # The monitor couples SYNCHRONIZATION on automatically (CUPTI only emits
            # CUDA_EVENT when it is enabled). DEVICE_TIMESTAMP is omitted: it is
            # disabled by default and cuptiActivityEnableCudaEventDeviceTimestamps is
            # NOT_COMPATIBLE under the UDR subscriber, so it would only ever read 0.
            # eventId resolves a graph start event to its collective; correlation_id
            # maps an eager start event to its external id.
            fields[ActivityKind.CUDA_EVENT] = {
                CudaEvent.EVENT_ID,
                CudaEvent.CORRELATION_ID,
            }
        super().__init__(fields)

    def start(self) -> None:
        """Start the background quiescence thread (idempotent; no-op without a configured
        ``quiescence_timeout_s`` or when the monitor is unavailable). The thread becomes
        the observer's poll driver, so callers must not also poll concurrently -- read
        completed records via :meth:`past`.

        NOT auto-started at construction: the thread flushes CUPTI on its cadence, which
        must not race process/comm initialization (e.g. ``ncclCommInitRank``). Call this
        once the comm is up and collectives are flowing."""
        if (
            self._quiescence_timeout_s is None
            or self._thread is not None
            or self._monitor is None
        ):
            return
        self._stop.clear()
        self._last_activity = self._clock()
        self._thread = threading.Thread(
            target=self._loop, name="comms-observer", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background quiescence thread (idempotent)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._quiescence_interval_s + 5.0)
            self._thread = None

    def close(self) -> None:
        """Stop the quiescence thread, then unregister from the monitor."""
        self.stop()
        super().close()

    def _loop(self) -> None:
        while not self._stop.wait(self._quiescence_interval_s):
            try:
                self._quiescence_tick()
            except Exception:
                logger.exception("comms observer quiescence tick failed")

    def _quiescence_tick(self) -> None:
        """One background cycle: drain + dispatch; if the observer has gone quiet for the
        timeout while collectives are still in flight, plain-flush once to nudge any
        completion sitting in an undelivered buffer, then -- a tick later, since CUPTI's
        delivery is async -- if still quiet, publish the in-flight set via on_progress."""
        self.poll()
        if not self._stalled():
            self._suspected = False
            return
        if not self._suspected:
            # First detection: a completion may merely be undelivered. Plain-flush to
            # nudge it (never forced/sync -- the device may be hung). Delivery is async,
            # so defer the verdict to the next tick; a real completion lands in the next
            # poll(), fires on_end, and resets the deadline -- clearing the stall.
            self._suspected = True
            self.poll(flush=True)
            return
        # Still stalled a tick after the suspicion flush -> a real stall.
        self._fire_progress()

    def _stalled(self) -> bool:
        return (
            self._quiescence_timeout_s is not None
            and self._clock() - self._last_activity >= self._quiescence_timeout_s
            and bool(self.in_flight())
        )

    def _fire_progress(self) -> None:
        """Publish the in-flight set as the stall heartbeat. Unlike the lifecycle
        callbacks this does NOT reset the quiescence deadline (it IS the stall signal),
        so it re-fires each cadence while the stall persists -- a hang detector dedups."""
        snapshot = self.in_flight()
        for plugin in self._plugins:
            fn = getattr(plugin, "on_progress", None)
            if fn is not None:
                fn(snapshot)

    def set_event_resolver(
        self,
        fn: Callable[[int], tuple[int, frozenset[int], dict[str, Any]] | None],
    ) -> None:
        """Set the graph start-event resolver (``event_id -> (coll_id, {graph_node_id},
        metadata)``). The anchor's resolver is only populated after capture/finalize, so
        a caller that built the observer before then wires it here."""
        self._event_resolver = fn

    def set_metadata_resolver(self, fn: Callable[[int], str | None]) -> None:
        """Set the ``graph_node_id -> blob`` resolver used to attach metadata to graph
        collectives' on_end (E). Like the event resolver, the anchor populates it only
        after capture/finalize, so a caller that built the observer before then wires it
        here."""
        self._meta_resolver = fn

    def set_wait_source(self, drain: Callable[[], list[int]]) -> None:
        """Set the CPU-wait source -- the comms hook's :meth:`CommMonitorHook.drain_waits`
        (``() -> list[external_id]``). :meth:`poll` drains it and fires ``on_wait`` for
        each waited collective. Without it, ``on_wait`` never fires."""
        self._wait_source = drain

    def add_plugin(self, plugin: Any) -> None:
        """Register a ``CommRecordPlugin`` (from ``torch.profiler._cupti.comms``)
        notified of each collective's lifecycle on every :meth:`poll` (schedule, end,
        progress -- watchdog, dump, slow-collective reporting)."""
        self._plugins.append(plugin)

    def _fire(self, method: str, *args: Any) -> None:
        # Any lifecycle callback is forward progress -- reset the quiescence deadline.
        # (The stall heartbeat on_progress goes through _fire_progress, which
        # deliberately does NOT, since it IS the signal that progress has stopped.)
        self._last_activity = self._clock()
        for plugin in self._plugins:
            getattr(plugin, method)(*args)

    def _seed_wait_meta(self, external_id: int, metadata: dict[str, Any]) -> None:
        """Cache an eager collective's metadata under its external id so a CPU wait
        arriving after completion still has it for on_wait. Eviction is FIFO past the
        cap."""
        if not external_id:
            return
        self._wait_meta[external_id] = metadata
        self._wait_meta.move_to_end(external_id)
        while len(self._wait_meta) > _WAIT_META_MAX:
            self._wait_meta.popitem(last=False)

    def _on_activities(self, columns: dict[Any, dict[int, Any]]) -> None:
        kernels: list[tuple[Any, Any, Any, Any, Any]] = []
        ext: list[tuple[Any, Any]] = []
        cuda_events: list[tuple[Any, Any]] = []
        for kind, cols in columns.items():
            k = int(kind)
            if k == _KERNEL:
                start = cols.get(int(Kernel.START))
                end = cols.get(int(Kernel.END))
                corr = cols.get(int(Kernel.CORRELATION_ID))
                name = cols.get(int(Kernel.NAME))
                gnode = cols.get(int(Kernel.GRAPH_NODE_ID))
                if start is not None and end is not None and corr is not None:
                    kernels.append((start, end, corr, gnode, name))
            elif k == _EXTERNAL:
                eid = cols.get(int(ExternalCorrelation.EXTERNAL_ID))
                corr = cols.get(int(ExternalCorrelation.CORRELATION_ID))
                if eid is not None and corr is not None:
                    ext.append((eid, corr))
            elif k == _CUDA_EVENT:
                ev = cols.get(int(CudaEvent.EVENT_ID))
                corr = cols.get(int(CudaEvent.CORRELATION_ID))
                if ev is not None:
                    cuda_events.append((ev, corr))
        if kernels or ext or cuda_events:
            with self._lock:
                self._kernels.extend(kernels)
                self._ext.extend(ext)
                self._cuda_events.extend(cuda_events)

    def _take(self):
        with self._lock:
            kernels, self._kernels = self._kernels, []
            ext, self._ext = self._ext, []
        self._update_corr_to_ext(ext)
        return kernels, ext

    def _update_corr_to_ext(self, ext_chunks: list[tuple[Any, Any]]) -> None:
        """Fold this drain's EXTERNAL_CORRELATION records into the persistent, bounded
        ``correlation_id -> external_id`` map (the kernel correlation and the eager
        CUDA_EVENT on_start path both read it)."""
        import numpy as np

        m = self._corr_to_ext
        for eid_col, corr_col in ext_chunks:
            eids = np.asarray(eid_col).tolist()
            corrs = np.asarray(corr_col).tolist()
            for eid, corr in zip(eids, corrs):
                m[int(corr)] = int(eid)
        while len(m) > _CORR_TO_EXT_MAX:
            m.popitem(last=False)

    def _take_cuda_events(self) -> list[tuple[int, int]]:
        """The (event_id, correlation_id) of CUDA_EVENT (start) records delivered since
        the last call, reset. Internal to :meth:`poll`'s on_start path -- the observer
        is the sole consumer."""
        import numpy as np

        with self._lock:
            chunks, self._cuda_events = self._cuda_events, []
        out: list[tuple[int, int]] = []
        for ev_col, corr_col in chunks:
            evs = np.asarray(ev_col).tolist()
            corrs = (
                np.asarray(corr_col).tolist()
                if corr_col is not None
                else [0] * len(evs)
            )
            out.extend((int(e), int(c)) for e, c in zip(evs, corrs))
        return out

    def drain_collectives(self, flush: bool = False) -> list[dict[str, Any]]:
        """Low-level raw timing: attribute each collective kernel to its collective and
        return ``[{external_id, start_ns, end_ns, graph_node_id, name}, ...]`` (and
        reset). Eager collectives carry the pushed ``external_id``; CUDA-graph
        collectives carry ``external_id == 0`` and a stable ``graph_node_id``.
        ``flush`` nudges CUPTI to hand over completed buffers first (a plain
        ``cuptiActivityFlushAll``, not a sync fence -- never a forced flush, which
        would deliver and consume in-progress records). :meth:`poll` builds CommRecords
        on top of this; use this directly only if you want timing without metadata."""
        if flush and self._monitor is not None:
            self._monitor.flush()
        kernels, _ext = self._take()  # _take folds ext into the persistent map
        if not kernels:
            return []
        # A kernel is a collective iff the instrumentation marked it: an external id
        # carrying metadata (CollectiveMeasurer / a hook -- folded into _in_flight) or a
        # registered graph node. Prefer the metadata resolver as the registered-node set:
        # the anchor fills it at finalize(), before the first replay, so a graph kernel
        # that drains in the same poll as its start event is still kept. Fall back to the
        # lazy set (_graph_coll_nodes) only when no resolver is wired -- it lags, since
        # poll fills it from start events after this drain.
        keep_graph_nodes: Any = (
            _ResolverNodes(self._meta_resolver)
            if self._meta_resolver is not None
            else frozenset(
                n for nodes in self._graph_coll_nodes.values() for n in nodes
            )
        )
        attribute = _attribution(
            self._corr_to_ext,
            keep_ext_ids=frozenset(self._in_flight),
            keep_graph_nodes=keep_graph_nodes,
            chain_resolver=(
                self._monitor.external_id_chain if self._monitor is not None else None
            ),
        )
        return _correlate_kernels(kernels, attribute)

    def poll(self, flush: bool = False) -> list[CommRecord]:
        """Absorb newly-issued collective metadata, pair completed timing with it into
        :class:`CommRecord`\\ s (appended to the past ring), drive the per-collective
        plugin lifecycle (``on_schedule`` / ``on_start`` / ``on_end`` / ``on_wait``), and
        return the records completed since the last poll. ``flush`` nudges CUPTI to
        deliver completed buffers first. For a deterministic boundary call
        ``monitor.flush(sync=True)`` before polling. The stall heartbeat ``on_progress``
        is NOT fired here -- it is published by the background quiescence thread (see
        :meth:`_quiescence_tick`) only after the timeout elapses without progress."""
        if self._monitor is not None:
            for eid, blob in self._monitor.take_external_metadata().items():
                self._in_flight[int(eid)] = _parse(blob)
        completed: list[CommRecord] = []
        for timed in self.drain_collectives(flush=flush):
            record = _join_record(timed, self._in_flight, self._meta_resolver)
            eid = int(timed["external_id"])
            gnode = int(timed["graph_node_id"])
            # Schedule-before-end: a collective issued and completed in the same poll
            # still gets on_schedule first (graph collectives have eid 0 -> end only).
            if eid and eid not in self._scheduled:
                meta = self._in_flight.get(eid, record.metadata)
                self._seed_wait_meta(eid, meta)
                self._fire("on_schedule", eid, meta)
                self._scheduled.add(eid)
            # An eager collective clears from in-flight on its kernel's REAL completion
            # -- drain_collectives yields only non-zero-end records. A hung collective's
            # kernel never completes, so it lingers -> the stall heartbeat surfaces it.
            if eid:
                self._in_flight.pop(eid, None)
                self._scheduled.discard(eid)
            elif gnode:
                # Graph completion: a replay kernel finished -> bump this node's count so
                # the matching outstanding start clears from the graph in-flight view.
                self._graph_complete_count[gnode] = (
                    self._graph_complete_count.get(gnode, 0) + 1
                )
            self._past.append(record)
            completed.append(record)
            # eid present iff eager; seed metadata so a late wait still resolves it.
            self._seed_wait_meta(record.coll_id if eid else 0, record.metadata)
            self._fire("on_end", record)
        # Start events (kernel began on device). CUDA_EVENT (start) and the kernel's
        # CONCURRENT_KERNEL (end) arrive in undefined order, so on_start may follow
        # on_end -- it still fires (plugins no-op a late start on a completed entry).
        # Per record (no once-suppression): graph replays repeat the same id.
        for event_id, correlation_id in self._take_cuda_events():
            ext = self._corr_to_ext.get(int(correlation_id)) if correlation_id else None
            if ext:
                # Eager. Schedule-before-start only if still in flight (unscheduled and
                # not yet completed) -- a late start after on_end must not re-schedule.
                if ext not in self._scheduled and ext in self._in_flight:
                    self._scheduled.add(ext)
                    self._fire("on_schedule", ext, self._in_flight[ext])
                self._fire("on_start", ext, self._in_flight.get(ext, {}))
            elif self._event_resolver is not None:
                role = self._event_resolver(int(event_id))
                if role is not None:
                    coll_id, nodes, meta = role
                    self._fire("on_start", coll_id, meta)
                    # Graph start: bump this collective's start count and record its
                    # kernel nodes + metadata so in_flight() can tell it's outstanding
                    # until each node's completion count catches up.
                    self._graph_start_count[coll_id] = (
                        self._graph_start_count.get(coll_id, 0) + 1
                    )
                    self._graph_coll_nodes[coll_id] = nodes
                    if meta:
                        self._graph_meta[coll_id] = meta
        # Newly-issued, still-in-flight collectives: schedule each exactly once.
        for eid, meta in self._in_flight.items():
            if eid not in self._scheduled:
                self._scheduled.add(eid)
                self._seed_wait_meta(eid, meta)
                self._fire("on_schedule", eid, meta)
        # CPU waits (work.wait()) recorded by the comms hook's per-work wait hook: one
        # on_wait per wait occurrence, with the collective's cached metadata if known.
        if self._wait_source is not None:
            for eid in self._wait_source():
                self._fire("on_wait", eid, self._wait_meta.get(eid, {}))
        return completed

    def in_flight(self) -> dict[int, dict[str, Any]]:
        """Collectives issued/running but not yet completed -- the current/pending set
        the stall heartbeat (``on_progress``) publishes; an entry that never clears is a
        stuck collective. Eager collectives are keyed by their external id; graph-replay
        collectives (a resolved start with no matching kernel completion yet) by their
        resolver ``coll_id``."""
        snapshot = dict(self._in_flight)
        for coll_id, starts in self._graph_start_count.items():
            nodes = self._graph_coll_nodes.get(coll_id, frozenset())
            if any(starts > self._graph_complete_count.get(gn, 0) for gn in nodes):
                snapshot[coll_id] = self._graph_meta.get(coll_id, {})
        return snapshot

    def past(self) -> list[CommRecord]:
        """The most recent completed CommRecords (bounded by ``max_records``)."""
        return list(self._past)


class _ResolverNodes:
    """A membership view over a ``graph_node_id -> blob`` resolver (the anchor's,
    populated at ``finalize()`` before any replay): ``node in self`` iff the resolver
    has it. Lets the kernel keep treat registered graph nodes as a set without
    enumerating the resolver, which is a callable, not an iterable."""

    __slots__ = ("_resolver",)

    def __init__(self, resolver: Callable[[int], str | None]) -> None:
        self._resolver = resolver

    def __contains__(self, node: int) -> bool:
        return self._resolver(int(node)) is not None


def _attribution(
    corr_to_ext: dict[int, int] | Any,
    *,
    keep_ext_ids: frozenset[int] = frozenset(),
    keep_graph_nodes: Any = frozenset(),
    chain_resolver: Callable[[int], tuple[int, ...]] | None = None,
) -> Callable[[int, int], int | None]:
    """Build the per-kernel keep + attribution fn ``(correlation_id, graph_node_id) ->
    external_id``: the collective external id to record (``0`` for a CUDA-graph
    collective, keyed by its node), or ``None`` to drop. A graph-replay kernel
    (``graph_node_id != 0``) carries no external id, so it is kept iff its node is a
    registered collective (``keep_graph_nodes`` -- any container, e.g. a set or a
    resolver-backed membership view). An eager kernel (``graph_node_id == 0``) is kept
    iff a collective id (``keep_ext_ids``) was active when its id was pushed: CUPTI tags
    a kernel with only the *innermost* active id, which need not be the collective's
    (a nested tracer/op region may be pushed inside the collective), so we resolve the
    innermost id's full push chain (``chain_resolver`` -- the monitor's
    ``external_id_chain``; without one the chain is just the innermost id) and attribute
    to the innermost collective id in it. There is no name heuristic, the mark is
    authoritative.

    ``corr_to_ext`` is the prebuilt (persistent) ``correlation_id -> external_id`` map;
    a list of ``(external_id col, correlation_id col)`` chunks is also accepted (it is
    folded into a local map) for direct callers/tests."""
    import numpy as np

    if not isinstance(corr_to_ext, dict):
        chunks, corr_to_ext = corr_to_ext, {}
        for eid_col, corr_col in chunks:
            eids = np.asarray(eid_col).tolist()
            corrs = np.asarray(corr_col).tolist()
            for eid, corr in zip(eids, corrs):
                corr_to_ext[int(corr)] = int(eid)

    def attribute(correlation_id: int, graph_node_id: int) -> int | None:
        if graph_node_id:
            return 0 if graph_node_id in keep_graph_nodes else None
        innermost = corr_to_ext.get(correlation_id)
        if innermost is None:
            return None
        chain = (
            chain_resolver(innermost) if chain_resolver is not None else (innermost,)
        )
        for cid in reversed(chain):
            if cid in keep_ext_ids:
                return cid
        return None

    return attribute


def _correlate_kernels(
    kernel_chunks: list[tuple[Any, Any, Any, Any, Any]],
    attribute: Callable[[int, int], int | None],
) -> list[dict[str, Any]]:
    """Build raw timing dicts for the collective kernels in ``kernel_chunks``, attributing
    each to its collective via ``attribute`` (see :func:`_attribution`): kept kernels get
    ``external_id`` (the collective's, or ``0`` for a graph collective keyed by
    ``graph_node_id``); kernels ``attribute`` maps to ``None`` are dropped."""
    import numpy as np

    out: list[dict[str, Any]] = []
    for start_col, end_col, corr_col, gnode_col, name_col in kernel_chunks:
        n = len(start_col)
        names = name_col if name_col is not None else [""] * n
        gnodes = np.asarray(gnode_col).tolist() if gnode_col is not None else [0] * n
        starts = np.asarray(start_col).tolist()
        ends = np.asarray(end_col).tolist()
        corrs = np.asarray(corr_col).tolist()
        for i in range(n):
            if int(ends[i]) == 0:
                # Defensive: a record with a zero end timestamp is an in-progress
                # kernel (not yet run to completion), not a completion -- counting one
                # would mask a hang. The monitor only ever plain-flushes (completed
                # records), so this should not occur; the guard costs nothing.
                continue
            gnode = int(gnodes[i])
            ext_id = attribute(int(corrs[i]), gnode)
            if ext_id is None:
                continue
            out.append(
                {
                    "external_id": ext_id,
                    "start_ns": int(starts[i]),
                    "end_ns": int(ends[i]),
                    "graph_node_id": gnode,
                    "name": str(names[i]),
                }
            )
    return out
