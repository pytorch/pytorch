# mypy: allow-untyped-defs
"""Per-collective external-correlation tagging for torchcomms.

:class:`CommMonitorHook` registers on a torchcomms ``TorchComm`` via its native
pre/post hooks so each collective call is bracketed by a unique external-
correlation push/pop -- the same push/pop the user-annotation path uses, mirroring
how ``SubgraphTimer.measure()`` brackets a region. The push does two things for the
per-collective metadata join:

  * NCCL's profiler plugin fires its ``startEvent`` synchronously inside the
    collective call (same thread, inside the push/pop window), reads the pushed id
    via ``cuptiMonitorCurrentExternalId()``, and stores the collective's metadata
    blob (func/algo/proto/...) under it -- so the id is the per-collective key.
  * CUPTI emits an EXTERNAL_CORRELATION record linking that id to the launched
    kernels' correlation ids.

The monitor then joins blob + GPU timing per collective. The hook also reads
host-side metadata (input/output sizes + dtypes, peer/root, comm identity) from
the typed pre-hook args -- the fields the NCCL descriptor can't supply -- and
merges them onto the same id (see :func:`_hook_metadata`).

External ids do not survive CUDA-graph capture/replay, so the eager push/pop
can't key a captured collective. When ``watchdog`` is enabled, a
:class:`_GraphCommAnchor` instead, for each *captured* collective, records a
single external CUDA *start* event (off the collective stream's critical path via
:class:`~torch.profiler._cupti.utils.graph_side_stream.GraphSideStream`) and
node-walks the capture to learn the collective kernel's ``graph_node_id``(s). The
collective's *completion* is the kernel's own ``CONCURRENT_KERNEL`` activity
record -- no end event needed. The observer's graph in-flight accounting flags
such a collective (a resolved start with no kernel ``graph_node_id`` completion
stays in flight; a hung collective's kernel never completes), and the
``HangDetectorPlugin`` reports it at quiescence. This module imports nothing from
torchcomms (it duck-types the comm + hook args), so it adds no dependency and is a
no-op when the monitor isn't running.
"""

from __future__ import annotations

import functools
import json
import logging
import threading
import traceback
from contextlib import contextmanager
from typing import Any

from torch.profiler._cupti.utils.graph_nodes import HAVE_NODE_TOOLS_ID, remap_to_exec


try:
    from cuda.bindings import runtime as _rt  # pyrefly: ignore[missing-import]
except ImportError:
    _rt: Any = None


logger = logging.getLogger(__name__)


def _check(ret: Any) -> Any:
    """Unwrap a ``cuda.bindings`` ``(err, *rest)`` return, raising on error."""
    err, *rest = ret if isinstance(ret, (tuple, list)) else (ret,)
    if err != _rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cuda.bindings runtime call failed: {err}")
    return rest[0] if len(rest) == 1 else tuple(rest)


# --- Capture-time node walk: the kernel/memcpy nodes a collective adds to the graph,
# and their cudaGraphNodeGetToolsId (== the CUPTI profiler's graph_node_id). A focused
# port of torch.cuda._graph_annotations; used only by _GraphCommAnchor below.


def _capture_state(stream_handle: int):
    """``(graph, frontier_nodes)`` for ``stream_handle`` (an int cudaStream_t) if
    actively capturing, else ``None``."""
    h = _rt.cudaStream_t(init_value=stream_handle)
    status, _id, graph, deps, _edge, num = _check(_rt.cudaStreamGetCaptureInfo(h))
    if status != _rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        return None
    return graph, list(deps[:num])


def _dependents(node: Any) -> list[Any]:
    _, _, num = _check(_rt.cudaGraphNodeGetDependentNodes(node))
    if num == 0:
        return []
    deps, _edge, num = _check(
        _rt.cudaGraphNodeGetDependentNodes(node, pNumDependentNodes=num)
    )
    return list(deps[:num])


def _root_nodes(graph: Any) -> list[Any]:
    _, num = _check(_rt.cudaGraphGetRootNodes(graph))
    if num == 0:
        return []
    roots, num = _check(_rt.cudaGraphGetRootNodes(graph, pNumRootNodes=num))
    return list(roots[:num])


def _collect_descendants(start_nodes, existing, include_start=False) -> dict[int, Any]:
    """New descendant nodes reachable from ``start_nodes``, skipping the edges in
    ``existing`` (node key -> set of direct-dependent keys present at entry)."""
    seen = {int(n) for n in start_nodes}
    out: dict[int, Any] = {}
    if include_start:
        for n in start_nodes:
            out[int(n)] = n
    stack = list(start_nodes)
    while stack:
        node = stack.pop()
        old = existing.get(int(node), set())
        for dep in _dependents(node):
            k = int(dep)
            if k in old or k in seen:
                continue
            seen.add(k)
            out[k] = dep
            stack.append(dep)
    return out


def _annotatable_types() -> set:
    return {
        _rt.cudaGraphNodeType.cudaGraphNodeTypeKernel,
        _rt.cudaGraphNodeType.cudaGraphNodeTypeMemcpy,
    }


class _CollectiveKernelScope:
    """Capture-time scope that records the kernel/memcpy nodes added between
    ``__enter__`` and ``__exit__`` on ``stream`` and exposes their (capture)
    ``graph_node_id`` toolsIds via :attr:`tools_ids`. A no-op (empty) when not
    capturing or when ``cudaGraphNodeGetToolsId`` is unavailable. Used only by
    :class:`_GraphCommAnchor`."""

    def __init__(self, stream: Any) -> None:
        self._stream_handle = int(stream.cuda_stream)
        self._graph: Any = None
        self._frontier: list[Any] = []
        self._existing: dict[int, set[int]] = {}
        self._entry_roots: set[int] | None = None
        self.tools_ids: list[int] = []

    def __enter__(self) -> _CollectiveKernelScope:  # noqa: PYI034
        if not HAVE_NODE_TOOLS_ID:
            return self
        state = _capture_state(self._stream_handle)
        if state is None:
            return self
        self._graph, self._frontier = state
        self._existing = {
            int(n): {int(d) for d in _dependents(n)} for n in self._frontier
        }
        if not self._frontier:
            self._entry_roots = {int(n) for n in _root_nodes(self._graph)}
        return self

    def __exit__(self, *exc: Any) -> None:
        if not HAVE_NODE_TOOLS_ID or self._graph is None:
            return
        if self._frontier:
            nodes = _collect_descendants(self._frontier, self._existing)
        else:
            new_roots = [
                n
                for n in _root_nodes(self._graph)
                if int(n) not in (self._entry_roots or set())
            ]
            nodes = _collect_descendants(new_roots, {}, include_start=True)
        annotatable = _annotatable_types()
        for node in nodes.values():
            if _check(_rt.cudaGraphNodeGetType(node)) in annotatable:
                self.tools_ids.append(_check(_rt.cudaGraphNodeGetToolsId(node)))


class _GraphCommAnchor:
    """Anchors per-collective hang-detection signals into a captured CUDA graph for the
    observer's graph in-flight accounting (where the eager external-correlation id is
    gone and all collectives share one ``cudaGraphLaunch``).

    Per captured collective, :meth:`collective` records a single external CUDA
    *start* event (off the collective kernel's critical path via
    :class:`GraphSideStream`) and node-walks the capture to find the collective
    kernel's ``graph_node_id``(s). After capture, :meth:`finalize` remaps those to
    the exec graph and learns each start event's CUPTI ``eventId`` (a one-shot
    eager record -- CUPTI assigns the id at runtime and capture can't eager-record),
    filling the ``eventId -> (coll_id, {graph_node_id}, metadata)`` registry that
    :meth:`event_resolver` serves. The collective's *completion* signal is the
    kernel's own ``CONCURRENT_KERNEL`` record (the watchdog matches by
    ``graph_node_id``); a hung collective's kernel never completes."""

    def __init__(self, monitor: Any) -> None:
        from torch.profiler._cupti.utils.graph_side_stream import GraphSideStream

        self._monitor = monitor
        self._gss = GraphSideStream()
        # (start_event, coll_id, capture_tools_ids, metadata) awaiting finalize().
        self._pending: list[tuple[Any, int, list[int], dict[str, Any]]] = []
        # eventId -> (coll_id, frozenset(exec graph_node_ids), metadata).
        self._registry: dict[int, tuple[int, frozenset[int], dict[str, Any]]] = {}
        # exec graph_node_id -> metadata JSON, for the observer's on_end (E) join, which
        # keys by graph_node_id (not eventId). Built alongside _registry in finalize().
        self._node_metadata: dict[int, str] = {}
        self._next_coll_id = 0

    @contextmanager
    def collective(self, stream: Any, metadata: dict[str, Any]):
        """Bracket one captured collective: record its start event and node-walk the
        kernel node(s) it adds to the graph."""
        coll_id = self._next_coll_id
        self._next_coll_id += 1
        start_event = self._gss.record_external_event(stream)
        scope = _CollectiveKernelScope(stream)
        scope.__enter__()
        try:
            yield
        finally:
            scope.__exit__(None, None, None)
            self._pending.append(
                (start_event, coll_id, list(scope.tools_ids), metadata)
            )

    def finalize(self, cuda_graph: Any) -> None:
        """Remap captured kernel ids to ``cuda_graph``'s exec graph and learn each
        start event's CUPTI ``eventId``. Call once after capture (the graph is
        instantiated) and before starting the watchdog; must NOT run during capture
        (the eager records would be captured)."""
        if not self._pending:
            return
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        import torch
        from torch.profiler._cupti.records import CudaEvent
        from torch.profiler._cupti.utils.graph_side_stream import eager_record

        exec_handle = cuda_graph.raw_cuda_graph_exec()
        seen: list[int] = []

        def _cb(columns: dict[Any, dict[int, Any]]) -> None:
            cols = columns.get(ActivityKind.CUDA_EVENT)
            if cols is not None and int(CudaEvent.EVENT_ID) in cols:
                seen.extend(int(v) for v in cols[int(CudaEvent.EVENT_ID)])

        obs = self._monitor.register(
            {ActivityKind.CUDA_EVENT: {CudaEvent.KIND, CudaEvent.EVENT_ID}}, _cb
        )
        stream = torch.cuda.current_stream()
        try:
            self._monitor.flush(sync=True)  # drain stragglers before mapping
            seen.clear()
            for start_event, _c, _t, _m in self._pending:
                eager_record(start_event, stream)
            torch.cuda.synchronize()
            self._monitor.flush(sync=True)
            if len(seen) != len(self._pending):
                logger.warning(
                    "comm watchdog finalize: expected %d start-event ids, got %d; "
                    "some graph collectives may be unwatched",
                    len(self._pending),
                    len(seen),
                )
            for (_start, coll_id, tools_ids, metadata), event_id in zip(
                self._pending, seen
            ):
                nodes = frozenset(remap_to_exec(tools_ids, exec_handle))
                self._registry[event_id] = (coll_id, nodes, metadata)
                blob = json.dumps(metadata)
                for node in nodes:
                    self._node_metadata[node] = blob
        finally:
            self._monitor.unregister(obs)
        self._pending.clear()

    def event_resolver(
        self, event_id: int
    ) -> tuple[int, frozenset[int], dict[str, Any]] | None:
        """Graph start-event resolver for the observer (``set_event_resolver``): map a
        replayed start event's id to ``(coll_id, {kernel graph_node_id}, metadata)``."""
        return self._registry.get(event_id)

    def metadata_resolver(self, graph_node_id: int) -> str | None:
        """Metadata-by-graph_node_id resolver for the observer's on_end (E) join
        (``set_metadata_resolver``): map a replayed kernel's ``graph_node_id`` to its
        captured collective's metadata blob."""
        return self._node_metadata.get(graph_node_id)

    def close(self) -> None:
        """Destroy the start events and the side stream."""
        self._gss.close()


def _as_tensor_list(t: Any) -> list[Any]:
    if t is None:
        return []
    return list(t) if isinstance(t, (list, tuple)) else [t]


def _tensor_dims(t: Any) -> list[list[int]]:
    return [list(x.shape) for x in _as_tensor_list(t)]


def _tensor_dtypes(t: Any) -> list[str]:
    return [str(x.dtype) for x in _as_tensor_list(t)]


def _comm_identity(comm: Any) -> dict[str, Any]:
    """The process-group fields a collective record carries, read from the comm once
    at registration: ``process_group`` ([name, desc]), this ``rank``, and the PG's
    global ranks (``process_group_ranks``, from ``comm.ranks``) so the FlightRecorder
    analyzer can compute expected-ranks and flag a rank that never issued a
    collective."""
    # torchcomms exposes the comm name via get_name() (== C++ getCommName(), what the
    # native clog uses, e.g. "...::split::0_DP_0"); the test fakes use a ``name`` attr.
    get_name = getattr(comm, "get_name", None)
    if callable(get_name):
        try:
            name = get_name() or "0"
        except Exception:
            name = "0"
    else:
        name = getattr(comm, "name", "0") or "0"
    pg: dict[str, Any] = {"process_group": [str(name), ""]}
    get_rank = getattr(comm, "get_rank", None)
    if callable(get_rank):
        try:
            pg["rank"] = get_rank()
        except Exception:
            pass
    try:
        ranks = getattr(comm, "ranks", None)
        if ranks is not None:
            pg["process_group_ranks"] = list(ranks)
    except Exception:
        pass
    return pg


def _hook_metadata(op_name: Any, args: Any, pg: dict[str, Any]) -> dict[str, Any]:
    """Per-collective metadata read from a torchcomms pre-hook's typed args: input/
    output tensor sizes + dtypes, peer/root, and is_p2p. The NCCL profiler plugin
    contributes the rest (func/algo/proto/...) onto the same id and wins on conflict
    (it runs after this), so these are the host-side fields it can't see -- and for
    p2p, which the plugin doesn't record at all, these stand alone."""
    meta: dict[str, Any] = dict(pg)
    # op_name is a torchcomms ``OpName`` enum (or a plain string); use the clean member
    # name. Recorded so consumers have the op for collectives the NCCL profiler plugin
    # doesn't describe (broadcast, p2p, ...); the plugin's "func" still wins when present.
    name = getattr(op_name, "name", None) or str(op_name)
    meta["name"] = name
    meta["is_p2p"] = "send" in name.lower() or "recv" in name.lower()
    tensor = getattr(args, "tensor", None)
    if tensor is not None:  # in-place collective: one tensor is both in and out
        dims, dtypes = _tensor_dims(tensor), _tensor_dtypes(tensor)
        meta["input_sizes"] = meta["output_sizes"] = dims
        meta["input_dtypes"] = meta["output_dtypes"] = dtypes
    else:
        inp, out = getattr(args, "input", None), getattr(args, "output", None)
        if inp is not None:
            meta["input_sizes"] = _tensor_dims(inp)
            meta["input_dtypes"] = _tensor_dtypes(inp)
        if out is not None:
            meta["output_sizes"] = _tensor_dims(out)
            meta["output_dtypes"] = _tensor_dtypes(out)
    for key in ("peer", "root", "async_op", "red_op"):
        value = getattr(args, key, None)
        if value is not None:
            meta[key] = value
    return meta


def _capture_frames(max_frames: int) -> list[dict[str, Any]]:
    """The user's call stack at the collective call site, innermost-first (matching
    c10d's FlightRecorder ``frames`` convention), as ``{name, filename, line}`` dicts.
    The most-recent ``max_frames`` are kept. ``extract_stack()`` is outermost-first and
    ends with this function plus the hook's own frames (``_capture_frames`` ->
    ``_pre_hook`` -> the comm's hook dispatch); the last two entries -- this function
    and ``_pre_hook`` -- are dropped so the top frame is the user's collective call."""
    stack = traceback.extract_stack()[:-2]
    frames = [
        {"name": fs.name, "filename": fs.filename, "line": fs.lineno}
        for fs in reversed(stack)
    ]
    return frames[:max_frames]


class CommMonitorHook:
    """Feeds the CUPTI monitor from a torchcomms ``TorchComm`` via its native pre/post
    hooks -- the integration point for the CollTrace-replacement monitor.

    :meth:`register_with_comm` registers a pre-hook that, on the calling thread before
    each collective is enqueued, pushes a unique external-correlation id (so CUPTI
    keys the kernel and the NCCL-plugin metadata by this collective) and records the
    host-side metadata the profiler descriptor can't supply -- input/output sizes +
    dtypes, peer/root, is_p2p, and the comm's identity -- read straight from the typed
    hook args; the post-hook pops the id. Mirrors the clog/fr hooks
    (``register_with_comm``), and adds nothing when no monitor is running.

    With ``watchdog=True`` each collective issued during CUDA-graph capture is also
    anchored (start event + kernel ``graph_node_id`` node-walk) so the observer's graph
    in-flight accounting (and the ``HangDetectorPlugin``) can see replayed-graph
    collectives; after capturing the graph ``g`` call
    :meth:`finalize_watchdog(g)` and pass :attr:`watchdog_event_resolver` to the
    observer (``CommsObserver.set_event_resolver``), which resolves the start events and
    feeds the watchdog. ``metadata_fn(op_name, args) -> dict | None`` is an escape hatch
    for any extra fields (e.g. ``process_group_ranks``).

    With ``capture_frames=True`` the pre-hook also records the Python call stack at the
    collective call site (innermost-first, ``max_frames`` most-recent) under the
    ``"frames"`` metadata key, so the FlightRecorder dump can source-localize a culprit
    collective. Off by default: ``traceback.extract_stack()`` per collective is not
    free, so enable it only when the source localization is worth the per-call cost."""

    def __init__(
        self,
        monitor: Any = None,
        *,
        watchdog: bool = False,
        metadata_fn: Any = None,
        capture_frames: bool = False,
        max_frames: int = 32,
    ) -> None:
        if monitor is None:
            from torch.profiler._cupti.monitor import get_monitor

            monitor = get_monitor()
        self._monitor = monitor
        self._metadata_fn = metadata_fn
        self._capture_frames_enabled = capture_frames
        self._max_frames = max_frames
        self._anchor = (
            _GraphCommAnchor(monitor) if (watchdog and monitor is not None) else None
        )
        # id()s of comm objects already hooked, to dedup register_with_comm (a comm can be
        # reached more than once, e.g. an explicit register plus the split post-hook).
        self._registered_comms: set[int] = set()
        # op_id -> live anchor scope, for collectives captured into a CUDA graph
        # (the scope spans pre_hook..post_hook).
        self._active: dict[int, Any] = {}
        # op_id -> the external-correlation id pushed for that collective, so the
        # post-hook can wire the work's wait hook to this collective's id.
        self._op_eid: dict[int, int] = {}
        # External ids whose work.wait() the CPU has waited on: recorded by the per-work
        # wait hook (on the waiting thread) and drained by the observer's poll thread via
        # :meth:`drain_waits`. Its own lock so the wait callback never contends with the
        # comm-hook path. The wait channel lives here (the producer), not in the generic
        # monitor; an observer pulls it with ``observer.set_wait_source(hook.drain_waits)``.
        self._waits: list[int] = []
        self._waits_lock = threading.Lock()
        # Reusable eager start event: recorded under each eager collective's external
        # id so CUPTI emits a CUDA_EVENT the observer maps (by correlation) to fire
        # on_start. Created lazily on the first eager call (None until then / no CUDA).
        self._start_event: Any = None

    def register_with_comm(self, comm: Any) -> None:
        """Register the pre/post hooks on ``comm`` (and call again for each split
        group). A no-op when no monitor is running. The comm's identity is bound into the
        per-comm pre-hook closure (one hook serves many comms, so it can't be stored on
        the hook), so each collective is tagged with the comm it actually ran on."""
        if self._monitor is None:
            return
        # Dedup by object identity, NOT comm name: distinct comm objects can share a name
        # (e.g. the model's torchcomms DP instance vs the c10d backend's DP comm that
        # functional/FSDP collectives route through), and we must hook BOTH -- a name
        # dedup would drop the second and lose all its collectives.
        if id(comm) in self._registered_comms:
            return
        self._registered_comms.add(id(comm))
        pg = _comm_identity(comm)
        logger.info("CUPTI comm hook registered on %s", pg["process_group"][0])
        # functools.partial (a C callable, no Python stack frame) binds this comm's pg
        # without disturbing _capture_frames' fixed trailing-frame drop -- a lambda would
        # add a "<lambda>" frame and mis-report the call site.
        comm.register_pre_hook(functools.partial(self._pre_hook, pg))
        comm.register_post_hook(self._post_hook)

    def _pre_hook(
        self, pg: dict[str, Any], op_name: Any, op_id: int, args: Any
    ) -> None:
        monitor = self._monitor
        if monitor is None:
            return
        import torch

        meta = _hook_metadata(op_name, args, pg)
        if self._metadata_fn is not None:
            extra = self._metadata_fn(op_name, args)
            if extra:
                meta.update(extra)
        if self._capture_frames_enabled:
            meta["frames"] = _capture_frames(self._max_frames)

        if self._anchor is not None and torch.cuda.is_current_stream_capturing():
            # Graph capture: anchor the collective (start event + kernel node-walk) and
            # stash the host metadata for the graph metadata_resolver -- the kernel is keyed
            # by graph_node_id on replay. Deliberately do NOT push an external id here: the
            # anchor doesn't use one, and pushing would make the NCCL plugin record a
            # capture-time descriptor (no comm, never completes) that surfaces as a phantom
            # comm=None signature.
            scope = self._anchor.collective(
                torch.cuda.current_stream(), {"name": str(op_name), **meta}
            )
            scope.__enter__()
            self._active[op_id] = scope
            return
        # Eager: push BEFORE the enqueue so CUPTI tags this collective's kernel, and write
        # the metadata now so the NCCL plugin's descriptor (fired during the enqueue) merges
        # on top of it.
        eid = monitor.push_external_correlation_id()
        if eid is not None:
            self._op_eid[op_id] = eid
        if meta:
            monitor.add_collective_metadata(**meta)
        self._record_eager_start()

    def _record_eager_start(self) -> None:
        """Record a reusable CUDA start event on the current stream under this eager
        collective's pushed external id, so CUPTI emits a CUDA_EVENT the observer maps
        (by correlation) to fire on_start. Cheap and re-recorded each call; no-op
        without CUDA."""
        import torch

        if not torch.cuda.is_available():
            return
        if self._start_event is None:
            self._start_event = torch.cuda.Event()
        self._start_event.record()

    def _post_hook(self, op_id: int, args: Any) -> None:
        monitor = self._monitor
        if monitor is None:
            return
        scope = self._active.pop(op_id, None)
        if scope is not None:
            # Capture path: no external id was pushed, so don't pop (and nothing to wait on).
            scope.__exit__(None, None, None)
            return
        monitor.pop_external_correlation_id()
        # A split operation: auto-attach to the new child comm (mirrors the native
        # ClogHook's split post-hook) so collectives on comms created outside the wrapper
        # -- DeviceMesh / FSDP / new_group sub-groups -- are tagged with their comm too.
        new_comm = getattr(args, "new_comm", None)
        if new_comm is not None:
            self.register_with_comm(new_comm)
            return
        # Wire this collective's work.wait() to our wait channel so the CPU-wait
        # lifecycle (on_wait) fires per wait. The work is exposed on the collective
        # post-hook args; None when already released (eager fast path).
        eid = self._op_eid.pop(op_id, None)
        if eid is None:
            return
        work = getattr(args, "work", None)
        if work is None:
            return
        try:
            work.register_wait_post_hook(lambda eid=eid: self._note_wait(eid))
        except Exception:
            pass

    def _note_wait(self, external_id: int) -> None:
        """Record (on the waiting host thread) that the CPU waited on the collective
        tagged with ``external_id``; the observer drains these via :meth:`drain_waits`."""
        with self._waits_lock:
            self._waits.append(external_id)

    def drain_waits(self) -> list[int]:
        """Move out the external ids waited on since the last call. Pass this to an
        observer (``observer.set_wait_source(hook.drain_waits)``) so it fires
        ``on_wait`` for each at poll time."""
        with self._waits_lock:
            waits = self._waits
            self._waits = []
            return waits

    def finalize_watchdog(self, cuda_graph: Any) -> None:
        """Resolve anchored collectives for ``cuda_graph`` (no-op without
        ``watchdog=True``). Call once after capture, before relying on hang detection."""
        if self._anchor is not None:
            self._anchor.finalize(cuda_graph)

    def close_watchdog(self) -> None:
        """Destroy the anchored events + side stream (no-op without ``watchdog``)."""
        if self._anchor is not None:
            self._anchor.close()

    @property
    def watchdog_event_resolver(self) -> Any:
        """The start-event ``event_resolver`` to pass to the observer
        (``CommsObserver.set_event_resolver``); ``None`` without ``watchdog=True``."""
        return self._anchor.event_resolver if self._anchor is not None else None

    @property
    def watchdog_metadata_resolver(self) -> Any:
        """The ``graph_node_id -> blob`` resolver to pass to the observer
        (``CommsObserver.set_metadata_resolver``) so graph collectives' on_end (E) carries
        metadata; ``None`` without ``watchdog=True``."""
        return self._anchor.metadata_resolver if self._anchor is not None else None
