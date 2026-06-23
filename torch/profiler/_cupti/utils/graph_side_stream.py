# mypy: allow-untyped-defs
"""A CUDA graph-capture "side stream", mirroring meta::comms GraphSideStream.

Records work onto a side stream *during* CUDA-graph capture so it is captured
into the graph but does NOT serialize subsequent ops on the main stream. The
CUPTI comm watchdog uses this to anchor a per-collective start/end external CUDA
event (recorded with ``cudaEventRecordExternal`` so each emits a CUDA_EVENT
activity record per replay) into a captured graph without putting the event on
the collective kernel's critical path.

Each :meth:`fork_from` call:
  1. checks whether ``stream`` is under capture (if not, runs ``fn(stream)``);
  2. forks ``stream -> side`` (``dep.record(stream)`` + ``side.wait_event(dep)``);
  3. snapshots ``stream``'s post-fork dependency set
     (``cudaStreamGetCaptureInfo``);
  4. runs ``fn(side)`` -- the caller's captured work;
  5. rejoins ``side -> stream`` (required: ``cudaStreamEndCapture`` rejects an
     unjoined fork);
  6. restores the snapshotted deps (``cudaStreamUpdateCaptureDependencies``,
     SET) so the rejoin node, though present in the graph DAG, is not a
     predecessor of ``stream``'s next op -- keeping side work off main's path.

Streams use the public ``torch.cuda`` API; the events, capture-info and
dependency APIs (no torch wrapper, or torch's lazy event creation is unsafe mid
capture) go through ``cuda.bindings.runtime`` -- the same path as
``torch.cuda._graph_annotations``. The external events are created EXPLICITLY
(``cudaEventCreateWithFlags``): ``torch.cuda.Event`` creates its ``cudaEvent_t``
lazily on first record, and that implicit ``cudaEventCreate`` -- riding a
``cudaEventRecord`` on the capturing stream -- poisons the graph and deadlocks
replay. An explicit ``cudaEventCreate`` is not a stream op, so it is safe
mid-capture.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from typing import Any

import torch


try:
    from cuda.bindings import runtime as _rt  # pyrefly: ignore[missing-import]

    _HAS_CUDA_BINDINGS = True
except ImportError:
    _rt: Any = None
    _HAS_CUDA_BINDINGS = False


# cudaEventRecordExternal: the event becomes a real event-record node in a
# captured graph (so it emits a CUDA_EVENT activity per replay) rather than an
# internal cross-stream dependency.
_CUDA_EVENT_RECORD_EXTERNAL = 1


def _check(ret: Any) -> Any:
    """Unwrap a ``cuda.bindings`` ``(err, *rest)`` return, raising on error."""
    err, *rest = ret if isinstance(ret, (tuple, list)) else (ret,)
    if err != _rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cuda.bindings runtime call failed: {err}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else tuple(rest)


class GraphSideStream:
    """A side stream for recording external CUDA events off the main stream's
    critical path during graph capture. Construct OUTSIDE an active capture (it
    allocates a stream); one instance per producer is enough. :meth:`close`
    destroys the events it created."""

    def __init__(self, priority: int = 0) -> None:
        self._side = torch.cuda.Stream(priority=priority)
        # Raw cudaEvent_t handles created here, destroyed on close().
        self._events: list[Any] = []

    def _capture_state(self, stream: torch.cuda.Stream):
        """Return ``(graph, deps, edge_data)`` if ``stream`` is actively
        capturing, else ``None``. ``deps``/``edge_data`` are lists valid only
        until the next API call on the stream."""
        if not _HAS_CUDA_BINDINGS:
            return None
        handle = _rt.cudaStream_t(init_value=stream.cuda_stream)
        status, _id, graph, deps, edge, num = _check(
            _rt.cudaStreamGetCaptureInfo(handle)
        )
        if status != _rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
            return None
        return graph, list(deps[:num]), (list(edge[:num]) if edge else None)

    def fork_from(
        self, stream: torch.cuda.Stream, fn: Callable[[torch.cuda.Stream], None]
    ) -> None:
        """Run ``fn(side_stream)`` with fork/save/rejoin/restore scaffolding. If
        ``stream`` is not under capture, ``fn(stream)`` runs directly (no fork)."""
        state = self._capture_state(stream)
        if state is None:
            fn(stream)
            return

        # Fork main -> side. A FRESH event per fork/rejoin (not one reused across
        # calls): in a captured graph every record/wait becomes a node, and
        # aliasing one event across many cycles produces ambiguous event state
        # that DEADLOCKS at replay -- the eager "record+wait consume immediately"
        # invariant doesn't hold once all nodes coexist and replay together.
        fork_dep = torch.cuda.Event()
        fork_dep.record(stream)
        self._side.wait_event(fork_dep)

        # Re-query after the fork (the eventRecord invalidated the prior pointers)
        # and snapshot the deps to restore after the rejoin.
        graph, deps, edge = self._capture_state(stream) or (None, [], None)

        fn(self._side)

        # Rejoin side -> main (required for cudaStreamEndCapture to accept).
        rejoin_dep = torch.cuda.Event()
        rejoin_dep.record(self._side)
        stream.wait_event(rejoin_dep)

        # Restore main's pre-rejoin deps so the rejoin node is not a predecessor
        # of main's next captured op.
        handle = _rt.cudaStream_t(init_value=stream.cuda_stream)
        _check(
            _rt.cudaStreamUpdateCaptureDependencies(
                handle,
                deps,
                edge,
                len(deps),
                _rt.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
            )
        )

    def _create_external_event(self) -> Any:
        """Create the raw ``cudaEvent_t`` explicitly. ``torch.cuda.Event`` creates
        its handle lazily on first record, and that implicit ``cudaEventCreate`` --
        riding a ``cudaEventRecord`` on the capturing stream -- poisons the captured
        graph and DEADLOCKS replay. ``cudaEventCreate`` itself is not a stream op,
        so calling it directly mid-capture is safe."""
        event = _check(_rt.cudaEventCreateWithFlags(_rt.cudaEventDisableTiming))
        self._events.append(event)
        return event

    def record_external_event(self, stream: torch.cuda.Stream) -> Any:
        """Create and record an external CUDA event off ``stream``'s critical path
        (forked onto the side stream) and return its raw ``cudaEvent_t``. Recorded
        with ``cudaEventRecordExternal`` so it emits a CUDA_EVENT activity record on
        every replay; the caller learns its CUPTI ``eventId`` via a one-shot eager
        record (warmup)."""
        if not _HAS_CUDA_BINDINGS:
            raise RuntimeError("GraphSideStream requires the cuda.bindings package")
        event = self._create_external_event()

        def _record(side: torch.cuda.Stream) -> None:
            _check(
                _rt.cudaEventRecordWithFlags(
                    event,
                    _rt.cudaStream_t(init_value=side.cuda_stream),
                    _CUDA_EVENT_RECORD_EXTERNAL,
                )
            )

        self.fork_from(stream, _record)
        return event

    def close(self) -> None:
        """Destroy the events created by :meth:`record_external_event`."""
        for event in self._events:
            _rt.cudaEventDestroy(event)
        self._events.clear()


def eager_record(event: Any, stream: torch.cuda.Stream) -> None:
    """Record a raw ``cudaEvent_t`` on ``stream`` (no external flag), used OUTSIDE
    capture to force the event's CUPTI ``eventId`` to be emitted so a consumer can
    learn the (replay-stable) handle->eventId mapping."""
    _check(
        _rt.cudaEventRecordWithFlags(
            event, _rt.cudaStream_t(init_value=stream.cuda_stream), 0
        )
    )
