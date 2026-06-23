# mypy: allow-untyped-defs
"""The :class:`CommRecordPlugin` extension point for a ``CommsObserver``.

A plugin observes a collective's lifecycle: ``on_schedule`` (issued / host-enqueued),
``on_start`` (kernel began on the device), ``on_progress`` (the stall heartbeat -- the
observer's quiescence thread publishes the in-flight set after the timeout elapses
without progress), ``on_end`` (completed, with timing + metadata), and ``on_wait`` (the
CPU waited on the collective). The observer (the producer) dispatches duck-typed to
registered plugins, so it does not import this base; :mod:`flight_recorder` (the OSS
FlightRecorder serializer) and :mod:`hang_detector` subclass it. Service-specific
serializers (torchcomms ``clog``, ncclx ``comm_dump``) live with their consumers.
"""

from __future__ import annotations

from typing import Any

from torch.profiler._cupti.observers.comms import CommRecord  # noqa: TC001


class CommRecordPlugin:
    """Lifecycle extension point for ``CommsObserver``. Subclass and override the hooks
    you need (all default to no-ops). Each collective fires ``on_schedule`` once before
    its ``on_end``; ``on_progress`` is the stall heartbeat (the observer's quiescence
    thread publishes the in-flight set after the timeout) -- the seam a hang/timeout scan
    consumes. Ready-made consumers in this package:
    :class:`~torch.profiler._cupti.comms.FlightRecorderPlugin` (serializes to the OSS
    FlightRecorder schema for ``fr_trace``) and
    :class:`~torch.profiler._cupti.comms.HangDetectorPlugin`. Service-specific
    serializers (torchcomms ``clog``, ncclx ``comm_dump``) live with their consumers as
    subclasses of this base."""

    def on_schedule(self, coll_id: int, metadata: dict[str, Any]) -> None:
        """A collective was issued (host-enqueued); its kernel has not been timed yet.
        Fires exactly once per collective, always before its :meth:`on_end`. Graph
        collectives (no eager external id) are only ever observed at completion, so they
        get :meth:`on_end` only."""

    def on_start(self, coll_id: int, metadata: dict[str, Any]) -> None:
        """The collective's kernel began on the device -- the per-collective device-start
        signal. Fires only when per-collective start events are captured (the observer's
        ``start_events`` / ``watchdog`` / ``event_resolver`` paths). Start and end
        records arrive in undefined order, so this may fire after :meth:`on_end`; and for
        graph replays it fires once per replay (no once-suppression)."""

    def on_progress(self, in_flight: dict[int, dict[str, Any]]) -> None:
        """Periodic heartbeat: fired once per poll with the issued-but-not-yet-completed
        collectives (external_id -> metadata) -- the current/pending view a hang/timeout
        check needs."""

    def on_end(self, record: CommRecord) -> None:
        """A collective completed (timing + metadata joined into a ``CommRecord``)."""

    def on_wait(self, coll_id: int, metadata: dict[str, Any]) -> None:
        """The CPU waited on this collective (``work.wait()``). A HOST event from the
        torchcomms work wait-hook, not a CUPTI signal: :class:`CommMonitorHook`'s
        per-work wait hook records the wait on the waiting thread (the hook's own wait
        channel) and the observer drains it at :meth:`poll` on its own thread (wired via
        ``observer.set_wait_source(hook.drain_waits)``), firing this per wait occurrence.
        Only fires when torchcomms exposes the ``TorchWork`` wait hooks (the work on the
        collective post-hook args). May fire after :meth:`on_end`; not all collectives
        are explicitly waited on, so it may not fire at all."""
