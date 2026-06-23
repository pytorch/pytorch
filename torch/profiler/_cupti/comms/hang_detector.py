# mypy: allow-untyped-defs
"""``HangDetectorPlugin`` -- flag collectives stuck past the observer's quiescence timeout.

A :class:`~torch.profiler._cupti.comms.CommRecordPlugin` that consumes the
``CommsObserver``'s stall heartbeat (``on_progress``), which the observer's quiescence
thread publishes only after the timeout elapses without any lifecycle callback. Every
collective in that snapshot has been in flight through the whole quiescence window with
no completion -- i.e. stuck -- and the observer has already plain-flushed to rule out an
undelivered completion before publishing, so the verdict is final. The detector fires
``on_timeout`` once per stuck collective.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable  # noqa: TC003
from typing import Any

from torch.profiler._cupti.comms.plugin import CommRecordPlugin


logger = logging.getLogger(__name__)


def _log_stuck_collective(
    coll_id: int, metadata: dict[str, Any], elapsed_s: float
) -> None:
    logger.error(
        "collective %d stuck: in flight %.1fs without completing (%s)",
        coll_id,
        elapsed_s,
        metadata or "no metadata",
    )


class HangDetectorPlugin(CommRecordPlugin):
    """Fire ``on_timeout(coll_id, metadata, elapsed_s)`` once per collective the observer
    reports stuck via ``on_progress`` (default: logs an error); the callback decides the
    action (dump, abort). Requires the observer to run its quiescence thread
    (``CommsObserver(quiescence_timeout_s=...)``) -- that thread is what turns silence
    into the heartbeat this consumes. The graph-replay path additionally needs an
    ``event_resolver`` on the observer so captured collectives appear in the in-flight
    snapshot; eager works regardless."""

    def __init__(
        self,
        on_timeout: Callable[[int, dict[str, Any], float], None] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_timeout = on_timeout or _log_stuck_collective
        self._clock = clock
        # coll_id -> first heartbeat it appeared stuck in (for an elapsed estimate).
        self._first_seen: dict[int, float] = {}
        # coll_ids already reported, so on_timeout fires once per stuck collective even
        # as the heartbeat re-fires each cadence while the stall persists.
        self._tripped: set[int] = set()

    def on_progress(self, in_flight: dict[int, dict[str, Any]]) -> None:
        now = self._clock()
        present = set(in_flight)
        # Drop bookkeeping for collectives that have since cleared (a later heartbeat no
        # longer lists them), so a reused id can trip again.
        for coll_id in [c for c in self._first_seen if c not in present]:
            del self._first_seen[coll_id]
            self._tripped.discard(coll_id)
        for coll_id, metadata in in_flight.items():
            first = self._first_seen.setdefault(coll_id, now)
            if coll_id not in self._tripped:
                self._tripped.add(coll_id)
                self._on_timeout(coll_id, metadata, now - first)

    def on_end(self, record: Any) -> None:
        # A completion clears any stuck bookkeeping for this collective.
        self._first_seen.pop(record.coll_id, None)
        self._tripped.discard(record.coll_id)
