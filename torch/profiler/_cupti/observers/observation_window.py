# mypy: allow-untyped-defs
"""Shared window-finalization machinery for CUPTI monitor observers.

An observer that publishes per *window* (a span ended by a boundary -- a training step or
profiling window) stamps each boundary in CUPTI's native record clock and finalizes the
window only once its records are *naturally* delivered -- no device sync on the measured
timeline (CUPTI delivers buffers in fill order, so a delivered record starting at/after a
boundary means every earlier one is in hand). This mixin owns the boundary queue, poll
thread, cover-detection, and teardown; the subclass supplies ``_collect_delivered(sync)``,
``_window_watermark_ns()`` (max delivered record start, native clock), and
``_finalize_window(window_id, boundary_ns)``. Boundaries use ``now_native_ns()`` from
:class:`CuptiMonitorObserver`, the same timebase as record START/END.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable  # noqa: TC003
from typing import TYPE_CHECKING


logger = logging.getLogger(__name__)


class _PollThread(threading.Thread):
    """Daemon thread that calls ``fn`` every ``interval_s`` until stopped."""

    def __init__(self, fn: Callable[[], None], interval_s: float, name: str) -> None:
        super().__init__(name=name, daemon=True)
        self._fn = fn
        self._interval_s = interval_s
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self._fn()
            except Exception:
                logger.exception("CUPTI window poll cycle failed")

    def stop(self) -> None:
        self._stop_event.set()


class WindowFinalizerMixin:
    """Boundary queue + background poller + cover-and-finalize loop. See the module
    docstring. Subclasses call :meth:`_init_observation_window` in ``__init__`` and implement
    ``_collect_delivered`` / ``_window_watermark_ns`` / ``_finalize_window``."""

    def _init_observation_window(
        self,
        *,
        poll_interval_ms: int = 50,
        thread_name: str = "cupti-window-poller",
        auto_start_poller: bool = True,
    ) -> None:
        # auto_start_poller=False: never spin up the poll thread; the consumer finalizes
        # synchronously via _stop_observation_window (e.g. a synchronous export).
        self._auto_start_poller = auto_start_poller
        self._win_lock = threading.Lock()
        # (window_id, boundary_ns) per ended window, oldest first; the poller pops one
        # once its boundary is covered by delivered records.
        self._boundaries: deque[tuple[int, int]] = deque()
        self._next_window_id = 0
        self._poll_interval_s = max(1, poll_interval_ms) / 1000.0
        self._poll_thread: _PollThread | None = None
        self._poll_thread_name = thread_name

    def mark_boundary(self) -> int:
        """Stamp the native record clock as a window boundary and queue it for finalization
        once delivered records cover it. Returns the window id. Lazily starts the poller on
        the first call (unless auto_start_poller is False). Does NOT flush -- the caller
        decides whether to nudge a flush so the poller sees the records."""
        boundary = self._boundary_clock_ns()
        with self._win_lock:
            window_id = self._next_window_id
            self._next_window_id += 1
            self._boundaries.append((window_id, boundary))
            if self._poll_thread is None and self._auto_start_poller:
                self._poll_thread = _PollThread(
                    self._poll_once, self._poll_interval_s, self._poll_thread_name
                )
                self._poll_thread.start()
        return window_id

    def _poll_once(self, drain_all: bool = False, *, sync: bool = False) -> None:
        """One poll cycle: collect delivered records, then finalize every queued boundary
        they now cover. ``drain_all`` (teardown) finalizes every remaining boundary
        regardless of watermark. ``sync`` sync-flushes CUPTI in the collect (only safe on
        the monitor-flusher thread); leave False to rely on naturally-delivered records."""
        self._collect_delivered(sync=sync)
        watermark = self._window_watermark_ns()
        ready: list[tuple[int, int]] = []
        with self._win_lock:
            while self._boundaries and (
                drain_all or self._boundaries[0][1] <= watermark
            ):
                ready.append(self._boundaries.popleft())
        for window_id, boundary in ready:
            self._finalize_window(window_id, boundary)

    def _stop_observation_window(self, *, sync: bool = True) -> None:
        """Stop the poller and finalize whatever remains. Idempotent. ``sync`` (default)
        sync-flushes CUPTI in the final drain -- correct on the monitor-flusher thread (e.g.
        teardown on the training thread). ``sync=False`` finalizes only naturally-delivered
        records, for off-thread use where a flush would race the monitor."""
        thread = self._poll_thread
        if thread is not None:
            thread.stop()
            thread.join(timeout=5.0)
            self._poll_thread = None
        self._poll_once(drain_all=True, sync=sync)

    def _boundary_clock_ns(self) -> int:
        """Clock a boundary is stamped in -- must match :meth:`_window_watermark_ns`.
        Default: CUPTI's native record clock (matches raw START/END). Override to a
        converted clock if the subclass stores records converted (convert the boundary the
        same way; convert_time is monotonic, so the comparison stays order-equivalent)."""
        return self.now_native_ns()

    if TYPE_CHECKING:
        # Provided by the co-class CuptiMonitorObserver in the MRO; declared here only for
        # the type checker (a real def would shadow it at runtime).
        def now_native_ns(self) -> int: ...

    def _collect_delivered(self, *, sync: bool) -> None:
        raise NotImplementedError

    def _window_watermark_ns(self) -> int:
        raise NotImplementedError

    def _finalize_window(self, window_id: int, boundary_ns: int) -> None:
        raise NotImplementedError
