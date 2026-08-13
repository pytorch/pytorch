# This module is experimental and subject to change.
"""
Pure-Python watchdog for detecting hung operations in symmetric memory
kernels, Python-based distributed backends and related
distributed primitives.

The watchdog runs an asyncio event loop on a background daemon thread,
polls CUDA events to detect stuck GPU work, and provides CPU-side timers
for host-blocking operations. It fires user-configurable
callbacks when a timeout is exceeded.

A second daemon thread ("health watchdog") periodically pings the
event loop. If the ping doesn't complete, the event loop is stuck and
the health watchdog takes a configurable action.

Usage:

    from torch.distributed._watchdog import (
        init,
        stream_timeout,
        cpu_timeout,
        op_timeout,
    )

    # Optional: configure programmatically (overrides env vars)
    init(poll_interval=0.5, timeout_action="abort")

    # Detect a hung GPU kernel: record event AFTER the kernel launch
    some_kernel_launch()
    handle = stream_timeout(60.0, lambda: print("kernel hung!"))
    # When operation completes:
    handle.cancel()

    # Detect a hung CPU operation
    handle = cpu_timeout(60.0, lambda: print("rendezvous hung!"))
    blocking_rendezvous_call()
    handle.cancel()

    # Context manager form
    with op_timeout(60.0, lambda: print("timed out!")):
        some_operation()
"""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import queue
import sys
import threading
import time
from collections.abc import Callable, Generator  # noqa: TC003
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta

import torch


logger = logging.getLogger(__name__)

__all__ = [
    "_get_watchdog",
    "init",
    "shutdown",
    "stream_timeout",
    "cpu_timeout",
    "op_timeout",
]


def _default_timeout_callback() -> None:
    logger.error("Watchdog timeout -- dumping all thread stack traces")
    faulthandler.dump_traceback()


@dataclass
class _StreamMonitor:
    event: object  # torch.cuda.Event
    deadline: float
    callback: Callable[[], None]
    cancelled: bool = False
    monitor_id: int = field(default=0, init=False)


class _CancelHandle:
    """
    Handle returned by stream_timeout/cpu_timeout for cancelling a pending timeout.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._timer_handle: asyncio.TimerHandle | None = None
        self._stream_monitor: _StreamMonitor | None = None

    def _set_timer_handle(self, timer_handle: asyncio.TimerHandle) -> None:
        with self._lock:
            if self._cancelled:
                timer_handle.cancel()
            else:
                self._timer_handle = timer_handle

    def _set_stream_monitor(self, monitor: _StreamMonitor) -> None:
        with self._lock:
            if self._cancelled:
                monitor.cancelled = True
            else:
                self._stream_monitor = monitor

    def cancel(self) -> None:
        """
        Cancel the timeout. Safe to call from any thread.
        """
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            if self._timer_handle is not None:
                self._timer_handle.cancel()
                self._timer_handle = None
            if self._stream_monitor is not None:
                self._stream_monitor.cancelled = True
                self._stream_monitor = None

    @property
    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled


class _Watchdog:
    """Process-wide watchdog for Python distributed backends.

    Manages an asyncio event loop on a daemon thread for CPU timeouts and
    periodic CUDA event polling, plus a health watchdog on a second daemon
    thread.
    """

    def __init__(
        self,
        *,
        poll_interval: float | None = None,
        health_interval: float | None = None,
        stuck_action: str | None = None,
    ) -> None:
        self._poll_interval = (
            poll_interval
            if poll_interval is not None
            else float(os.environ.get("TORCH_WATCHDOG_POLL_INTERVAL_SECS", "1.0"))
        )
        self._health_interval = (
            health_interval
            if health_interval is not None
            else float(os.environ.get("TORCH_WATCHDOG_HEALTH_INTERVAL_SECS", "30.0"))
        )
        self._stuck_action = (
            stuck_action or os.environ.get("TORCH_WATCHDOG_STUCK_ACTION", "log")
        ).lower()

        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._stop_event = threading.Event()

        self._stream_monitors: dict[int, _StreamMonitor] = {}
        self._next_monitor_id: int = 0
        self._poll_handle: asyncio.TimerHandle | None = None

        self._del_queue: queue.SimpleQueue[object] = queue.SimpleQueue()

        self._loop_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="pt_pywd_loop"
        )
        self._loop_thread.start()

        self._health_thread = threading.Thread(
            target=self._health_watchdog_loop, daemon=True, name="pt_pywd_health"
        )
        self._health_thread.start()

        logger.info(
            "Watchdog started (poll=%.1fs, health=%.1fs, stuck_action=%s)",
            self._poll_interval,
            self._health_interval,
            self._stuck_action,
        )

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def stream_timeout(
        self, timeout: float | timedelta, callback: Callable[[], None] | None = None
    ) -> _CancelHandle:
        if callback is None:
            callback = _default_timeout_callback
        if isinstance(timeout, timedelta):
            timeout = timeout.total_seconds()
        self._drain_del_queue()
        handle = _CancelHandle()
        event = torch.cuda.Event(enable_timing=False)
        event.record()
        monitor = _StreamMonitor(
            event=event, deadline=time.monotonic() + timeout, callback=callback
        )
        self._loop.call_soon_threadsafe(self._add_monitor, monitor, handle)
        return handle

    def _add_monitor(self, monitor: _StreamMonitor, handle: _CancelHandle) -> None:
        mid = self._next_monitor_id
        self._next_monitor_id += 1
        monitor.monitor_id = mid
        handle._set_stream_monitor(monitor)
        self._stream_monitors[mid] = monitor
        if self._poll_handle is None:
            self._schedule_poll()

    def _schedule_poll(self) -> None:
        self._poll_handle = self._loop.call_later(
            self._poll_interval, self._poll_monitors
        )

    def _poll_monitors(self) -> None:
        self._poll_handle = None
        now = time.monotonic()
        to_remove: list[int] = []

        for mid, monitor in self._stream_monitors.items():
            if monitor.cancelled:
                to_remove.append(mid)
                continue

            completed = False
            try:
                completed = monitor.event.query()  # type: ignore[union-attr]
            except RuntimeError:
                # event.query() raises during CUDA graph capture
                pass

            if completed:
                to_remove.append(mid)
                continue

            if now >= monitor.deadline:
                self._fire_callback(monitor.callback, "stream", monitor.monitor_id)
                to_remove.append(mid)

        for mid in to_remove:
            m = self._stream_monitors.pop(mid)
            self._del_queue.put(m.event)

        if self._stream_monitors:
            self._schedule_poll()

    def cpu_timeout(
        self, timeout: float | timedelta, callback: Callable[[], None] | None = None
    ) -> _CancelHandle:
        if callback is None:
            callback = _default_timeout_callback
        if isinstance(timeout, timedelta):
            timeout = timeout.total_seconds()
        handle = _CancelHandle()
        self._loop.call_soon_threadsafe(
            self._register_cpu_timeout, callback, timeout, handle
        )
        return handle

    def _register_cpu_timeout(
        self, callback: Callable[[], None], timeout: float, handle: _CancelHandle
    ) -> None:
        def fire() -> None:
            self._fire_callback(callback, "cpu", -1)

        timer_handle = self._loop.call_later(timeout, fire)
        handle._set_timer_handle(timer_handle)

    def _fire_callback(self, callback: Callable[[], None], kind: str, mid: int) -> None:
        logger.warning("Watchdog %s timeout fired (id=%d)", kind, mid)
        try:
            callback()
        except Exception:
            logger.exception("Exception in %s timeout callback (id=%d)", kind, mid)

        if callback is _default_timeout_callback:
            logger.error("Default timeout callback fired, calling os.abort()")
            os.abort()

    def _health_watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            healthy = threading.Event()
            try:
                self._loop.call_soon_threadsafe(healthy.set)
            except RuntimeError:
                return

            self._stop_event.wait(timeout=self._health_interval)
            if self._stop_event.is_set():
                return

            if not healthy.is_set():
                self._handle_stuck_loop()

    def _handle_stuck_loop(self) -> None:
        msg = (
            "Watchdog event loop appears stuck "
            f"(no response within {self._health_interval:.1f}s)"
        )
        if self._stuck_action == "abort":
            logger.error("%s -- aborting process", msg)
            os.abort()
        elif self._stuck_action == "exit":
            logger.error("%s -- exiting process", msg)
            sys.exit(1)
        else:
            logger.warning(msg)

    def _drain_del_queue(self) -> int:
        """Destroy retired CUDA events from the caller's thread.

        CUDA events must be destroyed on the thread that owns the CUDA context.
        Destroying them on the asyncio loop thread can block on the CUDA driver
        and stall the event loop, so retired events are queued and deleted here.
        """
        count = 0
        while True:
            try:
                obj = self._del_queue.get_nowait()
                del obj
                count += 1
            except queue.Empty:
                break
        return count

    def shutdown(self) -> None:
        logger.info("Watchdog shutting down")
        self._stop_event.set()
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass
        self._loop_thread.join(timeout=5.0)
        self._health_thread.join(timeout=5.0)
        self._loop.close()
        self._drain_del_queue()
        logger.info("Watchdog shut down")


_watchdog: _Watchdog | None = None
_watchdog_lock = threading.Lock()


def _get_watchdog() -> _Watchdog:
    """
    Get or create the process-wide watchdog singleton.
    """
    global _watchdog
    if _watchdog is not None:
        return _watchdog
    with _watchdog_lock:
        if _watchdog is None:
            _watchdog = _Watchdog()
        return _watchdog


def shutdown() -> None:
    """
    Shut down the watchdog singleton. After this, _get_watchdog() creates a fresh instance.
    """
    global _watchdog
    wd: _Watchdog | None = None
    with _watchdog_lock:
        if _watchdog is not None:
            wd = _watchdog
            _watchdog = None
    if wd is not None:
        wd.shutdown()


def init(
    *,
    poll_interval: float | None = None,
    health_interval: float | None = None,
    stuck_action: str | None = None,
) -> None:
    """Configure and start the watchdog.

    Programmatic settings override environment variables.

    If a watchdog is already running it is shut down first.

    Args:
        poll_interval: Seconds between CUDA event polls (env: TORCH_WATCHDOG_POLL_INTERVAL_SECS, default 1.0).
        health_interval: Seconds between health pings (env: TORCH_WATCHDOG_HEALTH_INTERVAL_SECS, default 30.0).
        stuck_action: Action when event loop is stuck: "log", "abort", or "exit" (env: TORCH_WATCHDOG_STUCK_ACTION, default "log").

    Example::
        from torch.distributed._watchdog import init

        init(poll_interval=0.5)
    """
    global _watchdog
    old: _Watchdog | None = None
    with _watchdog_lock:
        old = _watchdog
        _watchdog = _Watchdog(
            poll_interval=poll_interval,
            health_interval=health_interval,
            stuck_action=stuck_action,
        )
    if old is not None:
        old.shutdown()


def stream_timeout(
    timeout: float | timedelta, callback: Callable[[], None] | None = None
) -> _CancelHandle:
    """Record a CUDA event and fire callback if the stream hasn't completed by deadline.

    Args:
        timeout: Seconds (float) or timedelta until the callback fires.
        callback: Called on timeout. Defaults to logging stack traces and aborting.

    Example::

        some_kernel_launch()
        handle = stream_timeout(60.0)
        # ... when the operation completes:
        handle.cancel()
    """
    return _get_watchdog().stream_timeout(timeout, callback)


def cpu_timeout(
    timeout: float | timedelta, callback: Callable[[], None] | None = None
) -> _CancelHandle:
    """Schedule callback to fire after timeout unless cancelled.

    Args:
        timeout: Seconds (float) or timedelta until the callback fires.
        callback: Called on timeout. Defaults to logging stack traces and aborting.

    Example::

        handle = cpu_timeout(30.0)
        blocking_rendezvous_call()
        handle.cancel()
    """
    return _get_watchdog().cpu_timeout(timeout, callback)


@contextmanager
def op_timeout(
    timeout: float | timedelta, callback: Callable[[], None] | None = None
) -> Generator[None, None, None]:
    """CPU timeout guards the block; stream timeout monitors GPU work after exit.

    Example::

        with op_timeout(60.0):
            foo()  # cpu timeout fires if this hangs
        torch.cuda.synchronize()  # stream timeout fires if this hangs
    """
    cpu_handle = cpu_timeout(timeout, callback)
    try:
        yield
    finally:
        cpu_handle.cancel()
        stream_timeout(timeout, callback)
