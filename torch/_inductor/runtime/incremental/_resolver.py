from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from ._utils import log
from .config import _IDLE_TIMEOUT_S


if TYPE_CHECKING:
    import torch


TimingCallback = Callable[[float], None]

_QueueItem = tuple[TimingCallback, "torch.cuda.Event", "torch.cuda.Event"]

_global_event_queue: queue.Queue[_QueueItem] = queue.Queue()
_global_resolver_lock: threading.Lock = threading.Lock()
_global_resolver_thread: threading.Thread | None = None


def submit_event(
    callback: TimingCallback,
    start_event: torch.cuda.Event,
    end_event: torch.cuda.Event,
) -> None:
    """Enqueue a CUDA event pair for async timing resolution.

    The daemon synchronizes on ``end_event``, computes elapsed time, and
    invokes ``callback(elapsed_ms)``. Starts the global resolver daemon
    if it is not already running.
    """
    # Put-then-ensure ordering closes a termination race: if we ensured first
    # and a daemon was idling out concurrently, it could exit between our
    # ensure and our put, leaving the item orphaned. Putting first guarantees
    # the daemon's empty()-check (also under _global_resolver_lock) sees it.
    _global_event_queue.put((callback, start_event, end_event))
    _ensure_daemon()


def _ensure_daemon() -> None:
    global _global_resolver_thread
    with _global_resolver_lock:
        if _global_resolver_thread is not None:
            return
        t = threading.Thread(
            target=_global_event_resolver_loop,
            daemon=True,
            name="autotune-event-resolver",
        )
        t.start()
        _global_resolver_thread = t
        log.info(
            "Incremental autotune event resolver started (thread id=%d)",
            t.ident,
        )


def _global_event_resolver_loop() -> None:
    global _global_resolver_thread

    while True:
        try:
            item = _global_event_queue.get(timeout=_IDLE_TIMEOUT_S)
        except queue.Empty:
            # Consumer side of the put-then-ensure race close: a producer may
            # have called put() between our get() raising Empty and our
            # acquiring the lock. Re-check empty() under the lock so we don't
            # exit while an item is sitting in the queue.
            with _global_resolver_lock:
                if _global_event_queue.empty():
                    _global_resolver_thread = None
                    log.info(
                        "Incremental autotune event resolver stopped (idle timeout)"
                    )
                    return
                continue

        callback, start_event, end_event = item
        try:
            end_event.synchronize()
            elapsed_ms: float = start_event.elapsed_time(end_event)
            callback(elapsed_ms)
        except Exception:
            # Silently drop. We assume any underlying CUDA fault will
            # resurface on the main thread's next CUDA op (e.g.
            # ``torch.cuda.synchronize``) and would rather not surface it
            # twice from a background thread with no clear ownership.
            log.debug(
                "Incremental autotune: timing resolution failed",
                exc_info=True,
            )
