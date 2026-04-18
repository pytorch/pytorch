from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

from ._utils import log
from .config import _IDLE_TIMEOUT_S

if TYPE_CHECKING:
    import torch

    from ._launcher import Launcher
    from ._state import IncrementalAutotuneState

_global_event_queue: queue.Queue[
    tuple[IncrementalAutotuneState, Launcher, torch.cuda.Event, torch.cuda.Event]
] = queue.Queue()
_global_resolver_lock: threading.Lock = threading.Lock()
_global_resolver_thread: threading.Thread | None = None


def submit_event(
    state: IncrementalAutotuneState,
    launcher: Launcher,
    start_event: torch.cuda.Event,
    end_event: torch.cuda.Event,
) -> None:
    """Enqueue a CUDA event pair for async timing resolution.

    Starts the global resolver daemon if it is not already running.
    """
    # Put-then-ensure ordering closes a termination race: if we ensured first
    # and a daemon was idling out concurrently, it could exit between our
    # ensure and our put, leaving the item orphaned. Putting first guarantees
    # the daemon's empty()-check (also under _global_resolver_lock) sees it.
    _global_event_queue.put((state, launcher, start_event, end_event))
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
    background_error: Exception | None = None

    while True:
        try:
            item = _global_event_queue.get(timeout=_IDLE_TIMEOUT_S)
        except queue.Empty:
            with _global_resolver_lock:
                if _global_event_queue.empty():
                    _global_resolver_thread = None
                    log.info("Incremental autotune event resolver stopped (idle timeout)")
                    return
                continue

        state, launcher, start_event, end_event = item
        if background_error is not None:
            state.set_background_error(background_error)
            continue
        try:
            end_event.synchronize()
            elapsed_ms: float = start_event.elapsed_time(end_event)
            launcher.resolve_timing(elapsed_ms)
            state.decrement_pending()
        except RuntimeError as exc:
            # CUDA event ops surface as RuntimeError. Stamp the originating
            # state so the next dispatch raises; future items inherit
            # background_error and skip event resolution.
            log.debug(
                "Incremental autotune: exception resolving timing"
                " for state id=%d, launcher id=%d",
                id(state),
                id(launcher),
                exc_info=True,
            )
            background_error = RuntimeError("Incremental autotune event resolver failed")
            background_error.__cause__ = exc
            state.set_background_error(background_error)
