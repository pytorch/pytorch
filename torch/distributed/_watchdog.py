"""Process-wide watchdog for CPU and accelerator stream timeouts.

This module exposes timeout guards backed by a libuv timer loop that runs on a
background thread. The guards operate on a single process-wide watchdog
instance.

The watchdog is only available when PyTorch is built with the libuv backend;
importing this module otherwise raises ImportError from the underlying bindings.

All callbacks fire on the watchdog's background thread and therefore must not
block, otherwise other timeouts may not be serviced.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import timedelta

from torch._C._distributed_c10d_watchdog import _Watchdog, _WatchdogHandle


__all__ = ["cpu_timeout", "stream_timeout", "op_timeout"]


@contextmanager
def cpu_timeout(
    callback: Callable[[], None], timeout: timedelta
) -> Generator[None, None, None]:
    """Run ``callback`` if the guarded block takes longer than ``timeout``.

    The timer is cancelled on exit, so ``callback`` only fires when the guarded
    (typically blocking) section overruns. ``callback`` runs on the watchdog
    background thread and must not block.
    """
    handle = _Watchdog._singleton().cpu_timeout(timeout, callback)
    try:
        yield
    finally:
        handle.cancel()


def stream_timeout(callback: Callable[[], None], timeout: timedelta) -> _WatchdogHandle:
    """Run ``callback`` if the accelerator work enqueued on the current stream so
    far does not complete within ``timeout``.

    Call this after enqueuing the work to be bounded; it records an event on the
    current stream and monitors it. Returns a handle whose ``cancel()`` stops
    monitoring. ``callback`` runs on the watchdog background thread and must not
    block.
    """
    return _Watchdog._singleton().stream_timeout(timeout, callback)


@contextmanager
def op_timeout(
    callback: Callable[[], None], timeout: timedelta
) -> Generator[None, None, None]:
    """Run ``callback`` if an accelerator op does not launch and complete within
    ``timeout``.

    On entry a start event is recorded on the current stream and a launch
    deadline is armed: if the op never begins executing within ``timeout``,
    ``callback`` fires. On exit (the op has been enqueued) an end event is
    recorded on the current stream. Once the start event completes a completion
    deadline begins, and if the end event has not completed within ``timeout``
    ``callback`` fires. ``callback`` fires at most once and the monitor always
    removes itself. ``callback`` runs on the watchdog background thread and must
    not block.
    """
    handle = _Watchdog._singleton().op_timeout(timeout, callback)
    try:
        yield
    finally:
        handle.completed()
