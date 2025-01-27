# mypy: allow-untyped-defs
import contextlib

import torch


__all__ = [
    "start",
    "stop",
    "profile",
    "metal_capture",
    "is_metal_capture_enabled",
    "is_capturing_metal",
]


def start(mode: str = "interval", wait_until_completed: bool = False) -> None:
    r"""Start OS Signpost tracing from MPS backend.

    The generated OS Signposts could be recorded and viewed in
    XCode Instruments Logging tool.

    Args:
        mode(str): OS Signpost tracing mode could be "interval", "event",
            or both "interval,event".
            The interval mode traces the duration of execution of the operations,
            whereas event mode marks the completion of executions.
            See document `Recording Performance Data`_ for more info.
        wait_until_completed(bool): Waits until the MPS Stream complete
            executing each encoded GPU operation. This helps generating single
            dispatches on the trace's timeline.
            Note that enabling this option would affect the performance negatively.

    .. _Recording Performance Data:
       https://developer.apple.com/documentation/os/logging/recording_performance_data
    """
    mode_normalized = mode.lower().replace(" ", "")
    torch._C._mps_profilerStartTrace(mode_normalized, wait_until_completed)


def stop():
    r"""Stops generating OS Signpost tracing from MPS backend."""
    torch._C._mps_profilerStopTrace()


@contextlib.contextmanager
def profile(mode: str = "interval", wait_until_completed: bool = False):
    r"""Context Manager to enabling generating OS Signpost tracing from MPS backend.

    Args:
        mode(str): OS Signpost tracing mode could be "interval", "event",
            or both "interval,event".
            The interval mode traces the duration of execution of the operations,
            whereas event mode marks the completion of executions.
            See document `Recording Performance Data`_ for more info.
        wait_until_completed(bool): Waits until the MPS Stream complete
            executing each encoded GPU operation. This helps generating single
            dispatches on the trace's timeline.
            Note that enabling this option would affect the performance negatively.

    .. _Recording Performance Data:
       https://developer.apple.com/documentation/os/logging/recording_performance_data
    """
    try:
        start(mode, wait_until_completed)
        yield
    finally:
        stop()


def is_metal_capture_enabled() -> bool:
    """Checks if `metal_capture` context manager is usable
    To enable metal capture, set MTL_CAPTURE_ENABLED envvar
    """
    return torch._C._mps_isCaptureEnabled()  # type: ignore[attr-defined]


def is_capturing_metal() -> bool:
    """Cheks if metal capture is in progress"""
    return torch._C._mps_isCapturing()  # type: ignore[attr-defined]


@contextlib.contextmanager
def metal_capture(fname: str):
    """Conext manager that enables capturing of Metal calls into gputrace"""
    try:
        torch._C._mps_startCapture(fname)  # type: ignore[attr-defined]
        yield
        # Drain all the work that were enqueued during the context call
        torch.mps.synchronize()
    finally:
        torch._C._mps_stopCapture()  # type: ignore[attr-defined]
