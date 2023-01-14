from contextlib import contextmanager

try:
    from torch._C import _nvtx
except ImportError:
    class _NVTXStub:
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError("NVTX functions not installed. Are you sure you have a CUDA build?")

        rangePush = _fail
        rangePop = _fail
        mark = _fail

    _nvtx = _NVTXStub()  # type: ignore[assignment]

__all__ = ["range_push", "range_pop", "range_start", "range_end", "mark", "range"]


def range_push(msg, domain=None, category=None, color=None):
    """
    Pushes a range onto a stack of nested range span.  Returns zero-based
    depth of the range that is started.

    Args:
        msg (str): ASCII message to associate with range
        domain(str): ASCII domain name
        category(str): ASCII category name
        color(int) ARGB color
    """
    return _nvtx.rangePush(msg, domain, category, color)


def range_pop(domain=None):
    """
    Pops a range off of a stack of nested range spans.  Returns the
    zero-based depth of the range that is ended.

    Args:
        domain(str): ASCII domain name
    """
    return _nvtx.rangePop(domain)


def range_start(msg, domain=None, category=None, color=None) -> int:
    """
    Mark the start of a range with string message. It returns an unique handle
    for this range to pass to the corresponding call to rangeEnd().

    A key difference between this and range_push/range_pop is that the
    range_start/range_end version supports range across threads (start on one
    thread and end on another thread).

    Returns: A range handle (uint64_t) that can be passed to range_end().

    Args:
        msg (str): ASCII message to associate with the range.
        domain(str): ASCII domain name
        category(str): ASCII category name
        color(int) ARGB color
    """
    return _nvtx.rangeStart(msg, domain, category, color)


def range_end(range_id, domain=None) -> None:
    """
    Mark the end of a range for a given range_id.

    Args:
        range_id (int): an unique handle for the start range.
        domain(str): ASCII domain name
    """
    _nvtx.rangeEnd(range_id, domain)


def mark(msg, domain=None, category=None, color=None):
    """
    Describe an instantaneous event that occurred at some point.

    Args:
        msg (str): ASCII message to associate with the event.
        domain(str): ASCII domain name
        category(str): ASCII category name
        color(int) ARGB color
    """
    return _nvtx.mark(msg, domain, category, color)


@contextmanager
def range(msg, domain=None, category=None, color=None, *args, **kwargs):
    """
    Context manager / decorator that pushes an NVTX range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
    range_push(msg.format(*args, **kwargs), domain, category, color)
    try:
        yield
    finally:
        range_pop(domain)
