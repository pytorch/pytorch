# mypy: allow-untyped-defs
r"""This package adds support for NVIDIA Tools Extension (NVTX) used in profiling."""

from contextlib import contextmanager
import functools
from typing import Any, Callable, Optional, TypeVar, Union, cast

import torch


# Global flag to determine whether functions should return tensors
_RETURN_TENSORS = False

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def enable_tensor_returns() -> None:
    """
    Enable tensor returns for NVTX functions to support torch._dynamo.

    This resolves 'torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor'
    errors that occur when compiling code with torch.compile() that contains NVTX functions.
    """
    global _RETURN_TENSORS
    _RETURN_TENSORS = True


def disable_tensor_returns() -> None:
    """Disable tensor returns for NVTX functions (default behavior)."""
    global _RETURN_TENSORS
    _RETURN_TENSORS = False


def tensor_compatible(default_val: Optional[Any] = None) -> Callable[[F], F]:
    """
    Decorator to make a function compatible with torch._dynamo by ensuring it returns a tensor
    when _RETURN_TENSORS is enabled.

    This resolves 'torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor'
    errors that occur when using torch._dynamo with NVTX functions.

    Args:
        default_val: The default value to convert to a tensor if the function returns None

    Returns:
        Decorated function that returns a tensor when _RETURN_TENSORS is True
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            if _RETURN_TENSORS:
                if result is None:
                    # Use provided default value or 0 if none provided
                    val = None if default_val is None else torch.tensor(default_val)
                    return val
                if isinstance(result, torch.Tensor):
                    return result
                try:
                    return torch.tensor(result)
                except (TypeError, ValueError):
                    # If we can't convert to tensor, return a dummy tensor
                    return torch.tensor(0)
            return result
        return cast(F, wrapper)
    return decorator


try:
    from torch._C import _nvtx
except ImportError:

    class _NVTXStub:
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError(
                "NVTX functions not installed. Are you sure you have a CUDA build?"
            )

        rangePushA = _fail
        rangePop = _fail
        markA = _fail

    _nvtx = _NVTXStub()  # type: ignore[assignment]

__all__ = ["range_push", "range_pop", "range_start", "range_end", "mark", "range"]


@tensor_compatible()
def range_push(msg):
    """
    Push a range onto a stack of nested range span.  Returns zero-based depth of the range that is started.

    Args:
        msg (str): ASCII message to associate with range
    """
    return _nvtx.rangePushA(msg)


@tensor_compatible()
def range_pop():
    """Pop a range off of a stack of nested range spans.  Returns the  zero-based depth of the range that is ended."""
    return _nvtx.rangePop()


@tensor_compatible()
def range_start(msg) -> int:
    """
    Mark the start of a range with string message. It returns an unique handle
    for this range to pass to the corresponding call to rangeEnd().

    A key difference between this and range_push/range_pop is that the
    range_start/range_end version supports range across threads (start on one
    thread and end on another thread).

    Returns: A range handle (uint64_t) that can be passed to range_end().

    Args:
        msg (str): ASCII message to associate with the range.
    """
    return _nvtx.rangeStartA(msg)


@tensor_compatible()
def range_end(range_id) -> None:
    """
    Mark the end of a range for a given range_id.

    Args:
        range_id (int): an unique handle for the start range.
    """
    _nvtx.rangeEnd(range_id)



def _device_range_start(msg: str, stream: int = 0) -> object:
    """
    Marks the start of a range with string message.
    It returns an opaque heap-allocated handle for this range
    to pass to the corresponding call to device_range_end().

    A key difference between this and range_start is that the
    range_start marks the range right away, while _device_range_start
    marks the start of the range as soon as all the tasks on the
    CUDA stream are completed.

    Returns: An opaque heap-allocated handle that should be passed to _device_range_end().

    Args:
        msg (str): ASCII message to associate with the range.
        stream (int): CUDA stream id.
    """
    return _nvtx.deviceRangeStart(msg, stream)


def _device_range_end(range_handle: object, stream: int = 0) -> None:
    """
    Mark the end of a range for a given range_handle as soon as all the tasks
    on the CUDA stream are completed.

    Args:
        range_handle: an unique handle for the start range.
        stream (int): CUDA stream id.
    """
    _nvtx.deviceRangeEnd(range_handle, stream)


@tensor_compatible()
def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Args:
        msg (str): ASCII message to associate with the event.
    """
    return _nvtx.markA(msg)


@contextmanager
def range(msg, *args, **kwargs):
    """
    Context manager / decorator that pushes an NVTX range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
    range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        range_pop()
