from contextlib import contextmanager

import torch
from torch._utils import _dummy_type


__all__ = [
    "Stream",
    "stream",
    "current_stream",
    "default_stream",
    "set_stream",
]

if not hasattr(torch._C, "_MPSStreamBase"):
    torch._C.__dict__["_MPSStreamBase"] = _dummy_type("_MPSStreamBase")


class Stream(torch._C._MPSStreamBase):
    r"""Wrapper around an MPS stream.

    An MPS stream is a linear sequence of execution that is independent from
    other streams. Use the :meth:`stream` context manager to ensure operators
    run on the corresponding stream.
    """

    def synchronize(self) -> None:
        r"""Wait for all the kernels in this stream to complete."""
        super().synchronize()

    @property
    def stream_id(self) -> int:
        return super().stream_id

    def __repr__(self):
        return f"<torch.mps.Stream stream_id={self.stream_id}>"


@contextmanager
def stream(stream: Stream | None):
    r"""Context manager that selects a given stream.

    MPS operators called within this context will be enqueued on the selected
    stream.

    Args:
        stream (Stream): selected stream. This manager is a no-op if it's ``None``.
    """
    prev_stream = torch.mps.current_stream()

    if stream is not None:
        torch.mps.set_stream(stream)

    try:
        yield
    finally:
        if stream is not None:
            torch.mps.set_stream(prev_stream)


def current_stream() -> Stream:
    r"""Return the currently selected :class:`Stream`."""
    stream_base = torch._C._mps_getCurrentStream()
    return Stream(stream_id=stream_base.stream_id)


def default_stream() -> Stream:
    r"""Return the default :class:`Stream`."""
    return Stream(stream_id=0)


def set_stream(stream: Stream | None) -> None:
    r"""Set the current stream. Usage of this function is discouraged in favor
    of the :meth:`stream` context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op if this
            argument is ``None``.
    """
    if stream is None:
        return
    torch._C._mps_setStream(stream)
