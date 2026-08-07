from typing import Any, Optional

import torch
from torch._utils import _dummy_type


__all__ = [
    "Stream",
    "StreamContext",
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


class StreamContext:
    r"""Context-manager that selects a given stream.

    MPS operators called within this context will be enqueued on the selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
        ``None``.
    """

    def __init__(self, stream: Optional["torch.mps.Stream"]):
        self.stream = stream

        self.prev_stream = (
            None if not torch.jit.is_scripting() else torch.mps.default_stream()
        )

    def __enter__(self):
        if self.stream is None:
            return
        self.prev_stream = torch.mps.current_stream()

        torch.mps.set_stream(self.stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        if self.stream is None:
            return
        torch.mps.set_stream(self.prev_stream)  # type: ignore[arg-type]


def stream(stream: Optional["torch.mps.Stream"]) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    """
    return StreamContext(stream)


def current_stream() -> Stream:
    r"""Return the currently selected :class:`Stream`."""
    stream_base = torch._C._mps_getCurrentStream()
    return Stream(stream_id=stream_base.stream_id)


def default_stream() -> Stream:
    r"""Return the default :class:`Stream`."""
    return Stream(stream_id=0)


def set_stream(stream: Stream):
    r"""Set the current stream. Usage of this function is discouraged in favor
    of the :meth:`stream` context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op if this
            argument is ``None``.
    """
    if stream is None:
        return
    torch._C._mps_setStream(stream)
