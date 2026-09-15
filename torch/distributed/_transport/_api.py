from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING
from typing_extensions import Self

import torch

from ._work import _WorkQueue, Work


if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class MemoryView(Protocol):
    """A read-only view of registered memory."""

    def size(self) -> int: ...


@runtime_checkable
class MutableMemoryView(MemoryView, Protocol):
    """A writable view of registered memory."""


@runtime_checkable
class RemoteBuffer(Protocol):
    """A serializable descriptor for registered memory on a peer."""


@runtime_checkable
class Memory(Protocol):
    """Memory registered with a transport."""

    def to_view(
        self, offset: int | None = None, length: int | None = None
    ) -> MemoryView: ...

    def to_mutable_view(
        self, offset: int | None = None, length: int | None = None
    ) -> MutableMemoryView: ...

    def to_remote_buffer(self) -> RemoteBuffer: ...

    def reused_registration(self) -> bool: ...


class Transport(ABC):
    """Base class for one-sided tensor transports.

    ``read`` and ``write`` return zero on synchronous success. With
    ``async_op=True``, they return a :class:`torch.distributed.Work` whose
    ``wait`` blocks until completion and propagates transfer errors.
    ``is_completed`` includes failed operations; ``get_future`` resolves to
    an empty list on success.

    Built-in backends serialize asynchronous operations on a worker thread.
    Local buffers are retained until completion and must not be modified or
    resized meanwhile. CUDA operations wait for the submitting stream's prior
    work; completion includes device work. Async CUDA graph capture is unsupported.
    Peers must keep exposed buffers ready and alive until all accesses finish.
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device) if device is not None else None
        self._work_queue = _WorkQueue()

    def _run_transfer(
        self,
        operation: Callable[[], int],
        device: torch.device,
        *,
        async_op: bool,
    ) -> int | Work:
        return self._work_queue.run(operation, device, async_op=async_op)

    def _close_work(self) -> None:
        self._work_queue.close()

    def _check_device(self, device: torch.device) -> None:
        if self.device is not None and (
            device.type != self.device.type
            or (self.device.index is not None and device.index != self.device.index)
        ):
            raise ValueError(f"expected a tensor on {self.device}, got {device}")

    @staticmethod
    @abstractmethod
    def supported() -> bool:
        """Return whether the transport can be used in this process."""

    @abstractmethod
    def bind(self) -> bytes:
        """Bind the endpoint and return its opaque connection URL."""

    @abstractmethod
    def connect(self, peer_url: bytes) -> int:
        """Connect to a bound peer and return zero on success."""

    @abstractmethod
    def connected(self) -> bool:
        """Return whether the endpoint is connected to its peer."""

    @abstractmethod
    def register_memory(self, tensor: torch.Tensor) -> Memory:
        """Register a contiguous tensor for transport operations."""

    @abstractmethod
    def write(
        self,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        async_op: bool = False,
    ) -> int | Work:
        """Write a local view; return Work for async_op=True, otherwise zero."""

    @abstractmethod
    def read(
        self,
        local_buffer: MutableMemoryView,
        remote_buffer: RemoteBuffer,
        *,
        async_op: bool = False,
    ) -> int | Work:
        """Read into a local view; return Work for async_op=True, otherwise zero."""

    @abstractmethod
    def close(self) -> None:
        """Drain outstanding operations and release transport resources."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
