from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast, Protocol, runtime_checkable
from typing_extensions import Self

import torch
from torch.distributed import Work

from ._work import _validate_timeout, wait_all


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

    def to_remote_buffer(self, *, timeout: float | None = None) -> RemoteBuffer: ...

    def reused_registration(self) -> bool: ...


class Transport(ABC):
    """Base class for one-sided tensor transports.

    ``read`` and ``write`` return zero on synchronous success. With
    ``async_op=True``, they return a :class:`torch.distributed.Work` whose
    ``wait`` blocks until completion and propagates transfer errors.
    ``is_completed`` includes failed operations. Work futures resolve to an
    empty list on success.

    ``read_async`` and ``write_async`` are asyncio coroutines. Cancellation or
    timeout may leave transfers pending. ``wait_all`` awaits their completion
    futures without blocking the event loop.

    Operations use byte ranges, not tensor shapes or dtypes. One endpoint has
    one outgoing peer; no process group, ranks, or matching receives are needed.
    Exchange connection bytes and remote descriptors through a trusted control
    plane. Registration is valid until close, including after bind/connect.

    ``timeout`` is a nonnegative, finite number of seconds; ``None`` selects the
    backend default. A timeout bounds the caller's wait, not the transfer's
    lifetime: it does not cancel DMA. Pending work retains its local buffers.
    After a timeout, wait for the Work or successfully close the transport before
    reusing buffers. A timed-out close rejects new operations but retains resources
    until close is retried successfully.
    Independent transfers may overlap; wait before submitting dependent or
    overlapping reads/writes. There is no implicit completion ordering.

    Never resize, replace storage, or modify buffers while registered/exposed to
    a peer. The application must coordinate remote access and notify peers before
    close; local completion does not establish that a peer has stopped accessing
    this endpoint. Descriptors are invalid after their owner closes.
    CUDA stream, graph capture, and tracing semantics are not part of this API.
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device) if device is not None else None
        self._default_timeout: float | None = None

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
    def bind(self, *, timeout: float | None = None) -> bytes:
        """Bind the endpoint and return its opaque connection URL."""

    @abstractmethod
    def connect(self, peer_url: bytes, *, timeout: float | None = None) -> int:
        """Connect to a bound peer and return zero on success."""

    @abstractmethod
    def connected(self) -> bool:
        """Return whether the endpoint is connected to its peer."""

    @abstractmethod
    def register_memory(
        self, tensor: torch.Tensor, *, timeout: float | None = None
    ) -> Memory:
        """Register a contiguous tensor, including after bind/connect or transfers.

        Exchange its remote-buffer descriptor with the peer before remote access.
        """

    @abstractmethod
    def write(
        self,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        async_op: bool = False,
        timeout: float | None = None,
    ) -> int | Work:
        """Write a local view; return Work for async_op=True, otherwise zero."""

    @abstractmethod
    def read(
        self,
        local_buffer: MutableMemoryView,
        remote_buffer: RemoteBuffer,
        *,
        async_op: bool = False,
        timeout: float | None = None,
    ) -> int | Work:
        """Read into a local view; return Work for async_op=True, otherwise zero."""

    async def write_async(
        self,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        timeout: float | None = None,
    ) -> None:
        """Write a local view, yielding to asyncio until the transfer finishes."""
        _validate_timeout(timeout)
        work = cast(Work, self.write(local_buffer, remote_buffer, async_op=True))
        await wait_all(
            (work,), timeout=self._default_timeout if timeout is None else timeout
        )

    async def read_async(
        self,
        local_buffer: MutableMemoryView,
        remote_buffer: RemoteBuffer,
        *,
        timeout: float | None = None,
    ) -> None:
        """Read into a local view, yielding to asyncio until the transfer finishes."""
        _validate_timeout(timeout)
        work = cast(Work, self.read(local_buffer, remote_buffer, async_op=True))
        await wait_all(
            (work,), timeout=self._default_timeout if timeout is None else timeout
        )

    @abstractmethod
    def close(self, *, timeout: float | None = None) -> None:
        """Drain outstanding operations and release transport resources."""

    async def close_async(self, *, timeout: float | None = None) -> None:
        """Await cleanup when supported by the backend.

        Unlike ``close``, this must not block the event loop while waiting for
        transfers. Blocking prototype backends do not implement this method.
        """
        raise NotImplementedError("async close is unsupported by this transport")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
