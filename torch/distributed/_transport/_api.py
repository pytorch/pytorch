from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, cast, Literal, Protocol, runtime_checkable, TYPE_CHECKING
from typing_extensions import Self

import torch
from torch.distributed import Work

from ._work import _validate_timeout, wait_all


if TYPE_CHECKING:
    from ._cuda_stream import _CudaStreamOrdering


@runtime_checkable
class MemoryView(Protocol):
    """A read-only view of registered memory."""

    def size(self) -> int: ...


@runtime_checkable
class MutableMemoryView(MemoryView, Protocol):
    """A view explicitly permitting writes to its registered memory."""

    @property
    def writable(self) -> Literal[True]: ...


@runtime_checkable
class RemoteBuffer(Protocol):
    """A backend-defined wire descriptor, excluding tensor contents/native handles.

    Exchange ``serialize()`` bytes and reconstruct them with the matching backend
    descriptor class's ``deserialize()``. Decoders validate their schema and
    version without executing code or importing classes named by the payload.
    Descriptors grant memory access: exchange them only with authorized peers.
    """

    def serialize(self) -> bytes: ...

    @classmethod
    def deserialize(cls, data: bytes) -> Self: ...


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
    plane. Registration is valid until unregistration or close, including after bind/connect.

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
    unregistering memory or closing; local completion does not establish that a peer has stopped accessing
    this endpoint. Descriptors are invalid after unregistration or their owner closes.

    ``read_stream`` and ``write_stream`` order transfers on the current CUDA
    stream: the transfer starts after prior work on the stream, and later work
    waits for it to complete. They return after enqueueing, without holding SMs
    while the transfer is in flight, and can be captured in CUDA graphs.
    Transfer failures terminate the process, since consumers may already be
    enqueued. Ordinary ``read`` and ``write`` do not interact with CUDA streams.
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device) if device is not None else None
        self._default_timeout: float | None = None
        self._cuda_streams: dict[tuple[int, int], _CudaStreamOrdering] = {}

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
        """Connect to a bound peer and return zero on success.

        Rank-to-endpoint lookup belongs in a separate application control-plane
        adapter; this interface addresses peers using opaque connection bytes.
        """

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

    def unregister_memory(
        self, memory: Memory, *, timeout: float | None = None
    ) -> None:
        """Unregister memory after coordinating with peers to stop remote access.

        Wait for locally submitted transfers using this registration first.
        Reused registrations share one lifetime: unregistering any handle
        invalidates all aliases, existing views, and exported descriptors.
        Register the tensor again and exchange fresh descriptors before reuse.
        Repeating unregistration on the same handle is a no-op while open.
        Backends without this operation raise ``NotImplementedError``.
        """
        raise NotImplementedError("transport does not support memory unregistration")

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

    def write_stream(
        self,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        stream: torch.cuda.Stream | None = None,
        timeout: float | None = None,
    ) -> None:
        """Write a local view in order on ``stream`` (default: current).

        Returns once enqueued. Backends with native stream support may override this.
        """
        _validate_timeout(timeout)
        if timeout == 0:
            raise ValueError("stream transfers require a positive timeout")
        self._check_transfer(local_buffer, remote_buffer, mutable=False)
        submit = partial(self.write_async, local_buffer, remote_buffer, timeout=timeout)
        self._cuda_stream(stream or torch.cuda.current_stream()).enqueue(
            submit, local_buffer
        )

    def read_stream(
        self,
        local_buffer: MutableMemoryView,
        remote_buffer: RemoteBuffer,
        *,
        stream: torch.cuda.Stream | None = None,
        timeout: float | None = None,
    ) -> None:
        """Read into a local view in order on ``stream`` (default: current)."""
        _validate_timeout(timeout)
        if timeout == 0:
            raise ValueError("stream transfers require a positive timeout")
        self._check_transfer(local_buffer, remote_buffer, mutable=True)
        submit = partial(self.read_async, local_buffer, remote_buffer, timeout=timeout)
        self._cuda_stream(stream or torch.cuda.current_stream()).enqueue(
            submit, local_buffer
        )

    def _check_transfer(  # noqa: B027
        self, local_buffer: MemoryView, remote_buffer: RemoteBuffer, *, mutable: bool
    ) -> None:
        """Validate a transfer before it is enqueued on a CUDA stream."""

    def _cuda_stream(self, stream: torch.cuda.Stream) -> _CudaStreamOrdering:
        from ._cuda_stream import _CudaStreamOrdering

        key = (stream.device.index, stream.cuda_stream)
        if key not in self._cuda_streams:
            self._cuda_streams[key] = _CudaStreamOrdering(stream)
        return self._cuda_streams[key]

    def _cuda_stream_views(self) -> list[MemoryView]:
        """Return views retained by enqueued or captured stream transfers."""
        return [
            view
            for ordering in self._cuda_streams.values()
            for view in ordering.retained_views()
        ]

    def _close_cuda_streams(self) -> None:
        for ordering in self._cuda_streams.values():
            ordering.close()
        self._cuda_streams.clear()

    async def write_async(
        self,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        timeout: float | None = None,
    ) -> None:
        """Write a local view, yielding to asyncio until the transfer finishes."""
        _validate_timeout(timeout)
        timeout = self._default_timeout if timeout is None else timeout
        started = asyncio.get_running_loop().time()
        work = cast(
            Work,
            self.write(local_buffer, remote_buffer, async_op=True, timeout=timeout),
        )
        remaining = (
            None
            if timeout is None
            else max(0.0, timeout - (asyncio.get_running_loop().time() - started))
        )
        await wait_all((work,), timeout=remaining)

    async def read_async(
        self,
        local_buffer: MutableMemoryView,
        remote_buffer: RemoteBuffer,
        *,
        timeout: float | None = None,
    ) -> None:
        """Read into a local view, yielding to asyncio until the transfer finishes."""
        _validate_timeout(timeout)
        timeout = self._default_timeout if timeout is None else timeout
        started = asyncio.get_running_loop().time()
        work = cast(
            Work,
            self.read(local_buffer, remote_buffer, async_op=True, timeout=timeout),
        )
        remaining = (
            None
            if timeout is None
            else max(0.0, timeout - (asyncio.get_running_loop().time() - started))
        )
        await wait_all((work,), timeout=remaining)

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
