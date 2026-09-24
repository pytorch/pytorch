from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, TYPE_CHECKING

from ._blocking import _BlockingTransport
from ._serialization import _WireDescriptor


if TYPE_CHECKING:
    import torch

    from ._api import Memory, MemoryView, MutableMemoryView, RemoteBuffer, Work


def _load_backend() -> Any:
    try:
        return import_module("torchcomms._transport")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "torchcomms transport requires a torchcomms build with RDMA support"
        ) from error


@dataclass(frozen=True)
class TorchCommsRemoteBuffer(_WireDescriptor):
    _backend = "torchcomms"
    _fields = {"address": int, "length": int, "access_key": str}

    address: int
    length: int
    access_key: str


class _View:
    def __init__(self, memory: _Memory, view: Any) -> None:
        self._memory = memory
        self.native = view

    def size(self) -> int:
        return self.native.size()


class _MutableView(_View):
    writable: Literal[True] = True


class _Memory:
    def __init__(self, tensor: torch.Tensor, memory: Any) -> None:
        self._tensor = tensor
        self.native = memory

    def to_view(
        self, offset: int | None = None, length: int | None = None
    ) -> MemoryView:
        return _View(self, self.native.to_view(offset, length))

    def to_mutable_view(
        self, offset: int | None = None, length: int | None = None
    ) -> MutableMemoryView:
        return _MutableView(self, self.native.to_mutable_view(offset, length))

    def to_remote_buffer(self, *, timeout: float | None = None) -> RemoteBuffer:
        if timeout is not None:
            raise NotImplementedError(
                "per-call timeout is not supported by this prototype"
            )
        # Native bindings expose exactly (ptr, len, access_key); copy those
        # explicit fields, without invoking pickle or decoding native objects.
        descriptor = TorchCommsRemoteBuffer(
            *self.native.to_remote_buffer().__getstate__()
        )
        descriptor.serialize()  # Validate the native field types.
        return descriptor

    def reused_registration(self) -> bool:
        return self.native.reused_registration()


class TorchCommsTransport(_BlockingTransport):
    """Adapter for torchcomms' RDMA transport."""

    def __init__(self, device: torch.device | str | None = None) -> None:
        super().__init__(device)
        if self.device is None:
            raise ValueError("torchcomms transport requires an explicit CUDA device")
        backend = _load_backend()
        self._remote_type = backend.RdmaRemoteBuffer
        self._memory_type = backend.RdmaMemory
        self._transport_type = backend.RdmaTransport
        self._transport: Any = None
        self._closed = False

    @staticmethod
    def supported() -> bool:
        try:
            return bool(_load_backend().RdmaTransport.supported())
        except Exception:
            return False

    def _native(self) -> Any:
        if self._closed:
            raise RuntimeError("transport is closed")
        if self._transport is None:
            self._transport = self._transport_type(self.device)
        return self._transport

    def bind(self, *, timeout: float | None = None) -> bytes:
        if timeout is not None:
            raise NotImplementedError(
                "per-call timeout is not supported by this prototype"
            )
        return self._native().bind()

    def connect(self, peer_url: bytes, *, timeout: float | None = None) -> int:
        if timeout is not None:
            raise NotImplementedError(
                "per-call timeout is not supported by this prototype"
            )
        return self._native().connect(peer_url)

    def connected(self) -> bool:
        return self._transport is not None and self._transport.connected()

    def register_memory(
        self, tensor: torch.Tensor, *, timeout: float | None = None
    ) -> Memory:
        if timeout is not None:
            raise NotImplementedError(
                "per-call timeout is not supported by this prototype"
            )
        if self._closed:
            raise RuntimeError("transport is closed")
        return _Memory(tensor, self._memory_type(tensor))

    def write(
        self,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        async_op: bool = False,
        timeout: float | None = None,
    ) -> int | Work:
        if timeout is not None:
            raise NotImplementedError(
                "per-call timeout is not supported by this prototype"
            )
        if not isinstance(local_buffer, _View):
            raise TypeError("local_buffer was not registered by this transport")
        if not isinstance(remote_buffer, TorchCommsRemoteBuffer):
            raise TypeError("remote_buffer must be TorchCommsRemoteBuffer")
        native_remote = self._remote_type(
            remote_buffer.address, remote_buffer.length, remote_buffer.access_key
        )
        return self._run_transfer(
            lambda: self._native().write(local_buffer.native, native_remote),
            local_buffer._memory._tensor.device,
            async_op=async_op,
        )

    def read(
        self,
        local_buffer: MutableMemoryView,
        remote_buffer: RemoteBuffer,
        *,
        async_op: bool = False,
        timeout: float | None = None,
    ) -> int | Work:
        if timeout is not None:
            raise NotImplementedError(
                "per-call timeout is not supported by this prototype"
            )
        if not isinstance(local_buffer, _MutableView):
            raise TypeError("local_buffer was not registered by this transport")
        if not isinstance(remote_buffer, TorchCommsRemoteBuffer):
            raise TypeError("remote_buffer must be TorchCommsRemoteBuffer")
        native_remote = self._remote_type(
            remote_buffer.address, remote_buffer.length, remote_buffer.access_key
        )
        return self._run_transfer(
            lambda: self._native().read(local_buffer.native, native_remote),
            local_buffer._memory._tensor.device,
            async_op=async_op,
        )

    def close(self, *, timeout: float | None = None) -> None:
        if timeout is not None:
            raise NotImplementedError(
                "per-call timeout is not supported by this prototype"
            )
        self._close_work()
        self._closed = True
        self._transport = None


__all__ = ["TorchCommsTransport"]
