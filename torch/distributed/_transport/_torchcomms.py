from __future__ import annotations

from importlib import import_module
from typing import Any, TYPE_CHECKING

from ._api import Memory, MemoryView, MutableMemoryView, RemoteBuffer, Transport


if TYPE_CHECKING:
    import torch


def _load_backend() -> Any:
    try:
        return import_module("torchcomms._transport")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "torchcomms transport requires a torchcomms build with RDMA support"
        ) from error


class _View:
    def __init__(self, memory: _Memory, view: Any) -> None:
        self._memory = memory
        self.native = view

    def size(self) -> int:
        return self.native.size()


class _MutableView(_View):
    pass


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

    def to_remote_buffer(self) -> RemoteBuffer:
        return self.native.to_remote_buffer()

    def reused_registration(self) -> bool:
        return self.native.reused_registration()


class TorchCommsTransport(Transport):
    """Adapter for torchcomms' RDMA transport."""

    def __init__(self, device: torch.device | str) -> None:
        super().__init__(device)
        backend = _load_backend()
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

    def bind(self) -> bytes:
        return self._native().bind()

    def connect(self, peer_url: bytes) -> int:
        return self._native().connect(peer_url)

    def connected(self) -> bool:
        return self._transport is not None and self._transport.connected()

    def register_memory(self, tensor: torch.Tensor) -> Memory:
        if self._closed:
            raise RuntimeError("transport is closed")
        return _Memory(tensor, self._memory_type(tensor))

    def write(self, local_buffer: MemoryView, remote_buffer: RemoteBuffer) -> int:
        if not isinstance(local_buffer, _View):
            raise TypeError("local_buffer was not registered by this transport")
        return self._native().write(local_buffer.native, remote_buffer)

    def read(self, local_buffer: MutableMemoryView, remote_buffer: RemoteBuffer) -> int:
        if not isinstance(local_buffer, _MutableView):
            raise TypeError("local_buffer was not registered by this transport")
        return self._native().read(local_buffer.native, remote_buffer)

    def close(self) -> None:
        self._closed = True
        self._transport = None


__all__ = ["TorchCommsTransport"]
