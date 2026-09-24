from __future__ import annotations

import socket
from dataclasses import dataclass
from importlib import import_module
from threading import Lock
from typing import Any, Literal, TYPE_CHECKING

import torch

from ._blocking import _BlockingTransport


if TYPE_CHECKING:
    from ._api import MemoryView, MutableMemoryView, RemoteBuffer, Work


def _load_backend() -> Any:
    try:
        return import_module("mooncake.engine")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Mooncake transport requires the mooncake-transfer-engine package"
        ) from error


def _check_status(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"Mooncake {operation} failed with status {status}")


@dataclass(frozen=True)
class MooncakeRemoteBuffer:
    """A serializable descriptor for memory registered by a Mooncake peer."""

    endpoint: str
    address: int
    length: int


class MooncakeMemoryView:
    """A byte range in Mooncake-registered memory."""

    def __init__(self, memory: MooncakeMemory, offset: int, length: int) -> None:
        self._memory = memory
        self._offset = offset
        self._length = length

    def size(self) -> int:
        return self._length


class MooncakeMutableMemoryView(MooncakeMemoryView):
    """A writable byte range in Mooncake-registered memory."""

    writable: Literal[True] = True


class MooncakeMemory:
    """A tensor registered with a Mooncake transport."""

    def __init__(
        self, transport: MooncakeTransport, tensor: torch.Tensor, reused: bool
    ) -> None:
        self._transport = transport
        self._address = tensor.data_ptr()
        self._length = tensor.nbytes
        self._device = tensor.device
        self._reused = reused

    def _range(self, offset: int | None, length: int | None) -> tuple[int, int]:
        offset = 0 if offset is None else int(offset)
        if offset < 0 or offset > self._length:
            raise ValueError("offset is outside the registered memory")
        length = self._length - offset if length is None else int(length)
        if length < 0 or length > self._length - offset:
            raise ValueError("view exceeds the registered memory")
        return offset, length

    def to_view(
        self, offset: int | None = None, length: int | None = None
    ) -> MooncakeMemoryView:
        return MooncakeMemoryView(self, *self._range(offset, length))

    def to_mutable_view(
        self, offset: int | None = None, length: int | None = None
    ) -> MooncakeMutableMemoryView:
        return MooncakeMutableMemoryView(self, *self._range(offset, length))

    def to_remote_buffer(self, *, timeout: float | None = None) -> MooncakeRemoteBuffer:
        if timeout is not None:
            raise NotImplementedError(
                "per-call timeout is not supported by this prototype"
            )
        self._transport._ensure_open()
        return MooncakeRemoteBuffer(
            self._transport._endpoint, self._address, self._length
        )

    def reused_registration(self) -> bool:
        return self._reused


class MooncakeTransport(_BlockingTransport):
    """One-sided CPU/CUDA transfers using Mooncake's P2P metadata."""

    def __init__(
        self,
        device: torch.device | str | None = None,
        *,
        host: str | None = None,
        protocol: str = "rdma",
        device_name: str = "",
    ) -> None:
        super().__init__(device)
        if self.device is not None and self.device.type not in ("cpu", "cuda"):
            raise ValueError("Mooncake transport requires a CPU or CUDA device")
        host = socket.gethostname() if host is None else host
        if not host or not protocol:
            raise ValueError("host and protocol cannot be empty")
        host = host.strip("[]")
        host = f"[{host}]" if ":" in host else host
        engine = _load_backend().TransferEngine()
        _check_status(
            engine.initialize(host, "P2PHANDSHAKE", protocol, device_name),
            "initialization",
        )
        self._engine: Any = engine
        self._endpoint = f"{host}:{engine.get_rpc_port()}"
        self._peer: str | None = None
        self._registrations: dict[int, torch.UntypedStorage] = {}
        self._operation_lock = Lock()

    @staticmethod
    def supported() -> bool:
        try:
            return hasattr(_load_backend(), "TransferEngine")
        except Exception:
            return False

    def _ensure_open(self) -> Any:
        if self._engine is None:
            raise RuntimeError("transport is closed")
        return self._engine

    def bind(self, *, timeout: float | None = None) -> bytes:
        if timeout is not None:
            raise NotImplementedError(
                "per-call timeout is not supported by this prototype"
            )
        self._ensure_open()
        return self._endpoint.encode()

    def connect(self, peer_url: bytes, *, timeout: float | None = None) -> int:
        if timeout is not None:
            raise NotImplementedError(
                "per-call timeout is not supported by this prototype"
            )
        with self._operation_lock:
            self._ensure_open()
            if self._peer is not None:
                raise RuntimeError("transport is already connected")
            peer = peer_url.decode()
            if not peer:
                raise ValueError("peer endpoint cannot be empty")
            # Open the segment on first transfer, after peers register memory.
            self._peer = peer
        return 0

    def connected(self) -> bool:
        return self._peer is not None and self._engine is not None

    def register_memory(
        self, tensor: torch.Tensor, *, timeout: float | None = None
    ) -> MooncakeMemory:
        if timeout is not None:
            raise NotImplementedError(
                "per-call timeout is not supported by this prototype"
            )
        with self._operation_lock:
            engine = self._ensure_open()
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("tensor must be a torch.Tensor")
            if not tensor.is_contiguous():
                raise ValueError("Mooncake transport requires a contiguous tensor")
            if tensor.numel() == 0:
                raise ValueError("cannot register an empty tensor")
            self._check_device(tensor.device)
            if tensor.device.type not in ("cpu", "cuda"):
                raise ValueError("Mooncake transport requires CPU or CUDA tensors")
            if tensor.is_cuda:
                torch.cuda.current_stream(tensor.device).synchronize()
            storage = tensor.untyped_storage()
            address = storage.data_ptr()
            reused = address in self._registrations
            if not reused:
                # Register the allocation once, including overlapping tensor views.
                _check_status(
                    engine.register_memory(address, storage.nbytes()), "registration"
                )
                self._registrations[address] = storage
            return MooncakeMemory(self, tensor, reused)

    def _transfer(
        self,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        read: bool,
    ) -> int:
        with self._operation_lock:
            engine = self._ensure_open()
            expected = MooncakeMutableMemoryView if read else MooncakeMemoryView
            if not isinstance(local_buffer, expected) or (
                local_buffer._memory._transport is not self
            ):
                raise TypeError("local_buffer was not registered by this transport")
            if not isinstance(remote_buffer, MooncakeRemoteBuffer):
                raise TypeError("remote_buffer was not registered by Mooncake")
            if self._peer != remote_buffer.endpoint:
                raise ValueError("remote buffer does not belong to the connected peer")
            if local_buffer.size() > remote_buffer.length:
                raise ValueError("local view does not fit in the remote buffer")
            if local_buffer.size() == 0:
                return 0
            if local_buffer._memory._device.type == "cuda":
                torch.cuda.current_stream(local_buffer._memory._device).synchronize()
            transfer = engine.transfer_sync_read if read else engine.transfer_sync_write
            _check_status(
                transfer(
                    remote_buffer.endpoint,
                    local_buffer._memory._address + local_buffer._offset,
                    remote_buffer.address,
                    local_buffer.size(),
                ),
                "read" if read else "write",
            )
        return 0

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
        if not isinstance(local_buffer, MooncakeMemoryView):
            raise TypeError("local_buffer was not registered by this transport")
        return self._run_transfer(
            lambda: self._transfer(local_buffer, remote_buffer, read=False),
            local_buffer._memory._device,
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
        if not isinstance(local_buffer, MooncakeMutableMemoryView):
            raise TypeError("local_buffer was not registered by this transport")
        return self._run_transfer(
            lambda: self._transfer(local_buffer, remote_buffer, read=True),
            local_buffer._memory._device,
            async_op=async_op,
        )

    def close(self, *, timeout: float | None = None) -> None:
        if timeout is not None:
            raise NotImplementedError(
                "per-call timeout is not supported by this prototype"
            )
        self._close_work()
        with self._operation_lock:
            if self._engine is None:
                return
            error = None
            for address in self._registrations:
                try:
                    _check_status(
                        self._engine.unregister_memory(address), "unregistration"
                    )
                except Exception as exception:
                    if error is None:
                        error = exception
            # Destroy the engine before releasing storage, even if unregister fails.
            self._engine = None
            self._peer = None
            self._registrations.clear()
            if error is not None:
                raise error


__all__ = ["MooncakeTransport"]
