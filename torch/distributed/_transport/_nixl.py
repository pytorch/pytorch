from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from importlib import import_module
from threading import Lock
from typing import Any

import torch

from ._api import MemoryView, MutableMemoryView, RemoteBuffer, Transport


def _load_backend() -> Any:
    try:
        return import_module("nixl")
    except (ImportError, OSError) as error:
        raise RuntimeError("NIXL transport requires the nixl package") from error


def _agent_name(name: str | bytes) -> str:
    return name.decode() if isinstance(name, bytes) else name


@dataclass(frozen=True)
class NIXLRemoteBuffer:
    """A serializable descriptor for memory registered by a NIXL peer."""

    agent_name: str
    address: int
    length: int
    device_id: int
    memory_type: str
    metadata: bytes


@dataclass
class _Registration:
    tensor: torch.Tensor
    descs: Any
    address: int
    length: int
    device_id: int
    memory_type: str


class NIXLMemoryView:
    """A byte range in NIXL-registered memory."""

    def __init__(self, memory: NIXLMemory, offset: int, length: int) -> None:
        self._memory = memory
        self._offset = offset
        self._length = length

    def size(self) -> int:
        return self._length


class NIXLMutableMemoryView(NIXLMemoryView):
    """A writable byte range in NIXL-registered memory."""


class NIXLMemory:
    """A tensor registered with a NIXL transport."""

    def __init__(
        self, transport: NIXLTransport, registration: _Registration, reused: bool
    ) -> None:
        self._transport = transport
        self._registration = registration
        self._reused = reused

    def _range(self, offset: int | None, length: int | None) -> tuple[int, int]:
        offset = 0 if offset is None else int(offset)
        if offset < 0 or offset > self._registration.length:
            raise ValueError("offset is outside the registered memory")
        length = self._registration.length - offset if length is None else int(length)
        if length < 0 or length > self._registration.length - offset:
            raise ValueError("view exceeds the registered memory")
        return offset, length

    def to_view(
        self, offset: int | None = None, length: int | None = None
    ) -> NIXLMemoryView:
        return NIXLMemoryView(self, *self._range(offset, length))

    def to_mutable_view(
        self, offset: int | None = None, length: int | None = None
    ) -> NIXLMutableMemoryView:
        return NIXLMutableMemoryView(self, *self._range(offset, length))

    def to_remote_buffer(self) -> NIXLRemoteBuffer:
        return self._transport._remote_buffer(self._registration)

    def reused_registration(self) -> bool:
        return self._reused


class NIXLTransport(Transport):
    """One-sided tensor transport backed by a NIXL plugin."""

    def __init__(
        self,
        device: torch.device | str | None = None,
        *,
        plugin: str = "UCX",
        agent_name: str | None = None,
        num_threads: int = 0,
        enable_prog_thread: bool = True,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(device)
        if self.device is not None and self.device.type not in ("cpu", "cuda"):
            raise ValueError("NIXL transport requires a CPU or CUDA device")
        if not plugin:
            raise ValueError("plugin cannot be empty")
        if num_threads < 0:
            raise ValueError("num_threads must be nonnegative")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        backend = _load_backend()
        self._plugin = plugin.upper()
        config = backend.nixl_agent_config(
            backends=[self._plugin],
            num_threads=num_threads,
            enable_prog_thread=enable_prog_thread,
        )
        self._agent = backend.nixl_agent(
            agent_name or f"torch-{uuid.uuid4().hex}", config
        )
        if self._plugin not in self._agent.backends:
            raise RuntimeError(f"NIXL plugin {self._plugin!r} is unavailable")
        self._timeout = timeout
        self._peer_name: str | None = None
        self._registrations: dict[tuple[int, int, str], _Registration] = {}
        self._transfers: dict[tuple[Any, ...], Any] = {}
        self._operation_lock = Lock()
        self._closed = False

    @staticmethod
    def supported() -> bool:
        try:
            backend = _load_backend()
            return all(
                hasattr(backend, name) for name in ("nixl_agent", "nixl_agent_config")
            )
        except Exception:
            return False

    def _ensure_open(self) -> Any:
        if self._closed:
            raise RuntimeError("transport is closed")
        return self._agent

    def bind(self) -> bytes:
        return self._ensure_open().get_agent_metadata()

    def connect(self, peer_url: bytes) -> int:
        if self._peer_name is not None:
            raise RuntimeError("transport is already connected")
        name = _agent_name(self._ensure_open().add_remote_agent(peer_url))
        self._peer_name = name
        return 0

    def connected(self) -> bool:
        return self._peer_name is not None and not self._closed

    def register_memory(self, tensor: torch.Tensor) -> NIXLMemory:
        agent = self._ensure_open()
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        if not tensor.is_contiguous():
            raise ValueError("NIXL transport requires a contiguous tensor")
        if tensor.numel() == 0:
            raise ValueError("cannot register an empty tensor")
        self._check_device(tensor.device)
        if tensor.device.type not in ("cpu", "cuda"):
            raise ValueError("NIXL transport requires CPU or CUDA tensors")
        length = tensor.numel() * tensor.element_size()
        key = tensor.data_ptr(), length, str(tensor.device)
        if registration := self._registrations.get(key):
            return NIXLMemory(self, registration, reused=True)
        descs = agent.register_memory(tensor, backends=[self._plugin])
        registration = _Registration(
            tensor,
            descs,
            tensor.data_ptr(),
            length,
            max(tensor.get_device(), 0),
            "VRAM" if tensor.is_cuda else "DRAM",
        )
        self._registrations[key] = registration
        return NIXLMemory(self, registration, reused=False)

    def _remote_buffer(self, registration: _Registration) -> NIXLRemoteBuffer:
        metadata = self._ensure_open().get_partial_agent_metadata(
            registration.descs,
            inc_conn_info=True,
            backends=[self._plugin],
        )
        return NIXLRemoteBuffer(
            self._agent.name,
            registration.address,
            registration.length,
            registration.device_id,
            registration.memory_type,
            metadata,
        )

    def _transfer(
        self,
        operation: str,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        mutable: bool,
    ) -> int:
        agent = self._ensure_open()
        expected_type = NIXLMutableMemoryView if mutable else NIXLMemoryView
        if not isinstance(local_buffer, expected_type) or (
            local_buffer._memory._transport is not self
        ):
            raise TypeError("local_buffer was not registered by this transport")
        if not isinstance(remote_buffer, NIXLRemoteBuffer):
            raise TypeError("remote_buffer was not registered by a NIXL transport")
        if self._peer_name != remote_buffer.agent_name:
            raise ValueError("remote buffer does not belong to the connected peer")
        if local_buffer.size() > remote_buffer.length:
            raise ValueError("local view does not fit in the remote buffer")
        registration = local_buffer._memory._registration
        key = (
            operation,
            id(registration),
            local_buffer._offset,
            local_buffer.size(),
            remote_buffer.agent_name,
            remote_buffer.address,
            remote_buffer.length,
            remote_buffer.device_id,
            remote_buffer.memory_type,
        )
        with self._operation_lock:
            handle = self._transfers.get(key)
            if handle is None:
                loaded_name = _agent_name(
                    agent.add_remote_agent(remote_buffer.metadata)
                )
                if loaded_name != remote_buffer.agent_name:
                    raise ValueError("remote buffer metadata names a different agent")
                local_descs = agent.get_xfer_descs(
                    [
                        (
                            registration.address + local_buffer._offset,
                            local_buffer.size(),
                            registration.device_id,
                        )
                    ],
                    mem_type=registration.memory_type,
                )
                remote_descs = agent.get_xfer_descs(
                    [
                        (
                            remote_buffer.address,
                            local_buffer.size(),
                            remote_buffer.device_id,
                        )
                    ],
                    mem_type=remote_buffer.memory_type,
                )
                handle = agent.initialize_xfer(
                    operation,
                    local_descs,
                    remote_descs,
                    remote_buffer.agent_name,
                    backends=[self._plugin],
                )
                self._transfers[key] = handle
            try:
                state = agent.transfer(handle)
                deadline = time.monotonic() + self._timeout
                while state == "PROC":
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"NIXL {operation.lower()} timed out")
                    state = agent.check_xfer_state(handle)
                if state != "DONE":
                    raise RuntimeError(f"NIXL {operation.lower()} failed")
            except BaseException:
                self._transfers.pop(key, None)
                try:
                    agent.release_xfer_handle(handle)
                except Exception:
                    pass
                raise
        return 0

    def write(self, local_buffer: MemoryView, remote_buffer: RemoteBuffer) -> int:
        return self._transfer("WRITE", local_buffer, remote_buffer, mutable=False)

    def read(self, local_buffer: MutableMemoryView, remote_buffer: RemoteBuffer) -> int:
        return self._transfer("READ", local_buffer, remote_buffer, mutable=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        agent = self._agent
        peer_name = self._peer_name
        self._peer_name = None
        for handle in self._transfers.values():
            agent.release_xfer_handle(handle)
        self._transfers.clear()
        for registration in self._registrations.values():
            agent.deregister_memory(registration.descs, backends=[self._plugin])
        self._registrations.clear()
        if peer_name is not None:
            agent.remove_remote_agent(peer_name)
        self._agent = None


__all__ = ["NIXLTransport"]
