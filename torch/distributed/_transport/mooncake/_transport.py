from __future__ import annotations

import asyncio
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import timedelta
from importlib import import_module
from threading import RLock
from typing import Any, cast, TYPE_CHECKING

import torch

from .._api import MemoryView, MutableMemoryView, RemoteBuffer, Transport, Work
from .._work import _validate_timeout, wait_all
from ._memory import (
    _Registration,
    MooncakeMemory,
    MooncakeMemoryView,
    MooncakeMutableMemoryView,
    MooncakeRemoteBuffer,
)
from ._work import _live_transports, _MooncakeWork


if TYPE_CHECKING:
    from .._api import Memory


def _load_backend() -> Any:
    try:
        return import_module("mooncake.engine")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Mooncake transport requires the mooncake-transfer-engine package"
        ) from error


def _format_host(host: str) -> str:
    host = host.strip("[]")
    return f"[{host}]" if ":" in host else host


class MooncakeTransport(Transport):
    """One-sided tensor transport backed by Mooncake's Transfer Engine.

    Uses Mooncake's P2P metadata exchange: ``bind`` returns the endpoint's
    ``host:rpc_port`` and ``connect`` records the peer endpoint without an
    external metadata server. Peers are addressed by endpoint and their
    registered segments are reached directly, so remote descriptors carry no
    opaque metadata blob. Mooncake is an optional dependency, imported only when
    selected. Transfers require registered views; raw tensors are rejected rather
    than allocated or registered implicitly.

    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        *,
        host: str | None = None,
        protocol: str = "rdma",
        device_name: str = "",
        metadata_server: str = "P2PHANDSHAKE",
        timeout: float = 30.0,
        max_concurrent_transfers: int = 16,
    ) -> None:
        super().__init__(device)
        if self.device is not None and self.device.type not in ("cpu", "cuda"):
            raise ValueError("Mooncake transport requires a CPU or CUDA device")
        host = socket.gethostname() if host is None else host
        if not host or not protocol:
            raise ValueError("host and protocol cannot be empty")
        if max_concurrent_transfers < 1:
            raise ValueError("max_concurrent_transfers must be positive")
        _validate_timeout(timeout)
        host = _format_host(host)
        backend = _load_backend()
        engine = backend.TransferEngine()
        if engine.initialize(host, metadata_server, protocol, device_name) != 0:
            raise RuntimeError("Mooncake transport initialization failed")
        self._backend = backend
        self._engine: Any = engine
        self._endpoint = f"{host}:{engine.get_rpc_port()}"
        self._protocol = protocol
        self._timeout = timeout
        self._default_timeout = timeout
        self._peer: str | None = None
        self._registrations: dict[tuple[int, int, str], _Registration] = {}
        # In-flight transfers; completed entries are pruned lazily on submit.
        self._pending: dict[int, _MooncakeWork] = {}
        # Async transfers run their blocking transfer_sync on this pool; the
        # TransferEngine supports concurrent transfers across worker threads.
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_transfers, thread_name_prefix="mooncake-xfer"
        )
        self._operation_lock = RLock()
        # Closing rejects new operations and is terminal.
        # Registrations are released only by unregister or close,
        # independent of Work completion
        self._closed = False
        self._closing = False

    @staticmethod
    def supported() -> bool:
        try:
            return hasattr(_load_backend(), "TransferEngine")
        except Exception:
            return False

    def _ensure_open(self) -> Any:
        if self._closed or self._closing:
            raise RuntimeError("transport is closed")
        return self._engine

    @contextmanager
    def _locked(self, timeout: float | None):
        timeout = self._timeout if timeout is None else timeout
        _validate_timeout(timeout)
        if not self._operation_lock.acquire(timeout=timeout):
            raise TimeoutError("transport lock wait timed out")
        try:
            yield
        finally:
            self._operation_lock.release()

    @asynccontextmanager
    async def _locked_async(self, deadline: float):
        while not self._operation_lock.acquire(blocking=False):
            if time.monotonic() >= deadline:
                raise TimeoutError("transport lock wait timed out")
            await asyncio.sleep(0)
        try:
            yield
        finally:
            self._operation_lock.release()

    def _call(self, operation: Any, timeout: float | None) -> Any:
        with self._locked(timeout):
            self._ensure_open()
            return operation()

    def bind(self, *, timeout: float | None = None) -> bytes:
        return self._call(lambda: self._endpoint.encode(), timeout)

    def connect(self, peer_url: bytes, *, timeout: float | None = None) -> int:
        return self._call(lambda: self._connect(peer_url), timeout)

    def _connect(self, peer_url: bytes) -> int:
        if self._peer is not None:
            raise RuntimeError("transport is already connected")
        # Mooncake handshakes lazily on the first transfer, hence record the endpoint
        self._peer = peer_url.decode()
        return 0

    def connected(self) -> bool:
        return self._peer is not None and not self._closed and not self._closing

    def register_memory(
        self, tensor: torch.Tensor, *, timeout: float | None = None
    ) -> MooncakeMemory:
        return self._call(lambda: self._register_memory(tensor), timeout)

    def _register_memory(self, tensor: torch.Tensor) -> MooncakeMemory:
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
        length = tensor.numel() * tensor.element_size()
        key = tensor.data_ptr(), length, str(tensor.device)
        if registration := self._registrations.get(key):
            return MooncakeMemory(self, registration)
        device_id = max(tensor.get_device(), 0)
        location = f"cuda:{device_id}" if tensor.is_cuda else "cpu:0"
        if engine.register_memory(tensor.data_ptr(), length, location) != 0:
            raise RuntimeError("Mooncake memory registration failed")
        registration = _Registration(
            tensor,
            tensor.untyped_storage(),
            tensor.data_ptr(),
            length,
            device_id,
            "VRAM" if tensor.is_cuda else "DRAM",
        )
        self._registrations[key] = registration
        _live_transports.add(self)
        return MooncakeMemory(self, registration)

    def unregister_memory(
        self, memory: Memory, *, timeout: float | None = None
    ) -> None:
        with self._locked(timeout):
            engine = self._ensure_open()
            if not isinstance(memory, MooncakeMemory) or memory._transport is not self:
                raise TypeError("memory was not registered by this transport")
            registration = memory._registration
            if not registration.active:
                return
            if any(
                cast(MooncakeMemoryView, view)._memory._registration is registration
                for view in self._cuda_stream_views()
            ):
                raise RuntimeError(
                    "memory is retained by CUDA stream transfers or graphs"
                )
            if any(
                work._buffers[0]._memory._registration is registration
                for work in self._pending.values()
                if not work.is_completed()
            ):
                raise RuntimeError(
                    "memory has pending transfers; wait before unregistering"
                )
            if engine.unregister_memory(registration.address) != 0:
                raise RuntimeError("Mooncake memory unregistration failed")
            registration.active = False
            key = next(
                k for k, value in self._registrations.items() if value is registration
            )
            del self._registrations[key]
            if not self._registrations:
                _live_transports.discard(self)

    def _remote_buffer(self, registration: _Registration) -> MooncakeRemoteBuffer:
        registration.ensure_active()
        return MooncakeRemoteBuffer(
            self._endpoint,
            registration.address,
            registration.length,
            registration.device_id,
            registration.memory_type,
        )

    def _check_transfer(
        self, local_buffer: MemoryView, remote_buffer: RemoteBuffer, *, mutable: bool
    ) -> None:
        expected_type = MooncakeMutableMemoryView if mutable else MooncakeMemoryView
        if not isinstance(local_buffer, expected_type) or (
            local_buffer._memory._transport is not self
        ):
            raise TypeError("local_buffer was not registered by this transport")
        if not isinstance(remote_buffer, MooncakeRemoteBuffer):
            raise TypeError("remote_buffer was not registered by a Mooncake transport")
        if self._peer != remote_buffer.endpoint:
            raise ValueError("remote buffer does not belong to the connected peer")
        if local_buffer.size() > remote_buffer.length:
            raise ValueError("local view does not fit in the remote buffer")
        registration = local_buffer._memory._registration
        registration.ensure_active()
        if registration.tensor.data_ptr() != registration.address or (
            registration.tensor.numel() * registration.tensor.element_size()
            != registration.length
        ):
            raise RuntimeError(
                "registered tensor was resized or its storage was replaced"
            )

    def _submit(
        self,
        opcode: str,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        mutable: bool,
        timeout: float,
    ) -> _MooncakeWork:
        engine = self._ensure_open()
        self._check_transfer(local_buffer, remote_buffer, mutable=mutable)
        local_buffer = cast(MooncakeMemoryView, local_buffer)
        remote_buffer = cast(MooncakeRemoteBuffer, remote_buffer)
        registration = local_buffer._memory._registration
        size = local_buffer.size()
        if size == 0:
            return _MooncakeWork(self, local_buffer, remote_buffer, None, timeout)
        operation = (
            self._backend.TransferOpcode.Read
            if opcode == "READ"
            else self._backend.TransferOpcode.Write
        )
        endpoint = remote_buffer.endpoint
        local_address = registration.address + local_buffer._offset
        remote_address = remote_buffer.address

        # Mooncake's transfer APIs block until completion, so run the blocking
        # call on the executor; the Work settles when the worker thread returns.
        def run() -> None:
            status = engine.transfer_sync(
                endpoint, local_address, remote_address, size, operation
            )
            if status != 0:
                raise RuntimeError(f"Mooncake transfer failed: {status}")

        self._pending = {
            key: pending
            for key, pending in self._pending.items()
            if not pending.is_completed()
        }
        future = self._executor.submit(run)
        work = _MooncakeWork(self, local_buffer, remote_buffer, future, timeout)
        self._pending[id(work)] = work
        _live_transports.add(self)
        return work

    def _start(
        self,
        opcode: str,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        mutable: bool,
        async_op: bool,
        timeout: float | None,
    ) -> int | Work:
        timeout = self._timeout if timeout is None else timeout
        _validate_timeout(timeout)
        deadline = time.monotonic() + timeout
        with self._locked(timeout):
            work = self._submit(
                opcode, local_buffer, remote_buffer, mutable=mutable, timeout=timeout
            )
        if async_op:
            return work
        work._timeout = max(0.0, deadline - time.monotonic())
        work.wait()
        return 0

    def write(
        self,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        async_op: bool = False,
        timeout: float | None = None,
    ) -> int | Work:
        return self._start(
            "WRITE",
            local_buffer,
            remote_buffer,
            mutable=False,
            async_op=async_op,
            timeout=timeout,
        )

    def read(
        self,
        local_buffer: MutableMemoryView,
        remote_buffer: RemoteBuffer,
        *,
        async_op: bool = False,
        timeout: float | None = None,
    ) -> int | Work:
        return self._start(
            "READ",
            local_buffer,
            remote_buffer,
            mutable=True,
            async_op=async_op,
            timeout=timeout,
        )

    async def _transfer_async(
        self,
        opcode: str,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        mutable: bool,
        timeout: float | None,
    ) -> None:
        timeout = self._timeout if timeout is None else timeout
        _validate_timeout(timeout)
        deadline = time.monotonic() + timeout
        async with self._locked_async(deadline):
            work = self._submit(
                opcode, local_buffer, remote_buffer, mutable=mutable, timeout=timeout
            )
        await wait_all([work], timeout=max(0.0, deadline - time.monotonic()))

    async def write_async(
        self,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        timeout: float | None = None,
    ) -> None:
        await self._transfer_async(
            "WRITE", local_buffer, remote_buffer, mutable=False, timeout=timeout
        )

    async def read_async(
        self,
        local_buffer: MutableMemoryView,
        remote_buffer: RemoteBuffer,
        *,
        timeout: float | None = None,
    ) -> None:
        await self._transfer_async(
            "READ", local_buffer, remote_buffer, mutable=True, timeout=timeout
        )

    def _begin_close(self, timeout: float) -> list[_MooncakeWork]:
        self._closing = True
        with self._locked(timeout):
            return list(self._pending.values())

    def close(self, *, timeout: float | None = None) -> None:
        self._close_cuda_streams()
        timeout = self._timeout if timeout is None else timeout
        _validate_timeout(timeout)
        deadline = time.monotonic() + timeout
        pending = self._begin_close(timeout)
        error: BaseException | None = None
        for work in pending:
            try:
                work.wait(
                    timedelta(seconds=max(0.0, deadline - time.monotonic()))
                    or timedelta(microseconds=1)
                )
            except BaseException as failure:
                if not work.is_completed():
                    raise
                if error is None:
                    error = failure
        with self._locked(max(0.0, deadline - time.monotonic())):
            self._release_resources()
        if error is not None:
            raise error

    async def close_async(self, *, timeout: float | None = None) -> None:
        """Await outstanding DMA before synchronous native resource cleanup.

        Cancellation/timeout retains resources and rejects new submissions.
        Retry close after pending work completes. Native cleanup is not
        interruptible, peer access must already have been stopped externally.
        """
        self._close_cuda_streams()
        timeout = self._timeout if timeout is None else timeout
        _validate_timeout(timeout)
        deadline = time.monotonic() + timeout
        self._closing = True
        async with self._locked_async(deadline):
            pending = list(self._pending.values())
        error: BaseException | None = None
        try:
            await wait_all(pending, timeout=max(0.0, deadline - time.monotonic()))
        except BaseException as failure:
            if isinstance(failure, asyncio.CancelledError) or any(
                not w.is_completed() for w in pending
            ):
                raise
            error = failure
        async with self._locked_async(deadline):
            self._release_resources()
        if error is not None:
            raise error

    def _release_resources(self) -> int:
        if self._closed:
            return 0
        # Pending transfers were already drained by close()/close_async(), so the
        # workers are idle; shut down without joining to keep close_async from
        # blocking the event loop.
        self._executor.shutdown(wait=False)
        engine = self._engine
        for key, registration in list(self._registrations.items()):
            engine.unregister_memory(registration.address)
            registration.active = False
            del self._registrations[key]
        self._peer = None
        # Mooncake has no explicit shutdown
        self._engine = None
        self._closed = True
        _live_transports.discard(self)
        return 0
