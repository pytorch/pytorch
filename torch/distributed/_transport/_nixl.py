from __future__ import annotations

import asyncio
import operator
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import timedelta
from importlib import import_module
from threading import RLock
from typing import Any

import torch

from ._api import MemoryView, MutableMemoryView, RemoteBuffer, Transport, Work
from ._work import _validate_timeout, wait_all


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
    storage: Any
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
        offset = 0 if offset is None else operator.index(offset)
        if offset < 0 or offset > self._registration.length:
            raise ValueError("offset is outside the registered memory")
        length = (
            self._registration.length - offset
            if length is None
            else operator.index(length)
        )
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

    def to_remote_buffer(self, *, timeout: float | None = None) -> NIXLRemoteBuffer:
        return self._transport._call(
            lambda: self._transport._remote_buffer(self._registration), timeout
        )

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
        capture_telemetry: bool = False,
        backend_options: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(device)
        if self.device is not None and self.device.type not in ("cpu", "cuda"):
            raise ValueError("NIXL transport requires a CPU or CUDA device")
        if not plugin:
            raise ValueError("plugin cannot be empty")
        if num_threads < 0:
            raise ValueError("num_threads must be nonnegative")
        _validate_timeout(timeout)
        if backend_options and num_threads:
            raise ValueError(
                "set plugin thread parameters in backend_options, not num_threads"
            )
        if backend_options is not None and not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in backend_options.items()
        ):
            raise TypeError("backend_options must map strings to strings")
        backend = _load_backend()
        self._plugin = plugin.upper()
        config = backend.nixl_agent_config(
            backends=[] if backend_options else [self._plugin],
            num_threads=num_threads,
            enable_prog_thread=enable_prog_thread,
            capture_telemetry=capture_telemetry,
        )
        self._agent = backend.nixl_agent(
            agent_name or f"torch-{uuid.uuid4().hex}", config
        )
        if backend_options:
            self._agent.create_backend(self._plugin, dict(backend_options))
        if self._plugin not in self._agent.backends:
            raise RuntimeError(f"NIXL plugin {self._plugin!r} is unavailable")
        self._timeout = timeout
        self._default_timeout = timeout
        self._peer_name: str | None = None
        self._registrations: dict[tuple[int, int, str], _Registration] = {}
        self._transfers: dict[int, Any] = {}
        self._pending: dict[int, _NIXLWork] = {}
        self._operation_lock = RLock()
        self._closed = False
        self._closing = False

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
        if self._closed or self._closing:
            raise RuntimeError("transport is closed")
        return self._agent

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
            await asyncio.sleep(0.001)
        try:
            yield
        finally:
            self._operation_lock.release()

    def _call(self, operation: Any, timeout: float | None) -> Any:
        # NIXL metadata/registration APIs are synchronous. The deadline bounds
        # lock acquisition, not execution inside the native library.
        with self._locked(timeout):
            self._ensure_open()
            return operation()

    def bind(self, *, timeout: float | None = None) -> bytes:
        return self._call(lambda: self._ensure_open().get_agent_metadata(), timeout)

    def connect(self, peer_url: bytes, *, timeout: float | None = None) -> int:
        return self._call(lambda: self._connect(peer_url), timeout)

    def _connect(self, peer_url: bytes) -> int:
        if self._peer_name is not None:
            raise RuntimeError("transport is already connected")
        name = _agent_name(self._ensure_open().add_remote_agent(peer_url))
        self._peer_name = name
        return 0

    def connected(self) -> bool:
        return self._peer_name is not None and not self._closed and not self._closing

    def register_memory(
        self, tensor: torch.Tensor, *, timeout: float | None = None
    ) -> NIXLMemory:
        return self._call(lambda: self._register_memory(tensor), timeout)

    def _register_memory(self, tensor: torch.Tensor) -> NIXLMemory:
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
            tensor.untyped_storage(),
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

    def _submit(
        self,
        operation: str,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        mutable: bool,
        timeout: float,
    ) -> _NIXLWork:
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
        if registration.tensor.data_ptr() != registration.address or (
            registration.tensor.numel() * registration.tensor.element_size()
            != registration.length
        ):
            raise RuntimeError(
                "registered tensor was resized or its storage was replaced"
            )
        work = _NIXLWork(self, local_buffer, remote_buffer, timeout)
        if local_buffer.size() == 0:
            work._done = True
            return work
        loaded_name = _agent_name(agent.add_remote_agent(remote_buffer.metadata))
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
            [(remote_buffer.address, local_buffer.size(), remote_buffer.device_id)],
            mem_type=remote_buffer.memory_type,
        )
        # A request handle belongs to exactly one live Work. Reusing a handle
        # while its previous transfer is pending corrupts native request state.
        work._handle = agent.initialize_xfer(
            operation,
            local_descs,
            remote_descs,
            remote_buffer.agent_name,
            backends=[self._plugin],
        )
        work._descriptors = (local_descs, remote_descs)
        self._transfers[id(work)] = work._handle
        self._pending[id(work)] = work
        _live_transports.add(self)
        try:
            work._state = agent.transfer(work._handle)
        except BaseException as error:
            # Dispatch may have started DMA before raising. Keep polling and
            # retain buffers until a native terminal state establishes safety.
            work._error = error
        return work

    def _start(
        self,
        operation: str,
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
                operation, local_buffer, remote_buffer, mutable=mutable, timeout=timeout
            )
        if async_op:
            return work
        # Native submission itself is synchronous and cannot be preempted.
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
        operation: str,
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
                operation, local_buffer, remote_buffer, mutable=mutable, timeout=timeout
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

    def _begin_close(self, timeout: float) -> list[_NIXLWork]:
        self._closing = True
        with self._locked(timeout):
            return list(self._pending.values())

    def close(self, *, timeout: float | None = None) -> None:
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
        interruptible; peer access must already have been stopped externally.
        """
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
        agent = self._agent
        for key, handle in list(self._transfers.items()):
            agent.release_xfer_handle(handle)
            del self._transfers[key]
        for key, registration in list(self._registrations.items()):
            agent.deregister_memory(registration.descs, backends=[self._plugin])
            del self._registrations[key]
        if self._peer_name is not None:
            agent.remove_remote_agent(self._peer_name)
            self._peer_name = None
        self._agent = None
        self._closed = True
        return 0


# Dropping a Work or transport must not free memory still used by DMA. Entries
# are removed only after polling establishes completion; users must wait/close.
_live_transports: set[NIXLTransport] = set()


class _PollingWork(Work):
    """Work that resolves a future from backend status checks.

    Subclasses implement a nonblocking, thread-safe ``_poll`` and record a
    terminal error in ``_error``. A failed status query must not report completion
    unless it establishes that the backend has stopped accessing memory.
    """

    def __init__(self, timeout: float | None = None) -> None:
        super().__init__()
        _validate_timeout(timeout)
        self._timeout = timeout
        self._error: BaseException | None = None
        self._future: torch.futures.Future[Any] = torch.futures.Future()
        self._future_lock = threading.Lock()
        self._future_completed = False
        self._progress_task: asyncio.Task[None] | None = None

    def _poll(self) -> bool:
        raise NotImplementedError

    def is_completed(self) -> bool:
        if not self._poll():
            return False
        with self._future_lock:
            notify = not self._future_completed
            self._future_completed = True
        # Callbacks may reenter the transport: never invoke them under its lock.
        if notify:
            if self._error is None:
                self._future.set_result([])
            else:
                error = self._error
                if not isinstance(error, Exception):
                    error = RuntimeError(str(error))
                self._future.set_exception(error)
        return True

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        seconds = timeout.total_seconds()
        _validate_timeout(seconds)
        seconds = seconds or self._timeout
        deadline = None if seconds is None else time.monotonic() + seconds
        while not self.is_completed():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "transport wait timed out; operation remains pending"
                )
            time.sleep(0.001)
        if self._error is not None:
            raise self._error
        return True

    def is_success(self) -> bool:
        return self.is_completed() and self._error is None

    def exception(self) -> BaseException | None:
        return self._error if self.is_completed() else None

    async def _drive_future(self) -> None:
        while not self.is_completed():
            await asyncio.sleep(0.001)

    def get_future(self) -> torch.futures.Future[list[torch.Tensor]]:
        if self.is_completed():
            return self._future
        loop = asyncio.get_running_loop()
        if self._progress_task is None or self._progress_task.done():
            self._progress_task = loop.create_task(self._drive_future())
        return self._future

    def result(self) -> list[torch.Tensor]:
        self.wait()
        return []

    def synchronize(self) -> None:
        self.wait()


class _NIXLWork(_PollingWork):
    def __init__(
        self,
        transport: NIXLTransport,
        local: MemoryView,
        remote: NIXLRemoteBuffer,
        timeout: float,
    ) -> None:
        super().__init__(timeout)
        self._transport = transport
        self._buffers = (local, remote)
        self._descriptors: Any = None
        self._handle: Any = None
        self._state = "PROC"
        self._done = False

    def _poll(self) -> bool:
        transport = self._transport
        if not transport._operation_lock.acquire(blocking=False):
            return False
        try:
            if self._done:
                return True
            if self._state == "PROC":
                try:
                    self._state = transport._agent.check_xfer_state(self._handle)
                except BaseException as error:
                    if self._error is None:
                        self._error = error
                    return False
            if self._state == "PROC":
                return False
            if self._state != "DONE" and self._error is None:
                self._error = RuntimeError(f"NIXL transfer failed: {self._state}")
            try:
                transport._agent.release_xfer_handle(self._handle)
                transport._transfers.pop(id(self))
            except BaseException as error:
                # Keep unreleased handles for close; disallow new requests so
                # an old Work's identity cannot be reused as a new handle key.
                transport._closing = True
                if self._error is None:
                    self._error = error
            self._done = True
            transport._pending.pop(id(self))
            if not transport._pending:
                _live_transports.discard(transport)
            return True
        finally:
            transport._operation_lock.release()


__all__ = ["NIXLTransport"]
