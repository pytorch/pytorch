from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import Future as _ThreadFuture
from contextlib import asynccontextmanager, contextmanager
from datetime import timedelta
from importlib import import_module
from threading import Condition, current_thread, RLock, Thread
from typing import Any, cast, TYPE_CHECKING

import torch

from .._api import (
    MemoryView,
    MutableMemoryView,
    Notification,
    PeerId,
    RemoteBuffer,
    Transport,
    Work,
)
from .._work import _FutureWork, _validate_timeout, wait_all
from ._memory import (
    _Registration,
    NIXLMemory,
    NIXLMemoryView,
    NIXLMutableMemoryView,
    NIXLRemoteBuffer,
)
from ._work import _live_transports, _NIXLWork


if TYPE_CHECKING:
    from .._api import Memory


# Fits every NIXL network plugin: GPUNetIO caps a notification, including the
# sender's name, at 8 KiB.
_MAX_NOTIFICATION_SIZE = 4 << 10


def _load_backend() -> Any:
    try:
        return import_module("nixl")
    except (ImportError, OSError) as error:
        raise RuntimeError("NIXL transport requires the nixl package") from error


def _agent_name(name: str | bytes) -> str:
    return name.decode() if isinstance(name, bytes) else name


class NIXLTransport(Transport):
    """One-sided tensor transport backed by a NIXL plugin.

    Backend options accepted by :func:`new_transport` include ``plugin="UCX"``,
    ``agent_name=None``, ``num_threads=0``, ``enable_prog_thread=True``,
    ``capture_telemetry=False``, ``backend_options=None``, and ``timeout=30.0``.
    ``backend_options`` maps plugin parameter names to strings and is forwarded
    to NIXL's ``create_backend``. NIXL is an optional dependency, imported only
    when selected. Transfers require registered views; raw tensors are rejected
    rather than allocated or registered implicitly.

    Notifications use the plugin's NIXL support; NIXL rejects them on plugins
    without it, such as storage plugins. Peer ids are agent names. A plugin may
    send a transfer's notification only when the initiator checks the transfer,
    as UCX does through NIXL 1.5.2 (ai-dynamo/nixl#2439), and NIXL cannot signal
    arriving notifications, so a thread polls transfers that carry a
    notification, and incoming notifications while a ``wait_notifications`` Work
    is pending.
    """

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
        self._peers: set[str] = set()
        self._remote_agents: set[str] = set()
        self._registrations: dict[tuple[int, int, str], _Registration] = {}
        self._transfers: dict[int, Any] = {}
        self._pending: dict[int, _NIXLWork] = {}
        self._operation_lock = RLock()
        self._name: str = self._agent.name
        # Notification state, guarded by _operation_lock.
        self._progress_cv = Condition(self._operation_lock)
        self._progress_thread: Thread | None = None
        self._notifying: set[_NIXLWork] = set()
        self._received: list[Notification] = []
        # Pending wait_notifications Works share this future; it is replaced
        # once it fires.
        self._available: _ThreadFuture[None] | None = None
        # Lifecycle: open -> closing -> closed. Closing rejects new operations and
        # is terminal; a failed or timed-out close retains unreleased resources
        # and retrying close only finishes cleanup. Failed transfer-handle or
        # peer-metadata cleanup also starts closing. Registrations are released
        # only by unregister or close, independent of Work completion.
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
            await asyncio.sleep(0)
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
        name = _agent_name(self._ensure_open().add_remote_agent(peer_url))
        self._peers.add(name)
        self._remote_agents.add(name)
        return 0

    def connected(self) -> bool:
        return bool(self._peers) and not self._closed and not self._closing

    @property
    def peer_id(self) -> PeerId:
        return self._name

    def max_notification_size(self) -> int:
        return _MAX_NOTIFICATION_SIZE

    def send_notification(
        self, peer: PeerId, payload: bytes, *, timeout: float | None = None
    ) -> Work:
        self._check_notification(payload)

        def send() -> _FutureWork:
            if peer not in self._peers:
                raise ValueError("peer is not connected")
            # No backend argument: NIXL's send_notif fails with one
            # (ai-dynamo/nixl#2484), and this agent has only the one backend.
            self._agent.send_notif(peer, payload)
            future: _ThreadFuture[None] = _ThreadFuture()
            future.set_result(None)
            return _FutureWork(future, None)

        return self._call(send, timeout)

    def poll_notifications(self) -> list[Notification]:
        with self._locked(None):
            self._ensure_open()
            self._drain_notifications()
            received, self._received = self._received, []
        return received

    def wait_notifications(self, *, timeout: float | None = None) -> Work:
        with self._locked(timeout):
            self._ensure_open()
            self._drain_notifications()
            if self._received:
                future: _ThreadFuture[None] = _ThreadFuture()
                future.set_result(None)
            else:
                if self._available is None:
                    self._available = _ThreadFuture()
                future = self._available
                self._start_progress()
        return _FutureWork(future, self._timeout if timeout is None else timeout)

    def _drain_notifications(self) -> None:
        # Caller holds _operation_lock.
        notifications = self._agent.get_new_notifs(backends=[self._plugin])
        for name, messages in notifications.items():
            peer = _agent_name(name)
            self._received += [Notification(peer, message) for message in messages]

    def _start_progress(self) -> None:
        # Caller holds _operation_lock.
        if self._progress_thread is None:
            self._progress_thread = Thread(
                target=self._progress, name="nixl-progress", daemon=True
            )
            self._progress_thread.start()

    def _progress(self) -> None:
        delay = 0.0
        try:
            while True:
                available, error = None, None
                with self._progress_cv:
                    if delay:
                        self._progress_cv.wait(delay)
                    if self._closed or (
                        not self._notifying and self._available is None
                    ):
                        self._progress_thread = None
                        return
                    done = [work for work in self._notifying if work._poll()]
                    self._notifying.difference_update(done)
                    if self._available is not None:
                        try:
                            self._drain_notifications()
                        except Exception as failure:
                            error = failure
                        if error is not None or self._received:
                            available, self._available = self._available, None
                # Outside the lock: completion runs Work callbacks, which may
                # call back into this transport.
                for work in done:
                    work.is_completed()
                if available is not None:
                    if error is None:
                        available.set_result(None)
                    else:
                        available.set_exception(error)
                delay = 0.0 if done or available else min(max(delay * 2, 1e-6), 1e-3)
        finally:
            with self._operation_lock:
                if self._progress_thread is current_thread():
                    self._progress_thread = None

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
            return NIXLMemory(self, registration)
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
        _live_transports.add(self)
        return NIXLMemory(self, registration)

    def unregister_memory(
        self, memory: Memory, *, timeout: float | None = None
    ) -> None:
        with self._locked(timeout):
            agent = self._ensure_open()
            if not isinstance(memory, NIXLMemory) or memory._transport is not self:
                raise TypeError("memory was not registered by this transport")
            registration = memory._registration
            if not registration.active:
                return
            if any(
                cast(NIXLMemoryView, view)._memory._registration is registration
                for view in self._cuda_stream_views()
            ):
                raise RuntimeError(
                    "memory is retained by CUDA stream transfers or graphs"
                )
            if any(
                work._buffers[0]._memory._registration is registration
                for work in self._pending.values()
            ):
                raise RuntimeError(
                    "memory has pending transfers; wait before unregistering"
                )
            agent.deregister_memory(registration.descs, backends=[self._plugin])
            registration.active = False
            key = next(
                k for k, value in self._registrations.items() if value is registration
            )
            del self._registrations[key]
            if not self._registrations and not self._transfers:
                _live_transports.discard(self)

    def _remote_buffer(self, registration: _Registration) -> NIXLRemoteBuffer:
        registration.ensure_active()
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

    def _check_transfer(
        self, local_buffer: MemoryView, remote_buffer: RemoteBuffer, *, mutable: bool
    ) -> None:
        expected_type = NIXLMutableMemoryView if mutable else NIXLMemoryView
        if not isinstance(local_buffer, expected_type) or (
            local_buffer._memory._transport is not self
        ):
            raise TypeError("local_buffer was not registered by this transport")
        if not isinstance(remote_buffer, NIXLRemoteBuffer):
            raise TypeError("remote_buffer was not registered by a NIXL transport")
        if remote_buffer.agent_name not in self._peers:
            raise ValueError("remote buffer does not belong to a connected peer")
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
        operation: str,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        mutable: bool,
        timeout: float,
        notification: bytes | None,
    ) -> _NIXLWork:
        agent = self._ensure_open()
        self._check_transfer(local_buffer, remote_buffer, mutable=mutable)
        if notification is not None:
            self._check_notification(notification)
            if local_buffer.size() == 0:
                raise ValueError("zero-byte transfers cannot carry notifications")
        local_buffer = cast(NIXLMemoryView, local_buffer)
        remote_buffer = cast(NIXLRemoteBuffer, remote_buffer)
        registration = local_buffer._memory._registration
        work = _NIXLWork(self, local_buffer, remote_buffer, timeout)
        if local_buffer.size() == 0:
            work._done = True
            return work
        loaded_name = _agent_name(agent.add_remote_agent(remote_buffer.metadata))
        if loaded_name != remote_buffer.agent_name:
            # Keep agents loaded by other peers; they may have transfers in flight.
            if loaded_name not in self._remote_agents:
                self._remote_agents.add(loaded_name)
                try:
                    agent.remove_remote_agent(loaded_name)
                except Exception:
                    # Preserve the agent for retryable close, including failed cleanup.
                    self._closing = True
                    _live_transports.add(self)
                    raise
                self._remote_agents.remove(loaded_name)
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
            work._state = agent.transfer(work._handle, notif_msg=notification or b"")
        except BaseException as error:
            # Dispatch may have started DMA before raising. Keep polling and
            # retain buffers until a native terminal state establishes safety.
            work._error = error
        if notification is not None:
            self._notifying.add(work)
            self._start_progress()
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
        notification: bytes | None,
    ) -> int | Work:
        timeout = self._timeout if timeout is None else timeout
        _validate_timeout(timeout)
        deadline = time.monotonic() + timeout
        with self._locked(timeout):
            work = self._submit(
                operation,
                local_buffer,
                remote_buffer,
                mutable=mutable,
                timeout=timeout,
                notification=notification,
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
        notification: bytes | None = None,
    ) -> int | Work:
        return self._start(
            "WRITE",
            local_buffer,
            remote_buffer,
            mutable=False,
            async_op=async_op,
            timeout=timeout,
            notification=notification,
        )

    def read(
        self,
        local_buffer: MutableMemoryView,
        remote_buffer: RemoteBuffer,
        *,
        async_op: bool = False,
        timeout: float | None = None,
        notification: bytes | None = None,
    ) -> int | Work:
        return self._start(
            "READ",
            local_buffer,
            remote_buffer,
            mutable=True,
            async_op=async_op,
            timeout=timeout,
            notification=notification,
        )

    async def _transfer_async(
        self,
        operation: str,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        mutable: bool,
        timeout: float | None,
        notification: bytes | None,
    ) -> None:
        timeout = self._timeout if timeout is None else timeout
        _validate_timeout(timeout)
        deadline = time.monotonic() + timeout
        async with self._locked_async(deadline):
            work = self._submit(
                operation,
                local_buffer,
                remote_buffer,
                mutable=mutable,
                timeout=timeout,
                notification=notification,
            )
        await wait_all([work], timeout=max(0.0, deadline - time.monotonic()))

    async def write_async(
        self,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        timeout: float | None = None,
        notification: bytes | None = None,
    ) -> None:
        await self._transfer_async(
            "WRITE",
            local_buffer,
            remote_buffer,
            mutable=False,
            timeout=timeout,
            notification=notification,
        )

    async def read_async(
        self,
        local_buffer: MutableMemoryView,
        remote_buffer: RemoteBuffer,
        *,
        timeout: float | None = None,
        notification: bytes | None = None,
    ) -> None:
        await self._transfer_async(
            "READ",
            local_buffer,
            remote_buffer,
            mutable=True,
            timeout=timeout,
            notification=notification,
        )

    def _begin_close(self, timeout: float) -> list[_NIXLWork]:
        self._closing = True
        with self._locked(timeout):
            available, self._available = self._available, None
            pending = list(self._pending.values())
        if available is not None:
            available.set_exception(RuntimeError("transport is closed"))
        return pending

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
        interruptible; peer access must already have been stopped externally.
        """
        self._close_cuda_streams()
        timeout = self._timeout if timeout is None else timeout
        _validate_timeout(timeout)
        deadline = time.monotonic() + timeout
        self._closing = True
        async with self._locked_async(deadline):
            available, self._available = self._available, None
            pending = list(self._pending.values())
        if available is not None:
            available.set_exception(RuntimeError("transport is closed"))
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
            registration.active = False
            del self._registrations[key]
        for name in list(self._remote_agents):
            agent.remove_remote_agent(name)
            self._remote_agents.remove(name)
        self._peers.clear()
        self._notifying.clear()
        self._agent = None
        self._closed = True
        _live_transports.discard(self)
        return 0
