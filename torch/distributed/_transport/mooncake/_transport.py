from __future__ import annotations

import asyncio
import base64
import logging
import os
import socket
import time
from concurrent.futures import Future as _ThreadFuture, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
from importlib import import_module
from threading import Condition, RLock, Thread
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
    MooncakeMemory,
    MooncakeMemoryView,
    MooncakeMutableMemoryView,
    MooncakeRemoteBuffer,
)
from ._work import _live_transports, _MooncakeWork


logger = logging.getLogger(__name__)

_pinned_registrations: list[tuple[Any, _Registration]] = []

# Payloads travel base64-encoded because Mooncake returns notify messages as
# str. 2 KiB encodes to well under Mooncake's default 4 KiB RDMA notify slot.
_MAX_NOTIFICATION_SIZE = 2048


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
    if ":" not in host:
        return host
    if "MC_USE_IPV6" not in os.environ:
        raise ValueError("Mooncake requires MC_USE_IPV6 to be set for IPv6 hosts")
    return f"[{host}]"


class MooncakeTransport(Transport):
    """One-sided tensor transport backed by Mooncake's Transfer Engine.

    Uses Mooncake's P2P metadata exchange: ``bind`` returns the endpoint's
    ``host:rpc_port`` and ``connect`` records a peer endpoint without an
    external metadata server. An endpoint may connect to several peers;
    reconnecting with the same ``bind()`` bytes is a no-op. Peers are addressed
    by endpoint and their registered segments are reached directly, so remote
    descriptors carry no opaque metadata blob. Mooncake is an optional
    dependency, imported only when selected. Transfers require registered views;
    raw tensors are rejected rather than allocated or registered implicitly.

    There is no per-peer disconnect; ``close`` releases all peers. A failed
    transfer fails only its Work. A timed-out transfer also pins its local
    memory, which is never unregistered and rejects further transfers, because
    Mooncake may still be accessing it. Descriptors are not tied to the peer that
    sent them, so an endpoint serving several peers must authorize each request,
    for example by checking ``remote_buffer.endpoint``.

    Mooncake has no completion callback or fd yet, so transfers are submitted
    with ``batch_transfer_async`` and a single poller thread per transport checks
    them with ``transfer_check_status``, backing off while nothing completes and
    sleeping while nothing is in flight. The poller settles each Work, which
    wakes ``write_async``/``read_async``/``close_async`` through the asyncio
    future bridge without blocking the loop. The short lock-guarded
    submit/cleanup sections are offloaded with ``asyncio.to_thread`` rather than
    spinning on the lock.

    Notifications are sent by zero-length transfers carrying a Mooncake notify,
    on a small thread pool, since Mooncake's Python API has no standalone
    notify. A transfer's notification is sent the same way after the transfer
    succeeds: in Mooncake 0.3.14 a notify attached to a nonempty RDMA transfer
    can be lost. Mooncake cannot signal arriving notifies either, so a thread
    polls for them only while a ``wait_notifications`` Work is pending. Peer
    ids are ``host:rpc_port`` endpoints. Mooncake keeps at most 1024 undrained
    notifies on some paths, dropping the oldest, so receivers must drain them
    regularly.
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
    ) -> None:
        super().__init__(device)
        if self.device is not None and self.device.type not in ("cpu", "cuda"):
            raise ValueError("Mooncake transport requires a CPU or CUDA device")
        host = socket.gethostname() if host is None else host
        if not host or not protocol:
            raise ValueError("host and protocol cannot be empty")
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
        self._peers: set[str] = set()
        self._registrations: dict[tuple[int, int, str], _Registration] = {}
        # In-flight transfers; each entry is pruned by a done callback when its
        # transfer settles (see _submit), so a completed work is never revisited.
        self._pending: dict[int, _MooncakeWork] = {}
        self._batches: dict[int, tuple[_ThreadFuture[None], _Registration]] = {}
        self._poll_cv = Condition()
        self._poller: Thread | None = None
        self._poller_stopped = False
        # Threads start on first use; 4 keeps one slow peer from stalling the rest.
        self._notifier = ThreadPoolExecutor(4, thread_name_prefix="mooncake-notify")
        self._notify_cv = Condition()
        self._received: list[Notification] = []
        # Pending wait_notifications Works share this future, so a timed-out
        # wait leaves nothing behind; it is replaced once it fires.
        self._available: _ThreadFuture[None] | None = None
        self._watcher: Thread | None = None
        # Guarded by _operation_lock, so close() sees every submitted send.
        self._sends: set[_FutureWork] = set()
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

    def _run_locked(self, operation: Any, timeout: float) -> Any:
        with self._locked(timeout):
            return operation()

    def _call(self, operation: Any, timeout: float | None) -> Any:
        with self._locked(timeout):
            self._ensure_open()
            return operation()

    def bind(self, *, timeout: float | None = None) -> bytes:
        return self._call(lambda: self._endpoint.encode(), timeout)

    def connect(self, peer_url: bytes, *, timeout: float | None = None) -> int:
        return self._call(lambda: self._connect(peer_url), timeout)

    def _connect(self, peer_url: bytes) -> int:
        try:
            endpoint = peer_url.decode()
        except UnicodeDecodeError:
            endpoint = ""
        host, _, port = endpoint.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError("peer_url is not a Mooncake bind() endpoint")
        # Mooncake handshakes lazily on the first transfer, hence record the endpoint
        self._peers.add(endpoint)
        return 0

    def connected(self) -> bool:
        return bool(self._peers) and not self._closed and not self._closing

    @property
    def peer_id(self) -> PeerId:
        return self._endpoint

    def max_notification_size(self) -> int:
        return _MAX_NOTIFICATION_SIZE

    def send_notification(
        self, peer: PeerId, payload: bytes, *, timeout: float | None = None
    ) -> Work:
        self._check_notification(payload)
        _validate_timeout(timeout)
        wait_timeout = self._timeout if timeout is None else timeout

        def submit() -> _FutureWork:
            if peer not in self._peers:
                raise ValueError("peer is not connected")
            future = self._send_notify(self._engine, peer, payload)
            work = _FutureWork(future, wait_timeout)

            def untrack(_: _ThreadFuture[None]) -> None:
                with self._operation_lock:
                    self._sends.discard(work)

            self._sends.add(work)
            work._future.add_done_callback(untrack)
            return work

        return self._call(submit, timeout)

    def _send_notify(
        self, engine: Any, peer: str, payload: bytes
    ) -> _ThreadFuture[None]:
        notify = self._backend.TransferNotify(
            self._endpoint, base64.b64encode(payload).decode()
        )

        def send() -> None:
            if engine.transfer_sync(
                peer, 0, 0, 0, self._backend.TransferOpcode.Write, notify
            ):
                raise RuntimeError("Mooncake notification send failed")

        return self._notifier.submit(send)

    def _then_notify(
        self,
        transfer: _ThreadFuture[None],
        engine: Any,
        peer: str,
        payload: bytes,
    ) -> _ThreadFuture[None]:
        result: _ThreadFuture[None] = _ThreadFuture()

        def forward(done: _ThreadFuture[None]) -> None:
            if (error := done.exception()) is not None:
                result.set_exception(error)
            else:
                result.set_result(None)

        def notify(done: _ThreadFuture[None]) -> None:
            if done.exception() is not None:
                forward(done)
                return
            try:
                self._send_notify(engine, peer, payload).add_done_callback(forward)
            except RuntimeError as error:  # The notifier was shut down.
                result.set_exception(error)

        transfer.add_done_callback(notify)
        return result

    def poll_notifications(self) -> list[Notification]:
        with self._notify_cv:
            self._ensure_open()
            self._drain_notifies()
            received, self._received = self._received, []
        return received

    def wait_notifications(self, *, timeout: float | None = None) -> Work:
        _validate_timeout(timeout)
        with self._notify_cv:
            self._ensure_open()
            self._drain_notifies()
            if self._received:
                future: _ThreadFuture[None] = _ThreadFuture()
                future.set_result(None)
            else:
                if self._available is None:
                    self._available = _ThreadFuture()
                future = self._available
                if self._watcher is None:
                    self._watcher = Thread(
                        target=self._watch, name="mooncake-notify-watch", daemon=True
                    )
                    self._watcher.start()
        return _FutureWork(future, self._timeout if timeout is None else timeout)

    def _drain_notifies(self) -> None:
        # Caller holds _notify_cv.
        try:
            notifies = self._engine.get_notifies()
        except UnicodeDecodeError:
            # Mooncake's RDMA path raises on a non-UTF-8 notify from a sender
            # that is not a torch transport, losing the rest of this drain.
            logger.warning("dropped Mooncake notifies after a non-UTF-8 message")
            return
        for notify in notifies:
            try:
                payload = base64.b64decode(notify.msg, validate=True)
            except ValueError:
                continue  # Not sent by a torch transport.
            if payload:
                self._received.append(Notification(notify.name, payload))

    def _watch(self) -> None:
        delay = 1e-6
        available, error = None, None
        with self._notify_cv:
            try:
                while self._available is not None:
                    self._drain_notifies()
                    if self._received:
                        available, self._available = self._available, None
                        break
                    self._notify_cv.wait(delay)
                    delay = min(delay * 2, 1e-3)
            except Exception as failure:
                # Fail the pending waits; the next wait_notifications restarts
                # the watcher.
                available, self._available, error = self._available, None, failure
            finally:
                self._watcher = None
        # Outside the lock: completion runs Work callbacks, which may call back
        # into this transport.
        if available is not None:
            if error is None:
                available.set_result(None)
            else:
                available.set_exception(error)

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
            if registration.pinned:
                _pinned_registrations.append((engine, registration))
            elif engine.unregister_memory(registration.address) != 0:
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
        if remote_buffer.endpoint not in self._peers:
            raise ValueError("remote buffer does not belong to a connected peer")
        if local_buffer.size() > remote_buffer.length:
            raise ValueError("local view does not fit in the remote buffer")
        registration = local_buffer._memory._registration
        registration.ensure_active()
        if registration.pinned:
            raise RuntimeError(
                "memory is pinned after a timed-out transfer; Mooncake may still "
                "be accessing it"
            )
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
        notification: bytes | None,
    ) -> _MooncakeWork:
        engine = self._ensure_open()
        self._check_transfer(local_buffer, remote_buffer, mutable=mutable)
        if notification is not None:
            self._check_notification(notification)
            if local_buffer.size() == 0:
                raise ValueError("zero-byte transfers cannot carry notifications")
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
        batch_id = engine.batch_transfer_async(
            remote_buffer.endpoint,
            [registration.address + local_buffer._offset],
            [remote_buffer.address],
            [size],
            operation,
        )
        if batch_id == 0:
            raise RuntimeError("Mooncake transfer submission failed")
        transfer: _ThreadFuture[None] = _ThreadFuture()
        future = (
            transfer
            if notification is None
            else self._then_notify(
                transfer, engine, remote_buffer.endpoint, notification
            )
        )
        work = _MooncakeWork(self, local_buffer, remote_buffer, future, timeout)
        self._pending[id(work)] = work
        # Prune on completion so a settled (including failed) transfer is not
        # revisited by close(); fires on the poller thread.
        future.add_done_callback(lambda _: self._prune(id(work)))
        _live_transports.add(self)
        with self._poll_cv:
            self._batches[batch_id] = (transfer, registration)
            if self._poller is None:
                self._poller = Thread(target=self._poll, name="mooncake", daemon=True)
                self._poller.start()
            self._poll_cv.notify()
        return work

    def _poll(self) -> None:
        engine = self._engine
        delay = 0.0
        while True:
            with self._poll_cv:
                if delay:
                    self._poll_cv.wait(delay)
                while not self._batches and not self._poller_stopped:
                    self._poll_cv.wait()
                    delay = 0.0
                if self._poller_stopped:
                    return
                batches = list(self._batches.items())
            settled = []
            for batch_id, (future, registration) in batches:
                # Every non-zero status is final: Mooncake frees the batch, so it
                # must not be polled again. 1 and -1 mean every slice of the one
                # request has completed or failed. -2 (TIMEOUT, with
                # MC_SLICE_TIMEOUT) or an unknown status may leave slices in
                # flight, so the local memory stays pinned.
                status = engine.transfer_check_status(batch_id)
                if status != 0:
                    settled.append((batch_id, future, registration, status))
            with self._poll_cv:
                for batch_id, future, *_ in settled:
                    # A batch id is a BatchDesc address that Mooncake freed in
                    # the status check, so a concurrent submit may reuse it.
                    entry = self._batches.get(batch_id)
                    if entry is not None and entry[0] is future:
                        del self._batches[batch_id]
            for _, future, registration, status in settled:
                if status == 1:
                    future.set_result(None)
                    continue
                message = "Mooncake transfer failed"
                if status != -1:
                    registration.pinned = True
                    message += (
                        f" with status {status}; it may still be running, so its "
                        "local memory stays pinned"
                    )
                future.set_exception(RuntimeError(message))
            delay = 0.0 if settled else min(max(delay * 2, 1e-6), 1e-3)

    def _prune(self, key: int) -> None:
        with self._operation_lock:
            self._pending.pop(key, None)

    def _start(
        self,
        opcode: str,
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
                opcode,
                local_buffer,
                remote_buffer,
                mutable=mutable,
                timeout=timeout,
                notification=notification,
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
        opcode: str,
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
        # Offload the lock-guarded submit to a thread so a contended lock never
        # blocks the event loop; the poller settles the Work.
        work = await asyncio.to_thread(
            self._run_locked,
            lambda: self._submit(
                opcode,
                local_buffer,
                remote_buffer,
                mutable=mutable,
                timeout=timeout,
                notification=notification,
            ),
            max(0.0, deadline - time.monotonic()),
        )
        try:
            await wait_all([work], timeout=max(0.0, deadline - time.monotonic()))
        finally:
            # Mirror the sync wait() prune: a settled transfer (success or
            # failure) is observed here, so drop it before returning. A timed-out
            # or cancelled transfer is still running and stays pending.
            if work.is_completed():
                await asyncio.to_thread(self._prune, id(work))

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

    def _begin_close(self, timeout: float) -> list[_FutureWork]:
        self._closing = True
        with self._notify_cv:
            available, self._available = self._available, None
        if available is not None:
            available.set_exception(RuntimeError("transport is closed"))
        with self._locked(timeout):
            return [*self._pending.values(), *self._sends]

    def close(self, *, timeout: float | None = None) -> None:
        if self._closed:
            return
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
        if self._closed:
            return
        self._close_cuda_streams()
        timeout = self._timeout if timeout is None else timeout
        _validate_timeout(timeout)
        deadline = time.monotonic() + timeout
        self._closing = True
        pending = await asyncio.to_thread(
            self._begin_close, max(0.0, deadline - time.monotonic())
        )
        error: BaseException | None = None
        try:
            await wait_all(pending, timeout=max(0.0, deadline - time.monotonic()))
        except BaseException as failure:
            if isinstance(failure, asyncio.CancelledError) or any(
                not w.is_completed() for w in pending
            ):
                raise
            error = failure
        await asyncio.to_thread(
            self._run_locked,
            self._release_resources,
            max(0.0, deadline - time.monotonic()),
        )
        if error is not None:
            raise error

    def _release_resources(self) -> int:
        if self._closed:
            return 0
        with self._poll_cv:
            self._poller_stopped = True
            self._poll_cv.notify()
        self._notifier.shutdown(wait=False)
        engine = self._engine
        for key, registration in list(self._registrations.items()):
            if registration.pinned:
                _pinned_registrations.append((engine, registration))
            else:
                engine.unregister_memory(registration.address)
            registration.active = False
            del self._registrations[key]
        self._pending.clear()
        self._peers.clear()
        # Mooncake has no explicit shutdown
        self._engine = None
        self._closed = True
        _live_transports.discard(self)
        return 0
