from __future__ import annotations

import asyncio
import os
import socket
import time
from concurrent.futures import Future as _ThreadFuture
from contextlib import contextmanager
from datetime import timedelta
from importlib import import_module
from threading import Condition, RLock, Thread
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


_pinned_registrations: list[tuple[Any, _Registration]] = []


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
        batch_id = engine.batch_transfer_async(
            remote_buffer.endpoint,
            [registration.address + local_buffer._offset],
            [remote_buffer.address],
            [size],
            operation,
        )
        if batch_id == 0:
            raise RuntimeError("Mooncake transfer submission failed")
        future: _ThreadFuture[None] = _ThreadFuture()
        work = _MooncakeWork(self, local_buffer, remote_buffer, future, timeout)
        self._pending[id(work)] = work
        # Prune on completion so a settled (including failed) transfer is not
        # revisited by close(); fires on the poller thread.
        future.add_done_callback(lambda _: self._prune(id(work)))
        _live_transports.add(self)
        with self._poll_cv:
            self._batches[batch_id] = (future, registration)
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
                for batch_id, *_ in settled:
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
        # Offload the lock-guarded submit to a thread so a contended lock never
        # blocks the event loop; the poller settles the Work.
        work = await asyncio.to_thread(
            self._run_locked,
            lambda: self._submit(
                opcode, local_buffer, remote_buffer, mutable=mutable, timeout=timeout
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
