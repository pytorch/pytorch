from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import timedelta
from importlib import import_module
from threading import RLock
from typing import Any, cast

import torch
from torch.distributed._transport import (
    Memory,
    MemoryView,
    MutableMemoryView,
    RemoteBuffer,
    Transport,
    wait_all,
)
from torch.distributed._transport._work import _validate_timeout

from ._memory import (
    _Endpoint,
    _Registration,
    MooncakeMemory,
    MooncakeMemoryView,
    MooncakeMutableMemoryView,
    MooncakeRemoteBuffer,
)
from ._work import _live_transports, MooncakeWork


class MooncakeTransport(Transport):
    """Direct Transfer Engine implementation of PyTorch's experimental Transport.

    Requires PyTorch 2.15 and Mooncake's managed ``submit_transfer`` binding.
    ``hostname`` must be reachable by peers (loopback is the local-test default).
    Native calls are synchronous; timeouts bound waiting, not native execution.
    Coordinate with peers before unregistering exposed memory or closing.
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        *,
        hostname: str = "127.0.0.1",
        protocol: str = "rdma",
        device_name: str = "",
        timeout: float = 30.0,
    ) -> None:
        super().__init__(device)
        if self.device is not None and self.device.type not in ("cpu", "cuda"):
            raise ValueError("Mooncake transport requires CPU or CUDA tensors")
        _validate_timeout(timeout)
        if timeout is None:
            raise ValueError("constructor timeout must be a finite number")
        TransferEngine = import_module("mooncake.engine").TransferEngine

        if not hasattr(TransferEngine, "submit_transfer"):
            raise RuntimeError(
                "rebuild Mooncake with the managed submit_transfer binding"
            )
        self._engine: Any = TransferEngine()
        if self._engine.initialize(hostname, "P2PHANDSHAKE", protocol, device_name):
            raise RuntimeError("Transfer Engine initialization failed")
        self._endpoint = _Endpoint(uuid.uuid4().hex, self._engine.get_endpoint())
        self._peers: set[_Endpoint] = set()
        self._generation = 0
        self._remote_generations: dict[_Endpoint, int] = {}
        self._default_timeout = timeout
        self._lock = RLock()
        self._registrations: dict[tuple[int, int, str], _Registration] = {}
        self._pending: dict[int, MooncakeWork] = {}
        self._closing = False
        self._closed = False

    @staticmethod
    def supported() -> bool:
        try:
            TransferEngine = import_module("mooncake.engine").TransferEngine

            return hasattr(TransferEngine, "submit_transfer")
        except (ImportError, OSError):
            return False

    def _seconds(self, timeout: float | None) -> float:
        _validate_timeout(timeout)
        seconds = self._default_timeout if timeout is None else timeout
        if seconds is None:
            raise ValueError("transport timeout must be a finite number")
        return seconds

    @contextmanager
    def _locked(self, timeout):
        if not self._lock.acquire(timeout=self._seconds(timeout)):
            raise TimeoutError("transport lock wait timed out")
        try:
            yield
        finally:
            self._lock.release()

    @asynccontextmanager
    async def _locked_async(self, deadline):
        while not self._lock.acquire(blocking=False):
            if time.monotonic() >= deadline:
                raise TimeoutError("transport lock wait timed out")
            await asyncio.sleep(0.001)
        try:
            yield
        finally:
            self._lock.release()

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("transport is closed")

    def bind(self, *, timeout: float | None = None) -> bytes:
        with self._locked(timeout):
            self._ensure_open()
            return self._endpoint.serialize()

    def connect(self, peer_url: bytes, *, timeout: float | None = None) -> int:
        with self._locked(timeout):
            self._ensure_open()
            peer = _Endpoint.deserialize(peer_url)
            if not peer.segment or not peer.session:
                raise ValueError("empty peer endpoint")
            # TE opens the remote segment on submission, including registrations
            # made after connect. No separate ProcessGroup or NIXL agent exists.
            self._peers.add(peer)
            return 0

    def connected(self) -> bool:
        return bool(self._peers) and not self._closing and not self._closed

    def register_memory(
        self, tensor: torch.Tensor, *, timeout: float | None = None
    ) -> MooncakeMemory:
        with self._locked(timeout):
            self._ensure_open()
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("tensor must be a torch.Tensor")
            self._check_device(tensor.device)
            if tensor.device.type not in ("cpu", "cuda"):
                raise ValueError("Mooncake transport requires CPU or CUDA tensors")
            if not tensor.is_contiguous() or not tensor.numel():
                raise ValueError("register a nonempty contiguous tensor")
            address, length = tensor.data_ptr(), tensor.numel() * tensor.element_size()
            key = address, length, str(tensor.device)
            if key in self._registrations:
                return MooncakeMemory(self, self._registrations[key])
            for registration in self._registrations.values():
                if address < registration.address + registration.length and (
                    registration.address < address + length
                ):
                    raise ValueError(
                        "overlapping registration; use registered memory views"
                    )
            location = str(tensor.device) if tensor.is_cuda else "*"
            if self._engine.register_memory(address, length, location):
                raise RuntimeError("Transfer Engine memory registration failed")
            registration = _Registration(
                tensor, tensor.untyped_storage(), address, length
            )
            self._registrations[key] = registration
            self._generation += 1
            _live_transports.add(self)
            return MooncakeMemory(self, registration)

    def unregister_memory(
        self, memory: Memory, *, timeout: float | None = None
    ) -> None:
        with self._locked(timeout):
            self._ensure_open()
            if not isinstance(memory, MooncakeMemory) or memory._transport is not self:
                raise TypeError("memory was not registered by this transport")
            registration = memory._registration
            if not registration.active:
                return
            views = self._cuda_stream_views() + [
                w._buffers[0] for w in self._pending.values()
            ]
            if any(
                cast(MooncakeMemoryView, view)._memory._registration is registration
                for view in views
            ):
                raise RuntimeError("memory is retained by pending work or CUDA graphs")
            self._unregister(registration)

    def _unregister(self, registration: _Registration) -> None:
        if self._engine.unregister_memory(registration.address):
            raise RuntimeError("Transfer Engine memory unregistration failed")
        registration.active = False
        self._generation += 1
        key = next(
            k for k, value in self._registrations.items() if value is registration
        )
        del self._registrations[key]
        if not self._registrations and not self._pending:
            _live_transports.discard(self)

    def _check_transfer(self, local_buffer, remote_buffer, *, mutable):
        self._ensure_open()
        kind = MooncakeMutableMemoryView if mutable else MooncakeMemoryView
        if (
            not isinstance(local_buffer, kind)
            or local_buffer._memory._transport is not self
        ):
            raise TypeError("local view has the wrong owner or is not writable")
        if not isinstance(remote_buffer, MooncakeRemoteBuffer):
            raise TypeError("expected a MooncakeRemoteBuffer")
        remote_buffer.serialize()  # Validate directly constructed descriptors too.
        if _Endpoint(remote_buffer.session, remote_buffer.segment) not in self._peers:
            raise ValueError("remote buffer does not belong to the connected peer")
        if remote_buffer.address == 0 or local_buffer.size() > remote_buffer.length:
            raise ValueError("invalid remote address or insufficient remote capacity")
        registration = local_buffer._memory._registration
        registration.check()
        if not 0 <= local_buffer._offset <= registration.length or not (
            0 <= local_buffer.size() <= registration.length - local_buffer._offset
        ):
            raise ValueError("local view exceeds registered memory")

    def _start(self, local, remote, read, async_op, timeout):
        seconds = self._seconds(timeout)
        deadline = time.monotonic() + seconds
        with self._locked(seconds):
            self._check_transfer(local, remote, mutable=read)
            work = MooncakeWork(self, local, remote, seconds)
            if local.size():
                # Classic TE caches segment registrations. Refresh once when a
                # peer exports a newer registration generation, not per transfer.
                peer = _Endpoint(remote.session, remote.segment)
                if remote.generation > self._remote_generations.get(peer, -1):
                    self._engine.refresh_remote(remote.segment)
                    self._remote_generations[peer] = remote.generation
                self._pending[id(work)] = work
                try:
                    opcode = (
                        self._engine.TransferOpcode.Read
                        if read
                        else self._engine.TransferOpcode.Write
                    )
                    work._operation = self._engine.submit_transfer(
                        remote.segment,
                        local._memory._registration.address + local._offset,
                        remote.address,
                        local.size(),
                        opcode,
                    )
                except BaseException:
                    # submit_transfer owns a draining RAII operation: unwinding
                    # cannot return while published DMA still references buffers.
                    self._pending.pop(id(work))
                    raise
            else:
                work._done = True
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
    ) -> int | MooncakeWork:
        return self._start(local_buffer, remote_buffer, False, async_op, timeout)

    def read(
        self,
        local_buffer: MutableMemoryView,
        remote_buffer: RemoteBuffer,
        *,
        async_op: bool = False,
        timeout: float | None = None,
    ) -> int | MooncakeWork:
        return self._start(local_buffer, remote_buffer, True, async_op, timeout)

    async def _transfer_async(self, local, remote, read, timeout):
        deadline = time.monotonic() + self._seconds(timeout)
        async with self._locked_async(deadline):
            work = self._start(
                local, remote, read, True, max(0.0, deadline - time.monotonic())
            )
        await wait_all([work], timeout=max(0.0, deadline - time.monotonic()))

    async def write_async(
        self,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        timeout: float | None = None,
    ) -> None:
        await self._transfer_async(local_buffer, remote_buffer, False, timeout)

    async def read_async(
        self,
        local_buffer: MutableMemoryView,
        remote_buffer: RemoteBuffer,
        *,
        timeout: float | None = None,
    ) -> None:
        await self._transfer_async(local_buffer, remote_buffer, True, timeout)

    def _release_resources(self) -> None:
        for registration in list(self._registrations.values()):
            self._unregister(registration)
        self._engine = None
        self._peers.clear()
        self._remote_generations.clear()
        self._closed = True
        _live_transports.discard(self)

    def close(self, *, timeout: float | None = None) -> None:
        seconds = self._seconds(timeout)
        self._close_cuda_streams()
        deadline = time.monotonic() + seconds
        self._closing = True
        with self._locked(seconds):
            pending = list(self._pending.values())
        error = None
        for work in pending:
            try:
                remaining = max(0.000001, deadline - time.monotonic())
                work.wait(timedelta(seconds=remaining))
            except Exception as failure:
                if not work.is_completed():
                    raise
                error = error or failure
        with self._locked(max(0.0, deadline - time.monotonic())):
            self._release_resources()
        if error is not None:
            raise error

    async def close_async(self, *, timeout: float | None = None) -> None:
        seconds = self._seconds(timeout)
        # Match PyTorch's NIXL lifecycle: drain queued stream callbacks before
        # latching shutdown. They progress on PyTorch's separate background loop.
        # This native CUDA synchronization is not interruptible by asyncio.
        self._close_cuda_streams()
        deadline = time.monotonic() + seconds
        self._closing = True
        async with self._locked_async(deadline):
            pending = list(self._pending.values())
        error = None
        try:
            await wait_all(pending, timeout=max(0.0, deadline - time.monotonic()))
        except Exception as failure:
            if any(not work.is_completed() for work in pending):
                raise
            error = failure
        async with self._locked_async(deadline):
            self._release_resources()
        if error is not None:
            raise error
