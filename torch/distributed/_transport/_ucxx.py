from __future__ import annotations

import asyncio
import json
import os
import secrets
import struct
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from importlib import import_module
from threading import current_thread, Event, Thread
from typing import Any, cast, TYPE_CHECKING, TypeVar

import torch

from ._api import MemoryView, MutableMemoryView, RemoteBuffer, Transport


if TYPE_CHECKING:
    from collections.abc import Coroutine


_T = TypeVar("_T")
_HEADER = struct.Struct("!BQQQQQ")
_WRITE = 1
_READ = 2
_READY = 3
_DONE = 4
_DATA = 5
_ERROR = 6
_MAX_ERROR_SIZE = 4096


def _load_backend() -> Any:
    try:
        return import_module("ucxx")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "UCXX transport requires the ucxx-cu12 or ucxx-cu13 package"
        ) from error


@dataclass(frozen=True)
class UCXXRemoteBuffer:
    """A serializable descriptor for memory registered by a UCXX peer."""

    buffer_id: int
    length: int
    access_key: int


@dataclass
class _RegisteredMemory:
    tensor: torch.Tensor
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class UCXXMemoryView:
    """A byte range in UCXX-registered memory."""

    def __init__(self, memory: UCXXMemory, offset: int, length: int) -> None:
        self._memory = memory
        self._offset = offset
        self._length = length

    def size(self) -> int:
        return self._length


class UCXXMutableMemoryView(UCXXMemoryView):
    """A writable byte range in UCXX-registered memory."""


class UCXXMemory:
    """A tensor registered with a UCXX transport."""

    def __init__(
        self,
        transport: UCXXTransport,
        tensor: torch.Tensor,
        remote: UCXXRemoteBuffer,
        registered: _RegisteredMemory,
        reused: bool,
    ) -> None:
        self._transport = transport
        self._tensor = tensor
        self._remote = remote
        self._registered = registered
        self._reused = reused

    def _range(self, offset: int | None, length: int | None) -> tuple[int, int]:
        offset = 0 if offset is None else offset
        length = self._remote.length - offset if length is None else length
        if offset < 0 or length < 0 or offset + length > self._remote.length:
            raise ValueError("memory view is outside the registered tensor")
        return offset, length

    def to_view(
        self, offset: int | None = None, length: int | None = None
    ) -> UCXXMemoryView:
        offset, length = self._range(offset, length)
        return UCXXMemoryView(self, offset, length)

    def to_mutable_view(
        self, offset: int | None = None, length: int | None = None
    ) -> UCXXMutableMemoryView:
        offset, length = self._range(offset, length)
        return UCXXMutableMemoryView(self, offset, length)

    def to_remote_buffer(self) -> UCXXRemoteBuffer:
        return self._remote

    def reused_registration(self) -> bool:
        return self._reused


class _CudaBuffer:
    def __init__(
        self, tensor: torch.Tensor, offset: int, length: int, *, readonly: bool
    ) -> None:
        self._tensor = tensor
        self.__cuda_array_interface__ = {
            "shape": (length,),
            "strides": None,
            "typestr": "|u1",
            "data": (tensor.data_ptr() + offset, readonly),
            "version": 3,
        }


class UCXXTransport(Transport):
    """One-sided transport emulated with UCXX tagged messages."""

    def __init__(
        self,
        device: torch.device | str,
        *,
        host: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("UCXX transport requires a CPU or CUDA device")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._backend = _load_backend()
        self._host = host or os.environ.get(
            "TORCH_DISTRIBUTED_TRANSPORT_UCXX_HOST", "127.0.0.1"
        )
        self._timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._started = Event()
        self._startup_error: BaseException | None = None
        self._failure: BaseException | None = None
        self._listener: Any = None
        self._url: bytes | None = None
        self._outgoing: Any = None
        self._incoming: list[Any] = []
        self._registered: dict[tuple[int, int], _RegisteredMemory] = {}
        self._registrations: dict[tuple[int, int, str], UCXXRemoteBuffer] = {}
        self._next_request_id = 1
        self._operation_lock = asyncio.Lock()
        self._state_lock = threading.Lock()
        self._closed = False

    @staticmethod
    def supported() -> bool:
        try:
            backend = _load_backend()
            return all(
                hasattr(backend, name)
                for name in ("create_endpoint", "create_listener", "get_address")
            )
        except Exception:
            return False

    def _thread_main(self) -> None:
        try:
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
        except BaseException as error:
            self._startup_error = error
            self._started.set()
            return
        self._started.set()
        try:
            loop.run_forever()
        finally:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()
            if tasks:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.close()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        start_thread = False
        with self._state_lock:
            if self._closed:
                raise RuntimeError("transport is closed")
            if self._failure is not None:
                raise RuntimeError("UCXX transport failed") from self._failure
            if self._thread is None:
                self._thread = Thread(
                    target=self._thread_main,
                    name="torch-transport-ucxx",
                    daemon=True,
                )
                start_thread = True
            thread = self._thread
        if thread is None:
            raise RuntimeError("failed to create the UCXX event loop thread")
        if start_thread:
            thread.start()
        if not self._started.is_set():
            self._started.wait()
        if self._startup_error is not None:
            raise RuntimeError(
                "failed to start the UCXX event loop"
            ) from self._startup_error
        if self._loop is None:
            raise RuntimeError("failed to start the UCXX event loop")
        return self._loop

    def _run(self, coroutine: Coroutine[Any, Any, _T], operation: str) -> _T:
        try:
            loop = self._ensure_loop()
        except BaseException:
            coroutine.close()
            raise
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=self._timeout)
        except FutureTimeoutError as error:
            future.cancel()
            timeout = TimeoutError(f"UCXX {operation} timed out")
            with self._state_lock:
                self._failure = timeout
            asyncio.run_coroutine_threadsafe(self._close(), loop)
            raise timeout from error

    async def _accept(self, endpoint: Any) -> None:
        with self._state_lock:
            closed = self._closed
            if not closed:
                self._incoming.append(endpoint)
        if closed:
            await endpoint.close()
            return
        try:
            await self._serve(endpoint)
        except BaseException:
            with self._state_lock:
                closed = self._closed
            if not closed and not endpoint.closed:
                await endpoint.close()
        finally:
            with self._state_lock:
                if endpoint in self._incoming:
                    self._incoming.remove(endpoint)

    async def _bind(self) -> bytes:
        self._listener = self._backend.create_listener(self._accept)
        address = self._host
        return json.dumps(
            {"address": address, "port": self._listener.port}, separators=(",", ":")
        ).encode()

    def bind(self) -> bytes:
        if self._closed:
            raise RuntimeError("transport is closed")
        if self._url is not None:
            return self._url
        url = self._run(self._bind(), "bind")
        self._url = url
        return url

    @staticmethod
    def _parse_url(peer_url: bytes) -> tuple[str, int]:
        try:
            value = json.loads(peer_url)
            address = value["address"]
            port = value["port"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid UCXX transport URL") from error
        if not isinstance(address, str) or not isinstance(port, int):
            raise ValueError("invalid UCXX transport URL")
        if not 0 < port < 65536:
            raise ValueError("invalid UCXX transport URL")
        return address, port

    async def _connect(self, address: str, port: int) -> None:
        with self._state_lock:
            if self._outgoing is not None:
                raise RuntimeError("transport is already connected")
        endpoint = await self._backend.create_endpoint(address, port)
        with self._state_lock:
            if self._closed or self._outgoing is not None:
                close_endpoint = True
            else:
                self._outgoing = endpoint
                close_endpoint = False
        if close_endpoint:
            await endpoint.close()
            raise RuntimeError("transport closed while connecting")

    def connect(self, peer_url: bytes) -> int:
        address, port = self._parse_url(peer_url)
        self._run(self._connect(address, port), "connect")
        return 0

    def connected(self) -> bool:
        with self._state_lock:
            if self._closed or self._failure is not None:
                return False
            endpoints = [self._outgoing, *self._incoming]
        return any(
            endpoint is not None and not endpoint.closed for endpoint in endpoints
        )

    def register_memory(self, tensor: torch.Tensor) -> UCXXMemory:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        if not tensor.is_contiguous():
            raise ValueError("UCXX transport requires a contiguous tensor")
        if tensor.device.type != self.device.type or (
            self.device.index is not None and tensor.device.index != self.device.index
        ):
            raise ValueError(f"expected a tensor on {self.device}, got {tensor.device}")
        length = tensor.numel() * tensor.element_size()
        registration_key = (tensor.data_ptr(), length, str(tensor.device))
        with self._state_lock:
            if self._closed:
                raise RuntimeError("transport is closed")
            remote = self._registrations.get(registration_key)
            reused = remote is not None
            if remote is None:
                while True:
                    remote = UCXXRemoteBuffer(
                        secrets.randbits(64), length, secrets.randbits(64)
                    )
                    key = (remote.buffer_id, remote.access_key)
                    if key not in self._registered:
                        break
                registered = _RegisteredMemory(tensor)
                self._registrations[registration_key] = remote
                self._registered[key] = registered
            else:
                registered = self._registered[(remote.buffer_id, remote.access_key)]
        return UCXXMemory(self, tensor, remote, registered, reused)

    def write(self, local_buffer: MemoryView, remote_buffer: RemoteBuffer) -> int:
        local = self._local_view(local_buffer, mutable=False)
        remote = self._remote_buffer(remote_buffer)
        if local.size() > remote.length:
            raise ValueError("local view does not fit in the remote buffer")
        self._run(self._write(local, remote), "write")
        return 0

    async def _write(self, local: UCXXMemoryView, remote: UCXXRemoteBuffer) -> None:
        async with self._operation_lock:
            endpoint = self._connection()
            request_id = self._new_request_id()
            await self._send_header(
                endpoint,
                _WRITE,
                request_id,
                remote.buffer_id,
                remote.access_key,
                0,
                local.size(),
            )
            await self._receive_response(endpoint, request_id, _READY)
            async with local._memory._registered.lock:
                await endpoint.send(self._view_buffer(local, readonly=True))
            await self._receive_response(endpoint, request_id, _DONE)

    def read(self, local_buffer: MutableMemoryView, remote_buffer: RemoteBuffer) -> int:
        local = cast(
            UCXXMutableMemoryView, self._local_view(local_buffer, mutable=True)
        )
        remote = self._remote_buffer(remote_buffer)
        if local.size() > remote.length:
            raise ValueError("local view does not fit in the remote buffer")
        self._run(self._read(local, remote), "read")
        return 0

    async def _read(
        self, local: UCXXMutableMemoryView, remote: UCXXRemoteBuffer
    ) -> None:
        async with self._operation_lock:
            endpoint = self._connection()
            request_id = self._new_request_id()
            await self._send_header(
                endpoint,
                _READ,
                request_id,
                remote.buffer_id,
                remote.access_key,
                0,
                local.size(),
            )
            await self._receive_response(endpoint, request_id, _DATA)
            async with local._memory._registered.lock:
                await endpoint.recv(self._view_buffer(local, readonly=False))

    async def _serve(self, endpoint: Any) -> None:
        while not endpoint.closed:
            with self._state_lock:
                if self._closed:
                    return
            (
                opcode,
                request_id,
                buffer_id,
                access_key,
                offset,
                length,
            ) = await self._receive_header(endpoint)
            try:
                if opcode == _WRITE:
                    await self._serve_write(
                        endpoint,
                        request_id,
                        buffer_id,
                        access_key,
                        offset,
                        length,
                    )
                elif opcode == _READ:
                    await self._serve_read(
                        endpoint,
                        request_id,
                        buffer_id,
                        access_key,
                        offset,
                        length,
                    )
                else:
                    raise RuntimeError(f"unknown UCXX transport opcode {opcode}")
            except Exception as error:
                await self._send_error(endpoint, request_id, str(error))

    async def _serve_write(
        self,
        endpoint: Any,
        request_id: int,
        buffer_id: int,
        access_key: int,
        offset: int,
        length: int,
    ) -> None:
        registered = self._lookup_memory(buffer_id, access_key, offset, length)
        async with registered.lock:
            buffer = self._tensor_buffer(
                registered.tensor, offset, length, readonly=False
            )
            await self._send_header(endpoint, _READY, request_id)
            await endpoint.recv(buffer)
        await self._send_header(endpoint, _DONE, request_id)

    async def _serve_read(
        self,
        endpoint: Any,
        request_id: int,
        buffer_id: int,
        access_key: int,
        offset: int,
        length: int,
    ) -> None:
        registered = self._lookup_memory(buffer_id, access_key, offset, length)
        async with registered.lock:
            buffer = self._tensor_buffer(
                registered.tensor, offset, length, readonly=True
            )
            await self._send_header(endpoint, _DATA, request_id)
            await endpoint.send(buffer)

    async def _send_header(
        self,
        endpoint: Any,
        opcode: int,
        request_id: int,
        buffer_id: int = 0,
        access_key: int = 0,
        offset: int = 0,
        length: int = 0,
    ) -> None:
        await endpoint.send(
            _HEADER.pack(opcode, request_id, buffer_id, access_key, offset, length)
        )

    @staticmethod
    async def _receive_header(endpoint: Any) -> tuple[int, int, int, int, int, int]:
        header = bytearray(_HEADER.size)
        await endpoint.recv(header)
        return _HEADER.unpack(header)

    async def _receive_response(
        self, endpoint: Any, request_id: int, expected: int
    ) -> None:
        response = await self._receive_header(endpoint)
        opcode, response_id, _, _, _, length = response
        if response_id != request_id:
            raise RuntimeError(f"unexpected UCXX request {response_id}")
        if opcode == _ERROR:
            if length > _MAX_ERROR_SIZE:
                raise RuntimeError("invalid UCXX error response")
            message = bytearray(length)
            if length:
                await endpoint.recv(message)
            raise RuntimeError(f"UCXX peer: {message.decode('utf-8', 'replace')}")
        if opcode != expected:
            raise RuntimeError(f"unexpected UCXX transport opcode {opcode}")

    async def _send_error(self, endpoint: Any, request_id: int, message: str) -> None:
        payload = message.encode("utf-8")[:_MAX_ERROR_SIZE]
        await self._send_header(endpoint, _ERROR, request_id, length=len(payload))
        if payload:
            await endpoint.send(payload)

    def _new_request_id(self) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    def _connection(self) -> Any:
        with self._state_lock:
            endpoint = self._outgoing
            closed = self._closed
        if closed:
            raise RuntimeError("transport is closed")
        if endpoint is None or endpoint.closed:
            raise RuntimeError("UCXX transport has no outgoing connection")
        return endpoint

    def _lookup_memory(
        self,
        buffer_id: int,
        access_key: int,
        offset: int,
        length: int,
    ) -> _RegisteredMemory:
        with self._state_lock:
            registered = self._registered.get((buffer_id, access_key))
        if registered is None:
            raise ValueError("unknown remote buffer")
        registered_length = registered.tensor.numel() * registered.tensor.element_size()
        if offset > registered_length or length > registered_length - offset:
            raise ValueError("operation exceeds the registered buffer")
        return registered

    def _local_view(
        self, view: MemoryView | MutableMemoryView, *, mutable: bool
    ) -> UCXXMemoryView | UCXXMutableMemoryView:
        expected_type = UCXXMutableMemoryView if mutable else UCXXMemoryView
        if not isinstance(view, expected_type) or view._memory._transport is not self:
            raise TypeError("memory view belongs to another transport")
        return view

    @staticmethod
    def _remote_buffer(remote: RemoteBuffer) -> UCXXRemoteBuffer:
        if not isinstance(remote, UCXXRemoteBuffer):
            raise TypeError("remote buffer is not a UCXX descriptor")
        return remote

    def _view_buffer(self, view: UCXXMemoryView, *, readonly: bool) -> Any:
        return self._tensor_buffer(
            view._memory._tensor, view._offset, view._length, readonly=readonly
        )

    @staticmethod
    def _tensor_buffer(
        tensor: torch.Tensor, offset: int, length: int, *, readonly: bool
    ) -> Any:
        if tensor.device.type == "cuda":
            return _CudaBuffer(tensor, offset, length, readonly=readonly)
        byte_tensor = tensor.detach().reshape(-1).view(torch.uint8)
        return memoryview(byte_tensor.numpy())[offset : offset + length]

    async def _close(self) -> None:
        with self._state_lock:
            endpoints = [self._outgoing, *self._incoming]
            self._outgoing = None
            self._incoming.clear()
            listener = self._listener
            self._listener = None
        for endpoint in endpoints:
            if endpoint is not None and not endpoint.closed:
                await endpoint.close()
        await asyncio.sleep(0)
        if listener is not None:
            listener.close()

    def close(self) -> None:
        with self._state_lock:
            thread = self._thread
            if thread is not None and current_thread() is thread:
                raise RuntimeError("cannot close UCXX transport from its event loop")
            if self._closed:
                return
            loop = self._loop
            self._closed = True
            self._registered.clear()
            self._registrations.clear()
        if loop is None or thread is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._close(), loop)
        error: BaseException | None = None
        try:
            future.result(timeout=self._timeout)
        except FutureTimeoutError:
            future.cancel()
            error = TimeoutError("UCXX close timed out")
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=self._timeout)
        if thread.is_alive():
            raise TimeoutError("UCXX event loop did not stop")
        if error is not None:
            raise error


__all__ = [
    "UCXXMemory",
    "UCXXMemoryView",
    "UCXXMutableMemoryView",
    "UCXXRemoteBuffer",
    "UCXXTransport",
]
