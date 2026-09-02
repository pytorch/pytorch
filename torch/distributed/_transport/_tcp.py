from __future__ import annotations

import os
import queue
import secrets
import socket
import struct
import threading
from dataclasses import dataclass, field
from typing import cast
from urllib.parse import parse_qs, urlencode, urlsplit

import torch

from ._api import MemoryView, MutableMemoryView, RemoteBuffer, Transport


_MAGIC = b"PTTCP001"
_HANDSHAKE = struct.Struct("!8s16s16sII")
_FRAME_LENGTH = struct.Struct("!I")
_FRAME_HEADER = struct.Struct("!BQQQQQQ")
_WRITE = 1
_WRITE_ACK = 2
_READ = 3
_READ_DATA = 4
_ERROR = 5
_CLOSE = 6
_DEFAULT_FLOWS = 16
_DEFAULT_CHUNK_SIZE = 4 << 20


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _recv_exact(sock: socket.socket, length: int) -> bytearray:
    data = bytearray(length)
    _recv_into(sock, memoryview(data))
    return data


def _recv_into(sock: socket.socket, view: memoryview) -> None:
    offset = 0
    while offset < len(view):
        received = sock.recv_into(view[offset:])
        if received == 0:
            raise ConnectionError("TCP peer closed the connection")
        offset += received


def _segments(length: int, flows: int, chunk_size: int):
    if length == 0:
        yield 0, 0, 0
        return
    for flow in range(flows):
        start = length * flow // flows
        end = length * (flow + 1) // flows
        for offset in range(start, end, chunk_size):
            yield flow, offset, min(end, offset + chunk_size) - offset


@dataclass(frozen=True)
class TCPRemoteBuffer:
    """A serializable descriptor for memory registered by a TCP peer."""

    buffer_id: int
    length: int
    access_key: int


@dataclass
class _RegisteredMemory:
    tensor: torch.Tensor
    lock: threading.Lock = field(default_factory=threading.Lock)


class TCPMemoryView:
    """A byte range in TCP-registered memory."""

    def __init__(self, memory: TCPMemory, offset: int, length: int) -> None:
        self._memory = memory
        self._offset = offset
        self._length = length

    def size(self) -> int:
        return self._length


class TCPMutableMemoryView(TCPMemoryView):
    """A writable byte range in TCP-registered memory."""


class TCPMemory:
    """A tensor registered with a TCP transport."""

    def __init__(
        self,
        transport: TCPTransport,
        tensor: torch.Tensor,
        remote: TCPRemoteBuffer,
        reused: bool,
    ) -> None:
        self._transport = transport
        self._tensor = tensor
        self._remote = remote
        self._reused = reused

    def _range(self, offset: int | None, length: int | None) -> tuple[int, int]:
        offset = 0 if offset is None else offset
        length = self._remote.length - offset if length is None else length
        if offset < 0 or length < 0 or offset + length > self._remote.length:
            raise ValueError("memory view is outside the registered tensor")
        return offset, length

    def to_view(
        self, offset: int | None = None, length: int | None = None
    ) -> TCPMemoryView:
        offset, length = self._range(offset, length)
        return TCPMemoryView(self, offset, length)

    def to_mutable_view(
        self, offset: int | None = None, length: int | None = None
    ) -> TCPMutableMemoryView:
        offset, length = self._range(offset, length)
        return TCPMutableMemoryView(self, offset, length)

    def to_remote_buffer(self) -> TCPRemoteBuffer:
        return self._remote

    def reused_registration(self) -> bool:
        return self._reused


@dataclass
class _Packet:
    header: bytes
    payload: memoryview | bytes = b""


@dataclass
class _Flow:
    sock: socket.socket
    outgoing: queue.Queue[_Packet | None] = field(default_factory=queue.Queue)


@dataclass
class _Completion:
    mutable: TCPMutableMemoryView | None = None
    expected: int = 0
    received: int = 0
    error: BaseException | None = None
    event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _IncomingWrite:
    buffer_id: int
    access_key: int
    remote_offset: int
    expected: int
    received: int = 0


class TCPTransport(Transport):
    """A striped, asynchronous-I/O TCP transport."""

    def __init__(
        self,
        device: torch.device | str,
        *,
        num_flows: int | None = None,
        host: str | None = None,
        chunk_size: int | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(device)
        self._num_flows = (
            _env_int("TORCH_DISTRIBUTED_TRANSPORT_TCP_FLOWS", _DEFAULT_FLOWS)
            if num_flows is None
            else num_flows
        )
        self._chunk_size = (
            _env_int("TORCH_DISTRIBUTED_TRANSPORT_TCP_CHUNK_SIZE", _DEFAULT_CHUNK_SIZE)
            if chunk_size is None
            else chunk_size
        )
        if self._num_flows < 1 or self._chunk_size < 1 or timeout <= 0:
            raise ValueError("num_flows, chunk_size, and timeout must be positive")
        self._timeout = timeout
        self._bind_host = host or os.environ.get(
            "TORCH_DISTRIBUTED_TRANSPORT_TCP_HOST", "127.0.0.1"
        )
        if self._bind_host in ("0.0.0.0", "::"):
            raise ValueError("host must be an address peers can connect to")
        self._endpoint_id = secrets.token_bytes(16)
        self._listener: socket.socket | None = None
        self._url: bytes | None = None
        self._flows: list[_Flow] = []
        self._active_flows: list[_Flow] = []
        self._passive_flows: list[_Flow] = []
        self._pending_sockets: dict[bytes, dict[int, socket.socket]] = {}
        self._registered: dict[tuple[int, int], _RegisteredMemory] = {}
        self._registrations: dict[tuple[int, int, str], TCPRemoteBuffer] = {}
        self._completions: dict[int, _Completion] = {}
        self._expired_requests: set[int] = set()
        self._incoming_writes: dict[int, _IncomingWrite] = {}
        self._rejected_writes: set[int] = set()
        self._next_request_id = 1
        self._state_lock = threading.Lock()
        self._connected = threading.Event()
        self._closed = threading.Event()

    @staticmethod
    def supported() -> bool:
        return True

    def bind(self) -> bytes:
        with self._state_lock:
            if self._closed.is_set():
                raise RuntimeError("transport is closed")
            if self._url is not None:
                return self._url
            family = socket.AF_INET6 if ":" in self._bind_host else socket.AF_INET
            listener = socket.socket(family, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self._bind_host, 0))
            listener.listen(self._num_flows)
            self._listener = listener
            advertise_host = self._bind_host
            if ":" in advertise_host:
                advertise_host = f"[{advertise_host}]"
            query = urlencode(
                {"flows": self._num_flows, "token": self._endpoint_id.hex()}
            )
            port = listener.getsockname()[1]
            self._url = f"tcp://{advertise_host}:{port}?{query}".encode()
            url = self._url
        threading.Thread(
            target=self._accept_loop, name="tcp-transport-accept", daemon=True
        ).start()
        return url

    def connect(self, peer_url: bytes) -> int:
        host, port, flow_count, remote_id = self._parse_url(peer_url)
        with self._state_lock:
            if self._closed.is_set():
                raise RuntimeError("transport is closed")
            if self._active_flows:
                raise RuntimeError("transport is already connected")
        sockets: list[socket.socket] = []
        try:
            for index in range(flow_count):
                sock = socket.create_connection((host, port), timeout=self._timeout)
                self._configure_socket(sock)
                handshake = _HANDSHAKE.pack(
                    _MAGIC, remote_id, self._endpoint_id, index, flow_count
                )
                sock.sendall(handshake)
                sock.settimeout(None)
                sockets.append(sock)
            self._install_connection(sockets, active=True)
        except BaseException:
            for sock in sockets:
                sock.close()
            raise
        return 0

    def connected(self) -> bool:
        return self._connected.is_set() and not self._closed.is_set()

    def register_memory(self, tensor: torch.Tensor) -> TCPMemory:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        if not tensor.is_contiguous():
            raise ValueError("TCP transport requires a contiguous tensor")
        if tensor.device.type != self.device.type or (
            self.device.index is not None and tensor.device.index != self.device.index
        ):
            raise ValueError(f"expected a tensor on {self.device}, got {tensor.device}")
        length = tensor.numel() * tensor.element_size()
        registration_key = (tensor.data_ptr(), length, str(tensor.device))
        with self._state_lock:
            if self._closed.is_set():
                raise RuntimeError("transport is closed")
            remote = self._registrations.get(registration_key)
            reused = remote is not None
            if remote is None:
                while True:
                    remote = TCPRemoteBuffer(
                        secrets.randbits(64), length, secrets.randbits(64)
                    )
                    key = (remote.buffer_id, remote.access_key)
                    if key not in self._registered:
                        break
                self._registrations[registration_key] = remote
                self._registered[key] = _RegisteredMemory(tensor)
        return TCPMemory(self, tensor, remote, reused)

    def write(self, local_buffer: MemoryView, remote_buffer: RemoteBuffer) -> int:
        local = self._local_view(local_buffer, mutable=False)
        remote = self._remote_buffer(remote_buffer)
        if local.size() > remote.length:
            raise ValueError("local view does not fit in the remote buffer")
        request_id, completion = self._new_completion()
        try:
            flows = self._connection()
            for flow_index, offset, length in _segments(
                local.size(), len(flows), self._chunk_size
            ):
                payload = self._view_bytes(local, offset, length)
                self._queue_frame(
                    flows[flow_index],
                    _WRITE,
                    request_id,
                    remote.buffer_id,
                    remote.access_key,
                    0,
                    local.size(),
                    offset,
                    payload,
                )
            self._wait(request_id, completion)
        except BaseException:
            self._discard_completion(request_id)
            raise
        return 0

    def read(self, local_buffer: MutableMemoryView, remote_buffer: RemoteBuffer) -> int:
        local = cast(TCPMutableMemoryView, self._local_view(local_buffer, mutable=True))
        remote = self._remote_buffer(remote_buffer)
        if local.size() > remote.length:
            raise ValueError("local view does not fit in the remote buffer")
        request_id, completion = self._new_completion(local, local.size())
        try:
            flow = self._connection()[0]
            self._queue_frame(
                flow,
                _READ,
                request_id,
                remote.buffer_id,
                remote.access_key,
                0,
                local.size(),
                0,
            )
            self._wait(request_id, completion)
        except BaseException:
            self._discard_completion(request_id)
            raise
        return 0

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        with self._state_lock:
            listener = self._listener
            self._listener = None
            flows = self._active_flows + self._passive_flows
            self._flows = []
            self._active_flows = []
            self._passive_flows = []
            pending = [
                sock
                for group in self._pending_sockets.values()
                for sock in group.values()
            ]
            self._pending_sockets.clear()
            completions = list(self._completions.values())
            self._completions.clear()
            self._incoming_writes.clear()
            self._rejected_writes.clear()
            self._registered.clear()
            self._registrations.clear()
        if listener is not None:
            listener.close()
        for completion in completions:
            completion.error = ConnectionError("TCP transport closed")
            completion.event.set()
        for flow in flows:
            flow.outgoing.put(None)
            try:
                flow.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            flow.sock.close()
        for sock in pending:
            sock.close()
        self._connected.clear()

    def _parse_url(self, peer_url: bytes) -> tuple[str, int, int, bytes]:
        try:
            parsed = urlsplit(peer_url.decode("ascii"))
            query = parse_qs(parsed.query, strict_parsing=True)
            if parsed.scheme != "tcp" or parsed.hostname is None or parsed.port is None:
                raise ValueError
            flows = int(query["flows"][0])
            token = bytes.fromhex(query["token"][0])
            if flows < 1 or len(token) != 16:
                raise ValueError
            return parsed.hostname, parsed.port, flows, token
        except (KeyError, ValueError) as error:
            raise ValueError("invalid TCP transport URL") from error

    @staticmethod
    def _configure_socket(sock: socket.socket) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                sock, _ = listener.accept()
                self._configure_socket(sock)
                handshake = _recv_exact(sock, _HANDSHAKE.size)
                magic, server_id, client_id, flow_index, flow_count = _HANDSHAKE.unpack(
                    handshake
                )
                if (
                    magic != _MAGIC
                    or server_id != self._endpoint_id
                    or flow_count != self._num_flows
                    or flow_index >= flow_count
                ):
                    sock.close()
                    continue
                with self._state_lock:
                    group = self._pending_sockets.setdefault(client_id, {})
                    if flow_index in group:
                        sock.close()
                        continue
                    group[flow_index] = sock
                    if len(group) != flow_count:
                        continue
                    sockets = [group[index] for index in range(flow_count)]
                    del self._pending_sockets[client_id]
                self._install_connection(sockets, active=False)
            except OSError:
                if not self._closed.is_set():
                    self._fail_connection(ConnectionError("TCP listener failed"))
                return
            except BaseException as error:
                self._fail_connection(error)
                return

    def _install_connection(
        self, sockets: list[socket.socket], *, active: bool
    ) -> None:
        flows = [_Flow(sock) for sock in sockets]
        with self._state_lock:
            if self._closed.is_set():
                raise RuntimeError("transport is closed")
            current = self._active_flows if active else self._passive_flows
            if current:
                raise RuntimeError("transport is already connected")
            if active:
                self._active_flows = flows
                self._flows = flows
            else:
                self._passive_flows = flows
                if not self._flows:
                    self._flows = flows
            self._connected.set()
        for index, flow in enumerate(flows):
            threading.Thread(
                target=self._send_loop,
                args=(flow,),
                name=f"tcp-transport-send-{index}",
                daemon=True,
            ).start()
            threading.Thread(
                target=self._receive_loop,
                args=(flow,),
                name=f"tcp-transport-receive-{index}",
                daemon=True,
            ).start()

    def _send_loop(self, flow: _Flow) -> None:
        try:
            while (packet := flow.outgoing.get()) is not None:
                flow.sock.sendall(packet.header)
                if packet.payload:
                    flow.sock.sendall(packet.payload)
        except BaseException as error:
            if not self._closed.is_set():
                self._fail_connection(error)

    def _receive_loop(self, flow: _Flow) -> None:
        try:
            while not self._closed.is_set():
                frame_length = _FRAME_LENGTH.unpack(
                    _recv_exact(flow.sock, _FRAME_LENGTH.size)
                )[0]
                if (
                    not _FRAME_HEADER.size
                    <= frame_length
                    <= (_FRAME_HEADER.size + self._chunk_size)
                ):
                    raise RuntimeError(f"invalid TCP frame length {frame_length}")
                header = _recv_exact(flow.sock, _FRAME_HEADER.size)
                fields = _FRAME_HEADER.unpack(header)
                self._dispatch_frame(flow, fields, frame_length - _FRAME_HEADER.size)
        except BaseException as error:
            if not self._closed.is_set():
                self._fail_connection(error)

    def _dispatch_frame(
        self, flow: _Flow, fields: tuple[int, ...], payload_length: int
    ) -> None:
        opcode, request_id, buffer_id, access_key, remote_offset, total, offset = fields
        if opcode == _WRITE:
            with self._state_lock:
                rejected = request_id in self._rejected_writes
            if rejected:
                _recv_exact(flow.sock, payload_length)
                return
            try:
                self._receive_write(
                    flow,
                    request_id,
                    buffer_id,
                    access_key,
                    remote_offset,
                    total,
                    offset,
                    payload_length,
                )
            except BaseException as error:
                with self._state_lock:
                    self._incoming_writes.pop(request_id, None)
                    self._rejected_writes.add(request_id)
                self._send_error(flow, request_id, str(error))
            return
        if opcode == _READ:
            try:
                self._receive_read(
                    request_id,
                    buffer_id,
                    access_key,
                    remote_offset,
                    total,
                )
            except BaseException as error:
                self._send_error(flow, request_id, str(error))
            return
        if opcode == _WRITE_ACK:
            if payload_length:
                raise RuntimeError("write acknowledgement contains data")
            self._receive_ack(request_id)
        elif opcode == _READ_DATA:
            self._receive_read_data(
                flow.sock, request_id, total, offset, payload_length
            )
        elif opcode == _ERROR:
            payload = _recv_exact(flow.sock, payload_length)
            message = bytes(payload).decode("utf-8", "replace")
            self._receive_error(request_id, message)
        elif opcode == _CLOSE:
            if payload_length:
                raise RuntimeError("close frame contains data")
            raise ConnectionError("TCP peer closed the transport")
        else:
            raise RuntimeError(f"unknown TCP transport opcode {opcode}")

    def _receive_write(
        self,
        flow: _Flow,
        request_id: int,
        buffer_id: int,
        access_key: int,
        remote_offset: int,
        total: int,
        offset: int,
        payload_length: int,
    ) -> None:
        try:
            registered = self._lookup_memory(buffer_id, access_key)
            registered_length = (
                registered.tensor.numel() * registered.tensor.element_size()
            )
            if remote_offset + total > registered_length:
                raise ValueError("write exceeds the registered buffer")
            if offset + payload_length > total:
                raise ValueError("write chunk exceeds the request")
            with self._state_lock:
                incoming = self._incoming_writes.get(request_id)
                if incoming is None:
                    incoming = _IncomingWrite(
                        buffer_id, access_key, remote_offset, total
                    )
                    self._incoming_writes[request_id] = incoming
                elif (
                    incoming.buffer_id != buffer_id
                    or incoming.access_key != access_key
                    or incoming.remote_offset != remote_offset
                    or incoming.expected != total
                ):
                    raise RuntimeError("inconsistent write chunks")
        except BaseException:
            _recv_exact(flow.sock, payload_length)
            raise
        self._recv_tensor(
            flow.sock, registered.tensor, remote_offset + offset, payload_length
        )
        with self._state_lock:
            incoming.received += payload_length
            if incoming.received > total:
                raise RuntimeError("duplicate write data")
            complete = incoming.received == total
            if complete:
                del self._incoming_writes[request_id]
        if complete:
            self._queue_frame(flow, _WRITE_ACK, request_id)

    def _receive_ack(self, request_id: int) -> None:
        completion = self._get_completion(request_id)
        if completion is None:
            return
        if completion.mutable is not None:
            raise RuntimeError("received a write acknowledgement for a read")
        completion.event.set()

    def _receive_read(
        self,
        request_id: int,
        buffer_id: int,
        access_key: int,
        remote_offset: int,
        total: int,
    ) -> None:
        registered = self._lookup_memory(buffer_id, access_key)
        registered_length = registered.tensor.numel() * registered.tensor.element_size()
        if remote_offset + total > registered_length:
            raise ValueError("read exceeds the registered buffer")
        flows = self._connection()
        with registered.lock:
            for flow_index, offset, length in _segments(
                total, len(flows), self._chunk_size
            ):
                payload = self._tensor_bytes(
                    registered.tensor, remote_offset + offset, length
                )
                self._queue_frame(
                    flows[flow_index],
                    _READ_DATA,
                    request_id,
                    0,
                    0,
                    0,
                    total,
                    offset,
                    payload,
                )

    def _receive_read_data(
        self,
        sock: socket.socket,
        request_id: int,
        total: int,
        offset: int,
        payload_length: int,
    ) -> None:
        completion = self._get_completion(request_id)
        if completion is None:
            _recv_exact(sock, payload_length)
            return
        if completion.mutable is None or total != completion.expected:
            _recv_exact(sock, payload_length)
            raise RuntimeError("invalid read response")
        if offset + payload_length > total:
            _recv_exact(sock, payload_length)
            raise ValueError("read chunk exceeds the request")
        memory = completion.mutable._memory
        self._recv_tensor(
            sock,
            memory._tensor,
            completion.mutable._offset + offset,
            payload_length,
        )
        with completion.lock:
            completion.received += payload_length
            if completion.received > total:
                raise RuntimeError("duplicate read data")
            if completion.received == total:
                completion.event.set()

    def _receive_error(self, request_id: int, message: str) -> None:
        with self._state_lock:
            completion = self._completions.get(request_id)
        if completion is None:
            return
        completion.error = RuntimeError(f"TCP peer: {message}")
        completion.event.set()

    def _send_error(self, flow: _Flow, request_id: int, message: str) -> None:
        payload = message.encode("utf-8")[: self._chunk_size]
        self._queue_frame(flow, _ERROR, request_id, payload=payload)

    def _queue_frame(
        self,
        flow: _Flow,
        opcode: int,
        request_id: int,
        buffer_id: int = 0,
        access_key: int = 0,
        remote_offset: int = 0,
        total: int = 0,
        offset: int = 0,
        payload: memoryview | bytes = b"",
    ) -> None:
        frame_length = _FRAME_HEADER.size + len(payload)
        header = _FRAME_LENGTH.pack(frame_length) + _FRAME_HEADER.pack(
            opcode,
            request_id,
            buffer_id,
            access_key,
            remote_offset,
            total,
            offset,
        )
        flow.outgoing.put(_Packet(header, payload))

    def _new_completion(
        self, mutable: TCPMutableMemoryView | None = None, expected: int = 0
    ) -> tuple[int, _Completion]:
        with self._state_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            completion = _Completion(mutable, expected)
            self._completions[request_id] = completion
        return request_id, completion

    def _get_completion(self, request_id: int) -> _Completion | None:
        with self._state_lock:
            completion = self._completions.get(request_id)
            expired = request_id in self._expired_requests
        if expired:
            return None
        if completion is None:
            raise RuntimeError(f"unknown TCP request {request_id}")
        return completion

    def _wait(self, request_id: int, completion: _Completion) -> None:
        if not completion.event.wait(self._timeout):
            with self._state_lock:
                self._completions.pop(request_id, None)
                self._expired_requests.add(request_id)
            raise TimeoutError(f"TCP request {request_id} timed out")
        self._discard_completion(request_id)
        if completion.error is not None:
            raise completion.error

    def _discard_completion(self, request_id: int) -> None:
        with self._state_lock:
            self._completions.pop(request_id, None)

    def _connection(self) -> list[_Flow]:
        with self._state_lock:
            flows = list(self._flows)
        if not flows or not self.connected():
            raise RuntimeError("TCP transport is not connected")
        return flows

    def _lookup_memory(self, buffer_id: int, access_key: int) -> _RegisteredMemory:
        with self._state_lock:
            registered = self._registered.get((buffer_id, access_key))
        if registered is None:
            raise ValueError("unknown remote buffer")
        return registered

    def _local_view(
        self, view: MemoryView | MutableMemoryView, *, mutable: bool
    ) -> TCPMemoryView | TCPMutableMemoryView:
        expected_type = TCPMutableMemoryView if mutable else TCPMemoryView
        if not isinstance(view, expected_type) or view._memory._transport is not self:
            raise TypeError("memory view belongs to another transport")
        return view

    @staticmethod
    def _remote_buffer(remote: RemoteBuffer) -> TCPRemoteBuffer:
        if not isinstance(remote, TCPRemoteBuffer):
            raise TypeError("remote buffer is not a TCP descriptor")
        return remote

    def _view_bytes(self, view: TCPMemoryView, offset: int, length: int) -> memoryview:
        return self._tensor_bytes(view._memory._tensor, view._offset + offset, length)

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor, offset: int, length: int) -> memoryview:
        byte_tensor = tensor.detach().reshape(-1).view(torch.uint8)
        byte_tensor = byte_tensor[offset : offset + length]
        if tensor.device.type != "cpu":
            byte_tensor = byte_tensor.cpu()
        return memoryview(byte_tensor.numpy())

    @staticmethod
    def _copy_to_tensor(tensor: torch.Tensor, offset: int, data: memoryview) -> None:
        destination = (
            tensor.detach().reshape(-1).view(torch.uint8)[offset : offset + len(data)]
        )
        if tensor.device.type == "cpu":
            memoryview(destination.numpy())[:] = data
            return
        source = torch.frombuffer(data, dtype=torch.uint8).to(tensor.device)
        destination.copy_(source)
        torch.cuda.current_stream(tensor.device).synchronize()

    @classmethod
    def _recv_tensor(
        cls, sock: socket.socket, tensor: torch.Tensor, offset: int, length: int
    ) -> None:
        if tensor.device.type != "cpu":
            cls._copy_to_tensor(tensor, offset, memoryview(_recv_exact(sock, length)))
            return
        byte_tensor = tensor.detach().reshape(-1).view(torch.uint8)
        destination = memoryview(byte_tensor.numpy())[offset : offset + length]
        _recv_into(sock, destination)

    def _fail_connection(self, error: BaseException) -> None:
        with self._state_lock:
            flows = self._active_flows + self._passive_flows
            self._flows = []
            self._active_flows = []
            self._passive_flows = []
            completions = list(self._completions.values())
        self._connected.clear()
        for completion in completions:
            completion.error = error
            completion.event.set()
        for flow in flows:
            flow.outgoing.put(None)
            try:
                flow.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            flow.sock.close()


__all__ = [
    "TCPMemory",
    "TCPMemoryView",
    "TCPMutableMemoryView",
    "TCPRemoteBuffer",
    "TCPTransport",
]
