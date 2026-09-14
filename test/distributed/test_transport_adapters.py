# Owner(s): ["oncall: distributed"]

import asyncio
import gc
import pickle
import queue
import threading
import time
import weakref
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

from transport_test_utils import TransportTestMixin

import torch
from torch.distributed._transport import _torchcomms, _ucxx
from torch.testing._internal.common_utils import run_tests, TestCase


@dataclass(frozen=True)
class _NativeRemote:
    key: tuple[int, int]


class _NativeView:
    def __init__(self, tensor, offset, length):
        self.tensor = tensor
        self.offset = offset
        self._size = length

    def size(self):
        return self._size


class _NativeMemory:
    registrations = {}

    def __init__(self, tensor):
        self.tensor = tensor
        self.key = tensor.data_ptr(), tensor.nbytes
        self.reused = self.key in self.registrations
        self.registrations[self.key] = tensor

    def to_view(self, offset, length):
        offset = offset or 0
        length = self.tensor.nbytes - offset if length is None else length
        if offset < 0 or length < 0 or offset + length > self.tensor.nbytes:
            raise ValueError("view is outside tensor")
        return _NativeView(self.tensor, offset, length)

    def to_mutable_view(self, offset, length):
        return self.to_view(offset, length)

    def to_remote_buffer(self):
        return _NativeRemote(self.key)

    def reused_registration(self):
        return self.reused


class _NativeTransport:
    @staticmethod
    def supported():
        return True

    def __init__(self, device):
        self.device = device
        self.calls = []

    def bind(self):
        return b"address"

    def connect(self, url):
        self.calls.append(("connect", url))
        return 0

    def connected(self):
        return True

    def write(self, local, remote):
        self.calls.append(("write", local, remote))
        source = local.tensor.view(torch.uint8).flatten()
        destination = (
            _NativeMemory.registrations[remote.key].view(torch.uint8).flatten()
        )
        destination[: local.size()].copy_(
            source[local.offset : local.offset + local.size()].clone()
        )
        return 0

    def read(self, local, remote):
        self.calls.append(("read", local, remote))
        source = _NativeMemory.registrations[remote.key].view(torch.uint8).flatten()
        destination = local.tensor.view(torch.uint8).flatten()
        destination[local.offset : local.offset + local.size()].copy_(
            source[: local.size()].clone()
        )
        return 0


class TestTorchCommsTransport(TransportTestMixin, TestCase):
    def setUp(self):
        super().setUp()
        _NativeMemory.registrations.clear()

    def backend(self):
        return SimpleNamespace(RdmaMemory=_NativeMemory, RdmaTransport=_NativeTransport)

    def make_transport_pair(self):
        with patch.object(_torchcomms, "_load_backend", return_value=self.backend()):
            first = _torchcomms.TorchCommsTransport("cpu")
            second = _torchcomms.TorchCommsTransport("cpu")
        self.assertEqual(first.connect(second.bind()), 0)
        self.assertEqual(second.connect(first.bind()), 0)
        return first, second

    def test_supported(self):
        with patch.object(_torchcomms, "_load_backend", return_value=self.backend()):
            self.assertTrue(_torchcomms.TorchCommsTransport.supported())

    def test_forwards_operations_and_keeps_memory_alive(self):
        with patch.object(_torchcomms, "_load_backend", return_value=self.backend()):
            transport = _torchcomms.TorchCommsTransport("cpu")
        tensor = torch.arange(8)
        memory = transport.register_memory(tensor)
        view = memory.to_view(1, 3)
        mutable_view = memory.to_mutable_view(2, 4)

        self.assertEqual(view.size(), 3)
        self.assertEqual(mutable_view.size(), 4)
        remote = memory.to_remote_buffer()
        self.assertEqual(remote, _NativeRemote((tensor.data_ptr(), tensor.nbytes)))
        self.assertFalse(memory.reused_registration())
        self.assertIs(view._memory, memory)
        self.assertEqual(transport.write(view, remote), 0)
        self.assertEqual(transport.read(mutable_view, remote), 0)
        self.assertEqual(transport.bind(), b"address")
        self.assertEqual(transport.connect(b"peer"), 0)
        self.assertTrue(transport.connected())
        transport.close()
        self.assertFalse(transport.connected())


class _UCXXEndpoint:
    def __init__(self, incoming, outgoing):
        self._incoming = incoming
        self._outgoing = outgoing
        self.closed = False

    async def send(self, buffer):
        if self.closed:
            raise ConnectionError("endpoint is closed")
        self._outgoing.put(bytes(buffer))

    async def recv(self, buffer):
        data = await asyncio.to_thread(self._incoming.get)
        if data is None:
            self.closed = True
            raise ConnectionError("endpoint is closed")
        view = memoryview(buffer)
        if view.format != "B":
            view = view.cast("B")
        view[:] = data

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self._incoming.put(None)
        self._outgoing.put(None)


class _UCXXListener:
    def __init__(self, backend, callback, port):
        self._backend = backend
        self._callback = callback
        self._loop = asyncio.get_running_loop()
        self.port = port
        self.closed = False

    def close(self):
        self.closed = True
        self._backend.listeners.pop(self.port, None)


class _HangingEndpoint:
    def __init__(self):
        self.closed = False

    async def send(self, buffer):
        pass

    async def recv(self, buffer):
        await asyncio.Event().wait()

    async def close(self):
        self.closed = True


class _UCXXBackend:
    def __init__(self):
        self.listeners = {}
        self._next_port = 1234
        self._lock = threading.Lock()

    def create_listener(self, callback):
        with self._lock:
            port = self._next_port
            self._next_port += 1
            listener = _UCXXListener(self, callback, port)
            self.listeners[port] = listener
        return listener

    async def create_endpoint(self, address, port):
        self.assert_address(address)
        listener = self.listeners[port]
        to_server = queue.Queue()
        to_client = queue.Queue()
        client = _UCXXEndpoint(to_client, to_server)
        server = _UCXXEndpoint(to_server, to_client)
        asyncio.run_coroutine_threadsafe(listener._callback(server), listener._loop)
        return client

    @staticmethod
    def assert_address(address):
        if address != "127.0.0.1":
            raise AssertionError(f"unexpected address {address}")

    @staticmethod
    def get_address():
        return "127.0.0.1"


class TestUCXXTransport(TransportTestMixin, TestCase):
    def connect(self, *, symmetric=False, timeout=5.0):
        backend = _UCXXBackend()
        with patch.object(_ucxx, "_load_backend", return_value=backend):
            server = _ucxx.UCXXTransport("cpu", host="127.0.0.1", timeout=timeout)
            client = _ucxx.UCXXTransport("cpu", host="127.0.0.1", timeout=timeout)
        server_url = server.bind()
        client_url = client.bind() if symmetric else None
        self.assertEqual(client.connect(server_url), 0)
        if client_url is not None:
            self.assertEqual(server.connect(client_url), 0)
        deadline = time.monotonic() + timeout
        while not server.connected() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(server.connected())
        self.assertTrue(client.connected())
        return server, client

    def make_transport_pair(self):
        return self.connect(symmetric=True)

    def test_default_host_is_loopback(self):
        backend = _UCXXBackend()
        with patch.object(_ucxx, "_load_backend", return_value=backend):
            transport = _ucxx.UCXXTransport("cpu")
        try:
            self.assertIn(b'"address":"127.0.0.1"', transport.bind())
        finally:
            transport.close()

    def test_close_releases_registered_tensors(self):
        backend = _UCXXBackend()
        with patch.object(_ucxx, "_load_backend", return_value=backend):
            transport = _ucxx.UCXXTransport("cpu", host="127.0.0.1")
        tensor = torch.arange(8)
        memory = transport.register_memory(tensor)
        tensor_ref = weakref.ref(tensor)
        del tensor, memory

        self.assertIsNotNone(tensor_ref())
        transport.close()
        gc.collect()
        self.assertIsNone(tensor_ref())

    def test_read_write(self):
        server, client = self.connect()
        try:
            source = torch.arange(16, dtype=torch.int32)
            destination = torch.zeros(8, dtype=torch.int32)
            source_memory = client.register_memory(source)
            destination_memory = server.register_memory(destination)
            remote = destination_memory.to_remote_buffer()

            self.assertEqual(pickle.loads(pickle.dumps(remote)), remote)
            self.assertEqual(client.write(source_memory.to_view(32, 32), remote), 0)
            self.assertEqual(destination, source[8:])

            result = torch.full((10,), -1, dtype=torch.int32)
            result_memory = client.register_memory(result)
            self.assertEqual(
                client.read(result_memory.to_mutable_view(4, 32), remote), 0
            )
            self.assertEqual(result[1:9], destination)
            self.assertEqual(result[0], -1)
            self.assertEqual(result[9], -1)
        finally:
            client.close()
            server.close()

    def test_symmetric_connection_and_registration_errors(self):
        server, client = self.connect(symmetric=True)
        try:
            source = server.register_memory(torch.arange(16, dtype=torch.uint8))
            destination_tensor = torch.zeros(16, dtype=torch.uint8)
            first = client.register_memory(destination_tensor)
            second = client.register_memory(destination_tensor)

            self.assertFalse(first.reused_registration())
            self.assertTrue(second.reused_registration())
            self.assertEqual(first.to_remote_buffer(), second.to_remote_buffer())
            invalid = _ucxx.UCXXRemoteBuffer(1, 16, 2)
            with self.assertRaisesRegex(RuntimeError, "unknown remote buffer"):
                server.write(source.to_view(), invalid)
            self.assertEqual(
                server.write(source.to_view(), first.to_remote_buffer()), 0
            )
            self.assertEqual(destination_tensor, torch.arange(16, dtype=torch.uint8))
            with self.assertRaisesRegex(ValueError, "outside"):
                source.to_view(17)
        finally:
            client.close()
            server.close()

    def test_timeout(self):
        endpoint = _HangingEndpoint()

        async def create_endpoint(address, port):
            return endpoint

        backend = SimpleNamespace(
            create_endpoint=create_endpoint,
            create_listener=Mock(),
            get_address=Mock(),
        )
        with patch.object(_ucxx, "_load_backend", return_value=backend):
            transport = _ucxx.UCXXTransport("cpu", timeout=0.01)
        try:
            url = b'{"address":"127.0.0.1","port":1234}'
            self.assertEqual(transport.connect(url), 0)
            source = transport.register_memory(torch.arange(1))
            remote = _ucxx.UCXXRemoteBuffer(1, source.to_view().size(), 2)
            with self.assertRaisesRegex(TimeoutError, "write timed out"):
                transport.write(source.to_view(), remote)
            self.assertFalse(transport.connected())
            with self.assertRaisesRegex(RuntimeError, "transport failed"):
                transport.write(source.to_view(), remote)
        finally:
            transport.close()


if __name__ == "__main__":
    run_tests()
