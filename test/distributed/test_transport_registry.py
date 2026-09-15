# Owner(s): ["oncall: distributed"]

import gc
import threading
import weakref
from datetime import timedelta
from typing import Any, cast
from unittest.mock import patch

import torch
from torch.distributed._transport import (
    _registry,
    available_transports,
    new_transport,
    register_transport,
    Transport,
    Work,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class _TestTransport(Transport):
    is_supported = True

    def __init__(self, device=None, *, value=None, operation=None):
        super().__init__(device)
        self.value = value
        self.operation = operation or (lambda local, remote: 0)
        self.closed = False

    @staticmethod
    def supported() -> bool:
        return _TestTransport.is_supported

    def bind(self) -> bytes:
        return b"test://"

    def connect(self, peer_url: bytes) -> int:
        return 0

    def connected(self) -> bool:
        return True

    def register_memory(self, tensor):
        return tensor

    def write(self, local_buffer, remote_buffer, *, async_op=False):
        device = (
            local_buffer.device
            if isinstance(local_buffer, torch.Tensor)
            else torch.device("cpu")
        )
        return self._run_transfer(
            lambda: self.operation(local_buffer, remote_buffer),
            device,
            async_op=async_op,
        )

    def read(self, local_buffer, remote_buffer, *, async_op=False):
        return self.write(local_buffer, remote_buffer, async_op=async_op)

    def close(self) -> None:
        self._close_work()
        self.closed = True


class _EntryPoint:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def load(self):
        return self.value


class TestTransportRegistry(TestCase):
    def setUp(self):
        self.registered = patch.dict(_registry._registered_transports, clear=True)
        self.registered.start()

    def tearDown(self):
        self.registered.stop()

    def test_register_and_create(self):
        register_transport("test", _TestTransport)
        transport = new_transport("TEST", "cpu", value=3)
        self.assertIsInstance(transport, _TestTransport)
        self.assertEqual(transport.device, torch.device("cpu"))
        self.assertEqual(transport.value, 3)

    def test_factory_without_device(self):
        register_transport("test", lambda *, value: _TestTransport(value=value))
        transport = new_transport("test", value=3)
        self.assertIsNone(transport.device)
        self.assertEqual(transport.value, 3)

    def test_duplicate_registration(self):
        register_transport("test", _TestTransport)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_transport("test", _TestTransport)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_transport("tcp", _TestTransport)

    def test_factory_must_be_callable(self):
        with self.assertRaisesRegex(TypeError, "must be callable"):
            register_transport("test", cast(Any, None))

    def test_entry_point(self):
        entry_point = _EntryPoint("external", _TestTransport)
        with patch.object(
            _registry, "_iter_entry_points", return_value=iter([entry_point])
        ):
            transport = new_transport("external", "cpu")
        self.assertIsInstance(transport, _TestTransport)

    def test_entry_point_must_return_transport(self):
        entry_point = _EntryPoint("broken", lambda **kwargs: object())
        with patch.object(
            _registry, "_iter_entry_points", return_value=iter([entry_point])
        ):
            with self.assertRaisesRegex(TypeError, "expected a Transport"):
                new_transport("broken", "cpu")

    def test_duplicate_entry_points(self):
        entry_points = [
            _EntryPoint("duplicate", _TestTransport),
            _EntryPoint("duplicate", _TestTransport),
        ]
        with patch.object(
            _registry, "_iter_entry_points", return_value=iter(entry_points)
        ):
            with self.assertRaisesRegex(RuntimeError, "multiple entry points"):
                new_transport("duplicate", "cpu")

    def test_unknown_transport(self):
        with patch.object(_registry, "_iter_entry_points", return_value=iter(())):
            with self.assertRaisesRegex(ValueError, "unknown transport"):
                new_transport("missing", "cpu")

    def test_unsupported_transport_is_closed(self):
        register_transport("test", _TestTransport)
        _TestTransport.is_supported = False
        try:
            with self.assertRaisesRegex(RuntimeError, "is not supported"):
                new_transport("test", "cpu")
        finally:
            _TestTransport.is_supported = True

    def test_available_transports(self):
        entry_point = _EntryPoint("external", _TestTransport)
        with patch.object(
            _registry, "_iter_entry_points", return_value=iter([entry_point])
        ):
            self.assertEqual(
                available_transports(),
                ("external", "ibverbs", "tcp", "torchcomms", "ucxx"),
            )


class TestTransportWork(TestCase):
    def blocked_transport(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def operation(local, remote):
            started.set()
            if not release.wait(5):
                raise RuntimeError("test operation was not released")
            calls.append(local)
            return 0

        transport = _TestTransport(operation=operation)
        self.addCleanup(transport.close)
        self.addCleanup(release.set)
        return transport, started, release, calls

    def test_work_completion_and_timeout(self):
        transport, started, release, _ = self.blocked_transport()
        work = transport.write(None, None, async_op=True)
        self.assertIsInstance(work, Work)
        self.assertTrue(started.wait(5))
        self.assertFalse(work.is_completed())
        self.assertFalse(work.is_success())
        self.assertFalse(work.get_future().done())
        with self.assertRaises(TimeoutError):
            work.wait(timedelta(milliseconds=1))
        self.assertFalse(work.is_completed())
        release.set()
        self.assertTrue(work.wait())
        self.assertTrue(work.wait())
        self.assertTrue(work.is_completed())
        self.assertTrue(work.is_success())
        self.assertIsNone(work.exception())
        self.assertEqual(work.get_future().wait(), [])
        self.assertEqual(work.result(), [])

    def test_work_error(self):
        def operation(local, remote):
            raise ValueError("transfer failed")

        with _TestTransport(operation=operation) as transport:
            work = transport.read(None, None, async_op=True)
            with self.assertRaisesRegex(ValueError, "transfer failed"):
                work.wait()
            self.assertTrue(work.is_completed())
            self.assertFalse(work.is_success())
            self.assertIsInstance(work.exception(), ValueError)
            with self.assertRaisesRegex(ValueError, "transfer failed"):
                work.get_future().wait()

    def test_work_error_status(self):
        with _TestTransport(operation=lambda local, remote: -1) as transport:
            work = transport.write(None, None, async_op=True)
            with self.assertRaisesRegex(RuntimeError, "status -1"):
                work.wait()
            with self.assertRaisesRegex(RuntimeError, "status -1"):
                work.get_future().wait()
            self.assertFalse(work.is_success())

    def test_close_drains_work(self):
        transport, started, release, calls = self.blocked_transport()
        first = transport.write(1, None, async_op=True)
        self.assertTrue(started.wait(5))
        second = transport.write(2, None, async_op=True)
        closed = threading.Event()

        def close():
            transport.close()
            closed.set()

        thread = threading.Thread(target=close, daemon=True)
        thread.start()
        try:
            self.assertFalse(closed.wait(0.05))
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(closed.is_set())
        self.assertTrue(first.wait())
        self.assertTrue(second.wait())
        self.assertEqual(calls, [1, 2])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            transport.write(3, None, async_op=True)

    def test_sync_follows_queued_work(self):
        transport, started, release, calls = self.blocked_transport()
        first = transport.write(1, None, async_op=True)
        self.assertTrue(started.wait(5))
        second = transport.write(2, None, async_op=True)
        release.set()
        self.assertEqual(transport.write(3, None), 0)
        self.assertEqual(calls, [1, 2, 3])
        self.assertTrue(first.is_completed())
        self.assertTrue(second.is_completed())

    def test_work_retains_buffer(self):
        transport, started, release, _ = self.blocked_transport()
        first = transport.write(None, None, async_op=True)
        self.assertTrue(started.wait(5))
        tensor = torch.ones(16)
        reference = weakref.ref(tensor)
        work = transport.write(tensor, None, async_op=True)
        del tensor, work
        gc.collect()
        self.assertIsNotNone(reference())
        release.set()
        first.wait()
        transport.close()

    def test_callback_cannot_block_its_worker(self):
        transport, started, release, _ = self.blocked_transport()
        first = transport.write(1, None, async_op=True)
        self.assertTrue(started.wait(5))
        second = transport.write(2, None, async_op=True)

        def callback(future):
            with self.assertRaisesRegex(RuntimeError, "synchronous transfer"):
                transport.write(3, None)
            with self.assertRaisesRegex(RuntimeError, "pending work"):
                second.wait()
            with self.assertRaisesRegex(RuntimeError, "close a transport"):
                transport.close()
            return 1

        done = first.get_future().then(callback)
        release.set()
        self.assertEqual(done.wait(), 1)
        self.assertTrue(second.wait())


class TestTransportWorkDevice(TestCase):
    def test_stream_ordering(self, device):
        source = torch.zeros(1024, device=device)
        destination = torch.zeros_like(source)

        def operation(local, remote):
            remote.copy_(local)
            return 0

        with _TestTransport(operation=operation) as transport:
            if source.is_cuda:
                producer = torch.cuda.Stream(device=device)
                producer.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(producer):
                    torch.cuda._sleep(10_000_000)
                    source.fill_(7)
                    work = transport.write(source, destination, async_op=True)
            else:
                source.fill_(7)
                work = transport.write(source, destination, async_op=True)
            self.assertTrue(work.wait())
            self.assertEqual(destination, torch.full_like(destination, 7))

    @onlyCUDA
    def test_cuda_graph_after_async(self, device):
        source = torch.ones(1024, device=device)
        destination = torch.zeros_like(source)

        def operation(local, remote):
            remote.copy_(local)
            return 0

        with _TestTransport(operation=operation) as transport:
            transport.write(source, destination, async_op=True).wait()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.assertEqual(transport.write(source, destination), 0)
            source.fill_(9)
            graph.replay()
            self.assertEqual(destination, torch.full_like(destination, 9))
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot be captured"):
                    transport.write(source, destination, async_op=True)


instantiate_device_type_tests(TestTransportWorkDevice, globals())


if __name__ == "__main__":
    run_tests()
