# Owner(s): ["oncall: distributed"]

import asyncio
import json
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
from torch.distributed._transport import (
    _registry,
    available_transports,
    MemoryView,
    MutableMemoryView,
    new_transport,
    register_transport,
    Transport,
    wait_all,
)
from torch.distributed._transport._serialization import _WireDescriptor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@dataclass(frozen=True)
class _Descriptor(_WireDescriptor):
    _backend = "test"
    _fields = {"address": int, "name": str, "metadata": bytes}

    address: int
    name: str
    metadata: bytes


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

    def write(self, local_buffer, remote_buffer, *, async_op=False, timeout=None):
        self.operation(local_buffer, remote_buffer)
        return _completed_work() if async_op else 0

    def read(self, local_buffer, remote_buffer, *, async_op=False, timeout=None):
        return self.write(
            local_buffer, remote_buffer, async_op=async_op, timeout=timeout
        )

    def close(self) -> None:
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

    def test_unregister_unsupported(self):
        with _TestTransport() as transport:
            with self.assertRaisesRegex(NotImplementedError, "unregistration"):
                transport.unregister_memory(None)

    def test_mutable_view_protocol(self):
        readonly = SimpleNamespace(size=lambda: 8)
        mutable = SimpleNamespace(size=lambda: 8, writable=True)
        self.assertIsInstance(readonly, MemoryView)
        self.assertNotIsInstance(readonly, MutableMemoryView)
        self.assertIsInstance(mutable, MutableMemoryView)

    def test_descriptor_roundtrip(self):
        descriptor = _Descriptor(2**64 - 1, "peer\u2603", b"\x00\xffdata")
        encoded = descriptor.serialize()
        self.assertIsInstance(encoded, bytes)
        self.assertEqual(_Descriptor.deserialize(encoded), descriptor)
        self.assertEqual(json.loads(encoded)["backend"], "test")

    def test_descriptor_rejects_invalid_wire_data(self):
        valid = json.loads(_Descriptor(1, "peer", b"data").serialize())
        cases = [
            b"not json",
            b"\xff",
            b"[]",
            b"null",
            b'{"version":1,"version":1,"backend":"test","fields":{}}',
        ]
        for key, value in [
            ("version", 2),
            ("version", True),
            ("backend", "nixl"),
            ("fields", []),
            ("unexpected", 0),
        ]:
            cases.append(json.dumps({**valid, key: value}).encode())
        for key, value in [
            ("address", -1),
            ("address", True),
            ("address", 1.5),
            ("name", 1),
            ("metadata", "%%%"),
            ("metadata", []),
            ("extra", 0),
        ]:
            cases.append(
                json.dumps(
                    {**valid, "fields": {**valid["fields"], key: value}}
                ).encode()
            )
        cases.append(json.dumps({**valid, "fields": {}}).encode())
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                _Descriptor.deserialize(data)
        with self.assertRaises(TypeError):
            _Descriptor.deserialize("not bytes")
        with self.assertRaises(ValueError):
            _Descriptor(-1, "peer", b"").serialize()

    def test_factory_without_device(self):
        register_transport("test", lambda *, value: _TestTransport(value=value))
        transport = new_transport("test", value=3)
        self.assertIsNone(transport.device)
        self.assertEqual(transport.value, 3)

    def test_duplicate_registration(self):
        register_transport("test", _TestTransport)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_transport("test", _TestTransport)

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
        register_transport("local", _TestTransport)
        entry_point = _EntryPoint("EXTERNAL", _TestTransport)
        with patch.object(
            _registry, "_iter_entry_points", return_value=iter([entry_point])
        ):
            self.assertEqual(available_transports(), ("external", "local"))

    def test_entry_point_registration_requires_replace(self):
        entry_point = _EntryPoint("EXTERNAL", _TestTransport)
        with patch.object(
            _registry, "_iter_entry_points", side_effect=lambda: iter([entry_point])
        ):
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_transport("external", _TestTransport)
            register_transport("external", _TestTransport, replace=True)
            self.assertIsInstance(new_transport("external"), _TestTransport)

    def test_no_implicit_backends(self):
        with patch.object(
            _registry, "_iter_entry_points", side_effect=lambda: iter(())
        ):
            self.assertEqual(available_transports(), ())
            with self.assertRaisesRegex(ValueError, "unknown transport"):
                new_transport("nixl")


def _completed_work(error=None):
    future = torch.futures.Future()
    future.set_result([])
    if error is not None:

        def fail(_):
            raise error

        future = future.then(fail)
    return torch._C._distributed_c10d._create_work_from_future(future)


@instantiate_parametrized_tests
class TestTransportWork(TestCase):
    @parametrize("timeout", [-1, float("nan"), float("inf")])
    def test_invalid_timeout_does_not_submit(self, timeout):
        with _TestTransport() as transport:
            with patch.object(transport, "write") as write:
                with self.assertRaises(ValueError):
                    asyncio.run(transport.write_async(None, None, timeout=timeout))
                write.assert_not_called()

    @parametrize("operation", ["read", "write"])
    def test_async_timeout_reaches_submission(self, operation):
        async def run():
            with _TestTransport() as transport:
                with patch.object(
                    transport, operation, return_value=_completed_work()
                ) as submit:
                    await getattr(transport, operation + "_async")(
                        None, None, timeout=0.5
                    )
                    submit.assert_called_once_with(
                        None, None, async_op=True, timeout=0.5
                    )

        asyncio.run(run())

    @parametrize("operation", ["read", "write"])
    def test_async_submission_consumes_timeout(self, operation):
        async def run():
            with _TestTransport() as transport:
                loop = asyncio.get_running_loop()
                with (
                    patch.object(transport, operation, return_value=_completed_work()),
                    patch("torch.distributed._transport._api.wait_all") as wait,
                    patch.object(loop, "time", side_effect=[10, 10.25]),
                ):
                    await getattr(transport, operation + "_async")(
                        None, None, timeout=0.5
                    )
                    self.assertEqual(wait.call_args.kwargs["timeout"], 0.25)

        asyncio.run(run())

    def test_async_close_requires_backend_support(self):
        with _TestTransport() as transport:
            with self.assertRaisesRegex(NotImplementedError, "async close"):
                asyncio.run(transport.close_async())

    def test_future_only_work_does_not_poll(self):
        future = torch.futures.Future()

        class FutureOnlyWork(torch.distributed.Work):
            def get_future(self):
                return future

            def is_completed(self):
                raise AssertionError("must use the completion future")

            def wait(self, *args):
                raise AssertionError("must not block on Work.wait")

        async def run():
            task = asyncio.create_task(wait_all([FutureOnlyWork()]))
            await asyncio.sleep(0)
            thread = threading.Thread(target=lambda: future.set_result([]))
            thread.start()
            try:
                await asyncio.wait_for(task, 1)
            finally:
                thread.join(5)

        asyncio.run(run())

    def test_c10d_future_wrapped_work(self):
        future = torch.futures.Future()
        work = torch._C._distributed_c10d._create_work_from_future(future)

        async def run():
            asyncio.get_running_loop().call_soon(future.set_result, [])
            await wait_all([work], timeout=1)

        asyncio.run(run())

    def test_completed_future_zero_timeout(self):
        asyncio.run(wait_all([_completed_work()], timeout=0))
        with self.assertRaisesRegex(RuntimeError, "failed"):
            asyncio.run(
                wait_all([_completed_work(error=RuntimeError("failed"))], timeout=0)
            )

    def test_cancelled_waiter_does_not_cancel_shared_future(self):
        future = torch.futures.Future()
        work = torch._C._distributed_c10d._create_work_from_future(future)

        async def run():
            first = asyncio.create_task(wait_all([work]))
            second = asyncio.create_task(wait_all([work]))
            await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertFalse(future.done())
            future.set_result([])
            await asyncio.wait_for(second, 1)

        asyncio.run(run())

    def test_future_completion_after_loop_closes(self):
        future = torch.futures.Future()
        work = torch._C._distributed_c10d._create_work_from_future(future)
        with self.assertRaises(TimeoutError):
            asyncio.run(wait_all([work], timeout=0.001))
        future.set_result([])
        asyncio.run(wait_all([work], timeout=0))

    def test_empty_batch(self):
        asyncio.run(wait_all([]))


if __name__ == "__main__":
    run_tests()
