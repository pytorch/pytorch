# Owner(s): ["oncall: distributed"]

import asyncio
import threading
from datetime import timedelta
from typing import Any, cast
from unittest.mock import patch

import torch
from torch.distributed._transport import (
    _registry,
    _work,
    available_transports,
    new_transport,
    register_transport,
    Transport,
    wait_all,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


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
        self.operation(local_buffer, remote_buffer)
        return _ManualWork(done=True) if async_op else 0

    def read(self, local_buffer, remote_buffer, *, async_op=False):
        return self.write(local_buffer, remote_buffer, async_op=async_op)

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
                ("external", "nixl", "tcp", "torchcomms", "ucxx"),
            )


class _ManualWork(_work._PollingWork):
    def __init__(self, *, done=False, error=None):
        super().__init__()
        self.done = done
        self._error = error
        self.polls = 0

    def _poll(self):
        self.polls += 1
        return self.done


@instantiate_parametrized_tests
class TestTransportWork(TestCase):
    def test_wait_timeout_and_retry(self):
        work = _ManualWork()
        with self.assertRaises(TimeoutError):
            work.wait(timedelta(milliseconds=1))
        self.assertFalse(work.is_completed())
        work.done = True
        self.assertTrue(work.wait())
        self.assertTrue(work.is_success())
        self.assertEqual(work.result(), [])
        self.assertEqual(work.get_future().wait(), [])

    def test_failure_is_terminal(self):
        work = _ManualWork(done=True, error=RuntimeError("native failure"))
        self.assertTrue(work.is_completed())
        self.assertFalse(work.is_success())
        self.assertIsInstance(work.exception(), RuntimeError)
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            work.wait()
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            work.get_future().wait()

    def test_pending_future_requires_event_loop(self):
        with self.assertRaisesRegex(RuntimeError, "running event loop"):
            _ManualWork().get_future()

    def test_future_driven_by_event_loop(self):
        async def run():
            work = _ManualWork()
            future = work.get_future()
            callback = asyncio.Event()
            future.add_done_callback(lambda _: callback.set())
            self.assertIs(work.get_future(), future)
            self.assertFalse(future.done())
            work.done = True
            await asyncio.wait_for(callback.wait(), 1)
            self.assertEqual(future.wait(), [])

        asyncio.run(run())

    def test_future_reentrant_callback(self):
        async def run():
            work = _ManualWork()
            future = work.get_future()
            seen = []
            future.add_done_callback(lambda _: seen.append(work.wait()))
            work.done = True
            await wait_all([work])
            self.assertEqual(seen, [True])

        asyncio.run(run())

    def test_async_wait_yields(self):
        async def run():
            work = _ManualWork()

            async def complete():
                await asyncio.sleep(0.01)
                work.done = True

            task = asyncio.create_task(complete())
            await wait_all([work], timeout=1)
            await task
            self.assertGreater(work.polls, 1)

        asyncio.run(run())

    def test_timeout_does_not_complete_work(self):
        async def run():
            work = _ManualWork()
            with self.assertRaises(TimeoutError):
                await wait_all([work], timeout=0.001)
            self.assertFalse(work.is_completed())
            work.done = True
            await wait_all([work])

        asyncio.run(run())

    def test_cancellation_does_not_complete_work(self):
        async def run():
            work = _ManualWork()
            task = asyncio.create_task(wait_all([work]))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(work.is_completed())
            work.done = True
            await wait_all([work])

        asyncio.run(run())

    def test_error_drains_other_work(self):
        async def run():
            good = _ManualWork()
            bad = _ManualWork(done=True, error=ValueError("bad"))
            asyncio.get_running_loop().call_later(0.01, setattr, good, "done", True)
            with self.assertRaisesRegex(ValueError, "bad"):
                await wait_all([bad, good], timeout=1)
            self.assertTrue(good.is_completed())

        asyncio.run(run())

    def test_generator_error_drains_submitted_work(self):
        work = _ManualWork()

        def items():
            yield work
            raise ValueError("generator failed")

        async def run():
            asyncio.get_running_loop().call_later(0.01, setattr, work, "done", True)
            with self.assertRaisesRegex(ValueError, "generator failed"):
                await wait_all(items(), timeout=1)
            self.assertTrue(work.is_completed())

        asyncio.run(run())

    @parametrize("timeout", [-1, float("nan"), float("inf")])
    def test_invalid_timeout_does_not_submit(self, timeout):
        with _TestTransport() as transport:
            with patch.object(transport, "write") as write:
                with self.assertRaises(ValueError):
                    asyncio.run(transport.write_async(None, None, timeout=timeout))
                write.assert_not_called()

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
        asyncio.run(wait_all([_ManualWork(done=True)], timeout=0))
        with self.assertRaisesRegex(RuntimeError, "failed"):
            asyncio.run(
                wait_all(
                    [_ManualWork(done=True, error=RuntimeError("failed"))], timeout=0
                )
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
