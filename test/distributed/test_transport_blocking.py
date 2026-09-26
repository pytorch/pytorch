# Owner(s): ["oncall: distributed"]

import asyncio
import gc
import threading
import weakref
from datetime import timedelta
from unittest.mock import patch

import torch
from torch.distributed._transport import _blocking as _work, wait_all, Work
from torch.distributed._transport._blocking import _BlockingTransport as Transport
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
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

    def write(self, local_buffer, remote_buffer, *, async_op=False, timeout=None):
        device = (
            local_buffer.device
            if isinstance(local_buffer, torch.Tensor)
            else torch.device("cpu")
        )
        return self._run_transfer(
            lambda: self.operation(local_buffer, remote_buffer),
            device,
            async_op=async_op,
            timeout=timeout,
        )

    def read(self, local_buffer, remote_buffer, *, async_op=False, timeout=None):
        return self.write(
            local_buffer, remote_buffer, async_op=async_op, timeout=timeout
        )

    def close(self) -> None:
        self._close_work()
        self.closed = True


class _EntryPoint:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def load(self):
        return self.value


@instantiate_parametrized_tests
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

    def test_synchronous_backend_timeout_preserved(self):
        def operation(local, remote):
            raise TimeoutError("backend operation timed out")

        with _TestTransport(operation=operation) as transport:
            with self.assertRaisesRegex(TimeoutError, "backend operation timed out"):
                transport.write(None, None)

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

    def test_work_normalizes_legacy_future_timeout(self):
        class LegacyTimeoutError(Exception):
            pass

        transport, started, release, _ = self.blocked_transport()
        work = transport.write(None, None, async_op=True)
        self.assertTrue(started.wait(5))
        with (
            patch.object(_work, "FutureTimeoutError", LegacyTimeoutError),
            patch.object(work._future, "result", side_effect=LegacyTimeoutError),
            self.assertRaisesRegex(TimeoutError, "transport wait timed out"),
        ):
            work.wait(timedelta(milliseconds=1))
        release.set()
        self.assertTrue(work.wait())

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

    @parametrize("operation", ["read_async", "write_async"])
    def test_asyncio_transfer_yields(self, operation):
        transport, started, release, calls = self.blocked_transport()

        async def run():
            task = asyncio.create_task(getattr(transport, operation)(1, None))
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            self.assertFalse(task.done())
            release.set()
            self.assertIsNone(await task)

        asyncio.run(run())
        self.assertEqual(calls, [1])

    @parametrize("failure", ["transfer", "dispatch"])
    def test_asyncio_errors_drain_batch(self, failure):
        transport, started, release, _ = self.blocked_transport()

        def fail(local, remote):
            raise ValueError("transfer failed")

        broken = _TestTransport(operation=fail)
        self.addCleanup(broken.close)

        def submit():
            if failure == "transfer":
                yield broken.write(None, None, async_op=True)
            yield transport.write(None, None, async_op=True)
            if failure == "dispatch":
                raise ValueError("dispatch failed")

        async def run():
            task = asyncio.create_task(wait_all(submit()))
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaisesRegex(ValueError, f"{failure} failed"):
                await task

        asyncio.run(run())

    def test_asyncio_cancellation_preserves_other_waiters(self):
        transport, started, release, _ = self.blocked_transport()
        work = transport.write(None, None, async_op=True)

        async def run():
            cancelled = asyncio.create_task(wait_all([work]))
            other = asyncio.create_task(wait_all([work]))
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            await asyncio.sleep(0)
            cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled
            self.assertFalse(work.is_completed())
            self.assertFalse(other.done())
            release.set()
            self.assertIsNone(await other)
            self.assertTrue(work.is_success())

        asyncio.run(run())

    def test_explicit_asyncio_timeout_does_not_drain(self):
        transport, started, release, _ = self.blocked_transport()
        work = transport.write(None, None, async_op=True)

        async def run():
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            with self.assertRaises(TimeoutError):
                await wait_all([work], timeout=0.001)
            self.assertFalse(work.is_completed())
            release.set()
            await wait_all([work], timeout=5)

        asyncio.run(run())

    def test_explicit_asyncio_cancellation_does_not_cancel_work(self):
        transport, started, release, _ = self.blocked_transport()
        work = transport.write(None, None, async_op=True)

        async def run():
            task = asyncio.create_task(wait_all([work], timeout=5))
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(work.is_completed())
            release.set()
            await wait_all([work])

        asyncio.run(run())

    @parametrize("timeout", [-1, float("nan"), float("inf")])
    def test_invalid_asyncio_timeout_does_not_submit(self, timeout):
        with _TestTransport() as transport:
            with patch.object(transport, "write") as write:
                with self.assertRaises(ValueError):
                    asyncio.run(transport.write_async(None, None, timeout=timeout))
                write.assert_not_called()

    def test_asyncio_completed_work_and_empty_batch(self):
        with _TestTransport() as transport:
            work = transport.write(None, None, async_op=True)
            work.wait()
            asyncio.run(wait_all([work]))
            asyncio.run(wait_all([work]))
            asyncio.run(wait_all([]))


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
