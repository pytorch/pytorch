# Owner(s): ["oncall: distributed"]

import asyncio
import itertools
import threading
from concurrent.futures import Future
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from torch.distributed._transport import _api, Transport, wait_all
from torch.distributed._transport._work import _FutureWork
from torch.distributed._transport.nixl._work import _PollingWork
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class _View:
    def size(self) -> int:
        return 8


_PAIR = (_View(), None)


class _ChainedWork(_FutureWork):
    """A transfer whose future comes from ``then()``, so its error is held in C++."""

    def __init__(self, transfer: Future[None]) -> None:
        super().__init__(transfer, None)

    def get_future(self):
        return super().get_future().then(lambda future: future.wait())

    def wait(self, timeout=timedelta(0)):
        # There is no default timeout, so this wait would hang the test.
        if not timeout and not self._future.done():
            raise AssertionError("unbounded wait on a pending transfer")
        return super().wait(timeout)


class _PolledWork(_PollingWork):
    """A transfer whose future, like NIXL's, is driven from the calling loop."""

    def __init__(self, transfer: Future[None]) -> None:
        super().__init__()
        self._transfer = transfer

    def _poll(self) -> bool:
        return self._transfer.done()


class _ScriptedTransport(Transport):
    """A transport without a native batch path whose transfers the test settles."""

    def __init__(self, work=_ChainedWork, fail_dispatch_at=None) -> None:
        super().__init__()
        self.work = work
        self.fail_dispatch_at = fail_dispatch_at
        self.transfers: list[Future[None]] = []
        self.timeouts: list[float | None] = []

    @staticmethod
    def supported() -> bool:
        return True

    def bind(self, *, timeout=None) -> bytes:
        return b""

    def connect(self, peer_url, *, timeout=None) -> int:
        return 0

    def connected(self) -> bool:
        return True

    def register_memory(self, tensor, *, timeout=None):
        raise NotImplementedError

    def write(self, local_buffer, remote_buffer, *, async_op=False, timeout=None):
        if len(self.transfers) == self.fail_dispatch_at:
            raise RuntimeError("dispatch failed")
        self.timeouts.append(timeout)
        self.transfers.append(Future())
        return self.work(self.transfers[-1])

    read = write

    def close(self, *, timeout=None) -> None:
        pass


class TestTransportBatchFallback(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_failure_waits_for_every_transfer(self):
        transport = _ScriptedTransport()
        work = transport.write_batch([_PAIR] * 2, async_op=True)
        future = work.get_future()
        failed, pending = transport.transfers
        failed.set_exception(RuntimeError("first failed"))
        self.assertFalse(work.is_completed())
        self.assertIsNone(work.exception())
        self.assertFalse(future.done())
        with self.assertRaises(TimeoutError):
            work.wait(timedelta(milliseconds=10))

        pending.set_result(None)
        self.assertTrue(work.is_completed())
        for wait in (work.wait, future.wait):
            with self.assertRaisesRegex(RuntimeError, "first failed"):
                wait()

    def test_dispatch_failure(self):
        with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
            _ScriptedTransport(fail_dispatch_at=0).write_batch([_PAIR], async_op=True)

        transport = _ScriptedTransport(fail_dispatch_at=1)
        work = transport.write_batch([_PAIR] * 3, async_op=True)
        self.assertEqual(len(transport.transfers), 1)
        self.assertFalse(work.is_completed())
        with self.assertRaises(TimeoutError):
            work.wait(timedelta(milliseconds=10))
        transport.transfers[0].set_exception(RuntimeError("transfer failed"))
        self.assertRegex(str(work.exception()), "dispatch failed")
        for wait in (work.wait, work.get_future().wait):
            with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
                wait()

    def test_timeout_bounds_the_whole_batch(self):
        transport = _ScriptedTransport()
        ticks = itertools.count()
        clock = SimpleNamespace(monotonic=lambda: float(next(ticks)))
        with patch.object(_api, "time", clock):
            transport.write_batch([_PAIR] * 2, async_op=True, timeout=10)
        # Each dispatch gets what is left of the batch's deadline.
        self.assertEqual(transport.timeouts, [9.0, 8.0])

        with self.assertRaises(TimeoutError):
            transport.write_batch([_PAIR] * 2, timeout=0)
        batch = transport.write_batch_async([_PAIR], timeout=0)
        with self.assertRaisesRegex(TimeoutError, "remain pending"):
            asyncio.run(asyncio.wait_for(batch, 5))

    def test_future_outlives_its_event_loop(self):
        transport = _ScriptedTransport(work=_PolledWork)
        work = transport.write_batch([_PAIR], async_op=True)
        with self.assertRaisesRegex(RuntimeError, "no running event loop"):
            work.get_future()
        with self.assertRaises(TimeoutError):
            asyncio.run(wait_all([work], timeout=0))
        transport.transfers[0].set_result(None)
        asyncio.run(wait_all([work], timeout=5))

    def test_async_dispatch_leaves_the_event_loop_running(self):
        transport = _ScriptedTransport()
        released = threading.Event()
        write = transport.write

        def blocking_write(*args, **kwargs):
            # The event loop releases this, so it must keep running meanwhile.
            if not released.wait(5):
                raise RuntimeError("dispatch blocked the event loop")
            work = write(*args, **kwargs)
            transport.transfers[-1].set_result(None)
            return work

        async def run():
            batch = asyncio.create_task(transport.write_batch_async([_PAIR]))
            await asyncio.sleep(0)
            released.set()
            await batch

        with patch.object(transport, "write", blocking_write):
            asyncio.run(run())


if __name__ == "__main__":
    run_tests()
