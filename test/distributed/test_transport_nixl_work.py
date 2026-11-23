# Owner(s): ["oncall: distributed"]

import asyncio
from datetime import timedelta
from unittest.mock import patch

from torch.distributed._transport import wait_all
from torch.distributed._transport.nixl._work import _PollingWork
from torch.testing._internal.common_utils import run_tests, TestCase


class _ManualWork(_PollingWork):
    def __init__(self, *, done=False, error=None):
        super().__init__()
        self.done = done
        self._error = error
        self.polls = 0

    def _poll(self):
        self.polls += 1
        return self.done


class TestNIXLPollingWork(TestCase):
    def test_polling_backoff(self):
        work = _ManualWork()
        with patch(
            "torch.distributed._transport.nixl._work.time.sleep",
            side_effect=lambda _: setattr(work, "done", True),
        ) as sleep:
            work.wait()
            sleep.assert_called_once_with(0.001)

        async def run():
            work = _ManualWork()
            with patch(
                "torch.distributed._transport.nixl._work.asyncio.sleep",
                side_effect=lambda _: setattr(work, "done", True),
            ) as sleep:
                await work._drive_future()
                sleep.assert_awaited_once_with(0.001)

        asyncio.run(run())

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


if __name__ == "__main__":
    run_tests()
