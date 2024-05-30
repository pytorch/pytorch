# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing as mp
import signal
import time
import unittest
import unittest.mock as mock

import torch.distributed.elastic.timer as timer
from torch.distributed.elastic.timer.api import TimerRequest
from torch.distributed.elastic.timer.local_timer import MultiprocessingRequestQueue
from torch.testing._internal.common_utils import (
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_TSAN,
    TestCase,
)


# timer is not supported on windows or macos
if not (IS_WINDOWS or IS_MACOS or TEST_WITH_DEV_DBG_ASAN):
    # func2 should time out
    def func2(n, mp_queue):
        if mp_queue is not None:
            timer.configure(timer.LocalTimerClient(mp_queue))
        if n > 0:
            with timer.expires(after=0.1):
                func2(n - 1, None)
                time.sleep(0.2)

    class LocalTimerTest(TestCase):
        def setUp(self):
            super().setUp()
            self.ctx = mp.get_context("spawn")
            self.mp_queue = self.ctx.Queue()
            self.max_interval = 0.01
            self.server = timer.LocalTimerServer(self.mp_queue, self.max_interval)
            self.server.start()

        def tearDown(self):
            super().tearDown()
            self.server.stop()

        def test_exception_propagation(self):
            with self.assertRaises(Exception, msg="foobar"):
                with timer.expires(after=1):
                    raise Exception("foobar")  # noqa: TRY002

        def test_no_client(self):
            # no timer client configured; exception expected
            timer.configure(None)
            with self.assertRaises(RuntimeError):
                with timer.expires(after=1):
                    pass

        def test_client_interaction(self):
            # no timer client configured but one passed in explicitly
            # no exception expected
            timer_client = timer.LocalTimerClient(self.mp_queue)
            timer_client.acquire = mock.MagicMock(wraps=timer_client.acquire)
            timer_client.release = mock.MagicMock(wraps=timer_client.release)
            with timer.expires(after=1, scope="test", client=timer_client):
                pass

            timer_client.acquire.assert_called_once_with("test", mock.ANY)
            timer_client.release.assert_called_once_with("test")

        def test_happy_path(self):
            timer.configure(timer.LocalTimerClient(self.mp_queue))
            with timer.expires(after=0.5):
                time.sleep(0.1)

        def test_get_timer_recursive(self):
            """
            If a function acquires a countdown timer with default scope,
            then recursive calls to the function should re-acquire the
            timer rather than creating a new one. That is only the last
            recursive call's timer will take effect.
            """
            self.server.start()
            timer.configure(timer.LocalTimerClient(self.mp_queue))

            # func should not time out
            def func(n):
                if n > 0:
                    with timer.expires(after=0.1):
                        func(n - 1)
                        time.sleep(0.05)

            func(4)

            p = self.ctx.Process(target=func2, args=(2, self.mp_queue))
            p.start()
            p.join()
            self.assertEqual(-signal.SIGKILL, p.exitcode)

        @staticmethod
        def _run(mp_queue, timeout, duration):
            client = timer.LocalTimerClient(mp_queue)
            timer.configure(client)

            with timer.expires(after=timeout):
                time.sleep(duration)

        @unittest.skipIf(TEST_WITH_TSAN, "test is tsan incompatible")
        def test_timer(self):
            timeout = 0.1
            duration = 1
            p = mp.Process(target=self._run, args=(self.mp_queue, timeout, duration))
            p.start()
            p.join()
            self.assertEqual(-signal.SIGKILL, p.exitcode)

    def _enqueue_on_interval(mp_queue, n, interval, sem):
        """
        enqueues ``n`` timer requests into ``mp_queue`` one element per
        interval seconds. Releases the given semaphore once before going to work.
        """
        sem.release()
        for i in range(0, n):
            mp_queue.put(TimerRequest(i, "test_scope", 0))
            time.sleep(interval)


# timer is not supported on windows or macos
if not (IS_WINDOWS or IS_MACOS or TEST_WITH_DEV_DBG_ASAN):

    class MultiprocessingRequestQueueTest(TestCase):
        def test_get(self):
            mp_queue = mp.Queue()
            request_queue = MultiprocessingRequestQueue(mp_queue)

            requests = request_queue.get(1, timeout=0.01)
            self.assertEqual(0, len(requests))

            request = TimerRequest(1, "test_scope", 0)
            mp_queue.put(request)
            requests = request_queue.get(2, timeout=0.01)
            self.assertEqual(1, len(requests))
            self.assertIn(request, requests)

        @unittest.skipIf(
            TEST_WITH_TSAN,
            "test incompatible with tsan",
        )
        def test_get_size(self):
            """
            Creates a "producer" process that enqueues ``n`` elements
            every ``interval`` seconds. Asserts that a ``get(n, timeout=n*interval+delta)``
            yields all ``n`` elements.
            """
            mp_queue = mp.Queue()
            request_queue = MultiprocessingRequestQueue(mp_queue)
            n = 10
            interval = 0.1
            sem = mp.Semaphore(0)

            p = mp.Process(
                target=_enqueue_on_interval, args=(mp_queue, n, interval, sem)
            )
            p.start()

            sem.acquire()  # blocks until the process has started to run the function
            timeout = interval * (n + 1)
            start = time.time()
            requests = request_queue.get(n, timeout=timeout)
            self.assertLessEqual(time.time() - start, timeout + interval)
            self.assertEqual(n, len(requests))

        def test_get_less_than_size(self):
            """
            Tests slow producer.
            Creates a "producer" process that enqueues ``n`` elements
            every ``interval`` seconds. Asserts that a ``get(n, timeout=(interval * n/2))``
            yields at most ``n/2`` elements.
            """
            mp_queue = mp.Queue()
            request_queue = MultiprocessingRequestQueue(mp_queue)
            n = 10
            interval = 0.1
            sem = mp.Semaphore(0)

            p = mp.Process(
                target=_enqueue_on_interval, args=(mp_queue, n, interval, sem)
            )
            p.start()

            sem.acquire()  # blocks until the process has started to run the function
            requests = request_queue.get(n, timeout=(interval * (n / 2)))
            self.assertLessEqual(n / 2, len(requests))


# timer is not supported on windows or macos
if not (IS_WINDOWS or IS_MACOS or TEST_WITH_DEV_DBG_ASAN):

    class LocalTimerServerTest(TestCase):
        def setUp(self):
            super().setUp()
            self.mp_queue = mp.Queue()
            self.max_interval = 0.01
            self.server = timer.LocalTimerServer(self.mp_queue, self.max_interval)

        def tearDown(self):
            super().tearDown()
            self.server.stop()

        def test_watchdog_call_count(self):
            """
            checks that the watchdog function ran wait/interval +- 1 times
            """
            self.server._run_watchdog = mock.MagicMock(wraps=self.server._run_watchdog)

            wait = 0.1

            self.server.start()
            time.sleep(wait)
            self.server.stop()
            watchdog_call_count = self.server._run_watchdog.call_count
            self.assertGreaterEqual(
                watchdog_call_count, int(wait / self.max_interval) - 1
            )
            self.assertLessEqual(watchdog_call_count, int(wait / self.max_interval) + 1)

        def test_watchdog_empty_queue(self):
            """
            checks that the watchdog can run on an empty queue
            """
            self.server._run_watchdog()

        def _expired_timer(self, pid, scope):
            expired = time.time() - 60
            return TimerRequest(worker_id=pid, scope_id=scope, expiration_time=expired)

        def _valid_timer(self, pid, scope):
            valid = time.time() + 60
            return TimerRequest(worker_id=pid, scope_id=scope, expiration_time=valid)

        def _release_timer(self, pid, scope):
            return TimerRequest(worker_id=pid, scope_id=scope, expiration_time=-1)

        @mock.patch("os.kill")
        def test_expired_timers(self, mock_os_kill):
            """
            tests that a single expired timer on a process should terminate
            the process and clean up all pending timers that was owned by the process
            """
            test_pid = -3
            self.mp_queue.put(self._expired_timer(pid=test_pid, scope="test1"))
            self.mp_queue.put(self._valid_timer(pid=test_pid, scope="test2"))

            self.server._run_watchdog()

            self.assertEqual(0, len(self.server._timers))
            mock_os_kill.assert_called_once_with(test_pid, signal.SIGKILL)

        @mock.patch("os.kill")
        def test_acquire_release(self, mock_os_kill):
            """
            tests that:
            1. a timer can be acquired then released (should not terminate process)
            2. a timer can be vacuously released (e.g. no-op)
            """
            test_pid = -3
            self.mp_queue.put(self._valid_timer(pid=test_pid, scope="test1"))
            self.mp_queue.put(self._release_timer(pid=test_pid, scope="test1"))
            self.mp_queue.put(self._release_timer(pid=test_pid, scope="test2"))

            self.server._run_watchdog()

            self.assertEqual(0, len(self.server._timers))
            mock_os_kill.assert_not_called()

        @mock.patch("os.kill")
        def test_valid_timers(self, mock_os_kill):
            """
            tests that valid timers are processed correctly and the process is left alone
            """
            self.mp_queue.put(self._valid_timer(pid=-3, scope="test1"))
            self.mp_queue.put(self._valid_timer(pid=-3, scope="test2"))
            self.mp_queue.put(self._valid_timer(pid=-2, scope="test1"))
            self.mp_queue.put(self._valid_timer(pid=-2, scope="test2"))

            self.server._run_watchdog()

            self.assertEqual(4, len(self.server._timers))
            self.assertTrue((-3, "test1") in self.server._timers)
            self.assertTrue((-3, "test2") in self.server._timers)
            self.assertTrue((-2, "test1") in self.server._timers)
            self.assertTrue((-2, "test2") in self.server._timers)
            mock_os_kill.assert_not_called()


if __name__ == "__main__":
    run_tests()
