# Owner(s): ["oncall: r2p"]

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing as mp
import os
import signal
import time
import unittest
import unittest.mock as mock
import uuid

import torch.distributed.elastic.timer as timer
from torch.testing._internal.common_utils import (
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    TEST_WITH_TSAN,
    TestCase,
)


# timer is not supported on windows or macos
if not (IS_WINDOWS or IS_MACOS):
    # func2 should time out
    def func2(n, file_path):
        if file_path is not None:
            timer.configure(timer.FileTimerClient(file_path))
        if n > 0:
            with timer.expires(after=0.1):
                func2(n - 1, None)
                time.sleep(0.2)

    class FileTimerTest(TestCase):
        def setUp(self):
            super().setUp()
            self.max_interval = 0.01
            self.file_path = f"/tmp/test_file_path_{os.getpid()}_{uuid.uuid4()}"
            self.server = timer.FileTimerServer(
                self.file_path, "test", self.max_interval
            )
            self.server.start()

        def tearDown(self):
            super().tearDown()
            self.server.stop()

        def test_exception_propagation(self):
            with self.assertRaises(RuntimeError, msg="foobar"):
                with timer.expires(after=1):
                    raise RuntimeError("foobar")

        def test_no_client(self):
            # no timer client configured; exception expected
            timer.configure(None)
            with self.assertRaises(RuntimeError):
                with timer.expires(after=1):
                    pass

        def test_client_interaction(self):
            # no timer client configured but one passed in explicitly
            # no exception expected
            timer_client = timer.FileTimerClient(self.file_path)
            timer_client.acquire = mock.MagicMock(wraps=timer_client.acquire)
            timer_client.release = mock.MagicMock(wraps=timer_client.release)
            with timer.expires(after=1, scope="test", client=timer_client):
                pass

            timer_client.acquire.assert_called_once_with("test", mock.ANY)
            timer_client.release.assert_called_once_with("test")

        def test_happy_path(self):
            timer.configure(timer.FileTimerClient(self.file_path))
            with timer.expires(after=0.5):
                time.sleep(0.1)

        def test_get_timer_recursive(self):
            """
            If a function acquires a countdown timer with default scope,
            then recursive calls to the function should re-acquire the
            timer rather than creating a new one. That is only the last
            recursive call's timer will take effect.
            """
            timer.configure(timer.FileTimerClient(self.file_path))

            # func should not time out
            def func(n):
                if n > 0:
                    with timer.expires(after=0.1):
                        func(n - 1)
                        time.sleep(0.05)

            func(4)

            p = mp.Process(target=func2, args=(2, self.file_path))
            p.start()
            p.join()
            self.assertEqual(-signal.SIGKILL, p.exitcode)

        def test_multiple_clients_interaction(self):
            # func should not time out
            def func(n, file_path):
                if file_path is not None:
                    timer.configure(timer.FileTimerClient(file_path))
                if n > 0:
                    with timer.expires(after=100):
                        func(n - 1, None)
                        time.sleep(0.01)

            num_clients = 10
            num_requests_per_client = 10
            processes = []
            for _ in range(num_clients):
                p = mp.Process(
                    target=func, args=(num_requests_per_client, self.file_path)
                )
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            self.server.run_once()  # Allows the server to process all requests
            self.assertEqual(
                2 * num_clients * num_requests_per_client, self.server._request_count
            )

        @mock.patch("torch.distributed.elastic.timer.FileTimerServer._reap_worker")
        def test_exit_before_release(self, mock_reap):
            def func1(file_path):
                client = timer.FileTimerClient(file_path)
                timer.configure(client)
                expire = time.time() + 2
                client.acquire("test_scope", expire)
                time.sleep(1)

            p = mp.Process(target=func1, args=(self.file_path,))
            p.start()
            p.join()

            time.sleep(2)
            self.server.run_once()  # Allows the server to process all requests
            mock_reap.assert_not_called()
            self.assertEqual(0, len(self.server._timers))

        @mock.patch("torch.distributed.elastic.timer.FileTimerServer._reap_worker")
        @mock.patch(
            "torch.distributed.elastic.timer.FileTimerServer.is_process_running"
        )
        def test_exit_before_release_reap(self, mock_pid_exists, mock_reap):
            def func1(file_path):
                client = timer.FileTimerClient(file_path)
                timer.configure(client)
                expire = time.time() + 2
                client.acquire("test_scope", expire)
                time.sleep(1)

            mock_pid_exists.return_value = True
            p = mp.Process(target=func1, args=(self.file_path,))
            p.start()
            p.join()

            time.sleep(2)
            self.server.run_once()  # Allows the server to process all requests
            mock_reap.assert_called()
            self.assertEqual(0, len(self.server._timers))

        @staticmethod
        def _run(file_path, timeout, duration):
            client = timer.FileTimerClient(file_path)
            timer.configure(client)
            with timer.expires(after=timeout):
                time.sleep(duration)

        @unittest.skipIf(TEST_WITH_TSAN, "test is tsan incompatible")
        def test_timer(self):
            timeout = 0.1
            duration = 1
            p = mp.Process(target=self._run, args=(self.file_path, timeout, duration))
            p.start()
            p.join()
            self.assertEqual(-signal.SIGKILL, p.exitcode)

    def _request_on_interval(file_path, n, interval, sem):
        """
        enqueues ``n`` timer requests into ``mp_queue`` one element per
        interval seconds. Releases the given semaphore once before going to work.
        """
        client = timer.FileTimerClient(file_path)
        sem.release()
        for _ in range(0, n):
            client.acquire("test_scope", 0)
            time.sleep(interval)

    class FileTimerClientTest(TestCase):
        def test_send_request_without_server(self):
            client = timer.FileTimerClient("test_file")
            timer.configure(client)
            with self.assertRaises(BrokenPipeError):
                with timer.expires(after=0.1):
                    time.sleep(0.1)

    class FileTimerServerTest(TestCase):
        def setUp(self):
            super().setUp()
            self.file_path = f"/tmp/test_file_path_{os.getpid()}_{uuid.uuid4()}"
            self.max_interval = 0.01
            self.server = timer.FileTimerServer(
                self.file_path, "test", self.max_interval
            )

        def tearDown(self):
            super().tearDown()
            self.server.stop()

        def test_watchdog_call_count(self):
            """
            checks that the watchdog function ran wait/interval +- 1 times
            """
            self.server._run_watchdog = mock.MagicMock(wraps=self.server._run_watchdog)
            self.server.start()

            test_pid = -3
            client = timer.FileTimerClient(self.file_path)
            client._send_request(self._valid_timer(pid=test_pid, scope="test0"))

            wait = 0.1
            time.sleep(wait)
            self.server.stop()
            watchdog_call_count = self.server._run_watchdog.call_count
            self.assertGreaterEqual(
                watchdog_call_count, int(wait / self.max_interval) - 1
            )
            self.assertLessEqual(watchdog_call_count, int(wait / self.max_interval) + 1)

        def test_watchdog_empty_queue(self):
            """
            checks that the watchdog can run on an empty pipe
            """
            self.server.start()

        def _expired_timer(self, pid, scope):
            expired = time.time() - 60
            return timer.FileTimerRequest(
                worker_pid=pid,
                scope_id=scope,
                expiration_time=expired,
                signal=signal.SIGKILL,
            )

        def _valid_timer(self, pid, scope):
            valid = time.time() + 60
            return timer.FileTimerRequest(
                worker_pid=pid,
                scope_id=scope,
                expiration_time=valid,
                signal=signal.SIGKILL,
            )

        def _release_timer(self, pid, scope):
            return timer.FileTimerRequest(
                worker_pid=pid, scope_id=scope, expiration_time=-1
            )

        @mock.patch("os.kill")
        @mock.patch(
            "torch.distributed.elastic.timer.file_based_local_timer.log_debug_info_for_expired_timers"
        )
        def test_expired_timers(self, mock_debug_info, mock_os_kill):
            """
            tests that a single expired timer on a process should terminate
            the process and clean up all pending timers that was owned by the process
            """
            self.server.start()

            test_pid = -3
            client = timer.FileTimerClient(self.file_path)
            client._send_request(self._expired_timer(pid=test_pid, scope="test1"))
            client._send_request(self._valid_timer(pid=test_pid, scope="test2"))

            self.server.run_once()  # Allows the server to process all requests
            self.assertEqual(0, len(self.server._timers))
            mock_os_kill.assert_called_once_with(test_pid, signal.SIGKILL)
            mock_debug_info.assert_called()

        @mock.patch("os.kill")
        def test_send_request_release(self, mock_os_kill):
            """
            tests that:
            1. a timer can be acquired then released (should not terminate process)
            2. a timer can be vacuously released (e.g. no-op)
            """
            self.server.start()

            client = timer.FileTimerClient(self.file_path)
            test_pid = -3
            client._send_request(self._valid_timer(pid=test_pid, scope="test1"))
            client._send_request(self._release_timer(pid=test_pid, scope="test1"))
            client._send_request(self._release_timer(pid=test_pid, scope="test2"))

            self.assertEqual(0, len(self.server._timers))
            mock_os_kill.assert_not_called()

        @mock.patch(
            "torch.distributed.elastic.timer.FileTimerServer.is_process_running"
        )
        @mock.patch("os.kill")
        def test_valid_timers(self, mock_os_kill, mock_pid_exists):
            """
            tests that valid timers are processed correctly and the process is left alone
            """
            self.server.start()
            mock_pid_exists.return_value = True

            client = timer.FileTimerClient(self.file_path)
            client._send_request(self._valid_timer(pid=-3, scope="test1"))
            client._send_request(self._valid_timer(pid=-3, scope="test2"))
            client._send_request(self._valid_timer(pid=-2, scope="test1"))
            client._send_request(self._valid_timer(pid=-2, scope="test2"))

            self.server.run_once()  # Allows the server to process all requests
            self.assertEqual(4, len(self.server._timers))
            self.assertTrue((-3, "test1") in self.server._timers)
            self.assertTrue((-3, "test2") in self.server._timers)
            self.assertTrue((-2, "test1") in self.server._timers)
            self.assertTrue((-2, "test2") in self.server._timers)
            mock_os_kill.assert_not_called()


if __name__ == "__main__":
    run_tests()
