# Owner(s): ["oncall: r2p"]

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing as mp
import os
import signal
import tempfile
import time
import unittest
import unittest.mock as mock

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
    def func2(n, file_path, scope_id):
        if file_path is not None:
            timer.configure(timer.FileTimerClient(file_path, 0.1, scope_id))
        if n > 0:
            with timer.expires(after=0.1):
                func2(n - 1, None, None)
                time.sleep(0.2)

    class FileTimerTest(TestCase):
        def setUp(self):
            super().setUp()
            self.max_interval = 0.01
            self.num_clients = 10
            td = tempfile.mkdtemp(prefix="test-watchdog")
            self.server = timer.FileTimerServer(td, self.num_clients, self.max_interval)
            self.scope_ids = [f"rank{i}" for i in range(self.num_clients)]
            self.file_paths = [f"{td}/{scope}" for scope in self.scope_ids]
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
            timer_client = timer.FileTimerClient(self.file_paths[0], 1, self.scope_ids[0])
            timer_client.acquire = mock.MagicMock(wraps=timer_client.acquire)
            timer_client.release = mock.MagicMock(wraps=timer_client.release)
            with timer.expires(after=1, scope="test", client=timer_client):
                pass

            timer_client.acquire.assert_called_once_with("test", mock.ANY)
            timer_client.release.assert_called_once_with("test")

        def test_happy_path(self):
            timer.configure(timer.FileTimerClient(self.file_paths[0], 0.5, self.scope_ids[0]))
            with timer.expires(after=0.5):
                time.sleep(0.1)

        def test_get_timer_recursive(self):
            timer_client = timer.FileTimerClient(self.file_paths[0], 1, self.scope_ids[0])

            # func should not time out
            def func(n):
                if n > 0:
                    print("here...", n)
                    timer_client.acquire("test", 1)
                    time.sleep(0.5)
                    func(n - 1)

            func(4)
            self.server.stop()

        def test_get_timer_recursive_process(self):
            p = mp.Process(target=func2, args=(2, self.file_paths[0], self.scope_ids[0]))
            p.start()
            p.join()
            self.assertEqual(-signal.SIGKILL, p.exitcode)

        def test_multiple_clients_interaction(self):
            # func should not time out
            def func(n, file_path, scope_id):
                if file_path is not None:
                    timer.configure(timer.FileTimerClient(file_path, 100, scope_id))
                if n > 0:
                    with timer.expires(after=100):
                        func(n - 1, None)
                        time.sleep(0.01)

            num_requests_per_client = 10
            processes = []
            for i in range(self.num_clients):
                p = mp.Process(target=func, args=(num_requests_per_client, self.file_paths[i], self.scope_ids[i]))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            self.server.run_once()  # Allows the server to process all requests
            self.assertEqual(self.num_clients, len(self.server._timers))

        @staticmethod
        def _run(file_path, scope_id, timeout, duration):
            client = timer.FileTimerClient(file_path, timeout, scope_id)
            timer.configure(client)
            with timer.expires(after=timeout):
                time.sleep(duration)

        @unittest.skipIf(TEST_WITH_TSAN, "test is tsan incompatible")
        def test_timer(self):
            timeout = 0.1
            duration = 1
            p = mp.Process(target=self._run, args=(self.file_paths[0], self.scope_ids[0], timeout, duration))
            p.start()
            p.join()
            self.assertEqual(-signal.SIGKILL, p.exitcode)


    class FileTimerClientTest(TestCase):
        def test_send_request_without_server(self):
            with self.assertRaises(FileNotFoundError):
                client = timer.FileTimerClient("/tmp/watchdognotexist/test_file", 0.1, "test")
                timer.configure(client)
                with timer.expires(after=0.1):
                    time.sleep(0.1)


    class FileTimerServerTest(TestCase):
        def setUp(self):
            super().setUp()
            td = tempfile.mkdtemp(prefix="test-watchdog")
            self.max_interval = 0.01
            self.num_clients = 4
            self.server = timer.FileTimerServer(td, self.num_clients, self.max_interval)
            self.scope_ids = [f"rank{i}" for i in range(self.num_clients)]
            self.file_paths = [f"{td}/{scope}" for scope in self.scope_ids]

        def tearDown(self):
            super().tearDown()
            self.server.stop()

        def test_watchdog_call_count(self):
            """
            checks that the watchdog function ran wait/interval +- 1 times
            """
            self.server._run_watchdog = mock.MagicMock(wraps=self.server._run_watchdog)
            self.server.start()

            client = timer.FileTimerClient(self.file_paths[0], 1, self.scope_ids[0])
            client.acquire("test0", 60)

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
            checks that the watchdog can run on an empty file
            """
            self.server.start()

        @mock.patch("os.kill")
        def test_expired_timers(self, mock_os_kill):
            """
            tests that a single expired timer on a process should terminate
            the process and clean up all pending timers that was owned by the process
            """
            self.server.start()

            client1 = timer.FileTimerClient(self.file_paths[0], 1, self.scope_ids[0])
            client2 = timer.FileTimerClient(self.file_paths[1], 2, self.scope_ids[1])

            client1.acquire("test1", 1)
            client2.acquire("test2", 1)
            time.sleep(1.5)

            self.server.run_once()  # Allows the server to process all requests
            print(self.server._timers)
            self.assertEqual(1, len(self.server._timers))
            mock_os_kill.assert_called_once_with(os.getpid(), signal.SIGKILL)

        @mock.patch("os.kill")
        def test_valid_timers(self, mock_os_kill):
            """
            tests that valid timers are processed correctly and the process is left alone
            """
            self.server.start()

            client1 = timer.FileTimerClient(self.file_paths[0], 60, self.scope_ids[0])
            client2 = timer.FileTimerClient(self.file_paths[1], 60, self.scope_ids[1])
            client3 = timer.FileTimerClient(self.file_paths[2], 60, self.scope_ids[2])
            client4 = timer.FileTimerClient(self.file_paths[3], 60, self.scope_ids[3])

            self.server.run_once()  # Allows the server to process all requests
            self.assertEqual(4, len(self.server._timers))
            for scope in self.scope_ids:
                self.assertTrue(scope in self.server._timers)
            mock_os_kill.assert_not_called()


if __name__ == "__main__":
    run_tests()
