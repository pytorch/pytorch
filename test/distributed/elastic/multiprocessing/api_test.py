#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import ctypes
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import time
import unittest
from itertools import product
from typing import Dict, List
from unittest import mock
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing import ProcessFailure, start_processes
from torch.distributed.elastic.multiprocessing.api import (
    MultiprocessContext,
    RunProcsResult,
    Std,
    _validate_full_rank,
    to_map,
    _wrap,
)
from torch.distributed.elastic.multiprocessing.errors.error_handler import _write_error
from torch.testing._internal.common_utils import (
    NO_MULTIPROCESSING_SPAWN,
    TEST_WITH_ASAN,
    TEST_WITH_TSAN,
    IS_PYTORCH_CI,
    IS_WINDOWS,
    IS_MACOS,
)
from torch.testing._internal.common_utils import run_tests


class RunProcResultsTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_is_failed(self):
        pr_success = RunProcsResult(return_values={0: "a", 1: "b"})
        self.assertFalse(pr_success.is_failed())

        fail0 = ProcessFailure(
            local_rank=0, pid=998, exitcode=1, error_file="ignored.json"
        )
        pr_fail = RunProcsResult(failures={0: fail0})
        self.assertTrue(pr_fail.is_failed())

    @patch("torch.distributed.elastic.multiprocessing.errors.log")
    def test_get_failures(self, log_mock):
        with mock.patch("time.time", side_effect=[3, 2, 1]):
            error_file0 = os.path.join(self.test_dir, "error0.json")
            error_file1 = os.path.join(self.test_dir, "error1.json")
            _write_error(RuntimeError("error 0"), error_file0)
            _write_error(RuntimeError("error 1"), error_file1)

            fail0 = ProcessFailure(
                local_rank=0, pid=997, exitcode=1, error_file=error_file0
            )
            fail1 = ProcessFailure(
                local_rank=1, pid=998, exitcode=3, error_file=error_file1
            )
            fail2 = ProcessFailure(
                local_rank=2, pid=999, exitcode=15, error_file="no_exist.json"
            )

            self.assertEqual(3, fail0.timestamp)
            self.assertEqual(2, fail1.timestamp)
            self.assertEqual(1, fail2.timestamp)


class StdTest(unittest.TestCase):
    def test_from_value(self):
        self.assertEqual(Std.NONE, Std.from_str("0"))
        self.assertEqual(Std.OUT, Std.from_str("1"))
        self.assertEqual(Std.ERR, Std.from_str("2"))
        self.assertEqual(Std.ALL, Std.from_str("3"))

    def test_from_value_map(self):
        self.assertEqual({0: Std.OUT}, Std.from_str("0:1"))
        self.assertEqual({0: Std.OUT, 1: Std.OUT}, Std.from_str("0:1,1:1"))

    def test_from_str_bad_input(self):
        bad_inputs = ["0:1,", "11", "0:1,1", "1,0:1"]
        for bad in bad_inputs:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Std.from_str(bad)


def echo0(msg: str) -> None:
    """
    void function
    """
    print(msg)


def echo1(msg: str, exitcode: int = 0) -> str:
    """
    returns ``msg`` or exits with the given exitcode (if nonzero)
    """

    rank = int(os.environ["RANK"])
    if exitcode != 0:
        print(f"exit {exitcode} from {rank}", file=sys.stderr)
        sys.exit(exitcode)
    else:
        print(f"{msg} stdout from {rank}")
        print(f"{msg} stderr from {rank}", file=sys.stderr)
        return f"{msg}_{rank}"


def echo2(msg: str, fail: bool = False) -> str:
    """
    returns ``msg`` or raises a RuntimeError if ``fail`` is set
    """
    if fail:
        raise RuntimeError(msg)
    return msg


def echo_large(size: int) -> Dict[int, str]:
    """
    returns a large output ({0: test0", 1: "test1", ..., (size-1):f"test{size-1}"})
    """
    out = {}
    for idx in range(0, size):
        out[idx] = f"test{idx}"
    return out


def echo3(msg: str, fail: bool = False) -> str:
    """
    returns ``msg`` or induces a SIGSEGV if ``fail`` is set
    """
    if fail:
        ctypes.string_at(0)
    return msg


def dummy_compute() -> torch.Tensor:
    """
    returns a predefined size random Tensor
    """
    return torch.rand(100, 100)


def redirects_oss_test() -> List[Std]:
    return [
        Std.NONE,
    ]


def redirects_all() -> List[Std]:
    return [
        Std.NONE,
        Std.OUT,
        Std.ERR,
        Std.ALL,
    ]


@unittest.skipIf(
    TEST_WITH_ASAN or TEST_WITH_TSAN or IS_WINDOWS or IS_MACOS,
    "tests incompatible with tsan or asan",
)
class StartProcessesTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")

        if NO_MULTIPROCESSING_SPAWN:  # python 2.7 doesn't have spawn
            self._start_methods = ["fork"]
        else:
            self._start_methods = ["fork", "spawn"]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def log_dir(self):
        return tempfile.mkdtemp(dir=self.test_dir)

    def assert_in_file(self, expected: List[str], filename: str) -> None:
        expected = [f"{line.rstrip()}\n" for line in expected]
        with open(filename, "r") as fp:
            actual = fp.readlines()
            for line in expected:
                self.assertIn(line, actual)

    def assert_pids_noexist(self, pids: Dict[int, int]):
        for local_rank, pid in pids.items():
            with self.assertRaises(
                OSError, msg=f"local_rank: {local_rank} pid: {pid} should not exist"
            ):
                os.kill(pid, 0)

    def test_to_map(self):
        local_world_size = 2
        self.assertEqual({0: Std.OUT, 1: Std.OUT}, to_map(Std.OUT, local_world_size))
        self.assertEqual(
            {0: Std.NONE, 1: Std.OUT}, to_map({1: Std.OUT}, local_world_size)
        )
        self.assertEqual(
            {0: Std.ERR, 1: Std.OUT}, to_map({0: Std.ERR, 1: Std.OUT}, local_world_size)
        )

    def test_invalid_log_dir(self):
        with tempfile.NamedTemporaryFile(dir=self.test_dir) as not_a_dir:
            cases = {
                "does_not_exist": FileNotFoundError,
                not_a_dir.name: NotADirectoryError,
                # test_dir is not empty since we touched not_a_dir file
                self.test_dir: RuntimeError,
            }

            for (log_dir, expected_error) in cases.items():
                with self.subTest(log_dir=log_dir, expected_error=expected_error):
                    with self.assertRaises(expected_error):
                        start_processes(
                            name="echo",
                            entrypoint=echo1,
                            args={0: ("hello",)},
                            envs={0: {"RANK": "0"}},
                            log_dir=log_dir,
                        )

    def test_args_env_len_mismatch(self):
        cases = [
            # 1 x args; 2 x envs
            {
                "args": {0: ("hello",)},
                "envs": {0: {"RANK": "0"}, 1: {"RANK": "1"}},
            },
            # 2 x args; 1 x envs
            {
                "args": {0: ("hello",), 1: ("world",)},
                "envs": {0: {"RANK": "0"}},
            },
        ]

        for kwds in cases:
            args = kwds["args"]
            envs = kwds["envs"]
            with self.subTest(args=args, envs=envs):
                with self.assertRaises(RuntimeError):
                    start_processes(
                        name="echo",
                        entrypoint=echo1,
                        args=args,
                        envs=envs,
                        log_dir=self.log_dir(),
                    )

    def test_pcontext_wait(self):
        pc = start_processes(
            name="sleep",
            entrypoint=time.sleep,
            args={0: (1,)},
            envs={0: {}},
            log_dir=self.log_dir(),
            start_method="fork",
        )

        self.assertIsNone(pc.wait(timeout=0.1, period=0.01))
        self.assertIsNotNone(pc.wait(period=0.1))
        self.assertTrue(pc._stderr_tail.stopped())
        self.assertTrue(pc._stdout_tail.stopped())

    def test_multiprocess_context_close(self):
        pc = start_processes(
            name="sleep",
            entrypoint=time.sleep,
            args={0: (1,)},
            envs={0: {}},
            log_dir=self.log_dir(),
            start_method="fork",
        )

        pids = pc.pids()
        pc.close()
        self.assert_pids_noexist(pids)
        self.assertTrue(pc._stderr_tail.stopped())
        self.assertTrue(pc._stdout_tail.stopped())

    def test_function_with_tensor(self):
        for start_method in self._start_methods:
            pc = start_processes(
                name="dummy_compute",
                entrypoint=dummy_compute,
                args={},
                envs={},
                log_dir=self.log_dir(),
                start_method=start_method,
            )

            results = pc.wait()
            self.assert_pids_noexist(pc.pids())
            for return_value in results.return_values.values():
                self.assertIsInstance(return_value, torch.Tensor)
                self.assertEqual((100, 100), return_value.shape)

    def test_void_function(self):
        for start_method in self._start_methods:
            with self.subTest(start_method=start_method):
                pc = start_processes(
                    name="echo",
                    entrypoint=echo0,
                    args={0: ("hello",), 1: ("world",)},
                    envs={0: {}, 1: {}},
                    log_dir=self.log_dir(),
                    start_method=start_method,
                )

                results = pc.wait(period=0.1)
                self.assertEqual({0: None, 1: None}, results.return_values)

    @unittest.skipIf(
        TEST_WITH_ASAN or TEST_WITH_TSAN, "tests incompatible with tsan or asan"
    )
    def test_function_large_ret_val(self):
        # python multiprocessing.queue module uses pipes and actually PipedQueues
        # This means that if a single object is greater than a pipe size
        # the writer process will block until reader process will start
        # reading the pipe.
        # This test makes a worker fn to return huge output, around ~10 MB

        size = 200000
        for start_method in self._start_methods:
            with self.subTest(start_method=start_method):
                pc = start_processes(
                    name="echo",
                    entrypoint=echo_large,
                    args={0: (size,), 1: (size,), 2: (size,), 3: (size,)},
                    envs={0: {}, 1: {}, 2: {}, 3: {}},
                    log_dir=self.log_dir(),
                    start_method=start_method,
                )

                results = pc.wait(period=0.1)
                for i in range(pc.nprocs):
                    self.assertEqual(size, len(results.return_values[i]))

    def test_function_raise(self):
        """
        run 2x copies of echo2, raise an exception on the first
        """
        RAISE = True

        for start_method in self._start_methods:
            with self.subTest(start_method=start_method):
                log_dir = self.log_dir()
                pc = start_processes(
                    name="echo",
                    entrypoint=echo2,
                    args={0: ("hello", RAISE), 1: ("world",)},
                    envs={0: {}, 1: {}},
                    log_dir=log_dir,
                    start_method=start_method,
                )

                results = pc.wait(period=0.1)

                self.assert_pids_noexist(pc.pids())
                self.assertEqual(1, len(results.failures))
                self.assertFalse(results.return_values)

                failure = results.failures[0]
                error_file = failure.error_file
                error_file_data = failure.error_file_data

                self.assertEqual(1, failure.exitcode)
                self.assertEqual("<N/A>", failure.signal_name())
                self.assertEqual(pc.pids()[0], failure.pid)
                self.assertEqual(os.path.join(log_dir, "0", "error.json"), error_file)
                self.assertEqual(
                    int(error_file_data["message"]["extraInfo"]["timestamp"]),
                    int(failure.timestamp),
                )
                self.assertTrue(pc._stderr_tail.stopped())
                self.assertTrue(pc._stdout_tail.stopped())

    ########################################
    # start_processes as binary tests
    ########################################

    def bin(self, name: str):
        dir = os.path.dirname(__file__)
        return os.path.join(dir, "bin", name)

    def test_binary_exit(self):
        FAIL = 138
        pc = start_processes(
            name="echo",
            entrypoint=self.bin("echo1.py"),
            args={0: ("--exitcode", FAIL, "foo"), 1: ("--exitcode", 0, "bar")},
            envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
            log_dir=self.log_dir(),
            redirects={0: Std.ALL},
        )

        results = pc.wait(period=0.1)

        self.assertTrue(results.is_failed())
        self.assertEqual(1, len(results.failures))

        failure = results.failures[0]
        self.assertEqual(138, failure.exitcode)
        self.assertEqual("<N/A>", failure.signal_name())
        self.assertEqual("<NONE>", failure.error_file_data["message"])
        self.assert_in_file([f"exit {FAIL} from 0"], results.stderrs[0])
        self.assert_in_file([], results.stdouts[0])
        self.assertFalse(results.stderrs[1])
        self.assertFalse(results.stdouts[1])
        self.assertTrue(pc._stderr_tail.stopped())
        self.assertTrue(pc._stdout_tail.stopped())

    def test_binary_raises(self):
        pc = start_processes(
            name="echo",
            entrypoint=self.bin("echo2.py"),
            args={0: ("--raises", "true", "foo"), 1: ("bar",)},
            envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
            log_dir=self.log_dir(),
        )

        results = pc.wait(period=0.1)

        self.assert_pids_noexist(pc.pids())
        self.assertTrue(results.is_failed())
        self.assertEqual(1, len(results.failures))

        failure = results.failures[0]
        self.assertEqual(1, failure.exitcode)
        self.assertEqual("<NONE>", failure.error_file_data["message"])
        self.assertEqual("<N/A>", failure.signal_name())

    def test_binary_incorrect_entrypoint(self):
        with self.assertRaises(FileNotFoundError):
            start_processes(
                name="echo",
                entrypoint="does_not_exist.py",
                args={0: ("foo"), 1: ("bar",)},
                envs={0: {}, 1: {}},
                log_dir=self.log_dir(),
            )

    def test_validate_full_rank(self):
        with self.assertRaises(RuntimeError):
            _validate_full_rank({}, 10, "")

    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN,
        "Disabled for environments that \
                     don't support multiprocessing with spawn start method",
    )
    def test_multiprocessing_context_poll_raises_exception(self):
        mp_context = MultiprocessContext(
            name="test_mp",
            entrypoint=echo0,
            args={0: (0, 1)},
            envs={},
            stdouts={0: {}},
            stderrs={0: {}},
            tee_stdouts={0: "tee_stdout"},
            tee_stderrs={0: "tee_stderr"},
            error_files={0: "test_file"},
            start_method="spawn",
        )
        mp_context._pc = mock.Mock()
        # Using mock since we cannot just set exitcode on process
        mock_process = mock.Mock()
        mock_process.exitcode = -1
        mp_context._pc.processes = [mock_process]
        e = mp.ProcessRaisedException(msg="test msg", error_index=0, error_pid=123)
        mp_context._pc.join.side_effect = e
        with mock.patch.object(mp_context, "close"):
            run_result = mp_context._poll()
            self.assertEqual(1, len(run_result.failures))
            failure = run_result.failures[0]
            self.assertEqual("Signal 1 (SIGHUP) received by PID 123", failure.message)


@unittest.skipIf(
    TEST_WITH_ASAN or TEST_WITH_TSAN or IS_WINDOWS or IS_MACOS,
    "tests incompatible with tsan or asan, the redirect functionality does not work on macos or windows",
)
class StartProcessesListTest(StartProcessesTest):
    ########################################
    # start_processes as binary tests
    ########################################
    def test_function(self):
        for start_method, redirs in product(self._start_methods, redirects_oss_test()):
            with self.subTest(start_method=start_method, redirs=redirs):
                pc = start_processes(
                    name="echo",
                    entrypoint=echo1,
                    args={0: ("hello",), 1: ("hello",)},
                    envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                    log_dir=self.log_dir(),
                    start_method=start_method,
                    redirects=redirs,
                )

                results = pc.wait(period=0.1)
                nprocs = pc.nprocs

                self.assert_pids_noexist(pc.pids())
                self.assertEqual(
                    {i: f"hello_{i}" for i in range(nprocs)}, results.return_values
                )

                for i in range(nprocs):
                    if redirs & Std.OUT != Std.OUT:
                        self.assertFalse(results.stdouts[i])
                    if redirs & Std.ERR != Std.ERR:
                        self.assertFalse(results.stderrs[i])
                    if redirs & Std.OUT == Std.OUT:
                        self.assert_in_file(
                            [f"hello stdout from {i}"], results.stdouts[i]
                        )
                    if redirs & Std.ERR == Std.ERR:
                        self.assert_in_file(
                            [f"hello stderr from {i}"], results.stderrs[i]
                        )

    def test_binary(self):
        for redirs in redirects_oss_test():
            with self.subTest(redirs=redirs):
                pc = start_processes(
                    name="echo",
                    entrypoint=self.bin("echo1.py"),
                    args={0: ("hello",), 1: ("hello",)},
                    envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                    log_dir=self.log_dir(),
                    redirects=redirs,
                )

                results = pc.wait(period=0.1)

                self.assert_pids_noexist(pc.pids())
                # currently binaries return {rank: None}
                self.assertEqual(2, len(results.return_values))
                self.assertFalse(results.is_failed())

                nprocs = pc.nprocs
                for i in range(nprocs):
                    if redirs & Std.OUT != Std.OUT:
                        self.assertFalse(results.stdouts[i])
                    if redirs & Std.ERR != Std.ERR:
                        self.assertFalse(results.stderrs[i])
                    if redirs & Std.OUT == Std.OUT:
                        self.assert_in_file(
                            [f"hello stdout from {i}"], results.stdouts[i]
                        )
                    if redirs & Std.ERR == Std.ERR:
                        self.assert_in_file(
                            [f"hello stderr from {i}"], results.stderrs[i]
                        )

    def test_binary_redirect_and_tee(self):
        pc = start_processes(
            name="trainer",
            entrypoint=self.bin("echo1.py"),
            args={0: ("hello",), 1: ("world",)},
            envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
            log_dir=self.log_dir(),
            start_method="fork",
            redirects={0: Std.ERR, 1: Std.NONE},
            tee={0: Std.OUT, 1: Std.ERR},
        )

        result = pc.wait()

        self.assertFalse(result.is_failed())
        self.assert_in_file(["hello stdout from 0"], pc.stdouts[0])
        self.assert_in_file(["hello stderr from 0"], pc.stderrs[0])
        self.assert_in_file(["world stderr from 1"], pc.stderrs[1])
        self.assertFalse(pc.stdouts[1])
        self.assertTrue(pc._stderr_tail.stopped())
        self.assertTrue(pc._stdout_tail.stopped())


@unittest.skipIf(
    TEST_WITH_ASAN or TEST_WITH_TSAN or IS_WINDOWS or IS_MACOS or IS_PYTORCH_CI,
    "tests incompatible with tsan or asan, the redirect functionality does not work on macos or windows",
)
class StartProcessesNotCITest(StartProcessesTest):
    def test_wrap_bad(self):
        none = ""
        stdout_log = os.path.join(self.test_dir, "stdout.log")
        stderr_log = os.path.join(self.test_dir, "stderr.log")
        redirs = [
            (none, none),
            (none, stderr_log),
            (stdout_log, none),
            (stdout_log, stderr_log),
        ]

        for stdout_redir, stderr_redir in redirs:
            queue = multiprocessing.SimpleQueue()
            worker_finished_event_mock = mock.Mock()
            _wrap(
                local_rank=0,
                fn=echo1,
                args={0: ("hello",)},
                envs={0: {"RANK": "0"}},
                stdout_redirects={0: stdout_redir},
                stderr_redirects={0: stderr_redir},
                ret_vals={0: queue},
                queue_finished_reading_event=worker_finished_event_mock,
            )
            self.assertEqual("hello_0", queue.get())
            if stdout_redir:
                self.assert_in_file(["hello stdout from 0"], stdout_log)
            if stderr_redir:
                self.assert_in_file(["hello stderr from 0"], stderr_log)
            worker_finished_event_mock.wait.assert_called_once()

    def test_function_failure_signal(self):
        """
        run 2x copies of echo3, induce a segfault on first
        """
        SEGFAULT = True
        for start_method, redirs in product(self._start_methods, redirects_all()):
            with self.subTest(start_method=start_method):
                log_dir = self.log_dir()
                pc = start_processes(
                    name="echo",
                    entrypoint=echo3,
                    args={0: ("hello", SEGFAULT), 1: ("world",)},
                    envs={0: {}, 1: {}},
                    log_dir=log_dir,
                    start_method=start_method,
                    redirects=redirs,
                )

                results = pc.wait(period=0.1)

                self.assert_pids_noexist(pc.pids())
                self.assertEqual(1, len(results.failures))
                self.assertFalse(results.return_values)

                failure = results.failures[0]
                error_file = failure.error_file

                self.assertEqual(-signal.SIGSEGV, failure.exitcode)
                self.assertEqual("SIGSEGV", failure.signal_name())
                self.assertEqual(pc.pids()[0], failure.pid)
                self.assertEqual(os.path.join(log_dir, "0", "error.json"), error_file)

    def test_binary_signal(self):
        pc = start_processes(
            name="echo",
            entrypoint=self.bin("echo3.py"),
            args={0: ("--segfault", "true", "foo"), 1: ("bar",)},
            envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
            log_dir=self.log_dir(),
        )

        results = pc.wait(period=0.1)

        self.assert_pids_noexist(pc.pids())
        self.assertTrue(results.is_failed())
        self.assertEqual(1, len(results.failures))

        failure = results.failures[0]
        self.assertNotEqual(signal.SIGSEGV, failure.exitcode)
        self.assertEqual("SIGSEGV", failure.signal_name())
        self.assertEqual("<NONE>", failure.error_file_data["message"])

    def test_function_redirect_and_tee(self):
        for start_method in self._start_methods:
            with self.subTest(start_method=start_method):
                log_dir = self.log_dir()
                pc = start_processes(
                    name="trainer",
                    entrypoint=echo1,
                    args={0: ("hello",), 1: ("world",)},
                    envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                    log_dir=log_dir,
                    start_method="fork",
                    redirects={0: Std.ERR, 1: Std.NONE},
                    tee={0: Std.OUT, 1: Std.ERR},
                )

                result = pc.wait()

                self.assertFalse(result.is_failed())
                self.assert_in_file(["hello stdout from 0"], pc.stdouts[0])
                self.assert_in_file(["hello stderr from 0"], pc.stderrs[0])
                self.assert_in_file(["world stderr from 1"], pc.stderrs[1])
                self.assertFalse(pc.stdouts[1])
                self.assertTrue(pc._stderr_tail.stopped())
                self.assertTrue(pc._stdout_tail.stopped())

    def test_function(self):
        for start_method, redirs in product(self._start_methods, redirects_all()):
            with self.subTest(start_method=start_method, redirs=redirs):
                pc = start_processes(
                    name="echo",
                    entrypoint=echo1,
                    args={0: ("hello",), 1: ("hello",)},
                    envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                    log_dir=self.log_dir(),
                    start_method=start_method,
                    redirects=redirs,
                )

                results = pc.wait(period=0.1)
                nprocs = pc.nprocs

                self.assert_pids_noexist(pc.pids())
                self.assertEqual(
                    {i: f"hello_{i}" for i in range(nprocs)}, results.return_values
                )

                for i in range(nprocs):
                    if redirs & Std.OUT != Std.OUT:
                        self.assertFalse(results.stdouts[i])
                    if redirs & Std.ERR != Std.ERR:
                        self.assertFalse(results.stderrs[i])
                    if redirs & Std.OUT == Std.OUT:
                        self.assert_in_file(
                            [f"hello stdout from {i}"], results.stdouts[i]
                        )
                    if redirs & Std.ERR == Std.ERR:
                        self.assert_in_file(
                            [f"hello stderr from {i}"], results.stderrs[i]
                        )

    def test_function_exit(self):
        """
        run 2x copies of echo1 fail (exit) the first
        functions that exit from python do not generate an error file
        (even if they are decorated with @record)
        """

        FAIL = 138
        for start_method in self._start_methods:
            with self.subTest(start_method=start_method):
                log_dir = self.log_dir()
                pc = start_processes(
                    name="echo",
                    entrypoint=echo1,
                    args={0: ("hello", FAIL), 1: ("hello",)},
                    envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                    log_dir=log_dir,
                    start_method=start_method,
                    redirects={0: Std.ERR},
                )

                results = pc.wait(period=0.1)

                self.assert_pids_noexist(pc.pids())
                self.assertTrue(results.is_failed())
                self.assertEqual(1, len(results.failures))
                self.assertFalse(results.return_values)

                failure = results.failures[0]
                error_file = failure.error_file

                self.assertEqual(FAIL, failure.exitcode)
                self.assertEqual("<N/A>", failure.signal_name())
                self.assertEqual(pc.pids()[0], failure.pid)
                self.assertEqual("<N/A>", error_file)
                self.assertEqual(
                    f"Process failed with exitcode {FAIL}", failure.message
                )
                self.assertLessEqual(failure.timestamp, int(time.time()))

                self.assert_in_file([f"exit {FAIL} from 0"], results.stderrs[0])
                self.assertFalse(results.stdouts[0])
                self.assertFalse(results.stderrs[1])
                self.assertFalse(results.stdouts[1])
                self.assertTrue(pc._stderr_tail.stopped())
                self.assertTrue(pc._stdout_tail.stopped())


if __name__ == "__main__":
    run_tests()
