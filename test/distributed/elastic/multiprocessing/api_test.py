#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import ctypes
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import time
from collections.abc import Callable
from itertools import product
from typing import Union
from unittest import mock

import torch
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing import ProcessFailure, start_processes
from torch.distributed.elastic.multiprocessing.api import (
    _validate_full_rank,
    _wrap,
    DefaultLogsSpecs,
    MultiprocessContext,
    RunProcsResult,
    SignalException,
    Std,
    to_map,
)
from torch.distributed.elastic.multiprocessing.errors import ErrorHandler
from torch.testing._internal.common_utils import (
    IS_CI,
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skip_if_pytest,
    TEST_WITH_ASAN,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_TSAN,
    TestCase,
)


class RunProcResultsTest(TestCase):
    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.test_dir)

    def test_is_failed(self):
        pr_success = RunProcsResult(return_values={0: "a", 1: "b"})
        self.assertFalse(pr_success.is_failed())

        fail0 = ProcessFailure(
            local_rank=0, pid=998, exitcode=1, error_file="ignored.json"
        )
        pr_fail = RunProcsResult(failures={0: fail0})
        self.assertTrue(pr_fail.is_failed())

    def test_get_failures(self):
        error_file0 = os.path.join(self.test_dir, "error0.json")
        error_file1 = os.path.join(self.test_dir, "error1.json")
        eh = ErrorHandler()
        with mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": error_file0}):
            eh.record_exception(RuntimeError("error 0"))

        with mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": error_file0}):
            eh.record_exception(RuntimeError("error 1"))

        fail0 = ProcessFailure(
            local_rank=0, pid=997, exitcode=1, error_file=error_file0
        )
        fail1 = ProcessFailure(
            local_rank=1, pid=998, exitcode=3, error_file=error_file1
        )
        fail2 = ProcessFailure(
            local_rank=2, pid=999, exitcode=15, error_file="no_exist.json"
        )

        self.assertLessEqual(fail0.timestamp, fail1.timestamp)
        self.assertLessEqual(fail1.timestamp, fail2.timestamp)


class StdTest(TestCase):
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
        for m in msg.split(","):
            print(f"{m} stdout from {rank}")
            print(f"{m} stderr from {rank}", file=sys.stderr)
        return f"{msg}_{rank}"


def echo2(msg: str, fail: bool = False) -> str:
    """
    returns ``msg`` or raises a RuntimeError if ``fail`` is set
    """
    if fail:
        raise RuntimeError(msg)
    return msg


def echo_large(size: int) -> dict[int, str]:
    """
    returns a large output ({0: test0", 1: "test1", ..., (size-1):f"test{size-1}"})
    """
    out = {}
    for idx in range(size):
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


def redirects_oss_test() -> list[Std]:
    return [
        Std.NONE,
    ]


def redirects_all() -> list[Std]:
    return [
        Std.NONE,
        Std.OUT,
        Std.ERR,
        Std.ALL,
    ]


def bin(name: str):
    dir = os.path.dirname(__file__)
    return os.path.join(dir, "bin", name)


def wait_fn(wait_time: int = 300) -> None:
    time.sleep(wait_time)
    print("Finished waiting")


def start_processes_zombie_test(
    idx: int,
    entrypoint: Union[str, Callable],
    mp_queue: mp.Queue,
    log_dir: str,
    nproc: int = 2,
) -> None:
    """
    Starts processes
    """

    args = {}
    envs = {}
    for idx in range(nproc):
        args[idx] = ()
        envs[idx] = {}

    pc = start_processes(
        name="zombie_test",
        entrypoint=entrypoint,
        args=args,
        envs=envs,
        logs_specs=DefaultLogsSpecs(log_dir=log_dir),
    )
    my_pid = os.getpid()
    mp_queue.put(my_pid)
    for child_pid in pc.pids().values():
        mp_queue.put(child_pid)

    try:
        pc.wait(period=1, timeout=300)
    except SignalException as e:
        pc.close(e.sigval)


class _StartProcessesTest(TestCase):
    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")
        self._start_methods = ["spawn"]

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.test_dir)

    def log_dir(self):
        return tempfile.mkdtemp(dir=self.test_dir)

    def assert_in_file(self, expected: list[str], filename: str) -> None:
        expected = [f"{line.rstrip()}\n" for line in expected]
        with open(filename) as fp:
            actual = fp.readlines()
            for line in expected:
                self.assertIn(line, actual)

    def assert_not_in_file(self, lines: list[str], filename: str) -> None:
        lines = [f"{line.rstrip()}\n" for line in lines]
        with open(filename) as fp:
            actual = fp.readlines()
            for line in lines:
                self.assertNotIn(line, actual)

    def assert_pids_noexist(self, pids: dict[int, int]):
        for local_rank, pid in pids.items():
            with self.assertRaises(
                OSError, msg=f"local_rank: {local_rank} pid: {pid} should not exist"
            ):
                os.kill(pid, 0)

    def _test_zombie_workflow(
        self, entrypoint: Union[str, Callable], signal_to_send: signal.Signals
    ) -> None:
        mp_queue = mp.get_context("spawn").Queue()
        child_nproc = 2
        mp.spawn(
            start_processes_zombie_test,
            nprocs=1,
            args=(entrypoint, mp_queue, self.log_dir(), child_nproc),
            join=False,
        )
        total_processes = child_nproc + 1
        pids = []
        for _ in range(total_processes):
            pids.append(mp_queue.get(timeout=120))
        parent_pid = pids[0]
        child_pids = pids[1:]

        os.kill(parent_pid, signal.SIGTERM)
        # Wait to give time for signal handlers to finish work
        time.sleep(5)
        for child_pid in child_pids:
            # Killing parent should kill all children, we expect that each call to
            # os.kill would raise OSError
            with self.assertRaises(OSError):
                os.kill(child_pid, 0)


# tests incompatible with tsan or asan
if not (TEST_WITH_DEV_DBG_ASAN or IS_WINDOWS or IS_MACOS):

    class StartProcessesAsFuncTest(_StartProcessesTest):
        def test_to_map(self):
            local_world_size = 2
            self.assertEqual(
                {0: Std.OUT, 1: Std.OUT}, to_map(Std.OUT, local_world_size)
            )
            self.assertEqual(
                {0: Std.NONE, 1: Std.OUT}, to_map({1: Std.OUT}, local_world_size)
            )
            self.assertEqual(
                {0: Std.ERR, 1: Std.OUT},
                to_map({0: Std.ERR, 1: Std.OUT}, local_world_size),
            )

        def test_invalid_log_dir(self):
            with tempfile.NamedTemporaryFile(dir=self.test_dir) as not_a_dir:
                cases = {
                    not_a_dir.name: NotADirectoryError,
                }

                for log_dir, expected_error in cases.items():
                    with self.subTest(log_dir=log_dir, expected_error=expected_error):
                        with self.assertRaises(expected_error):
                            pc = None
                            try:
                                pc = start_processes(
                                    name="echo",
                                    entrypoint=echo1,
                                    args={0: ("hello",)},
                                    envs={0: {"RANK": "0"}},
                                    logs_specs=DefaultLogsSpecs(log_dir=log_dir),
                                )
                            finally:
                                if pc:
                                    pc.close()

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
                            logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
                        )

        def test_pcontext_wait(self):
            pc = start_processes(
                name="sleep",
                entrypoint=time.sleep,
                args={0: (1,)},
                envs={0: {}},
                logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
                start_method="spawn",
            )

            self.assertIsNone(pc.wait(timeout=0.1, period=0.01))
            self.assertIsNotNone(pc.wait(period=0.1))
            for tail_log in pc._tail_logs:
                self.assertTrue(tail_log.stopped())

        def test_pcontext_wait_on_a_child_thread(self):
            asyncio.run(asyncio.to_thread(self.test_pcontext_wait))

        def test_multiprocess_context_close(self):
            pc = start_processes(
                name="sleep",
                entrypoint=time.sleep,
                args={0: (1,)},
                envs={0: {}},
                logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
                start_method="spawn",
            )

            pids = pc.pids()
            pc.close()
            self.assert_pids_noexist(pids)
            for tail_log in pc._tail_logs:
                self.assertTrue(tail_log.stopped())

        def test_function_with_tensor(self):
            for start_method in self._start_methods:
                pc = start_processes(
                    name="dummy_compute",
                    entrypoint=dummy_compute,
                    args={0: ()},
                    envs={0: {}},
                    logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
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
                        logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
                        start_method=start_method,
                    )

                    results = pc.wait(period=0.1)
                    self.assertEqual({0: None, 1: None}, results.return_values)

        @skip_but_pass_in_sandcastle_if(
            TEST_WITH_DEV_DBG_ASAN, "tests incompatible with asan"
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
                        logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
                        name="echo",
                        entrypoint=echo_large,
                        args={0: (size,), 1: (size,), 2: (size,), 3: (size,)},
                        envs={0: {}, 1: {}, 2: {}, 3: {}},
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
                        envs={
                            0: {"TORCHELASTIC_RUN_ID": "run_id"},
                            1: {"TORCHELASTIC_RUN_ID": "run_id"},
                        },
                        logs_specs=DefaultLogsSpecs(log_dir=log_dir),
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
                    self.assertTrue(
                        error_file.startswith(os.path.join(log_dir, "run_id_"))
                    )
                    self.assertTrue(error_file.endswith("attempt_0/0/error.json"))
                    self.assertEqual(
                        int(error_file_data["message"]["extraInfo"]["timestamp"]),
                        int(failure.timestamp),
                    )
                    for tail_log in pc._tail_logs:
                        self.assertTrue(tail_log.stopped())

        def test_wait_for_all_child_procs_to_exit(self):
            """
            Tests that MultiprocessingContext actually waits for
            the child process to exit (not just that the entrypoint fn has
            finished running).
            """

            mpc = MultiprocessContext(
                name="echo",
                entrypoint=echo0,
                args={},
                envs={},
                start_method="spawn",
                logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
            )

            with (
                mock.patch.object(mpc, "_is_done", return_value=True),
                mock.patch.object(mpc, "_pc"),
                mock.patch.object(
                    mpc._pc, "join", side_effect=[True, False, False, True]
                ) as mock_join,
            ):
                mpc._poll()
                self.assertEqual(4, mock_join.call_count)

        def test_multiprocessing_context_poll_raises_exception(self):
            mp_context = MultiprocessContext(
                name="test_mp",
                entrypoint=echo0,
                args={0: (0, 1)},
                envs={0: {}},
                logs_specs=DefaultLogsSpecs(
                    log_dir=self.log_dir(), redirects=Std.ALL, tee=Std.ALL
                ),
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
                self.assertEqual(
                    "Signal 1 (SIGHUP) received by PID 123", failure.message
                )

    class StartProcessesAsBinaryTest(_StartProcessesTest):
        ########################################
        # start_processes as binary tests
        ########################################

        def test_subprocess_context_close(self):
            pc = start_processes(
                name="sleep",
                entrypoint=bin("zombie_test.py"),
                args={0: (1,)},
                envs={0: {}},
                logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
            )

            pids = pc.pids()
            pc.close()
            self.assert_pids_noexist(pids)

        def test_binary_exit(self):
            FAIL = 138
            pc = start_processes(
                name="echo",
                entrypoint=bin("echo4.py"),
                args={0: ("--exitcode", FAIL, "foo"), 1: ("--exitcode", 0, "bar")},
                envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                logs_specs=DefaultLogsSpecs(
                    log_dir=self.log_dir(),
                    redirects={0: Std.ALL},
                ),
            )

            results = pc.wait(period=0.1)
            self.assertTrue(results.is_failed())
            self.assertEqual(2, len(results.failures))

            failure = results.failures[0]
            self.assertEqual(138, failure.exitcode)
            self.assertEqual("<N/A>", failure.signal_name())
            self.assertEqual("<NONE>", failure.error_file_data["message"])
            self.assert_in_file([f"exit {FAIL} from 0"], results.stderrs[0])
            self.assert_in_file([], results.stdouts[0])
            self.assertFalse(results.stderrs[1])
            self.assertFalse(results.stdouts[1])
            for tail_log in pc._tail_logs:
                self.assertTrue(tail_log.stopped())

            failure = results.failures[1]
            self.assertEqual(-15, failure.exitcode)
            self.assertEqual("SIGTERM", failure.signal_name())
            self.assertEqual("<NONE>", failure.error_file_data["message"])
            # Assert that the failure message contains expected substrings
            self.assertIn("Signal 15 (SIGTERM) received by PID", failure.message)

        def test_binary_raises(self):
            pc = start_processes(
                name="echo",
                entrypoint=bin("echo2.py"),
                args={0: ("--raises", "true", "foo"), 1: ("bar",)},
                envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
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
                    logs_specs=DefaultLogsSpecs(log_dir=self.log_dir()),
                )

        def test_validate_full_rank(self):
            with self.assertRaises(RuntimeError):
                _validate_full_rank({}, 10, "")


# tests incompatible with tsan or asan, the redirect functionality does not work on macos or windows
if not (TEST_WITH_DEV_DBG_ASAN or IS_WINDOWS or IS_MACOS):

    class StartProcessesListAsFuncTest(_StartProcessesTest):
        def test_function(self):
            for start_method, redirs in product(
                self._start_methods, redirects_oss_test()
            ):
                with self.subTest(start_method=start_method, redirs=redirs):
                    pc = start_processes(
                        name="echo",
                        entrypoint=echo1,
                        args={0: ("hello",), 1: ("hello",)},
                        envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                        logs_specs=DefaultLogsSpecs(
                            log_dir=self.log_dir(),
                            redirects=redirs,
                        ),
                        start_method=start_method,
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

    class StartProcessesListAsBinaryTest(_StartProcessesTest):
        ########################################
        # start_processes as binary tests
        ########################################
        def test_binary(self):
            for redirs in redirects_oss_test():
                with self.subTest(redirs=redirs):
                    pc = start_processes(
                        name="echo",
                        entrypoint=bin("echo1.py"),
                        args={0: ("hello",), 1: ("hello",)},
                        envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                        logs_specs=DefaultLogsSpecs(
                            log_dir=self.log_dir(),
                            redirects=redirs,
                        ),
                        log_line_prefixes={0: "[rank0]:", 1: "[rank1]:"},
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
                entrypoint=bin("echo1.py"),
                args={0: ("hello",), 1: ("world",)},
                envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                logs_specs=DefaultLogsSpecs(
                    log_dir=self.log_dir(),
                    redirects={0: Std.ERR, 1: Std.NONE},
                    tee={0: Std.OUT, 1: Std.ERR},
                ),
                log_line_prefixes={0: "[rank0]:", 1: "[rank1]:"},
                start_method="spawn",
            )

            result = pc.wait()

            self.assertFalse(result.is_failed())
            self.assert_in_file(["hello stdout from 0"], pc.stdouts[0])
            self.assert_in_file(["hello stderr from 0"], pc.stderrs[0])
            self.assert_in_file(["world stderr from 1"], pc.stderrs[1])
            self.assertFalse(pc.stdouts[1])
            for tail_log in pc._tail_logs:
                self.assertTrue(tail_log.stopped())

        def test_binary_duplicate_log_filters(self):
            pc = start_processes(
                name="trainer",
                entrypoint=bin("echo1.py"),
                args={0: ("helloA,helloB",), 1: ("worldA,worldB",)},
                envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                logs_specs=DefaultLogsSpecs(
                    log_dir=self.log_dir(),
                    redirects={0: Std.ERR, 1: Std.NONE},
                    tee={0: Std.OUT, 1: Std.ERR},
                ),
                log_line_prefixes={0: "[rank0]:", 1: "[rank1]:"},
                duplicate_stdout_filters=["helloA"],
                duplicate_stderr_filters=["worldA", "B"],
                start_method="spawn",
            )

            result = pc.wait()

            self.assertFalse(result.is_failed())
            self.assert_in_file(["[rank0]:helloA stdout from 0"], pc.filtered_stdout)
            self.assert_not_in_file(
                ["[rank0]:helloB stdout from 0"], pc.filtered_stdout
            )
            self.assert_in_file(["[rank1]:worldA stderr from 1"], pc.filtered_stderr)
            self.assert_in_file(["[rank1]:worldB stderr from 1"], pc.filtered_stderr)
            for tail_log in pc._tail_logs:
                self.assertTrue(tail_log.stopped())


# tests incompatible with tsan or asan, the redirect functionality does not work on macos or windows
if not (TEST_WITH_DEV_DBG_ASAN or IS_WINDOWS or IS_MACOS or IS_CI):

    class StartProcessesNotCIAsFuncTest(_StartProcessesTest):
        @skip_if_pytest
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
                    numa_options=None,
                )
                self.assertEqual("hello_0", queue.get())
                if stdout_redir:
                    self.assert_in_file(["hello stdout from 0"], stdout_log)
                if stderr_redir:
                    self.assert_in_file(["hello stderr from 0"], stderr_log)
                worker_finished_event_mock.wait.assert_called_once()

        def test_function_redirect_and_tee(self):
            for start_method in self._start_methods:
                with self.subTest(start_method=start_method):
                    pc = start_processes(
                        name="trainer",
                        entrypoint=echo1,
                        args={0: ("hello",), 1: ("world",)},
                        envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                        logs_specs=DefaultLogsSpecs(
                            log_dir=self.log_dir(),
                            redirects={0: Std.ERR, 1: Std.NONE},
                            tee={0: Std.OUT, 1: Std.ERR},
                        ),
                        start_method="spawn",
                    )

                    result = pc.wait()

                    self.assertFalse(result.is_failed())
                    self.assert_in_file(["hello stdout from 0"], pc.stdouts[0])
                    self.assert_in_file(["hello stderr from 0"], pc.stderrs[0])
                    self.assert_in_file(["world stderr from 1"], pc.stderrs[1])
                    self.assertFalse(pc.stdouts[1])
                    for tail_log in pc._tail_logs:
                        self.assertTrue(tail_log.stopped())

        def test_function_duplicate_log_filters(self):
            for start_method in self._start_methods:
                with self.subTest(start_method=start_method):
                    pc = start_processes(
                        name="trainer",
                        entrypoint=echo1,
                        args={0: ("helloA,helloB",), 1: ("worldA,worldB",)},
                        envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                        logs_specs=DefaultLogsSpecs(
                            log_dir=self.log_dir(),
                            redirects={0: Std.ERR, 1: Std.NONE},
                            tee={0: Std.OUT, 1: Std.ERR},
                        ),
                        duplicate_stdout_filters=["helloA"],
                        duplicate_stderr_filters=["worldA", "B"],
                        start_method="spawn",
                    )

                    result = pc.wait()

                    self.assertFalse(result.is_failed())
                    self.assert_in_file(
                        ["[trainer0]:helloA stdout from 0"], pc.filtered_stdout
                    )
                    self.assert_not_in_file(
                        ["[trainer0]:helloB stdout from 0"], pc.filtered_stdout
                    )
                    self.assert_in_file(
                        ["[trainer1]:worldA stderr from 1"], pc.filtered_stderr
                    )
                    self.assert_in_file(
                        ["[trainer1]:worldB stderr from 1"], pc.filtered_stderr
                    )
                    for tail_log in pc._tail_logs:
                        self.assertTrue(tail_log.stopped())

        def test_function(self):
            for start_method, redirs in product(self._start_methods, redirects_all()):
                with self.subTest(start_method=start_method, redirs=redirs):
                    pc = start_processes(
                        name="echo",
                        entrypoint=echo1,
                        args={0: ("hello",), 1: ("hello",)},
                        envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                        start_method=start_method,
                        logs_specs=DefaultLogsSpecs(
                            log_dir=self.log_dir(),
                            redirects=redirs,
                        ),
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
                    pc = start_processes(
                        name="echo",
                        entrypoint=echo1,
                        args={0: ("hello", FAIL), 1: ("hello",)},
                        envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                        logs_specs=DefaultLogsSpecs(
                            log_dir=self.log_dir(),
                            redirects={0: Std.ERR},
                        ),
                        start_method=start_method,
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
                        "To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html",
                        failure.message,
                    )
                    self.assertLessEqual(failure.timestamp, int(time.time()))

                    self.assert_in_file([f"exit {FAIL} from 0"], results.stderrs[0])
                    self.assertFalse(results.stdouts[0])
                    self.assertFalse(results.stderrs[1])
                    self.assertFalse(results.stdouts[1])
                    for tail_log in pc._tail_logs:
                        self.assertTrue(tail_log.stopped())

        def test_no_zombie_process_function(self):
            signals = [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT]
            for s in signals:
                self._test_zombie_workflow(wait_fn, s)

    class StartProcessesNotCIAsBinaryTest(_StartProcessesTest):
        def test_binary_signal(self):
            pc = start_processes(
                name="echo",
                entrypoint=bin("echo3.py"),
                args={0: ("--segfault", "true", "foo"), 1: ("bar",)},
                envs={0: {"RANK": "0"}, 1: {"RANK": "1"}},
                logs_specs=DefaultLogsSpecs(
                    log_dir=self.log_dir(),
                ),
            )

            results = pc.wait(period=0.1)

            self.assert_pids_noexist(pc.pids())
            self.assertTrue(results.is_failed())
            self.assertEqual(1, len(results.failures))

            failure = results.failures[0]
            self.assertNotEqual(signal.SIGSEGV, failure.exitcode)
            if TEST_WITH_ASAN or TEST_WITH_TSAN:
                # ASAN/TSAN exit code is 1.
                self.assertEqual("<N/A>", failure.signal_name())
            else:
                self.assertEqual("SIGSEGV", failure.signal_name())
            self.assertEqual("<NONE>", failure.error_file_data["message"])

        def test_no_zombie_process_binary(self):
            signals = [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT]
            for s in signals:
                self._test_zombie_workflow(bin("zombie_test.py"), s)

    class ForkServerTest(
        StartProcessesAsFuncTest,
        StartProcessesListAsFuncTest,
        StartProcessesNotCIAsFuncTest,
    ):
        def setUp(self):
            super().setUp()
            self._start_methods = ["forkserver"]
            self.orig_paralell_env_val = os.environ.get(mp.ENV_VAR_PARALLEL_START)
            os.environ[mp.ENV_VAR_PARALLEL_START] = "1"

        def tearDown(self):
            super().tearDown()
            if self.orig_paralell_env_val is None:
                del os.environ[mp.ENV_VAR_PARALLEL_START]
            else:
                os.environ[mp.ENV_VAR_PARALLEL_START] = self.orig_paralell_env_val


if __name__ == "__main__":
    run_tests()
