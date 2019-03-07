from __future__ import absolute_import, division, print_function, unicode_literals

import os
import random
import signal
import sys
import time
import unittest

from common_utils import (TestCase, run_tests, IS_WINDOWS, NO_MULTIPROCESSING_SPAWN)
import torch.multiprocessing as mp


def test_success_func(i):
    pass


def test_success_single_arg_func(i, arg):
    if arg:
        arg.put(i)


def test_exception_single_func(i, arg):
    if i == arg:
        raise ValueError("legitimate exception from process %d" % i)
    time.sleep(1.0)


def test_exception_all_func(i):
    time.sleep(random.random() / 10)
    raise ValueError("legitimate exception from process %d" % i)


def test_terminate_signal_func(i):
    if i == 0:
        os.kill(os.getpid(), signal.SIGABRT)
    time.sleep(1.0)


def test_terminate_exit_func(i, arg):
    if i == 0:
        sys.exit(arg)
    time.sleep(1.0)


def test_success_first_then_exception_func(i, arg):
    if i == 0:
        return
    time.sleep(0.1)
    raise ValueError("legitimate exception")


def test_nested_child_body(i, ready_queue, nested_child_sleep):
    ready_queue.put(None)
    time.sleep(nested_child_sleep)


def test_nested_spawn(i, pids_queue, nested_child_sleep):
    context = mp.get_context("spawn")
    nested_child_ready_queue = context.Queue()
    nprocs = 2
    spawn_context = mp.spawn(
        fn=test_nested_child_body,
        args=(nested_child_ready_queue, nested_child_sleep),
        nprocs=nprocs,
        join=False,
        daemon=False,
    )
    pids_queue.put(spawn_context.pids())

    # Wait for both children to have spawned, to ensure that they
    # have called prctl(2) to register a parent death signal.
    for _ in range(nprocs):
        nested_child_ready_queue.get()

    # Kill self. This should take down the child processes as well.
    os.kill(os.getpid(), signal.SIGTERM)


@unittest.skipIf(
    NO_MULTIPROCESSING_SPAWN,
    "Disabled for environments that don't support the spawn start method")
class SpawnTest(TestCase):
    def test_success(self):
        mp.spawn(test_success_func, nprocs=2)

    def test_success_non_blocking(self):
        spawn_context = mp.spawn(test_success_func, nprocs=2, join=False)

        # After all processes (nproc=2) have joined it must return True
        spawn_context.join(timeout=None)
        spawn_context.join(timeout=None)
        self.assertTrue(spawn_context.join(timeout=None))

    def test_first_argument_index(self):
        context = mp.get_context("spawn")
        queue = context.SimpleQueue()
        mp.spawn(test_success_single_arg_func, args=(queue,), nprocs=2)
        self.assertEqual([0, 1], sorted([queue.get(), queue.get()]))

    def test_exception_single(self):
        nprocs = 2
        for i in range(nprocs):
            with self.assertRaisesRegex(
                Exception,
                "\nValueError: legitimate exception from process %d$" % i,
            ):
                mp.spawn(test_exception_single_func, args=(i,), nprocs=nprocs)

    def test_exception_all(self):
        with self.assertRaisesRegex(
            Exception,
            "\nValueError: legitimate exception from process (0|1)$",
        ):
            mp.spawn(test_exception_all_func, nprocs=2)

    def test_terminate_signal(self):
        # SIGABRT is aliased with SIGIOT
        message = "process 0 terminated with signal (SIGABRT|SIGIOT)"

        # Termination through with signal is expressed as a negative exit code
        # in multiprocessing, so we know it was a signal that caused the exit.
        # This doesn't appear to exist on Windows, where the exit code is always
        # positive, and therefore results in a different exception message.
        # Exit code 22 means "ERROR_BAD_COMMAND".
        if IS_WINDOWS:
            message = "process 0 terminated with exit code 22"

        with self.assertRaisesRegex(Exception, message):
            mp.spawn(test_terminate_signal_func, nprocs=2)

    def test_terminate_exit(self):
        exitcode = 123
        with self.assertRaisesRegex(
            Exception,
            "process 0 terminated with exit code %d" % exitcode,
        ):
            mp.spawn(test_terminate_exit_func, args=(exitcode,), nprocs=2)

    def test_success_first_then_exception(self):
        exitcode = 123
        with self.assertRaisesRegex(
            Exception,
            "ValueError: legitimate exception",
        ):
            mp.spawn(test_success_first_then_exception_func, args=(exitcode,), nprocs=2)

    @unittest.skipIf(
        sys.platform != "linux",
        "Only runs on Linux; requires prctl(2)",
    )
    def test_nested_spawn(self):
        context = mp.get_context("spawn")
        pids_queue = context.Queue()
        nested_child_sleep = 20.0
        spawn_context = mp.spawn(
            fn=test_nested_spawn,
            args=(pids_queue, nested_child_sleep),
            nprocs=1,
            join=False,
            daemon=False,
        )

        # Wait for nested children to terminate in time
        pids = pids_queue.get()
        start = time.time()
        while len(pids) > 0:
            for pid in pids:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pids.remove(pid)
                    break

            # This assert fails if any nested child process is still
            # alive after (nested_child_sleep / 2) seconds. By
            # extension, this test times out with an assertion error
            # after (nested_child_sleep / 2) seconds.
            self.assertLess(time.time() - start, nested_child_sleep / 2)
            time.sleep(0.1)


if __name__ == '__main__':
    run_tests()
