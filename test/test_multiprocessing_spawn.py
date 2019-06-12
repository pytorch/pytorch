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


if __name__ == '__main__':
    run_tests()
