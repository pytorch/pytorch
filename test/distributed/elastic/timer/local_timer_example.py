#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import multiprocessing as mp
import signal
import time

import torch.distributed.elastic.timer as timer
import torch.multiprocessing as torch_mp
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
    IS_WINDOWS,
    IS_MACOS,
    skip_but_pass_in_sandcastle_if,
    TestCase
)


logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s %(module)s: %(message)s"
)


def _happy_function(rank, mp_queue):
    timer.configure(timer.LocalTimerClient(mp_queue))
    with timer.expires(after=1):
        time.sleep(0.5)


def _stuck_function(rank, mp_queue):
    timer.configure(timer.LocalTimerClient(mp_queue))
    with timer.expires(after=1):
        time.sleep(5)


# timer is not supported on macos or windows
if not (IS_WINDOWS or IS_MACOS):
    class LocalTimerExample(TestCase):
        """
        Demonstrates how to use LocalTimerServer and LocalTimerClient
        to enforce expiration of code-blocks.

        Since torch multiprocessing's ``start_process`` method currently
        does not take the multiprocessing context as parameter argument
        there is no way to create the mp.Queue in the correct
        context BEFORE spawning child processes. Once the ``start_process``
        API is changed in torch, then re-enable ``test_torch_mp_example``
        unittest. As of now this will SIGSEGV.
        """

        @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, "test is asan incompatible")
        def test_torch_mp_example(self):
            # in practice set the max_interval to a larger value (e.g. 60 seconds)
            mp_queue = mp.get_context("spawn").Queue()
            server = timer.LocalTimerServer(mp_queue, max_interval=0.01)
            server.start()

            world_size = 8

            # all processes should complete successfully
            # since start_process does NOT take context as parameter argument yet
            # this method WILL FAIL (hence the test is disabled)
            torch_mp.spawn(
                fn=_happy_function, args=(mp_queue,), nprocs=world_size, join=True
            )

            with self.assertRaises(Exception):
                # torch.multiprocessing.spawn kills all sub-procs
                # if one of them gets killed
                torch_mp.spawn(
                    fn=_stuck_function, args=(mp_queue,), nprocs=world_size, join=True
                )

            server.stop()

        @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, "test is asan incompatible")
        def test_example_start_method_spawn(self):
            self._run_example_with(start_method="spawn")

        # @skip_but_pass_in_sandcastle_if(TEST_WITH_DEV_DBG_ASAN, "test is asan incompatible")
        # def test_example_start_method_forkserver(self):
        #     self._run_example_with(start_method="forkserver")

        def _run_example_with(self, start_method):
            spawn_ctx = mp.get_context(start_method)
            mp_queue = spawn_ctx.Queue()
            server = timer.LocalTimerServer(mp_queue, max_interval=0.01)
            server.start()

            world_size = 8
            processes = []
            for i in range(0, world_size):
                if i % 2 == 0:
                    p = spawn_ctx.Process(target=_stuck_function, args=(i, mp_queue))
                else:
                    p = spawn_ctx.Process(target=_happy_function, args=(i, mp_queue))
                p.start()
                processes.append(p)

            for i in range(0, world_size):
                p = processes[i]
                p.join()
                if i % 2 == 0:
                    self.assertEqual(-signal.SIGKILL, p.exitcode)
                else:
                    self.assertEqual(0, p.exitcode)

            server.stop()


if __name__ == "__main__":
    run_tests()
