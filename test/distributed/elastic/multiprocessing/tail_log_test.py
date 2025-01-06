#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import io
import os
import shutil
import sys
import tempfile
import time
import unittest
from concurrent.futures import wait
from concurrent.futures._base import ALL_COMPLETED
from concurrent.futures.thread import ThreadPoolExecutor
from unittest import mock

from torch.distributed.elastic.multiprocessing.tail_log import TailLog


def write(max: int, sleep: float, file: str):
    with open(file, "w") as fp:
        for i in range(max):
            print(i, file=fp, flush=True)
            time.sleep(sleep)


class TailLogTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")
        self.threadpool = ThreadPoolExecutor()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_tail(self):
        """
        writer() writes 0 - max (on number on each line) to a log file.
        Run nprocs such writers and tail the log files into an IOString
        and validate that all lines are accounted for.
        """
        nprocs = 32
        max = 1000
        interval_sec = 0.0001

        log_files = {
            local_rank: os.path.join(self.test_dir, f"{local_rank}_stdout.log")
            for local_rank in range(nprocs)
        }

        dst = io.StringIO()
        tail = TailLog(
            name="writer", log_files=log_files, dst=dst, interval_sec=interval_sec
        ).start()
        # sleep here is intentional to ensure that the log tail
        # can gracefully handle and wait for non-existent log files
        time.sleep(interval_sec * 10)

        futs = []
        for local_rank, file in log_files.items():
            f = self.threadpool.submit(
                write, max=max, sleep=interval_sec * local_rank, file=file
            )
            futs.append(f)

        wait(futs, return_when=ALL_COMPLETED)
        self.assertFalse(tail.stopped())
        tail.stop()

        dst.seek(0)
        actual: dict[int, set[int]] = {}

        for line in dst.readlines():
            header, num = line.split(":")
            nums = actual.setdefault(header, set())
            nums.add(int(num))

        self.assertEqual(nprocs, len(actual))
        self.assertEqual(
            {f"[writer{i}]": set(range(max)) for i in range(nprocs)}, actual
        )
        self.assertTrue(tail.stopped())

    def test_tail_with_custom_prefix(self):
        """
        writer() writes 0 - max (on number on each line) to a log file.
        Run nprocs such writers and tail the log files into an IOString
        and validate that all lines are accounted for.
        """
        nprocs = 3
        max = 10
        interval_sec = 0.0001

        log_files = {
            local_rank: os.path.join(self.test_dir, f"{local_rank}_stdout.log")
            for local_rank in range(nprocs)
        }

        dst = io.StringIO()
        log_line_prefixes = {n: f"[worker{n}][{n}]:" for n in range(nprocs)}
        tail = TailLog(
            "writer",
            log_files,
            dst,
            interval_sec=interval_sec,
            log_line_prefixes=log_line_prefixes,
        ).start()
        # sleep here is intentional to ensure that the log tail
        # can gracefully handle and wait for non-existent log files
        time.sleep(interval_sec * 10)
        futs = []
        for local_rank, file in log_files.items():
            f = self.threadpool.submit(
                write, max=max, sleep=interval_sec * local_rank, file=file
            )
            futs.append(f)
        wait(futs, return_when=ALL_COMPLETED)
        self.assertFalse(tail.stopped())
        tail.stop()
        dst.seek(0)

        headers: set[str] = set()
        for line in dst.readlines():
            header, _ = line.split(":")
            headers.add(header)
        self.assertEqual(nprocs, len(headers))
        for i in range(nprocs):
            self.assertIn(f"[worker{i}][{i}]", headers)
        self.assertTrue(tail.stopped())

    def test_tail_no_files(self):
        """
        Ensures that the log tail can gracefully handle no log files
        in which case it does nothing.
        """
        tail = TailLog("writer", log_files={}, dst=sys.stdout).start()
        self.assertFalse(tail.stopped())
        tail.stop()
        self.assertTrue(tail.stopped())

    def test_tail_logfile_never_generates(self):
        """
        Ensures that we properly shutdown the threadpool
        even when the logfile never generates.
        """

        tail = TailLog("writer", log_files={0: "foobar.log"}, dst=sys.stdout).start()
        tail.stop()
        self.assertTrue(tail.stopped())
        self.assertTrue(tail._threadpool._shutdown)

    @mock.patch("torch.distributed.elastic.multiprocessing.tail_log.logger")
    def test_tail_logfile_error_in_tail_fn(self, mock_logger):
        """
        Ensures that when there is an error in the tail_fn (the one that runs in the
        threadpool), it is dealt with and raised properly.
        """

        # try giving tail log a directory (should fail with an IsADirectoryError
        tail = TailLog("writer", log_files={0: self.test_dir}, dst=sys.stdout).start()
        tail.stop()

        mock_logger.error.assert_called_once()
