#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from concurrent.futures._base import Future
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Event
from typing import Dict, List, Optional, TextIO

__all__ = ["tail_logfile", "TailLog"]

log = logging.getLogger(__name__)


def tail_logfile(
    header: str, file: str, dst: TextIO, finished: Event, interval_sec: float
):

    while not os.path.exists(file):
        if finished.is_set():
            return
        time.sleep(interval_sec)

    with open(file) as fp:
        while True:
            line = fp.readline()

            if line:
                dst.write(f"{header}{line}")
            else:  # reached EOF
                if finished.is_set():
                    # log line producer is finished
                    break
                else:
                    # log line producer is still going
                    # wait for a bit before looping again
                    time.sleep(interval_sec)


class TailLog:
    """
    Tail the given log files.

    The log files do not have to exist when the ``start()`` method is called. The tail-er will gracefully wait until
    the log files are created by the producer and will tail the contents of the
    log files until the ``stop()`` method is called.

    .. warning:: ``TailLog`` will wait indefinitely for the log file to be created!

    Each log file's line will be suffixed with a header of the form: ``[{name}{idx}]:``,
    where the ``name`` is user-provided and ``idx`` is the index of the log file
    in the ``log_files`` mapping. ``log_line_prefixes`` can be used to override the
    header for each log file.

    Usage:

    ::

     log_files = {0: "/tmp/0_stdout.log", 1: "/tmp/1_stdout.log"}
     tailer = TailLog("trainer", log_files, sys.stdout).start()
     # actually run the trainers to produce 0_stdout.log and 1_stdout.log
     run_trainers()
     tailer.stop()

     # once run_trainers() start writing the ##_stdout.log files
     # the tailer will print to sys.stdout:
     # >>> [trainer0]:log_line1
     # >>> [trainer1]:log_line1
     # >>> [trainer0]:log_line2
     # >>> [trainer0]:log_line3
     # >>> [trainer1]:log_line2

    .. note:: Due to buffering log lines between files may not necessarily
              be printed out in order. You should configure your application's
              logger to suffix each log line with a proper timestamp.

    """

    def __init__(
        self,
        name: str,
        log_files: Dict[int, str],
        dst: TextIO,
        log_line_prefixes: Optional[Dict[int, str]] = None,
        interval_sec: float = 0.1,
    ):
        n = len(log_files)
        self._threadpool = None
        if n > 0:
            self._threadpool = ThreadPoolExecutor(
                max_workers=n,
                thread_name_prefix=f"{self.__class__.__qualname__}_{name}",
            )

        self._name = name
        self._dst = dst
        self._log_files = log_files
        self._log_line_prefixes = log_line_prefixes
        self._finished_events: Dict[int, Event] = {
            local_rank: Event() for local_rank in log_files.keys()
        }
        self._futs: List[Future] = []
        self._interval_sec = interval_sec
        self._stopped = False

    def start(self) -> "TailLog":
        if not self._threadpool:
            return self

        for local_rank, file in self._log_files.items():
            header = f"[{self._name}{local_rank}]:"
            if self._log_line_prefixes and local_rank in self._log_line_prefixes:
                header = self._log_line_prefixes[local_rank]
            self._futs.append(
                self._threadpool.submit(
                    tail_logfile,
                    header=header,
                    file=file,
                    dst=self._dst,
                    finished=self._finished_events[local_rank],
                    interval_sec=self._interval_sec,
                )
            )
        return self

    def stop(self) -> None:
        for finished in self._finished_events.values():
            finished.set()

        for local_rank, f in enumerate(self._futs):
            try:
                f.result()
            except Exception as e:
                log.error(
                    "error in log tailor for %s%s. %s: %s",
                    self._name, local_rank,
                    e.__class__.__qualname__, e,
                )

        if self._threadpool:
            self._threadpool.shutdown(wait=True)

        self._stopped = True

    def stopped(self) -> bool:
        return self._stopped
