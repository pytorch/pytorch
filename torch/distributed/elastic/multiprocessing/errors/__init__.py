#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Each host in a distributed PyTorch job runs with a single TorchElastic agent,
and multiple workers (as children processes of the TorchElastic agent).
Since the workers are user-provided (your PyTorch script/job), TorchElastic
has a way to propagate errors on the trainers through the agent and up to the
scheduler, which ultimately informs the end-user about the state of the job
and applies any retry policies.

TorchElastic categorizes errors into 3 categories:

+----------------+----------------+--------------------------------------------------------------+
| Category       | Sub-Category   |  Description                                                 |
+================+================+==============================================================+
| User Error     | Input Error    | invalid inputs to TorchElastic APIs (e.g. min > max nodes)   |
|                +----------------+--------------------------------------------------------------+
|                | Worker Failure | any failures on the worker child process                     |
+----------------+----------------+--------------------------------------------------------------+
| Platform Error |      n/a       | failures caused by the agent                                 |
+----------------+----------------+--------------------------------------------------------------+
| Infra Error    |      n/a       | failures outside the domain of the agent and workers         |
|                |                | (e.g. host failures)                                         |
+----------------+----------------+--------------------------------------------------------------+

All errors other than "Worker Failure" are either raised canonically from the
agent process or implicitly or explicitly crash the agent process. So the
standard language (python) provided exception handling strategies apply.

Worker Failures are special because the exception/failure originates on a different
process from the agent so the error needs to be propagated inter-process
(e.g. the agent cannot simply ``try-catch`` an exception raised on the worker process).

TorchElastic agents use :func:`torch.distributed.elastic.multiprocessing.start_processes`
to launch the workers which has a simple file based inter-process error propagation
built-in.

Any function or binary entrypoint decorated with :func:`record`
will write uncaught exceptions (with the trace information) to a file specified by the
environment variable ``TORCHELASTIC_ERROR_FILE``. The parent process (e.g. agent)
sets this env var on each child it launches, then aggregates the error files for all
children, and propagates the one with the **smallest** timestamp (e.g. the **first** error).
"""

import json
import os
import signal
import socket
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from string import Template
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from torch.distributed.elastic.utils.logging import get_logger

from .error_handler import ErrorHandler  # noqa: F401
from .handlers import get_error_handler  # noqa: F401

__all__ = ["ProcessFailure", "ChildFailedError", "record", "ErrorHandler", "get_error_handler"]

log = get_logger(__name__)


JSON = Dict

_EMPTY_ERROR_DATA = {"message": "<NONE>"}
_NOT_AVAILABLE = "<N/A>"

T = TypeVar("T")


@dataclass
class ProcessFailure:
    """
    Represents the failed process result. When the worker process fails,
    it may record failure root cause into the file.
    Tries to read the failure timestamp from the provided ``error_file``,
    if the ``error_file`` does not exist, the timestamp is the current
    timestamp (seconds since epoch).

    The ``message`` field is a concise explanation of the failure. If
    the error file exists then the message is obtained from the error file.
    Otherwise one is generated based on the failure signature.

    .. note:: It is assumed that the ``error_file`` is written by
              ``torch.distributed.elastic.multiprocessing.errors.error_handler.ErrorHandler``.
              Otherwise the behavior is undefined.

    """

    local_rank: int
    pid: int
    exitcode: int
    error_file: str
    error_file_data: JSON = field(init=False)
    message: str = field(init=False)
    timestamp: int = field(init=False)

    def __post_init__(self):
        self.error_file_data = _EMPTY_ERROR_DATA
        if os.path.isfile(self.error_file):
            try:
                with open(self.error_file, "r") as fp:
                    self.error_file_data = json.load(fp)
                    log.debug(
                        "User process failed with error data: %s", json.dumps(self.error_file_data, indent=2)
                    )
                    self.message, self.timestamp = self._get_error_data(
                        self.error_file_data
                    )
            except Exception:
                log.exception("Failed to parse reply file: %s", self.error_file)
                raise
        else:
            self._set_no_reply_file()

        # make up an informative message if not already present
        if not self.message:
            # signals typically do not generate an error file message
            if self.exitcode < 0:
                self.message = (
                    f"Signal {-self.exitcode} ({self.signal_name()})"
                    f" received by PID {self.pid}"
                )
            else:
                self.message = "To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html"

    def _get_error_data(self, error_file_data: Dict[str, Any]) -> Tuple[str, int]:
        message = error_file_data["message"]
        if isinstance(message, str):
            timestamp = int(error_file_data.get("timestamp", 0))
        else:
            timestamp = int(message["extraInfo"]["timestamp"])
        return (message, timestamp)

    def _set_no_reply_file(self):
        self.error_file = _NOT_AVAILABLE
        self.error_file_data = _EMPTY_ERROR_DATA
        self.message = ""
        self.timestamp = int(time.time())

    def signal_name(self) -> str:
        if self.exitcode < 0:
            return signal.Signals(-self.exitcode).name
        else:
            return _NOT_AVAILABLE

    def timestamp_isoformat(self):
        """
        Returns timestamp in ISO format (YYYY-MM-DD_HH:MM:SS)
        """
        return datetime.fromtimestamp(self.timestamp).isoformat(sep="_")


GlobalRank = int

_FAILURE_FORMAT_TEMPLATE = """[${idx}]:
  time      : ${time}
  host      : ${hostname}
  rank      : ${rank} (local_rank: ${local_rank})
  exitcode  : ${exitcode} (pid: ${pid})
  error_file: ${error_file}
  traceback : ${message}"""

# extra new lines before and after are intentional
_MSG_FORMAT_TEMPLATE = """
${boarder}
${title}
${section}
Failures:
${other_failures}
${section}
Root Cause (first observed failure):
${root_failure}
${boarder}"""


class ChildFailedError(Exception):
    """
    Special exception type that can be raised from a function annotated with the
    ``@record`` decorator to have the child process' (root exception) propagate
    up the stack as-is (e.g. without being wrapped in the parent's traceback).

    Useful in cases where the parent is a simple nanny process
    and the child (worker) processes are actually doing meaningful compute.
    In this case, errors typically occur on the child process as the parent
    is not doing anything non-trivial, and child errors should be propagated
    to the scheduler for accurate root cause diagnostics.

    .. note:: The propagation relies on error files rather than exception handling to
              support both function and binary launches.

    Example:

    ::

     # process tree on a host (container)
     0: scheduler-init-process:
                |- 1: torchelastic_agent:
                         |- 2: trainer_0 (ok)
                         |- 3: trainer_1 (fail) -> error.json
                         |- ...
                         |- n+2: trainer_n (ok)
                |- n+3: other processes
                |- ...

    In the example above, trainer 1's failure (written into error.json) is
    the root cause and should be reported to the scheduler's init process.
    The torchelastic agent raises a ``ChildFailedError("trainer", {1: "trainer_1/error.json"})``
    upon detecting trainer 1's failure which would propagate the contents
    of trainer 1's error file to the scheduler's init process.
    """

    def __init__(self, name: str, failures: Dict[GlobalRank, ProcessFailure]):
        self.name = name
        self.failures = failures
        assert (
            self.failures
        )  # does not make sense to create a ChildFaileError with no failures
        super().__init__(self.format_msg())

    def get_first_failure(self) -> Tuple[GlobalRank, ProcessFailure]:
        rank = min(self.failures.keys(), key=lambda r: self.failures[r].timestamp)
        return rank, self.failures[rank]

    def format_msg(self, boarder_delim="=", section_delim="-"):
        title = f"{self.name} FAILED"
        root_rank, root_failure = self.get_first_failure()

        root_failure_fmt: str = ""
        other_failures_fmt: List[str] = []
        width = len(title)
        for idx, (rank, failure) in enumerate(self.failures.items()):
            fmt, w = self._format_failure(idx, rank, failure)
            width = max(width, w)
            if rank == root_rank:
                root_failure_fmt = fmt
            else:
                other_failures_fmt.append(fmt)

        # upper boundary on width
        width = min(width, 60)

        return Template(_MSG_FORMAT_TEMPLATE).substitute(
            boarder=boarder_delim * width,
            title=title,
            section=section_delim * width,
            root_failure=root_failure_fmt,
            other_failures="\n".join(other_failures_fmt or ["  <NO_OTHER_FAILURES>"]),
        )

    def _format_failure(
        self, idx: int, rank: int, failure: ProcessFailure
    ) -> Tuple[str, int]:

        # failure.message is either a str (when the failure does not generate a traceback - e.g. signals)
        # or a dict (json) of the form
        # {"message": $ERROR_MSG, "extraInfo": {"py_callstack": $TRACEBACK, timestamp: $TS}}
        # so the display logic is:
        # 1. if failure.message is not a dict (it is a str) just show it as is
        # 2. else try to get the traceback (py_callstack)
        # 3.      if the traceback is not there, use the message
        # 4.      if the message  is not there show <N/A>
        msg = failure.message
        if isinstance(failure.message, dict):
            msg = (
                failure.message.get("extraInfo", {})
                .get("py_callstack", failure.message.get("message", "<N/A>"))
                .replace("\n", "\n  ")  # to properly indent the traceback
            )

        fmt = Template(_FAILURE_FORMAT_TEMPLATE).substitute(
            idx=idx,
            time=failure.timestamp_isoformat(),
            hostname=socket.getfqdn(),
            rank=rank,
            local_rank=failure.local_rank,
            exitcode=failure.exitcode,
            pid=failure.pid,
            error_file=failure.error_file,
            message=msg,
        )
        width = 0
        for line in fmt.split("\n"):
            width = max(width, len(line))
        return fmt, width


def record(
    fn: Callable[..., T], error_handler: Optional[ErrorHandler] = None
) -> Callable[..., T]:
    """
    Syntactic sugar to record errors/exceptions that happened in the decorated
    function using the provided ``error_handler``.

    Using this decorator is equivalent to:

    ::

     error_handler = get_error_handler()
     error_handler.initialize()
     try:
        foobar()
     except ChildFailedError as e:
        _, failure = e.get_first_failure()
        error_handler.dump_error_file(failure.error_file, failure.exitcode)
        raise
     except Exception as e:
        error_handler.record(e)
        raise

    .. important:: use this decorator once per process at the top level method,
                   typically this is the main method.

    Example

    ::

     @record
     def main():
         pass

     if __name__=="__main__":
        main()

    """

    if not error_handler:
        error_handler = get_error_handler()

    def wrap(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            assert error_handler is not None  # assertion for mypy type checker
            error_handler.initialize()
            try:
                return f(*args, **kwargs)
            except ChildFailedError as e:
                rank, failure = e.get_first_failure()
                if failure.error_file != _NOT_AVAILABLE:
                    error_handler.dump_error_file(failure.error_file, failure.exitcode)
                else:
                    log.info(
                        (
                            "local_rank %s FAILED with no error file."
                            " Decorate your entrypoint fn with @record for traceback info."
                            " See: https://pytorch.org/docs/stable/elastic/errors.html",
                            rank
                        )
                    )
                raise
            except Exception as e:
                error_handler.record_exception(e)
                raise

        return wrapper

    return wrap(fn)
