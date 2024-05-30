#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntFlag
from multiprocessing import synchronize
from types import FrameType
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
from abc import ABC, abstractmethod

import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure, record
from torch.distributed.elastic.multiprocessing.redirects import (
    redirect_stderr,
    redirect_stdout,
)

from torch.distributed.elastic.multiprocessing.subprocess_handler import SubprocessHandler, get_subprocess_handler
from torch.distributed.elastic.multiprocessing.tail_log import TailLog

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


logger = logging.getLogger(__name__)

__all__ = [
    "DefaultLogsSpecs",
    "SignalException",
    "Std",
    "to_map",
    "RunProcsResult",
    "PContext",
    "get_std_cm",
    "MultiprocessContext",
    "SubprocessContext",
    "LogsDest",
    "LogsSpecs",
]

class SignalException(Exception):
    """
    Exception is raised inside the torchelastic agent process by the termination handler
    if the death signal got received by the process.
    """

    def __init__(self, msg: str, sigval: signal.Signals) -> None:
        super().__init__(msg)
        self.sigval = sigval


def _terminate_process_handler(signum: int, frame: Optional[FrameType]) -> None:
    """Termination handler that raises exceptions on the main process.

    When the process receives death signal(SIGTERM, SIGINT), this termination handler will
    be invoked. It raises the ``SignalException`` exception that should be processed by the
    user code. Python does not terminate process after the termination handler is finished,
    so the exception should not be silently ignored, otherwise the process will never
    be terminated.
    """
    sigval = signal.Signals(signum)
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)


def _get_kill_signal() -> signal.Signals:
    """Get the kill signal. SIGKILL for unix, CTRL_C_EVENT for windows."""
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined] # noqa: F821
    else:
        return signal.SIGKILL


def _get_default_signal() -> signal.Signals:
    """Get the default termination signal. SIGTERM for unix, CTRL_C_EVENT for windows."""
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined] # noqa: F821
    else:
        return signal.SIGTERM


def _validate_full_rank(d: Dict[int, Any], nprocs: int, what: str):
    actual_keys = set(d.keys())
    expected_keys = set(range(nprocs))

    if actual_keys != expected_keys:
        raise RuntimeError(
            f"{what}, local rank mapping mismatch,"
            f" expected: {expected_keys}, actual: {actual_keys}"
        )


_MAPPING_REGEX = r"^(\d:[0123],)*(\d:[0123])$"
_VALUE_REGEX = r"^[0123]$"


class Std(IntFlag):
    NONE = 0
    OUT = 1
    ERR = 2
    ALL = OUT | ERR

    @classmethod
    def from_str(cls, vm: str) -> Union["Std", Dict[int, "Std"]]:
        """
        Example:
        ::

         from_str("0") -> Std.NONE
         from_str("1") -> Std.OUT
         from_str("0:3,1:0,2:1,3:2") -> {0: Std.ALL, 1: Std.NONE, 2: Std.OUT, 3: Std.ERR}

        Any other input raises an exception
        """

        def to_std(v: str) -> Std:  # type: ignore[return]
            s = Std(int(v))
            if s in Std:
                return s
            # return None -> should NEVER reach here since we regex check input

        if re.match(_VALUE_REGEX, vm):  # vm is a number (e.g. 0)
            return to_std(vm)
        elif re.match(_MAPPING_REGEX, vm):  # vm is a mapping (e.g. 0:1,1:2)
            d: Dict[int, Std] = {}
            for m in vm.split(","):
                i, v = m.split(":")
                d[int(i)] = to_std(v)
            return d
        else:
            raise ValueError(
                f"{vm} does not match: <{_VALUE_REGEX}> or <{_MAPPING_REGEX}>"
            )


def to_map(
    val_or_map: Union[Std, Dict[int, Std]], local_world_size: int
) -> Dict[int, Std]:
    """
    Certain APIs take redirect settings either as a single value (e.g. apply to all
    local ranks) or as an explicit user-provided mapping. This method is a convenience
    method that converts a value or mapping into a mapping.

    Example:
    ::

     to_map(Std.OUT, local_world_size=2) # returns: {0: Std.OUT, 1: Std.OUT}
     to_map({1: Std.OUT}, local_world_size=2) # returns: {0: Std.NONE, 1: Std.OUT}
     to_map({0: Std.OUT, 1: Std.OUT}, local_world_size=2) # returns: {0: Std.OUT, 1: Std.OUT}
    """
    if isinstance(val_or_map, Std):
        return dict.fromkeys(range(local_world_size), val_or_map)
    else:
        map = {}
        for i in range(local_world_size):
            map[i] = val_or_map.get(i, Std.NONE)
        return map


@dataclass
class LogsDest:
    """
    For each log type, holds mapping of local rank ids to file paths.
    """
    stdouts: Dict[int, str] = field(default_factory=dict)
    stderrs: Dict[int, str] = field(default_factory=dict)
    tee_stdouts: Dict[int, str] = field(default_factory=dict)
    tee_stderrs: Dict[int, str] = field(default_factory=dict)
    error_files: Dict[int, str] = field(default_factory=dict)


class LogsSpecs(ABC):
    """
    Defines logs processing and redirection for each worker process.

    Args:
        log_dir:
            Base directory where logs will be written.
        redirects:
            Streams to redirect to files. Pass a single ``Std``
            enum to redirect for all workers, or a mapping keyed
            by local_rank to selectively redirect.
        tee:
            Streams to duplicate to stdout/stderr.
            Pass a single ``Std`` enum to duplicate streams for all workers,
            or a mapping keyed by local_rank to selectively duplicate.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        redirects: Union[Std, Dict[int, Std]] = Std.NONE,
        tee: Union[Std, Dict[int, Std]] = Std.NONE,
        local_ranks_filter: Optional[Set[int]] = None,
    ) -> None:
        self._root_log_dir = log_dir
        self._redirects = redirects
        self._tee = tee
        self._local_ranks_filter = local_ranks_filter

    @abstractmethod
    def reify(self, envs: Dict[int, Dict[str, str]],) -> LogsDest:
        """
        Given the environment variables, builds destination of log files for each of the local ranks.

        Envs parameter contains env variables dict for each of the local ranks, where entries are defined in:
        :func:`~torchelastic.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent._start_workers`.
        """
        pass

    @property
    @abstractmethod
    def root_log_dir(self) -> str:
        pass

class DefaultLogsSpecs(LogsSpecs):
    """
    Default LogsSpecs implementation:

    - `log_dir` will be created if it doesn't exist
    - Generates nested folders for each attempt and rank.
    """
    def __init__(
        self,
        log_dir: Optional[str] = None,
        redirects: Union[Std, Dict[int, Std]] = Std.NONE,
        tee: Union[Std, Dict[int, Std]] = Std.NONE,
        local_ranks_filter: Optional[Set[int]] = None,
    ) -> None:
        if log_dir != os.devnull:
            if not log_dir:
                log_dir = tempfile.mkdtemp(prefix="torchelastic_")
            elif not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            else:
                if os.path.isfile(log_dir):
                    raise NotADirectoryError(f"log_dir: {log_dir} is a file")
        super().__init__(log_dir, redirects, tee, local_ranks_filter)
        # initialized only once
        self._run_log_dir = None

    @property
    def root_log_dir(self) -> str:
        return str(self._root_log_dir)

    def _make_log_dir(self, log_dir: Optional[str], rdzv_run_id: str):
        base_log_dir = log_dir or tempfile.mkdtemp(prefix="torchelastic_")
        os.makedirs(base_log_dir, exist_ok=True)
        dir = tempfile.mkdtemp(prefix=f"{rdzv_run_id}_", dir=base_log_dir)
        logger.info("log directory set to: %s", dir)
        return dir

    def reify(self, envs: Dict[int, Dict[str, str]],) -> LogsDest:
        """
        Uses following scheme to build log destination paths:

        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/stdout.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/stderr.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/error.json`
        """
        nprocs = len(envs)
        global_env = {}  # use only to query properies that are not dependent on a rank
        if nprocs > 0:
            global_env = envs[0]
        else:
            logger.warning("Empty envs map provided when defining logging destinations.")
        # Keys are always defined, but values can be missing in unit tests
        run_id = global_env.get("TORCHELASTIC_RUN_ID", "test_run_id")
        restart_count = global_env.get("TORCHELASTIC_RESTART_COUNT", "0")

        attempt_log_dir: str = ""
        if self._root_log_dir != os.devnull:
            if not self._run_log_dir:
                self._run_log_dir = self._make_log_dir(self._root_log_dir, run_id)

            attempt_log_dir = os.path.join(self._run_log_dir, f"attempt_{restart_count}")  # type: ignore[call-overload]
            shutil.rmtree(attempt_log_dir, ignore_errors=True)
            os.makedirs(attempt_log_dir)

        if self._root_log_dir == os.devnull:
            attempt_log_dir = os.devnull

        # create subdirs for each local rank in the logs_dir
        # logs_dir
        #       |- 0
        #          |- error.json
        #          |- stdout.log
        #          |- stderr.log
        #       |- ...
        #       |- (nprocs-1)
        redirs = to_map(self._redirects, nprocs)
        ts = to_map(self._tee, nprocs)

        # to tee stdout/stderr we first redirect into a file
        # then tail -f stdout.log/stderr.log so add tee settings to redirects
        for local_rank, tee_std in ts.items():
            redirect_std = redirs[local_rank]
            redirs[local_rank] = redirect_std | tee_std

        SYS_STREAM = ""  # special case to indicate to output to console
        stdouts = dict.fromkeys(range(nprocs), SYS_STREAM)
        stderrs = dict.fromkeys(range(nprocs), SYS_STREAM)
        tee_stdouts: Dict[int, str] = {}
        tee_stderrs: Dict[int, str] = {}
        error_files = {}

        for local_rank in range(nprocs):

            if attempt_log_dir == os.devnull:
                tee_stdouts[local_rank] = os.devnull
                tee_stderrs[local_rank] = os.devnull
                error_files[local_rank] = os.devnull
                envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = ""
            else:
                clogdir = os.path.join(attempt_log_dir, str(local_rank))
                os.mkdir(clogdir)

                rd = redirs[local_rank]
                if (rd & Std.OUT) == Std.OUT:
                    stdouts[local_rank] = os.path.join(clogdir, "stdout.log")
                if (rd & Std.ERR) == Std.ERR:
                    stderrs[local_rank] = os.path.join(clogdir, "stderr.log")

                t = ts[local_rank]
                if t & Std.OUT == Std.OUT:
                    tee_stdouts[local_rank] = stdouts[local_rank]
                if t & Std.ERR == Std.ERR:
                    tee_stderrs[local_rank] = stderrs[local_rank]

                if self._local_ranks_filter and local_rank not in self._local_ranks_filter:
                    # If stream is tee'd, only write to file, but don't tail
                    if local_rank in tee_stdouts:
                        tee_stdouts.pop(local_rank, None)
                    if local_rank in tee_stderrs:
                        tee_stderrs.pop(local_rank, None)

                    # If stream is not redirected, don't print
                    if stdouts[local_rank] == SYS_STREAM:
                        stdouts[local_rank] = os.devnull
                    if stderrs[local_rank] == SYS_STREAM:
                        stderrs[local_rank] = os.devnull

                error_file = os.path.join(clogdir, "error.json")
                error_files[local_rank] = error_file
                logger.info("Setting worker%s reply file to: %s", local_rank, error_file)
                envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = error_file

        return LogsDest(stdouts, stderrs, tee_stdouts, tee_stderrs, error_files)

    def __repr__(self) -> str:
        return (
            f"DefaultLogsSpecs(root_log_dir={self._root_log_dir}, redirects={self._redirects}, "
            f"tee={self._tee}, local_ranks_filter={self._local_ranks_filter})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DefaultLogsSpecs):
            return False

        return (
            self._root_log_dir == other._root_log_dir
            and self._redirects == other._redirects
            and self._tee == other._tee
            and self._local_ranks_filter == other._local_ranks_filter
        )


@dataclass
class RunProcsResult:
    """
    Results of a completed run of processes started with ``start_processes()``. Returned by ``PContext``.

    Note the following:

    1. All fields are mapped by local rank
    2. ``return_values`` - only populated for functions (not the binaries).
    3. ``stdouts`` - path to stdout.log (empty string if no redirect)
    4. ``stderrs`` - path to stderr.log (empty string if no redirect)

    """

    return_values: Dict[int, Any] = field(default_factory=dict)
    failures: Dict[int, ProcessFailure] = field(default_factory=dict)
    stdouts: Dict[int, str] = field(default_factory=dict)
    stderrs: Dict[int, str] = field(default_factory=dict)

    def is_failed(self) -> bool:
        return len(self.failures) > 0


class PContext(abc.ABC):
    """
    The base class that standardizes operations over a set of processes that are launched via different mechanisms.

    The name ``PContext`` is intentional to disambiguate with ``torch.multiprocessing.ProcessContext``.

    .. warning:: stdouts and stderrs should ALWAYS be a superset of
                 tee_stdouts and tee_stderrs (respectively) this is b/c
                 tee is implemented as a redirect + tail -f <stdout/stderr.log>
    """

    def __init__(
        self,
        name: str,
        entrypoint: Union[Callable, str],
        args: Dict[int, Tuple],
        envs: Dict[int, Dict[str, str]],
        logs_specs: LogsSpecs,
        log_line_prefixes: Optional[Dict[int, str]] = None,

    ):
        self.name = name
        # validate that all mappings have the same number of keys and
        # all local ranks are accounted for
        nprocs = len(args)

        # TODO log_line_prefixes can be exanded too
        logs_dest = logs_specs.reify(envs)

        _validate_full_rank(logs_dest.stdouts, nprocs, "stdouts")
        _validate_full_rank(logs_dest.stderrs, nprocs, "stderrs")

        self.entrypoint = entrypoint
        self.args = args
        self.envs = envs
        self.stdouts = logs_dest.stdouts
        self.stderrs = logs_dest.stderrs
        self.error_files = logs_dest.error_files
        self.nprocs = nprocs

        self._stdout_tail = TailLog(name, logs_dest.tee_stdouts, sys.stdout, log_line_prefixes)
        self._stderr_tail = TailLog(name, logs_dest.tee_stderrs, sys.stderr, log_line_prefixes)

    def start(self) -> None:
        """Start processes using parameters defined in the constructor."""
        signal.signal(signal.SIGTERM, _terminate_process_handler)
        signal.signal(signal.SIGINT, _terminate_process_handler)
        if not IS_WINDOWS:
            signal.signal(signal.SIGHUP, _terminate_process_handler)
            signal.signal(signal.SIGQUIT, _terminate_process_handler)
        self._start()
        self._stdout_tail.start()
        self._stderr_tail.start()

    @abc.abstractmethod
    def _start(self) -> None:
        """Start processes using strategy defined in a particular context."""
        raise NotImplementedError

    @abc.abstractmethod
    def _poll(self) -> Optional[RunProcsResult]:
        """
        Poll the run status of the processes running under this context.
        This method follows an "all-or-nothing" policy and returns
        a ``RunProcessResults`` object if either all processes complete
        successfully or any process fails. Returns ``None`` if
        all processes are still running.
        """
        raise NotImplementedError

    def wait(self, timeout: float = -1, period: float = 1) -> Optional[RunProcsResult]:
        """
        Wait for the specified ``timeout`` seconds, polling every ``period`` seconds
        for the processes to be done. Returns ``None`` if the processes are still running
        on timeout expiry. Negative timeout values are interpreted as "wait-forever".
        A timeout value of zero simply queries the status of the processes (e.g. equivalent
        to a poll).

        ..note: Multiprocessing library registers SIGTERM and SIGINT signal handlers that raise
                ``SignalException`` when the signals received. It is up to the consumer of the code
                to properly handle the exception. It is important not to swallow the exception otherwise
                the process would not terminate. Example of the typical workflow can be:

        .. code-block:: python
            pc = start_processes(...)
            try:
                pc.wait(1)
                .. do some other work
            except SignalException as e:
                pc.shutdown(e.sigval, timeout=30)

        If SIGTERM or SIGINT occurs, the code above will try to shutdown child processes by propagating
        received signal. If child processes will not terminate in the timeout time, the process will send
        the SIGKILL.
        """
        if timeout == 0:
            return self._poll()

        if timeout < 0:
            timeout = sys.maxsize

        expiry = time.time() + timeout
        while time.time() < expiry:
            pr = self._poll()
            if pr:
                return pr
            time.sleep(period)

        return None

    @abc.abstractmethod
    def pids(self) -> Dict[int, int]:
        """Return pids of processes mapped by their respective local_ranks."""
        raise NotImplementedError

    @abc.abstractmethod
    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        r"""
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).
        """
        raise NotImplementedError

    def close(
        self, death_sig: Optional[signal.Signals] = None, timeout: int = 30
    ) -> None:
        r"""
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).

        Args:
            death_sig: Death signal to terminate processes.
            timeout: Time to wait for processes to finish, if process is
                still alive after this time, it will be terminated via SIGKILL.
        """
        if not death_sig:
            death_sig = _get_default_signal()
        self._close(death_sig=death_sig, timeout=timeout)
        if self._stdout_tail:
            self._stdout_tail.stop()
        if self._stderr_tail:
            self._stderr_tail.stop()


def get_std_cm(std_rd: str, redirect_fn):
    if IS_WINDOWS or IS_MACOS or not std_rd:
        return nullcontext()
    else:
        return redirect_fn(std_rd)


def _wrap(
    local_rank: int,
    fn: Callable,
    args: Dict[int, Tuple],
    envs: Dict[int, Dict[str, str]],
    stdout_redirects: Dict[int, str],  # redirect file for stdout (to console if None)
    stderr_redirects: Dict[int, str],  # redirect file for stderr (to console if None)
    ret_vals: Dict[int, mp.SimpleQueue],
    queue_finished_reading_event: synchronize.Event,
) -> None:
    # get the per-rank params up front so we fail fast if no mapping is found
    args_ = args[local_rank]
    env_ = envs[local_rank]
    ret_val_ = ret_vals[local_rank]

    stdout_rd = stdout_redirects[local_rank]
    stderr_rd = stderr_redirects[local_rank]

    stdout_cm = get_std_cm(stdout_rd, redirect_stdout)
    stderr_cm = get_std_cm(stderr_rd, redirect_stderr)

    for k, v in env_.items():
        os.environ[k] = v

    with stdout_cm, stderr_cm:
        ret = record(fn)(*args_)
    ret_val_.put(ret)
    queue_finished_reading_event.wait()


class MultiprocessContext(PContext):
    """``PContext`` holding worker processes invoked as a function."""

    def __init__(
        self,
        name: str,
        entrypoint: Callable,
        args: Dict[int, Tuple],
        envs: Dict[int, Dict[str, str]],
        start_method: str,
        logs_specs: LogsSpecs,
        log_line_prefixes: Optional[Dict[int, str]] = None,
    ):
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            logs_specs,
            log_line_prefixes,
        )

        self.start_method = start_method
        # each ret_val queue will always contain a single element.
        self._ret_vals = {
            local_rank: mp.get_context(self.start_method).SimpleQueue()
            for local_rank in range(self.nprocs)
        }

        # see comments in ``join()`` for what this is
        self._return_values: Dict[int, Any] = {}
        self._pc: Optional[mp.ProcessContext] = None
        # Note: set method should ONLY be invoked for the use case when all processes finished
        # successfully. If any process died on event.wait() calling set() method will deadlock.
        self._worker_finished_event = mp.get_context(self.start_method).Event()

    def _start(self):
        if self._pc:
            raise ValueError(
                "The process context already initialized."
                " Most likely the start method got called twice."
            )
        self._pc = mp.start_processes(
            fn=_wrap,
            args=(
                self.entrypoint,
                self.args,
                self.envs,
                self.stdouts,
                self.stderrs,
                self._ret_vals,
                self._worker_finished_event,
            ),
            nprocs=self.nprocs,
            join=False,
            daemon=False,
            start_method=self.start_method,
        )

    def _is_done(self) -> bool:
        return len(self._return_values) == self.nprocs

    def _poll(self) -> Optional[RunProcsResult]:
        assert self._pc is not None  # assertion for mypy type checker

        try:
            # torch.mp.ProcessContext Throws an Exception if some/all of
            # worker processes failed
            # timeout < 0 checks worker status and return immediately
            # Join will never return success since we use synchronize.Event to wait
            # for all processes to finish.
            self._pc.join(-1)

            # IMPORTANT: we use multiprocessing.Queue to carry worker return values
            # back to the parent, the worker process will wait before terminating
            # until all the buffered items are fed by the feeder thread to the underlying
            # pipe. Hence to prevent deadlocks on large return values,
            # we opportunistically try queue.get on each join call
            # See: https://docs.python.org/2/library/multiprocessing.html#all-platforms
            for local_rank in range(0, self.nprocs):
                return_queue = self._ret_vals[local_rank]
                if not return_queue.empty():
                    # save the return values temporarily into a member var
                    self._return_values[local_rank] = return_queue.get()

            if self._is_done():
                # we should ALWAYS have ALL the return values when all the processes are done
                self._worker_finished_event.set()

                # At this point workers finished running the user function
                # But the child process might still have not exited. Wait for them.
                # pc.join() blocks [forever] until "a" proc exits. Loop until all of them exits.
                while not self._pc.join():
                    logger.debug("entrypoint fn finished, waiting for all child procs to exit...")

                _validate_full_rank(
                    self._return_values, self.nprocs, "return_value queue"
                )
                self.close()
                return RunProcsResult(
                    return_values=self._return_values,
                    stdouts=self.stdouts,
                    stderrs=self.stderrs,
                )
            else:
                return None
        except (mp.ProcessRaisedException, mp.ProcessExitedException) as e:
            failed_local_rank = e.error_index

            # entrypoint for MultiprocessContext will always be a Callable
            fn_name = self.entrypoint.__qualname__  # type: ignore[union-attr]
            failed_proc = self._pc.processes[failed_local_rank]
            error_filepath = self.error_files[failed_local_rank]

            logger.exception(
                "failed (exitcode: %s)"
                " local_rank: %s (pid: %s)"
                " of fn: %s (start_method: %s)",
                failed_proc.exitcode,
                failed_local_rank, e.pid,
                fn_name, self.start_method,
            )

            self.close()
            return RunProcsResult(
                failures={
                    failed_local_rank: ProcessFailure(
                        local_rank=failed_local_rank,
                        pid=e.pid,
                        exitcode=failed_proc.exitcode,
                        error_file=error_filepath,
                    )
                },
                stdouts=self.stdouts,
                stderrs=self.stderrs,
            )

    def pids(self) -> Dict[int, int]:
        assert self._pc is not None  # assertion for mypy type checking
        return dict(enumerate(self._pc.pids()))

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        if not self._pc:
            return
        for proc in self._pc.processes:
            if proc.is_alive():
                logger.warning("Closing process %s via signal %s", proc.pid, death_sig.name)
                try:
                    os.kill(proc.pid, death_sig)
                except ProcessLookupError:
                    # If the process exited because of some reason,
                    # `ProcessLookupError` will be raised, it is safe to ignore it.
                    pass
        end = time.monotonic() + timeout
        for proc in self._pc.processes:
            time_to_wait = end - time.monotonic()
            if time_to_wait <= 0:
                break
            proc.join(time_to_wait)
        for proc in self._pc.processes:
            if proc.is_alive():
                logger.warning(
                    "Unable to shutdown process %s via %s, forcefully exiting via %s",
                    proc.pid, death_sig, _get_kill_signal()
                )
                try:
                    os.kill(proc.pid, _get_kill_signal())
                except ProcessLookupError:
                    # If the process exited because of some reason,
                    # `ProcessLookupError` will be raised, it is safe to ignore it.
                    pass
            proc.join()

class SubprocessContext(PContext):
    """``PContext`` holding worker processes invoked as a binary."""

    def __init__(
        self,
        name: str,
        entrypoint: str,
        args: Dict[int, Tuple],
        envs: Dict[int, Dict[str, str]],
        logs_specs: LogsSpecs,
        log_line_prefixes: Optional[Dict[int, str]] = None,

    ):
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            logs_specs,
            log_line_prefixes,
        )

        # state vector; _vdone[local_rank] -> is local_rank finished or not
        self._running_local_ranks: Set[int] = set(range(self.nprocs))
        self._failures: Dict[int, ProcessFailure] = {}
        self.subprocess_handlers: Dict[int, SubprocessHandler] = {}

    def _start(self):
        if self.subprocess_handlers:
            raise ValueError(
                "The subprocess handlers already initialized. Most likely the start method got called twice."
            )
        self.subprocess_handlers = {
            local_rank: get_subprocess_handler(
                entrypoint=self.entrypoint,  # type: ignore[arg-type] # entrypoint is always a str
                args=self.args[local_rank],
                env=self.envs[local_rank],
                stdout=self.stdouts[local_rank],
                stderr=self.stderrs[local_rank],
                local_rank_id=local_rank,
            )
            for local_rank in range(self.nprocs)
        }

    def _poll(self) -> Optional[RunProcsResult]:
        done_local_ranks = set()
        for local_rank in self._running_local_ranks:
            handler = self.subprocess_handlers[local_rank]
            exitcode = handler.proc.poll()
            if exitcode is not None:
                done_local_ranks.add(local_rank)
                if exitcode != 0:  # failed or signaled
                    self._failures[local_rank] = ProcessFailure(
                        local_rank=local_rank,
                        pid=handler.proc.pid,
                        exitcode=exitcode,
                        error_file=self.error_files[local_rank],
                    )
                # else: --> succeeded; nothing to do

        self._running_local_ranks.difference_update(done_local_ranks)

        # if ALL procs are finished or ANY have failed
        if not self._running_local_ranks or self._failures:
            self.close()  # terminate all running procs
            result = RunProcsResult(
                failures=self._failures,
                stdouts=self.stdouts,
                stderrs=self.stderrs,
            )
            if result.is_failed():
                first_failure = min(result.failures.values(), key=lambda f: f.timestamp)
                logger.error(
                    "failed (exitcode: %s)"
                    " local_rank: %s (pid: %s)"
                    " of binary: %s",
                    first_failure.exitcode, first_failure.local_rank, first_failure.pid, self.entrypoint
                )
            else:
                # Populate return with dummy values. This provides consistency with MultiprocessingHandler
                result.return_values = dict.fromkeys(range(self.nprocs))

            return result
        else:  # there are no failures and procs still running
            return None

    def pids(self) -> Dict[int, int]:
        return {
            local_rank: sh.proc.pid
            for local_rank, sh in self.subprocess_handlers.items()
        }

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        if not self.subprocess_handlers:
            return
        for handler in self.subprocess_handlers.values():
            if handler.proc.poll() is None:
                logger.warning(
                    "Sending process %s closing signal %s", handler.proc.pid, death_sig.name
                )
                handler.close(death_sig=death_sig)
        end = time.monotonic() + timeout
        for handler in self.subprocess_handlers.values():
            time_to_wait = end - time.monotonic()
            if time_to_wait <= 0:
                break
            try:
                handler.proc.wait(time_to_wait)
            except subprocess.TimeoutExpired:
                # Ignore the timeout expired exception, since
                # the child process will be forcefully terminated via SIGKILL
                pass
        for handler in self.subprocess_handlers.values():
            if handler.proc.poll() is None:
                logger.warning(
                    "Unable to shutdown process %s via %s, forcefully exiting via %s",
                    handler.proc.pid, death_sig, _get_kill_signal()
                )
                handler.close(death_sig=_get_kill_signal())
                handler.proc.wait()
