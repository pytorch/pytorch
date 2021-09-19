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
import signal
import subprocess
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import IntFlag
from multiprocessing import synchronize
from types import FrameType
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure, record
from torch.distributed.elastic.multiprocessing.redirects import (
    redirect_stderr,
    redirect_stdout,
)
from torch.distributed.elastic.multiprocessing.tail_log import TailLog

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


log = logging.getLogger(__name__)


class SignalException(Exception):
    """
    Exception is raised inside the torchelastic agent process by the termination handler
    if the death signal got received by the process.
    """

    def __init__(self, msg: str, sigval: signal.Signals) -> None:
        super().__init__(msg)
        self.sigval = sigval


def _terminate_process_handler(signum: int, frame: FrameType) -> None:
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
    """
    Get the kill signal. SIGKILL for unix, CTRL_C_EVENT for windows.
    """
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined] # noqa: F821
    else:
        return signal.SIGKILL


def _get_default_signal() -> signal.Signals:
    """
    Get the default termination signal. SIGTERM for unix, CTRL_C_EVENT for windows.
    """
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

        def to_std(v):
            v = int(v)
            for s in Std:
                if s == v:
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
        return {i: val_or_map for i in range(local_world_size)}
    else:
        map = {}
        for i in range(local_world_size):
            map[i] = val_or_map.get(i, Std.NONE)
        return map


@dataclass
class RunProcsResult:
    """
    Results of a completed run of processes started with ``start_processes()``.
    Returned by ``PContext``.

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
    The base class that standardizes operations over a set of processes
    that are launched via different mechanisms. The name ``PContext``
    is intentional to disambiguate with ``torch.multiprocessing.ProcessContext``.

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
        stdouts: Dict[int, str],
        stderrs: Dict[int, str],
        tee_stdouts: Dict[int, str],
        tee_stderrs: Dict[int, str],
        error_files: Dict[int, str],
    ):
        self.name = name
        # validate that all mappings have the same number of keys and
        # all local ranks are accounted for
        nprocs = len(args)
        _validate_full_rank(stdouts, nprocs, "stdouts")
        _validate_full_rank(stderrs, nprocs, "stderrs")

        self.entrypoint = entrypoint
        self.args = args
        self.envs = envs
        self.stdouts = stdouts
        self.stderrs = stderrs
        self.error_files = error_files
        self.nprocs = nprocs

        self._stdout_tail = TailLog(name, tee_stdouts, sys.stdout)
        self._stderr_tail = TailLog(name, tee_stderrs, sys.stderr)

    def start(self) -> None:
        """
        Start processes using parameters defined in the constructor.
        """
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
        """
        Start processes using strategy defined in a particular context.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _poll(self) -> Optional[RunProcsResult]:
        """
        Polls the run status of the processes running under this context.
        This method follows an "all-or-nothing" policy and returns
        a ``RunProcessResults`` object if either all processes complete
        successfully or any process fails. Returns ``None`` if
        all processes are still running.
        """
        raise NotImplementedError()

    def wait(self, timeout: float = -1, period: float = 1) -> Optional[RunProcsResult]:
        """
        Waits for the specified ``timeout`` seconds, polling every ``period`` seconds
        for the processes to be done. Returns ``None`` if the processes are still running
        on timeout expiry. Negative timeout values are interpreted as "wait-forever".
        A timeout value of zero simply queries the status of the processes (e.g. equivalent
        to a poll).

        ..note: Multiprocesing library registers SIGTERM and SIGINT signal handlers that raise
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
        """
        Returns pids of processes mapped by their respective local_ranks
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        r"""
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).
        """
        raise NotImplementedError()

    def close(
        self, death_sig: Optional[signal.Signals] = None, timeout: int = 30
    ) -> None:
        r"""
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).

        Args:
            death_sig: Death signal to terminate porcesses.
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


class _nullcontext(AbstractContextManager):
    # TODO remove and replace in favor of contextlib.nullcontext
    # when torch drops support for python3.6
    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


def get_std_cm(std_rd: str, redirect_fn):
    if IS_WINDOWS or IS_MACOS or not std_rd:
        return _nullcontext()
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
    """
    ``PContext`` holding worker processes invoked as a function.
    """

    def __init__(
        self,
        name: str,
        entrypoint: Callable,
        args: Dict[int, Tuple],
        envs: Dict[int, Dict[str, str]],
        stdouts: Dict[int, str],
        stderrs: Dict[int, str],
        tee_stdouts: Dict[int, str],
        tee_stderrs: Dict[int, str],
        error_files: Dict[int, str],
        start_method: str,
    ):
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            stdouts,
            stderrs,
            tee_stdouts,
            tee_stderrs,
            error_files,
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
                # Wait untill all processes are finished. At this point workers finished executing
                # user function
                self._pc.join()
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

            log.error(
                f"failed (exitcode: {failed_proc.exitcode})"
                f" local_rank: {failed_local_rank} (pid: {e.pid})"
                f" of fn: {fn_name} (start_method: {self.start_method})",
                exc_info=True,
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
        return {local_rank: pid for local_rank, pid in enumerate(self._pc.pids())}

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        if not self._pc:
            return
        for proc in self._pc.processes:
            if proc.is_alive():
                log.warning(f"Closing process {proc.pid} via signal {death_sig.name}")
                try:
                    os.kill(proc.pid, death_sig)
                except ProcessLookupError:
                    # If the process exited because of some reason,
                    # `ProcessLookupError` will be rasied, it is safe to ignore it.
                    pass
        end = time.monotonic() + timeout
        for proc in self._pc.processes:
            time_to_wait = end - time.monotonic()
            if time_to_wait <= 0:
                break
            proc.join(time_to_wait)
        for proc in self._pc.processes:
            if proc.is_alive():
                log.warning(
                    f"Unable to shutdown process {proc.pid} via {death_sig}, forcefully exitting via {_get_kill_signal()}"
                )
                try:
                    os.kill(proc.pid, _get_kill_signal())
                except ProcessLookupError:
                    # If the process exited because of some reason,
                    # `ProcessLookupError` will be rasied, it is safe to ignore it.
                    pass
            proc.join()


class SubprocessHandler:
    """
    Convenience wrapper around python's ``subprocess.Popen``. Keeps track of
    meta-objects associated to the process (e.g. stdout and stderr redirect fds).
    """

    def __init__(
        self,
        entrypoint: str,
        args: Tuple,
        env: Dict[str, str],
        stdout: str,
        stderr: str,
    ):
        self._stdout = open(stdout, "w") if stdout else None
        self._stderr = open(stderr, "w") if stderr else None
        # inherit parent environment vars
        env_vars = os.environ.copy()
        env_vars.update(env)

        args_str = (entrypoint, *[str(e) for e in args])
        self.proc: subprocess.Popen = self._popen(args_str, env_vars)

    def _popen(self, args: Tuple, env: Dict[str, str]) -> subprocess.Popen:
        return subprocess.Popen(
            # pyre-fixme[6]: Expected `Union[typing.Sequence[Union[_PathLike[bytes],
            #  _PathLike[str], bytes, str]], bytes, str]` for 1st param but got
            #  `Tuple[str, *Tuple[Any, ...]]`.
            args=args,
            env=env,
            stdout=self._stdout,
            stderr=self._stderr,
        )

    def close(self, death_sig: Optional[signal.Signals] = None) -> None:
        if not death_sig:
            death_sig = _get_default_signal()
        self.proc.send_signal(death_sig)
        if self._stdout:
            self._stdout.close()
        if self._stderr:
            self._stderr.close()


class SubprocessContext(PContext):
    """
    ``PContext`` holding worker processes invoked as a binary.
    """

    def __init__(
        self,
        name: str,
        entrypoint: str,
        args: Dict[int, Tuple],
        envs: Dict[int, Dict[str, str]],
        stdouts: Dict[int, str],
        stderrs: Dict[int, str],
        tee_stdouts: Dict[int, str],
        tee_stderrs: Dict[int, str],
        error_files: Dict[int, str],
    ):
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            stdouts,
            stderrs,
            tee_stdouts,
            tee_stderrs,
            error_files,
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
            local_rank: SubprocessHandler(
                entrypoint=self.entrypoint,  # type: ignore[arg-type] # entrypoint is always a str
                args=self.args[local_rank],
                env=self.envs[local_rank],
                stdout=self.stdouts[local_rank],
                stderr=self.stderrs[local_rank],
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
                log.error(
                    f"failed (exitcode: {first_failure.exitcode})"
                    f" local_rank: {first_failure.local_rank} (pid: {first_failure.pid})"
                    f" of binary: {self.entrypoint}"
                )
            else:
                # Populate return with dummy values. This provides consistency with MultiprocessingHandler
                result.return_values = {
                    local_rank: None for local_rank in range(self.nprocs)
                }

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
                log.warning(
                    f"Sending process {handler.proc.pid} closing signal {death_sig.name}"
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
                log.warning(
                    f"Unable to shutdown process {handler.proc.pid} via {death_sig}, forcefully exitting via {_get_kill_signal()}"
                )
                handler.close(death_sig=_get_kill_signal())
                handler.proc.wait()
