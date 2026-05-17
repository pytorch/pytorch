import base64
import functools
import itertools
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, TypeVar
from typing_extensions import Never, ParamSpec

from torch._inductor import config
from torch._inductor.codecache import torch_key
from torch._inductor.compile_worker.subproc_pool_worker import (
    _recv_msg,
    _send_msg,
    _SubprocExceptionInfo,
    MsgHeader,
    SubprocException,
    SubprocKind,
    SubprocMain,
    SubprocPickler,
)
from torch._inductor.compile_worker.timer import Timer
from torch._inductor.utils import get_ld_library_path, python_subprocess_env
from torch._utils_internal import find_compile_subproc_binary
from torch.monitor import _WaitCounter, _WaitCounterTracker


log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")

__all__ = [
    "SubprocException",
    "SubprocKind",
    "SubprocMain",
    "SubprocPickler",
    "SubprocPool",
    "TestException",
    "raise_testexc",
]


class SubprocPool:
    """
    Mimic a concurrent.futures.ProcessPoolExecutor, but wrap it in
    a subprocess.Popen() to try to avoid issues with forking/spawning
    """

    def __init__(
        self,
        nprocs: int,
        pickler: SubprocPickler | None = None,
        kind: SubprocKind = SubprocKind.FORK,
        quiesce: bool = False,
    ) -> None:
        entry = os.path.join(os.path.dirname(__file__), "__main__.py")
        self.pickler = pickler or SubprocPickler()
        self.kind = kind

        subproc_read_fd, write_fd = os.pipe()
        read_fd, subproc_write_fd = os.pipe()
        self.write_pipe = os.fdopen(write_fd, "wb")
        self.read_pipe = os.fdopen(read_fd, "rb")
        torch_key_str = base64.b64encode(torch_key()).decode("utf-8")

        cmd = [
            sys.executable,
            entry,
        ]
        if (binary := find_compile_subproc_binary()) is not None:
            cmd = [binary]

        args = [
            f"--pickler={self.pickler.__class__.__module__}.{self.pickler.__class__.__name__}",
            f"--kind={self.kind.value}",
            f"--workers={nprocs}",
            f"--parent={os.getpid()}",
            f"--read-fd={str(subproc_read_fd)}",
            f"--write-fd={str(subproc_write_fd)}",
            f"--torch-key={torch_key_str}",
        ]
        cmd.extend(args)
        log_path = None
        self.log_file = None

        if config.worker_suppress_logging:
            log_path = os.devnull
            log.info("Suppressing compile worker output due to config")
        else:
            log_path = config.torchinductor_worker_logpath
            if not log_path:
                log_path = config.get_worker_log_path()

        if log_path:
            # pyrefly: ignore [bad-assignment]
            self.log_file = open(log_path, "w")  # noqa:SIM115

        self.process = subprocess.Popen(
            cmd,
            env={
                **python_subprocess_env(),
                # Safeguard against creating a SubprocPool in the subprocess.
                "TORCH_WARM_POOL": "0",
                # Some internal usages need a modified LD_LIBRARY_PATH.
                "LD_LIBRARY_PATH": get_ld_library_path(),
            },
            pass_fds=(subproc_read_fd, subproc_write_fd),
            stdout=self.log_file,
            stderr=self.log_file,
        )
        self.write_lock = threading.Lock()
        self.read_thread = threading.Thread(
            target=self._read_thread, name="InductorSubproc", daemon=True
        )

        self.futures_lock = threading.Lock()
        self.pending_futures: dict[int, Future[Any]] = {}
        # The pending waitcounter, is used to indicate the time when we have any specific job running.
        self.pending_waitcounters: dict[int, Any] = {}
        self.job_id_count = itertools.count()

        # The running waitcounter indicates the time when the SubProcPool object exists.
        self.running = True
        self.running_waitcounter = _WaitCounter(
            "pytorch.wait_counter.subproc_pool.running"
        ).guard()
        self.running_waitcounter.__enter__()

        # The quiesce waitcounter indicates when the job is in a quiesced state.
        self.quiesce_waitcounter: _WaitCounterTracker | None = None

        # Firstjob is used to capture the time from when the firstjob is queued, to when the first job is done.
        self.firstjob = True
        self.firstjob_id: int | None = None
        self.firstjob_waitcounter = _WaitCounter(
            "pytorch.wait_counter.subproc_pool.first_job"
        ).guard()

        if quiesce:
            self.timer: Timer | None = Timer(
                config.quiesce_async_compile_time, self.quiesce
            )
        else:
            self.timer = None

        # Start thread last to ensure all member variables are initialized
        # before any access.
        self.read_thread.start()

    def submit(
        self, job_fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
    ) -> Future[_T]:
        if args or kwargs:
            # pyrefly: ignore [bad-assignment]
            job_fn = functools.partial(job_fn, *args, **kwargs)
        job_data = self.pickler.dumps(job_fn)
        future: Future[_T]
        with self.futures_lock:
            job_id = next(self.job_id_count)
            self.pending_futures[job_id] = future = Future()
            self.pending_waitcounters[job_id] = _WaitCounter(
                "pytorch.wait_counter.subproc_pool.job"
            ).guard()
            self.pending_waitcounters[job_id].__enter__()
            if self.quiesce_waitcounter:
                self.firstjob = True
                self.quiesce_waitcounter.__exit__()
                self.quiesce_waitcounter = None
            # This can be entered from either quiesce wakeup, or from startup.
            if self.firstjob:
                self.firstjob_id = job_id
                self.firstjob_waitcounter.__enter__()
                self.firstjob = False
        future.set_running_or_notify_cancel()
        self._send(MsgHeader.JOB, job_id, job_data)
        return future

    def _send(self, msg_header: MsgHeader, job_id: int = -1, data: bytes = b"") -> None:
        with self.write_lock:
            if not self.running:
                raise RuntimeError("Attempting to use a closed pool")
            _send_msg(self.write_pipe, msg_header, job_id, data)

    def _read_thread(self) -> None:
        while True:
            data = b""
            job_id = -1
            try:
                msg_header, job_id, data = _recv_msg(self.read_pipe)
            except Exception:
                # Something went wrong during the read. There's no way we have a
                # valid msg.
                log.exception("failure in subproc_pool._recv_msg")
                msg_header = MsgHeader.ERROR

            if msg_header != MsgHeader.JOB:
                # read_pipe returned None or got exception
                if self.running:
                    log.warning("SubprocPool unclean exit")
                    self.running = False
                    self.running_waitcounter.__exit__()
                self.read_pipe.close()
                # Cancel all the pending futures.
                self.shutdown()
                return

            try:
                result = self.pickler.loads(data)
            except Exception as e:
                # Something went wrong unpickling. We have a job_id so just
                # notify that particular future and continue on.
                log.exception("unpickle failure in SubprocPool._read_thread")
                result = e

            with self.futures_lock:
                if not self.running:
                    return
                if self.timer:
                    self.timer.record_call()
                if isinstance(result, _SubprocExceptionInfo):
                    # An exception occurred in the submitted job
                    self.pending_futures[job_id].set_exception(
                        SubprocException(result.details)
                    )
                elif isinstance(result, Exception):
                    # An exception occurred in some of our subprocess machinery.
                    self.pending_futures[job_id].set_exception(result)
                else:
                    self.pending_futures[job_id].set_result(result)

                self.pending_waitcounters[job_id].__exit__()
                del self.pending_waitcounters[job_id]
                if self.firstjob_id == job_id:
                    self.firstjob_waitcounter.__exit__()

                del self.pending_futures[job_id]

    def quiesce(self) -> None:
        self._send(MsgHeader.QUIESCE)
        if self.quiesce_waitcounter is None:
            self.quiesce_waitcounter = _WaitCounter(
                "pytorch.wait_counter.subproc_pool.quiesced"
            ).guard()
            self.quiesce_waitcounter.__enter__()

    def wakeup(self) -> None:
        self._send(MsgHeader.WAKEUP)

    def shutdown(self) -> None:
        try:
            with self.write_lock:
                if not self.running:
                    return
                if self.timer:
                    self.timer.quit()
                self.running = False
                self.running_waitcounter.__exit__()
                _send_msg(self.write_pipe, MsgHeader.SHUTDOWN)
                self.write_pipe.close()
            self.process.wait(300)
            if self.log_file:
                self.log_file.close()
        except OSError:
            log.warning("Ignored OSError in pool shutdown", exc_info=True)
        finally:
            with self.futures_lock:
                for future in self.pending_futures.values():
                    if not future.cancel():
                        future.set_exception(RuntimeError("SubprocPool closed"))
                self.pending_futures.clear()


AnyPool = ProcessPoolExecutor | SubprocPool


class TestException(RuntimeError):
    pass


def raise_testexc() -> Never:
    raise TestException
