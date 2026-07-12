import base64
import functools
import itertools
import json
import logging
import multiprocessing
import os
import pickle
import struct
import subprocess
import sys
import threading
import time
import traceback
import typing
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, IO, TypeVar
from typing_extensions import Never, ParamSpec

# _thread_safe_fork is needed because the subprocesses in the pool can read
# justknobs, e.g., in the Triton compiler. For internal, the import installs
# functionality to destroy singletons before forking and re-enable them after.
import torch._thread_safe_fork  # noqa: F401
from torch._inductor import config
from torch._inductor.codecache import torch_key
from torch._inductor.compile_worker import watchdog
from torch._inductor.compile_worker.timer import Timer
from torch._inductor.compile_worker.tracked_process_pool import (
    TrackedProcessPoolExecutor,
)
from torch._inductor.compile_worker.utils import _async_compile_initializer
from torch._inductor.utils import get_ld_library_path, python_subprocess_env
from torch._logging import trace_structured
from torch._utils_internal import find_compile_subproc_binary
from torch.monitor import _WaitCounter, _WaitCounterTracker


log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")


class MsgHeader(IntEnum):
    ERROR = 0
    SHUTDOWN = 1
    QUIESCE = 2
    WAKEUP = 3
    JOB = 4
    # Sidecar -> parent watchdog report about a still-running job (out-of-band,
    # low volume). Payload is a pickled status dict; does not affect the future.
    STATUS = 5


def _current_compile_id() -> Any:
    # Snapshotted at submit so a later watchdog STATUS report (handled on the
    # read thread, which has no ambient compile context) can be attributed to the
    # right compile in tlparse.
    try:
        from torch._guards import CompileContext

        return CompileContext.current_compile_id()
    except Exception:
        return None


def _pack_msg(msg_header: MsgHeader, job_id: int, length: int) -> bytes:
    return struct.pack("nnn", int(msg_header), job_id, length)


def _unpack_msg(data: bytes) -> tuple[MsgHeader, int, int]:
    if not data:
        return MsgHeader.ERROR, -1, -1
    msg_header, job_id, length = struct.unpack("nnn", data)
    return MsgHeader(msg_header), job_id, length


msg_bytes = len(_pack_msg(MsgHeader.JOB, 0, 0))

# How often the parent polls the sidecar process for liveness. The sidecar
# dying is the only signal this catches, so a coarse interval is fine.
_SIDECAR_HEALTH_POLL_SECONDS = 2.0

# How long, in total, to let compile workers exit on SIGTERM during pool
# teardown before escalating to SIGKILL. A worker wedged holding a lock or in a
# C/GIL section won't honor SIGTERM, and an unbounded wait for it would stall the
# whole shutdown.
_WORKER_TERMINATE_GRACE_SECONDS = 10.0


def _send_msg(
    write_pipe: IO[bytes], msg_header: MsgHeader, job_id: int = -1, data: bytes = b""
) -> None:
    length = len(data)
    write_pipe.write(_pack_msg(msg_header, job_id, length))
    if length > 0:
        write_pipe.write(data)
    write_pipe.flush()


def _recv_msg(read_pipe: IO[bytes]) -> tuple[MsgHeader, int, bytes]:
    msg_header, job_id, length = _unpack_msg(read_pipe.read(msg_bytes))
    data = read_pipe.read(length) if length > 0 else b""
    return msg_header, job_id, data


class _SubprocExceptionInfo:
    """
    Carries exception info from subprocesses across the wire. traceback
    objects are not pickleable, so we store the trace as a string and
    use it for the message in the exception thrown in the main process.
    """

    def __init__(self, details: str) -> None:
        self.details = details


class SubprocException(Exception):
    """
    Thrown when a job in a subprocess raises an Exception.
    """

    def __init__(self, details: str, name: str = "<unknown>") -> None:
        self.details = details
        super().__init__(
            f"An exception occurred in a subprocess:\n\nName={name}\n{details}"
        )

    def with_name(self, name: str) -> "SubprocException":
        return SubprocException(self.details, name)


class SubprocPickler:
    """
    Allows a caller to provide a custom pickler for passing data with the
    subprocess.
    """

    def dumps(self, obj: object) -> bytes:
        return pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)

    def loads(self, data: bytes) -> object:
        return pickle.loads(data)


class SubprocKind(Enum):
    FORK = "fork"
    SPAWN = "spawn"


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
        # Kept so the liveness watchdog can tail worker output if the sidecar dies.
        self.log_path = log_path

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
        # The sidecar inherited these via pass_fds; the parent uses its own ends
        # (write_pipe/read_pipe) and must drop its copies. Otherwise the parent
        # itself keeps the result pipe's write end open, so read_pipe would never
        # EOF even after the sidecar and all workers are gone.
        os.close(subproc_read_fd)
        os.close(subproc_write_fd)

        self.write_lock = threading.Lock()
        self.read_thread = threading.Thread(
            target=self._read_thread, name="InductorSubproc", daemon=True
        )
        # Backstop for the sidecar dying. Closing the inherited pipe fds (in the
        # parent above and in each worker's initializer) means a dead sidecar
        # normally yields a clean EOF that _read_thread handles. This watchdog
        # covers the cases where EOF does not arrive -- e.g. a stray fd copy still
        # holding the write end open -- by detecting the dead process directly.
        self.health_thread = threading.Thread(
            target=self._health_monitor, name="InductorSubprocHealth", daemon=True
        )

        self.futures_lock = threading.Lock()
        self.pending_futures: dict[int, Future[Any]] = {}
        # Compile id captured at submit so watchdog STATUS reports (handled on the
        # read thread) attribute to the right compile in tlparse. Keyed by job_id.
        self._job_compile_id: dict[int, Any] = {}
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

        # Start threads last to ensure all member variables are initialized
        # before any access.
        self.read_thread.start()
        self.health_thread.start()

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
            self._job_compile_id[job_id] = _current_compile_id()
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

            if msg_header == MsgHeader.STATUS:
                # Out-of-band watchdog report; does not touch futures.
                self._handle_worker_status(job_id, data)
                continue

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
                self._job_compile_id.pop(job_id, None)

    def _health_monitor(self) -> None:
        # Poll the sidecar for liveness. If it dies while we still think we are
        # running, fail the pending compiles rather than let waiters block
        # forever (see health_thread comment in __init__).
        while self.running:
            returncode = self.process.poll()
            if returncode is not None:
                self._on_sidecar_death(returncode)
                return
            time.sleep(_SIDECAR_HEALTH_POLL_SECONDS)

    def _on_sidecar_death(self, returncode: int) -> None:
        with self.futures_lock:
            if not self.running:
                # Expected exit (shutdown) or already handled.
                return
            self.running = False
        # `running` is set under different locks across shutdown()/_read_thread/
        # here, so this exit can race another for the same transition. That's
        # safe: _WaitCounterTracker.__exit__ is idempotent (optional::reset()).
        self.running_waitcounter.__exit__()

        pid = self.process.pid
        log.error(
            "Inductor compile worker sidecar (pid %s) exited unexpectedly with "
            "code %s during compilation; failing pending compile jobs. Re-run "
            "with TORCHINDUCTOR_COMPILE_THREADS=1 to compile in the main process.",
            pid,
            returncode,
        )
        self._log_sidecar_death_diagnostics(returncode)

        exc = RuntimeError(
            f"Inductor compile worker sidecar (pid {pid}) exited unexpectedly "
            f"with code {returncode} during compilation. Re-run with "
            "TORCHINDUCTOR_COMPILE_THREADS=1 to compile in the main process."
        )
        with self.futures_lock:
            for job_id, future in self.pending_futures.items():
                if not future.cancel():
                    future.set_exception(exc)
                waitcounter = self.pending_waitcounters.pop(job_id, None)
                if waitcounter is not None:
                    waitcounter.__exit__()
            self.pending_futures.clear()
            self._job_compile_id.clear()
        # We intentionally do not close read_pipe here. _read_thread owns it and
        # may be mid-read; closing a buffered pipe out from under a concurrent
        # read can deadlock on the buffer lock. It is a daemon thread, so leaving
        # it is harmless now that the futures are resolved -- and if the sidecar's
        # death already produced an EOF, _read_thread will close the pipe itself.

    def _log_sidecar_death_diagnostics(self, returncode: int) -> None:
        tail = ""
        if self.log_path and self.log_path != os.devnull:
            try:
                with open(self.log_path, errors="replace") as f:
                    tail = "".join(f.readlines()[-50:])
            except OSError:
                pass
        if tail:
            log.error(
                "Last output from compile worker sidecar (pid %s):\n%s",
                self.process.pid,
                tail,
            )
        # Surface the same info to structured tracing so production jobs (which
        # only have tlparse access) can diagnose the crash.
        try:
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "compile_worker_sidecar_death",
                    "encoding": "string",
                },
                payload_fn=lambda: (
                    f"sidecar pid={self.process.pid} returncode={returncode}\n\n{tail}"
                ),
            )
        except Exception:
            log.warning("Failed to emit sidecar-death trace artifact", exc_info=True)

    def _handle_worker_status(self, job_id: int, data: bytes) -> None:
        # Best-effort, and runs on the read thread: a bad status report (corrupt
        # payload or a logging failure) must not escape and kill result
        # processing, so guard the whole thing.
        try:
            status = self.pickler.loads(data)
            if not isinstance(status, dict):
                return
            compile_id = self._job_compile_id.get(job_id)
            record = {**status, "compile_id": str(compile_id) if compile_id else None}
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "compile_worker_status",
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps(record),
                compile_id=compile_id,
                expect_trace_id=False,
                suppress_context=True,
                record_logging_overhead=False,
            )
        except Exception:
            log.warning("failed to report compile worker status", exc_info=True)

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
            try:
                self.process.wait(300)
            except subprocess.TimeoutExpired:
                # The sidecar did not exit in time (e.g. wedged tearing down its
                # own worker pool). Don't stall the whole process; kill it and
                # move on. TimeoutExpired is not an OSError, so it would
                # otherwise propagate uncaught out of shutdown().
                log.warning(
                    "Compile worker sidecar (pid %s) did not exit within 300s; "
                    "killing it.",
                    self.process.pid,
                )
                self.process.kill()
                self.process.wait()
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
                self._job_compile_id.clear()


class SubprocMain:
    """Communicates with a SubprocPool in the parent process, called by __main__.py"""

    def __init__(
        self,
        pickler: SubprocPickler,
        kind: SubprocKind,
        nprocs: int,
        read_pipe: IO[bytes],
        write_pipe: IO[bytes],
    ) -> None:
        self.pickler = pickler
        self.kind = kind
        self.read_pipe = read_pipe
        self.write_pipe = write_pipe
        self.write_lock = threading.Lock()
        self.nprocs = nprocs
        self.pool: ProcessPoolExecutor | None = None
        self.pool_finalizer: Any | None = None
        self.running = True
        # job_id -> monotonic submit time; scanned by the watchdog thread.
        self._inflight: dict[int, float] = {}
        self._inflight_lock = threading.Lock()
        self._watchdog_stop = threading.Event()

    def main(self) -> None:
        # Phase heartbeats rely on fork inheritance of the shared buffer; spawn
        # workers get no buffer and degrade to duration-only reporting.
        if self.kind == SubprocKind.FORK:
            watchdog.create(self.nprocs)
        self._start_watchdog()
        while True:
            msg_header, job_id, data = _recv_msg(self.read_pipe)
            if msg_header == MsgHeader.JOB:
                self.submit(job_id, data)
            elif msg_header == MsgHeader.WAKEUP:
                self._start_pool()
            elif msg_header == MsgHeader.QUIESCE:
                self._quiesce()
            else:
                return self._shutdown()

    def _quiesce(self) -> None:
        self._shutdown_pool(terminate_workers=False)

    def _shutdown(self) -> None:
        self._watchdog_stop.set()
        with self.write_lock:
            self.running = False
            try:
                _send_msg(self.write_pipe, MsgHeader.SHUTDOWN)
                self.write_pipe.close()
            except BrokenPipeError:
                pass  # parent process already shutdown
            self.read_pipe.close()
        self._shutdown_pool(terminate_workers=True)

    def _shutdown_pool(self, *, terminate_workers: bool) -> None:
        if self.pool is None:
            return

        pool = self.pool
        self.pool = None

        if self.pool_finalizer is not None:
            if terminate_workers:
                self.pool_finalizer.cancel()
            self.pool_finalizer = None

        if terminate_workers:
            # The sidecar is exiting, so do not let ProcessPoolExecutor's
            # interpreter finalization wait for running compiler workers.
            _terminate_process_pool(pool)
        else:
            pool.shutdown(wait=False)

    def submit(self, job_id: int, data: bytes) -> None:
        # Clock starts before _start_pool/_warm_process_pool, so the first job's
        # reported elapsed intentionally includes cold pool creation and the fork
        # of the workers -- that wait is real and worth surfacing.
        with self._inflight_lock:
            self._inflight[job_id] = time.monotonic()
        while self.running:
            try:
                self._submit_inner(job_id, data)
                return
            except BrokenProcessPool:
                # If any subprocess in the pool crashes, we get a BrokenProcessPool
                # exception and the whole pool becomes unusable. Handle crashes by
                # recreating the pool and resubmitting. Log it -- this was
                # previously silent, hiding repeated worker crashes that can stall
                # a job in this retry loop.
                log.warning(
                    "Compile worker pool broke while submitting job %s; "
                    "recreating the pool and retrying.",
                    job_id,
                    exc_info=True,
                )
                self.pool = None

    def _submit_inner(self, job_id: int, data: bytes) -> None:
        def callback(fut: Future[Any]) -> None:
            with self._inflight_lock:
                self._inflight.pop(job_id, None)
            if not self.running:
                return
            try:
                result = fut.result()
            except Exception as e:
                log.exception("Error in subprocess")
                result = self.pickler.dumps(e)
            if not isinstance(result, bytes):
                raise AssertionError(f"Expected bytes result, got {type(result)}")
            with self.write_lock:
                if self.running:
                    _send_msg(self.write_pipe, MsgHeader.JOB, job_id, result)
            return

        self._start_pool()
        if self.pool is None:
            raise AssertionError("pool must be initialized before submitting jobs")

        future = self.pool.submit(
            functools.partial(SubprocMain.do_job, self.pickler, job_id, data)
        )
        future.add_done_callback(callback)

    def _start_pool(self) -> None:
        if self.pool is not None:
            return

        # Recycle heartbeat slots before the new worker generation forks.
        watchdog.reset()

        # Only fork workers inherit the sidecar<->parent pipe fds and must close
        # them (see _async_compile_initializer). Under spawn the workers do not
        # inherit them (close_fds=True + O_CLOEXEC), and these integers would
        # refer to unrelated fds the fresh interpreter reused -- closing them
        # would be silent corruption, not a no-op.
        close_fds = (
            (self.read_pipe.fileno(), self.write_pipe.fileno())
            if self.kind == SubprocKind.FORK
            else ()
        )
        self.pool = TrackedProcessPoolExecutor(
            self.nprocs,
            mp_context=multiprocessing.get_context(self.kind.value),
            initializer=functools.partial(
                _async_compile_initializer, os.getpid(), close_fds
            ),
        )
        self.pool_finalizer = multiprocessing.util.Finalize(
            None, self.pool.shutdown, exitpriority=sys.maxsize
        )
        _warm_process_pool(self.pool, self.nprocs)

    def _start_watchdog(self) -> None:
        interval = config.compile_worker_watchdog_interval_seconds
        if interval <= 0:
            return
        threading.Thread(
            target=self._watchdog_loop,
            args=(interval,),
            name="InductorSubprocWatchdog",
            daemon=True,
        ).start()

    def _watchdog_loop(self, interval: float) -> None:
        # Every `interval` seconds, report any job still running past `interval`
        # so a stuck/slow worker leaves a breadcrumb (the parent turns these into
        # tlparse artifacts). Re-reports each tick with a growing elapsed time.
        while not self._watchdog_stop.wait(interval):
            now = time.monotonic()
            now_ns = time.monotonic_ns()
            heartbeats = watchdog.read_heartbeats()
            with self._inflight_lock:
                slow = [
                    (job_id, elapsed)
                    for job_id, start in self._inflight.items()
                    if (elapsed := now - start) >= interval
                ]
            for job_id, elapsed in slow:
                self._report_status(job_id, elapsed, heartbeats.get(job_id), now_ns)

    def _report_status(
        self,
        job_id: int,
        elapsed: float,
        heartbeat: tuple[int, int, int] | None,
        now_ns: int,
    ) -> None:
        status: dict[str, Any] = {"job_id": job_id, "elapsed_s": round(elapsed, 1)}
        if heartbeat is not None:
            phase, phase_start_ns, worker_pid = heartbeat
            status["phase"] = watchdog.Phase(phase).name.lower()
            status["phase_elapsed_s"] = round((now_ns - phase_start_ns) / 1e9, 1)
            status["worker_pid"] = worker_pid
        elif watchdog.enabled():
            # Heartbeats are active but this job isn't in any worker's slot. Almost
            # always that means it's still queued in the pool -- a running (fork)
            # worker stamps its slot before doing anything. It can also briefly
            # mean the opposite: a job that just finished (clear_current_job ran in
            # do_job's finally) but hasn't yet been popped from _inflight by the
            # sidecar callback. That window is negligibly short, so we don't
            # distinguish it.
            status["phase"] = "queued"
        # Otherwise phase tracking is unavailable (e.g. a spawn pool) and the
        # report carries duration only -- we can't tell queued from running.
        payload = self.pickler.dumps(status)
        try:
            with self.write_lock:
                if self.running:
                    _send_msg(self.write_pipe, MsgHeader.STATUS, job_id, payload)
        except (OSError, ValueError):
            # Parent gone / pipe closed; the watchdog is best-effort.
            pass

    @staticmethod
    def do_job(pickler: SubprocPickler, job_id: int, data: bytes) -> bytes:
        # do the pickle/unpickle in the sub-subproc
        watchdog.set_current_job(job_id)
        try:
            job = typing.cast(Callable[[], object], pickler.loads(data))
            try:
                result = job()
            except Exception:
                result = _SubprocExceptionInfo(traceback.format_exc())
            return pickler.dumps(result)
        finally:
            watchdog.clear_current_job()


AnyPool = ProcessPoolExecutor | SubprocPool


def _get_process_pool_processes(pool: ProcessPoolExecutor) -> list[Any]:
    processes = getattr(pool, "_processes", None)
    if processes is not None:
        return list(processes.values())

    manager_thread = getattr(pool, "_executor_manager_thread", None)
    manager_processes = getattr(manager_thread, "processes", None)
    if manager_processes is not None:
        return list(manager_processes.values())

    return []


def _terminate_process_pool(pool: ProcessPoolExecutor) -> None:
    processes = _get_process_pool_processes(pool)
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except (OSError, ValueError):
            log.warning("Ignored error terminating compile worker", exc_info=True)

    # Give workers a bounded, shared grace period to exit on the SIGTERM above,
    # then SIGKILL any that are wedged. Without this the pool.shutdown(wait=True)
    # below can block on an unreaped worker until the parent's shutdown wait
    # gives up (a ~300s stall).
    deadline = time.time() + _WORKER_TERMINATE_GRACE_SECONDS
    for process in processes:
        try:
            remaining = deadline - time.time()
            if remaining > 0:
                process.join(remaining)
            if process.is_alive():
                process.kill()
        except (OSError, ValueError, AssertionError):
            log.warning("Ignored error killing compile worker", exc_info=True)

    try:
        pool.shutdown(wait=True, cancel_futures=True)
    except Exception:
        log.warning("Ignored error shutting down compile worker pool", exc_info=True)


def _warm_process_pool(pool: ProcessPoolExecutor, n: int) -> None:
    # We have to fork processes for compiler workers, but the more memory and other resources that are loaded, the
    # slower the os.fork time is, quite drastically. It also holds the GIL so we can't put it on another thread.

    # Examples:
    # A simple x + x + x script: 10ms seconds in the middle of the program, 2ms at startup
    # tf_efficientnet_b0 benchmark: 50ms! in the middle of the program , 3ms at startup

    # So we want to start the workers early when it is still cheap, and also to allow the workers to get
    # ready before we have work for them.

    # ProcessPoolExecutor also does not launch the workers until it finds a point when all the workers are idle.
    # But if we waited until then fork time will be long and we will be waiting for the processes to initialize.

    # We force them to start here with some YOLOing of the internal methods.

    if hasattr(pool, "_start_queue_management_thread"):
        pool._start_queue_management_thread()
    else:
        for _ in range(n):
            pool._adjust_process_count()
        if hasattr(pool, "_start_executor_manager_thread"):
            pool._start_executor_manager_thread()


class TestException(RuntimeError):
    pass


def raise_testexc() -> Never:
    raise TestException


def _test_signal_then_sleep(signal_path: str, seconds: float) -> None:
    # Test helper: announce that this worker has begun executing by creating
    # signal_path, then block. Lets a test wait until a job is actually running
    # in a worker before acting on the pool (e.g. shutting it down).
    Path(signal_path).touch()
    time.sleep(seconds)


def _ignore_sigterm_and_sleep_for_test() -> None:
    # Test helper: a compile worker that ignores SIGTERM and blocks, forcing pool
    # teardown to escalate to SIGKILL instead of stalling on a graceful shutdown.
    import signal

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(3600)


def _report_phase_and_sleep_for_test(phase: int, seconds: float) -> None:
    # Test helper: report a heartbeat phase from a worker, then block, so the
    # sidecar watchdog observes and reports that phase.
    watchdog.report_phase(watchdog.Phase(phase))
    time.sleep(seconds)
