"""
Worker-side implementation for SubprocPool.

This module is imported by compile_worker.__main__ before it creates the inner
ProcessPoolExecutor, so top-level imports here must not import torch.
"""

import atexit
import concurrent
import dataclasses
import functools
import logging
import multiprocessing
import multiprocessing.util
import os
import pickle
import struct
import sys
import threading
import traceback
import typing
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from enum import Enum, IntEnum
from multiprocessing.context import BaseContext
from time import time
from typing import Any, IO


log = logging.getLogger(__name__)
_queue_stats_log = logging.getLogger(
    "torch._inductor.compile_worker.tracked_process_pool"
)


class MsgHeader(IntEnum):
    ERROR = 0
    SHUTDOWN = 1
    QUIESCE = 2
    WAKEUP = 3
    JOB = 4


def _pack_msg(msg_header: MsgHeader, job_id: int, length: int) -> bytes:
    return struct.pack("nnn", int(msg_header), job_id, length)


def _unpack_msg(data: bytes) -> tuple[MsgHeader, int, int]:
    if not data:
        return MsgHeader.ERROR, -1, -1
    msg_header, job_id, length = struct.unpack("nnn", data)
    return MsgHeader(msg_header), job_id, length


msg_bytes = len(_pack_msg(MsgHeader.JOB, 0, 0))


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


@dataclass
class _QueueStats:
    # Mapping from id(future) -> start time
    pending: dict[int, float] = dataclasses.field(default_factory=dict)
    timing: list[float] = dataclasses.field(default_factory=list)
    enqueue_count: int = 0
    dequeue_count: int = 0
    max_queue_depth: int = 0
    pool_count: int = 0


_queue_stats = _QueueStats()
_queue_stats_lock = threading.Lock()


class TrackedProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(
        self,
        max_workers: int | None = None,
        mp_context: BaseContext | None = None,
        initializer: Callable[[], object] | None = None,
    ) -> None:
        with _queue_stats_lock:
            _queue_stats.pool_count += 1
        super().__init__(max_workers, mp_context, initializer)

    def _record_dequeue(self, f: Future[Any]) -> None:
        now = time()
        with _queue_stats_lock:
            stats = _queue_stats
            if (start_time := stats.pending.pop(id(f), None)) is None:
                return
            stats.dequeue_count += 1
            duration = now - start_time
            stats.timing.append(duration)

    def _record_enqueue(self, f: Future[Any]) -> None:
        # Monkeypatch set_running_or_notify_cancel so we can track when the
        # Future moves out of PENDING.
        saved_running_or_notify_cancel = f.set_running_or_notify_cancel

        def set_running_or_notify_cancel() -> Any:
            self._record_dequeue(f)
            return saved_running_or_notify_cancel()

        now = time()
        with _queue_stats_lock:
            stats = _queue_stats
            stats.pending[id(f)] = now
            stats.enqueue_count += 1
            stats.max_queue_depth = max(stats.max_queue_depth, len(stats.pending))
            f.set_running_or_notify_cancel = set_running_or_notify_cancel  # type: ignore[method-assign]

        if f._state != concurrent.futures._base.PENDING:
            self._record_dequeue(f)

    def submit(
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        f = super().submit(fn, *args, **kwargs)
        self._record_enqueue(f)
        return f


@atexit.register
def _queue_stats_report() -> None:
    stats = _queue_stats
    if stats.pool_count == 0:
        return

    timing = stats.timing
    timing.sort()

    _queue_stats_log.info("AsyncCompile Metrics:")
    _queue_stats_log.info("  Pools %s", stats.pool_count)
    _queue_stats_log.info(
        "  Items %d enqueued / %d dequeued", stats.enqueue_count, stats.dequeue_count
    )
    _queue_stats_log.info("  Max Queue Depth: %d", stats.max_queue_depth)
    n = len(timing)
    if n > 0:
        _queue_stats_log.info("  Longest queue time: %0.2fs", timing[-1])
        _queue_stats_log.info("  P50: %0.2fs", timing[n // 2])
        if n >= 20:
            _queue_stats_log.info("  P95: %0.2fs", timing[n * 95 // 100])


class SubprocMain:
    """Communicates with a SubprocPool in the parent process, called by __main__.py"""

    def __init__(
        self,
        pickler: SubprocPickler,
        kind: SubprocKind,
        nprocs: int,
        read_pipe: IO[bytes],
        write_pipe: IO[bytes],
        torch_key_data: bytes,
    ) -> None:
        self.pickler = pickler
        self.kind = kind
        self.read_pipe = read_pipe
        self.write_pipe = write_pipe
        self.write_lock = threading.Lock()
        self.nprocs = nprocs
        self.torch_key_data = torch_key_data
        self.pool: TrackedProcessPoolExecutor | None = None
        self.running = True

    def main(self) -> None:
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
        if self.pool is not None:
            # A later wakeup may create a new fork-based pool. Wait for the old
            # executor manager thread to exit first so the sidecar is single-threaded
            # before it forks again.
            self.pool.shutdown(wait=True)
            self.pool = None

    def _shutdown(self) -> None:
        with self.write_lock:
            self.running = False
            try:
                _send_msg(self.write_pipe, MsgHeader.SHUTDOWN)
            except BrokenPipeError:
                pass  # parent process already shutdown
            finally:
                self.write_pipe.close()
            self.read_pipe.close()
        self._quiesce()

    def submit(self, job_id: int, data: bytes) -> None:
        while self.running:
            try:
                self._submit_inner(job_id, data)
                return
            except BrokenProcessPool:
                # If any subprocess in the pool crashes, we get a BrokenProcessPool
                # exception and the whole pool becomes unusable. Handle crashes by
                # recreating the pool and resubmitting.
                self.pool = None

    def _submit_inner(self, job_id: int, data: bytes) -> None:
        def callback(fut: Future[Any]) -> None:
            if not self.running:
                return
            try:
                result = fut.result()
            except Exception as e:
                log.exception("Error in subprocess")
                result = self.pickler.dumps(e)
            assert isinstance(result, bytes)
            with self.write_lock:
                if self.running:
                    _send_msg(self.write_pipe, MsgHeader.JOB, job_id, result)
            return

        self._start_pool()
        assert self.pool is not None

        future = self.pool.submit(
            functools.partial(SubprocMain.do_job, self.pickler, data)
        )
        future.add_done_callback(callback)

    def _start_pool(self) -> None:
        if self.pool is not None:
            return

        # Do not import tracked_process_pool here: it imports torch._thread_safe_fork
        # for parents that have already imported torch. The sidecar deliberately
        # has not, so that import would recreate the fork-after-thread-start issue.
        self.pool = TrackedProcessPoolExecutor(
            self.nprocs,
            mp_context=multiprocessing.get_context(self.kind.value),
            initializer=functools.partial(
                _worker_initializer, os.getpid(), self.torch_key_data
            ),
        )
        multiprocessing.util.Finalize(
            None, self.pool.shutdown, exitpriority=sys.maxsize
        )
        _warm_process_pool(self.pool, self.nprocs)

    @staticmethod
    def do_job(pickler: SubprocPickler, data: bytes) -> bytes:
        # do the pickle/unpickle in the sub-subproc
        job = typing.cast(Callable[[], object], pickler.loads(data))

        try:
            result = job()
        except Exception:
            result = _SubprocExceptionInfo(traceback.format_exc())
        return pickler.dumps(result)


def _worker_initializer(orig_ppid: int, torch_key_data: bytes) -> None:
    # Import torch-dependent setup only inside the already-created compile
    # worker. The sidecar process stays torch-free before the worker fork.
    from torch._inductor.async_compile import pre_fork_setup
    from torch._inductor.codecache import torch_key
    from torch._inductor.compile_worker.utils import _async_compile_initializer
    from torch._inductor.runtime.compile_tasks import _set_triton_ptxas_path

    _set_triton_ptxas_path()
    torch_key.set(torch_key_data)  # type: ignore[attr-defined]
    pre_fork_setup()
    _async_compile_initializer(orig_ppid)


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
