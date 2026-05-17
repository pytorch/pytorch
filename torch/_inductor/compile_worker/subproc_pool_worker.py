"""
Worker-side implementation for SubprocPool.

This module is imported by compile_worker.__main__ before it creates the inner
ProcessPoolExecutor, so top-level imports here must not import torch.
"""

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
from enum import Enum, IntEnum
from typing import Any, IO


log = logging.getLogger(__name__)


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
        self.pool: ProcessPoolExecutor | None = None
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
            self.pool.shutdown(wait=False)
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

        self.pool = ProcessPoolExecutor(
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
