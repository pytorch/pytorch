import functools
import itertools
import logging
import multiprocessing
import os
import pickle
import struct
import subprocess
import sys
import threading
import traceback
import typing
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from enum import Enum
from typing import Any, BinaryIO, Callable, Dict, Optional, Tuple, TypeVar
from typing_extensions import Never, ParamSpec

# _thread_safe_fork is needed because the subprocesses in the pool can read
# justknobs, e.g., in the Triton compiler. For internal, the import installs
# functionality to destroy singletons before forking and re-enable them after.
import torch._thread_safe_fork  # noqa: F401
from torch._inductor import config
from torch._inductor.compile_worker.watchdog import _async_compile_initializer


log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")


def _pack_msg(job_id: int, length: int) -> bytes:
    return struct.pack("nn", job_id, length)


def _unpack_msg(data: bytes) -> Tuple[int, int]:
    if not data:
        return -1, -1
    return struct.unpack("nn", data)


msg_bytes = len(_pack_msg(0, 0))


def _send_msg(write_pipe: BinaryIO, job_id: int, job_data: bytes = b"") -> None:
    length = len(job_data)
    write_pipe.write(_pack_msg(job_id, length))
    if length > 0:
        write_pipe.write(job_data)
    write_pipe.flush()


def _recv_msg(read_pipe: BinaryIO) -> Tuple[int, bytes]:
    job_id, length = _unpack_msg(read_pipe.read(msg_bytes))
    data = read_pipe.read(length) if length > 0 else b""
    return job_id, data


def _get_ld_library_path() -> str:
    path = os.environ.get("LD_LIBRARY_PATH", "")
    if config.is_fbcode():
        from libfb.py.parutil import get_runtime_path

        runtime_path = get_runtime_path()
        if runtime_path:
            lib_path = os.path.join(runtime_path, "runtime", "lib")
            path = os.pathsep.join([lib_path, path]) if path else lib_path

    return path


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

    def __init__(self, details: str) -> None:
        super().__init__(f"An exception occurred in a subprocess:\n\n{details}")


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
        pickler: Optional[SubprocPickler] = None,
        kind: SubprocKind = SubprocKind.FORK,
    ) -> None:
        entry = os.path.join(os.path.dirname(__file__), "__main__.py")
        self.pickler = pickler or SubprocPickler()
        self.kind = kind

        subproc_read_fd, write_fd = os.pipe()
        read_fd, subproc_write_fd = os.pipe()
        self.write_pipe = os.fdopen(write_fd, "wb")
        self.read_pipe = os.fdopen(read_fd, "rb")

        cmd = [
            sys.executable,
            entry,
            f"--pickler={self.pickler.__class__.__module__}.{self.pickler.__class__.__name__}",
            f"--kind={self.kind.value}",
            f"--workers={nprocs}",
            f"--parent={os.getpid()}",
            f"--read-fd={str(subproc_read_fd)}",
            f"--write-fd={str(subproc_write_fd)}",
        ]
        self.process = subprocess.Popen(
            cmd,
            env={
                **os.environ,
                # We need to set the PYTHONPATH so the subprocess can find torch.
                "PYTHONPATH": os.pathsep.join(sys.path),
                # We don't want to re-warm the pool when the subprocess imports
                # torch._inductor.codecache since the warming process is what
                # creates the SubprocPool in the first place.
                "TORCH_WARM_POOL": "0",
                # Some internal usages need a modified LD_LIBRARY_PATH.
                "LD_LIBRARY_PATH": _get_ld_library_path(),
            },
            pass_fds=(subproc_read_fd, subproc_write_fd),
        )
        self.write_lock = threading.Lock()
        self.read_thread = threading.Thread(target=self._read_thread, daemon=True)

        self.futures_lock = threading.Lock()
        self.pending_futures: Dict[int, Future[Any]] = {}
        self.job_id_count = itertools.count()

        self.running = True

        # Start thread last to ensure all member variables are initialized
        # before any access.
        self.read_thread.start()

    def submit(
        self, job_fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
    ) -> Future[_T]:
        if args or kwargs:
            job_fn = functools.partial(job_fn, *args, **kwargs)
        job_data = self.pickler.dumps(job_fn)
        future: Future[_T]
        with self.futures_lock:
            job_id = next(self.job_id_count)
            self.pending_futures[job_id] = future = Future()
        future.set_running_or_notify_cancel()
        with self.write_lock:
            if not self.running:
                raise RuntimeError("submit() on closed pool")
            _send_msg(self.write_pipe, job_id, job_data)
        return future

    def _read_thread(self) -> None:
        while True:
            data = b""
            try:
                job_id, data = _recv_msg(self.read_pipe)
            except Exception as e:
                # Something went wrong during the read. There's no way we have a
                # valid job_id.
                log.exception("failure in subproc_pool._recv_msg")
                job_id = -1

            if job_id < 0:
                # read_pipe returned None or got exception
                if self.running:
                    log.warning("SubprocPool unclean exit")
                    self.running = False
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
                del self.pending_futures[job_id]

    def shutdown(self) -> None:
        try:
            with self.write_lock:
                if not self.running:
                    return
                self.running = False
                _send_msg(self.write_pipe, -1)
                self.write_pipe.close()
            self.process.wait(300)
        except OSError as e:
            log.warning("Ignored OSError in pool shutdown:  %s", e)
        finally:
            with self.futures_lock:
                for future in self.pending_futures.values():
                    if not future.cancel():
                        future.set_exception(RuntimeError("SubprocPool closed"))
                self.pending_futures.clear()


class SubprocMain:
    """Communicates with a SubprocPool in the parent process, called by __main__.py"""

    def __init__(
        self,
        pickler: SubprocPickler,
        kind: SubprocKind,
        nprocs: int,
        read_pipe: BinaryIO,
        write_pipe: BinaryIO,
    ) -> None:
        self.pickler = pickler
        self.kind = kind
        self.read_pipe = read_pipe
        self.write_pipe = write_pipe
        self.write_lock = threading.Lock()
        self.nprocs = nprocs
        self.pool = self._new_pool(nprocs, True)
        self.running = True

    def _new_pool(self, nprocs: int, warm: bool) -> ProcessPoolExecutor:
        pool = ProcessPoolExecutor(
            nprocs,
            mp_context=multiprocessing.get_context(self.kind.value),
            initializer=functools.partial(_async_compile_initializer, os.getpid()),
        )
        multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)
        if warm:
            _warm_process_pool(pool, nprocs)
        return pool

    def main(self) -> None:
        while True:
            job_id, data = _recv_msg(self.read_pipe)
            if job_id < 0:
                return self._shutdown()
            self.submit(job_id, data)

    def _shutdown(self) -> None:
        with self.write_lock:
            self.running = False
            try:
                _send_msg(self.write_pipe, -1)
                self.write_pipe.close()
            except BrokenPipeError:
                pass  # parent process already shutdown
            self.read_pipe.close()
        self.pool.shutdown()

    def submit(self, job_id: int, data: bytes) -> None:
        while self.running:
            try:
                self._submit_inner(job_id, data)
                return
            except BrokenProcessPool:
                # If any subprocess in the pool crashes, we get a BrokenProcessPool
                # exception and the whole pool becomes unusable. Handle crashes by
                # recreating the pool and resubmitting.
                self.pool = self._new_pool(self.nprocs, False)

    def _submit_inner(self, job_id: int, data: bytes) -> None:
        future = self.pool.submit(
            functools.partial(SubprocMain.do_job, self.pickler, data)
        )

        def callback(_: Future[Any]) -> None:
            if not self.running:
                return
            try:
                result = future.result()
            except Exception as e:
                log.exception("Error in subprocess")
                result = self.pickler.dumps(e)
            assert isinstance(result, bytes)
            with self.write_lock:
                if self.running:
                    _send_msg(self.write_pipe, job_id, result)
            return

        future.add_done_callback(callback)

    @staticmethod
    def do_job(pickler: SubprocPickler, data: bytes) -> bytes:
        # do the pickle/unpickle in the sub-subproc
        job = typing.cast(Callable[[], object], pickler.loads(data))

        try:
            result = job()
        except Exception:
            result = _SubprocExceptionInfo(traceback.format_exc())
        return pickler.dumps(result)


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
