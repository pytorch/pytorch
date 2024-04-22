import functools
import itertools
import logging
import multiprocessing
import os
import pickle
import re
import struct
import subprocess
import sys
import threading
import typing
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict

from torch._inductor.compile_worker.watchdog import _async_compile_initializer

log = logging.getLogger(__name__)


def _pack_msg(job_id, length):
    return struct.pack("nn", job_id, length)


def _unpack_msg(data):
    if not data:
        return -1, -1
    return struct.unpack("nn", data)


msg_bytes = len(_pack_msg(0, 0))


class Pipe(typing.Protocol):
    def write(self, data: bytes):
        ...

    def read(self, n: int) -> bytes:
        ...

    def close(self):
        ...

    def flush(self):
        ...


class SubprocPool:
    """
    Mimic a concurrent.futures.ProcessPoolExecutor, but wrap it in
    a subprocess.Popen() to try to avoid issues with forking/spawning
    """

    def __init__(self, nprocs: int):
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                re.sub(r"[.][^.]*$", "", __name__),
                f"--workers={nprocs}",
                f"--parent={os.getpid()}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env={
                **os.environ,
            },
        )
        self.write_pipe: Pipe = typing.cast(Pipe, self.process.stdin)
        self.write_lock = threading.Lock()
        self.read_pipe: Pipe = typing.cast(Pipe, self.process.stdout)
        self.read_thread = threading.Thread(target=self._read_thread, daemon=True)
        self.read_thread.start()

        self.futures_lock = threading.Lock()
        self.pending_futures: Dict[int, Future[Any]] = {}
        self.job_id_count = itertools.count()

        self.running = True

    def submit(self, job_fn: Callable[..., Any], *args):
        if args:
            job_fn = functools.partial(job_fn, *args)
        job_data = pickle.dumps(job_fn, pickle.HIGHEST_PROTOCOL)
        future: Future[Any]
        with self.futures_lock:
            job_id = next(self.job_id_count)
            self.pending_futures[job_id] = future = Future()
        with self.write_lock:
            if not self.running:
                raise RuntimeError("submit() on closed pool")
            self.write_pipe.write(_pack_msg(job_id, len(job_data)))
            self.write_pipe.write(job_data)
            self.write_pipe.flush()
        future.set_running_or_notify_cancel()
        return future

    def _read_thread(self):
        try:
            while True:
                job_id, length = _unpack_msg(self.read_pipe.read(msg_bytes))
                if job_id < 0:
                    if self.running:
                        log.warning("SubprocPool unclean exit")
                    self.read_pipe.close()
                    return
                data = self.read_pipe.read(length)
                result = pickle.loads(data)
                with self.futures_lock:
                    if not self.running:
                        return
                    if isinstance(result, Exception):
                        self.pending_futures[job_id].set_exception(result)
                    else:
                        self.pending_futures[job_id].set_result(result)
                    del self.pending_futures[job_id]
        except Exception:
            log.exception("failure in SubprocPool._read_thread")

    def shutdown(self):
        try:
            with self.write_lock:
                if not self.running:
                    return
                self.running = False
                self.write_pipe.write(_pack_msg(-1, -1))
                self.write_pipe.close()
            self.process.wait(10)
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

    def __init__(self, nprocs: int, read_pipe: Pipe, write_pipe: Pipe):
        self.read_pipe = read_pipe
        self.write_pipe = write_pipe
        self.write_lock = threading.Lock()
        self.pool = ProcessPoolExecutor(
            nprocs,
            initializer=functools.partial(_async_compile_initializer, os.getpid()),
        )
        multiprocessing.util.Finalize(
            None, self.pool.shutdown, exitpriority=sys.maxsize
        )
        self.running = True
        _warm_process_pool(self.pool, nprocs)

    def main(self):
        while True:
            job_id, length = _unpack_msg(self.read_pipe.read(msg_bytes))
            if job_id < 0:
                return self._shutdown()
            data = self.read_pipe.read(length)
            self.submit(job_id, data)

    def _shutdown(self):
        with self.write_lock:
            self.running = False
            try:
                self.write_pipe.write(_pack_msg(-1, -1))
                self.write_pipe.close()
            except BrokenPipeError:
                pass  # parent process already shutdown
            self.read_pipe.close()
        self.pool.shutdown()

    def submit(self, job_id, data):
        future = self.pool.submit(functools.partial(SubprocMain.do_job, data))

        def callback(_):
            if not self.running:
                return
            try:
                result = future.result()
            except Exception as e:
                log.exception("Error in subprocess")
                result = pickle.dumps(e, pickle.HIGHEST_PROTOCOL)
            assert isinstance(result, bytes)
            with self.write_lock:
                if self.running:
                    self.write_pipe.write(_pack_msg(job_id, len(result)))
                    self.write_pipe.write(result)
                    self.write_pipe.flush()

        future.add_done_callback(callback)

    @staticmethod
    def do_job(data):
        # do the pickle/unpickle in the sub-subproc
        job = pickle.loads(data)
        result = job()
        return pickle.dumps(result, pickle.HIGHEST_PROTOCOL)


AnyPool = typing.Union[ProcessPoolExecutor, ThreadPoolExecutor, SubprocPool]


def _warm_process_pool(pool: AnyPool, n: int):
    if isinstance(pool, SubprocPool):
        return  # no need
    assert isinstance(pool, ProcessPoolExecutor)

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


def raise_testexc():
    raise TestException
