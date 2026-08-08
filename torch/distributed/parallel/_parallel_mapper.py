"""Parallel mapper for multi-threaded execution with collective operation serialization.

Enables parallel Python-side computation in separate threads while serializing
torch.distributed collective operations (all_reduce, etc.) on the main thread.
"""

import os
import threading
from typing import Any, Callable, List, Optional, TypeVar

from torch.distributed._collective_interceptor import (
    set_collective_interceptor,
    get_collective_interceptor,
    set_worker_id,
    get_worker_id,
    clear_worker_id,
)

R = TypeVar('R')

_thread_local = threading.local()

# Sync timeout (seconds). 0 or unset means no timeout.
# Used to detect deadlocks caused by precondition violations.
_SYNC_TIMEOUT: Optional[float] = None
_timeout_env = os.environ.get("PARALLEL_MAPPER_TIMEOUT", "")
if _timeout_env:
    _SYNC_TIMEOUT = float(_timeout_env)


# ============ sync_wrap ============

def sync_wrap(func: Callable) -> Callable:
    """Wrap a function to execute synchronously on the main thread.

    When called inside a parallel_map worker thread, the wrapped function
    will be submitted to the main thread for serialized execution.
    When called outside a worker thread, the original function runs directly.

    Usage:
        my_queue_put = sync_wrap(my_queue.put)
        my_custom_op = sync_wrap(some_module.sync_operation)
    """
    def _wrapper(*args, **kwargs):
        worker_id = get_worker_id()
        coordinator = getattr(_thread_local, '_coordinator', None)
        if worker_id is not None and coordinator is not None:
            return coordinator.submit(worker_id, func, *args, **kwargs)
        return func(*args, **kwargs)
    _wrapper.__wrapped__ = func
    _wrapper.__name__ = getattr(func, '__name__', str(func))
    return _wrapper


# ============ Exception propagation ============

class _PropagatedError(BaseException):
    """Internal exception: propagated to waiting workers when another worker fails.
    Inherits from BaseException to prevent being caught by user code's
    except Exception handlers."""
    def __init__(self, original: BaseException):
        self.original = original


# ============ Sync coordinator ============

class _SyncRequest:
    __slots__ = ['worker_id', 'func', 'args', 'kwargs', 'event', 'result', 'error']

    def __init__(
        self,
        worker_id: int,
        func: Callable,
        args: tuple,
        kwargs: dict,
        event: threading.Event,
    ):
        self.worker_id = worker_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.event = event
        self.result: Any = None
        self.error: Any = None


class _SyncCoordinator:
    """Waits for sync requests in worker_id order and dispatches execution.

    Worker 0's request executes immediately when it arrives without waiting
    for other workers. Supports exception propagation and timeout detection.
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self._cond = threading.Condition()
        self._pending: List[Optional[_SyncRequest]] = [None] * num_workers
        self._events = [threading.Event() for _ in range(num_workers)]
        self._next_id = 0
        self._done_count = 0
        self._error: Optional[BaseException] = None

    def submit(self, worker_id: int, func: Callable, *args, **kwargs) -> Any:
        event = self._events[worker_id]
        event.clear()
        req = _SyncRequest(worker_id, func, args, kwargs, event)
        with self._cond:
            if self._error is not None:
                raise _PropagatedError(self._error)
            self._pending[worker_id] = req
            self._cond.notify()
        event.wait()
        if req.error is not None:
            raise _PropagatedError(req.error)
        return req.result

    def notify_error(self, exc: BaseException):
        """Called when a worker errors out.

        Sets error flag and wakes the main thread.
        """
        with self._cond:
            if self._error is None:
                self._error = exc
            self._cond.notify()

    def notify_worker_done(self):
        with self._cond:
            self._done_count += 1
            self._cond.notify()

    def _release_all_pending(self, exc: BaseException):
        """Release all submitted but still waiting requests.

        Caller must hold _cond lock.
        """
        for i, req in enumerate(self._pending):
            if req is not None:
                req.error = exc
                req.event.set()
                self._pending[i] = None

    def get_next_request(self) -> Optional[_SyncRequest]:
        """Wait for the next sync request in worker_id order.
        Returns None when all workers are done or a worker has errored."""
        with self._cond:
            while True:
                if self._error is not None:
                    self._release_all_pending(self._error)
                    return None
                if self._pending[self._next_id] is not None:
                    req = self._pending[self._next_id]
                    self._pending[self._next_id] = None
                    self._next_id = (self._next_id + 1) % self.num_workers
                    return req
                if self._done_count == self.num_workers:
                    return None
                if not self._cond.wait(timeout=_SYNC_TIMEOUT):
                    submitted = [
                        i for i, r in enumerate(self._pending)
                        if r is not None
                    ]
                    raise TimeoutError(
                        f"parallel_mapper: sync timeout ({_SYNC_TIMEOUT}s). "
                        f"Waiting for worker {self._next_id}, "
                        f"submitted: {submitted}, "
                        f"done: {self._done_count}/{self.num_workers}. "
                        f"Likely cause: workers have inconsistent sync call patterns."
                    )


# ============ Worker thread ============

class _WorkerThread(threading.Thread):
    """Persistent worker thread.

    Receives tasks via task_event, signals completion via done_event.
    """

    def __init__(self, worker_id: int):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self._task_event = threading.Event()
        self._done_event = threading.Event()
        self._func: Optional[Callable] = None
        self._arg: Any = None
        self._result: Any = None
        self._exception: Optional[BaseException] = None
        self._coordinator: Optional[_SyncCoordinator] = None
        self.start()

    def assign(
        self,
        func: Callable,
        arg: Any,
        coordinator: _SyncCoordinator,
        cuda_device: Optional[int] = None,
    ):
        self._func = func
        self._arg = arg
        self._result = None
        self._exception = None
        self._coordinator = coordinator
        self._cuda_device = cuda_device
        self._done_event.clear()
        self._task_event.set()

    def wait_done(self):
        self._done_event.wait()
        return self._result

    def run(self):
        while True:
            self._task_event.wait()
            self._task_event.clear()

            # Propagate CUDA device from the main thread
            if self._cuda_device is not None:
                import torch
                torch.cuda.set_device(self._cuda_device)

            coordinator = self._coordinator
            set_worker_id(self.worker_id)
            _thread_local._coordinator = coordinator

            # Set PyTorch interceptor: collective ops are submitted to the coordinator
            set_collective_interceptor(
                lambda op_name, original_fn, *args,
                       _coord=coordinator, _wid=self.worker_id, **kwargs:
                    _coord.submit(_wid, original_fn, *args, **kwargs)
            )

            try:
                self._result = self._func(*self._arg)
            except _PropagatedError:
                # Exception propagated from another worker, no need to record
                pass
            except BaseException as e:
                self._exception = e
                if coordinator is not None:
                    coordinator.notify_error(e)
            finally:
                set_collective_interceptor(None)
                _thread_local._coordinator = None
                clear_worker_id()
                self._done_event.set()
                if coordinator is not None:
                    coordinator.notify_worker_done()
                    self._coordinator = None


# ============ Parallel pool ============

class _ParallelPool:
    """Parallel map thread pool with main thread coordinating sync operations.

    Only one _run_batch executes at a time to prevent worker state conflicts
    from concurrent calls.
    """

    def __init__(self, max_workers: int = 0):
        self._max_workers = max_workers
        self._workers: List[_WorkerThread] = []
        self._lock = threading.Lock()

    def _ensure_workers(self, n: int):
        while len(self._workers) < n:
            self._workers.append(_WorkerThread(len(self._workers)))

    def _run_batch(self, func: Callable, batch_args: List[Any]):
        num = len(batch_args)
        with self._lock:
            self._ensure_workers(num)
            coordinator = _SyncCoordinator(num)

            # Capture current CUDA device to propagate to worker threads
            cuda_device = None
            try:
                import torch
                if (
                    torch.cuda.is_available()
                    and torch.cuda.current_device() is not None
                ):
                    cuda_device = torch.cuda.current_device()
            except Exception:
                pass

            for i, arg in enumerate(batch_args):
                self._workers[i].assign(func, arg, coordinator, cuda_device)

            # Main thread: wait for, execute, and release requests in worker_id order
            req = None
            try:
                while True:
                    req = coordinator.get_next_request()
                    if req is None:
                        break
                    req.result = req.func(*req.args, **req.kwargs)
                    req.event.set()
                    req = None
            except BaseException as e:
                # Main thread exception during sync op (e.g. NCCL error, timeout).
                # Must release all waiting workers to prevent deadlock.
                coordinator.notify_error(e)
                if req is not None and not req.event.is_set():
                    req.error = e
                    req.event.set()
                # get_next_request detects _error, releases
                # remaining pending, returns None
                coordinator.get_next_request()

            # Wait for all workers, collect results
            results = []
            first_error = coordinator._error
            for i in range(num):
                self._workers[i].wait_done()
                exc = self._workers[i]._exception
                if exc is not None and first_error is None:
                    first_error = exc
                results.append(self._workers[i]._result)

            if first_error is not None:
                raise first_error

            return results

    def run(self, func: Callable, args: Optional[List[Any]] = None,
            max_workers: Optional[int] = None) -> List[Any]:
        # When args is None, invoke func with empty args
        # based on max_workers/_max_workers count
        if args is None:
            n = max_workers if max_workers is not None else self._max_workers
            if not n:
                raise ValueError("max_workers must be specified when args is None")
            args = [() for _ in range(n)]

        # Determine max concurrency for this run
        max_w = max_workers if max_workers is not None else (
            self._max_workers or len(args)
        )
        max_w = min(max_w, len(args))

        if max_w <= 1:
            return [func(*a) for a in args]

        results = []
        # Execute in batches, each batch has at most max_w workers
        for i in range(0, len(args), max_w):
            batch = args[i:i + max_w]
            results.extend(self._run_batch(func, batch))
        return results


# Global singleton
_pool = _ParallelPool()


# ============ Public API ============

def parallel_starmap(fn: Callable[..., R], args: Optional[List[Any]] = None,
                     max_workers: Optional[int] = None) -> List[R]:
    """Parallel starmap, analogous to itertools.starmap.

    Each element in args is a tuple that gets unpacked and passed to fn.

    Args:
        fn: The mapping function.
        args: List of argument tuples, e.g. [(a1, b1), (a2, b2)].
              Each tuple is unpacked: fn(*tuple). If None, fn is called
              max_workers times with no arguments.
        max_workers: Maximum concurrent threads. Excess items are batched.

    Returns:
        List of results.

    Example:
        parallel_starmap(fn, [(1, 'a'), (2, 'b')])  # fn(1,'a'), fn(2,'b')
    """
    return _pool.run(fn, args, max_workers)


def parallel_map(fn: Callable[[Any], R], iterable, max_workers: Optional[int] = None) -> List[R]:
    """Parallel map, analogous to builtin map(fn, iterable).

    Each element in iterable is passed as a single argument to fn.

    Args:
        fn: The mapping function.
        iterable: Input sequence. Each element is the sole argument to fn.
        max_workers: Maximum concurrent threads. Excess items are batched.

    Returns:
        List of results.

    Example:
        parallel_map(fn, [1, 2, 3])  # fn(1), fn(2), fn(3)
    """
    args = [(x,) for x in iterable]
    return _pool.run(fn, args, max_workers)


def parallel_multi_apply(func, *args, max_workers: Optional[int] = None, **kwargs):
    """Parallel version of multi_apply, interface compatible with mmdet.core.multi_apply.

    Args:
        func: The function to apply.
        *args: Multiple equal-length lists, zipped and passed element-wise to func.
        max_workers: Maximum concurrent threads.
        **kwargs: Extra keyword arguments bound to func via functools.partial.

    Returns:
        tuple: Tuple of result lists, one per return value position.

    Example:
        parallel_multi_apply(fn, [a1, a2], [b1, b2])
        # Equivalent to fn(a1, b1), fn(a2, b2), returns ([r1, r2], [s1, s2])
    """
    from functools import partial
    pfunc = partial(func, **kwargs) if kwargs else func
    num_args = len(args[0]) if args else 0
    if num_args == 0:
        return tuple()
    packed_args = [tuple(arg[i] for arg in args) for i in range(num_args)]
    map_results = _pool.run(pfunc, packed_args, max_workers)
    return tuple(map(list, zip(*map_results)))
