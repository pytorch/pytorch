"""Thread-local collective operation interceptor for parallel execution.

This module provides the interceptor primitives used by:
- torch/distributed/distributed_c10d.py: checks interceptor at collective op entry
- torch/distributed/parallel/: sets interceptor in worker threads

When a thread-local interceptor is set, collective operations (all_reduce, etc.)
are redirected through it instead of executing directly. This enables the
parallel mapper to serialize collective ops on the main thread while allowing
Python-side computation to run concurrently in worker threads.
"""

import threading
from typing import Any, Callable, Optional

_thread_local = threading.local()


def set_collective_interceptor(fn: Optional[Callable]) -> None:
    """Set the collective operation interceptor for the current thread.

    Args:
        fn: Interceptor callback with signature
            fn(op_name: str, original_fn: Callable, *args, **kwargs) -> Any
            Set to None to clear the interceptor.
    """
    _thread_local.interceptor = fn


def get_collective_interceptor() -> Optional[Callable]:
    """Get the collective operation interceptor for the current thread.

    Returns None if no interceptor is set (the common case),
    or if we are already inside an interceptor call (re-entrancy guard).
    """
    if getattr(_thread_local, "_in_interceptor", False):
        return None
    return getattr(_thread_local, "interceptor", None)


def call_collective_interceptor(
    interceptor: Callable,
    op_name: str,
    fn: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call the interceptor with re-entrancy protection.

    Sets a thread-local flag so that if the interceptor calls back into a
    collective op on the same thread, the interceptor check is skipped and
    the op executes directly. This prevents infinite recursion.
    """
    _thread_local._in_interceptor = True
    try:
        return interceptor(op_name, fn, *args, **kwargs)
    finally:
        _thread_local._in_interceptor = False


def set_worker_id(worker_id: int) -> None:
    """Set the parallel mapper worker id for the current thread."""
    _thread_local.worker_id = worker_id


def get_worker_id() -> Optional[int]:
    """Get the parallel mapper worker id for the current thread.

    Returns None if not running inside a parallel_map worker.
    """
    return getattr(_thread_local, "worker_id", None)


def clear_worker_id() -> None:
    """Clear the parallel mapper worker id for the current thread."""
    _thread_local.worker_id = None
