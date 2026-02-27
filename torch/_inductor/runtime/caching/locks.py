"""Lock acquisition utilities

This module provides safe and unsafe lock acquisition functions for both threading.Lock
and FileLock objects, with configurable timeout behaviors. It supports three timeout modes:
blocking (infinite wait), non-blocking (immediate), and blocking with timeout (finite wait).

The module offers both context manager and manual acquisition patterns:
- Safe acquisition: Uses context managers that automatically handle lock release
- Unsafe acquisition: Manual acquisition that requires explicit release by the caller
"""

from __future__ import annotations

from contextlib import _GeneratorContextManager, contextmanager, ExitStack
from typing import TYPE_CHECKING, TypeAlias
from typing_extensions import Protocol

from filelock import BaseFileLock, Timeout

from . import exceptions


if TYPE_CHECKING:
    from collections.abc import Generator
    from threading import Lock

    from .implementations import _CacheImpl


_LockContextManager: TypeAlias = _GeneratorContextManager[None, None, None]


class _LockProtocol(Protocol):  # noqa: PYI046
    def __call__(self, timeout: float | None = None) -> _LockContextManager: ...


# Infinite timeout - blocks indefinitely until lock is acquired.
_BLOCKING: float = -1
# No timeout - returns immediately if lock cannot be acquired.
_NON_BLOCKING: float = 0
# Finite timeout - blocks for a specified duration before raising a timeout error.
_BLOCKING_WITH_TIMEOUT: float = 60.0
# Default timeout for lock acquisition.
_DEFAULT_TIMEOUT: float = _BLOCKING_WITH_TIMEOUT


@contextmanager
def _acquire_lock_with_timeout(
    lock: Lock,
    timeout: float | None = None,
) -> Generator[None, None, None]:
    """Context manager that safely acquires a threading.Lock with timeout and automatically releases it.

    This function provides a safe way to acquire a lock with timeout support, ensuring
    the lock is always released even if an exception occurs during execution.

    Args:
        lock: The threading.Lock object to acquire
        timeout: Timeout in seconds. If None, uses _DEFAULT_TIMEOUT.
                - Use _BLOCKING (-1.0) for infinite wait
                - Use _NON_BLOCKING (0.0) for immediate return
                - Use positive value for finite timeout

    Yields:
        None: Yields control to the caller while holding the lock

    Raises:
        LockTimeoutError: If the lock cannot be acquired within the timeout period

    Example:
        with _acquire_lock_with_timeout(my_lock, timeout=30.0):
            # Critical section - lock is held
            perform_critical_operation()
        # Lock is automatically released here
    """
    _unsafe_acquire_lock_with_timeout(lock, timeout=timeout)

    try:
        yield
    finally:
        lock.release()


def _unsafe_acquire_lock_with_timeout(lock: Lock, timeout: float | None = None) -> None:
    """Acquire a threading.Lock with timeout without automatic release (unsafe).

    This function acquires a lock with timeout support but does NOT automatically
    release it. The caller is responsible for releasing the lock explicitly.
    Use this only when you need manual control over lock lifetime.

    Args:
        lock: The threading.Lock object to acquire
        timeout: Timeout in seconds. If None, uses _DEFAULT_TIMEOUT.
                - Use _BLOCKING (-1.0) for infinite wait
                - Use _NON_BLOCKING (0.0) for immediate return
                - Use positive value for finite timeout

    Raises:
        LockTimeoutError: If the lock cannot be acquired within the timeout period

    Warning:
        This is an "unsafe" function because it does not automatically release
        the lock. Always call lock.release() when done, preferably in a try/finally
        block or use the safe _acquire_lock_with_timeout context manager instead.

    Example:
        lock = Lock()
        try:
            _unsafe_acquire_lock_with_timeout(lock, timeout=30.0)
            # Critical section - lock is held
            perform_critical_operation()
        finally:
            lock.release()  # Must manually release!
    """
    _timeout: float = timeout if timeout is not None else _DEFAULT_TIMEOUT
    if not lock.acquire(timeout=_timeout):
        raise exceptions.LockTimeoutError(lock, _timeout)


@contextmanager
def _acquire_flock_with_timeout(
    flock: BaseFileLock,
    timeout: float | None = None,
) -> Generator[None, None, None]:
    """Context manager that safely acquires a FileLock with timeout and automatically releases it.

    This function provides a safe way to acquire a file lock with timeout support, ensuring
    the lock is always released even if an exception occurs during execution.

    Args:
        flock: The FileLock object to acquire
        timeout: Timeout in seconds. If None, uses _DEFAULT_TIMEOUT.
                - Use _BLOCKING (-1.0) for infinite wait
                - Use _NON_BLOCKING (0.0) for immediate return
                - Use positive value for finite timeout

    Yields:
        None: Yields control to the caller while holding the file lock

    Raises:
        FileLockTimeoutError: If the file lock cannot be acquired within the timeout period

    Example:
        flock = FileLock("/tmp/my_process.lock")
        with _acquire_flock_with_timeout(flock, timeout=30.0):
            # Critical section - file lock is held
            perform_exclusive_file_operation()
        # File lock is automatically released here
    """
    _unsafe_acquire_flock_with_timeout(flock, timeout=timeout)

    try:
        yield
    finally:
        flock.release()


def _unsafe_acquire_flock_with_timeout(
    flock: BaseFileLock,
    timeout: float | None,
) -> None:
    """Acquire a FileLock with timeout without automatic release (unsafe).

    This function acquires a file lock with timeout support but does NOT automatically
    release it. The caller is responsible for releasing the lock explicitly.
    Use this only when you need manual control over lock lifetime.

    Args:
        flock: The FileLock object to acquire
        timeout: Timeout in seconds. If None, uses _DEFAULT_TIMEOUT.
                - Use _BLOCKING (-1.0) for infinite wait
                - Use _NON_BLOCKING (0.0) for immediate return
                - Use positive value for finite timeout

    Raises:
        FileLockTimeoutError: If the file lock cannot be acquired within the timeout period

    Warning:
        This is an "unsafe" function because it does not automatically release
        the lock. Always call flock.release() when done, preferably in a try/finally
        block or use the safe _acquire_flock_with_timeout context manager instead.

    Example:
        flock = FileLock("/tmp/my_process.lock")
        try:
            _unsafe_acquire_flock_with_timeout(flock, timeout=30.0)
            # Critical section - file lock is held
            perform_exclusive_file_operation()
        finally:
            flock.release()  # Must manually release!
    """
    _timeout: float = timeout if timeout is not None else _DEFAULT_TIMEOUT
    try:
        _ = flock.acquire(timeout=_timeout)
    except Timeout as err:
        raise exceptions.FileLockTimeoutError(flock, _timeout) from err


@contextmanager
def _acquire_many_impl_locks_with_timeout(
    *impls: _CacheImpl,
    timeout: float | None = None,
) -> Generator[None, None, None]:
    with ExitStack() as stack:
        for impl in impls:
            stack.enter_context(impl.lock(timeout))
        yield
