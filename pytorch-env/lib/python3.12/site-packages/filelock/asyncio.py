"""An asyncio-based implementation of the file lock."""  # noqa: A005

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import dataclass
from threading import local
from typing import TYPE_CHECKING, Any, Callable, NoReturn, cast

from ._api import BaseFileLock, FileLockContext, FileLockMeta
from ._error import Timeout
from ._soft import SoftFileLock
from ._unix import UnixFileLock
from ._windows import WindowsFileLock

if TYPE_CHECKING:
    import sys
    from concurrent import futures
    from types import TracebackType

    if sys.version_info >= (3, 11):  # pragma: no cover (py311+)
        from typing import Self
    else:  # pragma: no cover (<py311)
        from typing_extensions import Self


_LOGGER = logging.getLogger("filelock")


@dataclass
class AsyncFileLockContext(FileLockContext):
    """A dataclass which holds the context for a ``BaseAsyncFileLock`` object."""

    #: Whether run in executor
    run_in_executor: bool = True

    #: The executor
    executor: futures.Executor | None = None

    #: The loop
    loop: asyncio.AbstractEventLoop | None = None


class AsyncThreadLocalFileContext(AsyncFileLockContext, local):
    """A thread local version of the ``FileLockContext`` class."""


class AsyncAcquireReturnProxy:
    """A context-aware object that will release the lock file when exiting."""

    def __init__(self, lock: BaseAsyncFileLock) -> None:  # noqa: D107
        self.lock = lock

    async def __aenter__(self) -> BaseAsyncFileLock:  # noqa: D105
        return self.lock

    async def __aexit__(  # noqa: D105
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.lock.release()


class AsyncFileLockMeta(FileLockMeta):
    def __call__(  # type: ignore[override] # noqa: PLR0913
        cls,  # noqa: N805
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        mode: int = 0o644,
        thread_local: bool = False,  # noqa: FBT001, FBT002
        *,
        blocking: bool = True,
        is_singleton: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        run_in_executor: bool = True,
        executor: futures.Executor | None = None,
    ) -> BaseAsyncFileLock:
        if thread_local and run_in_executor:
            msg = "run_in_executor is not supported when thread_local is True"
            raise ValueError(msg)
        instance = super().__call__(
            lock_file=lock_file,
            timeout=timeout,
            mode=mode,
            thread_local=thread_local,
            blocking=blocking,
            is_singleton=is_singleton,
            loop=loop,
            run_in_executor=run_in_executor,
            executor=executor,
        )
        return cast(BaseAsyncFileLock, instance)


class BaseAsyncFileLock(BaseFileLock, metaclass=AsyncFileLockMeta):
    """Base class for asynchronous file locks."""

    def __init__(  # noqa: PLR0913
        self,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        mode: int = 0o644,
        thread_local: bool = False,  # noqa: FBT001, FBT002
        *,
        blocking: bool = True,
        is_singleton: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        run_in_executor: bool = True,
        executor: futures.Executor | None = None,
    ) -> None:
        """
        Create a new lock object.

        :param lock_file: path to the file
        :param timeout: default timeout when acquiring the lock, in seconds. It will be used as fallback value in \
            the acquire method, if no timeout value (``None``) is given. If you want to disable the timeout, set it \
            to a negative value. A timeout of 0 means that there is exactly one attempt to acquire the file lock.
        :param mode: file permissions for the lockfile
        :param thread_local: Whether this object's internal context should be thread local or not. If this is set to \
            ``False`` then the lock will be reentrant across threads.
        :param blocking: whether the lock should be blocking or not
        :param is_singleton: If this is set to ``True`` then only one instance of this class will be created \
            per lock file. This is useful if you want to use the lock object for reentrant locking without needing \
            to pass the same object around.
        :param loop: The event loop to use. If not specified, the running event loop will be used.
        :param run_in_executor: If this is set to ``True`` then the lock will be acquired in an executor.
        :param executor: The executor to use. If not specified, the default executor will be used.

        """
        self._is_thread_local = thread_local
        self._is_singleton = is_singleton

        # Create the context. Note that external code should not work with the context directly and should instead use
        # properties of this class.
        kwargs: dict[str, Any] = {
            "lock_file": os.fspath(lock_file),
            "timeout": timeout,
            "mode": mode,
            "blocking": blocking,
            "loop": loop,
            "run_in_executor": run_in_executor,
            "executor": executor,
        }
        self._context: AsyncFileLockContext = (AsyncThreadLocalFileContext if thread_local else AsyncFileLockContext)(
            **kwargs
        )

    @property
    def run_in_executor(self) -> bool:
        """::return: whether run in executor."""
        return self._context.run_in_executor

    @property
    def executor(self) -> futures.Executor | None:
        """::return: the executor."""
        return self._context.executor

    @executor.setter
    def executor(self, value: futures.Executor | None) -> None:  # pragma: no cover
        """
        Change the executor.

        :param value: the new executor or ``None``
        :type value: futures.Executor | None

        """
        self._context.executor = value

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """::return: the event loop."""
        return self._context.loop

    async def acquire(  # type: ignore[override]
        self,
        timeout: float | None = None,
        poll_interval: float = 0.05,
        *,
        blocking: bool | None = None,
    ) -> AsyncAcquireReturnProxy:
        """
        Try to acquire the file lock.

        :param timeout: maximum wait time for acquiring the lock, ``None`` means use the default
            :attr:`~BaseFileLock.timeout` is and if ``timeout < 0``, there is no timeout and
            this method will block until the lock could be acquired
        :param poll_interval: interval of trying to acquire the lock file
        :param blocking: defaults to True. If False, function will return immediately if it cannot obtain a lock on the
         first attempt. Otherwise, this method will block until the timeout expires or the lock is acquired.
        :raises Timeout: if fails to acquire lock within the timeout period
        :return: a context object that will unlock the file when the context is exited

        .. code-block:: python

            # You can use this method in the context manager (recommended)
            with lock.acquire():
                pass

            # Or use an equivalent try-finally construct:
            lock.acquire()
            try:
                pass
            finally:
                lock.release()

        """
        # Use the default timeout, if no timeout is provided.
        if timeout is None:
            timeout = self._context.timeout

        if blocking is None:
            blocking = self._context.blocking

        # Increment the number right at the beginning. We can still undo it, if something fails.
        self._context.lock_counter += 1

        lock_id = id(self)
        lock_filename = self.lock_file
        start_time = time.perf_counter()
        try:
            while True:
                if not self.is_locked:
                    _LOGGER.debug("Attempting to acquire lock %s on %s", lock_id, lock_filename)
                    await self._run_internal_method(self._acquire)
                if self.is_locked:
                    _LOGGER.debug("Lock %s acquired on %s", lock_id, lock_filename)
                    break
                if blocking is False:
                    _LOGGER.debug("Failed to immediately acquire lock %s on %s", lock_id, lock_filename)
                    raise Timeout(lock_filename)  # noqa: TRY301
                if 0 <= timeout < time.perf_counter() - start_time:
                    _LOGGER.debug("Timeout on acquiring lock %s on %s", lock_id, lock_filename)
                    raise Timeout(lock_filename)  # noqa: TRY301
                msg = "Lock %s not acquired on %s, waiting %s seconds ..."
                _LOGGER.debug(msg, lock_id, lock_filename, poll_interval)
                await asyncio.sleep(poll_interval)
        except BaseException:  # Something did go wrong, so decrement the counter.
            self._context.lock_counter = max(0, self._context.lock_counter - 1)
            raise
        return AsyncAcquireReturnProxy(lock=self)

    async def release(self, force: bool = False) -> None:  # type: ignore[override]  # noqa: FBT001, FBT002
        """
        Releases the file lock. Please note, that the lock is only completely released, if the lock counter is 0.
        Also note, that the lock file itself is not automatically deleted.

        :param force: If true, the lock counter is ignored and the lock is released in every case/

        """
        if self.is_locked:
            self._context.lock_counter -= 1

            if self._context.lock_counter == 0 or force:
                lock_id, lock_filename = id(self), self.lock_file

                _LOGGER.debug("Attempting to release lock %s on %s", lock_id, lock_filename)
                await self._run_internal_method(self._release)
                self._context.lock_counter = 0
                _LOGGER.debug("Lock %s released on %s", lock_id, lock_filename)

    async def _run_internal_method(self, method: Callable[[], Any]) -> None:
        if asyncio.iscoroutinefunction(method):
            await method()
        elif self.run_in_executor:
            loop = self.loop or asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, method)
        else:
            method()

    def __enter__(self) -> NoReturn:
        """
        Replace old __enter__ method to avoid using it.

        NOTE: DO NOT USE `with` FOR ASYNCIO LOCKS, USE `async with` INSTEAD.

        :return: none
        :rtype: NoReturn
        """
        msg = "Do not use `with` for asyncio locks, use `async with` instead."
        raise NotImplementedError(msg)

    async def __aenter__(self) -> Self:
        """
        Acquire the lock.

        :return: the lock object

        """
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Release the lock.

        :param exc_type: the exception type if raised
        :param exc_value: the exception value if raised
        :param traceback: the exception traceback if raised

        """
        await self.release()

    def __del__(self) -> None:
        """Called when the lock object is deleted."""
        with contextlib.suppress(RuntimeError):
            loop = self.loop or asyncio.get_running_loop()
            if not loop.is_running():  # pragma: no cover
                loop.run_until_complete(self.release(force=True))
            else:
                loop.create_task(self.release(force=True))


class AsyncSoftFileLock(SoftFileLock, BaseAsyncFileLock):
    """Simply watches the existence of the lock file."""


class AsyncUnixFileLock(UnixFileLock, BaseAsyncFileLock):
    """Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems."""


class AsyncWindowsFileLock(WindowsFileLock, BaseAsyncFileLock):
    """Uses the :func:`msvcrt.locking` to hard lock the lock file on windows systems."""


__all__ = [
    "AsyncAcquireReturnProxy",
    "AsyncSoftFileLock",
    "AsyncUnixFileLock",
    "AsyncWindowsFileLock",
    "BaseAsyncFileLock",
]
