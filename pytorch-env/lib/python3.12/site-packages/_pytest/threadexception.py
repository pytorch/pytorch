from __future__ import annotations

import threading
import traceback
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Generator
from typing import TYPE_CHECKING
import warnings

import pytest


if TYPE_CHECKING:
    from typing_extensions import Self


# Copied from cpython/Lib/test/support/threading_helper.py, with modifications.
class catch_threading_exception:
    """Context manager catching threading.Thread exception using
    threading.excepthook.

    Storing exc_value using a custom hook can create a reference cycle. The
    reference cycle is broken explicitly when the context manager exits.

    Storing thread using a custom hook can resurrect it if it is set to an
    object which is being finalized. Exiting the context manager clears the
    stored object.

    Usage:
        with threading_helper.catch_threading_exception() as cm:
            # code spawning a thread which raises an exception
            ...
            # check the thread exception: use cm.args
            ...
        # cm.args attribute no longer exists at this point
        # (to break a reference cycle)
    """

    def __init__(self) -> None:
        self.args: threading.ExceptHookArgs | None = None
        self._old_hook: Callable[[threading.ExceptHookArgs], Any] | None = None

    def _hook(self, args: threading.ExceptHookArgs) -> None:
        self.args = args

    def __enter__(self) -> Self:
        self._old_hook = threading.excepthook
        threading.excepthook = self._hook
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self._old_hook is not None
        threading.excepthook = self._old_hook
        self._old_hook = None
        del self.args


def thread_exception_runtest_hook() -> Generator[None]:
    with catch_threading_exception() as cm:
        try:
            yield
        finally:
            if cm.args:
                thread_name = (
                    "<unknown>" if cm.args.thread is None else cm.args.thread.name
                )
                msg = f"Exception in thread {thread_name}\n\n"
                msg += "".join(
                    traceback.format_exception(
                        cm.args.exc_type,
                        cm.args.exc_value,
                        cm.args.exc_traceback,
                    )
                )
                warnings.warn(pytest.PytestUnhandledThreadExceptionWarning(msg))


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_setup() -> Generator[None]:
    yield from thread_exception_runtest_hook()


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_call() -> Generator[None]:
    yield from thread_exception_runtest_hook()


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown() -> Generator[None]:
    yield from thread_exception_runtest_hook()
