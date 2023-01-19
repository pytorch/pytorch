from contextlib import contextmanager

from pip._vendor.contextlib2 import ExitStack

from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import Iterator, ContextManager, TypeVar

    _T = TypeVar('_T', covariant=True)


class CommandContextMixIn(object):
    def __init__(self):
        # type: () -> None
        super(CommandContextMixIn, self).__init__()
        self._in_main_context = False
        self._main_context = ExitStack()

    @contextmanager
    def main_context(self):
        # type: () -> Iterator[None]
        assert not self._in_main_context

        self._in_main_context = True
        try:
            with self._main_context:
                yield
        finally:
            self._in_main_context = False

    def enter_context(self, context_provider):
        # type: (ContextManager[_T]) -> _T
        assert self._in_main_context

        return self._main_context.enter_context(context_provider)
