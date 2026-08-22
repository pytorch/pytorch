import sys
from collections.abc import Callable


__all__ = ["register_after_fork"]


def register_after_fork(func: Callable) -> None:
    """Register a callable to be executed in the child process after a fork.

    Works with processes created using the ``multiprocessing`` module and
    with ``os.fork()``.

    Args:
        func (function): Function taking no arguments to be called in the child after fork

    """
    if sys.platform == "win32":
        import multiprocessing.util as _util

        _util.register_after_fork(register_after_fork, lambda _: func())
    else:
        import os

        os.register_at_fork(after_in_child=func)
