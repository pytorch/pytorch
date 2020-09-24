import sys

__all__ = ['register_after_fork']

if sys.platform == 'win32' or sys.version_info < (3, 7):
    import multiprocessing.util as _util

    def _register(func):
        def wrapper(arg):
            func()
        _util.register_after_fork(_register, wrapper)
else:
    import os

    def _register(func):
        os.register_at_fork(after_in_child=func)

def register_after_fork(func):
    """Register a callable to be executed in the child process after a fork.

    Note:
        In python < 3.7 this will only work with processes created using the
        ``multiprocessing`` module. In python >= 3.7 it also works with
        ``os.fork()``.

    Arguments:
        func (function): Function taking no arguments to be called in the child after fork

    """
    _register(func)
