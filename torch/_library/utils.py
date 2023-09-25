import inspect
import sys
from typing import Callable


class Kernel:
    """Models a (function, source location)"""

    def __init__(self, func: Callable, source: str):
        self.func: Callable = func
        self.source: str = source

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def get_source(stacklevel: int) -> str:
    """Get a string that represents the caller.
    Example: "/path/to/foo.py:42"
    Use stacklevel=1 to get the caller's source
    Use stacklevel=2 to get the caller's caller's source
    etc.
    """
    frame = inspect.getframeinfo(sys._getframe(stacklevel))
    source = f"{frame.filename}:{frame.lineno}"
    return source
