import inspect
import sys


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
