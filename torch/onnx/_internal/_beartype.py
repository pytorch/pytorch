"""An internal wrapper for the beartype library.

The module returns a no-op decorator when the beartype library is not installed.
"""

try:
    from beartype import beartype
except ImportError:
    def _no_op_decorator(func):
        return func

    beartype = _no_op_decorator
