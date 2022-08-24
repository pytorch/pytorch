"""An internal wrapper for the beartype library.

The module returns a no-op decorator when the beartype library is not installed.
"""
import functools
import traceback
import warnings

from torch.onnx._globals import GLOBALS


class CallHintViolationWarning(UserWarning):
    pass


def _no_op_decorator(func):
    return func


if GLOBALS.runtime_type_check is False:
    beartype = _no_op_decorator
else:
    try:
        import beartype as beartype_lib  # type: ignore[import]
        from beartype import roar as _roar

        # Beartype warns when we import from typing because the types are deprecated
        # in Python 3.9. But there will be a long time until we can move to using
        # the native container types for type annotations (when 3.9 is the lowest
        # supported version). So we silence the warning.
        warnings.filterwarnings(
            "ignore",
            category=_roar.BeartypeDecorHintPep585DeprecationWarning,
        )

        if GLOBALS.runtime_type_check is True:
            # Enable runtime type checking which errors on any type hint violation.
            beartype = beartype_lib.beartype
        else:
            # GLOBALS.runtime_type_check is None, show warnings only.

            def beartype_warn(func):
                if "return" in func.__annotations__:
                    # Remove the return type from the func function's
                    # annotations so that the beartype decorator does not complain
                    # about the return type.
                    del func.__annotations__["return"]

                beartyped = beartype_lib.beartype(func)

                @functools.wraps(beartyped)
                def _coerce_beartype_exceptions_to_warnings(*args, **kwargs):
                    try:
                        return beartyped(*args, **kwargs)
                    except _roar.BeartypeCallHintViolation:
                        # Fall back to the original function if the beartype hint is violated.
                        warnings.warn(
                            traceback.format_exc(),
                            category=CallHintViolationWarning,
                            stacklevel=2,
                        )
                        return func(*args, **kwargs)

                return _coerce_beartype_exceptions_to_warnings

            beartype = beartype_warn

    except ImportError:
        # Beartype is not installed.
        beartype = _no_op_decorator

# Make sure that the beartype decorator is enabled whichever path we took.
assert beartype is not None
