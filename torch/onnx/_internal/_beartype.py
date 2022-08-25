"""An internal wrapper for the beartype library.

The module returns a no-op decorator when the beartype library is not installed.
"""
import functools
import traceback
import typing
import warnings

from torch.onnx import errors, _exporter_states
from torch.onnx._globals import GLOBALS


try:
    from beartype import roar as _roar
    import beartype as _beartype_lib

    # Beartype warns when we import from typing because the types are deprecated
    # in Python 3.9. But there will be a long time until we can move to using
    # the native container types for type annotations (when 3.9 is the lowest
    # supported version). So we silence the warning.
    warnings.filterwarnings(
        "ignore",
        category=_roar.BeartypeDecorHintPep585DeprecationWarning,
    )
    has_beartype = True
except ImportError:
    has_beartype = False


def _no_op_decorator(func):
    return func


def _create_beartype_decorator(
    runtime_check_state: _exporter_states.RuntimeTypeCheckState,
):

    if runtime_check_state == _exporter_states.RuntimeTypeCheckState.DISABLED:
        return _no_op_decorator
    elif runtime_check_state == _exporter_states.RuntimeTypeCheckState.ERRORS:
        # Enable runtime type checking which errors on any type hint violation.
        if has_beartype:
            return _beartype_lib.beartype
        else:
            # If the beartype library is not installed, return a no-op decorator
            warnings.warn(
                "TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK is set to '1', "
                "but the beartype library is not installed."
                "Install beartype with `pip install beartype` to enable runtime type checking."
            )
            return _no_op_decorator
    else:
        # Warning only
        if has_beartype:

            def beartype(func):
                """Warn on type hint violation."""

                if "return" in func.__annotations__:
                    # Remove the return type from the func function's
                    # annotations so that the beartype decorator does not complain
                    # about the return type.
                    return_type = func.__annotations__["return"]
                    del func.__annotations__["return"]
                    beartyped = _beartype_lib.beartype(func)
                    # Restore the return type to the func function's annotations
                    # FIXME(justinchuby): Make sure this is correct
                    func.__annotations__["return"] = return_type
                else:
                    beartyped = _beartype_lib.beartype(func)

                @functools.wraps(func)
                def _coerce_beartype_exceptions_to_warnings(*args, **kwargs):
                    try:
                        return beartyped(*args, **kwargs)
                    except _roar.BeartypeCallHintViolation:
                        # Fall back to the original function if the beartype hint is violated.
                        warnings.warn(
                            traceback.format_exc(),
                            category=errors.CallHintViolationWarning,
                            stacklevel=2,
                        )
                    finally:
                        return func(*args, **kwargs)  # noqa: B012

                return _coerce_beartype_exceptions_to_warnings

            return beartype

        else:
            # Beartype is not installed. Return a no-op decorator.
            return _no_op_decorator


if typing.TYPE_CHECKING:
    beartype = _no_op_decorator
else:
    beartype = _create_beartype_decorator(GLOBALS.runtime_type_check_state)
    # Make sure that the beartype decorator is enabled whichever path we took.
    assert beartype is not None
