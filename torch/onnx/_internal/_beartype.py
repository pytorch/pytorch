"""An internal wrapper for the beartype library.

The module returns a no-op decorator when the beartype library is not installed.
"""
import enum
import functools
import os
import traceback
import typing
import warnings
from types import ModuleType

try:
    import beartype as _beartype_lib  # type: ignore[import]
    from beartype import roar as _roar  # type: ignore[import]

    # Beartype warns when we import from typing because the types are deprecated
    # in Python 3.9. But there will be a long time until we can move to using
    # the native container types for type annotations (when 3.9 is the lowest
    # supported version). So we silence the warning.
    warnings.filterwarnings(
        "ignore",
        category=_roar.BeartypeDecorHintPep585DeprecationWarning,
    )
except ImportError:
    _beartype_lib = None  # type: ignore[assignment]
except Exception as e:
    # Warn errors that are not import errors (unexpected).
    warnings.warn(f"{e}", stacklevel=TO_BE_DETERMINED)
    _beartype_lib = None  # type: ignore[assignment]


@enum.unique
class RuntimeTypeCheckState(enum.Enum):
    """Runtime type check state."""

    # Runtime type checking is disabled.
    DISABLED = enum.auto()
    # Runtime type checking is enabled but warnings are shown only.
    WARNINGS = enum.auto()
    # Runtime type checking is enabled.
    ERRORS = enum.auto()


class CallHintViolationWarning(UserWarning):
    """Warning raised when a type hint is violated during a function call."""

    pass


def _no_op_decorator(func):
    return func


def _create_beartype_decorator(
    runtime_check_state: RuntimeTypeCheckState,
):
    # beartype needs to be imported outside of the function and aliased because
    # this module overwrites the name "beartype".

    if runtime_check_state == RuntimeTypeCheckState.DISABLED:
        return _no_op_decorator
    if _beartype_lib is None:
        # If the beartype library is not installed, return a no-op decorator
        return _no_op_decorator

    assert isinstance(_beartype_lib, ModuleType)

    if runtime_check_state == RuntimeTypeCheckState.ERRORS:
        # Enable runtime type checking which errors on any type hint violation.
        return _beartype_lib.beartype

    # Warnings only
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
            func.__annotations__["return"] = return_type
        else:
            beartyped = _beartype_lib.beartype(func)

        @functools.wraps(func)
        def _coerce_beartype_exceptions_to_warnings(*args, **kwargs):
            try:
                return beartyped(*args, **kwargs)
            except _roar.BeartypeCallHintParamViolation:
                # Fall back to the original function if the beartype hint is violated.
                warnings.warn(
                    traceback.format_exc(),
                    category=CallHintViolationWarning,
                    stacklevel=2,
                )

            return func(*args, **kwargs)  # noqa: B012

        return _coerce_beartype_exceptions_to_warnings

    return beartype


if typing.TYPE_CHECKING:
    # This is a hack to make mypy play nicely with the beartype decorator.
    def beartype(func):
        return func

else:
    _TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK = os.getenv(
        "TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK"
    )
    if _TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK == "WARNINGS":
        _runtime_type_check_state = RuntimeTypeCheckState.WARNINGS
    elif _TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK == "DISABLED":
        _runtime_type_check_state = RuntimeTypeCheckState.DISABLED
    else:
        _runtime_type_check_state = RuntimeTypeCheckState.ERRORS
    beartype = _create_beartype_decorator(_runtime_type_check_state)
    # Make sure that the beartype decorator is enabled whichever path we took.
    assert beartype is not None
