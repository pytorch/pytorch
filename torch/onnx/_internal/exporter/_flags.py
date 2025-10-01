"""Internal flags for ONNX export."""

from __future__ import annotations

import functools
from typing import Callable, TypeVar
from typing_extensions import ParamSpec


_is_onnx_exporting = False

# Use ParamSpec to preserve parameter types instead of erasing to Any
_P = ParamSpec("_P")
_R = TypeVar("_R")


def set_onnx_exporting_flag(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        global _is_onnx_exporting
        _is_onnx_exporting = True
        try:
            return func(*args, **kwargs)
        finally:
            # Ensure it resets even if an exception occurs
            _is_onnx_exporting = False

    return wrapper
