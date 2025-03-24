"""Internal flags for ONNX export."""

from __future__ import annotations
import functools
from typing import Callable

_is_onnx_exporting = False

def set_onnx_exporting_flag(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _is_onnx_exporting
        _is_onnx_exporting = True
        try:
            return func(*args, **kwargs)
        finally:
            # Ensure it resets even if an exception occurs
            _is_onnx_exporting = False
    return wrapper