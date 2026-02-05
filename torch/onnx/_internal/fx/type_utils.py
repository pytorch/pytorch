# mypy: allow-untyped-defs
"""Utilities for converting and operating on ONNX and torch types."""

from __future__ import annotations

from typing import Any

import torch


def is_torch_symbolic_type(value: Any) -> bool:
    return isinstance(value, (torch.SymBool, torch.SymInt, torch.SymFloat))


def from_scalar_type_to_torch_dtype(scalar_type: type) -> torch.dtype | None:
    return _SCALAR_TYPE_TO_TORCH_DTYPE.get(scalar_type)


_PYTHON_TYPE_TO_TORCH_DTYPE = {
    bool: torch.bool,
    int: torch.int64,
    float: torch.float32,
    complex: torch.complex64,
}

_SYM_TYPE_TO_TORCH_DTYPE = {
    torch.SymInt: torch.int64,
    torch.SymFloat: torch.float32,
    torch.SymBool: torch.bool,
}

_SCALAR_TYPE_TO_TORCH_DTYPE: dict[type, torch.dtype] = {
    **_PYTHON_TYPE_TO_TORCH_DTYPE,
    **_SYM_TYPE_TO_TORCH_DTYPE,  # type: ignore[dict-item]
}
