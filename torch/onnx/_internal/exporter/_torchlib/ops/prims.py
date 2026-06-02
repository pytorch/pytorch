"""torch.ops.prims operators."""
# mypy: disable-error-code="misc,arg-type,return-value,type-var"

from __future__ import annotations

from typing import Any

from onnxscript.onnx_opset import opset18 as op

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import (
    BFLOAT16,
    DOUBLE,
    FLOAT,
    FLOAT16,
    INT16,
    INT32,
    INT64,
    INT8,
    TRealOrUInt8,
)
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


prims = torch.ops.prims
_NARROW_SIGNED_INTEGER_DTYPES = frozenset({INT8.dtype, INT16.dtype})
_SIGNED_INTEGER_DTYPES = frozenset({INT8.dtype, INT16.dtype, INT32.dtype, INT64.dtype})


def _dtype(arg: Any) -> Any | None:
    return getattr(arg, "dtype", None)


def _is_integer_dtype(arg: Any) -> bool:
    dtype = _dtype(arg)
    return bool(dtype is not None and dtype.is_integer())


def _is_signed_integer_dtype(arg: Any) -> bool:
    return _dtype(arg) in _SIGNED_INTEGER_DTYPES


def _is_float_dtype(arg: Any) -> bool:
    dtype = _dtype(arg)
    return bool(dtype is not None and not dtype.is_integer())


def _is_integer_remainder(self: Any, other: Any) -> bool:
    if isinstance(self, float) or isinstance(other, float):
        return False
    if _dtype(self) is not None:
        return _is_integer_dtype(self)
    if _dtype(other) is not None:
        return _is_integer_dtype(other)
    return isinstance(self, int) and isinstance(other, int)


def _integer_result_dtype(self: Any, other: Any) -> Any | None:
    if _is_integer_dtype(self):
        return _dtype(self)
    if _is_integer_dtype(other):
        return _dtype(other)
    return None


def _is_signed_integer_remainder(self: Any, other: Any) -> bool:
    return _is_signed_integer_dtype(self) or _is_signed_integer_dtype(other)


def _default_float_dtype() -> int:
    dtype = torch.get_default_dtype()
    if dtype == torch.float16:
        return FLOAT16.dtype
    if dtype == torch.float32:
        return FLOAT.dtype
    if dtype == torch.float64:
        return DOUBLE.dtype
    if dtype == torch.bfloat16:
        return BFLOAT16.dtype
    raise AssertionError(f"Unsupported default dtype: {dtype}")


def _floating_result_dtype(self: Any, other: Any) -> Any | None:
    if (isinstance(self, float) and _is_integer_dtype(other)) or (
        isinstance(other, float) and _is_integer_dtype(self)
    ):
        return _default_float_dtype()
    if _is_float_dtype(self):
        return _dtype(self)
    if _is_float_dtype(other):
        return _dtype(other)
    if isinstance(self, float) or isinstance(other, float):
        return _default_float_dtype()
    return None


def _normalize_operands(self: Any, other: Any, *, integer: bool) -> tuple[Any, Any]:
    self_dtype = _dtype(self)
    other_dtype = _dtype(other)

    if integer:
        if self_dtype is None and other_dtype is not None:
            self = op.CastLike(self, other)
        elif other_dtype is None and self_dtype is not None:
            other = op.CastLike(other, self)
        return self, other

    if isinstance(self, float) and _is_integer_dtype(other):
        other = op.Cast(other, to=_default_float_dtype())
        self = op.CastLike(self, other)
    elif isinstance(other, float) and _is_integer_dtype(self):
        self = op.Cast(self, to=_default_float_dtype())
        other = op.CastLike(other, self)
    elif self_dtype is None and other_dtype is not None:
        self = op.CastLike(self, other)
    elif other_dtype is None and self_dtype is not None:
        other = op.CastLike(other, self)

    return self, other


def _floating_remainder(self: Any, other: Any) -> Any:
    fmod = op.Mod(self, other, fmod=1)
    zero = op.CastLike(0, fmod)
    nonzero = op.Not(op.Equal(fmod, zero))
    signs_differ = op.Xor(op.Less(fmod, zero), op.Less(other, zero))
    return op.Where(op.And(nonzero, signs_differ), op.Add(fmod, other), fmod)


def _signed_integer_remainder(self: Any, other: Any) -> Any:
    safe_other = op.Where(
        op.Equal(other, op.CastLike(-1, other)), op.CastLike(1, other), other
    )
    return op.Mod(self, safe_other)


@onnx_impl(prims.remainder.default, trace_only=True)
def prims_remainder(self: TRealOrUInt8, other: TRealOrUInt8) -> TRealOrUInt8:
    """remainder(Tensor self, Tensor other) -> Tensor"""

    integer = _is_integer_remainder(self, other)
    integer_result_dtype = _integer_result_dtype(self, other) if integer else None
    result_dtype = None if integer else _floating_result_dtype(self, other)
    self, other = _normalize_operands(self, other, integer=integer)

    if integer:
        if integer_result_dtype in _NARROW_SIGNED_INTEGER_DTYPES:
            result = _signed_integer_remainder(
                op.Cast(self, to=INT32.dtype), op.Cast(other, to=INT32.dtype)
            )
            return op.Cast(result, to=integer_result_dtype)
        if _is_signed_integer_remainder(self, other):
            return _signed_integer_remainder(self, other)
        return op.Mod(self, other)

    if result_dtype == BFLOAT16.dtype:
        result = _floating_remainder(
            op.Cast(self, to=FLOAT.dtype), op.Cast(other, to=FLOAT.dtype)
        )
        return op.Cast(result, to=BFLOAT16.dtype)

    return _floating_remainder(self, other)
