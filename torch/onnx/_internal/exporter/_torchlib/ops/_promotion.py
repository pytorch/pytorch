# Owner(s): ["module: onnx"]
"""Operator-owned dtype promotion without executing tensors or FX graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx_ir as ir

import torch
from torch import _prims_common as prims


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.onnx._internal.exporter._ir_builder import TranslationContext


_DTYPES = {
    ir.DataType.BOOL: torch.bool,
    ir.DataType.UINT8: torch.uint8,
    ir.DataType.INT8: torch.int8,
    ir.DataType.INT16: torch.int16,
    ir.DataType.INT32: torch.int32,
    ir.DataType.INT64: torch.int64,
    ir.DataType.UINT16: torch.uint16,
    ir.DataType.UINT32: torch.uint32,
    ir.DataType.UINT64: torch.uint64,
    ir.DataType.FLOAT16: torch.float16,
    ir.DataType.BFLOAT16: torch.bfloat16,
    ir.DataType.FLOAT: torch.float32,
    ir.DataType.DOUBLE: torch.float64,
    ir.DataType.COMPLEX64: torch.complex64,
    ir.DataType.COMPLEX128: torch.complex128,
}
_ONNX_DTYPES = {dtype: onnx_dtype for onnx_dtype, dtype in _DTYPES.items()}
_SOURCE_TYPE = "torch.onnx.original_type"
Scalar = bool | int | float | complex


def source_dtype(
    op: TranslationContext, value: ir.Value, *, packed_complex: bool = False
) -> torch.dtype:
    """Recover the logical dtype, not a packed complex tensor's component dtype."""
    original = value.meta.get(_SOURCE_TYPE)
    if isinstance(original, torch.dtype):
        return original
    if value.dtype not in _DTYPES:
        raise TypeError(
            f"Cannot promote ONNX dtype {value.dtype}. "
            "Set a supported numeric tensor dtype before translating the operator."
        )
    dtype = _DTYPES[value.dtype]
    return (
        prims.corresponding_complex_dtype(dtype)
        if packed_complex and not dtype.is_complex
        else dtype
    )


def dtypes(
    op: TranslationContext,
    *values: ir.Value | Scalar | None,
    kind: str = "DEFAULT",
    packed_complex: bool = False,
) -> tuple[torch.dtype, torch.dtype]:
    """Apply PyTorch's dimensioned/zero-dimensional/wrapped-scalar precedence.

    Only dtype and rank are inspected. The prims helpers implement the dtype
    lattice and operator-specific computation/result rules; no tensor is created.
    """
    highest_type = bool
    tensors = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, ir.Value):
            original = value.meta.get(_SOURCE_TYPE)
            if original in (bool, int, float, complex):
                highest_type = prims.get_higher_type(highest_type, original)
                continue
            dtype = source_dtype(op, value, packed_complex=packed_complex)
            highest_type = prims.get_higher_type(
                highest_type, prims.dtype_to_type(dtype)
            )
            tensors.append((value, dtype))
        elif isinstance(value, (bool, int, float, complex)):
            highest_type = prims.get_higher_type(highest_type, type(value))
        else:
            raise TypeError(
                f"Cannot promote {type(value)}. Pass IR tensors or Python numeric scalars."
            )

    dimensioned = None
    scalar = None
    for value, dtype in tensors:
        category = prims.dtype_to_type(dtype)
        if highest_type is complex and category in (float, complex):
            if category is float:
                dtype = prims.corresponding_complex_dtype(dtype)
        elif category is not highest_type:
            continue
        if value.shape is None:
            # Rank cannot affect the answer when all candidates have one dtype.
            if len({dtype for _, dtype in tensors}) > 1:
                raise ValueError(
                    "Promotion of mixed tensor dtypes requires known ranks. "
                    "Set ir.Value.shape (symbolic dimensions are supported)."
                )
            dimensioned = prims.get_higher_dtype(dimensioned, dtype)
        elif len(value.shape) == int(
            source_dtype(op, value, packed_complex=packed_complex).is_complex
        ):
            scalar = prims.get_higher_dtype(scalar, dtype)
        else:
            dimensioned = prims.get_higher_dtype(dimensioned, dtype)
    dtype = (
        dimensioned
        or scalar
        or {
            bool: torch.bool,
            int: torch.int64,
            float: torch.get_default_dtype(),
            complex: prims.corresponding_complex_dtype(torch.get_default_dtype()),
        }[highest_type]
    )
    promotion = prims.ELEMENTWISE_TYPE_PROMOTION_KIND[kind]
    if promotion is prims.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT:
        if prims.is_integer_dtype(dtype) or dtype is torch.bool:
            dtype = torch.get_default_dtype()
    if promotion is prims.ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG:
        if dtype is torch.bool:
            dtype = torch.int64
    computation = prims.get_computation_dtype(dtype)
    if promotion is prims.ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH:
        computation = dtype
    if promotion is prims.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
        return computation, torch.bool
    if promotion is prims.ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
        if dtype.is_complex:
            dtype = prims.corresponding_real_dtype(dtype)
    return computation, dtype


def cast(
    op: TranslationContext,
    value: ir.Value | Scalar,
    dtype: torch.dtype,
    *,
    packed_complex: bool = False,
) -> ir.Value:
    """Explicitly cast a logical tensor or wrapped scalar, preserving complex packing."""
    component = prims.corresponding_real_dtype(dtype) if dtype.is_complex else dtype
    onnx_dtype = _ONNX_DTYPES[component]
    if not isinstance(value, ir.Value):
        data = [value.real, value.imag] if isinstance(value, complex) else [value, 0]
        result = op.builder.constant(
            data if dtype.is_complex else value, dtype=onnx_dtype
        )
    else:
        original = source_dtype(op, value, packed_complex=packed_complex)
        if original == dtype and value.dtype == onnx_dtype:
            value.meta[_SOURCE_TYPE] = dtype
            return value
        result = value if value.dtype == onnx_dtype else op.Cast(value, to=onnx_dtype)
        result.shape = value.shape
        if dtype.is_complex and not original.is_complex:
            real = op.Unsqueeze(result, [-1])
            zero = op.builder.constant(0, dtype=onnx_dtype)
            result = op.Concat(real, op.Expand(zero, op.Shape(real)), axis=-1)
            if value.shape is not None:
                result.shape = ir.Shape([*value.shape, 2])
        elif original.is_complex and not dtype.is_complex:
            result = op.Squeeze(op.Slice(result, [0], [1], [-1]), [-1])
            if value.shape is not None:
                result.shape = ir.Shape(list(value.shape)[:-1])
    result.meta[_SOURCE_TYPE] = dtype
    return result


def promote(
    op: TranslationContext,
    *values: ir.Value | Scalar,
    kind: str = "DEFAULT",
    packed_complex: bool = False,
    opmath: bool = False,
) -> tuple[ir.Value, ...]:
    """Promote participating operands; operator lowerings own accumulation casts."""
    computation, dtype = dtypes(op, *values, kind=kind, packed_complex=packed_complex)
    if opmath:
        dtype = computation
    elif kind in ("ALWAYS_BOOL", "COMPLEX_TO_FLOAT"):
        dtype = dtypes(op, *values, packed_complex=packed_complex)[1]
    return tuple(
        cast(op, value, dtype, packed_complex=packed_complex) for value in values
    )


def promote_sequence(
    op: TranslationContext,
    values: Sequence[ir.Value] | ir.Value,
    *,
    packed_complex: bool = False,
) -> list[ir.Value] | ir.Value:
    if isinstance(values, ir.Value):
        if not isinstance(values.type, ir.SequenceType):
            raise TypeError(
                f"Expected an ONNX tensor sequence, got {values.type}. "
                "Pass a Python list of tensor values or a homogeneous SequenceType value."
            )
        return values
    dtype = dtypes(op, *values, kind="NO_OPMATH", packed_complex=packed_complex)[1]
    return [cast(op, value, dtype, packed_complex=packed_complex) for value in values]


def reduction_dtypes(
    op: TranslationContext,
    value: ir.Value,
    dtype: int | None = -1,
    *,
    integral_to_long: bool = False,
) -> tuple[ir.DataType, ir.DataType]:
    """Choose reduction accumulation and output dtypes independently."""
    result = (
        source_dtype(op, value) if dtype in (None, -1) else _DTYPES[ir.DataType(dtype)]
    )
    if dtype in (None, -1) and integral_to_long:
        if result is torch.bool or prims.is_integer_dtype(result):
            result = torch.int64
    computation = (
        torch.int64 if result is torch.bool else prims.get_computation_dtype(result)
    )
    return _ONNX_DTYPES[computation], _ONNX_DTYPES[result]


def reduction_result(
    op: TranslationContext, value: ir.Value, dtype: ir.DataType
) -> ir.Value:
    return value if value.dtype == dtype else op.Cast(value, to=dtype)


def boolean_reduction(
    op: TranslationContext, value: ir.Value, source: ir.Value
) -> ir.Value:
    dtype = ir.DataType.UINT8 if source.dtype == ir.DataType.UINT8 else ir.DataType.BOOL
    return reduction_result(op, value, dtype)
