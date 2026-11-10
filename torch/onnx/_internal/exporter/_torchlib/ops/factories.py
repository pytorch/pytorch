# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR factories translations."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import common as common_ops


if TYPE_CHECKING:
    from collections.abc import Sequence


aten = torch.ops.aten


@functools.cache
def _int64_arange_non_integral_uses_float_length() -> bool:
    """Whether torch computes the arange length in float when dtype is integral.

    Newer versions of torch return 4 elements for ``torch.arange(3.1, dtype=torch.int64)``,
    while older versions return 3. This is evaluated lazily to avoid running torch code
    at import time.
    """
    return torch.arange(3.1, dtype=torch.int64, device="cpu").numel() == 4


def _range_supported(dtype: int) -> bool:
    """Returns true if the dtype is supported by the ONNX Range op."""
    return dtype in {
        ir.DataType.DOUBLE,
        ir.DataType.FLOAT,
        ir.DataType.INT16,
        ir.DataType.INT32,
        ir.DataType.INT64,
    }


def _is_integral_dtype(dtype: int) -> bool:
    return dtype in {
        ir.DataType.INT8,
        ir.DataType.INT16,
        ir.DataType.INT32,
        ir.DataType.INT64,
    }


def _is_integral_scalar(arg: ir.Value | int | float) -> bool:
    if isinstance(arg, int):
        return True
    if isinstance(arg, float):
        return False
    return arg.dtype in {
        ir.DataType.INT8,
        ir.DataType.INT16,
        ir.DataType.INT32,
        ir.DataType.INT64,
    }


def _arange_integral_dtype_with_non_integral_args(
    start: ir.Value,
    end: ir.Value,
    step: ir.Value,
    dtype: int,
) -> ir.Value:
    """Implements torch.arange for integral dtypes when not all inputs are integral."""
    builder = current_builder()
    if dtype is None:
        dtype = -1
    start_float = builder.call("Cast", start, to=ir.DataType.FLOAT)[0]
    end_float = builder.call("Cast", end, to=ir.DataType.FLOAT)[0]
    step_float = builder.call("Cast", step, to=ir.DataType.FLOAT)[0]

    length = builder.call(
        "Cast",
        builder.call(
            "Ceil",
            builder.call(
                "Div", builder.call("Sub", end_float, start_float)[0], step_float
            )[0],
        )[0],
        to=ir.DataType.INT64,
    )[0]
    index = builder.call(
        "Range",
        builder.constant(0, dtype=ir.DataType.INT64),
        length,
        builder.constant(1, dtype=ir.DataType.INT64),
    )[0]

    if _range_supported(dtype):
        index = builder.call("Cast", index, to=dtype)[0]
        start = builder.call("Cast", start, to=dtype)[0]
        step = builder.call("Cast", step, to=dtype)[0]
        return builder.call("Add", start, builder.call("Mul", step, index)[0])[0]

    start = builder.call("Cast", start, to=ir.DataType.INT64)[0]
    step = builder.call("Cast", step, to=ir.DataType.INT64)[0]
    result = builder.call("Add", start, builder.call("Mul", step, index)[0])[0]
    return builder.call("Cast", result, to=dtype)[0]


def _integral_to_be_adjusted(dtype: int) -> bool:
    """Returns true if the dtype is special integral handled by torch."""
    return dtype in {
        ir.DataType.INT8,
        ir.DataType.INT16,
        ir.DataType.INT32,
    }


def _adjust_args_for_arange_int_dtype(
    start: ir.Value,
    end: ir.Value,
    step: ir.Value,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    builder = current_builder()
    zero = builder.call("Cast", 0.0, to=ir.DataType.FLOAT)[0]
    start = builder.call("Cast", start, to=ir.DataType.FLOAT)[0]
    end = builder.call("Cast", end, to=ir.DataType.FLOAT)[0]
    step = builder.call("Cast", step, to=ir.DataType.FLOAT)[0]

    start = builder.call(
        "Where",
        builder.call("Less", start, zero)[0],
        builder.call("Ceil", start)[0],
        start,
    )[0]
    start = builder.call(
        "Where",
        builder.call("Less", step, zero)[0],
        builder.call("Floor", start)[0],
        start,
    )[0]

    return (start, end, step)


@onnx_impl(aten.arange.default)
def aten_arange(
    end: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1
    if dtype == ir.DataType.DOUBLE and not isinstance(end, ir.Value):
        end = builder.constant(end, dtype=ir.DataType.DOUBLE)

    if dtype == -1:
        dtype = ir.DataType.INT64 if _is_integral_scalar(end) else ir.DataType.FLOAT
    if (
        _is_integral_dtype(dtype)
        and not _is_integral_scalar(end)
        and (
            dtype != ir.DataType.INT64 or _int64_arange_non_integral_uses_float_length()
        )
    ):
        result = _arange_integral_dtype_with_non_integral_args(0, end, 1, dtype)
    elif _range_supported(dtype):
        end = builder.call("Cast", end, to=dtype)[0]
        zero = builder.call("Cast", 0, to=dtype)[0]
        one = builder.call("Cast", 1, to=dtype)[0]
        result = builder.call("Range", zero, end, one)[0]
    else:
        # Cast input to float if dtype is not supported by Range,
        # because the input dtype may be e.g. bfloat16 / int8 etc.
        # which Range does not support. The output type is ensured because the output
        # is casted to the specified dtype.
        end = builder.call("Cast", end, to=ir.DataType.FLOAT)[0]
        zero = builder.constant(0.0, dtype=ir.DataType.FLOAT)
        one = builder.constant(1.0, dtype=ir.DataType.FLOAT)
        result = builder.call(
            "Cast", builder.call("Range", zero, end, one)[0], to=dtype
        )[0]

    return result


@onnx_impl(aten.arange.start)
def aten_arange_start(
    start: ir.Value,
    end: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1
    if dtype == ir.DataType.DOUBLE:
        start = (
            start
            if isinstance(start, ir.Value)
            else builder.constant(start, dtype=ir.DataType.DOUBLE)
        )
        end = (
            end
            if isinstance(end, ir.Value)
            else builder.constant(end, dtype=ir.DataType.DOUBLE)
        )

    if dtype == -1:
        dtype = (
            ir.DataType.INT64
            if _is_integral_scalar(start) and _is_integral_scalar(end)
            else ir.DataType.FLOAT
        )
    if (
        _is_integral_dtype(dtype)
        and not (_is_integral_scalar(start) and _is_integral_scalar(end))
        and (
            dtype != ir.DataType.INT64 or _int64_arange_non_integral_uses_float_length()
        )
    ):
        result = _arange_integral_dtype_with_non_integral_args(start, end, 1, dtype)
    elif _range_supported(dtype):
        end = builder.call("Cast", end, to=dtype)[0]
        start = builder.call("Cast", start, to=dtype)[0]
        one = builder.call("Cast", 1, to=dtype)[0]
        result = builder.call("Range", start, end, one)[0]
    else:
        # Cast input to float if dtype is not supported by Range,
        # because the input dtype may be e.g. bfloat16 / int8 etc.
        # which Range does not support. The output type is ensured because the output
        # is casted to the specified dtype.
        end = builder.call("Cast", end, to=ir.DataType.FLOAT)[0]
        start = builder.call("Cast", start, to=ir.DataType.FLOAT)[0]
        one = builder.constant(1.0, dtype=ir.DataType.FLOAT)
        result = builder.call(
            "Cast", builder.call("Range", start, end, one)[0], to=dtype
        )[0]

    return result


@onnx_impl(aten.arange.start_step)
def aten_arange_start_step(
    start: ir.Value,
    end: ir.Value,
    step: ir.Value = 1,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1
    if dtype == ir.DataType.DOUBLE:
        start = (
            start
            if isinstance(start, ir.Value)
            else builder.constant(start, dtype=ir.DataType.DOUBLE)
        )
        end = (
            end
            if isinstance(end, ir.Value)
            else builder.constant(end, dtype=ir.DataType.DOUBLE)
        )
        step = (
            step
            if isinstance(step, ir.Value)
            else builder.constant(step, dtype=ir.DataType.DOUBLE)
        )

    if dtype == -1:
        # Scalars remain Python values until an ONNX node consumes them;
        # dynamic inputs are already IR tensors here.
        # https://github.com/microsoft/onnxscript/issues/1914
        if isinstance(start, (int, float)):
            start_is_int = isinstance(start, int)
        else:
            start_is_int = start.dtype in {
                ir.DataType.INT16,
                ir.DataType.INT32,
                ir.DataType.INT64,
            }
        if isinstance(end, (int, float)):
            end_is_int = isinstance(end, int)
        else:
            end_is_int = end.dtype in {
                ir.DataType.INT16,
                ir.DataType.INT32,
                ir.DataType.INT64,
            }
        if isinstance(step, (int, float)):
            step_is_int = isinstance(step, int)
        else:
            step_is_int = step.dtype in {
                ir.DataType.INT16,
                ir.DataType.INT32,
                ir.DataType.INT64,
            }
        if start_is_int and end_is_int and step_is_int:
            start = builder.call("Cast", start, to=ir.DataType.INT64)[0]
            end = builder.call("Cast", end, to=ir.DataType.INT64)[0]
            step = builder.call("Cast", step, to=ir.DataType.INT64)[0]
            result = builder.call("Range", start, end, step)[0]
        else:
            # to float
            start = builder.call("Cast", start, to=ir.DataType.FLOAT)[0]
            end = builder.call("Cast", end, to=ir.DataType.FLOAT)[0]
            step = builder.call("Cast", step, to=ir.DataType.FLOAT)[0]
            result = builder.call("Range", start, end, step)[0]
    elif (
        _is_integral_dtype(dtype)
        and not (
            _is_integral_scalar(start)
            and _is_integral_scalar(end)
            and _is_integral_scalar(step)
        )
        and (
            dtype != ir.DataType.INT64 or _int64_arange_non_integral_uses_float_length()
        )
    ):
        result = _arange_integral_dtype_with_non_integral_args(start, end, step, dtype)
    elif dtype in (ir.DataType.INT64, ir.DataType.DOUBLE):
        end = builder.call("Cast", end, to=dtype)[0]
        start = builder.call("Cast", start, to=dtype)[0]
        step = builder.call("Cast", step, to=dtype)[0]
        result = builder.call("Range", start, end, step)[0]
    elif _integral_to_be_adjusted(dtype):
        # PyTorch arange op handles these integral types differently from INT64
        # when all arguments are integral.
        start, end, step = _adjust_args_for_arange_int_dtype(start, end, step)
        result = builder.call(
            "Cast", builder.call("Range", start, end, step)[0], to=dtype
        )[0]
    else:
        # Cast input to float if dtype is not supported by Range,
        # because the input dtype may be e.g. bfloat16,
        # which Range does not support. The output type is ensured because the output
        # is casted to the specified dtype.
        end = builder.call("Cast", end, to=ir.DataType.FLOAT)[0]
        start = builder.call("Cast", start, to=ir.DataType.FLOAT)[0]
        step = builder.call("Cast", step, to=ir.DataType.FLOAT)[0]
        result = builder.call(
            "Cast", builder.call("Range", start, end, step)[0], to=dtype
        )[0]

    return result


@onnx_impl(aten.linspace.default)
def aten_linspace(
    start: ir.Value,
    end: ir.Value,
    steps: int,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """linspace(Scalar start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1

    if dtype == -1 or dtype is None:
        dtype = ir.DataType.FLOAT
    if dtype == ir.DataType.DOUBLE:
        start = (
            start
            if isinstance(start, ir.Value)
            else builder.constant(start, dtype=ir.DataType.DOUBLE)
        )
        end = (
            end
            if isinstance(end, ir.Value)
            else builder.constant(end, dtype=ir.DataType.DOUBLE)
        )

    if isinstance(steps, int) and steps == 0:
        return aten_full(
            builder.constant([0], dtype=ir.DataType.INT64), 0.0, dtype=dtype
        )
    if isinstance(steps, int) and steps == 1:
        return aten_full(
            builder.constant([steps], dtype=ir.DataType.INT64), start, dtype=dtype
        )

    # For integer output dtypes, cast start/end to the target dtype first
    # This matches PyTorch's behavior where fractional start/end values
    # are truncated before computing the linspace
    dtype = ir.DataType(dtype)
    if dtype.is_integer():
        # Use double precision for computation to match PyTorch's internal precision
        compute_dtype = ir.DataType.DOUBLE
        # Cast to integer dtype first (truncation), then to compute dtype
        start_int = builder.call("Cast", start, to=dtype)[0]  # Truncate to int32/int64
        end_int = builder.call("Cast", end, to=dtype)[0]
        start_f = builder.call("Cast", start_int, to=compute_dtype)[0]  # Then to double
        end_f = builder.call("Cast", end_int, to=compute_dtype)[0]
    else:
        compute_dtype = dtype
        start_f = builder.call("Cast", start, to=compute_dtype)[0]
        end_f = builder.call("Cast", end, to=compute_dtype)[0]

    rg = aten_arange_start(0, steps, dtype=compute_dtype)
    steps_f = builder.call("Cast", steps, to=compute_dtype)[0]
    one = builder.call("Constant", value=ir.tensor(1, dtype=compute_dtype))[0]
    two = builder.call("Constant", value=ir.tensor(2, dtype=compute_dtype))[0]
    steps_minus_1 = builder.call("Sub", steps_f, one)[0]
    denominator = builder.call("Max", steps_minus_1, one)[0]
    step = builder.call("Div", builder.call("Sub", end_f, start_f)[0], denominator)[0]

    # Two-sided computation for numerical stability at endpoints
    # Use forward computation for first half, backward for second half
    lin_vals = builder.call(
        "Where",
        builder.call("Less", rg, builder.call("Div", steps_f, two)[0])[0],
        builder.call("Add", start_f, builder.call("Mul", step, rg)[0])[0],
        builder.call(
            "Sub",
            end_f,
            builder.call("Mul", step, builder.call("Sub", steps_minus_1, rg)[0])[0],
        )[0],
    )[0]

    return builder.call("Cast", lin_vals, to=dtype)[0]


@onnx_impl(aten.empty.memory_format)
def aten_empty(
    size: Sequence[ir.Value],
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """empty(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    builder = current_builder()
    if dtype is None or dtype == -1:
        dtype = ir.DataType.FLOAT
    size = common_ops.merge_dims(size)
    if dtype == -1:
        dtype = ir.DataType.FLOAT

    # using Zeros to simulate empty()
    zero = builder.call("Constant", value=ir.tensor(0, dtype=ir.DataType(dtype)))[0]
    size = common_ops.merge_dims(size)

    return builder.call("Expand", zero, size)[0]


@onnx_impl(aten.empty_like.default)
def aten_empty_like(
    self: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1

    if dtype == -1 or dtype is None:
        zero = builder.call("CastLike", 0, self)[0]
    else:
        zero = builder.call("Cast", 0, to=dtype)[0]

    shape = builder.call("Shape", self)[0]
    return builder.call("Expand", zero, shape)[0]


@onnx_impl(aten.empty_strided.default)
def aten_empty_strided(
    size: Sequence[ir.Value],
    stride: Sequence[ir.Value],
    layout: str = "",
    dtype: int = ir.DataType.FLOAT,
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    # empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    # using Zeros to simulate empty()

    builder = current_builder()
    if dtype is None or dtype == -1:
        dtype = ir.DataType.FLOAT
    size = common_ops.merge_dims(size)
    zero = builder.call("Constant", value=ir.tensor(0, dtype=ir.DataType(dtype)))[0]
    size = common_ops.merge_dims(size)

    return builder.call("Expand", zero, size)[0]


@onnx_impl((aten.fill.Tensor, aten.fill.Scalar))
def aten_fill(self: ir.Value, value: ir.Value) -> ir.Value:
    """fill.Tensor(Tensor self, Tensor value) -> Tensor"""
    builder = current_builder()

    # Cast the value before Expand so it can be constant folded
    value = builder.call("CastLike", value, self)[0]
    shape = builder.call("Shape", self)[0]
    return builder.call("Expand", value, shape)[0]


@onnx_impl(aten.full.default)
def aten_full(
    size: ir.Value,
    fill_value: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1
    size = common_ops.merge_dims(size)

    if dtype != -1:
        fill_value = builder.call("Cast", fill_value, to=dtype)[0]

    size = builder.call("Cast", size, to=ir.DataType.INT64)[0]
    return builder.call("Expand", fill_value, size)[0]


@onnx_impl(aten.full_like.default)
def aten_full_like(
    self: ir.Value,
    fill_value: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1

    if dtype == -1:
        fill_value = builder.call("CastLike", fill_value, self)[0]
    else:
        fill_value = builder.call("Cast", fill_value, to=dtype)[0]

    self_shape = builder.call("Shape", self)[0]
    return builder.call("Expand", fill_value, self_shape)[0]


@onnx_impl(aten.new_empty.default)
def aten_new_empty(
    self: ir.Value,
    size: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """new_empty(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1
    size = common_ops.merge_dims(size)

    # using zero to simulate empty array
    result = builder.call("ConstantOfShape", size)[0]
    if dtype == -1:
        return builder.call("CastLike", result, self)[0]
    return builder.call("Cast", result, to=dtype)[0]


@onnx_impl(aten.new_empty_strided.default)
def aten_new_empty_strided(
    self: ir.Value,
    size: Sequence[ir.Value],
    stride: Sequence[ir.Value],
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """new_empty_strided(Tensor self, SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1
    size = common_ops.merge_dims(size)

    # using zero to simulate empty array
    size = common_ops.merge_dims(size)
    zero = builder.call("ConstantOfShape", size)[0]
    if dtype == -1:
        return builder.call("CastLike", zero, self)[0]
    return builder.call("Cast", zero, to=dtype)[0]


@onnx_impl(aten.new_full.default)
def aten_new_full(
    self: ir.Value,
    size: ir.Value,
    fill_value: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    # new_full(Tensor self, SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    builder = current_builder()
    if dtype is None:
        dtype = -1
    size = common_ops.merge_dims(size)
    if dtype == -1:
        fill_value = builder.call("CastLike", fill_value, self)[0]
    else:
        fill_value = builder.call("Cast", fill_value, to=dtype)[0]
    return builder.call("Expand", fill_value, size)[0]


@onnx_impl(aten.new_ones.default)
def aten_new_ones(
    self: ir.Value,
    size: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """new_ones(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1
    size = common_ops.merge_dims(size)

    one = builder.constant(1.0, dtype=ir.DataType.FLOAT)
    result = builder.call("Expand", one, size)[0]
    if dtype == -1:
        return builder.call("CastLike", result, self)[0]
    return builder.call("Cast", result, to=dtype)[0]


@onnx_impl(aten.new_zeros.default)
def aten_new_zeros(
    self: ir.Value,
    size: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1
    size = common_ops.merge_dims(size)

    result = builder.call("ConstantOfShape", size)[0]
    if dtype == -1:
        return builder.call("CastLike", result, self)[0]
    return builder.call("Cast", result, to=dtype)[0]


@onnx_impl(aten.ones.default)
def aten_ones(
    size: ir.Value,
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
):
    """ones(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = ir.DataType.FLOAT
    size = common_ops.merge_dims(size)
    if dtype == -1:
        dtype = ir.DataType.FLOAT
    size = builder.call("Cast", size, to=ir.DataType.INT64)[0]
    one = builder.constant(1.0, dtype=ir.DataType.FLOAT)
    one = builder.call("Cast", one, to=dtype)[0]
    return builder.call("Expand", one, size)[0]


@onnx_impl(aten.ones_like.default)
def aten_ones_like(
    self: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    Note: dtype is an onnx enum. Users should convert torch dtype to onnx dtype
    before calling this function.
    """
    builder = current_builder()
    if dtype is None:
        dtype = -1
    if dtype is None:
        dtype = -1

    if dtype == -1:
        one = builder.call("CastLike", 1, self)[0]
    else:
        one = builder.call("Cast", 1, to=dtype)[0]
    shape = builder.call("Shape", self)[0]
    return builder.call("Expand", one, shape)[0]


@onnx_impl(aten.zeros.default)
def aten_zeros(
    size: Sequence[ir.Value],
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = ir.DataType.FLOAT
    size = common_ops.merge_dims(size)
    if dtype == -1:
        dtype = ir.DataType.FLOAT

    zero = builder.call("Constant", value=ir.tensor(0, dtype=ir.DataType(dtype)))[0]
    size = common_ops.merge_dims(size)

    return builder.call("Expand", zero, size)[0]


@onnx_impl(aten.zeros_like.default)
def aten_zeros_like(
    self: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1

    # Each branch explicitly casts the fill value to the requested output type.
    if dtype is None:
        dtype = -1

    if dtype == -1:
        zero = builder.call("CastLike", 0, self)[0]
    else:
        zero = builder.call("Cast", 0, to=dtype)[0]

    shape = builder.call("Shape", self)[0]
    return builder.call("Expand", zero, shape)[0]


@onnx_impl(aten.tensor.bool)
def aten_tensor_bool(self: bool, dtype: int) -> ir.Value:
    builder = current_builder()
    if dtype is None:
        dtype = ir.DataType.BOOL
    tensor = builder.constant(self, dtype=ir.DataType.INT64)
    return builder.call("Cast", tensor, to=dtype)[0]


@onnx_impl(aten.tensor.float)
def aten_tensor_float(self: float, dtype: int) -> ir.Value:
    builder = current_builder()
    if dtype is None:
        dtype = ir.DataType.FLOAT
    tensor = builder.constant(self, dtype=ir.DataType.FLOAT)
    return builder.call("Cast", tensor, to=dtype)[0]


@onnx_impl(aten.tensor.int)
def aten_tensor_int(self: int, dtype: int) -> ir.Value:
    builder = current_builder()
    if dtype is None:
        dtype = ir.DataType.INT64
    tensor = builder.constant(self, dtype=ir.DataType.INT64)
    return builder.call("Cast", tensor, to=dtype)[0]


@onnx_impl(aten.blackman_window.default)
def aten_blackman_window(
    window_length: int,
    dtype: int = 1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = -1

    if dtype is None or dtype == -1:
        dtype = 1
    return _window("BlackmanWindow", window_length, dtype)


@onnx_impl(aten.hamming_window.default)
def aten_hamming_window(
    window_length: int,
    dtype: int = 1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = -1

    if dtype is None or dtype == -1:
        dtype = 1
    # ONNX uses different alpha/beta values for the Hamming window
    # Whereas PyTorch uses alpha=0.54, beta=0.46, ONNX uses
    # alpha=0.543478, beta=0.456522. This causes a slight difference
    # in the output values, but we still uses the HammingWindow op for performance.
    return _window("HammingWindow", window_length, dtype)


@onnx_impl(aten.hann_window.default)
def aten_hann_window(
    window_length: int,
    dtype: int = 1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = -1

    if dtype is None or dtype == -1:
        dtype = 1
    return _window("HannWindow", window_length, dtype)


def _window(op_type: str, length: int | ir.Value, dtype: int) -> ir.Value:
    builder = current_builder()
    if isinstance(length, int) and length <= 1:
        if length < 0:
            raise ValueError(
                f"{op_type} requires a nonnegative length, got {length}; pass length >= 0."
            )
        return builder.constant([1] * length, dtype=ir.DataType(dtype))
    if isinstance(length, int):
        return builder.call(op_type, length, output_datatype=dtype)[0]
    # ONNX window schemas reject zero; slice a valid window for empty outputs.
    safe_length = builder.call("Max", length, 2)[0]
    window = builder.call(op_type, safe_length, output_datatype=dtype)[0]
    result = builder.call(
        "Slice", window, [0], builder.call("Reshape", length, [1])[0]
    )[0]
    return builder.call("Where", builder.call("Equal", length, 1)[0], 1, result)[0]


@onnx_impl(aten.bernoulli.default)
def aten_bernoulli(self: ir.Value) -> ir.Value:
    """Proximal implementation of aten::bernoulli.default

    Note that due to the limitation of ONNX, we ignore the `generator` argument in
      aten::bernoulli.default(Tensor self, *, Generator? generator=None) -> Tensor
    """
    builder = current_builder()
    return builder.call("Bernoulli", self)[0]


@onnx_impl(aten.bernoulli.p)
def aten_bernoulli_p(self: ir.Value, p: float) -> ir.Value:
    """Proximal implementation of aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None)

    Ignore `generator` due to the limit on ONNX expressiveness.
    """
    builder = current_builder()
    # NOTE: We will lose some precision when input is float64 but that's considered insignificant
    self_float = builder.call("Cast", self, to=ir.DataType.FLOAT)[0]
    rands = builder.call("RandomUniformLike", self_float, high=1.0, low=0.0)[0]
    sampled = builder.call("Less", rands, p)[0]
    return builder.call("CastLike", sampled, self)[0]


@onnx_impl(aten.normal_functional.default)
def aten_normal(
    self: ir.Value,
    mean: float = 0.0,
    std: float = 1.0,
) -> ir.Value:
    """normal_functional(Tensor self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor"""
    builder = current_builder()

    result = builder.call("RandomNormalLike", self, mean=mean, scale=std)[0]
    return result


@onnx_impl(aten.normal.float_float)
def aten_normal_float_float(
    mean: float,
    std: float,
    size: ir.Value,
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """normal.float_float(float mean, float std, SymInt[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = ir.DataType.FLOAT
    size = common_ops.merge_dims(size)

    if dtype == -1:
        dtype = ir.DataType.FLOAT
    # Create a dummy tensor for RandomNormalLike to get the shape
    dummy_tensor = builder.call("ConstantOfShape", size)[0]
    result = builder.call("RandomNormalLike", dummy_tensor, mean=mean, scale=std)[0]
    return builder.call("Cast", result, to=dtype)[0]


@onnx_impl(aten.normal.float_Tensor)
def aten_normal_float_tensor(mean: ir.Value, std: ir.Value) -> ir.Value:
    """normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor"""
    builder = current_builder()

    mean_casted = builder.call("CastLike", mean, std)[0]
    sampled = builder.call("RandomNormalLike", std, mean=0.0, scale=1.0)[0]
    # Transform the distribution to the mean and std
    return builder.call("Add", builder.call("Mul", std, sampled)[0], mean_casted)[0]


@onnx_impl(aten.normal.Tensor_float)
def aten_normal_tensor_float(mean: ir.Value, std: ir.Value) -> ir.Value:
    """normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor"""
    builder = current_builder()

    sampled = builder.call("RandomNormalLike", mean, mean=0.0, scale=1.0)[0]
    # Transform the distribution to the mean and std
    return builder.call(
        "Add",
        builder.call("Mul", builder.call("CastLike", std, sampled)[0], sampled)[0],
        mean,
    )[0]


@onnx_impl(aten.normal.Tensor_Tensor)
def aten_normal_tensor_tensor(mean: ir.Value, std: ir.Value) -> ir.Value:
    """normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor"""
    builder = current_builder()

    sampled = builder.call("RandomNormalLike", mean, mean=0.0, scale=1.0)[0]
    # Transform the distribution to the mean and std
    return builder.call("Add", builder.call("Mul", std, sampled)[0], mean)[0]


@onnx_impl(aten.rand.default)
def aten_rand(
    size: ir.Value,
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None or dtype == -1:
        dtype = ir.DataType.FLOAT
    size = common_ops.merge_dims(size)
    if dtype == -1:
        dtype = ir.DataType.FLOAT
    shaper = builder.call("ConstantOfShape", size)[0]
    return builder.call("RandomUniformLike", shaper, dtype=dtype)[0]


@onnx_impl(aten.rand_like.default)
def aten_rand_like(
    self: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1

    if dtype == -1:
        return builder.call("RandomUniformLike", self)[0]
    return builder.call("RandomUniformLike", self, dtype=dtype)[0]


@onnx_impl(aten.randint.default)
def aten_randint(
    high: ir.Value,
    size: ir.Value,
    dtype: int = ir.DataType.INT64,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """randint(SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = ir.DataType.INT64
    size = common_ops.merge_dims(size)

    shaper = builder.call("ConstantOfShape", size)[0]
    rand = builder.call("RandomUniformLike", shaper)[0]
    # Scale to [0, high] first
    rand_scaled = builder.call("Mul", rand, builder.call("CastLike", high, rand)[0])[0]
    # Round to ints
    rand_int = builder.call("Floor", rand_scaled)[0]
    return builder.call("Cast", rand_int, to=dtype)[0]


@onnx_impl(aten.randint.low)
def aten_randint_low(
    low: ir.Value,
    high: ir.Value,
    size: ir.Value,
    dtype: int = ir.DataType.INT64,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """randint.low(SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = ir.DataType.INT64
    size = common_ops.merge_dims(size)

    shaper = builder.call("ConstantOfShape", size)[0]
    rand = builder.call("RandomUniformLike", shaper)[0]
    # Translate to [low, high] first
    high = builder.call("Cast", high, to=ir.DataType.FLOAT)[0]
    low = builder.call("Cast", low, to=ir.DataType.FLOAT)[0]
    rand_translated = builder.call(
        "Add", builder.call("Mul", rand, builder.call("Sub", high, low)[0])[0], low
    )[0]
    # Round to ints
    rand_int = builder.call("Floor", rand_translated)[0]
    return builder.call("Cast", rand_int, to=dtype)[0]


@onnx_impl(aten.randint_like.default)
def aten_randint_like(
    self: ir.Value,
    high: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """randint_like(Tensor self, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1

    self_float = builder.call("Cast", self, to=ir.DataType.FLOAT)[0]
    rand = builder.call("RandomUniformLike", self_float)[0]
    # Scale to [0, high] first
    rand_scaled = builder.call("Mul", rand, builder.call("CastLike", high, rand)[0])[0]
    # Round to ints
    rand_int = builder.call("Floor", rand_scaled)[0]
    if dtype == -1:
        return builder.call("CastLike", rand_int, self)[0]
    return builder.call("Cast", rand_int, to=dtype)[0]


@onnx_impl(aten.randint_like.low_dtype)
def aten_randint_like_low_dtype(
    self: ir.Value,
    low: ir.Value,
    high: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """randint_like.low_dtype(Tensor self, SymInt low, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor

    This is the TorchLib overload for aten::randint_like.low_dtype when dtype is None.
    """
    builder = current_builder()
    if dtype is None:
        dtype = -1

    self_float = builder.call("Cast", self, to=ir.DataType.FLOAT)[0]
    rand = builder.call("RandomUniformLike", self_float)[0]
    # Translate to [low, high] first
    high = builder.call("Cast", high, to=ir.DataType.FLOAT)[0]
    low = builder.call("Cast", low, to=ir.DataType.FLOAT)[0]
    rand_translated = builder.call(
        "Add", builder.call("Mul", rand, builder.call("Sub", high, low)[0])[0], low
    )[0]
    # Round to ints
    rand_int = builder.call("Floor", rand_translated)[0]
    if dtype == -1:
        return builder.call("CastLike", rand_int, self)[0]
    return builder.call("Cast", rand_int, to=dtype)[0]


@onnx_impl(aten.randn.default)
def aten_randn(
    size: ir.Value,
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype is None or dtype == -1:
        dtype = ir.DataType.FLOAT
    size = common_ops.merge_dims(size)

    shaper = builder.call("ConstantOfShape", size)[0]
    return builder.call("RandomNormalLike", shaper, dtype=dtype)[0]


@onnx_impl(aten.randn_like.default)
def aten_randn_like(
    self: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    builder = current_builder()
    if dtype is None:
        dtype = -1

    if dtype == -1:
        return builder.call("RandomNormalLike", self)[0]
    return builder.call("RandomNormalLike", self, dtype=dtype)[0]


@onnx_impl(aten.multinomial.default)
def aten_multinomial(
    self: ir.Value,
    num_samples: int,
    replacement: bool = False,
) -> ir.Value:
    """multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor"""
    builder = current_builder()
    # ONNX Multinomial doesn't support 1D input
    if len(self.shape) == 1:
        unsqueezed_input = builder.call("Unsqueeze", self, [0])[0]
    else:
        unsqueezed_input = self
    # ONNX multinomial expects log probability
    log_input = builder.call("Log", unsqueezed_input)[0]
    result = builder.call(
        "Multinomial", log_input, dtype=ir.DataType.INT64, sample_size=num_samples
    )[0]
    if len(self.shape) == 1:
        result = builder.call("Squeeze", result, [0])[0]
    return result
