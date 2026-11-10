# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned core ONNX IR translations."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence

import numpy as np
import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import common as common_ops


aten = torch.ops.aten
prims = torch.ops.prims


@onnx_impl((aten.abs.default, operator.abs))
def aten_abs(self: ir.Value) -> ir.Value:
    """abs(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Abs", self)[0]


@onnx_impl(aten.abs.default, complex=True)
def aten_abs_complex(self: ir.Value) -> ir.Value:
    """abs(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("ReduceL2", self, [-1], keepdims=False)[0]


@onnx_impl(aten._local_scalar_dense.default)
def aten__local_scalar_dense(self: ir.Value) -> ir.Value:
    """_local_scalar_dense(Tensor self) -> Scalar"""
    builder = current_builder()

    # Return the first element in tensor as a scalar.
    if self.dtype.is_floating_point():
        dtype = ir.DataType.FLOAT
    elif self.dtype == ir.DataType.BOOL:
        dtype = ir.DataType.BOOL
    else:
        dtype = ir.DataType.INT64
    return builder.call(
        "Cast",
        builder.call("Gather", builder.call("Reshape", self, [-1])[0], 0)[0],
        to=dtype,
    )[0]


@onnx_impl(aten.acos.default)
def aten_acos(self: ir.Value) -> ir.Value:
    """acos(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Acos", self)[0]


@onnx_impl(aten.acosh.default)
def aten_acosh(self: ir.Value) -> ir.Value:
    """acosh(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Acosh", self)[0]


@onnx_impl(aten.add.Tensor)
def aten_add(self: ir.Value, other: ir.Value, alpha: float = 1.0) -> ir.Value:
    """add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    builder = current_builder()

    if self.dtype == ir.DataType.BOOL:
        # alpha can also be bool
        if alpha == 0:
            return builder.call("Identity", self)[0]
        return builder.call("Or", self, other)[0]

    if alpha != 1.0:
        alpha = builder.call("CastLike", alpha, other)[0]
        other = builder.call("Mul", other, alpha)[0]
    return builder.call("Add", self, other)[0]


@onnx_impl(aten.add.Scalar)
def aten_add_scalar(self: ir.Value, other: float, alpha: float = 1.0) -> ir.Value:
    """add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor"""
    builder = current_builder()

    other = builder.constant(other, dtype=self.dtype)
    return aten_add(self, other, alpha=alpha)


@onnx_impl(operator.add)
def operator_add(self: ir.Value, other: ir.Value) -> ir.Value:
    builder = current_builder()
    return builder.call("Add", self, other)[0]


@onnx_impl((aten.add.Tensor, aten.add.Scalar), complex=True)
def aten_add_complex(
    self: ir.Value, other: ir.Value | complex | float | int | bool, alpha: float = 1.0
) -> ir.Value:
    """add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    other = common_ops.complex_scalar(other, dtype=self.dtype)
    return aten_add(self, other, alpha=alpha)


@onnx_impl(aten.addcdiv.default)
def aten_addcdiv(
    self: ir.Value, tensor1: ir.Value, tensor2: ir.Value, value: float = 1.0
) -> ir.Value:
    """addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor

    Performs the element-wise division of tensor1 by tensor2, multiplies the result
    by the scalar value and adds it to self.
    """
    builder = current_builder()

    return builder.call(
        "Add",
        self,
        builder.call("Mul", builder.call("Div", tensor1, tensor2)[0], value)[0],
    )[0]


@onnx_impl(aten.addcmul.default)
def aten_addcmul(
    self: ir.Value,
    tensor1: ir.Value,
    tensor2: ir.Value,
    value: float = 1.0,
) -> ir.Value:
    """addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor

    Performs the element-wise multiplication of tensor1 by tensor2, multiplies the
    result by the scalar value and adds it to self.
    """
    builder = current_builder()

    # Follow the order in https://github.com/pytorch/pytorch/blob/29e3fddb082b5a14262a7246bc62381a55199d45/aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp#L47
    # TODO(#811): Understand fp16 accuracy issue
    return builder.call(
        "Add",
        self,
        builder.call("Mul", builder.call("Mul", value, tensor1)[0], tensor2)[0],
    )[0]


@onnx_impl(aten.alias.default)
def aten_alias(self: ir.Value) -> ir.Value:
    """alias(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    return builder.call("Identity", self)[0]


@onnx_impl(aten.angle.default)
def aten_angle(self: ir.Value) -> ir.Value:
    """angle(Tensor self) -> Tensor"""
    builder = current_builder()

    zero = common_ops.constant(0.0, dtype=self.dtype)
    pi = common_ops.constant(math.pi, dtype=self.dtype)

    result = builder.call("Where", builder.call("Less", self, zero)[0], pi, zero)[0]
    return result


@onnx_impl(aten.angle.default, complex=True)
def aten_angle_complex(self: ir.Value) -> ir.Value:
    """angle(Tensor self) -> Tensor"""
    builder = current_builder()

    self_real = builder.call(
        "Squeeze", builder.call("Slice", self, [0], [1], [-1])[0], [-1]
    )[0]
    self_imag = builder.call(
        "Squeeze", builder.call("Slice", self, [1], [2], [-1])[0], [-1]
    )[0]

    result = _atan2(self_imag, self_real, dtype=self.dtype)
    return result


@onnx_impl(aten.asin.default)
def aten_asin(self: ir.Value) -> ir.Value:
    """asin(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Asin", self)[0]


@onnx_impl(aten.asinh.default)
def aten_asinh(self: ir.Value) -> ir.Value:
    """asinh(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Asinh", self)[0]


@onnx_impl(aten.atan.default)
def aten_atan(self: ir.Value) -> ir.Value:
    """atan(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Atan", self)[0]


@onnx_impl(aten.atan2.default)
def aten_atan2(self: ir.Value, other: ir.Value) -> ir.Value:
    """atan2(Tensor self, Tensor other) -> Tensor"""
    return _atan2(self, other, dtype=self.dtype)


def _atan2(y: ir.Value, x: ir.Value, dtype: ir.DataType) -> ir.Value:
    builder = current_builder()
    slope = builder.call("Div", y, x)[0]
    atan = builder.call("Atan", slope)[0]
    zero = common_ops.constant(0.0, dtype=dtype)
    pi = common_ops.constant(math.pi, dtype=dtype)

    second_third_quadrant = builder.call(
        "Where",
        builder.call("Greater", y, zero)[0],
        builder.call("Add", atan, pi)[0],
        builder.call("Sub", atan, pi)[0],
    )[0]
    result = builder.call(
        "Where", builder.call("Less", x, zero)[0], second_third_quadrant, atan
    )[0]

    # Map NaN to 0 to match PyTorch behavior
    result = builder.call("Where", builder.call("IsNaN", result)[0], zero, result)[0]

    return result


@onnx_impl(aten.atanh.default)
def aten_atanh(self: ir.Value) -> ir.Value:
    """atanh(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Atanh", self)[0]


@onnx_impl((aten.bitwise_and.Tensor, operator.and_))
def aten_bitwise_and(self: ir.Value, other: ir.Value) -> ir.Value:
    """bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if not (self.dtype == other.dtype or self.dtype is None or other.dtype is None):
        raise ValueError(
            f"aten_bitwise_and requires matching tensor dtypes, got {self.dtype} and {other.dtype}. Cast operands to the same dtype before translation."
        )
    dtype = self.dtype if self.dtype is not None else other.dtype
    if dtype is None:
        raise ValueError(
            "aten_bitwise_and requires a known tensor dtype. Set ir.Value.type before translation."
        )

    if dtype.is_integer():
        return builder.call("BitwiseAnd", self, other)[0]
    if dtype == ir.DataType.BOOL:
        return builder.call("And", self, other)[0]
    raise NotImplementedError(
        f"aten_bitwise_and does not support dtype {self.dtype}. Use integer or boolean tensors."
    )


@onnx_impl(aten.bitwise_and.Scalar)
def aten_bitwise_and_scalar(self: ir.Value, other: int) -> ir.Value:
    """bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = current_builder()

    other_tensor = builder.constant(other, dtype=self.dtype)
    return aten_bitwise_and(self, other_tensor)


@onnx_impl(aten.bitwise_and.Scalar_Tensor)
def aten_bitwise_and_scalar_tensor(self: float, other: ir.Value) -> ir.Value:
    """bitwise_and.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = current_builder()

    self_tensor = builder.constant(self, dtype=other.dtype)
    return aten_bitwise_and(self_tensor, other)


@onnx_impl((aten.bitwise_left_shift.Tensor, aten.__lshift__.Tensor, operator.lshift))
def aten_bitwise_left_shift(self: ir.Value, other: ir.Value) -> ir.Value:
    """bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()
    if not (self.dtype == other.dtype or self.dtype is None or other.dtype is None):
        raise ValueError(
            f"aten_bitwise_left_shift requires matching tensor dtypes, got {self.dtype} and {other.dtype}. Cast operands to the same dtype before translation."
        )
    dtype = self.dtype if self.dtype is not None else other.dtype
    if dtype is None:
        raise ValueError(
            "aten_bitwise_left_shift requires a known tensor dtype. Set ir.Value.type before translation."
        )

    if not dtype.is_signed():
        return builder.call("BitShift", self, other, direction="LEFT")[0]

    # assert other >= 0
    if dtype.bitwidth == 8:
        unsigned_dtype = ir.DataType.UINT8
        signed_dtype = ir.DataType.INT8
    elif dtype.bitwidth == 16:
        unsigned_dtype = ir.DataType.UINT16
        signed_dtype = ir.DataType.INT16
    elif dtype.bitwidth == 32:
        unsigned_dtype = ir.DataType.UINT32
        signed_dtype = ir.DataType.INT32
    elif dtype.bitwidth == 64:
        unsigned_dtype = ir.DataType.UINT64
        signed_dtype = ir.DataType.INT64
    else:
        raise NotImplementedError(
            f"aten_bitwise_left_shift does not support dtype {dtype}. Use 8-, 16-, 32-, or 64-bit integer tensors."
        )

    self = builder.call("Cast", self, to=unsigned_dtype)[0]
    other = builder.call("Cast", other, to=unsigned_dtype)[0]

    result = builder.call("BitShift", self, other, direction="LEFT")[0]

    return builder.call("Cast", result, to=signed_dtype)[0]


@onnx_impl((aten.bitwise_left_shift.Tensor_Scalar, aten.__lshift__.Scalar))
def aten_bitwise_left_shift_tensor_scalar(self: ir.Value, other: int) -> ir.Value:
    """bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = current_builder()
    other_tensor = builder.constant(other, dtype=self.dtype)
    return aten_bitwise_left_shift(self, other_tensor)


@onnx_impl(aten.bitwise_left_shift.Scalar_Tensor)
def aten_bitwise_left_shift_scalar_tensor(self: int, other: ir.Value) -> ir.Value:
    """bitwise_left_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = current_builder()
    self_tensor = builder.constant(self, dtype=other.dtype)
    return aten_bitwise_left_shift(self_tensor, other)


@onnx_impl(aten.bitwise_not.default)
def aten_bitwise_not(self: ir.Value) -> ir.Value:
    """bitwise_not(Tensor self) -> Tensor"""
    builder = current_builder()

    if self.dtype == ir.DataType.BOOL:
        return builder.call("Not", self)[0]
    if self.dtype.is_integer():
        return builder.call("BitwiseNot", self)[0]
    raise NotImplementedError(
        f"aten_bitwise_not does not support dtype {self.dtype}. Use integer or boolean tensors."
    )


@onnx_impl((aten.bitwise_or.Tensor, operator.or_))
def aten_bitwise_or(self: ir.Value, other: ir.Value) -> ir.Value:
    """bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if not (self.dtype == other.dtype or self.dtype is None or other.dtype is None):
        raise ValueError(
            f"aten_bitwise_or requires matching tensor dtypes, got {self.dtype} and {other.dtype}. Cast operands to the same dtype before translation."
        )
    dtype = self.dtype if self.dtype is not None else other.dtype
    if dtype is None:
        raise ValueError(
            "aten_bitwise_or requires a known tensor dtype. Set ir.Value.type before translation."
        )

    if dtype.is_integer():
        return builder.call("BitwiseOr", self, other)[0]
    if dtype == ir.DataType.BOOL:
        return builder.call("Or", self, other)[0]
    raise NotImplementedError(
        f"aten_bitwise_or does not support dtype {self.dtype}. Use integer or boolean tensors."
    )


@onnx_impl(aten.bitwise_or.Scalar)
def aten_bitwise_or_scalar(self: ir.Value, other: int) -> ir.Value:
    """bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = current_builder()
    other_tensor = builder.constant(other, dtype=self.dtype)
    return aten_bitwise_or(self, other_tensor)


@onnx_impl(aten.bitwise_or.Scalar_Tensor)
def aten_bitwise_or_scalar_tensor(self: int, other: ir.Value) -> ir.Value:
    """bitwise_or.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = current_builder()
    self_tensor = builder.constant(self, dtype=other.dtype)
    return aten_bitwise_or(self_tensor, other)


@onnx_impl((aten.bitwise_right_shift.Tensor, aten.__rshift__.Tensor, operator.rshift))
def aten_bitwise_right_shift(self: ir.Value, other: ir.Value) -> ir.Value:
    """bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()
    if not (self.dtype == other.dtype or self.dtype is None or other.dtype is None):
        raise ValueError(
            f"aten_bitwise_right_shift requires matching tensor dtypes, got {self.dtype} and {other.dtype}. Cast operands to the same dtype before translation."
        )
    dtype = self.dtype if self.dtype is not None else other.dtype
    if dtype is None:
        raise ValueError(
            "aten_bitwise_right_shift requires a known tensor dtype. Set ir.Value.type before translation."
        )

    if not dtype.is_signed():
        return builder.call("BitShift", self, other, direction="RIGHT")[0]

    if dtype.bitwidth == 8:
        unsigned_dtype = ir.DataType.UINT8
        signed_dtype = ir.DataType.INT8
        mask = builder.constant(0xFF, dtype=unsigned_dtype)
    elif dtype.bitwidth == 16:
        unsigned_dtype = ir.DataType.UINT16
        signed_dtype = ir.DataType.INT16
        mask = builder.constant(0xFFFF, dtype=unsigned_dtype)
    elif dtype.bitwidth == 32:
        unsigned_dtype = ir.DataType.UINT32
        signed_dtype = ir.DataType.INT32
        mask = builder.constant(0xFFFFFFFF, dtype=unsigned_dtype)
    elif dtype.bitwidth == 64:
        unsigned_dtype = ir.DataType.UINT64
        signed_dtype = ir.DataType.INT64
        mask = builder.constant(
            0xFFFFFFFFFFFFFFFF, dtype=unsigned_dtype
        )  # 0xFFFFFFFFFFFFFFFF
    else:
        raise NotImplementedError(
            f"aten_bitwise_right_shift does not support dtype {dtype}. Use 8-, 16-, 32-, or 64-bit integer tensors."
        )

    negative = builder.call("Less", self, 0)[0]
    self = builder.call("Cast", self, to=unsigned_dtype)[0]
    other = builder.call("Cast", other, to=unsigned_dtype)[0]

    # Simulate arithmetic shift using logical shift
    # Clear the lower bits of an all one mask to create the mask to simulate the sign bit shifting
    mask = builder.call("BitShift", mask, other, direction="RIGHT")[0]
    mask = builder.call("BitwiseNot", mask)[0]
    # Do logical shift
    shifted = builder.call("BitShift", self, other, direction="RIGHT")[0]
    # Compute the arithmetic shifted value assuming the sign bit was set
    negative_shifted = builder.call("BitwiseOr", shifted, mask)[0]
    # Choose the shifted value based on the sign bit
    return builder.call(
        "Where",
        negative,
        builder.call("Cast", negative_shifted, to=signed_dtype)[0],
        builder.call("Cast", shifted, to=signed_dtype)[0],
    )[0]


@onnx_impl((aten.bitwise_right_shift.Tensor_Scalar, aten.__rshift__.Scalar))
def aten_bitwise_right_shift_tensor_scalar(self: ir.Value, other: int) -> ir.Value:
    """bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = current_builder()
    other_tensor = builder.constant(other, dtype=self.dtype)
    return aten_bitwise_right_shift(self, other_tensor)


@onnx_impl(aten.bitwise_right_shift.Scalar_Tensor)
def aten_bitwise_right_shift_scalar_tensor(self: int, other: ir.Value) -> ir.Value:
    """bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = current_builder()
    self_tensor = builder.constant(self, dtype=other.dtype)
    return aten_bitwise_right_shift(self_tensor, other)


@onnx_impl(aten.bitwise_xor.Tensor)
def aten_bitwise_xor(self: ir.Value, other: ir.Value) -> ir.Value:
    """bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if not (self.dtype == other.dtype or self.dtype is None or other.dtype is None):
        raise ValueError(
            f"aten_bitwise_xor requires matching tensor dtypes, got {self.dtype} and {other.dtype}. Cast operands to the same dtype before translation."
        )
    dtype = self.dtype if self.dtype is not None else other.dtype
    if dtype is None:
        raise ValueError(
            "aten_bitwise_xor requires a known tensor dtype. Set ir.Value.type before translation."
        )

    if dtype.is_integer():
        return builder.call("BitwiseXor", self, other)[0]
    if dtype == ir.DataType.BOOL:
        return builder.call("Xor", self, other)[0]
    raise NotImplementedError(
        f"aten_bitwise_xor does not support dtype {self.dtype}. Use integer or boolean tensors."
    )


@onnx_impl(aten.bitwise_xor.Scalar)
def aten_bitwise_xor_scalar(self: ir.Value, other: int) -> ir.Value:
    """bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = current_builder()
    other_tensor = builder.constant(other, dtype=self.dtype)
    return aten_bitwise_xor(self, other_tensor)


@onnx_impl(aten.bitwise_xor.Scalar_Tensor)
def aten_bitwise_xor_scalar_tensor(self: int, other: ir.Value) -> ir.Value:
    """bitwise_xor.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = current_builder()
    self_tensor = builder.constant(self, dtype=other.dtype)
    return aten_bitwise_xor(self_tensor, other)


@onnx_impl(aten.ceil.default)
def aten_ceil(self: ir.Value) -> ir.Value:
    """ceil(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Ceil", self)[0]


@onnx_impl(math.ceil)
def python_math_ceil(self: ir.Value) -> ir.Value:
    """ceil(Tensor self) -> Tensor"""
    builder = current_builder()
    ceil = builder.call("Ceil", self)[0]
    return builder.call("Cast", ceil, to=ir.DataType.INT64)[0]


@onnx_impl(aten.clamp.default)
def aten_clamp(
    self: ir.Value, min: float | None = None, max: float | None = None
) -> ir.Value:
    """clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor"""
    builder = current_builder()

    if min is None and max is None:
        return builder.call("Identity", self)[0]

    if min is not None:
        min = builder.call("CastLike", min, self)[0]

    if max is not None:
        max = builder.call("CastLike", max, self)[0]

    return builder.call("Clip", self, min, max)[0]


@onnx_impl(aten.clamp.Tensor)
def aten_clamp_tensor(
    self: ir.Value, min: ir.Value | None = None, max: ir.Value | None = None
) -> ir.Value:
    """clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor"""
    builder = current_builder()

    if min is None and max is None:
        return builder.call("Identity", self)[0]

    clamped = self

    # If min is greater than max torch.clamp(..., min, max)
    # sets all elements in input to the value of max.
    # So this order is important.
    if min is not None:
        min_clamp = builder.call("CastLike", min, self)[0]
        clamped = builder.call("Max", clamped, min_clamp)[0]

    if max is not None:
        max_clamp = builder.call("CastLike", max, self)[0]
        clamped = builder.call("Min", clamped, max_clamp)[0]

    return clamped


@onnx_impl(aten.clamp_max.default)
def aten_clamp_max(self: ir.Value, max_: float) -> ir.Value:
    """clamp_max(Tensor self, Scalar max) -> Tensor"""
    builder = current_builder()

    # This implementation does not intend to handle when self is an empty tensor
    max_ = builder.call("CastLike", max_, self)[0]
    return builder.call("Clip", self, None, max_)[0]


@onnx_impl(aten.clamp_max.Tensor)
def aten_clamp_max_tensor(self: ir.Value, max_: ir.Value) -> ir.Value:
    """clamp_max.Tensor(Tensor self, Tensor max) -> Tensor"""
    builder = current_builder()

    # This implementation does not intend to handle when self is an empty tensor
    max_rank = len(max_.shape)
    if max_rank == 0:
        max_ = builder.call("CastLike", max_, self)[0]
        result = builder.call("Clip", self, None, max_)[0]
    else:
        result = builder.call("Min", self, max_)[0]

    return result


@onnx_impl(aten.clamp_min.default)
def aten_clamp_min(self: ir.Value, min_: float) -> ir.Value:
    """clamp_min(Tensor self, Scalar min) -> Tensor"""
    builder = current_builder()

    # This implementation does not intend to handle when self is an empty tensor
    min_ = builder.call("CastLike", min_, self)[0]
    return builder.call("Clip", self, min_, None)[0]


@onnx_impl(aten.clamp_min.Tensor)
def aten_clamp_min_tensor(self: ir.Value, min_: ir.Value) -> ir.Value:
    """clamp_min.Tensor(Tensor self, Tensor min) -> Tensor"""
    builder = current_builder()

    # This implementation does not intend to handle when self is an empty tensor
    min_rank = len(min_.shape)
    if min_rank == 0:
        min_ = builder.call("CastLike", min_, self)[0]
        result = builder.call("Clip", self, min_, None)[0]
    else:
        result = builder.call("Max", self, min_)[0]

    return result


@onnx_impl(aten.clone.default)
def aten_clone(
    self: ir.Value,
    memory_format: str = "",
) -> ir.Value:
    """clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"""
    builder = current_builder()

    return builder.call("Identity", self)[0]


@onnx_impl(aten.clone.default, complex=True)
def aten_clone_complex(
    self: ir.Value,
    memory_format: str = "",  # pylint: disable=unused-argument
) -> ir.Value:
    """clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"""
    builder = current_builder()
    return builder.call("Identity", self)[0]


@onnx_impl((aten.conj.default, aten._conj.default))
def aten_conj(self: ir.Value) -> ir.Value:
    """conj(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    return builder.call("Identity", self)[0]


def _complex_conjugate(self: ir.Value) -> ir.Value:
    builder = current_builder()
    zero = builder.constant([0], dtype=ir.DataType.INT64)
    one = builder.constant([1], dtype=ir.DataType.INT64)
    two = builder.constant([2], dtype=ir.DataType.INT64)
    neg_1 = builder.constant([-1], dtype=ir.DataType.INT64)
    # The last dimension is the real and imaginary parts

    real = builder.call("Slice", self, zero, one, neg_1)[0]
    imag = builder.call("Slice", self, one, two, neg_1)[0]
    conjugated = builder.call("Concat", real, builder.call("Neg", imag)[0], axis=-1)[0]

    return conjugated


@onnx_impl((aten.conj.default, aten._conj.default), complex=True)
def aten_conj_complex(self: ir.Value) -> ir.Value:
    """conj(Tensor(a) self) -> Tensor(a)"""
    return _complex_conjugate(self)


@onnx_impl(aten._to_copy.default)
def aten__to_copy(
    self: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor"""
    builder = current_builder()

    if dtype == -1:
        return builder.call("Identity", self)[0]
    else:
        return common_ops.cast_to(self, dtype=dtype)


@onnx_impl(aten.cos.default)
def aten_cos(self: ir.Value) -> ir.Value:
    """cos(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Cos", self)[0]


@onnx_impl(aten.cosh.default)
def aten_cosh(self: ir.Value) -> ir.Value:
    """cosh(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Cosh", self)[0]


@onnx_impl(aten.deg2rad.default)
def aten_deg2rad(self: ir.Value) -> ir.Value:
    """deg2rad(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call(
        "Mul", self, builder.call("CastLike", math.pi / 180.0, self)[0]
    )[0]


@onnx_impl(
    (
        aten.div.Tensor,
        aten.div.Scalar,
        aten.divide.Tensor,
        aten.divide.Scalar,
        aten.true_divide.Tensor,
        aten.true_divide.Scalar,
    )
)
def aten_div(self: ir.Value, other: ir.Value) -> ir.Value:
    """div.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    # Int inputs will be promoted to float by PyTorch
    return builder.call("Div", self, other)[0]


@onnx_impl(operator.truediv)
def operator_truediv(self: ir.Value, other: ir.Value) -> ir.Value:
    builder = current_builder()
    return builder.call(
        "Div",
        builder.call("Cast", self, to=ir.DataType.FLOAT)[0],
        builder.call("Cast", other, to=ir.DataType.FLOAT)[0],
    )[0]


@onnx_impl(
    (
        aten.div.Tensor,
        aten.div.Scalar,
        aten.divide.Tensor,
        aten.divide.Scalar,
        aten.true_divide.Tensor,
        aten.true_divide.Scalar,
    ),
    complex=True,
)
def aten_div_complex(
    self: ir.Value, other: ir.Value | complex | float | int | bool
) -> ir.Value:
    """div.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()
    other = common_ops.complex_scalar(other, dtype=self.dtype)

    # Complex division. PyTorch type promotion ensures both arguments are complex numbers
    self_real = builder.call("Slice", self, [0], [1], [-1])[0]
    self_imag = builder.call("Slice", self, [1], [2], [-1])[0]
    other_real = builder.call("Slice", other, [0], [1], [-1])[0]
    other_imag = builder.call("Slice", other, [1], [2], [-1])[0]

    # Complex division
    # (a + bi) / (c + di) = (ac + bd) / (c^2 + d^2) + (bc - ad) / (c^2 + d^2)i
    # https://mathworld.wolfram.com/ComplexDivision.html
    ac = builder.call("Mul", self_real, other_real)[0]
    bd = builder.call("Mul", self_imag, other_imag)[0]
    bc = builder.call("Mul", self_imag, other_real)[0]
    ad = builder.call("Mul", self_real, other_imag)[0]
    denominator = builder.call(
        "Add",
        builder.call("Mul", other_real, other_real)[0],
        builder.call("Mul", other_imag, other_imag)[0],
    )[0]
    real = builder.call("Div", builder.call("Add", ac, bd)[0], denominator)[0]
    imag = builder.call("Div", builder.call("Sub", bc, ad)[0], denominator)[0]

    return builder.call("Concat", real, imag, axis=-1)[0]


@onnx_impl((aten.div.Tensor_mode, aten.div.Scalar_mode))
def aten_div_mode(
    self: ir.Value, other: ir.Value, rounding_mode: str | None = None
) -> ir.Value:
    """div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor"""
    builder = current_builder()

    if rounding_mode not in {"trunc", "floor", None}:
        raise ValueError(
            f"Invalid rounding_mode={rounding_mode!r}. Use None, 'trunc', or 'floor'."
        )

    if self.dtype.is_integer():
        quotient = builder.call(
            "Div",
            builder.call("Cast", self, to=ir.DataType.FLOAT)[0],
            builder.call("Cast", other, to=ir.DataType.FLOAT)[0],
        )[0]

        if rounding_mode == "trunc":
            # Rounds the results of the division towards zero.
            # Equivalent to C-style integer division
            result = aten_trunc(quotient)
            return builder.call("CastLike", result, self)[0]
        if rounding_mode == "floor":
            result = builder.call("Floor", quotient)[0]
            return builder.call("CastLike", result, self)[0]

        if rounding_mode is not None:
            raise ValueError(
                "Expected rounding_mode=None after handling trunc and floor."
            )
        # When rounding_mode is None, the return type is float32
        return quotient

    # Float inputs

    if rounding_mode == "trunc":
        # Rounds the results of the division towards zero.
        # Equivalent to C-style integer division
        quotient = builder.call("Div", self, other)[0]
        if self.dtype == ir.DataType.FLOAT16:
            quotient = builder.call("Cast", quotient, to=ir.DataType.FLOAT16)[0]
        return aten_trunc(quotient)
    if rounding_mode == "floor":
        return builder.call("Floor", builder.call("Div", self, other)[0])[0]

    return builder.call("Div", self, other)[0]


@onnx_impl((aten.eq.default, aten.eq.Tensor, aten.eq.Scalar, operator.eq))
def aten_eq(self: ir.Value, other: ir.Value) -> ir.Value:
    """eq.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("Equal", self, other)[0]


@onnx_impl(aten.exp.default)
def aten_exp(self: ir.Value) -> ir.Value:
    """exp(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Exp", self)[0]


@onnx_impl(aten.exp2.default)
def aten_exp2(self: ir.Value) -> ir.Value:
    """exp2(Tensor self) -> Tensor"""
    builder = current_builder()

    two = builder.constant(2, dtype=ir.DataType.INT64)
    two = builder.call("CastLike", two, self)[0]
    return builder.call("Pow", two, self)[0]


@onnx_impl(aten.floor.default)
def aten_floor(self: ir.Value) -> ir.Value:
    """floor(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Floor", self)[0]


@onnx_impl(math.floor)
def python_math_floor(self: ir.Value) -> ir.Value:
    """floor(Tensor self) -> Tensor"""
    builder = current_builder()
    floor = builder.call("Floor", self)[0]
    return builder.call("Cast", floor, to=ir.DataType.INT64)[0]


@onnx_impl(aten.floor_divide.default)
def aten_floor_divide(self: ir.Value, other: ir.Value) -> ir.Value:
    """floor_divide(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if self.dtype.is_floating_point():
        return builder.call("Floor", builder.call("Div", self, other)[0])[0]

    if not self.dtype.is_integer():
        raise ValueError(
            f"floor_divide requires real numeric tensors, got {self.dtype}. Cast the input to an integer or floating-point dtype."
        )

    if not self.dtype.is_signed():
        return builder.call("Div", self, other)[0]

    # Convert truncation to flooring
    # Reference: https://github.com/pytorch/pytorch/blob/ffc645c870f0abd368606ba1e2b3b58cacb03046/torch/_refs/__init__.py#L1401C1-L1409C70
    # offset = (torch.signbit(a) != torch.signbit(b)).logical_and(torch.fmod(a, b) != 0)
    # return prims.div(a, b) - _maybe_convert_to_dtype(offset, a.dtype)
    # The sign-mismatch check ``signbit(a) != signbit(b)`` (for nonzero divisors)
    # is expressed as ``(a < 0) == (b > 0)``, which avoids Sign (limited integral
    # support on some EPs) and enables optimizations when ``self`` is provably
    # non-negative.
    offset = builder.call(
        "And",
        builder.call(
            "Equal",
            builder.call("Less", self, 0)[0],
            builder.call("Greater", other, 0)[0],
        )[0],
        builder.call("Cast", builder.call("Mod", self, other)[0], to=ir.DataType.BOOL)[
            0
        ],
    )[0]
    offset = builder.call("Cast", offset, to=self.dtype)[0]
    return builder.call("Sub", builder.call("Div", self, other)[0], offset)[0]


@onnx_impl(operator.floordiv)
def operator_floordiv(self: ir.Value, other: ir.Value) -> ir.Value:
    builder = current_builder()
    # We implement floor_divide only for positive inputs (using integer division)
    # because that is the usual intended case and is the most efficient.
    return builder.call("Div", self, other)[0]


@onnx_impl((aten.fmod.Tensor, aten.fmod.Scalar))
def aten_fmod(self: ir.Value, other: ir.Value) -> ir.Value:
    """fmod.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("Mod", self, other, fmod=1)[0]


@onnx_impl(aten.frac.default)
def aten_frac(self: ir.Value) -> ir.Value:
    """frac(Tensor self) -> Tensor

    Computes the fractional portion of each element in input.
    """
    builder = current_builder()

    # https://pytorch.org/docs/stable/generated/torch.frac.html
    return builder.call(
        "Sub",
        self,
        builder.call(
            "Mul",
            builder.call("Floor", builder.call("Abs", self)[0])[0],
            builder.call("Sign", self)[0],
        )[0],
    )[0]


@onnx_impl((aten.ge.Tensor, aten.ge.Scalar, aten.greater_equal.Tensor))
def aten_ge(self: ir.Value, other: ir.Value) -> ir.Value:
    """ge.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if self.dtype == ir.DataType.BOOL:
        # self, other, self >= other
        #    F,    F,    T
        #    F,    T,    F
        #    T,    F,    T
        #    T,    T,    T
        return builder.call("Or", self, builder.call("Not", other)[0])[0]

    return builder.call("GreaterOrEqual", self, other)[0]


@onnx_impl(operator.ge)
def operator_ge(self: ir.Value, other: ir.Value) -> ir.Value:
    builder = current_builder()
    # operator.ge for SymInt
    return builder.call("GreaterOrEqual", self, other)[0]


@onnx_impl((aten.gt.Tensor, aten.gt.Scalar, aten.greater.Tensor))
def aten_gt(self: ir.Value, other: ir.Value) -> ir.Value:
    """gt.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if self.dtype == ir.DataType.BOOL:
        # self, other, self > other
        #    F,    F,    F
        #    F,    T,    F
        #    T,    F,    T
        #    T,    T,    F

        return builder.call("And", self, builder.call("Not", other)[0])[0]

    return builder.call("Greater", self, other)[0]


@onnx_impl(operator.gt)
def operator_gt(self: ir.Value, other: ir.Value) -> ir.Value:
    builder = current_builder()
    # operator.gt for SymInt
    return builder.call("Greater", self, other)[0]


@onnx_impl(aten.isclose.default)
def aten_isclose(
    self: ir.Value,
    other: ir.Value,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> ir.Value:
    """isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor"""
    builder = current_builder()

    # torch builds isclose out of three terms in aten/src/ATen/native/TensorCompare.cpp:
    # exact equality, NaN against NaN when equal_nan is set, and the tolerance band
    # restricted to the elements whose error is finite.
    result = builder.call("Equal", self, other)[0]

    # |self - other| <= atol + rtol x |other|
    actual_error = builder.call("Abs", builder.call("Sub", self, other)[0])[0]
    allowed_error = builder.call(
        "Add",
        builder.call("CastLike", atol, other)[0],
        builder.call(
            "Mul",
            builder.call("CastLike", rtol, other)[0],
            builder.call("Abs", other)[0],
        )[0],
    )[0]

    if self.dtype.is_floating_point():
        # Comparing the two sides of the band by subtraction carries the finiteness
        # restriction on its own, so no IsInf term is needed. An infinity on either side
        # puts a NaN into the subtraction whichever way it arrives: the error and the
        # allowance are both infinite and cancel, or the allowance is already NaN because
        # rtol is zero and zero times infinity is NaN. Every comparison against NaN is
        # false, which leaves the equality term above as the only way an infinity is
        # reported close. On finite values the sign of the difference and the direct
        # comparison agree exactly, so the band is unchanged.
        zero = builder.constant(0, dtype=self.dtype)
        within_tolerance = builder.call(
            "LessOrEqual", builder.call("Sub", actual_error, allowed_error)[0], zero
        )[0]
        if equal_nan:
            result = builder.call(
                "Or",
                result,
                builder.call(
                    "And",
                    builder.call("IsNaN", self)[0],
                    builder.call("IsNaN", other)[0],
                )[0],
            )[0]
    else:
        # Integers are never infinite and never NaN, so they need no finiteness term, and
        # IsNaN does not accept them. The subtraction is left off here because it can
        # overflow a narrow integer type where the direct comparison cannot.
        within_tolerance = builder.call("LessOrEqual", actual_error, allowed_error)[0]

    return builder.call("Or", result, within_tolerance)[0]


@onnx_impl(aten.isfinite.default)
def aten_isfinite(self: ir.Value) -> ir.Value:
    """isfinite(Tensor self) -> Tensor"""
    builder = current_builder()

    # IsInf only supports FLOAT and DOUBLE
    not_inf = builder.call("Not", builder.call("IsInf", self)[0])[0]
    # TODO: The test case doesn't cover this condition
    not_nan = builder.call("Not", builder.call("IsNaN", self)[0])[0]
    return builder.call("And", not_inf, not_nan)[0]


@onnx_impl(aten.isinf.default)
def aten_isinf(self: ir.Value) -> ir.Value:
    """isinf(Tensor self) -> Tensor"""
    builder = current_builder()

    # Added Cast inside the function so it can support all real dtypes naturally
    self = builder.call("Cast", self, to=ir.DataType.FLOAT)[0]
    return builder.call("IsInf", self)[0]


@onnx_impl(aten.isnan.default)
def aten_isnan(self: ir.Value) -> ir.Value:
    """isnan(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("IsNaN", self)[0]


@onnx_impl(aten.isneginf.default)
def aten_isneginf(self: ir.Value) -> ir.Value:
    """isneginf(Tensor self) -> Tensor"""
    builder = current_builder()

    # Added Cast inside the function so it can support all real dtypes naturally
    self = builder.call("Cast", self, to=ir.DataType.FLOAT)[0]
    return builder.call(
        "And", builder.call("Less", self, 0)[0], builder.call("IsInf", self)[0]
    )[0]


@onnx_impl(aten.isposinf.default)
def aten_isposinf(self: ir.Value) -> ir.Value:
    """isposinf(Tensor self) -> Tensor"""
    builder = current_builder()

    # Added Cast inside the function so it can support all real dtypes naturally
    self = builder.call("Cast", self, to=ir.DataType.FLOAT)[0]
    return builder.call(
        "And", builder.call("Greater", self, 0)[0], builder.call("IsInf", self)[0]
    )[0]


@onnx_impl((aten.le.Tensor, aten.le.Scalar, aten.less_equal.Tensor))
def aten_le(self: ir.Value, other: ir.Value) -> ir.Value:
    """le.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if self.dtype == ir.DataType.BOOL:
        # self, other, self <= other
        #    F,    F,    T
        #    F,    T,    T
        #    T,    F,    F
        #    T,    T,    T

        return builder.call("Or", other, builder.call("Not", self)[0])[0]

    return builder.call("LessOrEqual", self, other)[0]


@onnx_impl(operator.le)
def operator_le(self: ir.Value, other: ir.Value) -> ir.Value:
    builder = current_builder()
    # operator.le for SymInt
    return builder.call("LessOrEqual", self, other)[0]


@onnx_impl((aten.lerp.Tensor, aten.lerp.Scalar))
def aten_lerp(self: ir.Value, end: ir.Value, weight: ir.Value) -> ir.Value:
    """lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor"""
    builder = current_builder()

    weight = builder.call("CastLike", weight, self)[0]
    diff = builder.call("Sub", end, self)[0]
    return builder.call(
        "Where",
        builder.call("Less", weight, 0.5)[0],
        builder.call("Add", self, builder.call("Mul", weight, diff)[0])[0],
        builder.call(
            "Sub",
            end,
            builder.call("Mul", diff, builder.call("Sub", 1.0, weight)[0])[0],
        )[0],
    )[0]


@onnx_impl(aten.log.default)
def aten_log(self: ir.Value) -> ir.Value:
    """log(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Log", self)[0]


@onnx_impl(aten.log10.default)
def aten_log10(self: ir.Value) -> ir.Value:
    """log10(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call(
        "Div",
        builder.call("Log", self)[0],
        builder.constant(math.log(10.0), dtype=self.dtype),
    )[0]


@onnx_impl(aten.log1p.default)
def aten_log1p(self: ir.Value) -> ir.Value:
    """log1p(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Log", builder.call("Add", self, 1.0)[0])[0]


@onnx_impl(aten.log2.default)
def aten_log2(self: ir.Value) -> ir.Value:
    """log2(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call(
        "Div",
        builder.call("Log", self)[0],
        builder.constant(math.log(2.0), dtype=self.dtype),
    )[0]


@onnx_impl(aten.logaddexp.default)
def aten_logaddexp(self: ir.Value, other: ir.Value) -> ir.Value:
    """logaddexp(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call(
        "Log",
        builder.call(
            "Add", builder.call("Exp", self)[0], builder.call("Exp", other)[0]
        )[0],
    )[0]


@onnx_impl(aten.logaddexp2.default)
def aten_logaddexp2(self: ir.Value, other: ir.Value) -> ir.Value:
    """logaddexp2(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()
    two = builder.call("CastLike", 2.0, self)[0]
    summation = builder.call(
        "Add", builder.call("Pow", two, self)[0], builder.call("Pow", two, other)[0]
    )[0]

    return builder.call(
        "Div", builder.call("Log", summation)[0], builder.call("Log", two)[0]
    )[0]


@onnx_impl(aten.logical_and.default)
def aten_logical_and(self: ir.Value, other: ir.Value) -> ir.Value:
    """logical_and(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if self.dtype != other.dtype:
        raise ValueError(
            f"aten_logical_and requires matching tensor dtypes, got {self.dtype} and {other.dtype}. Cast operands to the same dtype before translation."
        )

    if self.dtype == ir.DataType.BOOL:
        return builder.call("And", self, other)[0]
    return builder.call(
        "And",
        builder.call("Cast", self, to=ir.DataType.BOOL)[0],
        builder.call("Cast", other, to=ir.DataType.BOOL)[0],
    )[0]


@onnx_impl(aten.logical_not.default)
def aten_logical_not(self: ir.Value) -> ir.Value:
    """logical_not(Tensor self) -> Tensor"""
    builder = current_builder()

    if self.dtype == ir.DataType.BOOL:
        return builder.call("Not", self)[0]
    return builder.call("Not", builder.call("Cast", self, to=ir.DataType.BOOL)[0])[0]


@onnx_impl(aten.logical_or.default)
def aten_logical_or(self: ir.Value, other: ir.Value) -> ir.Value:
    """logical_or(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if self.dtype != other.dtype:
        raise ValueError(
            f"aten_logical_or requires matching tensor dtypes, got {self.dtype} and {other.dtype}. Cast operands to the same dtype before translation."
        )

    if self.dtype == ir.DataType.BOOL:
        return builder.call("Or", self, other)[0]
    return builder.call(
        "Or",
        builder.call("Cast", self, to=ir.DataType.BOOL)[0],
        builder.call("Cast", other, to=ir.DataType.BOOL)[0],
    )[0]


@onnx_impl(aten.logical_xor.default)
def aten_logical_xor(self: ir.Value, other: ir.Value) -> ir.Value:
    """logical_xor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if self.dtype != other.dtype:
        raise ValueError(
            f"aten_logical_xor requires matching tensor dtypes, got {self.dtype} and {other.dtype}. Cast operands to the same dtype before translation."
        )

    if self.dtype == ir.DataType.BOOL:
        return builder.call("Xor", self, other)[0]
    return builder.call(
        "Xor",
        builder.call("Cast", self, to=ir.DataType.BOOL)[0],
        builder.call("Cast", other, to=ir.DataType.BOOL)[0],
    )[0]


@onnx_impl(aten.logit.default)
def aten_logit(self: ir.Value, eps: float | None = None) -> ir.Value:
    """logit(Tensor self, float? eps=None) -> Tensor"""
    builder = current_builder()
    one = builder.constant(1, dtype=self.dtype)

    if eps is None:
        return builder.call(
            "Log", builder.call("Div", self, builder.call("Sub", one, self)[0])[0]
        )[0]

    one_minus_eps = builder.constant(1 - eps, dtype=self.dtype)
    eps = builder.constant(eps, dtype=self.dtype)

    # Logit's lower-bound branch takes precedence even when eps > 0.5.
    upper_clamped = builder.call(
        "Where",
        builder.call("Greater", self, one_minus_eps)[0],
        one_minus_eps,
        self,
    )[0]
    z = builder.call("Where", builder.call("Less", self, eps)[0], eps, upper_clamped)[0]

    return builder.call(
        "Log", builder.call("Div", z, builder.call("Sub", one, z)[0])[0]
    )[0]


@onnx_impl((aten.lt.Tensor, aten.lt.Scalar, aten.less.Tensor))
def aten_lt(self: ir.Value, other: ir.Value) -> ir.Value:
    """lt.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if self.dtype == ir.DataType.BOOL:
        # self, other, self < other
        #    F,    F,    F
        #    F,    T,    T
        #    T,    F,    F
        #    T,    T,    F
        return builder.call("And", other, builder.call("Not", self)[0])[0]

    return builder.call("Less", self, other)[0]


@onnx_impl(operator.lt)
def operator_lt(self: ir.Value, other: ir.Value) -> ir.Value:
    builder = current_builder()
    # operator.lt for SymInt
    return builder.call("Less", self, other)[0]


@onnx_impl(aten.maximum.default)
def aten_maximum(self: ir.Value, other: ir.Value) -> ir.Value:
    """maximum(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if self.dtype == ir.DataType.BOOL:
        return builder.call("Or", self, other)[0]

    return builder.call("Max", self, other)[0]


@onnx_impl(aten.minimum.default)
def aten_minimum(self: ir.Value, other: ir.Value) -> ir.Value:
    """minimum(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if self.dtype == ir.DataType.BOOL:
        return builder.call("And", self, other)[0]

    return builder.call("Min", self, other)[0]


@onnx_impl((aten.mul.default, aten.mul.Tensor, aten.multiply.Tensor))
def aten_mul(self: ir.Value, other: ir.Value) -> ir.Value:
    """mul.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    if self.dtype == ir.DataType.BOOL:
        return builder.call("And", self, other)[0]

    return builder.call("Mul", self, other)[0]


@onnx_impl(operator.mul)
def operator_mul(self: ir.Value, other: ir.Value) -> ir.Value:
    builder = current_builder()
    return builder.call("Mul", self, other)[0]


@onnx_impl((aten.mul.default, aten.mul.Tensor, aten.multiply.Tensor), complex=True)
def aten_mul_complex(
    self: ir.Value, other: ir.Value | complex | float | int | bool
) -> ir.Value:
    """mul.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()
    other = common_ops.complex_scalar(other, dtype=self.dtype)

    # TODO(justinchuby): Maybe use Split to simplify the logic
    self_real = builder.call("Slice", self, [0], [1], [-1])[0]
    self_imag = builder.call("Slice", self, [1], [2], [-1])[0]
    other_real = builder.call("Slice", other, [0], [1], [-1])[0]
    other_imag = builder.call("Slice", other, [1], [2], [-1])[0]

    # Complex multiplication
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i

    ac = builder.call("Mul", self_real, other_real)[0]
    bd = builder.call("Mul", self_imag, other_imag)[0]
    ad = builder.call("Mul", self_real, other_imag)[0]
    bc = builder.call("Mul", self_imag, other_real)[0]

    real = builder.call("Sub", ac, bd)[0]
    imag = builder.call("Add", ad, bc)[0]

    return builder.call("Concat", real, imag, axis=-1)[0]


@onnx_impl((aten.mul.Scalar, aten.multiply.Scalar))
def aten_mul_scalar(self: ir.Value, other: float) -> ir.Value:
    """mul.Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = current_builder()

    other = builder.constant(other, dtype=self.dtype)
    return aten_mul(self, other)


@onnx_impl((aten.ne.default, aten.ne.Scalar, aten.ne.Tensor, operator.ne))
def aten_ne(self: ir.Value, other: ir.Value) -> ir.Value:
    """ne.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("Not", builder.call("Equal", self, other)[0])[0]


@onnx_impl((aten.neg.default, operator.neg))
def aten_neg(self: ir.Value) -> ir.Value:
    """neg(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Neg", self)[0]


@onnx_impl((aten.neg.default, operator.neg), complex=True)
def aten_neg_complex(self: ir.Value) -> ir.Value:
    """neg(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Neg", self)[0]


@onnx_impl((aten.pow.Tensor_Tensor, operator.pow))
def aten_pow(self: ir.Value, exponent: ir.Value) -> ir.Value:
    """pow(Tensor self, Tensor exponent) -> Tensor"""
    builder = current_builder()
    # TODO(justinchuby): Add type promotion
    return builder.call("Pow", self, exponent)[0]


@onnx_impl(aten.pow.Tensor_Scalar)
def aten_pow_tensor_scalar(self: ir.Value, exponent: float) -> ir.Value:
    """pow(Tensor self, Scalar exponent) -> Tensor"""
    builder = current_builder()
    if self.dtype.is_floating_point():
        # Handle cases when e.g. (1) self is float16 or int
        return builder.call("Pow", self, builder.constant(exponent, dtype=self.dtype))[
            0
        ]
    # For integer types, we need to cast self to the exponent type
    if isinstance(exponent, int):
        # The scalar exponent can be an int
        return builder.call("Pow", self, builder.constant(exponent, dtype=self.dtype))[
            0
        ]

    # exponent is float so we cast self to match the exponent type.
    # More precisely if self is float64, we should cast exponent to float64; but
    # this is uncommon and should be fixed when we create a general type promotion
    # mechanism for torchlib
    return builder.call(
        "Pow", builder.call("Cast", self, to=ir.DataType.FLOAT)[0], exponent
    )[0]


@onnx_impl(aten.pow.Scalar)
def aten_pow_scalar(self: float, exponent: ir.Value) -> ir.Value:
    """pow.Scalar(Scalar self, Tensor exponent) -> Tensor"""
    builder = current_builder()
    if not isinstance(self, int) and not exponent.dtype.is_floating_point():
        # A float scalar outranks an integral exponent, so torch promotes the result to
        # the default float type instead of narrowing the scalar down to the exponent
        return builder.call(
            "Pow",
            builder.constant(self, dtype=ir.DataType.FLOAT),
            builder.call("Cast", exponent, to=ir.DataType.FLOAT)[0],
        )[0]
    if exponent.dtype == ir.DataType.BOOL:
        # Pow has no boolean inputs, and an int scalar over a boolean exponent
        # promotes to the default int type in torch
        return builder.call(
            "Pow",
            builder.constant(self, dtype=ir.DataType.INT64),
            builder.call("Cast", exponent, to=ir.DataType.INT64)[0],
        )[0]
    # The exponent is in the same or a higher type category than the scalar, so it
    # decides the result type. e.g. 2.0 ** float16 tensor is float16
    return builder.call("Pow", builder.constant(self, dtype=exponent.dtype), exponent)[
        0
    ]


@onnx_impl(aten.rad2deg.default)
def aten_rad2deg(self: ir.Value) -> ir.Value:
    """rad2deg(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call(
        "Mul", self, builder.call("CastLike", 180.0 / math.pi, self)[0]
    )[0]


@onnx_impl(aten.reciprocal.default)
def aten_reciprocal(self: ir.Value) -> ir.Value:
    """reciprocal(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Reciprocal", self)[0]


@onnx_impl(aten.reciprocal.default, complex=True)
def aten_reciprocal_complex(self: ir.Value) -> ir.Value:
    """reciprocal(Tensor self) -> Tensor"""
    builder = current_builder()

    self_real = builder.call("Slice", self, [0], [1], [-1])[0]
    self_imag = builder.call("Slice", self, [1], [2], [-1])[0]

    # Complex reciprocal
    # 1 / (c + di) = c / (c^2 + d^2) - d / (c^2 + d^2) i
    denominator = builder.call(
        "Add",
        builder.call("Mul", self_real, self_real)[0],
        builder.call("Mul", self_imag, self_imag)[0],
    )[0]
    real = builder.call("Div", self_real, denominator)[0]
    imag = builder.call("Div", builder.call("Neg", self_imag)[0], denominator)[0]

    return builder.call("Concat", real, imag, axis=-1)[0]


def _aten_remainder(self: ir.Value, other: ir.Value, integer: bool) -> ir.Value:
    builder = current_builder()
    if integer:
        return builder.call("Mod", self, other)[0]

    # TODO(justinchuby): Improve fp16 precision by following the logic in
    # https://github.com/pytorch/pytorch/blob/3a823e46170778cc32783f27596c77d0103084a9/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp#L264-L277

    # a - a.div(b, rounding_mode="floor") * b
    rounded_quotient = builder.call("Floor", builder.call("Div", self, other)[0])[0]

    return builder.call("Sub", self, builder.call("Mul", rounded_quotient, other)[0])[0]


@onnx_impl((aten.remainder.Tensor, prims.remainder.default))
def aten_remainder(self: ir.Value, other: ir.Value) -> ir.Value:
    """remainder.Tensor(Tensor self, Tensor other) -> Tensor"""
    return _aten_remainder(self, other, integer=self.dtype.is_integer())


@onnx_impl(aten.remainder.Scalar)
def aten_remainder_scalar(self: ir.Value, other: float) -> ir.Value:
    """remainder.Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = current_builder()

    other_tensor = builder.constant(other, dtype=self.dtype)
    return _aten_remainder(self, other_tensor, integer=self.dtype.is_integer())


@onnx_impl(aten.remainder.Scalar_Tensor)
def aten_remainder_scalar_tensor(self: float, other: ir.Value) -> ir.Value:
    """remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = current_builder()

    self_tensor = builder.constant(self, dtype=other.dtype)
    return _aten_remainder(self_tensor, other, integer=other.dtype.is_integer())


@onnx_impl(operator.mod)
def operator_mod(self: ir.Value, other: ir.Value) -> ir.Value:
    builder = current_builder()
    # Modulus operator % on SymInt
    return builder.call("Mod", self, other)[0]


@onnx_impl(aten.round.default)
def aten_round(self: ir.Value) -> ir.Value:
    """round(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Round", self)[0]


@onnx_impl(aten.round.decimals)
def aten_round_decimals(self: ir.Value, decimals: int = 0) -> ir.Value:
    """round.decimals(Tensor self, *, int decimals) -> Tensor"""
    builder = current_builder()

    # Scale the input by 10^decimals, round it, and scale it back.
    ten = builder.call("CastLike", 10.0, self)[0]
    scale = builder.call("Pow", ten, builder.call("CastLike", decimals, self)[0])[0]
    self_scaled = builder.call("Mul", self, scale)[0]
    rounded = builder.call("Round", self_scaled)[0]
    return builder.call("Div", rounded, scale)[0]


@onnx_impl(aten.rsqrt.default)
def aten_rsqrt(self: ir.Value) -> ir.Value:
    """rsqrt(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Reciprocal", builder.call("Sqrt", self)[0])[0]


@onnx_impl(aten.scalar_tensor.default)
def aten_scalar_tensor(
    s: ir.Value | float | int | bool,
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    if dtype == -1:
        dtype = ir.DataType.FLOAT

    if isinstance(s, float) and dtype in (ir.DataType.DOUBLE, ir.DataType.COMPLEX128):
        s = builder.constant(s, dtype=ir.DataType.DOUBLE)
    if not isinstance(s, ir.Value):
        real_dtype = ir.DataType(dtype)
        if real_dtype in (ir.DataType.COMPLEX64, ir.DataType.COMPLEX128):
            real_dtype = (
                ir.DataType.FLOAT
                if real_dtype == ir.DataType.COMPLEX64
                else ir.DataType.DOUBLE
            )
        s = builder.constant(s, dtype=real_dtype)
    return common_ops.cast_to(s, dtype=dtype)


@onnx_impl(aten.scalar_tensor.default, complex=True)
def aten_scalar_tensor_complex(
    s: ir.Value,
    dtype: int = ir.DataType.COMPLEX64,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = current_builder()
    # NOTE: When the input is originally in complex, this function is invoked.
    # On the other hand, when the input is originally in real, aten_scalar_tensor is used.
    # is invoked.
    if dtype == -1:
        dtype = ir.DataType.COMPLEX64
    if dtype == ir.DataType.COMPLEX128:
        result = builder.call("Cast", s, to=ir.DataType.DOUBLE)[0]
    elif dtype == ir.DataType.COMPLEX64:
        result = builder.call("Cast", s, to=ir.DataType.FLOAT)[0]
    else:
        # The source's no-op for non-complex dtype could hide an invalid conversion.
        raise TypeError(
            f"scalar_tensor with complex input cannot use dtype {dtype}. "
            "Use COMPLEX64 or COMPLEX128 for complex scalars."
        )
    return result


@onnx_impl(aten.sign.default)
def aten_sign(self: ir.Value) -> ir.Value:
    """sign(Tensor self) -> Tensor"""
    builder = current_builder()

    if self.dtype == ir.DataType.BOOL:
        return builder.call("Identity", self)[0]

    return builder.call("Sign", self)[0]


@onnx_impl(aten.sin.default)
def aten_sin(self: ir.Value) -> ir.Value:
    """sin(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Sin", self)[0]


@onnx_impl(aten.sinh.default)
def aten_sinh(self: ir.Value) -> ir.Value:
    """sinh(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Sinh", self)[0]


@onnx_impl(aten.sqrt.default)
def aten_sqrt(self: ir.Value) -> ir.Value:
    """sqrt(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Sqrt", self)[0]


@onnx_impl((aten.sub.Tensor, aten.subtract.Tensor, operator.sub))
def aten_sub(self: ir.Value, other: ir.Value, alpha: float = 1.0) -> ir.Value:
    """sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    builder = current_builder()
    if alpha != 1.0:
        alpha = builder.call("CastLike", alpha, other)[0]
        other = builder.call("Mul", other, alpha)[0]
    return builder.call("Sub", self, other)[0]


@onnx_impl((aten.sub.Scalar, aten.subtract.Scalar))
def aten_sub_scalar(self: ir.Value, other: float, alpha: float = 1.0) -> ir.Value:
    """sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor"""
    builder = current_builder()

    other = builder.constant(other, dtype=self.dtype)
    return aten_sub(self, other, alpha=alpha)


@onnx_impl(
    (aten.sub.Tensor, aten.sub.Scalar, aten.subtract.Tensor, aten.subtract.Scalar),
    complex=True,
)
def aten_sub_complex(
    self: ir.Value, other: ir.Value | complex | float | int | bool, alpha: float = 1.0
) -> ir.Value:
    """sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    other = common_ops.complex_scalar(other, dtype=self.dtype)
    return aten_sub(self, other, alpha=alpha)


@onnx_impl(aten.tan.default)
def aten_tan(self: ir.Value) -> ir.Value:
    """tan(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Tan", self)[0]


@onnx_impl(aten.tanh.default)
def aten_tanh(self: ir.Value) -> ir.Value:
    """tanh(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Tanh", self)[0]


@onnx_impl(aten.trunc.default)
def aten_trunc(self: ir.Value) -> ir.Value:
    """trunc(Tensor self) -> Tensor"""
    builder = current_builder()
    # Reference https://github.com/onnx/onnx/issues/4588#issuecomment-2658170591
    return builder.call(
        "Mul",
        builder.call("Floor", builder.call("Abs", self)[0])[0],
        builder.call("Sign", self)[0],
    )[0]


@onnx_impl(math.trunc)
def python_math_trunc(self: ir.Value) -> ir.Value:
    """trunc(Tensor self) -> Tensor"""
    builder = current_builder()
    # NOTE: This is used in SymInt/SymBool/SymFloat context, so
    # we don't expect overflow to happen here.
    return builder.call("Cast", self, to=ir.DataType.INT64)[0]


@onnx_impl(aten.type_as.default)
def aten_type_as(self: ir.Value, other: ir.Value) -> ir.Value:
    """type_as(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("CastLike", self, other)[0]


_INT64_MAX = 9223372036854775807
_INT64_MIN = -9223372036854775808
_AS_STRIDED_STATIC_INDEX_SIZE_LIMIT = 512 * 512


@onnx_impl(aten.all.default)
def aten_all(self: ir.Value) -> ir.Value:
    """all(Tensor self) -> Tensor"""
    builder = current_builder()

    if len(self.shape) == 0:
        result = builder.call("Cast", self, to=ir.DataType.BOOL)[0]
    else:
        self_bool = builder.call("Cast", self, to=ir.DataType.BOOL)[0]
        self_int = builder.call("Cast", self_bool, to=ir.DataType.INT64)[0]
        all_true = builder.call("ReduceMin", self_int, keepdims=False)[0]
        result = builder.call("Cast", all_true, to=ir.DataType.BOOL)[0]
    return result


@onnx_impl(aten.all.dim)
def aten_all_dim(self: ir.Value, dim: int, keepdim: bool = False) -> ir.Value:
    """all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor"""
    builder = current_builder()
    if len(self.shape) == 0:
        return builder.call("Cast", self, to=ir.DataType.BOOL)[0]

    self_bool = builder.call("Cast", self, to=ir.DataType.BOOL)[0]
    self_int = builder.call("Cast", self_bool, to=ir.DataType.INT64)[0]
    dims = builder.call(
        "Reshape", dim, builder.constant([-1], dtype=ir.DataType.INT64)
    )[0]
    all_true = builder.call("ReduceMin", self_int, dims, keepdims=keepdim)[0]
    return builder.call("Cast", all_true, to=ir.DataType.BOOL)[0]


@onnx_impl(aten.all.dims)
def aten_all_dims(
    self: ir.Value, dim: Sequence[int] = (), keepdim: bool = False
) -> ir.Value:
    """all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor"""
    builder = current_builder()

    if not dim:
        return _aten_all_dims_no_dim(self, keepdim)
    if len(self.shape) == 0:
        return builder.call("Cast", self, to=ir.DataType.BOOL)[0]
    self_bool = builder.call("Cast", self, to=ir.DataType.BOOL)[0]
    self_int = builder.call("Cast", self_bool, to=ir.DataType.INT64)[0]
    result = builder.call("ReduceMin", self_int, list(dim), keepdims=keepdim)[0]
    return builder.call("Cast", result, to=ir.DataType.BOOL)[0]


def _aten_all_dims_no_dim(self: ir.Value, keepdims: bool) -> ir.Value:
    """all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor"""
    builder = current_builder()

    if len(self.shape) == 0:
        result = builder.call("Cast", self, to=ir.DataType.BOOL)[0]
    else:
        self_bool = builder.call("Cast", self, to=ir.DataType.BOOL)[0]
        self_int = builder.call("Cast", self_bool, to=ir.DataType.INT64)[0]
        all_true = builder.call("ReduceMin", self_int, keepdims=keepdims)[0]
        result = builder.call("Cast", all_true, to=ir.DataType.BOOL)[0]
    return result


@onnx_impl(aten.amax.default)
def aten_amax(
    self: ir.Value, dim: ir.Value | None = None, keepdim: bool = False
) -> ir.Value:
    """amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"""
    builder = current_builder()

    if dim is None:
        # dim defaults to the empty list in the aten schema, which means reduce every
        # dimension. noop_with_empty_axes keeps its default of 0, so ReduceMax without
        # an axes input reduces all of them.
        return builder.call("ReduceMax", self, keepdims=keepdim)[0]
    # An explicitly empty dim arrives here and reduces every dimension for the same reason
    return builder.call("ReduceMax", self, dim, keepdims=keepdim)[0]


@onnx_impl(aten.amin.default)
def aten_amin(
    self: ir.Value, dim: ir.Value | None = None, keepdim: bool = False
) -> ir.Value:
    """amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"""
    builder = current_builder()

    if dim is None:
        # dim defaults to the empty list in the aten schema, which means reduce every
        # dimension. noop_with_empty_axes keeps its default of 0, so ReduceMin without
        # an axes input reduces all of them.
        return builder.call("ReduceMin", self, keepdims=keepdim)[0]
    # An explicitly empty dim arrives here and reduces every dimension for the same reason
    return builder.call("ReduceMin", self, dim, keepdims=keepdim)[0]


@onnx_impl(aten.any.default)
def aten_any(self: ir.Value) -> ir.Value:
    """any(Tensor self) -> Tensor"""
    builder = current_builder()

    if len(self.shape) == 0:
        result = builder.call("Cast", self, to=ir.DataType.BOOL)[0]
    else:
        self_bool = builder.call("Cast", self, to=ir.DataType.BOOL)[0]
        # op.ReduceMax() in the next step cannot process BOOL inputs, so convert to INT64
        self_int = builder.call("Cast", self_bool, to=ir.DataType.INT64)[0]
        any_true = builder.call("ReduceMax", self_int, keepdims=False)[0]
        result = builder.call("Greater", any_true, 0)[0]
    return result


@onnx_impl(aten.any.dim)
def aten_any_dim(self: ir.Value, dim: int, keepdim: bool = False) -> ir.Value:
    """any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor"""
    builder = current_builder()
    if len(self.shape) == 0:
        return builder.call("Cast", self, to=ir.DataType.BOOL)[0]

    self_bool = builder.call("Cast", self, to=ir.DataType.BOOL)[0]
    # op.ReduceMax() in the next step cannot process BOOL inputs, so convert to INT64
    self_int = builder.call("Cast", self_bool, to=ir.DataType.INT64)[0]
    # Change dim from int to INT64[1]
    dims = builder.call(
        "Reshape", dim, builder.constant([-1], dtype=ir.DataType.INT64)
    )[0]
    any_true = builder.call("ReduceMax", self_int, dims, keepdims=keepdim)[0]
    return builder.call("Greater", any_true, 0)[0]


@onnx_impl(aten.any.dims)
def aten_any_dims(
    self: ir.Value, dim: Sequence[int] = (), keepdim: bool = False
) -> ir.Value:
    """any.dims(Tensor self, int[1]? dim=None, bool keepdim=False) -> Tensor"""
    builder = current_builder()

    if not dim:
        return _aten_any_dims_no_dim(self, keepdim)
    if len(self.shape) == 0:
        return builder.call("Cast", self, to=ir.DataType.BOOL)[0]
    self_bool = builder.call("Cast", self, to=ir.DataType.BOOL)[0]
    self_int = builder.call("Cast", self_bool, to=ir.DataType.INT64)[0]
    result = builder.call("ReduceMax", self_int, list(dim), keepdims=keepdim)[0]
    return builder.call("Greater", result, 0)[0]


def _aten_any_dims_no_dim(self: ir.Value, keepdims: bool) -> ir.Value:
    builder = current_builder()
    if len(self.shape) == 0:
        result = builder.call("Cast", self, to=ir.DataType.BOOL)[0]
    else:
        self_bool = builder.call("Cast", self, to=ir.DataType.BOOL)[0]
        self_int = builder.call("Cast", self_bool, to=ir.DataType.INT64)[0]
        any_true = builder.call("ReduceMax", self_int, keepdims=keepdims)[0]
        result = builder.call("Greater", any_true, 0)[0]
    return result


@onnx_impl(aten.argmax.default)
def aten_argmax(
    self: ir.Value, dim: int | None = None, keepdim: bool = False
) -> ir.Value:
    """argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""

    if dim is None:
        result = _aten_argmax(self, keepdim)
    else:
        result = _aten_argmax_dim(self, dim, keepdim)
    return result


def _aten_argmax(self: ir.Value, keepdim: bool = False) -> ir.Value:
    """argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""
    builder = current_builder()

    self_is_scaler = len(self.shape) == 0
    self = builder.call(
        "Reshape", self, builder.constant([-1], dtype=ir.DataType.INT64)
    )[0]
    result = builder.call("ArgMax", self, keepdims=keepdim)[0]
    if self_is_scaler:
        result = builder.call("Squeeze", result)[0]

    return result


def _aten_argmax_dim(self: ir.Value, dim: int, keepdim: bool = False) -> ir.Value:
    """argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""
    builder = current_builder()

    self_is_scaler = len(self.shape) == 0
    if self_is_scaler:
        self = builder.call(
            "Reshape", self, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]

    result = builder.call("ArgMax", self, axis=dim, keepdims=keepdim)[0]
    if self_is_scaler:
        result = builder.call("Squeeze", result)[0]

    return result


@onnx_impl(aten.argmin.default)
def aten_argmin(
    self: ir.Value, dim: int | None = None, keepdim: bool = False
) -> ir.Value:
    """argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""

    if dim is None:
        result = _aten_argmin(self, keepdim)
    else:
        result = _aten_argmin_dim(self, dim, keepdim)
    return result


def _aten_argmin(self: ir.Value, keepdim: bool = False) -> ir.Value:
    """argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""
    builder = current_builder()

    self_is_scaler = len(self.shape) == 0
    self = builder.call(
        "Reshape", self, builder.constant([-1], dtype=ir.DataType.INT64)
    )[0]
    result = builder.call("ArgMin", self, keepdims=keepdim)[0]
    if self_is_scaler:
        result = builder.call("Squeeze", result)[0]

    return result


def _aten_argmin_dim(self: ir.Value, dim: int, keepdim: bool = False) -> ir.Value:
    """argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""
    builder = current_builder()

    self_is_scaler = len(self.shape) == 0
    if self_is_scaler:
        self = builder.call(
            "Reshape", self, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]

    result = builder.call("ArgMin", self, axis=dim, keepdims=keepdim)[0]
    if self_is_scaler:
        result = builder.call("Squeeze", result)[0]

    return result


@onnx_impl(aten.atleast_1d.default)
def aten_atleast_1d(self: ir.Value) -> ir.Value:
    """atleast_1d(Tensor self) -> Tensor"""
    builder = current_builder()

    if len(self.shape) == 0:
        self = builder.call(
            "Reshape", self, builder.constant([1], dtype=ir.DataType.INT64)
        )[0]
    return builder.call("Identity", self)[0]


@onnx_impl(aten.atleast_2d.default)
def aten_atleast_2d(self: ir.Value) -> ir.Value:
    """atleast_2d(Tensor self) -> Tensor"""
    builder = current_builder()

    if len(self.shape) <= 1:
        self = builder.call(
            "Reshape", self, builder.constant([1, -1], dtype=ir.DataType.INT64)
        )[0]
    return builder.call("Identity", self)[0]


@onnx_impl(aten.atleast_3d.default)
def aten_atleast_3d(self: ir.Value) -> ir.Value:
    """atleast_3d(Tensor self) -> Tensor"""
    builder = current_builder()

    rank = len(self.shape)
    if rank <= 1:
        self = builder.call(
            "Reshape", self, builder.constant([1, -1, 1], dtype=ir.DataType.INT64)
        )[0]
    elif rank == 2:
        self = builder.call(
            "Unsqueeze", self, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]
    return builder.call("Identity", self)[0]


@onnx_impl(aten.broadcast_to.default)
def aten_broadcast_to(self: ir.Value, size: Sequence[int | ir.Value]) -> ir.Value:
    """broadcast_to(Tensor(a) self, SymInt[] size) -> Tensor(a)"""
    builder = current_builder()
    size = common_ops.merge_dims(size)
    return builder.call("Expand", self, size)[0]


@onnx_impl(aten.cumsum.default)
def aten_cumsum(self: ir.Value, dim: ir.Value, dtype: int = -1) -> ir.Value:
    """cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()

    # The accumulation type for int32 is int64.
    if dtype != -1 and dtype is not None:
        cast = builder.call("Cast", self, to=dtype)[0]
    elif self.dtype.is_integer() or self.dtype == ir.DataType.BOOL:
        cast = builder.call("Cast", self, to=ir.DataType.INT64)[0]
    else:
        cast = self
    if len(self.shape) == 0:
        # A scalar
        result = builder.call("Identity", cast)[0]
    else:
        result = builder.call("CumSum", cast, dim)[0]
    return result


@onnx_impl(aten.expand.default)
def aten_expand(
    self: ir.Value, size: Sequence[int | ir.Value], implicit: bool = False
) -> ir.Value:
    """expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)"""
    builder = current_builder()
    # NOTE: PyTorch supports `not changing dim` by -1, but ONNX supports `not changing dim` by 1.
    # To support -1 dim, we need to convert -1 to 1.
    # Even though in theory a dynamic dim can still be -1, in practice it is very unlikely
    # and isn't expected to appear from correct usages of SymInt.
    size = [1 if isinstance(s, int) and s == -1 else s for s in size]
    return builder.call("Expand", self, common_ops.merge_dims(size))[0]


@onnx_impl(aten.expand_as.default)
def aten_expand_as(self: ir.Value, other: ir.Value) -> ir.Value:
    """expand_as(Tensor(a) self, Tensor other) -> Tensor(a)"""
    builder = current_builder()

    shape = builder.call("Shape", other)[0]
    result = builder.call("Expand", self, shape)[0]
    return result


@onnx_impl(aten.flatten.using_ints)
def aten_flatten(self: ir.Value, start_dim: int = 0, end_dim: int = -1) -> ir.Value:
    """flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)"""
    builder = current_builder()
    dim = len(self.shape)
    if dim == 0:
        return builder.call("Reshape", self, [1])[0]
    if start_dim < 0:
        start_dim += dim
    if dim == 1:
        return builder.call("Identity", self)[0]
    # use ONNX's Flatten operator for cases where the output shape is 2D
    if start_dim == 1:
        if end_dim in (-1, dim - 1):
            return builder.call("Flatten", self, axis=start_dim)[0]
    elif start_dim == 0:
        if end_dim in (-2, dim - 2):
            return builder.call("Flatten", self, axis=end_dim + 1)[0]

    # if end_dim is negative add dim
    if end_dim < 0:
        end_dim = dim + end_dim

    input_size = builder.call("Shape", self)[0]
    dim_head = builder.call(
        "Slice",
        input_size,
        builder.constant([0], dtype=ir.DataType.INT64),
        builder.constant([start_dim], dtype=ir.DataType.INT64),
        builder.constant([0], dtype=ir.DataType.INT64),
    )[0]
    final_dims = [dim_head, builder.constant([-1], dtype=ir.DataType.INT64)]
    if end_dim < dim - 1:
        dim_tail = builder.call(
            "Slice",
            input_size,
            builder.constant([end_dim + 1], dtype=ir.DataType.INT64),
            builder.constant([dim], dtype=ir.DataType.INT64),
            builder.constant([0], dtype=ir.DataType.INT64),
        )[0]
        final_dims = [
            dim_head,
            builder.constant([-1], dtype=ir.DataType.INT64),
            dim_tail,
        ]

    final_shape = builder.call("Concat", *final_dims, axis=0)[0]
    return builder.call("Reshape", self, final_shape)[0]


@onnx_impl(aten.flip.default)
def aten_flip(self: ir.Value, dims: Sequence[int]) -> ir.Value:
    """flip(Tensor self, int[] dims) -> Tensor"""
    builder = current_builder()

    if not dims:
        # Nothing to flip
        return builder.call("Identity", self)[0]

    rank = len(dims)
    starts = builder.constant(
        [-1] * rank, dtype=ir.DataType.INT64
    )  # something like [-1, -1, -1]
    steps = starts  # something like [-1, -1, -1]
    ends = builder.constant(
        [_INT64_MIN] * rank, dtype=ir.DataType.INT64
    )  # something like [-xxx, -xxx, -xxx]
    dims = builder.constant(dims, dtype=ir.DataType.INT64)
    return builder.call("Slice", self, starts, ends, dims, steps)[0]


@onnx_impl(aten.flip.default, complex=True)
def aten_flip_complex(self: ir.Value, dims: Sequence[int]) -> ir.Value:
    """flip(Tensor self, int[] dims) -> Tensor"""
    # if dims is negative, take into account the complex dimension
    dims = [d - 1 if d < 0 else d for d in dims]
    return aten_flip(self, dims)


@onnx_impl(aten.gather.default)
def aten_gather(
    self: ir.Value,
    dim: int,
    index: ir.Value,
    sparse_grad: bool = False,
) -> ir.Value:
    """gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor"""
    builder = current_builder()

    if len(self.shape) == 0:
        if len(index.shape) == 0:
            return builder.call("Identity", self)[0]
        else:
            return builder.call("Expand", self, builder.call("Shape", index)[0])[0]

    is_scalar_index = len(index.shape) == 0
    if is_scalar_index:
        index = builder.call("Unsqueeze", index, [0])[0]

    index = builder.call("Cast", index, to=ir.DataType.INT64)[0]
    result = builder.call("GatherElements", self, index, axis=dim)[0]

    if is_scalar_index:
        result = builder.call("Squeeze", result, [0])[0]
    return result


def _are_consecutive(sorted_list: Sequence[int]) -> bool:
    """Returns True if a sorted list contains consecutive numbers."""
    if not sorted_list:
        return True

    return sorted_list == list(range(min(sorted_list), max(sorted_list) + 1))


def _has_none_in_middle(indices: Sequence[ir.Value | None]) -> bool:
    """Returns True if there is a None in the middle of the list."""
    not_none_indices = [i for i, idx in enumerate(indices) if idx is not None]
    return not _are_consecutive(not_none_indices)


def _shape_of_broadcast_tensors(*args: ir.Value) -> ir.Value:
    """Returns the broadcasted shape of the given tensors."""
    builder = current_builder()
    broadcasted = builder.call("Max", *args)[0]
    return builder.call("Shape", broadcasted)[0]


def _aten_index_onnx(
    self: ir.Value,
    indices: Sequence[ir.Value | None],
    index_ranks: Sequence[int],
) -> ir.Value:
    builder = current_builder()
    self_rank = len(self.shape)
    advanced_indexing_rank = max(index_ranks)

    # reordered_positions is the permutation of the index positions where
    # positions with None are move to the end of the list
    # For example, if indices = [None, 1, None, 2], then reordered_positions = [1, 3, 0, 2]
    reordered_positions = sorted(
        range(len(indices)), key=lambda i: (indices[i] is None, i)
    )

    # Fill the list with the remaining indices up to the rank of the tensor self.
    # For example, if indices = [None, 1, None, 2], and the rank of self is 6,
    # then reordered_positions = [1, 3, 0, 2, 4, 5]
    reordered_positions = [
        *reordered_positions,
        *range(len(reordered_positions), self_rank),
    ]
    # Transpose self according to the reordered positions
    self = builder.call("Transpose", self, perm=reordered_positions)[0]

    # Broadcast the indices to the same shape then concatenate
    not_none_indices = [idx for idx in indices if idx is not None]
    broadcast_shape = _shape_of_broadcast_tensors(*not_none_indices)
    final_index = builder.call(
        "Concat",
        *(
            builder.call(
                "Unsqueeze", builder.call("Expand", idx, broadcast_shape)[0], [-1]
            )[0]
            for idx in not_none_indices
        ),
        axis=-1,
    )[0]

    self = builder.call("GatherND", self, final_index, batch_dims=0)[0]

    if _has_none_in_middle(indices):
        # If there is None in the middle, Advanced Indexing cannot decide where to put
        # the new dimensions. So it places them in the front, like GatherND does.
        return self

    # When the indices are consecutive, Advanced Indexing will place the new dimensions
    # (aka. the broadcasted shape) in the middle, replacing the original [x1, ..., xk] axes.
    #
    # Input index axes (three parts):
    #   [
    #      x_None_front_1, ... x_None_front_m,
    #      x1, ..., xk,
    #      x_None_back_1, ..., x_None_back_m
    #   ]
    # GatherND result axes:
    #   [
    #      *broadcasted_shape(x1, x2, ..., xk),
    #      x_None_front_1, ... x_None_front_m,
    #      x_None_back_1, ..., x_None_back_m
    #   ]
    # (Transpose here)
    # Advanced indexing result axes:
    #   [
    #      x_None_front_1, ... x_None_front_m,
    #      *brocasted_shape(x1, x2, ..., xk),
    #      x_None_back_1, ..., x_None_back_m
    #   ]
    #
    # Need to transpose the result of GatherND to match this axes ordering.
    first_not_none_position = reordered_positions[0]  # x_None_front_m + 1
    starting_position_of_none_in_back = (
        advanced_indexing_rank + first_not_none_position
    )  # x_None_back_1
    result_rank = self_rank - len(not_none_indices) + advanced_indexing_rank
    perm = [
        *range(
            advanced_indexing_rank, starting_position_of_none_in_back
        ),  # None_front_1...x_None_back_1
        *range(advanced_indexing_rank),  # 0...len(broadcasted_shape)
        *range(
            starting_position_of_none_in_back,
            result_rank,
        ),  # None_back_1...None_back_m
    ]

    return builder.call("Transpose", self, perm=perm)[0]


@onnx_impl((aten.index.Tensor, aten._unsafe_index.Tensor))
def aten_index(self: ir.Value, indices: Sequence[ir.Value | None]) -> ir.Value:
    """index.Tensor(Tensor self, Tensor?[] indices) -> Tensor

    NOTE: Understanding `aten::index`
    For `arg0` with shape `[7, 3, 4, 5, 6]`
    The indexing operation `arg0[0, :, 1:2, tensor([[4,5]])]` will be translated to

    ```
    +>  select: i64[3, 4, 5, 6] = torch.ops.aten.select.int(arg0, 0, 0);
    +>  slice_1: i64[3, 4, 5, 6] = torch.ops.aten.slice.Tensor(select, 0, 0, 9223372036854775807);
    +>  slice_2: i64[3, 1, 5, 6] = torch.ops.aten.slice.Tensor(slice_1, 1, 1, 2);
    +>  index: i64[3, 1, 1, 2, 6] = torch.ops.aten.index.Tensor(slice_2, [None, None, arg1]);
    ```

    Here,
    - `indices = [None, None, arg1]` is equivalent to `indices = [None, None, arg1, None]`
    - The operation `arg0[0, :, 1:2, tensor([[4,5]])]` is equivalent to `arg0[0, :, 1:2, tensor([[4,5]]), :]`

    None in `indices` are like fillers for dimensions that cannot be removed in the process.
    """
    # Handle Boolean indexing first
    if any(index is not None and index.dtype == ir.DataType.BOOL for index in indices):
        return _aten_index_bool(self, indices)

    index_ranks = [len(index.shape) for index in indices if index is not None]

    return _aten_index_onnx(self, indices, index_ranks)


def _aten_index_bool(self: ir.Value, indices: Sequence[ir.Value | None]) -> ir.Value:
    builder = current_builder()
    index_ranks = [len(index.shape) for index in indices if index is not None]

    if all(rank == 1 for rank in index_ranks):
        # indices contains scalar only.
        new_indices = [
            builder.call("Transpose", builder.call("NonZero", index)[0], perm=[1, 0])[0]
            if index is not None
            else None
            for index in indices
        ]
        new_indices = [
            builder.call("Squeeze", index, [1])[0] if index is not None else None
            for index in new_indices
        ]
        return _aten_index_onnx(self, new_indices, index_ranks)

    else:
        input_rank = len(self.shape)
        # Prepare perm for transposing self tensor.
        # In indices, None meaning skip the corresponding dimension,
        # so we need to move this dimension to the end of the list.
        # After we gathered the final results, we transpose it back.
        # For example,
        # self's shape is [5, 5, 5, 5], indices is [None, (5, 5)]
        # the final result's shape should be [5, 16, 5].
        trans_perm = list(range(input_rank))
        trans_perm.append(trans_perm.pop(0))
        count_of_none = 0
        for index in indices:
            if index is None:
                self = builder.call("Transpose", self, perm=trans_perm)[0]
                count_of_none += 1
                continue

            new_indices = builder.call(
                "Transpose", builder.call("NonZero", index)[0], perm=[1, 0]
            )[0]
            result = builder.call("GatherND", self, new_indices, batch_dims=0)[0]
            final_rank = input_rank - (len(index.shape) - 1)
            trans_perm = list(range(final_rank))
            trans_perm = trans_perm[-1:] + trans_perm[:-1]
            for _ in range(count_of_none):
                result = builder.call("Transpose", result, perm=trans_perm)[0]
            # FIXME(justinchuby): Even though this logic passes the tests, it still looks strange:
            # why does it return early here instead of continuing to process the remaining indices?
            # I think the assumption here is that there can be only one Boolean index in the indices list?
            return result


@onnx_impl(aten.index_select.default)
def aten_index_select(self: ir.Value, dim: int, index: ir.Value) -> ir.Value:
    """index_select(Tensor self, int dim, Tensor index) -> Tensor"""
    builder = current_builder()

    self_is_scalar = len(self.shape) == 0
    if self_is_scalar:
        self = builder.call(
            "Reshape", self, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]

    # Index may be a scalar. Reshape it to a rank 1 tensor.
    index = builder.call(
        "Reshape", index, builder.constant([-1], dtype=ir.DataType.INT64)
    )[0]
    index = builder.call("Cast", index, to=ir.DataType.INT64)[0]
    result = builder.call("Gather", self, index, axis=dim)[0]

    if self_is_scalar:
        result = builder.call("Squeeze", result)[0]

    return result


@onnx_impl(aten.logsumexp.default)
def aten_logsumexp(self: ir.Value, dim: ir.Value, keepdim: int = False) -> ir.Value:
    """logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor"""
    builder = current_builder()

    if len(self.shape) == 0:
        # A scalar
        result = self
    else:
        result = builder.call("ReduceLogSumExp", self, dim, keepdims=keepdim)[0]
    return result


@onnx_impl(aten.mH.default)
def aten_mH(self: ir.Value) -> ir.Value:
    """mH(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    # Taking the conjugate transpose of a real matrix is the same as the transpose
    return builder.call("Einsum", self, equation="...ij->...ji")[0]


@onnx_impl(aten.mH.default, complex=True)
def aten_mH_complex(self: ir.Value) -> ir.Value:
    """mH(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    # c is the last dimension being the real and imaginary parts
    transposed = builder.call("Einsum", self, equation="...ijc->...jic")[0]
    return _complex_conjugate(transposed)


@onnx_impl(aten.mT.default)
def aten_mT(self: ir.Value) -> ir.Value:
    """mT(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    return builder.call("Einsum", self, equation="...ij->...ji")[0]


@onnx_impl(aten.mT.default, complex=True)
def aten_mT_complex(self: ir.Value) -> ir.Value:
    """mT(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    # c is the last dimension being the real and imaginary parts
    return builder.call("Einsum", self, equation="...ijc->...jic")[0]


@onnx_impl((aten.masked_fill.Scalar, aten.masked_fill.Tensor))
def aten_masked_fill(self: ir.Value, mask: ir.Value, value: ir.Value) -> ir.Value:
    """masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor"""
    builder = current_builder()
    # NOTE: Do not attempt to cast `mask` to BOOL because mask should not take any other types.
    # `mask` coming in as other types is often an error and should fail the model.
    value_cast = builder.call("CastLike", value, self)[0]
    return builder.call("Where", mask, value_cast, self)[0]


@onnx_impl(aten.masked_scatter.default)
def aten_masked_scatter(self: ir.Value, mask: ir.Value, source: ir.Value) -> ir.Value:
    """masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor"""
    builder = current_builder()

    # Broadcast self and mask to their common shape so NonZero enumerates every
    # masked element. The previous rank-only check missed same-rank broadcasting
    # (e.g. mask (1, S, 1) vs self (1, S, D)): it left mask un-expanded, so only a
    # subset of masked positions were scattered (pytorch/pytorch#186146).
    self = builder.call("Expand", self, builder.call("Shape", mask)[0])[0]
    mask = builder.call("Expand", mask, builder.call("Shape", self)[0])[0]
    index = builder.call("Transpose", builder.call("NonZero", mask)[0], perm=[1, 0])[0]

    # NOTE: source can have more elements than needed.
    # It could also have arbitrary shape.
    # This is not supported by ONNX::ScatterND, so we need to flatten and slice source tensor.
    source = builder.call(
        "Reshape", source, builder.constant([-1], dtype=ir.DataType.INT64)
    )[0]
    axes = builder.constant([0], dtype=ir.DataType.INT64)
    starts = builder.constant([0], dtype=ir.DataType.INT64)
    ends = builder.call(
        "Gather",
        builder.call("Shape", index)[0],
        builder.constant([0], dtype=ir.DataType.INT64),
        axis=0,
    )[0]
    source = builder.call("Slice", source, starts, ends, axes)[0]

    return builder.call("ScatterND", self, index, source)[0]


@onnx_impl(aten.max.default)
def aten_max(self: ir.Value) -> ir.Value:
    """max(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("ReduceMax", self, keepdims=False)[0]


@onnx_impl(aten.max.dim)
def aten_max_dim(
    self: ir.Value, dim: int, keepdim: bool = False
) -> tuple[ir.Value, ir.Value]:
    """max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"""
    builder = current_builder()

    if len(self.shape) == 0:
        result = self
        indices = builder.constant(0, dtype=ir.DataType.INT64)
    else:
        dims = builder.call(
            "Reshape", dim, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]
        result = builder.call("ReduceMax", self, dims, keepdims=keepdim)[0]
        indices = builder.call("ArgMax", self, axis=dim, keepdims=keepdim)[0]
    return result, indices


@onnx_impl(aten.mean.default)
def aten_mean(self: ir.Value, dtype: int = -1) -> ir.Value:
    """mean(Tensor self, *, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()

    if dtype != -1 and dtype is not None:
        # Cast before reducing so that the accumulation happens in the requested
        # dtype, matching PyTorch. Casting the result afterwards would keep the
        # precision loss of the input dtype.
        self = builder.call("Cast", self, to=dtype)[0]

    result = builder.call("ReduceMean", self)[0]
    return builder.call("Squeeze", result)[0]


@onnx_impl(aten.mean.dim)
def aten_mean_dim(
    self: ir.Value, dim: ir.Value, keepdim: bool = False, dtype: int = -1
) -> ir.Value:
    """mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()
    self_is_scalar = len(self.shape) == 0
    if dtype != -1 and dtype is not None:
        self = builder.call("Cast", self, to=dtype)[0]

    if self_is_scalar:
        result = self
    elif dim is None:
        result = builder.call("ReduceMean", self, keepdims=keepdim)[0]
    else:
        if not isinstance(dim, ir.Value):
            dim = builder.constant(dim, dtype=ir.DataType.INT64)
        dims = builder.call(
            "Reshape", dim, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]
        result = builder.call("ReduceMean", self, dims, keepdims=keepdim)[0]

    if dtype != -1 and dtype is not None:
        result = builder.call("Cast", result, to=dtype)[0]

    return result


@onnx_impl(aten.mean.default, complex=True)
def aten_mean_complex(self: ir.Value, dtype: int = -1) -> ir.Value:
    """mean(Tensor self, *, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()
    rank = len(self.shape) - 1
    if rank == 0:
        result = builder.call("Identity", self)[0]
    else:
        dim = builder.constant(list(range(rank)), dtype=ir.DataType.INT64)
        result = builder.call("ReduceMean", self, dim, keepdims=False)[0]

    if dtype != -1 and dtype is not None:
        raise NotImplementedError(
            "Complex mean does not support the dtype argument; cast the input "
            "before export or omit dtype."
        )

    return result


@onnx_impl(aten.mean.dim, complex=True)
def aten_mean_dim_complex(
    self: ir.Value,
    dim: ir.Value | Sequence[int] | None,
    keepdim: bool = False,
    dtype: int = -1,
) -> ir.Value:
    """mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()
    rank = len(self.shape) - 1
    if rank == 0:
        result = builder.call("Identity", self)[0]
    else:
        if dim is None:
            dims = builder.constant(list(range(rank)), dtype=ir.DataType.INT64)
        else:
            if not isinstance(dim, ir.Value):
                dim = builder.constant(dim, dtype=ir.DataType.INT64)
            dims = builder.call("Reshape", dim, [-1])[0]
            dims = builder.call(
                "Where",
                builder.call("Less", dims, 0)[0],
                builder.call("Sub", dims, 1)[0],
                dims,
            )[0]
            end = builder.call(
                "Where",
                builder.call("Equal", builder.call("Size", dims)[0], 0)[0],
                rank,
                0,
            )[0]
            all_dims = builder.call("Range", 0, end, 1)[0]
            dims = builder.call("Concat", dims, all_dims, axis=0)[0]
        result = builder.call("ReduceMean", self, dims, keepdims=keepdim)[0]

    if dtype != -1 and dtype is not None:
        raise NotImplementedError(
            "Complex mean does not support the dtype argument; cast the input "
            "before export or omit dtype."
        )

    return result


@onnx_impl(aten.min.default)
def aten_min(self: ir.Value) -> ir.Value:
    """min(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("ReduceMin", self, keepdims=False)[0]


@onnx_impl(aten.min.dim)
def aten_min_dim(
    self: ir.Value, dim: int, keepdim: bool = False
) -> tuple[ir.Value, ir.Value]:
    """min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"""
    builder = current_builder()
    if len(self.shape) == 0:
        result = self
        indices = builder.constant(0, dtype=ir.DataType.INT64)
    else:
        dims = builder.call(
            "Reshape", dim, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]
        result = builder.call("ReduceMin", self, dims, keepdims=keepdim)[0]
        indices = builder.call("ArgMin", self, axis=dim, keepdims=keepdim)[0]

    return result, indices


@onnx_impl(aten.narrow.default)
def aten_narrow(
    self: ir.Value, dim: ir.Value, start: ir.Value, length: ir.Value
) -> ir.Value:
    """narrow(Tensor(a) self, int dim, SymInt start, SymInt length) -> Tensor(a)"""
    builder = current_builder()

    dim = builder.call("Reshape", dim, builder.constant([-1], dtype=ir.DataType.INT64))[
        0
    ]
    start = builder.call(
        "Reshape", start, builder.constant([-1], dtype=ir.DataType.INT64)
    )[0]
    length = builder.call(
        "Reshape", length, builder.constant([-1], dtype=ir.DataType.INT64)
    )[0]

    end = builder.call("Add", start, length)[0]
    return builder.call("Slice", self, start, end, dim)[0]


@onnx_impl(aten.nonzero.default)
def aten_nonzero(self: ir.Value) -> ir.Value:
    """nonzero(Tensor self) -> Tensor"""
    builder = current_builder()
    # NOTE: In torch the return shape is [n, d], while in onnx [d, n],
    # where `d` is rank of input tensor, `n` is number of nonzero elements.
    return builder.call("Transpose", builder.call("NonZero", self)[0], perm=[1, 0])[0]


@onnx_impl(aten.permute.default)
def aten_permute(self: ir.Value, dims: Sequence[int]) -> ir.Value:
    """permute(Tensor(a) self, int[] dims) -> Tensor(a)"""
    builder = current_builder()

    if not dims:
        return builder.call("Transpose", self)[0]

    # Handle negative axes
    dims = [axis + len(dims) if axis < 0 else axis for axis in dims]

    return builder.call("Transpose", self, perm=dims)[0]


@onnx_impl(aten.permute.default, complex=True)
def aten_permute_complex(self: ir.Value, dims: Sequence[int]) -> ir.Value:
    """permute(Tensor(a) self, int[] dims) -> Tensor(a)"""
    builder = current_builder()

    rank = len(self.shape) - 1

    # Handle negative axes
    dims = [axis + rank if axis < 0 else axis for axis in dims] + [rank]

    return builder.call("Transpose", self, perm=dims)[0]


@onnx_impl(aten.prod.default)
def aten_prod(self: ir.Value, dtype: int = -1) -> ir.Value:
    """prod(Tensor self, *, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()

    if dtype != -1 and dtype is not None:
        self = builder.call("Cast", self, to=dtype)[0]
    elif self.dtype.is_integer() or self.dtype == ir.DataType.BOOL:
        self = builder.call("Cast", self, to=ir.DataType.INT64)[0]
    return builder.call("ReduceProd", self, keepdims=False)[0]


@onnx_impl(aten.prod.dim_int)
def aten_prod_dim_int(
    self: ir.Value, dim: int, keepdim: bool = False, dtype: int = -1
) -> ir.Value:
    """prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()
    self_is_scalar = len(self.shape) == 0

    if dtype != -1 and dtype is not None:
        self = builder.call("Cast", self, to=dtype)[0]
    elif self.dtype.is_integer() or self.dtype == ir.DataType.BOOL:
        self = builder.call("Cast", self, to=ir.DataType.INT64)[0]
    if self_is_scalar:
        return builder.call("Identity", self)[0]
    return builder.call("ReduceProd", self, [dim], keepdims=keepdim)[0]


@onnx_impl(aten.repeat.default)
def aten_repeat(self: ir.Value, repeats: Sequence[int | ir.Value]) -> ir.Value:
    """repeat(Tensor self, SymInt[] repeats) -> Tensor"""
    builder = current_builder()

    if len(repeats) == 0:
        return self
    self_expanded = builder.call("Expand", self, [1] * len(repeats))[0]
    return builder.call("Tile", self_expanded, repeats)[0]


@onnx_impl(aten.reshape.default)
def aten_reshape(self: ir.Value, shape: Sequence[int | ir.Value]) -> ir.Value:
    """reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)"""
    builder = current_builder()
    shape = common_ops.merge_dims(shape)
    return builder.call("Reshape", self, shape)[0]


@onnx_impl(aten.scatter.src)
def aten_scatter_src(
    self: ir.Value,
    dim: int,  # we have to use int here because ScatterElements() will use this attribute
    index: ir.Value,
    src: ir.Value,
) -> ir.Value:
    """scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor"""
    builder = current_builder()
    if len(index.shape) == 0:
        index = builder.call("Unsqueeze", index, [0])[0]
    if len(src.shape) == 0:
        src = builder.call("Unsqueeze", src, [0])[0]
    return builder.call("ScatterElements", self, index, src, axis=dim)[0]


@onnx_impl(aten.scatter.value)
def aten_scatter_value(
    self: ir.Value,
    dim: int,  # we have to use int here because ScatterElements() will use this attribute
    index: ir.Value,
    value: float,
) -> ir.Value:
    """scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor"""
    builder = current_builder()
    # Ensure value is a scalar tensor and expand it to match index shape
    if len(index.shape) == 0:
        index = builder.call("Unsqueeze", index, [0])[0]
    scalar_tensor = ir.tensor([value], dtype=self.dtype)
    src = builder.call(
        "ConstantOfShape", builder.call("Shape", index)[0], value=scalar_tensor
    )[0]
    return builder.call("ScatterElements", self, index, src, axis=dim)[0]


@onnx_impl(aten.scatter_add.default)
def aten_scatter_add(
    self: ir.Value,
    dim: int,  # we have to use int here because ScatterElements() will use this attribute
    index: ir.Value,
    src: ir.Value,
) -> ir.Value:
    """scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor"""
    builder = current_builder()

    # if rank(self) == 0 will lead ORT failed, skipped
    return builder.call("ScatterElements", self, index, src, axis=dim, reduction="add")[
        0
    ]


@onnx_impl(aten.scatter_reduce.two)
def aten_scatter_reduce(
    self: ir.Value,
    dim: int,  # we have to use int here because ScatterElements() will use this attribute
    index: ir.Value,
    src: ir.Value,
    reduce: str,
    include_self: bool = True,
) -> ir.Value:
    """scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor"""
    builder = current_builder()

    reduce_mode = {  # convert torch string name to onnx string name
        "mean": "none",  # 'mean' doesn't support in ONNX 1.14 definition
        "sum": "add",
        "prod": "mul",
        "amin": "min",
        "amax": "max",
    }
    if reduce not in reduce_mode:
        raise ValueError(
            f"Unknown scatter reduction {reduce!r}. Use sum, prod, amin, amax, or mean."
        )
    onnx_reduce = reduce_mode[reduce]
    dtype = src.dtype or self.dtype
    if dtype is None:
        raise ValueError(
            "scatter_reduce needs a known input dtype. Set the input IR tensor type."
        )

    self_is_scalar = len(self.shape) == 0
    index_rank = max(len(index.shape), 1)
    if self_is_scalar:  # assert (index_rank == 0 and rank_src == 0)
        neg_1 = builder.constant([-1], dtype=ir.DataType.INT64)
        self = builder.call("Reshape", self, neg_1)[0]
        index = builder.call("Reshape", index, neg_1)[0]
        src = builder.call("Reshape", src, neg_1)[0]

    src = builder.call("Slice", src, [0] * index_rank, builder.call("Shape", index)[0])[
        0
    ]
    if reduce == "mean":
        original = self
        zeros = builder.call(
            "ConstantOfShape",
            builder.call("Shape", self)[0],
            value=ir.tensor([0], dtype=dtype),
        )[0]
        ones = builder.call(
            "ConstantOfShape",
            builder.call("Shape", src)[0],
            value=ir.tensor([1], dtype=dtype),
        )[0]
        counts = builder.call("Add", zeros, 1)[0] if include_self else zeros
        if not include_self:
            updates_zero = builder.call("Sub", ones, ones)[0]
            self = builder.call("ScatterElements", self, index, updates_zero, axis=dim)[
                0
            ]
        total = builder.call(
            "ScatterElements", self, index, src, axis=dim, reduction="add"
        )[0]
        counts = builder.call(
            "ScatterElements", counts, index, ones, axis=dim, reduction="add"
        )[0]
        divisor = builder.call("Max", counts, 1)[0]
        mean = builder.call("Div", total, divisor)[0]
        if dtype.is_integer():
            remainder = builder.call("Mod", total, divisor, fmod=0)[0]
            round_down = builder.call(
                "And",
                builder.call("Less", total, 0)[0],
                builder.call("Not", builder.call("Equal", remainder, 0)[0])[0],
            )[0]
            mean = builder.call(
                "Sub", mean, builder.call("CastLike", round_down, mean)[0]
            )[0]
        result = builder.call(
            "Where", builder.call("Equal", counts, 0)[0], original, mean
        )[0]
        return builder.call("Squeeze", result)[0] if self_is_scalar else result

    if not include_self:
        # onnx standard always assume the value from self is part of the reduction.
        # A first step is added to replace the impacted value by another one
        # chosen in a way that the results of the reduction is not changed
        # whether or not it takes part in it.
        # It is -inf if the reduction is max, inf for min, 0 for add, 1 for mul.
        # Mean is handled separately with explicit sums and counts.
        if onnx_reduce == "max":
            if dtype in {
                ir.DataType.FLOAT16,
                ir.DataType.FLOAT,
                ir.DataType.DOUBLE,
            }:
                value = ir.tensor([float("-inf")], dtype=dtype)
            elif dtype == ir.DataType.BFLOAT16:
                value = ir.tensor([float("-inf")], dtype=dtype)
            elif dtype == ir.DataType.BOOL:
                value = ir.tensor([False], dtype=dtype)
            else:
                value = ir.tensor([np.iinfo(dtype.numpy()).min], dtype=dtype)
            reduction_init = "none"
        elif onnx_reduce == "min":
            if dtype in {
                ir.DataType.FLOAT16,
                ir.DataType.FLOAT,
                ir.DataType.DOUBLE,
            }:
                value = ir.tensor([float("inf")], dtype=dtype)
            elif dtype == ir.DataType.BFLOAT16:
                value = ir.tensor([float("inf")], dtype=dtype)
            elif dtype == ir.DataType.BOOL:
                value = ir.tensor([True], dtype=dtype)
            else:
                value = ir.tensor([np.iinfo(dtype.numpy()).max], dtype=dtype)
            reduction_init = "none"
        elif onnx_reduce == "add":
            value = ir.tensor([0], dtype=dtype)
            reduction_init = "none"
        elif onnx_reduce == "mul":
            value = ir.tensor([1], dtype=dtype)
            reduction_init = "none"
        else:
            value = ir.tensor([0], dtype=dtype)
            reduction_init = "none"

        cst = builder.call(
            "ConstantOfShape", builder.call("Shape", src)[0], value=value
        )[0]
        self = builder.call(
            "ScatterElements", self, index, cst, axis=dim, reduction=reduction_init
        )[0]

    result = builder.call(
        "ScatterElements", self, index, src, axis=dim, reduction=onnx_reduce
    )[0]

    if self_is_scalar:
        result = builder.call("Squeeze", result)[0]
    return result


@onnx_impl(aten.select.int)
def aten_select(self: ir.Value, dim: int, index: int) -> ir.Value:
    """select(Tensor self, int dim, int index) -> Tensor"""
    builder = current_builder()

    return builder.call("Gather", self, index, axis=dim)[0]


@onnx_impl(aten.select_scatter.default)
def aten_select_scatter(
    self: ir.Value, src: ir.Value, dim: int, index: int
) -> ir.Value:
    """select_scatter(Tensor self, Tensor src, int dim, int index) -> Tensor"""
    builder = current_builder()

    # Change src rank to self rank according to dim
    # e.g. if self is [2,3,4], src is [2,4], dim=1, then update is [2,1,4]
    update = builder.call("Unsqueeze", src, [dim])[0]
    # Change index rank to the same as 'update' [2,1,4]
    indices = builder.call("Expand", index, builder.call("Shape", update)[0])[0]
    return builder.call(
        "ScatterElements", self, indices, update, axis=dim, reduction="none"
    )[0]


@onnx_impl(aten.slice.Tensor, complex=True)
def aten_slice_complex(
    self: ir.Value,
    dim: int = 0,
    start: ir.Value | None = None,
    end: ir.Value | None = None,
    step: ir.Value | None = None,
) -> ir.Value:
    """slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)"""
    if dim < 0:
        # Account for the complex dimension in ONNX
        dim = len(self.shape) + dim - 1
    return aten_slice(self, dim, start, end, step)


@onnx_impl(aten.slice.Tensor)
def aten_slice(
    self: ir.Value,
    dim: int = 0,
    start: ir.Value | None = None,
    end: ir.Value | None = None,
    step: ir.Value | None = None,
) -> ir.Value:
    """slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)"""
    builder = current_builder()

    # TODO: using OptionalHasElement() to check start/end value
    if start is not None:
        start = builder.call("Cast", start, to=ir.DataType.INT64)[0]
        start = builder.call(
            "Reshape", start, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]
    else:
        start = builder.constant([0], dtype=ir.DataType.INT64)

    if end is not None:
        end = builder.call("Cast", end, to=ir.DataType.INT64)[0]
        end = builder.call(
            "Reshape", end, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]
    else:
        end = builder.constant([_INT64_MAX], dtype=ir.DataType.INT64)

    dim = builder.call("Cast", dim, to=ir.DataType.INT64)[0]
    dim = builder.call("Reshape", dim, builder.constant([-1], dtype=ir.DataType.INT64))[
        0
    ]

    if step is not None:
        step = builder.call("Cast", step, to=ir.DataType.INT64)[0]
        step = builder.call(
            "Reshape", step, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]
    else:
        step = builder.constant([1], dtype=ir.DataType.INT64)

    return builder.call("Slice", self, start, end, dim, step)[0]


@onnx_impl(aten.slice_scatter.default)
def aten_slice_scatter(
    self: ir.Value,
    src: ir.Value,
    dim: int = 0,
    start: ir.Value | None = None,
    end: ir.Value | None = None,
    step: ir.Value = 1,
) -> ir.Value:
    """slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor"""
    builder = current_builder()

    # 'start' and 'end' are optional (aten schema: SymInt? start=None, SymInt? end=None).
    # When absent, default start to 0 and end to _INT64_MAX (a full slice), mirroring aten_slice.
    # Try torch sample to get more information:
    # https://pytorch.org/docs/master/generated/torch.slice_scatter.html?highlight=slice_scatter#torch.slice_scatter
    # Take (torch.zeros(8, 8), torch.ones(2, 8), 0, 6, 64, 1) as example:
    # Step 1: get 1D tensor from 0 to dim_size-1, then Slice it using start, end and step.
    # We cannot use Range(start, end, step) directly as start or end may out of range.
    # For the example, the output of this step is Slice([0, ..., 7], 6, 64, 1) = [6, 7]

    # Scatter ND
    zero = builder.constant([0], dtype=ir.DataType.INT64)
    self_shape = builder.call("Shape", self)[0]
    dim_shape = builder.call("Gather", self_shape, dim, axis=0)[0]
    index_base = builder.call("Range", 0, dim_shape, 1)[0]
    start_index = zero if start is None else builder.call("Unsqueeze", start, zero)[0]
    end_index = (
        builder.constant([_INT64_MAX], dtype=ir.DataType.INT64)
        if end is None
        else builder.call("Unsqueeze", end, zero)[0]
    )
    index_base = builder.call(
        "Slice",
        index_base,
        start_index,
        end_index,
        zero,
        builder.call("Unsqueeze", step, zero)[0],
    )[0]
    index_base = builder.call("Unsqueeze", index_base, [-1])[0]

    # Use trace only to construct the perm attribute in Transpose
    dims = None
    if dim != 0:
        src_rank = len(src.shape)  # type: ignore[attr-defined]

        if src_rank != 0:
            # Python code, change when onnxscript supports this
            dims = list(range(src_rank))
            dims[0], dims[dim] = dims[dim], dims[0]
            # Python code ends

            src = builder.call("Transpose", src, perm=dims)[0]
            self = builder.call("Transpose", self, perm=dims)[0]

    output = builder.call("ScatterND", self, index_base, src)[0]
    if dims is not None:
        output = builder.call("Transpose", output, perm=dims)[0]
    return output


@onnx_impl(aten.sort.default)
def aten_sort(
    self: ir.Value, dim: int = -1, descending: bool = False, stable: bool = False
) -> tuple[ir.Value, ir.Value]:
    """sort(Tensor self, int dim=-1, bool descending=False, bool stable=False) -> (Tensor values, Tensor indices)"""
    builder = current_builder()

    self_is_scalar = len(self.shape) == 0
    if self_is_scalar:
        return builder.call("Identity", self)[0], builder.constant(
            0, dtype=ir.DataType.INT64
        )
    shape = builder.call("Shape", self)[0]
    dim_size = builder.call("Gather", shape, dim, axis=0)[0]
    dim_size = builder.call(
        "Reshape", dim_size, builder.constant([1], dtype=ir.DataType.INT64)
    )[0]
    values, indices = builder.call(
        "TopK", self, dim_size, axis=dim, largest=descending, sorted=True
    )
    return values, indices


@onnx_impl(aten.squeeze.default)
def aten_squeeze(self: ir.Value) -> ir.Value:
    """squeeze(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    return builder.call("Squeeze", self)[0]


@onnx_impl(aten.squeeze.dim)
def aten_squeeze_dim(self: ir.Value, dim: int) -> ir.Value:
    builder = current_builder()
    if len(self.shape) == 0:
        return builder.call("Identity", self)[0]
    return builder.call("Squeeze", self, [dim])[0]


@onnx_impl(aten.squeeze.dim, complex=True)
def aten_squeeze_dim_complex(self: ir.Value, dim: int) -> ir.Value:
    builder = current_builder()
    if dim < 0:
        # Account for the complex dimension in ONNX
        dim = dim - 1

    if len(self.shape) == 1:
        # The single dimension is the complex dimension
        return builder.call("Identity", self)[0]
    return aten_squeeze_dim(self, dim)


@onnx_impl(aten.sum.default)
def aten_sum(self: ir.Value, dtype: int = -1) -> ir.Value:
    """sum(Tensor self, *, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()
    self_is_scalar = len(self.shape) == 0
    if dtype != -1 and dtype is not None:
        self = builder.call("Cast", self, to=dtype)[0]
    elif self.dtype.is_integer() or self.dtype == ir.DataType.BOOL:
        self = builder.call("Cast", self, to=ir.DataType.INT64)[0]
    if self_is_scalar:
        result = builder.call("Identity", self)[0]
    else:
        result = builder.call("ReduceSum", self, keepdims=False)[0]
    if dtype != -1 and dtype is not None:
        result = builder.call("Cast", result, to=dtype)[0]
    return result


@onnx_impl(aten.sum.default, complex=True)
def aten_sum_complex(self: ir.Value, dtype: int = -1) -> ir.Value:
    """sum(Tensor self, *, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()
    if len(self.shape) == 1:
        result = builder.call("Identity", self)[0]
    else:
        rank = len(self.shape) - 1
        dim = builder.constant(list(range(rank)), dtype=ir.DataType.INT64)
        result = builder.call("ReduceSum", self, dim, keepdims=False)[0]
    if dtype != -1 and dtype is not None:
        raise NotImplementedError(
            "support for the dtype argument is not implemented for complex tensors"
        )
    return result


@onnx_impl(aten.sum.dim_IntList)
def aten_sum_dim_IntList(
    self: ir.Value, dim: int | None = None, keepdim: bool = False, dtype: int = -1
) -> ir.Value:
    """sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()
    self_is_scalar = len(self.shape) == 0
    if dtype != -1 and dtype is not None:
        self = builder.call("Cast", self, to=dtype)[0]
    elif self.dtype.is_integer() or self.dtype == ir.DataType.BOOL:
        self = builder.call("Cast", self, to=ir.DataType.INT64)[0]
    if self_is_scalar:
        result = builder.call("Identity", self)[0]
    elif dim is None or isinstance(dim, Sequence) and len(dim) == 0:
        result = builder.call("ReduceSum", self, keepdims=keepdim)[0]
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = common_ops.constant(dim, dtype=ir.DataType.INT64)
        result = builder.call("ReduceSum", self, dim, keepdims=keepdim)[0]

    if dtype != -1 and dtype is not None:
        result = builder.call("Cast", result, to=dtype)[0]

    return result


@onnx_impl(aten.sum.dim_IntList, complex=True)
def aten_sum_dim_IntList_complex(
    self: ir.Value, dim: int | None = None, keepdim: bool = False, dtype: int = -1
) -> ir.Value:
    """sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()
    if len(self.shape) == 1:
        result = builder.call("Identity", self)[0]
    elif dim is None or isinstance(dim, Sequence) and len(dim) == 0:
        rank = len(self.shape) - 1
        dim = builder.constant(list(range(rank)), dtype=ir.DataType.INT64)
        result = builder.call("ReduceSum", self, dim, keepdims=keepdim)[0]
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = [d - 1 if d < 0 else d for d in dim]
        result = builder.call("ReduceSum", self, dim, keepdims=keepdim)[0]

    if dtype != -1 and dtype is not None:
        raise NotImplementedError

    return result


@onnx_impl(aten.sym_size.int)
def aten_sym_size(self: ir.Value, dim: int = 0) -> ir.Value:
    """sym_size.int(Tensor self, int dim) -> SymInt"""
    builder = current_builder()
    return builder.call("Gather", builder.call("Shape", self)[0], dim, axis=0)[0]


@onnx_impl(aten.sym_storage_offset.default)
def aten_sym_storage_offset(self: ir.Value, dim: int = 0) -> ir.Value:
    """sym_storage_offset(Tensor self, int dim) -> SymInt"""
    builder = current_builder()
    # storage offset is not used in onnx world.
    # the output of this function is not used.
    return builder.constant(0, dtype=ir.DataType.INT64)


@onnx_impl(aten.t.default)
def aten_t(self: ir.Value) -> ir.Value:
    """t(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    rank = len(self.shape)
    if rank == 2:
        result = builder.call("Transpose", self, perm=[1, 0])[0]
    else:
        # rank < 2
        result = self
    return result


@onnx_impl(aten.t.default, complex=True)
def aten_t_complex(self: ir.Value) -> ir.Value:
    """t(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    rank = len(self.shape) - 1
    if rank == 2:
        result = builder.call("Transpose", self, perm=[1, 0, 2])[0]
    else:
        # rank < 2
        result = builder.call("Identity", self)[0]
    return result


@onnx_impl(aten.topk.default)
def aten_topk(
    self: ir.Value, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
) -> tuple[ir.Value, ir.Value]:
    """topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)"""
    builder = current_builder()

    # We do not handle scalar inputs for topk
    values, indices = builder.call(
        "TopK", self, [k], axis=dim, largest=largest, sorted=sorted
    )
    return values, indices


@onnx_impl(aten.transpose.int)
def aten_transpose(self: ir.Value, dim0: int, dim1: int) -> ir.Value:
    """transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)"""
    builder = current_builder()

    # Use trace only to construct the prem attribute in Transpose
    self_rank = len(self.shape)  # type: ignore[attr-defined]

    if self_rank == 0:
        result = self
    else:
        # Python code, change when onnxscript supports this
        dims = list(range(self_rank))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        # Python code ends

        result = builder.call("Transpose", self, perm=dims)[0]

    return result


@onnx_impl(aten.transpose.int, complex=True)
def aten_transpose_complex(self: ir.Value, dim0: int, dim1: int) -> ir.Value:
    """transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)"""
    builder = current_builder()

    # Use trace only to construct the prem attribute in Transpose
    self_rank = len(self.shape)  # type: ignore[attr-defined]

    if self_rank == 0:
        result = self
    else:
        # Python code, change when onnxscript supports this
        # Handle when dim0 or dim1 is negative. ONNX uses the last axis to
        # represent to complex axis so we need to move the dim one axis toward the start.
        if dim0 < 0:
            dim0 = dim0 - 1
        if dim1 < 0:
            dim1 = dim1 - 1
        dims = list(range(self_rank))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        # Python code ends

        result = builder.call("Transpose", self, perm=dims)[0]

    return result


@onnx_impl(aten.unflatten.int)
def aten_unflatten(
    self: ir.Value, dim: int, sizes: Sequence[int | ir.Value]
) -> ir.Value:
    """unflatten(Tensor(a) self, int dim, SymInt[] sizes) -> Tensor(a)"""
    builder = current_builder()

    self_size = builder.call("Shape", self)[0]

    # PyTorch accepts negative dim as reversed counting
    self_rank = len(self.shape)
    if dim < 0:
        dim = self_rank + dim

    head_start_idx = builder.constant([0], dtype=ir.DataType.INT64)
    head_end_idx = builder.call(
        "Reshape", dim, builder.constant([1], dtype=ir.DataType.INT64)
    )[0]
    head_part_rank = builder.call("Slice", self_size, head_start_idx, head_end_idx)[0]

    tail_start_idx = builder.call(
        "Reshape", dim + 1, builder.constant([1], dtype=ir.DataType.INT64)
    )[0]
    tail_end_idx = builder.constant([_INT64_MAX], dtype=ir.DataType.INT64)
    tail_part_rank = builder.call("Slice", self_size, tail_start_idx, tail_end_idx)[0]

    sizes = [
        builder.call("Reshape", size, builder.constant([1], dtype=ir.DataType.INT64))[0]
        for size in sizes
    ]

    # corner case 1: head part is None
    if dim == 0:
        final_shape = builder.call("Concat", *sizes, tail_part_rank, axis=0)[0]
    # corner case 2: tail part is None
    elif dim == self_rank - 1:
        final_shape = builder.call("Concat", head_part_rank, *sizes, axis=0)[0]
    else:
        final_shape = builder.call(
            "Concat", head_part_rank, *sizes, tail_part_rank, axis=0
        )[0]
    return builder.call("Reshape", self, final_shape, allowzero=True)[0]


@onnx_impl(aten.unfold.default)
def aten_unfold(self: ir.Value, dimension: int, size: int, step: int) -> ir.Value:
    """unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)"""
    builder = current_builder()

    self_rank = len(self.shape)
    if self_rank == 0:
        result = builder.call("Unsqueeze", self, [0])[0]
    else:
        # Handle negative dimension
        if dimension < 0:
            dimension = dimension + self_rank

        input_shape = builder.call("Shape", self)[0]
        # Range requires rank 0 (scalar) inputs, so squeeze the [1] shaped
        # Gather result down to a scalar.
        dim_size = builder.call(
            "Squeeze",
            builder.call(
                "Gather",
                input_shape,
                builder.constant([dimension], dtype=ir.DataType.INT64),
            )[0],
            builder.constant([0], dtype=ir.DataType.INT64),
        )[0]

        # Create indices for each window
        window_starts = builder.call(
            "Range", 0, builder.call("Sub", dim_size, size - 1)[0], step
        )[0]

        # Create the base indices for one window
        window_indices = list(range(size))

        # Broadcast to create all indices
        starts_expanded = builder.call("Unsqueeze", window_starts, [1])[
            0
        ]  # [num_windows, 1]
        indices_expanded = builder.call("Unsqueeze", window_indices, [0])[
            0
        ]  # [1, size]
        all_indices = builder.call("Add", starts_expanded, indices_expanded)[
            0
        ]  # [num_windows, size]

        # Gather along the specified dimension
        result = builder.call("Gather", self, all_indices, axis=dimension)[0]

        # The result shape is now [..., num_windows, size, ...] with num_windows at position 'dimension'.
        # We need to move the size dimension to the end:
        #   Current shape: [..., num_windows, size, ...]
        #   Target shape:  [..., num_windows, ..., size]

        # Move the size dimension (at position dimension+1) to the end
        # perm need to be list[int], so have to be generated in trace_only mode
        perm = list(range(self_rank + 1))
        perm.append(perm.pop(dimension + 1))

        result = builder.call("Transpose", result, perm=perm)[0]

    return result


@onnx_impl(aten.unsqueeze.default)
def aten_unsqueeze(self: ir.Value, dim: int) -> ir.Value:
    """unsqueeze(Tensor(a) self, int dim) -> Tensor(a)"""
    builder = current_builder()

    return builder.call("Unsqueeze", self, [dim])[0]


@onnx_impl(aten.unsqueeze.default, complex=True)
def aten_unsqueeze_complex(self: ir.Value, dim: int) -> ir.Value:
    """unsqueeze(Tensor(a) self, int dim) -> Tensor(a)"""
    builder = current_builder()

    if dim < 0:
        # Account for the complex dimension in ONNX
        dim = dim - 1
    return builder.call("Unsqueeze", self, [dim])[0]


@onnx_impl((aten.view.default, aten._unsafe_view.default))
def aten_view(self: ir.Value, size: Sequence[int | ir.Value]) -> ir.Value:
    """view(Tensor(a) self, SymInt[] size) -> Tensor(a)"""
    builder = current_builder()

    size = common_ops.merge_dims(size)
    return builder.call("Reshape", self, size, allowzero=True)[0]


@onnx_impl((aten.view.default, aten._unsafe_view.default), complex=True)
def aten_view_complex(self: ir.Value, size: Sequence[int | ir.Value]) -> ir.Value:
    """view(Tensor(a) self, SymInt[] size) -> Tensor(a)"""
    builder = current_builder()

    complex_size = common_ops.merge_dims([*size, 2])
    return builder.call("Reshape", self, complex_size, allowzero=True)[0]


@onnx_impl(aten.view_as.default)
def aten_view_as(self: ir.Value, other: ir.Value) -> ir.Value:
    """view_as(Tensor(a) self, Tensor other) -> Tensor(a)"""
    builder = current_builder()

    size = builder.call("Shape", other)[0]
    return builder.call("Reshape", self, size, allowzero=True)[0]


@onnx_impl(aten.view_as_complex.default)
def aten_view_as_complex(self: ir.Value) -> ir.Value:
    """view_as_complex(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    # We always operate on the real representation of a complex number in torchlib
    # So this is a no-op
    return builder.call("Identity", self)[0]


@onnx_impl(aten.view_as_complex_copy.default)
def aten_view_as_complex_copy(self: ir.Value) -> ir.Value:
    """view_as_complex_copy(Tensor self) -> Tensor"""
    builder = current_builder()

    # We always operate on the real representation of a complex number in torchlib
    # So this is a no-op
    return builder.call("Identity", self)[0]


@onnx_impl(aten.view_as_real.default, complex=True)
def aten_view_as_real(self: ir.Value) -> ir.Value:
    """view_as_real(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    # We always operate on the real representation of a complex number in torchlib
    # So this is a no-op
    return builder.call("Identity", self)[0]


@onnx_impl(aten.view_as_real_copy.default, complex=True)
def aten_view_as_real_copy(self: ir.Value) -> ir.Value:
    """view_as_real_copy(Tensor self) -> Tensor"""
    builder = current_builder()

    # We always operate on the real representation of a complex number in torchlib
    # So this is a no-op
    return builder.call("Identity", self)[0]


@onnx_impl(aten.view_copy.default)
def aten_view_copy(self: ir.Value, size: Sequence[int | ir.Value]) -> ir.Value:
    """view_copy(Tensor self, SymInt[] size) -> Tensor"""
    builder = current_builder()

    size = common_ops.merge_dims(size)
    return builder.call("Reshape", self, size)[0]


@onnx_impl(
    (aten.where.Scalar, aten.where.ScalarSelf, aten.where.ScalarOther, aten.where.self)
)
def aten_where(condition: ir.Value, self: ir.Value, other: ir.Value) -> ir.Value:
    """where.self(Tensor condition, Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("Where", condition, self, other)[0]


@onnx_impl(aten.as_strided.default)
def aten_as_strided(
    self: ir.Value,
    size: Sequence[int | ir.Value],
    stride: Sequence[int | ir.Value],
    storage_offset: ir.Value | None = None,
) -> ir.Value:
    """as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)"""
    builder = current_builder()

    # torch.as_strided produces a view of `self`'s underlying linear storage
    # with the requested `size` (the output shape) and `stride` (the step, in
    # elements of storage, taken along each output dimension), starting at
    # `storage_offset` elements into the storage. For an output element at
    # position (i_0, ..., i_{n-1}) the element read from storage lives at the flat
    # index storage_offset + sum_d i_d * stride[d]. So if we flatten `self` to 1-D
    # and gather it with a tensor of those flat indices shaped like the output, we
    # reproduce the view as a single Gather. This avoids the hard-to-fold loop of
    # the previous implementation.
    rank = len(size)
    # ONNX exposes logical values, not backing storage. Flattening is correct only
    # when logical row-major order matches storage order; non-contiguous views remain
    # unsupported, as they were in the previous lowering.
    self_flatten = builder.call(
        "Reshape", self, builder.constant([-1], dtype=ir.DataType.INT64)
    )[0]

    # A missing storage_offset means "start at the beginning of the storage".
    if storage_offset is None:
        storage_offset = 0

    static_inputs = (
        all(isinstance(s, int) for s in size)
        and all(isinstance(s, int) for s in stride)
        and isinstance(storage_offset, int)
    )
    static_index_count = math.prod(size) if static_inputs else None
    if (
        static_index_count is not None
        and static_index_count <= _AS_STRIDED_STATIC_INDEX_SIZE_LIMIT
    ):
        # Static fast path: every size/stride/offset is known at trace time, so we
        # compute the full index tensor with NumPy and emit it as a single
        # constant that downstream passes can fold trivially.
        # Start from the storage_offset; the per-dimension contributions are added in.
        indices = np.array(storage_offset, dtype=np.int64)
        for dim, (dim_size, dim_stride) in enumerate(zip(size, stride)):
            # Contribution of dimension `dim`: index i_dim contributes i_dim * stride[dim].
            add_value = np.arange(dim_size, dtype=np.int64) * dim_stride
            # Reshape that 1-D contribution so it broadcasts along `dim` only
            # (length dim_size at position `dim`, length 1 everywhere else), which
            # lets the running sum build the full n-D index grid.
            broadcast_shape = [1] * rank
            broadcast_shape[dim] = dim_size
            indices = indices + add_value.reshape(broadcast_shape)
        indices = builder.call("Constant", value=ir.tensor(indices))[0]
    else:
        # Build runtime indices when a SymInt is dynamic or a static index tensor
        # would exceed the optimizer's default folded-output size limit. The loop
        # is unrolled at trace time because rank is static, so no Loop/Scan is emitted.
        zero = builder.constant(0, dtype=ir.DataType.INT64)
        one = builder.constant(1, dtype=ir.DataType.INT64)
        # `scalar_shape` reshapes a value to a 0-D scalar (shape []).
        scalar_shape = builder.call(
            "Constant", value=ir.tensor(np.array([], dtype=np.int64))
        )[0]
        # Start the running index from storage_offset as an INT64 scalar; SymInt
        # runtime values are assumed to be INT64.
        indices = builder.call("Reshape", storage_offset, scalar_shape)[0]
        for dim in range(rank):
            # Reshape this dimension's size and stride to INT64 scalars.
            dim_size = builder.call("Reshape", size[dim], scalar_shape)[0]
            dim_stride = builder.call("Reshape", stride[dim], scalar_shape)[0]
            # add_value = arange(dim_size) * dim_stride, a 1-D tensor of length dim_size
            # holding the storage offsets contributed by index 0..dim_size-1 along `dim`.
            add_value = builder.call(
                "Mul", builder.call("Range", zero, dim_size, one)[0], dim_stride
            )[0]
            # Insert singleton axes everywhere except `dim` so this 1-D contribution
            # broadcasts along dimension `dim` only when added to the running index,
            # matching the NumPy `reshape(broadcast_shape)` in the static path.
            unsqueeze_axes = [axis for axis in range(rank) if axis != dim]
            if unsqueeze_axes:
                add_value = builder.call(
                    "Unsqueeze",
                    add_value,
                    builder.constant(unsqueeze_axes, dtype=ir.DataType.INT64),
                )[0]
            indices = builder.call("Add", indices, add_value)[0]

    # `indices` now has shape `size`; gathering yields the strided view.
    return builder.call("Gather", self_flatten, indices)[0]


@onnx_impl((aten.diagonal.default, aten.diagonal_copy.default))
def aten_diagonal(
    self: ir.Value, offset: int = 0, dim1: int = 0, dim2: int = 1
) -> ir.Value:
    """diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)"""
    builder = current_builder()

    is_bool = self.dtype == ir.DataType.BOOL

    # perm is used to transpose the tensor to make dim1 and dim2 as the last 2 dims
    # [0,1,2] -> [2,0,1] when dim1=0 and dim2=1
    # [0,1,2] -> [1,0,2] when dim1=0 and dim2=2
    # [0,1,2] -> [0,1,2] when dim1=1 and dim2=2
    if dim1 < 0:
        dim1 = dim1 + len(self.shape)
    if dim2 < 0:
        dim2 = dim2 + len(self.shape)

    self_rank = len(self.shape)
    perm = list(range(self_rank))
    perm.remove(dim1)
    perm.remove(dim2)
    perm.append(dim1)
    perm.append(dim2)

    # If rank=2, then axes=[0]; if rank=3, then axes=[1]
    # This is because computing diagonal sum is on dim2 after transpose by perm
    axes = [self_rank - 2]

    neg_1 = builder.constant([-1], dtype=ir.DataType.INT64)
    dim1_size = builder.call(
        "Reshape",
        builder.call("Gather", builder.call("Shape", self)[0], dim1)[0],
        neg_1,
    )[0]  # row
    dim2_size = builder.call(
        "Reshape",
        builder.call("Gather", builder.call("Shape", self)[0], dim2)[0],
        neg_1,
    )[0]  # col
    mask_shape = builder.call("Concat", dim1_size, dim2_size, axis=0)[0]
    mask = builder.call(
        "EyeLike",
        builder.call("ConstantOfShape", mask_shape)[0],
        k=offset,
        _output_types=[ir.TensorType(ir.DataType.FLOAT)],
    )[0]

    if is_bool:
        self_int = builder.call("Cast", self, to=ir.DataType.INT64)[0]
        mask_int = builder.call("Cast", mask, to=ir.DataType.INT64)[0]
        self_int_t = builder.call("Transpose", self_int, perm=perm)[0]
        result = builder.call("Mul", self_int_t, mask_int)[0]
    else:
        mask = builder.call("CastLike", mask, self)[0]
        self_t = builder.call("Transpose", self, perm=perm)[0]
        result = builder.call("Mul", self_t, mask)[0]
    result = builder.call("ReduceSum", result, axes, keepdims=False)[0]
    # min(row, col)
    min_dim_size = builder.call("Min", dim1_size, dim2_size)[0]
    # take 2 tensors as example:
    # one is 3x5 in size, min_dim_size = 3, dim1_size = 3
    # the other is 5x3 in size, min_dim_size = 3, dim1_size = 5
    # 3 rows x 5 cols     5 rows x 3 cols
    # offset  diagonal    offset  diagonal
    # ----------------    ----------------
    # -4      0           -6      0
    # -3      0           -5      0
    # -2      1           -4      1
    # -1      2           -3      2
    # 0       3           -2      3
    # 1       3           -1      3
    # 2       3           0       3
    # 3       2           1       2
    # 4       1           2       1
    # 5       0           3       0
    # 6       0           4       0

    # From above table, we can get the logic below
    offset_val = builder.constant([offset], dtype=ir.DataType.INT64)
    if offset < 0:
        # row + offset
        length = builder.call("Add", dim1_size, offset_val)[0]
        start = builder.constant([0], dtype=ir.DataType.INT64)
    else:  # offset >= 0
        # col - offset
        length = builder.call("Sub", dim2_size, offset_val)[0]
        start = offset_val

    # max(min(length, min(row, col)), 0)
    length = builder.call(
        "Max",
        builder.call("Min", length, min_dim_size)[0],
        builder.constant([0], dtype=ir.DataType.INT64),
    )[0]
    end = builder.call("Add", start, length)[0]
    result = builder.call("Slice", result, start, end, axes)[0]

    if is_bool:
        result = builder.call("Cast", result, to=ir.DataType.BOOL)[0]

    return result


@onnx_impl((aten.index_put.default, aten._unsafe_index_put.default))
def aten_index_put(
    self: ir.Value,
    indices: Sequence[ir.Value | None],
    values: ir.Value,
    accumulate: bool = False,
) -> ir.Value:
    """index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor

    See implementation of `torch.onnx.symbolic_opset11.index_put
    <https://github.com/pytorch/pytorch/blob/main/torch/onnx/symbolic_opset11.py#L212>`_.
    """
    builder = current_builder()
    if any(index is not None and index.dtype == ir.DataType.BOOL for index in indices):
        return _aten_index_put_bool(self, indices, values, accumulate)

    # Ensure the number of indices matches the tensor rank by appending trailing Nones.
    self_rank = len(self.shape)
    if len(indices) < self_rank:
        indices = list(indices) + [None] * (self_rank - len(indices))

    # The behavior of the op is dependent on whether there are advanced indices (i.e., non-scalar tensors)
    # and whether these advanced indices are contiguous.

    # Identify advanced indices.
    def is_advanced_index(index):
        # Note: In this function, the index is assumed to be either None or an int64 Tensor.
        return index is not None

    advanced_indices: list[int] = []
    none_indices: list[int] = []
    num_advanced_indices = 0
    num_none_indices = 0

    for i, index in enumerate(indices):
        if is_advanced_index(index):
            advanced_indices.append(i)
            num_advanced_indices += 1
        elif index is None:
            none_indices.append(i)
            num_none_indices += 1
        else:
            raise ValueError(f"Unhandled index at position {i}: {index}")

    self_shape = builder.call("Shape", self)[0]
    if num_advanced_indices == 0:
        return builder.call("Expand", values, self_shape)[0]

    # More than one advanced index may require broadcasting of index values
    if num_advanced_indices > 1:
        # Check for special case where all advanced indices have same shape.
        # But need to ensure none of the shapes have None as a dimension, which
        # will invalidate equality-based check.
        first_shape = indices[advanced_indices[0]].shape

        def same_shape(other_shape: ir.Shape | None) -> bool:
            return (
                first_shape is not None
                and other_shape is not None
                and not any(d is None for d in first_shape)
                and not any(d is None for d in other_shape)
                and other_shape == first_shape
            )

        all_same_shape = all(same_shape(indices[i].shape) for i in advanced_indices)
        if not all_same_shape:
            # Broadcast advanced indices to a common shape.
            advanced_index_rank = max(len(indices[i].shape) for i in advanced_indices)
            shapes = []
            for i in advanced_indices:
                index = indices[i]
                index_rank = len(index.shape)
                index_shape = builder.call("Shape", index)[0]
                if index_rank < advanced_index_rank:
                    padding = builder.constant(
                        [1 for _ in range(advanced_index_rank - index_rank)],
                        dtype=ir.DataType.INT64,
                    )
                    index_shape = builder.call("Concat", padding, index_shape, axis=0)[
                        0
                    ]
                shapes.append(index_shape)
            advanced_indices_shape = builder.call("Max", *shapes)[0]
            indices = [
                builder.call("Expand", index, advanced_indices_shape)[0]
                if is_advanced_index(index)
                else index
                for index in indices
            ]
        else:
            advanced_indices_shape = builder.call(
                "Shape", indices[advanced_indices[0]]
            )[0]
            advanced_index_rank = len(indices[advanced_indices[0]].shape)
    else:
        advanced_indices_shape = builder.call("Shape", indices[advanced_indices[0]])[0]
        advanced_index_rank = len(indices[advanced_indices[0]].shape)

    # ONNX ScatterND supports only the case where all advanced indices appear first,
    # followed by None indices. So, we need to transpose self and values so that the
    # advanced indices appear first, and then transpose the result back to original
    # order at the end.

    none_indices_constant = builder.constant(none_indices, dtype=ir.DataType.INT64)
    none_indices_shape = builder.call(
        "Gather", self_shape, none_indices_constant, axis=0
    )[0]
    target_shape = builder.call(
        "Concat", advanced_indices_shape, none_indices_shape, axis=0
    )[0]
    target_rank = advanced_index_rank + num_none_indices

    # Generate indices tensor required by ONNX ScatterND by unsqueezing an extra dimension and
    # concatenating all advanced indices along this new dimension.
    minus_one = builder.constant([-1], dtype=ir.DataType.INT64)
    advanced_index_values = [
        builder.call("Unsqueeze", indices[i], minus_one)[0] for i in advanced_indices
    ]
    onnx_index = builder.call("Concat", *advanced_index_values, axis=-1)[0]

    # Check if advanced indices are contiguous:
    contiguous = True
    if advanced_indices:
        if advanced_indices[-1] - advanced_indices[0] + 1 != len(advanced_indices):
            contiguous = False

    # Bring advanced indices to front:
    perm = advanced_indices + none_indices
    transposed = builder.call("Transpose", self, perm=perm)[0]

    # Expand values to match target shape:
    # First, transpose values if necessary to match advanced indices order!
    if contiguous:
        # values may need to be transposed before expanding to target shape
        num_padded_dims = target_rank - len(values.shape)
        if num_padded_dims > 0:
            unsqueezed_dims = builder.constant(
                list(range(num_padded_dims)), dtype=ir.DataType.INT64
            )
            values = builder.call("Unsqueeze", values, unsqueezed_dims)[0]
        initial_none_index_positions = list(range(advanced_indices[0]))
        advanced_index_replacement_positions = list(
            range(advanced_indices[0], advanced_indices[0] + advanced_index_rank)
        )
        final_none_index_positions = list(
            range(advanced_indices[0] + advanced_index_rank, target_rank)
        )
        values_perm = (
            advanced_index_replacement_positions
            + initial_none_index_positions
            + final_none_index_positions
        )
        values = builder.call("Transpose", values, perm=values_perm)[0]

    expanded_values = builder.call("Expand", values, target_shape)[0]

    updated = builder.call(
        "ScatterND",
        transposed,
        onnx_index,
        expanded_values,
        reduction="add" if accumulate else None,
    )[0]

    # Inverse transpose to restore original dimension order:

    inverse_perm = [0] * self_rank
    for i, p in enumerate(perm):
        inverse_perm[p] = i
    result = builder.call("Transpose", updated, perm=inverse_perm)[0]
    return result


def _aten_index_put_bool(
    self: ir.Value,
    indices: Sequence[ir.Value | None],
    values: ir.Value,
    accumulate: bool = False,
) -> ir.Value:
    """index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"""
    builder = current_builder()

    bool_mask = indices[0]
    if len(indices) > 1:
        if any(index is None for index in indices):
            raise NotImplementedError(
                "Boolean index_put with multiple indices does not support None indices."
            )

        advanced_indices = []
        selected_positions = []
        minus_one = builder.constant([-1], dtype=ir.DataType.INT64)
        for index in indices:
            if index.dtype != ir.DataType.BOOL or len(index.shape) != 1:
                raise NotImplementedError(
                    "Boolean index_put with multiple indices supports only 1-D boolean masks."
                )
            positions = builder.call(
                "Reshape",
                builder.call(
                    "Transpose", builder.call("NonZero", index)[0], perm=[1, 0]
                )[0],
                minus_one,
            )[0]
            selected_positions.append(positions)
            advanced_indices.append(builder.call("Unsqueeze", positions, minus_one)[0])
        onnx_index = builder.call("Concat", *advanced_indices, axis=-1)[0]
        target_shape = builder.call(
            "Concat",
            builder.call("Shape", selected_positions[0])[0],
            builder.call(
                "Slice",
                builder.call("Shape", self)[0],
                [len(indices)],
                [len(self.shape)],
                [0],
            )[0],
            axis=0,
        )[0]
        expanded_values = builder.call("Expand", values, target_shape)[0]
        return builder.call(
            "ScatterND",
            self,
            onnx_index,
            expanded_values,
            reduction="add" if accumulate else None,
        )[0]

    if bool_mask is None or bool_mask.dtype != ir.DataType.BOOL:
        raise NotImplementedError(
            "Boolean index_put expects a boolean mask as the first index."
        )

    neg_1 = builder.constant([-1], dtype=ir.DataType.INT64)
    self_rank = len(self.shape)
    mask_rank = len(bool_mask.shape)

    # Expand a lower-rank mask (e.g. a row mask) across the trailing dimensions of self
    # so it selects whole slices, then collect the coordinates of every selected element.
    # NonZero returns them in row-major order.
    expanded_mask = bool_mask
    for _ in range(self_rank - mask_rank):
        expanded_mask = builder.call("Unsqueeze", expanded_mask, neg_1)[0]
    expanded_mask = builder.call(
        "Expand", expanded_mask, builder.call("Shape", self)[0]
    )[0]
    selected_indices = builder.call(
        "Transpose", builder.call("NonZero", expanded_mask)[0], perm=[1, 0]
    )[0]

    # Broadcast ``values`` to the selection shape ``[num_true, *self.shape[mask_rank:]]``
    # and flatten it to one update per selected element. This keeps scalar and
    # broadcastable ``values`` working, matching ``self[mask] = values`` semantics.
    num_true = builder.call(
        "ReduceSum",
        builder.call(
            "Cast", builder.call("Reshape", bool_mask, neg_1)[0], to=ir.DataType.INT64
        )[0],
        keepdims=1,
    )[0]
    trailing_shape = builder.call(
        "Slice", builder.call("Shape", self)[0], [mask_rank], [self_rank], [0]
    )[0]
    selection_shape = builder.call("Concat", num_true, trailing_shape, axis=0)[0]
    flat_values = builder.call(
        "Reshape", builder.call("Expand", values, selection_shape)[0], neg_1
    )[0]

    return builder.call(
        "ScatterND",
        self,
        selected_indices,
        flat_values,
        reduction="add" if accumulate else None,
    )[0]


def aten_std(self: ir.Value, unbiased: bool = True) -> ir.Value:
    """std(Tensor self, bool unbiased=True) -> Tensor"""
    builder = current_builder()
    var = _aten_var_onnx(self, correction=float(unbiased), keepdim=False)
    return builder.call("Sqrt", var)[0]


def aten_std_dim(
    self: ir.Value,
    dim: Sequence[int],
    unbiased: bool | None = True,
    keepdim: bool | None = False,
) -> ir.Value:
    """std.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor"""
    builder = current_builder()

    var = _aten_var_dim_onnx(
        self, dims=dim, correction=float(unbiased), keepdim=keepdim
    )
    return builder.call("Sqrt", var)[0]


def aten_std_correction(
    self: ir.Value,
    # FIXME(justinchuby): Make dim Optional[Sequence[int]]
    dim: int | None = None,
    correction: float | None = None,
    keepdim: bool = False,
) -> ir.Value:
    """std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor"""
    builder = current_builder()

    if correction is None:
        correction = 1.0

    if dim is None:
        var = _aten_var_onnx(self, correction=correction, keepdim=keepdim)
    else:
        var = _aten_var_dim_onnx(self, dims=dim, correction=correction, keepdim=keepdim)
    return builder.call("Sqrt", var)[0]


def aten_std_mean(self: ir.Value, unbiased: bool = True) -> tuple[ir.Value, ir.Value]:
    """std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)"""
    builder = current_builder()

    # Assume bool(True) and int(1) are same in ONNX, so pass "unbiased" directly as "correction"
    # If not this case, should be explicitly set correction value according to unbiased value
    var, mean = _aten_var_mean_onnx(self, correction=float(unbiased), keepdim=False)
    return builder.call("Sqrt", var)[0], mean


def aten_std_mean_dim(
    self: ir.Value, dim: Sequence[int], unbiased: bool = True, keepdim: bool = False
) -> tuple[ir.Value, ir.Value]:
    """std_mean.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)"""
    builder = current_builder()

    # Although dim is Optional in signature, but we assume it must have value for this overload
    # Assert(dim is not None)
    var, mean = _aten_var_mean_dim_onnx(
        self, dims=dim, correction=float(unbiased), keepdim=keepdim
    )
    return builder.call("Sqrt", var)[0], mean


def aten_std_mean_correction(
    self: ir.Value,
    # FIXME(justinchuby): Make dim Optional[Sequence[int]]
    dim: int | None = None,
    correction: float | None = None,
    keepdim: bool = False,
) -> tuple[ir.Value, ir.Value]:
    """std_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)"""
    builder = current_builder()

    if correction is None:
        correction = 1.0

    if dim is None:
        var, mean = _aten_var_mean_onnx(self, correction=correction, keepdim=keepdim)
    else:
        var, mean = _aten_var_mean_dim_onnx(
            self, dims=dim, correction=correction, keepdim=keepdim
        )
    return builder.call("Sqrt", var)[0], mean


def aten_var(self: ir.Value, unbiased: bool | None = True) -> ir.Value:
    """var(Tensor self, bool unbiased=True) -> Tensor"""

    # Assume bool(True) and int(1) are same in ONNX, so pass "unbiased" directly as "correction"
    # If not this case, should be explicitly set correction value according to unbiased value
    return _aten_var_onnx(self, correction=float(unbiased), keepdim=False)


def aten_var_dim(
    self: ir.Value,
    dim: Sequence[int],
    unbiased: bool | None = True,
    keepdim: bool | None = False,
) -> ir.Value:
    """var(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor"""

    return _aten_var_dim_onnx(
        self, dims=dim, correction=float(unbiased), keepdim=keepdim
    )


def aten_var_correction(
    self: ir.Value,
    # FIXME(justinchuby): Make dim Optional[Sequence[int]]
    dim: int | None = None,
    correction: float | None = None,
    keepdim: bool = False,
) -> ir.Value:
    """var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor"""

    if correction is None:
        correction = 1.0

    if dim is None:
        var = _aten_var_onnx(self, correction=correction, keepdim=keepdim)
    else:
        var = _aten_var_dim_onnx(self, dims=dim, correction=correction, keepdim=keepdim)
    return var


def _aten_var_onnx(
    self: ir.Value, correction: float, keepdim: bool = False
) -> ir.Value:
    builder = current_builder()
    mean = builder.call("ReduceMean", self, keepdims=keepdim)[0]
    sub_mean = builder.call("Sub", self, mean)[0]
    sqr_mean = builder.call("Mul", sub_mean, sub_mean)[0]
    var = builder.call("ReduceMean", sqr_mean, keepdims=keepdim)[0]
    # Adjust var according to correction value
    if correction > 0.0:
        self_shape = builder.call("Shape", self)[0]
        numel_float = builder.call(
            "CastLike", builder.call("ReduceProd", self_shape, keepdims=False)[0], self
        )[0]
        mul = builder.call("Mul", var, numel_float)[0]
        sub = builder.call(
            "Sub", numel_float, builder.call("CastLike", correction, self)[0]
        )[0]
        var = builder.call("Div", mul, sub)[0]

    return var


def _aten_var_dim_onnx(
    self: ir.Value, dims: Sequence[int], correction: float, keepdim: bool = False
) -> ir.Value:
    builder = current_builder()
    dims = builder.call(
        "Reshape",
        common_ops.merge_dims(dims),
        builder.constant([-1], dtype=ir.DataType.INT64),
    )[0]
    # Computer mean and var
    sub_mean = builder.call(
        "Sub", self, builder.call("ReduceMean", self, dims, keepdims=True)[0]
    )[0]
    sqr_mean = builder.call("Mul", sub_mean, sub_mean)[0]
    var = builder.call("ReduceMean", sqr_mean, dims, keepdims=keepdim)[0]
    # Adjust var according to correction value
    if correction > 0.0:
        self_shape = builder.call("Shape", self)[0]
        dim_size = builder.call("Gather", self_shape, dims, axis=0)[0]
        numel_float = builder.call(
            "CastLike", builder.call("ReduceProd", dim_size, keepdims=False)[0], self
        )[0]
        mul = builder.call("Mul", var, numel_float)[0]
        sub = builder.call(
            "Sub", numel_float, builder.call("CastLike", correction, self)[0]
        )[0]
        var = builder.call("Div", mul, sub)[0]

    return var


def aten_var_mean(self: ir.Value, unbiased: bool = True) -> tuple[ir.Value, ir.Value]:
    """var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)"""

    # Assume bool(True) and int(1) are same in ONNX, so pass "unbiased" directly as "correction"
    # If not this case, should be explicitly set correction value according to unbiased value
    return _aten_var_mean_onnx(self, correction=float(unbiased), keepdim=False)


def aten_var_mean_dim(
    self: ir.Value, dim: Sequence[int], unbiased: bool = True, keepdim: bool = False
) -> tuple[ir.Value, ir.Value]:
    """var_mean.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)"""

    # Although dim is Optional in signature, but we assume it must have value for this overload
    # Assert(dim is not None)
    return _aten_var_mean_dim_onnx(
        self, dims=dim, correction=float(unbiased), keepdim=keepdim
    )


def aten_var_mean_correction(
    self: ir.Value,
    # FIXME(justinchuby): Make dim Optional[Sequence[int]]
    dim: int | None = None,
    correction: float | None = None,
    keepdim: bool = False,
) -> tuple[ir.Value, ir.Value]:
    """var_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)"""

    if correction is None:
        correction = 1.0

    if dim is None:
        var, mean = _aten_var_mean_onnx(self, correction=correction, keepdim=keepdim)
    else:
        var, mean = _aten_var_mean_dim_onnx(
            self, dims=dim, correction=correction, keepdim=keepdim
        )
    return var, mean


def _aten_var_mean_onnx(
    self: ir.Value, correction: float = 1.0, keepdim: bool = False
) -> tuple[ir.Value, ir.Value]:
    builder = current_builder()
    # Compute mean and var
    mean = builder.call("ReduceMean", self, keepdims=keepdim)[0]
    sub_mean = builder.call("Sub", self, mean)[0]
    sqr_mean = builder.call("Mul", sub_mean, sub_mean)[0]
    var = builder.call("ReduceMean", sqr_mean, keepdims=keepdim)[0]
    # Adjust var according to correction value
    if correction > 0.0:
        self_shape = builder.call("Shape", self)[0]
        numel_float = builder.call(
            "CastLike", builder.call("ReduceProd", self_shape, keepdims=False)[0], self
        )[0]
        mul = builder.call("Mul", var, numel_float)[0]
        sub = builder.call(
            "Sub", numel_float, builder.call("CastLike", correction, self)[0]
        )[0]
        var = builder.call("Div", mul, sub)[0]

    return var, mean


def _aten_var_mean_dim_onnx(
    self: ir.Value, dims: Sequence[int], correction: float, keepdim: bool = False
) -> tuple[ir.Value, ir.Value]:
    builder = current_builder()
    dims = builder.call(
        "Reshape",
        common_ops.merge_dims(dims),
        builder.constant([-1], dtype=ir.DataType.INT64),
    )[0]
    # Computer mean and var
    mean = builder.call("ReduceMean", self, dims, keepdims=keepdim)[0]
    sub_mean = builder.call(
        "Sub", self, builder.call("ReduceMean", self, dims, keepdims=True)[0]
    )[0]
    sqr_mean = builder.call("Mul", sub_mean, sub_mean)[0]
    var = builder.call("ReduceMean", sqr_mean, dims, keepdims=keepdim)[0]
    # Adjust var according to correction value
    if correction > 0.0:
        self_shape = builder.call("Shape", self)[0]
        dim_size = builder.call("Gather", self_shape, dims, axis=0)[0]
        numel_float = builder.call(
            "CastLike", builder.call("ReduceProd", dim_size, keepdims=False)[0], self
        )[0]
        mul = builder.call("Mul", var, numel_float)[0]
        sub = builder.call(
            "Sub", numel_float, builder.call("CastLike", correction, self)[0]
        )[0]
        var = builder.call("Div", mul, sub)[0]

    return var, mean
