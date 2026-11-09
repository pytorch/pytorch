# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned arithmetic and elementwise ONNX IR translations."""

from __future__ import annotations

import math
import operator

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir
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
