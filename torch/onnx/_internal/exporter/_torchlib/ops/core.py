# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned core ONNX IR translations."""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import common as common_ops


if TYPE_CHECKING:
    from torch.onnx._internal.exporter._ir_builder import TranslationContext


aten = torch.ops.aten
prims = torch.ops.prims


@onnx_impl((aten.abs.default, operator.abs))
def aten_abs(op: TranslationContext, self: ir.Value) -> ir.Value:
    """abs(Tensor self) -> Tensor"""

    return op.Abs(self)


@onnx_impl(aten.abs.default, complex=True)
def aten_abs_complex(op: TranslationContext, self: ir.Value) -> ir.Value:
    """abs(Tensor self) -> Tensor"""

    return op.ReduceL2(self, [-1], keepdims=False)


@onnx_impl(aten._local_scalar_dense.default)
def aten__local_scalar_dense(op: TranslationContext, self: ir.Value) -> ir.Value:
    """_local_scalar_dense(Tensor self) -> Scalar"""

    # Return the first element in tensor as a scalar.
    if self.dtype.is_floating_point():
        dtype = ir.DataType.FLOAT
    elif self.dtype == ir.DataType.BOOL:
        dtype = ir.DataType.BOOL
    else:
        dtype = ir.DataType.INT64
    return op.Cast(
        op.Gather(op.Reshape(self, [-1]), 0),
        to=dtype,
    )


@onnx_impl(aten.acos.default)
def aten_acos(op: TranslationContext, self: ir.Value) -> ir.Value:
    """acos(Tensor self) -> Tensor"""

    return op.Acos(self)


@onnx_impl(aten.acosh.default)
def aten_acosh(op: TranslationContext, self: ir.Value) -> ir.Value:
    """acosh(Tensor self) -> Tensor"""

    return op.Acosh(self)


@onnx_impl(aten.add.Tensor)
def aten_add(
    op: TranslationContext, self: ir.Value, other: ir.Value, alpha: float = 1.0
) -> ir.Value:
    """add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""

    if self.dtype == ir.DataType.BOOL:
        # alpha can also be bool
        if alpha == 0:
            return op.Identity(self)
        return op.Or(self, other)

    if alpha != 1.0:
        alpha = op.CastLike(alpha, other)
        other = op.Mul(other, alpha)
    return op.Add(self, other)


@onnx_impl(aten.add.Scalar)
def aten_add_scalar(
    op: TranslationContext, self: ir.Value, other: float, alpha: float = 1.0
) -> ir.Value:
    """add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor"""
    builder = op.builder

    other = builder.constant(other, dtype=self.dtype)
    return aten_add(op, self, other, alpha=alpha)


@onnx_impl(operator.add)
def operator_add(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    return op.Add(self, other)


@onnx_impl((aten.add.Tensor, aten.add.Scalar), complex=True)
def aten_add_complex(
    op: TranslationContext,
    self: ir.Value,
    other: ir.Value | complex | float | int | bool,
    alpha: float = 1.0,
) -> ir.Value:
    """add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    other = common_ops.complex_scalar(op, other, dtype=self.dtype)
    return aten_add(op, self, other, alpha=alpha)


@onnx_impl(aten.addcdiv.default)
def aten_addcdiv(
    op: TranslationContext,
    self: ir.Value,
    tensor1: ir.Value,
    tensor2: ir.Value,
    value: float = 1.0,
) -> ir.Value:
    """addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor

    Performs the element-wise division of tensor1 by tensor2, multiplies the result
    by the scalar value and adds it to self.
    """

    return op.Add(
        self,
        op.Mul(op.Div(tensor1, tensor2), value),
    )


@onnx_impl(aten.addcmul.default)
def aten_addcmul(
    op: TranslationContext,
    self: ir.Value,
    tensor1: ir.Value,
    tensor2: ir.Value,
    value: float = 1.0,
) -> ir.Value:
    """addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor

    Performs the element-wise multiplication of tensor1 by tensor2, multiplies the
    result by the scalar value and adds it to self.
    """

    # Follow the order in https://github.com/pytorch/pytorch/blob/29e3fddb082b5a14262a7246bc62381a55199d45/aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp#L47
    # TODO(#811): Understand fp16 accuracy issue
    return op.Add(
        self,
        op.Mul(op.Mul(value, tensor1), tensor2),
    )


@onnx_impl(aten.alias.default)
def aten_alias(op: TranslationContext, self: ir.Value) -> ir.Value:
    """alias(Tensor(a) self) -> Tensor(a)"""

    return op.Identity(self)


@onnx_impl(aten.angle.default)
def aten_angle(op: TranslationContext, self: ir.Value) -> ir.Value:
    """angle(Tensor self) -> Tensor"""

    zero = common_ops.constant(op, 0.0, dtype=self.dtype)
    pi = common_ops.constant(op, math.pi, dtype=self.dtype)

    result = op.Where(op.Less(self, zero), pi, zero)
    return result


@onnx_impl(aten.angle.default, complex=True)
def aten_angle_complex(op: TranslationContext, self: ir.Value) -> ir.Value:
    """angle(Tensor self) -> Tensor"""

    self_real = op.Squeeze(op.Slice(self, [0], [1], [-1]), [-1])
    self_imag = op.Squeeze(op.Slice(self, [1], [2], [-1]), [-1])

    result = _atan2(op, self_imag, self_real, dtype=self.dtype)
    return result


@onnx_impl(aten.asin.default)
def aten_asin(op: TranslationContext, self: ir.Value) -> ir.Value:
    """asin(Tensor self) -> Tensor"""

    return op.Asin(self)


@onnx_impl(aten.asinh.default)
def aten_asinh(op: TranslationContext, self: ir.Value) -> ir.Value:
    """asinh(Tensor self) -> Tensor"""

    return op.Asinh(self)


@onnx_impl(aten.atan.default)
def aten_atan(op: TranslationContext, self: ir.Value) -> ir.Value:
    """atan(Tensor self) -> Tensor"""

    return op.Atan(self)


@onnx_impl(aten.atan2.default)
def aten_atan2(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """atan2(Tensor self, Tensor other) -> Tensor"""
    return _atan2(op, self, other, dtype=self.dtype)


def _atan2(
    op: TranslationContext, y: ir.Value, x: ir.Value, dtype: ir.DataType
) -> ir.Value:
    slope = op.Div(y, x)
    atan = op.Atan(slope)
    zero = common_ops.constant(op, 0.0, dtype=dtype)
    pi = common_ops.constant(op, math.pi, dtype=dtype)

    second_third_quadrant = op.Where(
        op.Greater(y, zero),
        op.Add(atan, pi),
        op.Sub(atan, pi),
    )
    result = op.Where(op.Less(x, zero), second_third_quadrant, atan)

    # Map NaN to 0 to match PyTorch behavior
    result = op.Where(op.IsNaN(result), zero, result)

    return result


@onnx_impl(aten.atanh.default)
def aten_atanh(op: TranslationContext, self: ir.Value) -> ir.Value:
    """atanh(Tensor self) -> Tensor"""

    return op.Atanh(self)


@onnx_impl((aten.bitwise_and.Tensor, operator.and_))
def aten_bitwise_and(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    """bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor"""

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
        return op.BitwiseAnd(self, other)
    if dtype == ir.DataType.BOOL:
        return op.And(self, other)
    raise NotImplementedError(
        f"aten_bitwise_and does not support dtype {self.dtype}. Use integer or boolean tensors."
    )


@onnx_impl(aten.bitwise_and.Scalar)
def aten_bitwise_and_scalar(
    op: TranslationContext, self: ir.Value, other: int
) -> ir.Value:
    """bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = op.builder

    other_tensor = builder.constant(other, dtype=self.dtype)
    return aten_bitwise_and(op, self, other_tensor)


@onnx_impl(aten.bitwise_and.Scalar_Tensor)
def aten_bitwise_and_scalar_tensor(
    op: TranslationContext, self: float, other: ir.Value
) -> ir.Value:
    """bitwise_and.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = op.builder

    self_tensor = builder.constant(self, dtype=other.dtype)
    return aten_bitwise_and(op, self_tensor, other)


@onnx_impl((aten.bitwise_left_shift.Tensor, aten.__lshift__.Tensor, operator.lshift))
def aten_bitwise_left_shift(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    """bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor"""
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
        return op.BitShift(self, other, direction="LEFT")

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

    self = op.Cast(self, to=unsigned_dtype)
    other = op.Cast(other, to=unsigned_dtype)

    result = op.BitShift(self, other, direction="LEFT")

    return op.Cast(result, to=signed_dtype)


@onnx_impl((aten.bitwise_left_shift.Tensor_Scalar, aten.__lshift__.Scalar))
def aten_bitwise_left_shift_tensor_scalar(
    op: TranslationContext, self: ir.Value, other: int
) -> ir.Value:
    """bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = op.builder
    other_tensor = builder.constant(other, dtype=self.dtype)
    return aten_bitwise_left_shift(op, self, other_tensor)


@onnx_impl(aten.bitwise_left_shift.Scalar_Tensor)
def aten_bitwise_left_shift_scalar_tensor(
    op: TranslationContext, self: int, other: ir.Value
) -> ir.Value:
    """bitwise_left_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = op.builder
    self_tensor = builder.constant(self, dtype=other.dtype)
    return aten_bitwise_left_shift(op, self_tensor, other)


@onnx_impl(aten.bitwise_not.default)
def aten_bitwise_not(op: TranslationContext, self: ir.Value) -> ir.Value:
    """bitwise_not(Tensor self) -> Tensor"""

    if self.dtype == ir.DataType.BOOL:
        return op.Not(self)
    if self.dtype.is_integer():
        return op.BitwiseNot(self)
    raise NotImplementedError(
        f"aten_bitwise_not does not support dtype {self.dtype}. Use integer or boolean tensors."
    )


@onnx_impl((aten.bitwise_or.Tensor, operator.or_))
def aten_bitwise_or(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    """bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor"""

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
        return op.BitwiseOr(self, other)
    if dtype == ir.DataType.BOOL:
        return op.Or(self, other)
    raise NotImplementedError(
        f"aten_bitwise_or does not support dtype {self.dtype}. Use integer or boolean tensors."
    )


@onnx_impl(aten.bitwise_or.Scalar)
def aten_bitwise_or_scalar(
    op: TranslationContext, self: ir.Value, other: int
) -> ir.Value:
    """bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = op.builder
    other_tensor = builder.constant(other, dtype=self.dtype)
    return aten_bitwise_or(op, self, other_tensor)


@onnx_impl(aten.bitwise_or.Scalar_Tensor)
def aten_bitwise_or_scalar_tensor(
    op: TranslationContext, self: int, other: ir.Value
) -> ir.Value:
    """bitwise_or.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = op.builder
    self_tensor = builder.constant(self, dtype=other.dtype)
    return aten_bitwise_or(op, self_tensor, other)


@onnx_impl((aten.bitwise_right_shift.Tensor, aten.__rshift__.Tensor, operator.rshift))
def aten_bitwise_right_shift(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    """bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor"""
    builder = op.builder
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
        return op.BitShift(self, other, direction="RIGHT")

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

    negative = op.Less(self, 0)
    self = op.Cast(self, to=unsigned_dtype)
    other = op.Cast(other, to=unsigned_dtype)

    # Simulate arithmetic shift using logical shift
    # Clear the lower bits of an all one mask to create the mask to simulate the sign bit shifting
    mask = op.BitShift(mask, other, direction="RIGHT")
    mask = op.BitwiseNot(mask)
    # Do logical shift
    shifted = op.BitShift(self, other, direction="RIGHT")
    # Compute the arithmetic shifted value assuming the sign bit was set
    negative_shifted = op.BitwiseOr(shifted, mask)
    # Choose the shifted value based on the sign bit
    return op.Where(
        negative,
        op.Cast(negative_shifted, to=signed_dtype),
        op.Cast(shifted, to=signed_dtype),
    )


@onnx_impl((aten.bitwise_right_shift.Tensor_Scalar, aten.__rshift__.Scalar))
def aten_bitwise_right_shift_tensor_scalar(
    op: TranslationContext, self: ir.Value, other: int
) -> ir.Value:
    """bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = op.builder
    other_tensor = builder.constant(other, dtype=self.dtype)
    return aten_bitwise_right_shift(op, self, other_tensor)


@onnx_impl(aten.bitwise_right_shift.Scalar_Tensor)
def aten_bitwise_right_shift_scalar_tensor(
    op: TranslationContext, self: int, other: ir.Value
) -> ir.Value:
    """bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = op.builder
    self_tensor = builder.constant(self, dtype=other.dtype)
    return aten_bitwise_right_shift(op, self_tensor, other)


@onnx_impl(aten.bitwise_xor.Tensor)
def aten_bitwise_xor(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    """bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor"""

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
        return op.BitwiseXor(self, other)
    if dtype == ir.DataType.BOOL:
        return op.Xor(self, other)
    raise NotImplementedError(
        f"aten_bitwise_xor does not support dtype {self.dtype}. Use integer or boolean tensors."
    )


@onnx_impl(aten.bitwise_xor.Scalar)
def aten_bitwise_xor_scalar(
    op: TranslationContext, self: ir.Value, other: int
) -> ir.Value:
    """bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = op.builder
    other_tensor = builder.constant(other, dtype=self.dtype)
    return aten_bitwise_xor(op, self, other_tensor)


@onnx_impl(aten.bitwise_xor.Scalar_Tensor)
def aten_bitwise_xor_scalar_tensor(
    op: TranslationContext, self: int, other: ir.Value
) -> ir.Value:
    """bitwise_xor.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = op.builder
    self_tensor = builder.constant(self, dtype=other.dtype)
    return aten_bitwise_xor(op, self_tensor, other)


@onnx_impl(aten.ceil.default)
def aten_ceil(op: TranslationContext, self: ir.Value) -> ir.Value:
    """ceil(Tensor self) -> Tensor"""

    return op.Ceil(self)


@onnx_impl(math.ceil)
def python_math_ceil(op: TranslationContext, self: ir.Value) -> ir.Value:
    """ceil(Tensor self) -> Tensor"""
    ceil = op.Ceil(self)
    return op.Cast(ceil, to=ir.DataType.INT64)


@onnx_impl(aten.clamp.default)
def aten_clamp(
    op: TranslationContext,
    self: ir.Value,
    min: float | None = None,
    max: float | None = None,
) -> ir.Value:
    """clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor"""

    if min is None and max is None:
        return op.Identity(self)

    if min is not None:
        min = op.CastLike(min, self)

    if max is not None:
        max = op.CastLike(max, self)

    return op.Clip(self, min, max)


@onnx_impl(aten.clamp.Tensor)
def aten_clamp_tensor(
    op: TranslationContext,
    self: ir.Value,
    min: ir.Value | None = None,
    max: ir.Value | None = None,
) -> ir.Value:
    """clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor"""

    if min is None and max is None:
        return op.Identity(self)

    clamped = self

    # If min is greater than max torch.clamp(..., min, max)
    # sets all elements in input to the value of max.
    # So this order is important.
    if min is not None:
        min_clamp = op.CastLike(min, self)
        clamped = op.Max(clamped, min_clamp)

    if max is not None:
        max_clamp = op.CastLike(max, self)
        clamped = op.Min(clamped, max_clamp)

    return clamped


@onnx_impl(aten.clamp_max.default)
def aten_clamp_max(op: TranslationContext, self: ir.Value, max_: float) -> ir.Value:
    """clamp_max(Tensor self, Scalar max) -> Tensor"""

    # This implementation does not intend to handle when self is an empty tensor
    max_ = op.CastLike(max_, self)
    return op.Clip(self, None, max_)


@onnx_impl(aten.clamp_max.Tensor)
def aten_clamp_max_tensor(
    op: TranslationContext, self: ir.Value, max_: ir.Value
) -> ir.Value:
    """clamp_max.Tensor(Tensor self, Tensor max) -> Tensor"""

    # This implementation does not intend to handle when self is an empty tensor
    max_rank = len(max_.shape)
    if max_rank == 0:
        max_ = op.CastLike(max_, self)
        result = op.Clip(self, None, max_)
    else:
        result = op.Min(self, max_)

    return result


@onnx_impl(aten.clamp_min.default)
def aten_clamp_min(op: TranslationContext, self: ir.Value, min_: float) -> ir.Value:
    """clamp_min(Tensor self, Scalar min) -> Tensor"""

    # This implementation does not intend to handle when self is an empty tensor
    min_ = op.CastLike(min_, self)
    return op.Clip(self, min_, None)


@onnx_impl(aten.clamp_min.Tensor)
def aten_clamp_min_tensor(
    op: TranslationContext, self: ir.Value, min_: ir.Value
) -> ir.Value:
    """clamp_min.Tensor(Tensor self, Tensor min) -> Tensor"""

    # This implementation does not intend to handle when self is an empty tensor
    min_rank = len(min_.shape)
    if min_rank == 0:
        min_ = op.CastLike(min_, self)
        result = op.Clip(self, min_, None)
    else:
        result = op.Max(self, min_)

    return result


@onnx_impl(aten.clone.default)
def aten_clone(
    op: TranslationContext,
    self: ir.Value,
    memory_format: str = "",
) -> ir.Value:
    """clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"""

    return op.Identity(self)


@onnx_impl(aten.clone.default, complex=True)
def aten_clone_complex(
    op: TranslationContext,
    self: ir.Value,
    memory_format: str = "",  # pylint: disable=unused-argument
) -> ir.Value:
    """clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"""
    return op.Identity(self)


@onnx_impl((aten.conj.default, aten._conj.default))
def aten_conj(op: TranslationContext, self: ir.Value) -> ir.Value:
    """conj(Tensor(a) self) -> Tensor(a)"""

    return op.Identity(self)


def _complex_conjugate(op: TranslationContext, self: ir.Value) -> ir.Value:
    builder = op.builder
    zero = builder.constant([0], dtype=ir.DataType.INT64)
    one = builder.constant([1], dtype=ir.DataType.INT64)
    two = builder.constant([2], dtype=ir.DataType.INT64)
    neg_1 = builder.constant([-1], dtype=ir.DataType.INT64)
    # The last dimension is the real and imaginary parts

    real = op.Slice(self, zero, one, neg_1)
    imag = op.Slice(self, one, two, neg_1)
    conjugated = op.Concat(real, op.Neg(imag), axis=-1)

    return conjugated


@onnx_impl((aten.conj.default, aten._conj.default), complex=True)
def aten_conj_complex(op: TranslationContext, self: ir.Value) -> ir.Value:
    """conj(Tensor(a) self) -> Tensor(a)"""
    return _complex_conjugate(op, self)


@onnx_impl(aten._to_copy.default)
def aten__to_copy(
    op: TranslationContext,
    self: ir.Value,
    dtype: int = -1,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: str = "",
) -> ir.Value:
    """_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor"""

    if dtype == -1:
        return op.Identity(self)
    else:
        return common_ops.cast_to(op, self, dtype=dtype)


@onnx_impl(aten.cos.default)
def aten_cos(op: TranslationContext, self: ir.Value) -> ir.Value:
    """cos(Tensor self) -> Tensor"""

    return op.Cos(self)


@onnx_impl(aten.cosh.default)
def aten_cosh(op: TranslationContext, self: ir.Value) -> ir.Value:
    """cosh(Tensor self) -> Tensor"""

    return op.Cosh(self)


@onnx_impl(aten.deg2rad.default)
def aten_deg2rad(op: TranslationContext, self: ir.Value) -> ir.Value:
    """deg2rad(Tensor self) -> Tensor"""

    return op.Mul(self, op.CastLike(math.pi / 180.0, self))


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
def aten_div(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """div.Tensor(Tensor self, Tensor other) -> Tensor"""

    # Int inputs will be promoted to float by PyTorch
    return op.Div(self, other)


@onnx_impl(operator.truediv)
def operator_truediv(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    return op.Div(
        op.Cast(self, to=ir.DataType.FLOAT),
        op.Cast(other, to=ir.DataType.FLOAT),
    )


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
    op: TranslationContext,
    self: ir.Value,
    other: ir.Value | complex | float | int | bool,
) -> ir.Value:
    """div.Tensor(Tensor self, Tensor other) -> Tensor"""
    other = common_ops.complex_scalar(op, other, dtype=self.dtype)

    # Complex division. PyTorch type promotion ensures both arguments are complex numbers
    self_real = op.Slice(self, [0], [1], [-1])
    self_imag = op.Slice(self, [1], [2], [-1])
    other_real = op.Slice(other, [0], [1], [-1])
    other_imag = op.Slice(other, [1], [2], [-1])

    # Complex division
    # (a + bi) / (c + di) = (ac + bd) / (c^2 + d^2) + (bc - ad) / (c^2 + d^2)i
    # https://mathworld.wolfram.com/ComplexDivision.html
    ac = op.Mul(self_real, other_real)
    bd = op.Mul(self_imag, other_imag)
    bc = op.Mul(self_imag, other_real)
    ad = op.Mul(self_real, other_imag)
    denominator = op.Add(
        op.Mul(other_real, other_real),
        op.Mul(other_imag, other_imag),
    )
    real = op.Div(op.Add(ac, bd), denominator)
    imag = op.Div(op.Sub(bc, ad), denominator)

    return op.Concat(real, imag, axis=-1)


@onnx_impl((aten.div.Tensor_mode, aten.div.Scalar_mode))
def aten_div_mode(
    op: TranslationContext,
    self: ir.Value,
    other: ir.Value,
    rounding_mode: str | None = None,
) -> ir.Value:
    """div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor"""

    if rounding_mode not in {"trunc", "floor", None}:
        raise ValueError(
            f"Invalid rounding_mode={rounding_mode!r}. Use None, 'trunc', or 'floor'."
        )

    if self.dtype.is_integer():
        quotient = op.Div(
            op.Cast(self, to=ir.DataType.FLOAT),
            op.Cast(other, to=ir.DataType.FLOAT),
        )

        if rounding_mode == "trunc":
            # Rounds the results of the division towards zero.
            # Equivalent to C-style integer division
            result = aten_trunc(op, quotient)
            return op.CastLike(result, self)
        if rounding_mode == "floor":
            result = op.Floor(quotient)
            return op.CastLike(result, self)

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
        quotient = op.Div(self, other)
        if self.dtype == ir.DataType.FLOAT16:
            quotient = op.Cast(quotient, to=ir.DataType.FLOAT16)
        return aten_trunc(op, quotient)
    if rounding_mode == "floor":
        return op.Floor(op.Div(self, other))

    return op.Div(self, other)


@onnx_impl((aten.eq.default, aten.eq.Tensor, aten.eq.Scalar, operator.eq))
def aten_eq(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """eq.Tensor(Tensor self, Tensor other) -> Tensor"""

    return op.Equal(self, other)


@onnx_impl(aten.exp.default)
def aten_exp(op: TranslationContext, self: ir.Value) -> ir.Value:
    """exp(Tensor self) -> Tensor"""

    return op.Exp(self)


@onnx_impl(aten.exp2.default)
def aten_exp2(op: TranslationContext, self: ir.Value) -> ir.Value:
    """exp2(Tensor self) -> Tensor"""
    builder = op.builder

    two = builder.constant(2, dtype=ir.DataType.INT64)
    two = op.CastLike(two, self)
    return op.Pow(two, self)


@onnx_impl(aten.floor.default)
def aten_floor(op: TranslationContext, self: ir.Value) -> ir.Value:
    """floor(Tensor self) -> Tensor"""

    return op.Floor(self)


@onnx_impl(math.floor)
def python_math_floor(op: TranslationContext, self: ir.Value) -> ir.Value:
    """floor(Tensor self) -> Tensor"""
    floor = op.Floor(self)
    return op.Cast(floor, to=ir.DataType.INT64)


@onnx_impl(aten.floor_divide.default)
def aten_floor_divide(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    """floor_divide(Tensor self, Tensor other) -> Tensor"""

    if self.dtype.is_floating_point():
        return op.Floor(op.Div(self, other))

    if not self.dtype.is_integer():
        raise ValueError(
            f"floor_divide requires real numeric tensors, got {self.dtype}. Cast the input to an integer or floating-point dtype."
        )

    if not self.dtype.is_signed():
        return op.Div(self, other)

    # Convert truncation to flooring
    # Reference: https://github.com/pytorch/pytorch/blob/ffc645c870f0abd368606ba1e2b3b58cacb03046/torch/_refs/__init__.py#L1401C1-L1409C70
    # offset = (torch.signbit(a) != torch.signbit(b)).logical_and(torch.fmod(a, b) != 0)
    # return prims.div(a, b) - _maybe_convert_to_dtype(offset, a.dtype)
    # The sign-mismatch check ``signbit(a) != signbit(b)`` (for nonzero divisors)
    # is expressed as ``(a < 0) == (b > 0)``, which avoids Sign (limited integral
    # support on some EPs) and enables optimizations when ``self`` is provably
    # non-negative.
    offset = op.And(
        op.Equal(
            op.Less(self, 0),
            op.Greater(other, 0),
        ),
        op.Cast(op.Mod(self, other), to=ir.DataType.BOOL),
    )
    offset = op.Cast(offset, to=self.dtype)
    return op.Sub(op.Div(self, other), offset)


@onnx_impl(operator.floordiv)
def operator_floordiv(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    # We implement floor_divide only for positive inputs (using integer division)
    # because that is the usual intended case and is the most efficient.
    return op.Div(self, other)


@onnx_impl((aten.fmod.Tensor, aten.fmod.Scalar))
def aten_fmod(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """fmod.Tensor(Tensor self, Tensor other) -> Tensor"""

    return op.Mod(self, other, fmod=1)


@onnx_impl(aten.frac.default)
def aten_frac(op: TranslationContext, self: ir.Value) -> ir.Value:
    """frac(Tensor self) -> Tensor

    Computes the fractional portion of each element in input.
    """

    # https://pytorch.org/docs/stable/generated/torch.frac.html
    return op.Sub(
        self,
        op.Mul(
            op.Floor(op.Abs(self)),
            op.Sign(self),
        ),
    )


@onnx_impl((aten.ge.Tensor, aten.ge.Scalar, aten.greater_equal.Tensor))
def aten_ge(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """ge.Tensor(Tensor self, Tensor other) -> Tensor"""

    if self.dtype == ir.DataType.BOOL:
        # self, other, self >= other
        #    F,    F,    T
        #    F,    T,    F
        #    T,    F,    T
        #    T,    T,    T
        return op.Or(self, op.Not(other))

    return op.GreaterOrEqual(self, other)


@onnx_impl(operator.ge)
def operator_ge(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    # operator.ge for SymInt
    return op.GreaterOrEqual(self, other)


@onnx_impl((aten.gt.Tensor, aten.gt.Scalar, aten.greater.Tensor))
def aten_gt(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """gt.Tensor(Tensor self, Tensor other) -> Tensor"""

    if self.dtype == ir.DataType.BOOL:
        # self, other, self > other
        #    F,    F,    F
        #    F,    T,    F
        #    T,    F,    T
        #    T,    T,    F

        return op.And(self, op.Not(other))

    return op.Greater(self, other)


@onnx_impl(operator.gt)
def operator_gt(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    # operator.gt for SymInt
    return op.Greater(self, other)


@onnx_impl(aten.trunc.default)
def aten_trunc(op: TranslationContext, self: ir.Value) -> ir.Value:
    """trunc(Tensor self) -> Tensor"""
    # Reference https://github.com/onnx/onnx/issues/4588#issuecomment-2658170591
    return op.Mul(
        op.Floor(op.Abs(self)),
        op.Sign(self),
    )
