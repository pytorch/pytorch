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
    from collections.abc import Sequence

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
    if common_ops.require_dtype(op, self).is_floating_point():
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

    other_tensor = builder.constant(other, dtype=common_ops.require_dtype(op, self))
    return aten_add(op, self, other_tensor, alpha=alpha)


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

    zero = common_ops.constant(op, 0.0, dtype=common_ops.require_dtype(op, self))
    pi = common_ops.constant(op, math.pi, dtype=common_ops.require_dtype(op, self))

    result = op.Where(op.Less(self, zero), pi, zero)
    return result


@onnx_impl(aten.angle.default, complex=True)
def aten_angle_complex(op: TranslationContext, self: ir.Value) -> ir.Value:
    """angle(Tensor self) -> Tensor"""

    self_real = op.Squeeze(op.Slice(self, [0], [1], [-1]), [-1])
    self_imag = op.Squeeze(op.Slice(self, [1], [2], [-1]), [-1])

    result = _atan2(op, self_imag, self_real, dtype=common_ops.require_dtype(op, self))
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
    return _atan2(op, self, other, dtype=common_ops.require_dtype(op, self))


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

    other_tensor = builder.constant(other, dtype=common_ops.require_dtype(op, self))
    return aten_bitwise_and(op, self, other_tensor)


@onnx_impl(aten.bitwise_and.Scalar_Tensor)
def aten_bitwise_and_scalar_tensor(
    op: TranslationContext, self: float, other: ir.Value
) -> ir.Value:
    """bitwise_and.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = op.builder

    self_tensor = builder.constant(self, dtype=common_ops.require_dtype(op, other))
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
    other_tensor = builder.constant(other, dtype=common_ops.require_dtype(op, self))
    return aten_bitwise_left_shift(op, self, other_tensor)


@onnx_impl(aten.bitwise_left_shift.Scalar_Tensor)
def aten_bitwise_left_shift_scalar_tensor(
    op: TranslationContext, self: int, other: ir.Value
) -> ir.Value:
    """bitwise_left_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = op.builder
    self_tensor = builder.constant(self, dtype=common_ops.require_dtype(op, other))
    return aten_bitwise_left_shift(op, self_tensor, other)


@onnx_impl(aten.bitwise_not.default)
def aten_bitwise_not(op: TranslationContext, self: ir.Value) -> ir.Value:
    """bitwise_not(Tensor self) -> Tensor"""

    if self.dtype == ir.DataType.BOOL:
        return op.Not(self)
    if common_ops.require_dtype(op, self).is_integer():
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
    other_tensor = builder.constant(other, dtype=common_ops.require_dtype(op, self))
    return aten_bitwise_or(op, self, other_tensor)


@onnx_impl(aten.bitwise_or.Scalar_Tensor)
def aten_bitwise_or_scalar_tensor(
    op: TranslationContext, self: int, other: ir.Value
) -> ir.Value:
    """bitwise_or.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = op.builder
    self_tensor = builder.constant(self, dtype=common_ops.require_dtype(op, other))
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
    other_tensor = builder.constant(other, dtype=common_ops.require_dtype(op, self))
    return aten_bitwise_right_shift(op, self, other_tensor)


@onnx_impl(aten.bitwise_right_shift.Scalar_Tensor)
def aten_bitwise_right_shift_scalar_tensor(
    op: TranslationContext, self: int, other: ir.Value
) -> ir.Value:
    """bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = op.builder
    self_tensor = builder.constant(self, dtype=common_ops.require_dtype(op, other))
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
    other_tensor = builder.constant(other, dtype=common_ops.require_dtype(op, self))
    return aten_bitwise_xor(op, self, other_tensor)


@onnx_impl(aten.bitwise_xor.Scalar_Tensor)
def aten_bitwise_xor_scalar_tensor(
    op: TranslationContext, self: int, other: ir.Value
) -> ir.Value:
    """bitwise_xor.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = op.builder
    self_tensor = builder.constant(self, dtype=common_ops.require_dtype(op, other))
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

    min_tensor = op.CastLike(min, self) if min is not None else None
    max_tensor = op.CastLike(max, self) if max is not None else None
    return op.Clip(self, min_tensor, max_tensor)


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
    max_rank = len(common_ops.require_shape(op, max_))
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
    min_rank = len(common_ops.require_shape(op, min_))
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

    if common_ops.require_dtype(op, self).is_integer():
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

    if common_ops.require_dtype(op, self).is_floating_point():
        return op.Floor(op.Div(self, other))

    if not common_ops.require_dtype(op, self).is_integer():
        raise ValueError(
            f"floor_divide requires real numeric tensors, got {self.dtype}. Cast the input to an integer or floating-point dtype."
        )

    if not common_ops.require_dtype(op, self).is_signed():
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


@onnx_impl(aten.isclose.default)
def aten_isclose(
    op: TranslationContext,
    self: ir.Value,
    other: ir.Value,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> ir.Value:
    """isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor"""
    builder = op.builder

    # torch builds isclose out of three terms in aten/src/ATen/native/TensorCompare.cpp:
    # exact equality, NaN against NaN when equal_nan is set, and the tolerance band
    # restricted to the elements whose error is finite.
    result = op.Equal(self, other)

    # |self - other| <= atol + rtol x |other|
    actual_error = op.Abs(op.Sub(self, other))
    allowed_error = op.Add(
        op.CastLike(atol, other),
        op.Mul(
            op.CastLike(rtol, other),
            op.Abs(other),
        ),
    )

    if common_ops.require_dtype(op, self).is_floating_point():
        # Comparing the two sides of the band by subtraction carries the finiteness
        # restriction on its own, so no IsInf term is needed. An infinity on either side
        # puts a NaN into the subtraction whichever way it arrives: the error and the
        # allowance are both infinite and cancel, or the allowance is already NaN because
        # rtol is zero and zero times infinity is NaN. Every comparison against NaN is
        # false, which leaves the equality term above as the only way an infinity is
        # reported close. On finite values the sign of the difference and the direct
        # comparison agree exactly, so the band is unchanged.
        zero = builder.constant(0, dtype=common_ops.require_dtype(op, self))
        within_tolerance = op.LessOrEqual(op.Sub(actual_error, allowed_error), zero)
        if equal_nan:
            result = op.Or(
                result,
                op.And(
                    op.IsNaN(self),
                    op.IsNaN(other),
                ),
            )
    else:
        # Integers are never infinite and never NaN, so they need no finiteness term, and
        # IsNaN does not accept them. The subtraction is left off here because it can
        # overflow a narrow integer type where the direct comparison cannot.
        within_tolerance = op.LessOrEqual(actual_error, allowed_error)

    return op.Or(result, within_tolerance)


@onnx_impl(aten.isfinite.default)
def aten_isfinite(op: TranslationContext, self: ir.Value) -> ir.Value:
    """isfinite(Tensor self) -> Tensor"""

    # IsInf only supports FLOAT and DOUBLE
    not_inf = op.Not(op.IsInf(self))
    # TODO: The test case doesn't cover this condition
    not_nan = op.Not(op.IsNaN(self))
    return op.And(not_inf, not_nan)


@onnx_impl(aten.isinf.default)
def aten_isinf(op: TranslationContext, self: ir.Value) -> ir.Value:
    """isinf(Tensor self) -> Tensor"""

    # Added Cast inside the function so it can support all real dtypes naturally
    self = op.Cast(self, to=ir.DataType.FLOAT)
    return op.IsInf(self)


@onnx_impl(aten.isnan.default)
def aten_isnan(op: TranslationContext, self: ir.Value) -> ir.Value:
    """isnan(Tensor self) -> Tensor"""

    return op.IsNaN(self)


@onnx_impl(aten.isneginf.default)
def aten_isneginf(op: TranslationContext, self: ir.Value) -> ir.Value:
    """isneginf(Tensor self) -> Tensor"""

    # Added Cast inside the function so it can support all real dtypes naturally
    self = op.Cast(self, to=ir.DataType.FLOAT)
    return op.And(op.Less(self, 0), op.IsInf(self))


@onnx_impl(aten.isposinf.default)
def aten_isposinf(op: TranslationContext, self: ir.Value) -> ir.Value:
    """isposinf(Tensor self) -> Tensor"""

    # Added Cast inside the function so it can support all real dtypes naturally
    self = op.Cast(self, to=ir.DataType.FLOAT)
    return op.And(op.Greater(self, 0), op.IsInf(self))


@onnx_impl((aten.le.Tensor, aten.le.Scalar, aten.less_equal.Tensor))
def aten_le(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """le.Tensor(Tensor self, Tensor other) -> Tensor"""

    if self.dtype == ir.DataType.BOOL:
        # self, other, self <= other
        #    F,    F,    T
        #    F,    T,    T
        #    T,    F,    F
        #    T,    T,    T

        return op.Or(other, op.Not(self))

    return op.LessOrEqual(self, other)


@onnx_impl(operator.le)
def operator_le(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    # operator.le for SymInt
    return op.LessOrEqual(self, other)


@onnx_impl((aten.lerp.Tensor, aten.lerp.Scalar))
def aten_lerp(
    op: TranslationContext, self: ir.Value, end: ir.Value, weight: ir.Value
) -> ir.Value:
    """lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor"""

    weight = op.CastLike(weight, self)
    diff = op.Sub(end, self)
    return op.Where(
        op.Less(weight, 0.5),
        op.Add(self, op.Mul(weight, diff)),
        op.Sub(
            end,
            op.Mul(diff, op.Sub(1.0, weight)),
        ),
    )


@onnx_impl(aten.log.default)
def aten_log(op: TranslationContext, self: ir.Value) -> ir.Value:
    """log(Tensor self) -> Tensor"""

    return op.Log(self)


@onnx_impl(aten.log10.default)
def aten_log10(op: TranslationContext, self: ir.Value) -> ir.Value:
    """log10(Tensor self) -> Tensor"""
    builder = op.builder

    return op.Div(
        op.Log(self),
        builder.constant(math.log(10.0), dtype=common_ops.require_dtype(op, self)),
    )


@onnx_impl(aten.log1p.default)
def aten_log1p(op: TranslationContext, self: ir.Value) -> ir.Value:
    """log1p(Tensor self) -> Tensor"""

    return op.Log(op.Add(self, 1.0))


@onnx_impl(aten.log2.default)
def aten_log2(op: TranslationContext, self: ir.Value) -> ir.Value:
    """log2(Tensor self) -> Tensor"""
    builder = op.builder

    return op.Div(
        op.Log(self),
        builder.constant(math.log(2.0), dtype=common_ops.require_dtype(op, self)),
    )


@onnx_impl(aten.logaddexp.default)
def aten_logaddexp(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """logaddexp(Tensor self, Tensor other) -> Tensor"""

    return op.Log(
        op.Add(op.Exp(self), op.Exp(other)),
    )


@onnx_impl(aten.logaddexp2.default)
def aten_logaddexp2(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    """logaddexp2(Tensor self, Tensor other) -> Tensor"""
    two = op.CastLike(2.0, self)
    summation = op.Add(op.Pow(two, self), op.Pow(two, other))

    return op.Div(op.Log(summation), op.Log(two))


@onnx_impl(aten.logical_and.default)
def aten_logical_and(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    """logical_and(Tensor self, Tensor other) -> Tensor"""

    if self.dtype != other.dtype:
        raise ValueError(
            f"aten_logical_and requires matching tensor dtypes, got {self.dtype} and {other.dtype}. Cast operands to the same dtype before translation."
        )

    if self.dtype == ir.DataType.BOOL:
        return op.And(self, other)
    return op.And(
        op.Cast(self, to=ir.DataType.BOOL),
        op.Cast(other, to=ir.DataType.BOOL),
    )


@onnx_impl(aten.logical_not.default)
def aten_logical_not(op: TranslationContext, self: ir.Value) -> ir.Value:
    """logical_not(Tensor self) -> Tensor"""

    if self.dtype == ir.DataType.BOOL:
        return op.Not(self)
    return op.Not(op.Cast(self, to=ir.DataType.BOOL))


@onnx_impl(aten.logical_or.default)
def aten_logical_or(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    """logical_or(Tensor self, Tensor other) -> Tensor"""

    if self.dtype != other.dtype:
        raise ValueError(
            f"aten_logical_or requires matching tensor dtypes, got {self.dtype} and {other.dtype}. Cast operands to the same dtype before translation."
        )

    if self.dtype == ir.DataType.BOOL:
        return op.Or(self, other)
    return op.Or(
        op.Cast(self, to=ir.DataType.BOOL),
        op.Cast(other, to=ir.DataType.BOOL),
    )


@onnx_impl(aten.logical_xor.default)
def aten_logical_xor(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    """logical_xor(Tensor self, Tensor other) -> Tensor"""

    if self.dtype != other.dtype:
        raise ValueError(
            f"aten_logical_xor requires matching tensor dtypes, got {self.dtype} and {other.dtype}. Cast operands to the same dtype before translation."
        )

    if self.dtype == ir.DataType.BOOL:
        return op.Xor(self, other)
    return op.Xor(
        op.Cast(self, to=ir.DataType.BOOL),
        op.Cast(other, to=ir.DataType.BOOL),
    )


@onnx_impl(aten.logit.default)
def aten_logit(
    op: TranslationContext, self: ir.Value, eps: float | None = None
) -> ir.Value:
    """logit(Tensor self, float? eps=None) -> Tensor"""
    builder = op.builder
    one = builder.constant(1, dtype=common_ops.require_dtype(op, self))

    if eps is None:
        return op.Log(op.Div(self, op.Sub(one, self)))

    one_minus_eps = builder.constant(1 - eps, dtype=common_ops.require_dtype(op, self))
    eps_tensor = builder.constant(eps, dtype=common_ops.require_dtype(op, self))

    # Logit's lower-bound branch takes precedence even when eps > 0.5.
    upper_clamped = op.Where(
        op.Greater(self, one_minus_eps),
        one_minus_eps,
        self,
    )
    z = op.Where(op.Less(self, eps_tensor), eps_tensor, upper_clamped)

    return op.Log(op.Div(z, op.Sub(one, z)))


@onnx_impl((aten.lt.Tensor, aten.lt.Scalar, aten.less.Tensor))
def aten_lt(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """lt.Tensor(Tensor self, Tensor other) -> Tensor"""

    if self.dtype == ir.DataType.BOOL:
        # self, other, self < other
        #    F,    F,    F
        #    F,    T,    T
        #    T,    F,    F
        #    T,    T,    F
        return op.And(other, op.Not(self))

    return op.Less(self, other)


@onnx_impl(operator.lt)
def operator_lt(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    # operator.lt for SymInt
    return op.Less(self, other)


@onnx_impl(aten.maximum.default)
def aten_maximum(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """maximum(Tensor self, Tensor other) -> Tensor"""

    if self.dtype == ir.DataType.BOOL:
        return op.Or(self, other)

    return op.Max(self, other)


@onnx_impl(aten.minimum.default)
def aten_minimum(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """minimum(Tensor self, Tensor other) -> Tensor"""

    if self.dtype == ir.DataType.BOOL:
        return op.And(self, other)

    return op.Min(self, other)


@onnx_impl((aten.mul.default, aten.mul.Tensor, aten.multiply.Tensor))
def aten_mul(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """mul.Tensor(Tensor self, Tensor other) -> Tensor"""

    if self.dtype == ir.DataType.BOOL:
        return op.And(self, other)

    return op.Mul(self, other)


@onnx_impl(operator.mul)
def operator_mul(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    return op.Mul(self, other)


@onnx_impl((aten.mul.default, aten.mul.Tensor, aten.multiply.Tensor), complex=True)
def aten_mul_complex(
    op: TranslationContext,
    self: ir.Value,
    other: ir.Value | complex | float | int | bool,
) -> ir.Value:
    """mul.Tensor(Tensor self, Tensor other) -> Tensor"""
    other = common_ops.complex_scalar(op, other, dtype=self.dtype)

    # TODO(justinchuby): Maybe use Split to simplify the logic
    self_real = op.Slice(self, [0], [1], [-1])
    self_imag = op.Slice(self, [1], [2], [-1])
    other_real = op.Slice(other, [0], [1], [-1])
    other_imag = op.Slice(other, [1], [2], [-1])

    # Complex multiplication
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i

    ac = op.Mul(self_real, other_real)
    bd = op.Mul(self_imag, other_imag)
    ad = op.Mul(self_real, other_imag)
    bc = op.Mul(self_imag, other_real)

    real = op.Sub(ac, bd)
    imag = op.Add(ad, bc)

    return op.Concat(real, imag, axis=-1)


@onnx_impl((aten.mul.Scalar, aten.multiply.Scalar))
def aten_mul_scalar(op: TranslationContext, self: ir.Value, other: float) -> ir.Value:
    """mul.Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = op.builder

    other_tensor = builder.constant(other, dtype=common_ops.require_dtype(op, self))
    return aten_mul(op, self, other_tensor)


@onnx_impl((aten.ne.default, aten.ne.Scalar, aten.ne.Tensor, operator.ne))
def aten_ne(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """ne.Tensor(Tensor self, Tensor other) -> Tensor"""

    return op.Not(op.Equal(self, other))


@onnx_impl((aten.neg.default, operator.neg))
def aten_neg(op: TranslationContext, self: ir.Value) -> ir.Value:
    """neg(Tensor self) -> Tensor"""

    return op.Neg(self)


@onnx_impl((aten.neg.default, operator.neg), complex=True)
def aten_neg_complex(op: TranslationContext, self: ir.Value) -> ir.Value:
    """neg(Tensor self) -> Tensor"""

    return op.Neg(self)


@onnx_impl((aten.pow.Tensor_Tensor, operator.pow))
def aten_pow(op: TranslationContext, self: ir.Value, exponent: ir.Value) -> ir.Value:
    """pow(Tensor self, Tensor exponent) -> Tensor"""
    # TODO(justinchuby): Add type promotion
    return op.Pow(self, exponent)


@onnx_impl(aten.pow.Tensor_Scalar)
def aten_pow_tensor_scalar(
    op: TranslationContext, self: ir.Value, exponent: float
) -> ir.Value:
    """pow(Tensor self, Scalar exponent) -> Tensor"""
    builder = op.builder
    dtype = common_ops.require_dtype(op, self)
    if dtype.is_floating_point():
        # Handle cases when e.g. (1) self is float16 or int
        return op.Pow(self, builder.constant(exponent, dtype=dtype))
    # For integer types, we need to cast self to the exponent type
    if isinstance(exponent, int):
        # The scalar exponent can be an int
        return op.Pow(self, builder.constant(exponent, dtype=dtype))

    # exponent is float so we cast self to match the exponent type.
    # More precisely if self is float64, we should cast exponent to float64; but
    # this is uncommon and should be fixed when we create a general type promotion
    # mechanism for torchlib
    return op.Pow(op.Cast(self, to=ir.DataType.FLOAT), exponent)


@onnx_impl(aten.pow.Scalar)
def aten_pow_scalar(
    op: TranslationContext, self: float, exponent: ir.Value
) -> ir.Value:
    """pow.Scalar(Scalar self, Tensor exponent) -> Tensor"""
    builder = op.builder
    dtype = common_ops.require_dtype(op, exponent)
    if not isinstance(self, int) and not dtype.is_floating_point():
        # A float scalar outranks an integral exponent, so torch promotes the result to
        # the default float type instead of narrowing the scalar down to the exponent
        return op.Pow(
            builder.constant(self, dtype=ir.DataType.FLOAT),
            op.Cast(exponent, to=ir.DataType.FLOAT),
        )
    if dtype == ir.DataType.BOOL:
        # Pow has no boolean inputs, and an int scalar over a boolean exponent
        # promotes to the default int type in torch
        return op.Pow(
            builder.constant(self, dtype=ir.DataType.INT64),
            op.Cast(exponent, to=ir.DataType.INT64),
        )
    # The exponent is in the same or a higher type category than the scalar, so it
    # decides the result type. e.g. 2.0 ** float16 tensor is float16
    return op.Pow(builder.constant(self, dtype=dtype), exponent)


@onnx_impl(aten.rad2deg.default)
def aten_rad2deg(op: TranslationContext, self: ir.Value) -> ir.Value:
    """rad2deg(Tensor self) -> Tensor"""

    return op.Mul(self, op.CastLike(180.0 / math.pi, self))


@onnx_impl(aten.reciprocal.default)
def aten_reciprocal(op: TranslationContext, self: ir.Value) -> ir.Value:
    """reciprocal(Tensor self) -> Tensor"""

    return op.Reciprocal(self)


@onnx_impl(aten.reciprocal.default, complex=True)
def aten_reciprocal_complex(op: TranslationContext, self: ir.Value) -> ir.Value:
    """reciprocal(Tensor self) -> Tensor"""

    self_real = op.Slice(self, [0], [1], [-1])
    self_imag = op.Slice(self, [1], [2], [-1])

    # Complex reciprocal
    # 1 / (c + di) = c / (c^2 + d^2) - d / (c^2 + d^2) i
    denominator = op.Add(
        op.Mul(self_real, self_real),
        op.Mul(self_imag, self_imag),
    )
    real = op.Div(self_real, denominator)
    imag = op.Div(op.Neg(self_imag), denominator)

    return op.Concat(real, imag, axis=-1)


def _aten_remainder(
    op: TranslationContext, self: ir.Value, other: ir.Value, integer: bool
) -> ir.Value:
    if integer:
        return op.Mod(self, other)

    # TODO(justinchuby): Improve fp16 precision by following the logic in
    # https://github.com/pytorch/pytorch/blob/3a823e46170778cc32783f27596c77d0103084a9/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp#L264-L277

    # a - a.div(b, rounding_mode="floor") * b
    rounded_quotient = op.Floor(op.Div(self, other))

    return op.Sub(self, op.Mul(rounded_quotient, other))


@onnx_impl((aten.remainder.Tensor, prims.remainder.default))
def aten_remainder(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """remainder.Tensor(Tensor self, Tensor other) -> Tensor"""
    return _aten_remainder(
        op, self, other, integer=common_ops.require_dtype(op, self).is_integer()
    )


@onnx_impl(aten.remainder.Scalar)
def aten_remainder_scalar(
    op: TranslationContext, self: ir.Value, other: float
) -> ir.Value:
    """remainder.Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = op.builder

    dtype = common_ops.require_dtype(op, self)
    other_tensor = builder.constant(other, dtype=dtype)
    return _aten_remainder(op, self, other_tensor, integer=dtype.is_integer())


@onnx_impl(aten.remainder.Scalar_Tensor)
def aten_remainder_scalar_tensor(
    op: TranslationContext, self: float, other: ir.Value
) -> ir.Value:
    """remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = op.builder

    dtype = common_ops.require_dtype(op, other)
    self_tensor = builder.constant(self, dtype=dtype)
    return _aten_remainder(op, self_tensor, other, integer=dtype.is_integer())


@onnx_impl(operator.mod)
def operator_mod(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    # Modulus operator % on SymInt
    return op.Mod(self, other)


@onnx_impl(aten.round.default)
def aten_round(op: TranslationContext, self: ir.Value) -> ir.Value:
    """round(Tensor self) -> Tensor"""

    return op.Round(self)


@onnx_impl(aten.round.decimals)
def aten_round_decimals(
    op: TranslationContext, self: ir.Value, decimals: int = 0
) -> ir.Value:
    """round.decimals(Tensor self, *, int decimals) -> Tensor"""

    # Scale the input by 10^decimals, round it, and scale it back.
    ten = op.CastLike(10.0, self)
    scale = op.Pow(ten, op.CastLike(decimals, self))
    self_scaled = op.Mul(self, scale)
    rounded = op.Round(self_scaled)
    return op.Div(rounded, scale)


@onnx_impl(aten.rsqrt.default)
def aten_rsqrt(op: TranslationContext, self: ir.Value) -> ir.Value:
    """rsqrt(Tensor self) -> Tensor"""

    return op.Reciprocal(op.Sqrt(self))


@onnx_impl(aten.scalar_tensor.default)
def aten_scalar_tensor(
    op: TranslationContext,
    s: ir.Value | float | int | bool,
    dtype: int = ir.DataType.FLOAT,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    builder = op.builder
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
    return common_ops.cast_to(op, s, dtype=dtype)


@onnx_impl(aten.scalar_tensor.default, complex=True)
def aten_scalar_tensor_complex(
    op: TranslationContext,
    s: ir.Value,
    dtype: int = ir.DataType.COMPLEX64,
    layout: str = "",
    device: str = "",
    pin_memory: bool = False,
) -> ir.Value:
    """scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    # NOTE: When the input is originally in complex, this function is invoked.
    # On the other hand, when the input is originally in real, aten_scalar_tensor is used.
    # is invoked.
    if dtype == -1:
        dtype = ir.DataType.COMPLEX64
    if dtype == ir.DataType.COMPLEX128:
        result = op.Cast(s, to=ir.DataType.DOUBLE)
    elif dtype == ir.DataType.COMPLEX64:
        result = op.Cast(s, to=ir.DataType.FLOAT)
    else:
        # The source's no-op for non-complex dtype could hide an invalid conversion.
        raise TypeError(
            f"scalar_tensor with complex input cannot use dtype {dtype}. "
            "Use COMPLEX64 or COMPLEX128 for complex scalars."
        )
    return result


@onnx_impl(aten.sign.default)
def aten_sign(op: TranslationContext, self: ir.Value) -> ir.Value:
    """sign(Tensor self) -> Tensor"""

    if self.dtype == ir.DataType.BOOL:
        return op.Identity(self)

    return op.Sign(self)


@onnx_impl(aten.sin.default)
def aten_sin(op: TranslationContext, self: ir.Value) -> ir.Value:
    """sin(Tensor self) -> Tensor"""

    return op.Sin(self)


@onnx_impl(aten.sinh.default)
def aten_sinh(op: TranslationContext, self: ir.Value) -> ir.Value:
    """sinh(Tensor self) -> Tensor"""

    return op.Sinh(self)


@onnx_impl(aten.sqrt.default)
def aten_sqrt(op: TranslationContext, self: ir.Value) -> ir.Value:
    """sqrt(Tensor self) -> Tensor"""

    return op.Sqrt(self)


@onnx_impl((aten.sub.Tensor, aten.subtract.Tensor, operator.sub))
def aten_sub(
    op: TranslationContext, self: ir.Value, other: ir.Value, alpha: float = 1.0
) -> ir.Value:
    """sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    if alpha != 1.0:
        alpha = op.CastLike(alpha, other)
        other = op.Mul(other, alpha)
    return op.Sub(self, other)


@onnx_impl((aten.sub.Scalar, aten.subtract.Scalar))
def aten_sub_scalar(
    op: TranslationContext, self: ir.Value, other: float, alpha: float = 1.0
) -> ir.Value:
    """sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor"""
    builder = op.builder

    other_tensor = builder.constant(other, dtype=common_ops.require_dtype(op, self))
    return aten_sub(op, self, other_tensor, alpha=alpha)


@onnx_impl(
    (aten.sub.Tensor, aten.sub.Scalar, aten.subtract.Tensor, aten.subtract.Scalar),
    complex=True,
)
def aten_sub_complex(
    op: TranslationContext,
    self: ir.Value,
    other: ir.Value | complex | float | int | bool,
    alpha: float = 1.0,
) -> ir.Value:
    """sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    other = common_ops.complex_scalar(op, other, dtype=self.dtype)
    return aten_sub(op, self, other, alpha=alpha)


@onnx_impl(aten.tan.default)
def aten_tan(op: TranslationContext, self: ir.Value) -> ir.Value:
    """tan(Tensor self) -> Tensor"""

    return op.Tan(self)


@onnx_impl(aten.tanh.default)
def aten_tanh(op: TranslationContext, self: ir.Value) -> ir.Value:
    """tanh(Tensor self) -> Tensor"""

    return op.Tanh(self)


@onnx_impl(aten.trunc.default)
def aten_trunc(op: TranslationContext, self: ir.Value) -> ir.Value:
    """trunc(Tensor self) -> Tensor"""
    # Reference https://github.com/onnx/onnx/issues/4588#issuecomment-2658170591
    return op.Mul(
        op.Floor(op.Abs(self)),
        op.Sign(self),
    )


@onnx_impl(math.trunc)
def python_math_trunc(op: TranslationContext, self: ir.Value) -> ir.Value:
    """trunc(Tensor self) -> Tensor"""
    # NOTE: This is used in SymInt/SymBool/SymFloat context, so
    # we don't expect overflow to happen here.
    return op.Cast(self, to=ir.DataType.INT64)


@onnx_impl(aten.type_as.default)
def aten_type_as(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """type_as(Tensor self, Tensor other) -> Tensor"""

    return op.CastLike(self, other)


_INT64_MAX = 9223372036854775807
_INT64_MIN = -9223372036854775808
_AS_STRIDED_STATIC_INDEX_SIZE_LIMIT = 512 * 512


@onnx_impl(aten.all.default)
def aten_all(op: TranslationContext, self: ir.Value) -> ir.Value:
    """all(Tensor self) -> Tensor"""

    if len(common_ops.require_shape(op, self)) == 0:
        result = op.Cast(self, to=ir.DataType.BOOL)
    else:
        self_bool = op.Cast(self, to=ir.DataType.BOOL)
        self_int = op.Cast(self_bool, to=ir.DataType.INT64)
        all_true = op.ReduceMin(self_int, keepdims=False)
        result = op.Cast(all_true, to=ir.DataType.BOOL)
    return result


@onnx_impl(aten.all.dim)
def aten_all_dim(
    op: TranslationContext, self: ir.Value, dim: int, keepdim: bool = False
) -> ir.Value:
    """all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor"""
    builder = op.builder
    if len(common_ops.require_shape(op, self)) == 0:
        return op.Cast(self, to=ir.DataType.BOOL)

    self_bool = op.Cast(self, to=ir.DataType.BOOL)
    self_int = op.Cast(self_bool, to=ir.DataType.INT64)
    dims = op.Reshape(dim, builder.constant([-1], dtype=ir.DataType.INT64))
    all_true = op.ReduceMin(self_int, dims, keepdims=keepdim)
    return op.Cast(all_true, to=ir.DataType.BOOL)


@onnx_impl(aten.all.dims)
def aten_all_dims(
    op: TranslationContext,
    self: ir.Value,
    dim: Sequence[int] = (),
    keepdim: bool = False,
) -> ir.Value:
    """all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor"""

    if not dim:
        return _aten_all_dims_no_dim(op, self, keepdim)
    if len(common_ops.require_shape(op, self)) == 0:
        return op.Cast(self, to=ir.DataType.BOOL)
    self_bool = op.Cast(self, to=ir.DataType.BOOL)
    self_int = op.Cast(self_bool, to=ir.DataType.INT64)
    result = op.ReduceMin(self_int, list(dim), keepdims=keepdim)
    return op.Cast(result, to=ir.DataType.BOOL)


def _aten_all_dims_no_dim(
    op: TranslationContext, self: ir.Value, keepdims: bool
) -> ir.Value:
    """all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor"""

    if len(common_ops.require_shape(op, self)) == 0:
        result = op.Cast(self, to=ir.DataType.BOOL)
    else:
        self_bool = op.Cast(self, to=ir.DataType.BOOL)
        self_int = op.Cast(self_bool, to=ir.DataType.INT64)
        all_true = op.ReduceMin(self_int, keepdims=keepdims)
        result = op.Cast(all_true, to=ir.DataType.BOOL)
    return result


@onnx_impl(aten.amax.default)
def aten_amax(
    op: TranslationContext,
    self: ir.Value,
    dim: ir.Value | None = None,
    keepdim: bool = False,
) -> ir.Value:
    """amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"""

    if dim is None:
        # dim defaults to the empty list in the aten schema, which means reduce every
        # dimension. noop_with_empty_axes keeps its default of 0, so ReduceMax without
        # an axes input reduces all of them.
        return op.ReduceMax(self, keepdims=keepdim)
    # An explicitly empty dim arrives here and reduces every dimension for the same reason
    return op.ReduceMax(self, dim, keepdims=keepdim)


@onnx_impl(aten.amin.default)
def aten_amin(
    op: TranslationContext,
    self: ir.Value,
    dim: ir.Value | None = None,
    keepdim: bool = False,
) -> ir.Value:
    """amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"""

    if dim is None:
        # dim defaults to the empty list in the aten schema, which means reduce every
        # dimension. noop_with_empty_axes keeps its default of 0, so ReduceMin without
        # an axes input reduces all of them.
        return op.ReduceMin(self, keepdims=keepdim)
    # An explicitly empty dim arrives here and reduces every dimension for the same reason
    return op.ReduceMin(self, dim, keepdims=keepdim)


@onnx_impl(aten.any.default)
def aten_any(op: TranslationContext, self: ir.Value) -> ir.Value:
    """any(Tensor self) -> Tensor"""

    if len(common_ops.require_shape(op, self)) == 0:
        result = op.Cast(self, to=ir.DataType.BOOL)
    else:
        self_bool = op.Cast(self, to=ir.DataType.BOOL)
        # op.ReduceMax() in the next step cannot process BOOL inputs, so convert to INT64
        self_int = op.Cast(self_bool, to=ir.DataType.INT64)
        any_true = op.ReduceMax(self_int, keepdims=False)
        result = op.Greater(any_true, 0)
    return result


@onnx_impl(aten.any.dim)
def aten_any_dim(
    op: TranslationContext, self: ir.Value, dim: int, keepdim: bool = False
) -> ir.Value:
    """any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor"""
    builder = op.builder
    if len(common_ops.require_shape(op, self)) == 0:
        return op.Cast(self, to=ir.DataType.BOOL)

    self_bool = op.Cast(self, to=ir.DataType.BOOL)
    # op.ReduceMax() in the next step cannot process BOOL inputs, so convert to INT64
    self_int = op.Cast(self_bool, to=ir.DataType.INT64)
    # Change dim from int to INT64[1]
    dims = op.Reshape(dim, builder.constant([-1], dtype=ir.DataType.INT64))
    any_true = op.ReduceMax(self_int, dims, keepdims=keepdim)
    return op.Greater(any_true, 0)


@onnx_impl(aten.any.dims)
def aten_any_dims(
    op: TranslationContext,
    self: ir.Value,
    dim: Sequence[int] = (),
    keepdim: bool = False,
) -> ir.Value:
    """any.dims(Tensor self, int[1]? dim=None, bool keepdim=False) -> Tensor"""

    if not dim:
        return _aten_any_dims_no_dim(op, self, keepdim)
    if len(common_ops.require_shape(op, self)) == 0:
        return op.Cast(self, to=ir.DataType.BOOL)
    self_bool = op.Cast(self, to=ir.DataType.BOOL)
    self_int = op.Cast(self_bool, to=ir.DataType.INT64)
    result = op.ReduceMax(self_int, list(dim), keepdims=keepdim)
    return op.Greater(result, 0)


def _aten_any_dims_no_dim(
    op: TranslationContext, self: ir.Value, keepdims: bool
) -> ir.Value:
    if len(common_ops.require_shape(op, self)) == 0:
        result = op.Cast(self, to=ir.DataType.BOOL)
    else:
        self_bool = op.Cast(self, to=ir.DataType.BOOL)
        self_int = op.Cast(self_bool, to=ir.DataType.INT64)
        any_true = op.ReduceMax(self_int, keepdims=keepdims)
        result = op.Greater(any_true, 0)
    return result


@onnx_impl(aten.argmax.default)
def aten_argmax(
    op: TranslationContext,
    self: ir.Value,
    dim: int | None = None,
    keepdim: bool = False,
) -> ir.Value:
    """argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""

    if dim is None:
        result = _aten_argmax(op, self, keepdim)
    else:
        result = _aten_argmax_dim(op, self, dim, keepdim)
    return result


def _aten_argmax(
    op: TranslationContext, self: ir.Value, keepdim: bool = False
) -> ir.Value:
    """argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""
    builder = op.builder

    self_is_scaler = len(common_ops.require_shape(op, self)) == 0
    self = op.Reshape(self, builder.constant([-1], dtype=ir.DataType.INT64))
    result = op.ArgMax(self, keepdims=keepdim)
    if self_is_scaler:
        result = op.Squeeze(result)

    return result


def _aten_argmax_dim(
    op: TranslationContext, self: ir.Value, dim: int, keepdim: bool = False
) -> ir.Value:
    """argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""
    builder = op.builder

    self_is_scaler = len(common_ops.require_shape(op, self)) == 0
    if self_is_scaler:
        self = op.Reshape(self, builder.constant([-1], dtype=ir.DataType.INT64))

    result = op.ArgMax(self, axis=dim, keepdims=keepdim)
    if self_is_scaler:
        result = op.Squeeze(result)

    return result


@onnx_impl(aten.argmin.default)
def aten_argmin(
    op: TranslationContext,
    self: ir.Value,
    dim: int | None = None,
    keepdim: bool = False,
) -> ir.Value:
    """argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""

    if dim is None:
        result = _aten_argmin(op, self, keepdim)
    else:
        result = _aten_argmin_dim(op, self, dim, keepdim)
    return result


def _aten_argmin(
    op: TranslationContext, self: ir.Value, keepdim: bool = False
) -> ir.Value:
    """argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""
    builder = op.builder

    self_is_scaler = len(common_ops.require_shape(op, self)) == 0
    self = op.Reshape(self, builder.constant([-1], dtype=ir.DataType.INT64))
    result = op.ArgMin(self, keepdims=keepdim)
    if self_is_scaler:
        result = op.Squeeze(result)

    return result


def _aten_argmin_dim(
    op: TranslationContext, self: ir.Value, dim: int, keepdim: bool = False
) -> ir.Value:
    """argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"""
    builder = op.builder

    self_is_scaler = len(common_ops.require_shape(op, self)) == 0
    if self_is_scaler:
        self = op.Reshape(self, builder.constant([-1], dtype=ir.DataType.INT64))

    result = op.ArgMin(self, axis=dim, keepdims=keepdim)
    if self_is_scaler:
        result = op.Squeeze(result)

    return result


@onnx_impl(aten.atleast_1d.default)
def aten_atleast_1d(op: TranslationContext, self: ir.Value) -> ir.Value:
    """atleast_1d(Tensor self) -> Tensor"""
    builder = op.builder

    if len(common_ops.require_shape(op, self)) == 0:
        self = op.Reshape(self, builder.constant([1], dtype=ir.DataType.INT64))
    return op.Identity(self)


@onnx_impl(aten.atleast_2d.default)
def aten_atleast_2d(op: TranslationContext, self: ir.Value) -> ir.Value:
    """atleast_2d(Tensor self) -> Tensor"""
    builder = op.builder

    if len(common_ops.require_shape(op, self)) <= 1:
        self = op.Reshape(self, builder.constant([1, -1], dtype=ir.DataType.INT64))
    return op.Identity(self)


@onnx_impl(aten.atleast_3d.default)
def aten_atleast_3d(op: TranslationContext, self: ir.Value) -> ir.Value:
    """atleast_3d(Tensor self) -> Tensor"""
    builder = op.builder

    rank = len(common_ops.require_shape(op, self))
    if rank <= 1:
        self = op.Reshape(self, builder.constant([1, -1, 1], dtype=ir.DataType.INT64))
    elif rank == 2:
        self = op.Unsqueeze(self, builder.constant([-1], dtype=ir.DataType.INT64))
    return op.Identity(self)


@onnx_impl(aten.broadcast_to.default)
def aten_broadcast_to(
    op: TranslationContext, self: ir.Value, size: Sequence[int | ir.Value]
) -> ir.Value:
    """broadcast_to(Tensor(a) self, SymInt[] size) -> Tensor(a)"""
    target_size = common_ops.merge_dims(op, size)
    return op.Expand(self, target_size)


@onnx_impl(aten.cumsum.default)
def aten_cumsum(
    op: TranslationContext, self: ir.Value, dim: ir.Value, dtype: int = -1
) -> ir.Value:
    """cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"""

    # The accumulation type for int32 is int64.
    if dtype != -1 and dtype is not None:
        cast = op.Cast(self, to=dtype)
    elif (
        common_ops.require_dtype(op, self).is_integer()
        or self.dtype == ir.DataType.BOOL
    ):
        cast = op.Cast(self, to=ir.DataType.INT64)
    else:
        cast = self
    if len(common_ops.require_shape(op, self)) == 0:
        # A scalar
        result = op.Identity(cast)
    else:
        result = op.CumSum(cast, dim)
    return result


@onnx_impl(aten.expand.default)
def aten_expand(
    op: TranslationContext,
    self: ir.Value,
    size: Sequence[int | ir.Value],
    implicit: bool = False,
) -> ir.Value:
    """expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)"""
    # NOTE: PyTorch supports `not changing dim` by -1, but ONNX supports `not changing dim` by 1.
    # To support -1 dim, we need to convert -1 to 1.
    # Even though in theory a dynamic dim can still be -1, in practice it is very unlikely
    # and isn't expected to appear from correct usages of SymInt.
    size = [1 if isinstance(s, int) and s == -1 else s for s in size]
    return op.Expand(self, common_ops.merge_dims(op, size))


@onnx_impl(aten.expand_as.default)
def aten_expand_as(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """expand_as(Tensor(a) self, Tensor other) -> Tensor(a)"""

    shape = op.Shape(other)
    result = op.Expand(self, shape)
    return result
