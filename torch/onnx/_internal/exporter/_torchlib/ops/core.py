# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned core ONNX IR translations."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
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
        builder.constant(math.log(10.0), dtype=self.dtype),
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
        builder.constant(math.log(2.0), dtype=self.dtype),
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
    one = builder.constant(1, dtype=self.dtype)

    if eps is None:
        return op.Log(op.Div(self, op.Sub(one, self)))

    one_minus_eps = builder.constant(1 - eps, dtype=self.dtype)
    eps = builder.constant(eps, dtype=self.dtype)

    # Logit's lower-bound branch takes precedence even when eps > 0.5.
    upper_clamped = op.Where(
        op.Greater(self, one_minus_eps),
        one_minus_eps,
        self,
    )
    z = op.Where(op.Less(self, eps), eps, upper_clamped)

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

    other = builder.constant(other, dtype=self.dtype)
    return aten_mul(op, self, other)


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
    if self.dtype.is_floating_point():
        # Handle cases when e.g. (1) self is float16 or int
        return op.Pow(self, builder.constant(exponent, dtype=self.dtype))
    # For integer types, we need to cast self to the exponent type
    if isinstance(exponent, int):
        # The scalar exponent can be an int
        return op.Pow(self, builder.constant(exponent, dtype=self.dtype))

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
    if not isinstance(self, int) and not exponent.dtype.is_floating_point():
        # A float scalar outranks an integral exponent, so torch promotes the result to
        # the default float type instead of narrowing the scalar down to the exponent
        return op.Pow(
            builder.constant(self, dtype=ir.DataType.FLOAT),
            op.Cast(exponent, to=ir.DataType.FLOAT),
        )
    if exponent.dtype == ir.DataType.BOOL:
        # Pow has no boolean inputs, and an int scalar over a boolean exponent
        # promotes to the default int type in torch
        return op.Pow(
            builder.constant(self, dtype=ir.DataType.INT64),
            op.Cast(exponent, to=ir.DataType.INT64),
        )
    # The exponent is in the same or a higher type category than the scalar, so it
    # decides the result type. e.g. 2.0 ** float16 tensor is float16
    return op.Pow(builder.constant(self, dtype=exponent.dtype), exponent)


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
    return _aten_remainder(op, self, other, integer=self.dtype.is_integer())


@onnx_impl(aten.remainder.Scalar)
def aten_remainder_scalar(
    op: TranslationContext, self: ir.Value, other: float
) -> ir.Value:
    """remainder.Scalar(Tensor self, Scalar other) -> Tensor"""
    builder = op.builder

    other_tensor = builder.constant(other, dtype=self.dtype)
    return _aten_remainder(op, self, other_tensor, integer=self.dtype.is_integer())


@onnx_impl(aten.remainder.Scalar_Tensor)
def aten_remainder_scalar_tensor(
    op: TranslationContext, self: float, other: ir.Value
) -> ir.Value:
    """remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor"""
    builder = op.builder

    self_tensor = builder.constant(self, dtype=other.dtype)
    return _aten_remainder(op, self_tensor, other, integer=other.dtype.is_integer())


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

    other = builder.constant(other, dtype=self.dtype)
    return aten_sub(op, self, other, alpha=alpha)


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

    if len(self.shape) == 0:
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
    if len(self.shape) == 0:
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
    if len(self.shape) == 0:
        return op.Cast(self, to=ir.DataType.BOOL)
    self_bool = op.Cast(self, to=ir.DataType.BOOL)
    self_int = op.Cast(self_bool, to=ir.DataType.INT64)
    result = op.ReduceMin(self_int, list(dim), keepdims=keepdim)
    return op.Cast(result, to=ir.DataType.BOOL)


def _aten_all_dims_no_dim(
    op: TranslationContext, self: ir.Value, keepdims: bool
) -> ir.Value:
    """all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor"""

    if len(self.shape) == 0:
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

    if len(self.shape) == 0:
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
    if len(self.shape) == 0:
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
    if len(self.shape) == 0:
        return op.Cast(self, to=ir.DataType.BOOL)
    self_bool = op.Cast(self, to=ir.DataType.BOOL)
    self_int = op.Cast(self_bool, to=ir.DataType.INT64)
    result = op.ReduceMax(self_int, list(dim), keepdims=keepdim)
    return op.Greater(result, 0)


def _aten_any_dims_no_dim(
    op: TranslationContext, self: ir.Value, keepdims: bool
) -> ir.Value:
    if len(self.shape) == 0:
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

    self_is_scaler = len(self.shape) == 0
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

    self_is_scaler = len(self.shape) == 0
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

    self_is_scaler = len(self.shape) == 0
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

    self_is_scaler = len(self.shape) == 0
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

    if len(self.shape) == 0:
        self = op.Reshape(self, builder.constant([1], dtype=ir.DataType.INT64))
    return op.Identity(self)


@onnx_impl(aten.atleast_2d.default)
def aten_atleast_2d(op: TranslationContext, self: ir.Value) -> ir.Value:
    """atleast_2d(Tensor self) -> Tensor"""
    builder = op.builder

    if len(self.shape) <= 1:
        self = op.Reshape(self, builder.constant([1, -1], dtype=ir.DataType.INT64))
    return op.Identity(self)


@onnx_impl(aten.atleast_3d.default)
def aten_atleast_3d(op: TranslationContext, self: ir.Value) -> ir.Value:
    """atleast_3d(Tensor self) -> Tensor"""
    builder = op.builder

    rank = len(self.shape)
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
    size = common_ops.merge_dims(op, size)
    return op.Expand(self, size)


@onnx_impl(aten.cumsum.default)
def aten_cumsum(
    op: TranslationContext, self: ir.Value, dim: ir.Value, dtype: int = -1
) -> ir.Value:
    """cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"""

    # The accumulation type for int32 is int64.
    if dtype != -1 and dtype is not None:
        cast = op.Cast(self, to=dtype)
    elif self.dtype.is_integer() or self.dtype == ir.DataType.BOOL:
        cast = op.Cast(self, to=ir.DataType.INT64)
    else:
        cast = self
    if len(self.shape) == 0:
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


@onnx_impl(aten.flatten.using_ints)
def aten_flatten(
    op: TranslationContext, self: ir.Value, start_dim: int = 0, end_dim: int = -1
) -> ir.Value:
    """flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)"""
    builder = op.builder
    dim = len(self.shape)
    if dim == 0:
        return op.Reshape(self, [1])
    if start_dim < 0:
        start_dim += dim
    if dim == 1:
        return op.Identity(self)
    # use ONNX's Flatten operator for cases where the output shape is 2D
    if start_dim == 1:
        if end_dim in (-1, dim - 1):
            return op.Flatten(self, axis=start_dim)
    elif start_dim == 0:
        if end_dim in (-2, dim - 2):
            return op.Flatten(self, axis=end_dim + 1)

    # if end_dim is negative add dim
    if end_dim < 0:
        end_dim = dim + end_dim

    input_size = op.Shape(self)
    dim_head = op.Slice(
        input_size,
        builder.constant([0], dtype=ir.DataType.INT64),
        builder.constant([start_dim], dtype=ir.DataType.INT64),
        builder.constant([0], dtype=ir.DataType.INT64),
    )
    final_dims = [dim_head, builder.constant([-1], dtype=ir.DataType.INT64)]
    if end_dim < dim - 1:
        dim_tail = op.Slice(
            input_size,
            builder.constant([end_dim + 1], dtype=ir.DataType.INT64),
            builder.constant([dim], dtype=ir.DataType.INT64),
            builder.constant([0], dtype=ir.DataType.INT64),
        )
        final_dims = [
            dim_head,
            builder.constant([-1], dtype=ir.DataType.INT64),
            dim_tail,
        ]

    final_shape = op.Concat(*final_dims, axis=0)
    return op.Reshape(self, final_shape)


@onnx_impl(aten.flip.default)
def aten_flip(op: TranslationContext, self: ir.Value, dims: Sequence[int]) -> ir.Value:
    """flip(Tensor self, int[] dims) -> Tensor"""
    builder = op.builder

    if not dims:
        # Nothing to flip
        return op.Identity(self)

    rank = len(dims)
    starts = builder.constant(
        [-1] * rank, dtype=ir.DataType.INT64
    )  # something like [-1, -1, -1]
    steps = starts  # something like [-1, -1, -1]
    ends = builder.constant(
        [_INT64_MIN] * rank, dtype=ir.DataType.INT64
    )  # something like [-xxx, -xxx, -xxx]
    dims = builder.constant(dims, dtype=ir.DataType.INT64)
    return op.Slice(self, starts, ends, dims, steps)


@onnx_impl(aten.flip.default, complex=True)
def aten_flip_complex(
    op: TranslationContext, self: ir.Value, dims: Sequence[int]
) -> ir.Value:
    """flip(Tensor self, int[] dims) -> Tensor"""
    # if dims is negative, take into account the complex dimension
    dims = [d - 1 if d < 0 else d for d in dims]
    return aten_flip(op, self, dims)


@onnx_impl(aten.gather.default)
def aten_gather(
    op: TranslationContext,
    self: ir.Value,
    dim: int,
    index: ir.Value,
    sparse_grad: bool = False,
) -> ir.Value:
    """gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor"""

    if len(self.shape) == 0:
        if len(index.shape) == 0:
            return op.Identity(self)
        else:
            return op.Expand(self, op.Shape(index))

    is_scalar_index = len(index.shape) == 0
    if is_scalar_index:
        index = op.Unsqueeze(index, [0])

    index = op.Cast(index, to=ir.DataType.INT64)
    result = op.GatherElements(self, index, axis=dim)

    if is_scalar_index:
        result = op.Squeeze(result, [0])
    return result


def _are_consecutive(op: TranslationContext, sorted_list: Sequence[int]) -> bool:
    """Returns True if a sorted list contains consecutive numbers."""
    if not sorted_list:
        return True

    return sorted_list == list(range(min(sorted_list), max(sorted_list) + 1))


def _has_none_in_middle(
    op: TranslationContext, indices: Sequence[ir.Value | None]
) -> bool:
    """Returns True if there is a None in the middle of the list."""
    not_none_indices = [i for i, idx in enumerate(indices) if idx is not None]
    return not _are_consecutive(op, not_none_indices)


def _shape_of_broadcast_tensors(op: TranslationContext, *args: ir.Value) -> ir.Value:
    """Returns the broadcasted shape of the given tensors."""
    broadcasted = op.Max(*args)
    return op.Shape(broadcasted)


def _aten_index_onnx(
    op: TranslationContext,
    self: ir.Value,
    indices: Sequence[ir.Value | None],
    index_ranks: Sequence[int],
) -> ir.Value:
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
    self = op.Transpose(self, perm=reordered_positions)

    # Broadcast the indices to the same shape then concatenate
    not_none_indices = [idx for idx in indices if idx is not None]
    broadcast_shape = _shape_of_broadcast_tensors(op, *not_none_indices)
    final_index = op.Concat(
        *(
            op.Unsqueeze(op.Expand(idx, broadcast_shape), [-1])
            for idx in not_none_indices
        ),
        axis=-1,
    )

    self = op.GatherND(self, final_index, batch_dims=0)

    if _has_none_in_middle(op, indices):
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

    return op.Transpose(self, perm=perm)


@onnx_impl((aten.index.Tensor, aten._unsafe_index.Tensor))
def aten_index(
    op: TranslationContext, self: ir.Value, indices: Sequence[ir.Value | None]
) -> ir.Value:
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
        return _aten_index_bool(op, self, indices)

    index_ranks = [len(index.shape) for index in indices if index is not None]

    return _aten_index_onnx(op, self, indices, index_ranks)


def _aten_index_bool(
    op: TranslationContext, self: ir.Value, indices: Sequence[ir.Value | None]
) -> ir.Value:
    index_ranks = [len(index.shape) for index in indices if index is not None]

    if all(rank == 1 for rank in index_ranks):
        # indices contains scalar only.
        new_indices = [
            op.Transpose(op.NonZero(index), perm=[1, 0]) if index is not None else None
            for index in indices
        ]
        new_indices = [
            op.Squeeze(index, [1]) if index is not None else None
            for index in new_indices
        ]
        return _aten_index_onnx(op, self, new_indices, index_ranks)

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
                self = op.Transpose(self, perm=trans_perm)
                count_of_none += 1
                continue

            new_indices = op.Transpose(op.NonZero(index), perm=[1, 0])
            result = op.GatherND(self, new_indices, batch_dims=0)
            final_rank = input_rank - (len(index.shape) - 1)
            trans_perm = list(range(final_rank))
            trans_perm = trans_perm[-1:] + trans_perm[:-1]
            for _ in range(count_of_none):
                result = op.Transpose(result, perm=trans_perm)
            # FIXME(justinchuby): Even though this logic passes the tests, it still looks strange:
            # why does it return early here instead of continuing to process the remaining indices?
            # I think the assumption here is that there can be only one Boolean index in the indices list?
            return result


@onnx_impl(aten.index_select.default)
def aten_index_select(
    op: TranslationContext, self: ir.Value, dim: int, index: ir.Value
) -> ir.Value:
    """index_select(Tensor self, int dim, Tensor index) -> Tensor"""
    builder = op.builder

    self_is_scalar = len(self.shape) == 0
    if self_is_scalar:
        self = op.Reshape(self, builder.constant([-1], dtype=ir.DataType.INT64))

    # Index may be a scalar. Reshape it to a rank 1 tensor.
    index = op.Reshape(index, builder.constant([-1], dtype=ir.DataType.INT64))
    index = op.Cast(index, to=ir.DataType.INT64)
    result = op.Gather(self, index, axis=dim)

    if self_is_scalar:
        result = op.Squeeze(result)

    return result


@onnx_impl(aten.logsumexp.default)
def aten_logsumexp(
    op: TranslationContext, self: ir.Value, dim: ir.Value, keepdim: int = False
) -> ir.Value:
    """logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor"""

    if len(self.shape) == 0:
        # A scalar
        result = self
    else:
        result = op.ReduceLogSumExp(self, dim, keepdims=keepdim)
    return result


@onnx_impl(aten.mH.default)
def aten_mH(op: TranslationContext, self: ir.Value) -> ir.Value:
    """mH(Tensor(a) self) -> Tensor(a)"""

    # Taking the conjugate transpose of a real matrix is the same as the transpose
    return op.Einsum(self, equation="...ij->...ji")


@onnx_impl(aten.mH.default, complex=True)
def aten_mH_complex(op: TranslationContext, self: ir.Value) -> ir.Value:
    """mH(Tensor(a) self) -> Tensor(a)"""

    # c is the last dimension being the real and imaginary parts
    transposed = op.Einsum(self, equation="...ijc->...jic")
    return _complex_conjugate(op, transposed)


@onnx_impl(aten.mT.default)
def aten_mT(op: TranslationContext, self: ir.Value) -> ir.Value:
    """mT(Tensor(a) self) -> Tensor(a)"""

    return op.Einsum(self, equation="...ij->...ji")


@onnx_impl(aten.mT.default, complex=True)
def aten_mT_complex(op: TranslationContext, self: ir.Value) -> ir.Value:
    """mT(Tensor(a) self) -> Tensor(a)"""

    # c is the last dimension being the real and imaginary parts
    return op.Einsum(self, equation="...ijc->...jic")


@onnx_impl((aten.masked_fill.Scalar, aten.masked_fill.Tensor))
def aten_masked_fill(
    op: TranslationContext, self: ir.Value, mask: ir.Value, value: ir.Value
) -> ir.Value:
    """masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor"""
    # NOTE: Do not attempt to cast `mask` to BOOL because mask should not take any other types.
    # `mask` coming in as other types is often an error and should fail the model.
    value_cast = op.CastLike(value, self)
    return op.Where(mask, value_cast, self)


@onnx_impl(aten.masked_scatter.default)
def aten_masked_scatter(
    op: TranslationContext, self: ir.Value, mask: ir.Value, source: ir.Value
) -> ir.Value:
    """masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor"""
    builder = op.builder

    # Broadcast self and mask to their common shape so NonZero enumerates every
    # masked element. The previous rank-only check missed same-rank broadcasting
    # (e.g. mask (1, S, 1) vs self (1, S, D)): it left mask un-expanded, so only a
    # subset of masked positions were scattered (pytorch/pytorch#186146).
    self = op.Expand(self, op.Shape(mask))
    mask = op.Expand(mask, op.Shape(self))
    index = op.Transpose(op.NonZero(mask), perm=[1, 0])

    # NOTE: source can have more elements than needed.
    # It could also have arbitrary shape.
    # This is not supported by ONNX::ScatterND, so we need to flatten and slice source tensor.
    source = op.Reshape(source, builder.constant([-1], dtype=ir.DataType.INT64))
    axes = builder.constant([0], dtype=ir.DataType.INT64)
    starts = builder.constant([0], dtype=ir.DataType.INT64)
    ends = op.Gather(
        op.Shape(index),
        builder.constant([0], dtype=ir.DataType.INT64),
        axis=0,
    )
    source = op.Slice(source, starts, ends, axes)

    return op.ScatterND(self, index, source)


@onnx_impl(aten.max.default)
def aten_max(op: TranslationContext, self: ir.Value) -> ir.Value:
    """max(Tensor self) -> Tensor"""

    return op.ReduceMax(self, keepdims=False)


@onnx_impl(aten.max.dim)
def aten_max_dim(
    op: TranslationContext, self: ir.Value, dim: int, keepdim: bool = False
) -> tuple[ir.Value, ir.Value]:
    """max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"""
    builder = op.builder

    if len(self.shape) == 0:
        result = self
        indices = builder.constant(0, dtype=ir.DataType.INT64)
    else:
        dims = op.Reshape(dim, builder.constant([-1], dtype=ir.DataType.INT64))
        result = op.ReduceMax(self, dims, keepdims=keepdim)
        indices = op.ArgMax(self, axis=dim, keepdims=keepdim)
    return result, indices


@onnx_impl(aten.mean.default)
def aten_mean(op: TranslationContext, self: ir.Value, dtype: int = -1) -> ir.Value:
    """mean(Tensor self, *, ScalarType? dtype=None) -> Tensor"""

    if dtype != -1 and dtype is not None:
        # Cast before reducing so that the accumulation happens in the requested
        # dtype, matching PyTorch. Casting the result afterwards would keep the
        # precision loss of the input dtype.
        self = op.Cast(self, to=dtype)

    result = op.ReduceMean(self)
    return op.Squeeze(result)


@onnx_impl(aten.mean.dim)
def aten_mean_dim(
    op: TranslationContext,
    self: ir.Value,
    dim: ir.Value,
    keepdim: bool = False,
    dtype: int = -1,
) -> ir.Value:
    """mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    builder = op.builder
    self_is_scalar = len(self.shape) == 0
    if dtype != -1 and dtype is not None:
        self = op.Cast(self, to=dtype)

    if self_is_scalar:
        result = self
    elif dim is None:
        result = op.ReduceMean(self, keepdims=keepdim)
    else:
        if not isinstance(dim, ir.Value):
            dim = builder.constant(dim, dtype=ir.DataType.INT64)
        dims = op.Reshape(dim, builder.constant([-1], dtype=ir.DataType.INT64))
        result = op.ReduceMean(self, dims, keepdims=keepdim)

    if dtype != -1 and dtype is not None:
        result = op.Cast(result, to=dtype)

    return result


@onnx_impl(aten.mean.default, complex=True)
def aten_mean_complex(
    op: TranslationContext, self: ir.Value, dtype: int = -1
) -> ir.Value:
    """mean(Tensor self, *, ScalarType? dtype=None) -> Tensor"""
    builder = op.builder
    rank = len(self.shape) - 1
    if rank == 0:
        result = op.Identity(self)
    else:
        dim = builder.constant(list(range(rank)), dtype=ir.DataType.INT64)
        result = op.ReduceMean(self, dim, keepdims=False)

    if dtype != -1 and dtype is not None:
        raise NotImplementedError(
            "Complex mean does not support the dtype argument; cast the input "
            "before export or omit dtype."
        )

    return result


@onnx_impl(aten.mean.dim, complex=True)
def aten_mean_dim_complex(
    op: TranslationContext,
    self: ir.Value,
    dim: ir.Value | Sequence[int] | None,
    keepdim: bool = False,
    dtype: int = -1,
) -> ir.Value:
    """mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    builder = op.builder
    rank = len(self.shape) - 1
    if rank == 0:
        result = op.Identity(self)
    else:
        if dim is None:
            dims = builder.constant(list(range(rank)), dtype=ir.DataType.INT64)
        else:
            if not isinstance(dim, ir.Value):
                dim = builder.constant(dim, dtype=ir.DataType.INT64)
            dims = op.Reshape(dim, [-1])
            dims = op.Where(
                op.Less(dims, 0),
                op.Sub(dims, 1),
                dims,
            )
            end = op.Where(
                op.Equal(op.Size(dims), 0),
                rank,
                0,
            )
            all_dims = op.Range(0, end, 1)
            dims = op.Concat(dims, all_dims, axis=0)
        result = op.ReduceMean(self, dims, keepdims=keepdim)

    if dtype != -1 and dtype is not None:
        raise NotImplementedError(
            "Complex mean does not support the dtype argument; cast the input "
            "before export or omit dtype."
        )

    return result


@onnx_impl(aten.min.default)
def aten_min(op: TranslationContext, self: ir.Value) -> ir.Value:
    """min(Tensor self) -> Tensor"""

    return op.ReduceMin(self, keepdims=False)


@onnx_impl(aten.min.dim)
def aten_min_dim(
    op: TranslationContext, self: ir.Value, dim: int, keepdim: bool = False
) -> tuple[ir.Value, ir.Value]:
    """min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"""
    builder = op.builder
    if len(self.shape) == 0:
        result = self
        indices = builder.constant(0, dtype=ir.DataType.INT64)
    else:
        dims = op.Reshape(dim, builder.constant([-1], dtype=ir.DataType.INT64))
        result = op.ReduceMin(self, dims, keepdims=keepdim)
        indices = op.ArgMin(self, axis=dim, keepdims=keepdim)

    return result, indices


@onnx_impl(aten.narrow.default)
def aten_narrow(
    op: TranslationContext,
    self: ir.Value,
    dim: ir.Value,
    start: ir.Value,
    length: ir.Value,
) -> ir.Value:
    """narrow(Tensor(a) self, int dim, SymInt start, SymInt length) -> Tensor(a)"""
    builder = op.builder

    dim = op.Reshape(dim, builder.constant([-1], dtype=ir.DataType.INT64))
    start = op.Reshape(start, builder.constant([-1], dtype=ir.DataType.INT64))
    length = op.Reshape(length, builder.constant([-1], dtype=ir.DataType.INT64))

    end = op.Add(start, length)
    return op.Slice(self, start, end, dim)


@onnx_impl(aten.nonzero.default)
def aten_nonzero(op: TranslationContext, self: ir.Value) -> ir.Value:
    """nonzero(Tensor self) -> Tensor"""
    # NOTE: In torch the return shape is [n, d], while in onnx [d, n],
    # where `d` is rank of input tensor, `n` is number of nonzero elements.
    return op.Transpose(op.NonZero(self), perm=[1, 0])


@onnx_impl(aten.permute.default)
def aten_permute(
    op: TranslationContext, self: ir.Value, dims: Sequence[int]
) -> ir.Value:
    """permute(Tensor(a) self, int[] dims) -> Tensor(a)"""

    if not dims:
        return op.Transpose(self)

    # Handle negative axes
    dims = [axis + len(dims) if axis < 0 else axis for axis in dims]

    return op.Transpose(self, perm=dims)


@onnx_impl(aten.permute.default, complex=True)
def aten_permute_complex(
    op: TranslationContext, self: ir.Value, dims: Sequence[int]
) -> ir.Value:
    """permute(Tensor(a) self, int[] dims) -> Tensor(a)"""

    rank = len(self.shape) - 1

    # Handle negative axes
    dims = [axis + rank if axis < 0 else axis for axis in dims] + [rank]

    return op.Transpose(self, perm=dims)


@onnx_impl(aten.prod.default)
def aten_prod(op: TranslationContext, self: ir.Value, dtype: int = -1) -> ir.Value:
    """prod(Tensor self, *, ScalarType? dtype=None) -> Tensor"""

    if dtype != -1 and dtype is not None:
        self = op.Cast(self, to=dtype)
    elif self.dtype.is_integer() or self.dtype == ir.DataType.BOOL:
        self = op.Cast(self, to=ir.DataType.INT64)
    return op.ReduceProd(self, keepdims=False)


@onnx_impl(aten.prod.dim_int)
def aten_prod_dim_int(
    op: TranslationContext,
    self: ir.Value,
    dim: int,
    keepdim: bool = False,
    dtype: int = -1,
) -> ir.Value:
    """prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    self_is_scalar = len(self.shape) == 0

    if dtype != -1 and dtype is not None:
        self = op.Cast(self, to=dtype)
    elif self.dtype.is_integer() or self.dtype == ir.DataType.BOOL:
        self = op.Cast(self, to=ir.DataType.INT64)
    if self_is_scalar:
        return op.Identity(self)
    return op.ReduceProd(self, [dim], keepdims=keepdim)


@onnx_impl(aten.repeat.default)
def aten_repeat(
    op: TranslationContext, self: ir.Value, repeats: Sequence[int | ir.Value]
) -> ir.Value:
    """repeat(Tensor self, SymInt[] repeats) -> Tensor"""

    if len(repeats) == 0:
        return self
    self_expanded = op.Expand(self, [1] * len(repeats))
    return op.Tile(self_expanded, repeats)


@onnx_impl(aten.reshape.default)
def aten_reshape(
    op: TranslationContext, self: ir.Value, shape: Sequence[int | ir.Value]
) -> ir.Value:
    """reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)"""
    shape = common_ops.merge_dims(op, shape)
    return op.Reshape(self, shape)


@onnx_impl(aten.scatter.src)
def aten_scatter_src(
    op: TranslationContext,
    self: ir.Value,
    dim: int,  # we have to use int here because ScatterElements() will use this attribute
    index: ir.Value,
    src: ir.Value,
) -> ir.Value:
    """scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor"""
    if len(index.shape) == 0:
        index = op.Unsqueeze(index, [0])
    if len(src.shape) == 0:
        src = op.Unsqueeze(src, [0])
    return op.ScatterElements(self, index, src, axis=dim)


@onnx_impl(aten.scatter.value)
def aten_scatter_value(
    op: TranslationContext,
    self: ir.Value,
    dim: int,  # we have to use int here because ScatterElements() will use this attribute
    index: ir.Value,
    value: float,
) -> ir.Value:
    """scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor"""
    # Ensure value is a scalar tensor and expand it to match index shape
    if len(index.shape) == 0:
        index = op.Unsqueeze(index, [0])
    scalar_tensor = ir.tensor([value], dtype=self.dtype)
    src = op.ConstantOfShape(op.Shape(index), value=scalar_tensor)
    return op.ScatterElements(self, index, src, axis=dim)


@onnx_impl(aten.scatter_add.default)
def aten_scatter_add(
    op: TranslationContext,
    self: ir.Value,
    dim: int,  # we have to use int here because ScatterElements() will use this attribute
    index: ir.Value,
    src: ir.Value,
) -> ir.Value:
    """scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor"""

    # if rank(self) == 0 will lead ORT failed, skipped
    return op.ScatterElements(self, index, src, axis=dim, reduction="add")


@onnx_impl(aten.scatter_reduce.two)
def aten_scatter_reduce(
    op: TranslationContext,
    self: ir.Value,
    dim: int,  # we have to use int here because ScatterElements() will use this attribute
    index: ir.Value,
    src: ir.Value,
    reduce: str,
    include_self: bool = True,
) -> ir.Value:
    """scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor"""
    builder = op.builder

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
        self = op.Reshape(self, neg_1)
        index = op.Reshape(index, neg_1)
        src = op.Reshape(src, neg_1)

    src = op.Slice(src, [0] * index_rank, op.Shape(index))
    if reduce == "mean":
        original = self
        zeros = op.ConstantOfShape(
            op.Shape(self),
            value=ir.tensor([0], dtype=dtype),
        )
        ones = op.ConstantOfShape(
            op.Shape(src),
            value=ir.tensor([1], dtype=dtype),
        )
        counts = op.Add(zeros, 1) if include_self else zeros
        if not include_self:
            updates_zero = op.Sub(ones, ones)
            self = op.ScatterElements(self, index, updates_zero, axis=dim)
        total = op.ScatterElements(self, index, src, axis=dim, reduction="add")
        counts = op.ScatterElements(counts, index, ones, axis=dim, reduction="add")
        divisor = op.Max(counts, 1)
        mean = op.Div(total, divisor)
        if dtype.is_integer():
            remainder = op.Mod(total, divisor, fmod=0)
            round_down = op.And(
                op.Less(total, 0),
                op.Not(op.Equal(remainder, 0)),
            )
            mean = op.Sub(mean, op.CastLike(round_down, mean))
        result = op.Where(op.Equal(counts, 0), original, mean)
        return op.Squeeze(result) if self_is_scalar else result

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

        cst = op.ConstantOfShape(op.Shape(src), value=value)
        self = op.ScatterElements(self, index, cst, axis=dim, reduction=reduction_init)

    result = op.ScatterElements(self, index, src, axis=dim, reduction=onnx_reduce)

    if self_is_scalar:
        result = op.Squeeze(result)
    return result


@onnx_impl(aten.select.int)
def aten_select(
    op: TranslationContext, self: ir.Value, dim: int, index: int
) -> ir.Value:
    """select(Tensor self, int dim, int index) -> Tensor"""

    return op.Gather(self, index, axis=dim)


@onnx_impl(aten.select_scatter.default)
def aten_select_scatter(
    op: TranslationContext, self: ir.Value, src: ir.Value, dim: int, index: int
) -> ir.Value:
    """select_scatter(Tensor self, Tensor src, int dim, int index) -> Tensor"""

    # Change src rank to self rank according to dim
    # e.g. if self is [2,3,4], src is [2,4], dim=1, then update is [2,1,4]
    update = op.Unsqueeze(src, [dim])
    # Change index rank to the same as 'update' [2,1,4]
    indices = op.Expand(index, op.Shape(update))
    return op.ScatterElements(self, indices, update, axis=dim, reduction="none")


@onnx_impl(aten.slice.Tensor, complex=True)
def aten_slice_complex(
    op: TranslationContext,
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
    return aten_slice(op, self, dim, start, end, step)


@onnx_impl(aten.slice.Tensor)
def aten_slice(
    op: TranslationContext,
    self: ir.Value,
    dim: int = 0,
    start: ir.Value | None = None,
    end: ir.Value | None = None,
    step: ir.Value | None = None,
) -> ir.Value:
    """slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)"""
    builder = op.builder

    # TODO: using OptionalHasElement() to check start/end value
    if start is not None:
        start = op.Cast(start, to=ir.DataType.INT64)
        start = op.Reshape(start, builder.constant([-1], dtype=ir.DataType.INT64))
    else:
        start = builder.constant([0], dtype=ir.DataType.INT64)

    if end is not None:
        end = op.Cast(end, to=ir.DataType.INT64)
        end = op.Reshape(end, builder.constant([-1], dtype=ir.DataType.INT64))
    else:
        end = builder.constant([_INT64_MAX], dtype=ir.DataType.INT64)

    dim = op.Cast(dim, to=ir.DataType.INT64)
    dim = op.Reshape(dim, builder.constant([-1], dtype=ir.DataType.INT64))

    if step is not None:
        step = op.Cast(step, to=ir.DataType.INT64)
        step = op.Reshape(step, builder.constant([-1], dtype=ir.DataType.INT64))
    else:
        step = builder.constant([1], dtype=ir.DataType.INT64)

    return op.Slice(self, start, end, dim, step)


@onnx_impl(aten.slice_scatter.default)
def aten_slice_scatter(
    op: TranslationContext,
    self: ir.Value,
    src: ir.Value,
    dim: int = 0,
    start: ir.Value | None = None,
    end: ir.Value | None = None,
    step: ir.Value = 1,
) -> ir.Value:
    """slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor"""
    builder = op.builder

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
    self_shape = op.Shape(self)
    dim_shape = op.Gather(self_shape, dim, axis=0)
    index_base = op.Range(0, dim_shape, 1)
    start_index = zero if start is None else op.Unsqueeze(start, zero)
    end_index = (
        builder.constant([_INT64_MAX], dtype=ir.DataType.INT64)
        if end is None
        else op.Unsqueeze(end, zero)
    )
    index_base = op.Slice(
        index_base,
        start_index,
        end_index,
        zero,
        op.Unsqueeze(step, zero),
    )
    index_base = op.Unsqueeze(index_base, [-1])

    # Use trace only to construct the perm attribute in Transpose
    dims = None
    if dim != 0:
        src_rank = len(src.shape)  # type: ignore[attr-defined]

        if src_rank != 0:
            # Python code, change when onnxscript supports this
            dims = list(range(src_rank))
            dims[0], dims[dim] = dims[dim], dims[0]
            # Python code ends

            src = op.Transpose(src, perm=dims)
            self = op.Transpose(self, perm=dims)

    output = op.ScatterND(self, index_base, src)
    if dims is not None:
        output = op.Transpose(output, perm=dims)
    return output


@onnx_impl(aten.sort.default)
def aten_sort(
    op: TranslationContext,
    self: ir.Value,
    dim: int = -1,
    descending: bool = False,
    stable: bool = False,
) -> tuple[ir.Value, ir.Value]:
    """sort(Tensor self, int dim=-1, bool descending=False, bool stable=False) -> (Tensor values, Tensor indices)"""
    builder = op.builder

    self_is_scalar = len(self.shape) == 0
    if self_is_scalar:
        return op.Identity(self), builder.constant(0, dtype=ir.DataType.INT64)
    shape = op.Shape(self)
    dim_size = op.Gather(shape, dim, axis=0)
    dim_size = op.Reshape(dim_size, builder.constant([1], dtype=ir.DataType.INT64))
    values, indices = op.TopK(self, dim_size, axis=dim, largest=descending, sorted=True)
    return values, indices


@onnx_impl(aten.squeeze.default)
def aten_squeeze(op: TranslationContext, self: ir.Value) -> ir.Value:
    """squeeze(Tensor(a) self) -> Tensor(a)"""

    return op.Squeeze(self)


@onnx_impl(aten.squeeze.dim)
def aten_squeeze_dim(op: TranslationContext, self: ir.Value, dim: int) -> ir.Value:
    if len(self.shape) == 0:
        return op.Identity(self)
    return op.Squeeze(self, [dim])


@onnx_impl(aten.squeeze.dim, complex=True)
def aten_squeeze_dim_complex(
    op: TranslationContext, self: ir.Value, dim: int
) -> ir.Value:
    if dim < 0:
        # Account for the complex dimension in ONNX
        dim = dim - 1

    if len(self.shape) == 1:
        # The single dimension is the complex dimension
        return op.Identity(self)
    return aten_squeeze_dim(op, self, dim)


@onnx_impl(aten.sum.default)
def aten_sum(op: TranslationContext, self: ir.Value, dtype: int = -1) -> ir.Value:
    """sum(Tensor self, *, ScalarType? dtype=None) -> Tensor"""
    self_is_scalar = len(self.shape) == 0
    if dtype != -1 and dtype is not None:
        self = op.Cast(self, to=dtype)
    elif self.dtype.is_integer() or self.dtype == ir.DataType.BOOL:
        self = op.Cast(self, to=ir.DataType.INT64)
    if self_is_scalar:
        result = op.Identity(self)
    else:
        result = op.ReduceSum(self, keepdims=False)
    if dtype != -1 and dtype is not None:
        result = op.Cast(result, to=dtype)
    return result


@onnx_impl(aten.sum.default, complex=True)
def aten_sum_complex(
    op: TranslationContext, self: ir.Value, dtype: int = -1
) -> ir.Value:
    """sum(Tensor self, *, ScalarType? dtype=None) -> Tensor"""
    builder = op.builder
    if len(self.shape) == 1:
        result = op.Identity(self)
    else:
        rank = len(self.shape) - 1
        dim = builder.constant(list(range(rank)), dtype=ir.DataType.INT64)
        result = op.ReduceSum(self, dim, keepdims=False)
    if dtype != -1 and dtype is not None:
        raise NotImplementedError(
            "support for the dtype argument is not implemented for complex tensors"
        )
    return result


@onnx_impl(aten.sum.dim_IntList)
def aten_sum_dim_IntList(
    op: TranslationContext,
    self: ir.Value,
    dim: int | None = None,
    keepdim: bool = False,
    dtype: int = -1,
) -> ir.Value:
    """sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    self_is_scalar = len(self.shape) == 0
    if dtype != -1 and dtype is not None:
        self = op.Cast(self, to=dtype)
    elif self.dtype.is_integer() or self.dtype == ir.DataType.BOOL:
        self = op.Cast(self, to=ir.DataType.INT64)
    if self_is_scalar:
        result = op.Identity(self)
    elif dim is None or isinstance(dim, Sequence) and len(dim) == 0:
        result = op.ReduceSum(self, keepdims=keepdim)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = common_ops.constant(op, dim, dtype=ir.DataType.INT64)
        result = op.ReduceSum(self, dim, keepdims=keepdim)

    if dtype != -1 and dtype is not None:
        result = op.Cast(result, to=dtype)

    return result
