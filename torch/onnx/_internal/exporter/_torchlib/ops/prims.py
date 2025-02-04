"""torch.ops.aten operators under the `prims` module."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# ruff: noqa: TCH001,TCH002
# flake8: noqa

from __future__ import annotations

from typing import Optional, Sequence

from onnxscript import INT64
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import BOOL, TensorType

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import RealType, TTensor
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import common as common_ops


prims = torch.ops.prims


@onnx_impl(prims.abs, trace_only=True)
def prims_abs(self: TTensor) -> TTensor:
    """abs(Tensor self) -> Tensor"""

    return op.Abs(self)


@onnx_impl(prims.acos, trace_only=True)
def prims_acos(self: TensorType) -> TensorType:
    """acos(Tensor self) -> Tensor"""

    return op.Acos(self)


@onnx_impl(prims.acosh, trace_only=True)
def prims_acosh(self: TensorType) -> TensorType:
    """acosh(Tensor self) -> Tensor"""

    return op.Acosh(self)


@onnx_impl(prims.add, trace_only=True)
def prims_add(self: TTensor, other: TTensor) -> TTensor:
    """add(Tensor self, Tensor other) -> Tensor"""

    return op.Add(self, other)


def prims_amax(
    inp: TensorType, dims: Optional[Sequence[int]], output_dtype: Optional[int] = None
) -> TensorType:
    """amax(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor"""

    raise NotImplementedError


def prims_amin(
    inp: TensorType, dims: Optional[Sequence[int]], output_dtype: Optional[int] = None
) -> TensorType:
    """amin(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor"""

    raise NotImplementedError


def prims_as_strided(
    a: TensorType, size: INT64, stride: INT64, storage_offset: INT64
) -> TensorType:
    """as_strided(Tensor a, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor"""

    raise NotImplementedError


def prims_as_strided_scatter(
    self: TensorType, src: TensorType, size: INT64, stride: INT64, storage_offset: INT64
) -> TensorType:
    """as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.asin, trace_only=True)
def prims_asin(self: TTensor) -> TTensor:
    """asin(Tensor self) -> Tensor"""

    return op.Asin(self)


@onnx_impl(prims.asinh, trace_only=True)
def prims_asinh(self: TTensor) -> TTensor:
    """asinh(Tensor self) -> Tensor"""

    return op.Asinh(self)


@onnx_impl(prims.atan, trace_only=True)
def prims_atan(self: TTensor) -> TTensor:
    """atan(Tensor self) -> Tensor"""

    return op.Atan(self)


def prims_atan2(self: TensorType, other: TensorType) -> TensorType:
    """atan2(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.atanh, trace_only=True)
def prims_atanh(self: TTensor) -> TTensor:
    """atanh(Tensor self) -> Tensor"""

    return op.Atanh(self)


def prims_bessel_i0(self: TensorType) -> TensorType:
    """bessel_i0(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_bessel_i0e(self: TensorType) -> TensorType:
    """bessel_i0e(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_bessel_i1(self: TensorType) -> TensorType:
    """bessel_i1(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_bessel_i1e(self: TensorType) -> TensorType:
    """bessel_i1e(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_bessel_j0(self: TensorType) -> TensorType:
    """bessel_j0(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_bessel_j1(self: TensorType) -> TensorType:
    """bessel_j1(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_bitwise_and(self: TensorType, other: TensorType) -> TensorType:
    """bitwise_and(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_bitwise_not(self: TensorType) -> TensorType:
    """bitwise_not(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_bitwise_or(self: TensorType, other: TensorType) -> TensorType:
    """bitwise_or(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_bitwise_xor(self: TensorType, other: TensorType) -> TensorType:
    """bitwise_xor(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_broadcast_in_dim(
    a: TensorType, shape: INT64, broadcast_dimensions: Sequence[int]
) -> TensorType:
    """broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)"""

    raise NotImplementedError


def prims_cat(tensors: Sequence[TensorType], dim: int) -> TensorType:
    """cat(Tensor[] tensors, int dim) -> Tensor"""

    raise NotImplementedError


def prims_cbrt(self: TensorType) -> TensorType:
    """cbrt(Tensor self) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.ceil, trace_only=True)
def prims_ceil(self: TTensor) -> TTensor:
    """ceil(Tensor self) -> Tensor"""

    return op.Ceil(self)


def prims_clone(self: TensorType, memory_format: Optional[str] = None) -> TensorType:
    """clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor"""

    raise NotImplementedError


def prims_collapse_view(a: TensorType, start: int, end: int) -> TensorType:
    """collapse_view(Tensor(a) a, int start, int end) -> Tensor(a)"""

    raise NotImplementedError


def prims_conj(a: TensorType) -> TensorType:
    """conj(Tensor(a) a) -> Tensor(a)"""

    raise NotImplementedError


def prims_conj_physical(self: TensorType) -> TensorType:
    """conj_physical(Tensor self) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.convert_element_type, trace_only=True)
def prims_convert_element_type(a: RealType, dtype: int) -> RealType:
    """convert_element_type(Tensor a, ScalarType dtype) -> Tensor"""

    # Set trace_only=True because different if branches return different dtypes
    # which is not supported in an ONNX function
    return common_ops.cast_to(a, dtype)


def prims_copy_strided(a: TensorType, stride: INT64) -> TensorType:
    """copy_strided(Tensor a, SymInt[] stride) -> Tensor"""

    raise NotImplementedError


def prims_copy_to(a: TensorType, b: TensorType) -> TensorType:
    """copy_to(Tensor a, Tensor b) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.cos, trace_only=True)
def prims_cos(self: TTensor) -> TTensor:
    """cos(Tensor self) -> Tensor"""

    return op.Cos(self)


@onnx_impl(prims.cosh, trace_only=True)
def prims_cosh(self: TTensor) -> TTensor:
    """cosh(Tensor self) -> Tensor"""

    return op.Cosh(self)


@onnx_impl(prims.device_put)
def prims_device_put(
    a: TTensor,
    device: str = "unspecified",  # pylint: disable=unused-argument
) -> TTensor:
    """device_put(Tensor a, Device device) -> Tensor"""

    # ONNX does not have the notion of a "device". So we just return the input
    return op.Identity(a)


def prims_digamma(self: TensorType) -> TensorType:
    """digamma(Tensor self) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.div, trace_only=True)
def prims_div(self: TTensor, other: TTensor) -> TTensor:
    """div(Tensor self, Tensor other) -> Tensor"""

    return op.Div(self, other)


def prims_empty(
    shape: INT64, dtype: int, device: str, requires_grad: bool
) -> TensorType:
    """empty(SymInt[] shape, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""

    raise NotImplementedError


def prims_empty_strided(
    shape: INT64, strides: INT64, dtype: int, device: str, requires_grad: bool
) -> TensorType:
    """empty_strided(SymInt[] shape, SymInt[] strides, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.eq, trace_only=True)
def prims_eq(self: TTensor, other: TTensor) -> TTensor:
    """eq(Tensor self, Tensor other) -> Tensor"""

    return op.Equal(self, other)


@onnx_impl(prims.erf, trace_only=True)
def prims_erf(self: TTensor) -> TTensor:
    """erf(Tensor self) -> Tensor"""

    return op.Erf(self)


def prims_erf_inv(self: TensorType) -> TensorType:
    """erf_inv(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_erfc(self: TensorType) -> TensorType:
    """erfc(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_erfcx(self: TensorType) -> TensorType:
    """erfcx(Tensor self) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.exp, trace_only=True)
def prims_exp(self: TTensor) -> TTensor:
    """exp(Tensor self) -> Tensor"""

    return op.Exp(self)


def prims_exp2(self: TensorType) -> TensorType:
    """exp2(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_expm1(self: TensorType) -> TensorType:
    """expm1(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_fft_c2c(self: TensorType, dim: Sequence[int], forward: bool) -> TensorType:
    """fft_c2c(Tensor self, *, int[] dim, bool forward) -> Tensor"""

    raise NotImplementedError


def prims_fft_c2r(
    self: TensorType, dim: Sequence[int], last_dim_size: INT64
) -> TensorType:
    """fft_c2r(Tensor self, *, int[] dim, SymInt last_dim_size) -> Tensor"""

    raise NotImplementedError


def prims_fft_r2c(self: TensorType, dim: Sequence[int], onesided: bool) -> TensorType:
    """fft_r2c(Tensor self, *, int[] dim, bool onesided) -> Tensor"""

    raise NotImplementedError


def prims_fill(self: TensorType, value: float) -> TensorType:
    """fill(Tensor self, Scalar value) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.floor, trace_only=True)
def prims_floor(self: TTensor) -> TTensor:
    """floor(Tensor self) -> Tensor"""

    return op.Floor(self)


def prims_fmax(self: TensorType, other: TensorType) -> TensorType:
    """fmax(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_fmin(self: TensorType, other: TensorType) -> TensorType:
    """fmin(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_fmod(self: TensorType, other: TensorType) -> TensorType:
    """fmod(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_full(
    shape: INT64, fill_value: float, dtype: int, device: str, requires_grad: bool
) -> TensorType:
    """full(SymInt[] shape, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""

    raise NotImplementedError


def prims_full_like(
    a: TensorType, fill_value: float, dtype: int, device: str, requires_grad: bool
) -> TensorType:
    """full_like(Tensor a, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""

    raise NotImplementedError


def prims_gcd(self: TensorType, other: TensorType) -> TensorType:
    """gcd(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.ge, trace_only=True)
def prims_ge(self: TTensor, other: TTensor) -> TTensor:
    """ge(Tensor self, Tensor other) -> Tensor"""

    return op.GreaterOrEqual(self, other)


@onnx_impl(prims.gt, trace_only=True)
def prims_gt(self: TTensor, other: TTensor) -> TTensor:
    """gt(Tensor self, Tensor other) -> Tensor"""

    return op.Greater(self, other)


def prims_hypot(self: TensorType, other: TensorType) -> TensorType:
    """hypot(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_igamma(self: TensorType, other: TensorType) -> TensorType:
    """igamma(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_igammac(self: TensorType, other: TensorType) -> TensorType:
    """igammac(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_imag(self: TensorType) -> TensorType:
    """imag(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_iota(
    length: INT64,
    start: INT64,
    step: INT64,
    dtype: int,
    device: str,
    requires_grad: bool,
) -> TensorType:
    """iota(SymInt length, *, SymInt start, SymInt step, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""

    raise NotImplementedError


def prims_isfinite(self: TensorType) -> TensorType:
    """isfinite(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_item(a: TensorType) -> float:
    """item(Tensor a) -> Scalar"""

    raise NotImplementedError


@onnx_impl(prims.le, trace_only=True)
def prims_le(self: TensorType, other: TensorType) -> TensorType:
    """le(Tensor self, Tensor other) -> Tensor"""

    return op.LessOrEqual(self, other)


def prims_lgamma(self: TensorType) -> TensorType:
    """lgamma(Tensor self) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.log, trace_only=True)
def prims_log(self: TensorType) -> TensorType:
    """log(Tensor self) -> Tensor"""

    return op.Log(self)


def prims_log10(self: TensorType) -> TensorType:
    """log10(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_log1p(self: TensorType) -> TensorType:
    """log1p(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_log2(self: TensorType) -> TensorType:
    """log2(Tensor self) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.lt, trace_only=True)
def prims_lt(self: TensorType, other: TensorType) -> TensorType:
    """lt(Tensor self, Tensor other) -> Tensor"""

    return op.Less(self, other)


def prims_maximum(self: TensorType, other: TensorType) -> TensorType:
    """maximum(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_maximum_value(dtype: int) -> float:
    """maximum_value(ScalarType dtype) -> Scalar"""

    raise NotImplementedError


def prims_minimum(self: TensorType, other: TensorType) -> TensorType:
    """minimum(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_minium_value(dtype: int) -> float:
    """minium_value(ScalarType dtype) -> Scalar"""

    raise NotImplementedError


@onnx_impl(prims.mul, trace_only=True)
def prims_mul(self: TTensor, other: TTensor) -> TTensor:
    """mul(Tensor self, Tensor other) -> Tensor"""

    return op.Mul(self, other)


def prims_ndtri(self: TensorType) -> TensorType:
    """ndtri(Tensor self) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.ne, trace_only=True)
def prims_ne(self: TTensor, other: TTensor) -> TTensor:
    """ne(Tensor self, Tensor other) -> Tensor"""

    return op.Not(op.Equal(self, other))


@onnx_impl(prims.neg, trace_only=True)
def prims_neg(self: TTensor) -> TTensor:
    """neg(Tensor self) -> Tensor"""

    return op.Neg(self)


def prims_nextafter(self: TensorType, other: TensorType) -> TensorType:
    """nextafter(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_normal(
    shape: INT64, mean: float, std: float, dtype: int, device: str, requires_grad: bool
) -> TensorType:
    """normal(SymInt[] shape, *, Scalar mean, Scalar std, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.pow, trace_only=True)
def prims_pow(self: TTensor, other: TTensor) -> TTensor:
    """pow(Tensor self, Tensor other) -> Tensor"""

    return op.Pow(self, other)


def prims_prod(
    inp: TensorType, dims: Optional[Sequence[int]], output_dtype: Optional[int] = None
) -> TensorType:
    """prod(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor"""

    raise NotImplementedError


def prims_real(self: TensorType) -> TensorType:
    """real(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_reciprocal(self: TensorType) -> TensorType:
    """reciprocal(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_remainder(self: TensorType, other: TensorType) -> TensorType:
    """remainder(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.reshape, trace_only=True)
def prims_reshape(a: TTensor, shape: INT64) -> TTensor:
    """reshape(Tensor a, SymInt[] shape) -> Tensor"""

    return op.Reshape(a, shape)


@onnx_impl(prims.resize, trace_only=True)
def prims_resize(a: TensorType, shape: INT64) -> TensorType:
    """resize(Tensor a, SymInt[] shape) -> Tensor"""

    return op.Expand(a, shape)


def prims_rev(a: TensorType, dims: Sequence[int]) -> TensorType:
    """rev(Tensor a, int[] dims) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.round, trace_only=True)
def prims_round(self: TensorType) -> TensorType:
    """round(Tensor self) -> Tensor"""

    return op.Round(self)


def prims_rsqrt(self: TensorType) -> TensorType:
    """rsqrt(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_scalar_tensor(
    s: float, dtype: Optional[int] = None, device: Optional[str] = None
) -> TensorType:
    """scalar_tensor(Scalar s, *, ScalarType? dtype=None, Device? device=None) -> Tensor"""

    raise NotImplementedError


def prims_shift_left(self: TensorType, other: TensorType) -> TensorType:
    """shift_left(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_shift_right_arithmetic(self: TensorType, other: TensorType) -> TensorType:
    """shift_right_arithmetic(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def prims_sign(self: TensorType) -> TensorType:
    """sign(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_signbit(self: TensorType) -> TensorType:
    """signbit(Tensor self) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.sin, trace_only=True)
def prims_sin(self: TTensor) -> TTensor:
    """sin(Tensor self) -> Tensor"""

    return op.Sin(self)


@onnx_impl(prims.sinh, trace_only=True)
def prims_sinh(self: TTensor) -> TTensor:
    """sinh(Tensor self) -> Tensor"""

    return op.Sinh(self)


def prims_slice(
    a: TensorType,
    start_indices: INT64,
    limit_indices: INT64,
    strides: Optional[INT64] = None,
) -> TensorType:
    """slice(Tensor(a) a, SymInt[] start_indices, SymInt[] limit_indices, SymInt[]? strides=None) -> Tensor(a)"""

    raise NotImplementedError


def prims_slice_in_dim(
    a: TensorType,
    start_index: INT64,
    limit_index: INT64,
    stride: int = 1,
    axis: int = 0,
) -> TensorType:
    """slice_in_dim(Tensor(a) a, SymInt start_index, SymInt limit_index, int stride=1, int axis=0) -> Tensor(a)"""

    raise NotImplementedError


def prims_spherical_bessel_j0(self: TensorType) -> TensorType:
    """spherical_bessel_j0(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_split_dim(a: TensorType, dim: int, outer_length: INT64) -> TensorType:
    """split_dim(Tensor(a) a, int dim, SymInt outer_length) -> Tensor(a)"""

    raise NotImplementedError


@onnx_impl(prims.sqrt, trace_only=True)
def prims_sqrt(self: TTensor) -> TTensor:
    """sqrt(Tensor self) -> Tensor"""

    return op.Sqrt(self)


@onnx_impl(prims.squeeze, trace_only=True)
def prims_squeeze(a: TTensor, dimensions: Sequence[int]) -> TTensor:
    """squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a)"""

    return op.Squeeze(a, axes=dimensions)


@onnx_impl(prims.sub, trace_only=True)
def prims_sub(self: TTensor, other: TTensor) -> TTensor:
    """sub(Tensor self, Tensor other) -> Tensor"""

    return op.Sub(self, other)


def prims_sum(
    inp: TensorType, dims: Optional[Sequence[int]], output_dtype: Optional[int] = None
) -> TensorType:
    """sum(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor"""

    raise NotImplementedError


def prims_svd(
    A: TensorType, full_matrices: bool
) -> tuple[TensorType, TensorType, TensorType]:
    """svd(Tensor A, *, bool full_matrices) -> (Tensor U, Tensor S, Tensor Vh)"""

    raise NotImplementedError


@onnx_impl(prims.tan, trace_only=True)
def prims_tan(self: TTensor) -> TTensor:
    """tan(Tensor self) -> Tensor"""

    return op.Tan(self)


@onnx_impl(prims.tanh, trace_only=True)
def prims_tanh(self: TTensor) -> TTensor:
    """tanh(Tensor self) -> Tensor"""

    return op.Tanh(self)


@onnx_impl(prims.transpose, trace_only=True)
def prims_transpose(a: TensorType, permutation: Sequence[int]) -> TensorType:
    """transpose(Tensor(a) a, int[] permutation) -> Tensor(a)"""

    return op.Transpose(a, perm=permutation)


def prims_trunc(self: TensorType) -> TensorType:
    """trunc(Tensor self) -> Tensor"""

    raise NotImplementedError


def prims_uniform(
    shape: INT64, low: float, high: float, dtype: int, device: str
) -> TensorType:
    """uniform(SymInt[] shape, *, Scalar low, Scalar high, ScalarType dtype, Device device) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.var, trace_only=True)
def prims_var(
    inp: TensorType,
    dims: Optional[Sequence[int]],
    correction: int,
    output_dtype: Optional[int] = None,
) -> TensorType:
    """var(Tensor inp, int[]? dims, *, int correction, ScalarType? output_dtype=None) -> Tensor"""

    if not dims:
        # dims can be empty in practice. We just use a None so it is not added in the ONNX graph
        dims = None
    sub_mean = op.Sub(inp, op.ReduceMean(inp, dims, keepdims=True))
    sqr_mean = op.Mul(sub_mean, sub_mean)
    var = op.ReduceMean(sqr_mean, dims, keepdims=False)
    # Adjust var according to correction value
    if correction != 0:
        inp_shape = op.Shape(inp)
        dim_size = op.Gather(inp_shape, dims, axis=0)
        numel_float = op.CastLike(op.ReduceProd(dim_size, keepdims=False), inp)
        mul = op.Mul(var, numel_float)
        # Subtract the correction value
        sub = op.Sub(numel_float, op.CastLike(correction, inp))
        var = op.Div(mul, sub)

    if output_dtype is not None and output_dtype != -1:
        var = op.Cast(var, to=output_dtype)

    return var


def prims_view_of(a: TensorType) -> TensorType:
    """view_of(Tensor(a) a) -> Tensor"""

    raise NotImplementedError


@onnx_impl(prims.where, trace_only=True)
def prims_where(pred: BOOL, a: TTensor, b: TTensor) -> TTensor:
    """where(Tensor pred, Tensor a, Tensor b) -> Tensor"""

    return op.Where(pred, a, b)


def prims_zeta(self: TensorType, other: TensorType) -> TensorType:
    """zeta(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError
