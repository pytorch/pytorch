"""Tuple-polymorphic pointwise math for :mod:`quack.epilogue.frontend` functions.

Values are Float32 scalars or two-lane tuple values such as ``F2`` and ``Pair``.
The default transcendental path requests precise Cute math; ``fast=True`` opts
into the corresponding approximate path. Output conversion normally belongs to
the EpiMod D/TileStore boundary; ``to_dtype`` is for FX-visible casts.
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32

from torch._vendor.quack import activation
from torch._vendor.quack.cute_dsl_utils import torch2cute_dtype_map
from torch._vendor.quack.rounding import (
    cvt_f32x2_ue8m0x2_rp_satfinite,
    cvt_f32x2_ue8m0x2_rz,
)


def _pair_like(value, lo, hi):
    """Rebuild a two-lane value without depending on a particular lane class."""
    return (lo, hi) if const_expr(type(value) is tuple) else type(value)(lo, hi)


def _lane(value, index):
    """Broadcast scalars across tuple lanes."""
    return value[index] if const_expr(isinstance(value, tuple)) else value


def abs(x, *, fast=False):
    """Elementwise absolute value."""
    if const_expr(isinstance(x, tuple)):
        return _pair_like(x, abs(x[0], fast=fast), abs(x[1], fast=fast))
    return cute.math.abs(x, fastmath=fast)


def reciprocal(x, *, fast=False):
    """Elementwise reciprocal; ``fast`` selects approximate reciprocal math."""
    if const_expr(isinstance(x, tuple)):
        return _pair_like(x, reciprocal(x[0], fast=fast), reciprocal(x[1], fast=fast))
    return cute.math.rcp(x, approx=fast)


def divide(a, b, *, fast=False):
    """Elementwise division with scalar broadcasting and explicit precision."""
    if const_expr(isinstance(a, tuple) or isinstance(b, tuple)):
        template = a if isinstance(a, tuple) else b
        return _pair_like(
            template,
            divide(_lane(a, 0), _lane(b, 0), fast=fast),
            divide(_lane(a, 1), _lane(b, 1), fast=fast),
        )
    return a * reciprocal(b, fast=True) if const_expr(fast) else a / b


def exp(x, *, fast=False):
    """Elementwise natural exponential, using the packed F2 intrinsic when fast."""
    if const_expr(isinstance(x, tuple)):
        if const_expr(fast):
            lo, hi = cute.arch.exp_packed_f32x2(x)
            return _pair_like(x, lo, hi)
        return _pair_like(x, exp(x[0]), exp(x[1]))
    return cute.math.exp(x, fastmath=fast)


def sqrt(x, *, fast=False):
    """Elementwise square root."""
    if const_expr(isinstance(x, tuple)):
        return _pair_like(x, sqrt(x[0], fast=fast), sqrt(x[1], fast=fast))
    return cute.math.sqrt(x, approx=fast)


def rsqrt(x, *, fast=False):
    """Elementwise reciprocal square root."""
    if const_expr(isinstance(x, tuple)):
        return _pair_like(x, rsqrt(x[0], fast=fast), rsqrt(x[1], fast=fast))
    return cute.math.rsqrt(x, approx=fast)


def log(x, *, fast=False):
    """Elementwise natural logarithm."""
    if const_expr(isinstance(x, tuple)):
        return _pair_like(x, log(x[0], fast=fast), log(x[1], fast=fast))
    return cute.math.log(x, fastmath=fast)


def log1p(x, *, fast=False):
    """Elementwise ``log(1 + x)``."""
    if const_expr(isinstance(x, tuple)):
        return _pair_like(x, log1p(x[0], fast=fast), log1p(x[1], fast=fast))
    return cute.math.log1p(x, fastmath=fast)


def erf(x, *, fast=False):
    """Elementwise error function."""
    if const_expr(isinstance(x, tuple)):
        return _pair_like(x, erf(x[0], fast=fast), erf(x[1], fast=fast))
    return cute.math.erf(x, fastmath=fast)


def tanh(x, *, fast=False):
    """Elementwise hyperbolic tangent."""
    if const_expr(isinstance(x, tuple)):
        return _pair_like(x, tanh(x[0], fast=fast), tanh(x[1], fast=fast))
    return activation.tanh(x) if const_expr(fast) else cute.math.tanh(x)


def sigmoid(x, *, fast=False):
    """Elementwise sigmoid; ``fast`` reuses QuACK's approximate activation."""
    if const_expr(isinstance(x, tuple)):
        return _pair_like(x, sigmoid(x[0], fast=fast), sigmoid(x[1], fast=fast))
    return activation.sigmoid(x) if const_expr(fast) else reciprocal(1.0 + exp(-x))


def relu(x, *, fast=False):
    """Elementwise ReLU; ``fast`` selects QuACK's store-optimized activation."""
    if const_expr(isinstance(x, tuple)):
        return _pair_like(x, relu(x[0], fast=fast), relu(x[1], fast=fast))
    return activation.relu(x) if const_expr(fast) else maximum(x, 0.0)


def minimum(a, b):
    """Elementwise IEEE ``minimum`` with NaN propagation."""
    if const_expr(isinstance(a, tuple) or isinstance(b, tuple)):
        template = a if isinstance(a, tuple) else b
        return _pair_like(
            template,
            cute.arch.fmin(_lane(a, 0), _lane(b, 0), nan=True),
            cute.arch.fmin(_lane(a, 1), _lane(b, 1), nan=True),
        )
    return cute.arch.fmin(a, b, nan=True)


def maximum(a, b):
    """Elementwise IEEE ``maximum`` with NaN propagation."""
    if const_expr(isinstance(a, tuple) or isinstance(b, tuple)):
        template = a if isinstance(a, tuple) else b
        return _pair_like(
            template,
            cute.arch.fmax(_lane(a, 0), _lane(b, 0), nan=True),
            cute.arch.fmax(_lane(a, 1), _lane(b, 1), nan=True),
        )
    return cute.arch.fmax(a, b, nan=True)


def min(a, b):
    """Alias for :func:`minimum` matching the pointwise vocabulary."""
    return minimum(a, b)


def max(a, b):
    """Alias for :func:`maximum` matching the pointwise vocabulary."""
    return maximum(a, b)


def clamp(x, min=None, max=None):
    """Clamp ``x`` to its optional inclusive lower and upper bounds."""
    if const_expr(min is not None):
        x = maximum(x, min)
    if const_expr(max is not None):
        x = minimum(x, max)
    return x


def clamp_min(x, min):
    """Clamp ``x`` below by ``min``."""
    return maximum(x, min)


def clamp_max(x, max):
    """Clamp ``x`` above by ``max``."""
    return minimum(x, max)


def _mx_e8m0_input(x, max_value, rounding):
    """Prepare one scalar for the selected packed E8M0 conversion."""
    if const_expr(rounding == "rceil"):
        return x / max_value
    max_power = const_expr(math.floor(math.log2(max_value)))
    scaled = x * (2.0**-max_power)
    if const_expr(max_power > 0):
        replacement = Float32(2.0 ** (128 - max_power))
        scaled = Float32(cutlass.select_(scaled == Float32(float("inf")), replacement, scaled))
    return scaled


def mx_e8m0_scale(x, max_value=448.0, rounding="rceil"):
    """Encode an MX scale with exact packed FLOOR or saturating RCEIL conversion."""
    if const_expr(rounding not in ("floor", "rceil")):
        raise ValueError(f"unsupported MX scale rounding {rounding!r}")
    is_pair = const_expr(isinstance(x, tuple))
    lo = _mx_e8m0_input(x[0] if is_pair else x, max_value, rounding)
    hi = _mx_e8m0_input(x[1] if is_pair else Float32(0.0), max_value, rounding)
    convert = (
        cvt_f32x2_ue8m0x2_rz if const_expr(rounding == "floor") else cvt_f32x2_ue8m0x2_rp_satfinite
    )
    result = convert(lo, hi)
    return _pair_like(x, *result) if is_pair else result[0]


def nvfp4_e4m3_scale(x, max_value=6.0, rounding="nearest"):
    """Encode an NVFP4 E4M3 scale and return its decoded Float32 value."""
    if const_expr(rounding != "nearest"):
        raise ValueError(f"unsupported NVFP4 scale rounding {rounding!r}")
    if const_expr(isinstance(x, tuple)):
        return _pair_like(
            x,
            nvfp4_e4m3_scale(x[0], max_value, rounding),
            nvfp4_e4m3_scale(x[1], max_value, rounding),
        )
    scaled = clamp(
        x / max_value,
        min=0.015625,
        max=448.0,
    )
    return to_dtype(scaled, cutlass.Float8E4M3FN)


def _nvfp4_e2m1_code(x):
    """Return one finite E2M1 nibble with round-to-nearest-even ties."""
    magnitude = abs(x)
    code = where(magnitude > 0.25, Float32(1.0), Float32(0.0))
    code = where(magnitude >= 0.75, Float32(2.0), code)
    code = where(magnitude > 1.25, Float32(3.0), code)
    code = where(magnitude >= 1.75, Float32(4.0), code)
    code = where(magnitude > 2.5, Float32(5.0), code)
    code = where(magnitude >= 3.5, Float32(6.0), code)
    code = where(magnitude > 5.0, Float32(7.0), code)
    return code + where(x < 0.0, Float32(8.0), Float32(0.0))


def nvfp4_pack(x):
    """Pack adjacent Float32 lanes into native E2M1 Uint8 storage."""
    if const_expr(isinstance(x, tuple)):
        return _nvfp4_e2m1_code(x[0]) + Float32(16.0) * _nvfp4_e2m1_code(x[1])
    packed = x.to(cutlass.Float4E2M1FN).bitcast(cutlass.Uint8)
    return packed.reshape((cute.size(packed.shape), 1, 1))


def eq(a, b):
    """Elementwise equality comparison."""
    if const_expr(isinstance(a, tuple) or isinstance(b, tuple)):
        template = a if isinstance(a, tuple) else b
        return _pair_like(template, _lane(a, 0) == _lane(b, 0), _lane(a, 1) == _lane(b, 1))
    return a == b


def ne(a, b):
    """Elementwise inequality comparison."""
    if const_expr(isinstance(a, tuple) or isinstance(b, tuple)):
        template = a if isinstance(a, tuple) else b
        return _pair_like(template, _lane(a, 0) != _lane(b, 0), _lane(a, 1) != _lane(b, 1))
    return a != b


def lt(a, b):
    """Elementwise less-than comparison."""
    if const_expr(isinstance(a, tuple) or isinstance(b, tuple)):
        template = a if isinstance(a, tuple) else b
        return _pair_like(template, _lane(a, 0) < _lane(b, 0), _lane(a, 1) < _lane(b, 1))
    return a < b


def le(a, b):
    """Elementwise less-than-or-equal comparison."""
    if const_expr(isinstance(a, tuple) or isinstance(b, tuple)):
        template = a if isinstance(a, tuple) else b
        return _pair_like(template, _lane(a, 0) <= _lane(b, 0), _lane(a, 1) <= _lane(b, 1))
    return a <= b


def gt(a, b):
    """Elementwise greater-than comparison."""
    if const_expr(isinstance(a, tuple) or isinstance(b, tuple)):
        template = a if isinstance(a, tuple) else b
        return _pair_like(template, _lane(a, 0) > _lane(b, 0), _lane(a, 1) > _lane(b, 1))
    return a > b


def ge(a, b):
    """Elementwise greater-than-or-equal comparison."""
    if const_expr(isinstance(a, tuple) or isinstance(b, tuple)):
        template = a if isinstance(a, tuple) else b
        return _pair_like(template, _lane(a, 0) >= _lane(b, 0), _lane(a, 1) >= _lane(b, 1))
    return a >= b


def logical_not(x):
    """Elementwise logical-not, following PyTorch's ``x == 0`` convention."""
    if const_expr(isinstance(x, tuple)):
        return _pair_like(x, x[0] == 0, x[1] == 0)
    return x == 0


@cute.jit
def where(condition, a, b):
    """Select ``a`` where ``condition`` is true and ``b`` otherwise."""
    if const_expr(isinstance(condition, tuple) or isinstance(a, tuple) or isinstance(b, tuple)):
        template = condition if isinstance(condition, tuple) else a if isinstance(a, tuple) else b
        return _pair_like(
            template,
            _lane(a, 0) if _lane(condition, 0) else _lane(b, 0),
            _lane(a, 1) if _lane(condition, 1) else _lane(b, 1),
        )
    return a if condition else b


def select(condition, a, b):
    """Alias for :func:`where`."""
    return where(condition, a, b)


def to_dtype(x, dtype):
    """Round through a torch or CuTe ``dtype`` while retaining Float32 SSA."""
    if const_expr(isinstance(x, tuple)):
        return _pair_like(x, to_dtype(x[0], dtype), to_dtype(x[1], dtype))
    target_dtype = torch2cute_dtype_map.get(dtype, dtype)
    converted = x.to(target_dtype)
    return converted if const_expr(target_dtype is Float32) else converted.to(Float32)


def convert_element_type(x, dtype):
    """Alias for :func:`to_dtype` matching the FX primitive name."""
    return to_dtype(x, dtype)


def store_cast(x, dtype):
    """Explicit pre-store cast; normal EpiMod stores convert from the destination dtype."""
    return to_dtype(x, dtype)
