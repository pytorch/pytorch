# CuTeDSL NaN-propagating max / min in one instruction where hardware supports it.
#
# torch's max/min family propagates NaN -- amax, amin, aminmax, maximum, minimum and the
# ord=+-inf norms all return NaN if any input is NaN -- and the obvious way to write that
# costs four instructions:
#
#     b if ((b > a) or (b != b)) else a          # two setp, an or.pred, a selp
#
# and the fully-branched form some traits used costs five. PTX has done the whole thing in
# one instruction since sm_80 (`max.NaN.f32`), and NVVM's MaxOp/MinOp expose the flag as an
# attribute -- but cute.arch.fmax/fmin do not pass it, so this reaches the dialect op the
# way those wrappers do. Verified against torch.maximum / torch.minimum over every
# combination of NaN, +-inf and +-0.
#
# The candidate difference is the SIGN OF ZERO on a tie: PTX max returns +0 for max(+0, -0)
# where the comparison form returns whichever operand it was handed first. Measured BIT FOR
# BIT against ATen on CUDA over all four +-0 combinations of maximum/minimum/fmax/fmin
# ATen agrees with the instruction, because its kernels
# are the same hardware semantics. So there is nothing to trade off. Comparing against
# torch.maximum with these overrides ACTIVE says otherwise -- that is the mismeasurement
# that made it look like a difference at all.
#
# The hardware path is FLOAT32. fp16/bf16 reach it through their fp32 compute
# type; other trait types retain the generic fallback.
#
# `dtype` is optional: the reduction traits pass their accumulator dtype, while the pointwise
# op functions receive only values (their signature is fn(*vals, *consts)), so those let it
# be inferred from the operand. Anything the inference does not recognise falls through to
# the comparison form, which is correct for every dtype.

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32
from cutlass._mlir.dialects import nvvm
from cutlass._mlir.extras import types as T


def _dt(a, dtype):
    # Deliberately NOT @cute.jit: an undecorated callee runs as ordinary Python at trace
    # time, which is what resolving a dtype is. (`const_expr` is for conditions -- handing it
    # a dtype class raises "received a dynamic expression".)
    return type(a) if dtype is None else dtype


@cute.jit
def fmax_nan(a, b, dtype: cutlass.Constexpr = None):
    """max(a, b), NaN-PROPAGATING: torch.maximum / amax / max.dim semantics."""
    dt = _dt(a, dtype)
    if const_expr(dt is Float32):
        return Float32(nvvm.fmax(T.f32(), a.ir_value(), b.ir_value(), nan=True))
    if const_expr(getattr(dt, "is_integer", False)):
        return max(a, b)
    return b if ((b > a) or (b != b)) else a


@cute.jit
def fmin_nan(a, b, dtype: cutlass.Constexpr = None):
    """min(a, b), NaN-PROPAGATING: torch.minimum / amin / min.dim semantics."""
    dt = _dt(a, dtype)
    if const_expr(dt is Float32):
        return Float32(nvvm.fmin(T.f32(), a.ir_value(), b.ir_value(), nan=True))
    if const_expr(getattr(dt, "is_integer", False)):
        return min(a, b)
    return b if ((b < a) or (b != b)) else a


@cute.jit
def fmax_suppress_nan(a, b, dtype: cutlass.Constexpr = None):
    """max(a, b), NaN-SUPPRESSING: C fmax / torch.fmax -- a NaN operand loses.

    This is plain PTX max.f32, i.e. the same NVVM op WITHOUT the nan flag. (cute.arch.fmax is
    exactly that, but there is no cute.arch.fmin to pair it with, so both go through the
    dialect op and stay symmetric.)
    """
    dt = _dt(a, dtype)
    if const_expr(dt is Float32):
        return Float32(nvvm.fmax(T.f32(), a.ir_value(), b.ir_value()))
    if const_expr(getattr(dt, "is_integer", False)):
        return max(a, b)
    return b if ((b > a) or (a != a)) else a


@cute.jit
def fmin_suppress_nan(a, b, dtype: cutlass.Constexpr = None):
    """min(a, b), NaN-SUPPRESSING: C fmin / torch.fmin."""
    dt = _dt(a, dtype)
    if const_expr(dt is Float32):
        return Float32(nvvm.fmin(T.f32(), a.ir_value(), b.ir_value()))
    if const_expr(getattr(dt, "is_integer", False)):
        return min(a, b)
    return b if ((b < a) or (a != a)) else a
