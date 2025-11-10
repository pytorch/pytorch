# mypy: allow-untyped-defs
import math
import operator
from typing import NoReturn

import sympy

import torch
from torch.utils._sympy.functions import (
    _keep_float,
    BitwiseFn_bitwise_and,
    BitwiseFn_bitwise_or,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    IntTrueDiv,
    Max,
    Min,
    Mod,
    OpaqueUnaryFn_exp,
    OpaqueUnaryFn_log,
    OpaqueUnaryFn_log2,
    OpaqueUnaryFn_sqrt,
    PowByNatural,
    RoundDecimal,
    RoundToInt,
    ToFloat,
    TruncToInt,
)


# The sympy interpretation of operators.  It will also sometimes work with
# plain int/float, but if you do certain operations you will get out a
# sympy.Basic in the end.  If you want the Python/FX traceable interpretation,
# check PythonReferenceAnalysis.
# NB: For magic methods this needs to use normal magic methods
# so that test_magic_methods works
class ReferenceAnalysis:
    @staticmethod
    def constant(c, dtype):
        return sympy.sympify(c)

    @staticmethod
    def or_(a, b):
        return a | b

    @staticmethod
    def and_(a, b):
        return a & b

    @staticmethod
    def eq(a, b):
        if isinstance(a, sympy.Expr) or isinstance(b, sympy.Expr):
            return sympy.Eq(a, b)
        return a == b

    @classmethod
    def ne(cls, a, b):
        return cls.not_(cls.eq(a, b))

    @staticmethod
    def lt(a, b):
        return a < b

    @staticmethod
    def gt(a, b):
        return a > b

    @staticmethod
    def le(a, b):
        return a <= b

    @staticmethod
    def ge(a, b):
        return a >= b

    @staticmethod
    def not_(a):
        if isinstance(a, bool):
            raise AssertionError("not_ needs sympy expr")
        return ~a

    @staticmethod
    def reciprocal(x):
        return FloatTrueDiv(1.0, x)

    @staticmethod
    def square(x):
        return PowByNatural(x, 2)

    @staticmethod
    def trunc_to_int(x, dtype):
        return TruncToInt(x)

    @staticmethod
    def ceil_to_int(x, dtype):
        return sympy.ceiling(x)

    @staticmethod
    def floor_to_int(x, dtype):
        return sympy.floor(x)

    @staticmethod
    def floor(x):
        return _keep_float(sympy.floor)(x)

    @staticmethod
    def ceil(x):
        return _keep_float(sympy.ceiling)(x)

    @staticmethod
    def to_dtype(x, dtype):
        if dtype == torch.float64:
            return ToFloat(x)
        raise NotImplementedError(f"to_dtype {dtype} NYI")

    @staticmethod
    def mod(x, y):
        return Mod(x, y)

    @staticmethod
    def abs(x):
        return abs(x)

    @staticmethod
    def neg(x):
        return -x

    @staticmethod
    def truediv(a, b):
        return FloatTrueDiv(a, b)

    @staticmethod
    def int_truediv(a, b):
        return IntTrueDiv(a, b)

    @staticmethod
    def floordiv(a, b):
        return FloorDiv(a, b)

    @staticmethod
    def truncdiv(a, b) -> NoReturn:
        raise NotImplementedError("TODO: truncdiv")

    @staticmethod
    def add(a, b):
        return _keep_float(operator.add)(a, b)

    @classmethod
    def sym_sum(cls, args):
        return sympy.Add(*args)

    @staticmethod
    def mul(a, b):
        return _keep_float(operator.mul)(a, b)

    @staticmethod
    def sub(a, b):
        return _keep_float(operator.sub)(a, b)

    @staticmethod
    def exp(x):
        return OpaqueUnaryFn_exp(x)

    @staticmethod
    def log(x):
        return OpaqueUnaryFn_log(x)

    @staticmethod
    def log2(x):
        return OpaqueUnaryFn_log2(x)

    @staticmethod
    def sqrt(x):
        return OpaqueUnaryFn_sqrt(x)

    @staticmethod
    def pow(a, b):
        # pyrefly: ignore [bad-argument-type]
        return _keep_float(FloatPow)(a, b)

    @staticmethod
    def pow_by_natural(a, b):
        return PowByNatural(a, b)

    @staticmethod
    def minimum(a, b):
        return Min(a, b)

    @staticmethod
    def maximum(a, b):
        return Max(a, b)

    @staticmethod
    def round_to_int(a, dtype):
        return RoundToInt(a)

    @staticmethod
    def round_decimal(a, b):
        return RoundDecimal(a, b)

    @staticmethod
    def bitwise_and(a, b):
        return BitwiseFn_bitwise_and(a, b)

    @staticmethod
    def bitwise_or(a, b):
        return BitwiseFn_bitwise_or(a, b)


# Unlike ReferenceAnalysis, does NOT sympyify, instead, works with plain
# Python types and is FX traceable.  Inheritance here is purely for code
# sharing (TODO: considering splitting out a BaseReferenceAnalysis).
class PythonReferenceAnalysis(ReferenceAnalysis):
    @staticmethod
    def constant(c, dtype):
        if dtype is torch.int64:
            return int(c)
        elif dtype is torch.double:
            return float(c)
        elif dtype is torch.bool:
            return bool(c)
        else:
            raise AssertionError(f"unrecognized dtype {dtype}")

    @staticmethod
    def not_(a):
        return torch.sym_not(a)

    @classmethod
    def sym_sum(cls, args):
        if len(args) == 0:
            return 0
        if len(args) == 1:
            return args[0]
        acc = cls.add(args[0], args[1])
        for i in range(2, len(args)):
            acc = cls.add(acc, args[i])
        return acc

    @staticmethod
    def floordiv(a, b):
        return a // b

    @staticmethod
    def mod(x, y):
        return x % y

    @staticmethod
    def truncdiv(a, b):
        return a / b

    @staticmethod
    def to_dtype(x, dtype):
        if dtype == torch.float64:
            return torch.sym_float(x)
        raise NotImplementedError(f"to_dtype {dtype} NYI")

    @staticmethod
    def exp(x) -> NoReturn:
        raise AssertionError("exp is not valid shape sympy expr")

    @staticmethod
    def log(x) -> NoReturn:
        raise AssertionError("log is not valid shape sympy expr")

    @staticmethod
    def log2(x):
        return torch._sym_log2(x)  # type: ignore[attr-defined]

    @staticmethod
    def sqrt(x):
        return torch._sym_sqrt(x)  # type: ignore[attr-defined]

    @staticmethod
    def minimum(a, b):
        return torch.sym_min(a, b)

    @staticmethod
    def maximum(a, b):
        return torch.sym_max(a, b)

    @staticmethod
    def floor_to_int(x, dtype):
        return math.floor(x)

    @staticmethod
    def ceil_to_int(x, dtype):
        return math.ceil(x)

    @staticmethod
    def floor(x):
        return float(math.floor(x))

    @staticmethod
    def ceil(x):
        return float(math.ceil(x))

    @staticmethod
    def truediv(a, b):
        return a / b

    @staticmethod
    def pow(a, b):
        return a**b

    @staticmethod
    def pow_by_natural(a, b):
        # Pray that safe_pow is not needed here lol.  In particular, this
        # never participates in VR low/high ranges, so overflow should be
        # unlikely
        return a**b

    @staticmethod
    def round_to_int(a, dtype):
        return round(a)

    @staticmethod
    def round_decimal(a, b):
        return round(a, ndigits=b)

    @staticmethod
    def bitwise_and(a, b):
        return a & b

    @staticmethod
    def bitwise_or(a, b):
        return a | b


# Like PythonReferenceAnalysis, but some export-unfriendly choices of
# operators to make things faster
class OptimizedPythonReferenceAnalysis(PythonReferenceAnalysis):
    @staticmethod
    def sym_sum(args):
        return torch.sym_sum(args)


def _to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.ops.prims.convert_element_type.default(x, dtype)


# Suppose we have some int/float arguments.  This diagram commutes:
#
#   int/float  -- PythonReferenceAnalysis.op -->  int/float
#       |                                           |
#       |                                           |
#      torch.tensor(..., dtype=torch.int64/torch.float64)
#       |                                           |
#       V                                           V
#    Tensor    -- TensorReferenceAnalysis.op -->  Tensor
#
# NB: int before and after must be representable in int64 (we will
# insert guards accordingly.)
#
# This is guaranteed to be FX traceable with OpOverloads only.
class TensorReferenceAnalysis:
    # NB: This is actually dead, because with Proxy tracing the factory
    # function isn't traced correctly.  Here for completeness.
    @staticmethod
    def constant(c, dtype):
        d: int | float | bool
        if dtype is torch.int64:
            d = int(c)
        elif dtype is torch.double:
            d = float(c)
        elif dtype is torch.bool:
            d = bool(c)
        else:
            raise AssertionError(f"unrecognized dtype {dtype}")
        return torch.ops.aten.scalar_tensor.default(d, dtype=dtype)

    @staticmethod
    def or_(a, b):
        return torch.ops.aten.logical_or.default(a, b)

    @staticmethod
    def and_(a, b):
        return torch.ops.aten.logical_and.default(a, b)

    @staticmethod
    def bitwise_and(a, b):
        return torch.ops.aten.bitwise_and(a, b)

    @staticmethod
    def bitwise_or(a, b):
        return torch.ops.aten.bitwise_or(a, b)

    @staticmethod
    def eq(a, b):
        return torch.ops.aten.eq.Tensor(a, b)

    @classmethod
    def ne(cls, a, b):
        return torch.ops.aten.ne.Tensor(a, b)

    @staticmethod
    def lt(a, b):
        return torch.ops.aten.lt.Tensor(a, b)

    @staticmethod
    def gt(a, b):
        return torch.ops.aten.gt.Tensor(a, b)

    @staticmethod
    def le(a, b):
        return torch.ops.aten.le.Tensor(a, b)

    @staticmethod
    def ge(a, b):
        return torch.ops.aten.ge.Tensor(a, b)

    @staticmethod
    def not_(a):
        return torch.ops.aten.logical_not.default(a)

    @staticmethod
    def reciprocal(x):
        return torch.ops.aten.reciprocal.default(x)

    @staticmethod
    def square(x):
        # TODO: maybe composite implicit autograd doesn't work here?
        return torch.ops.aten.square.default(x)

    @staticmethod
    def trunc_to_int(x, dtype):
        return _to_dtype(torch.ops.aten.trunc.default(x), dtype)

    @staticmethod
    def ceil_to_int(x, dtype):
        return _to_dtype(torch.ops.aten.ceil.default(x), dtype)

    @staticmethod
    def floor_to_int(x, dtype):
        return _to_dtype(torch.ops.aten.floor.default(x), dtype)

    @staticmethod
    def floor(x):
        return torch.ops.aten.floor.default(x)

    @staticmethod
    def ceil(x):
        return torch.ops.aten.ceil.default(x)

    @staticmethod
    def to_dtype(x, dtype):
        return _to_dtype(x, dtype)

    @staticmethod
    def mod(x, y) -> NoReturn:
        # TODO: https://github.com/pytorch/pytorch/pull/133654
        raise NotImplementedError(
            "no C-style modulus operation available from frontend atm"
        )

    @staticmethod
    def abs(x):
        return torch.ops.aten.abs.default(x)

    @staticmethod
    def neg(x):
        return torch.ops.aten.neg.default(x)

    @staticmethod
    def truediv(a, b):
        return torch.ops.aten.true_divide.Tensor(a, b)

    @staticmethod
    def int_truediv(a, b):
        raise NotImplementedError(
            "Python int truediv difficult to implement in PyTorch atm"
        )

        # TODO: This is wrong, CPython has a custom implementation of true
        # division that results in higher precision when the floats are
        # sufficiently large.  Short term fix: add a guard here
        return torch.ops.aten.true_divide.default(
            _to_dtype(a, torch.float64), _to_dtype(b, torch.float64)
        )

    @staticmethod
    def floordiv(a, b):
        return torch.ops.aten.div.Tensor_mode(a, b, rounding_mode="floor")

    @staticmethod
    def truncdiv(a, b) -> NoReturn:
        raise NotImplementedError(
            "no C-style truncdiv operation available from frontend atm"
        )

    @staticmethod
    def add(a, b):
        return torch.ops.aten.add.Tensor(a, b)

    @staticmethod
    def mul(a, b):
        return torch.ops.aten.mul.Tensor(a, b)

    @staticmethod
    def sub(a, b):
        return torch.ops.aten.sub.Tensor(a, b)

    @staticmethod
    def exp(x):
        return torch.ops.aten.exp.default(x)

    @staticmethod
    def log(x):
        return torch.ops.aten.log.default(x)

    @staticmethod
    def log2(x):
        return torch.ops.aten.log2.default(x)

    @staticmethod
    def sqrt(x):
        return torch.ops.aten.sqrt.default(x)

    @staticmethod
    def sin(x):
        return torch.ops.aten.sin.default(x)

    @staticmethod
    def cos(x):
        return torch.ops.aten.cos.default(x)

    @staticmethod
    def tanh(x):
        return torch.ops.aten.tanh.default(x)

    @staticmethod
    def sinh(x):
        return torch.ops.aten.sinh.default(x)

    @staticmethod
    def cosh(x):
        return torch.ops.aten.cosh.default(x)

    @staticmethod
    def tan(x):
        return torch.ops.aten.tan.default(x)

    @staticmethod
    def acos(x):
        return torch.ops.aten.acos.default(x)

    @staticmethod
    def atan(x):
        return torch.ops.aten.atan.default(x)

    @staticmethod
    def asin(x):
        return torch.ops.aten.asin.default(x)

    @staticmethod
    def pow(a, b):
        return torch.ops.aten.pow.Tensor_Tensor(a, b)

    @staticmethod
    def pow_by_natural(a, b):
        # NB: pow handles int x int fine
        return torch.ops.aten.pow.Tensor_Tensor(a, b)

    @staticmethod
    def minimum(a, b):
        return torch.ops.aten.minimum.default(a, b)

    @staticmethod
    def maximum(a, b):
        return torch.ops.aten.maximum.default(a, b)

    @staticmethod
    def round_to_int(a, dtype):
        return torch.ops.aten.round.default(a)

    @staticmethod
    def round_decimal(a, b) -> NoReturn:
        raise NotImplementedError(
            "round decimal doesn't support Tensor second argument atm"
        )

        # return torch.ops.aten.round.decimals(a, b)
