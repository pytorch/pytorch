import math

import sympy

import torch


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
        assert not isinstance(a, bool)
        return ~a

    @staticmethod
    def reciprocal(x):
        return 1 / x

    @staticmethod
    def square(x):
        return x * x

    @staticmethod
    def mod(x, y):
        return x % y

    @staticmethod
    def abs(x):
        return abs(x)

    @staticmethod
    def neg(x):
        return -x

    @staticmethod
    def truediv(a, b):
        return a / b

    @staticmethod
    def div(a, b):
        return ReferenceAnalysis.truediv(a, b)

    @staticmethod
    def floordiv(a, b):
        if b == 0:
            return sympy.nan if a == 0 else sympy.zoo
        return a // b

    @staticmethod
    def truncdiv(a, b):
        result = a / b
        if result.is_finite:
            result = sympy.Integer(result)

        return result

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def exp(x):
        return sympy.exp(x)

    @staticmethod
    def log(x):
        return sympy.log(x)

    @staticmethod
    def sqrt(x):
        return sympy.sqrt(x)

    @staticmethod
    def pow(a, b):
        return a**b

    @staticmethod
    def minimum(a, b):
        # Poorman's version of upcasting in Sympy
        # This won't do for sympy.Expr as the casting does nothing for those
        if a.is_Float or not a.is_finite or b.is_Float or not b.is_finite:
            result_type = sympy.Float
        else:
            assert a.is_Integer
            assert b.is_Integer
            result_type = sympy.Integer
        return sympy.Min(result_type(a), result_type(b))

    @staticmethod
    def maximum(a, b):
        # Poorman's version of upcasting in Sympy
        # This won't do for sympy.Expr as the casting does nothing for those
        if a.is_Float or not a.is_finite or b.is_Float or not b.is_finite:
            result_type = sympy.Float
        else:
            assert a.is_Integer
            assert b.is_Integer
            result_type = sympy.Integer
        return sympy.Max(result_type(a), result_type(b))

    @staticmethod
    def floor(x):
        return sympy.floor(x)

    @staticmethod
    def ceil(x):
        return sympy.ceiling(x)


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

    @staticmethod
    def floordiv(a, b):
        return a // b

    @staticmethod
    def truncdiv(a, b):
        return a / b

    @staticmethod
    def exp(x):
        raise AssertionError("exp is not valid shape sympy expr")

    @staticmethod
    def log(x):
        raise AssertionError("log is not valid shape sympy expr")

    @staticmethod
    def sqrt(x):
        return torch.sym_sqrt(x)

    @staticmethod
    def minimum(a, b):
        return torch.sym_min(a, b)

    @staticmethod
    def maximum(a, b):
        return torch.sym_max(a, b)

    @staticmethod
    def floor(x):
        return math.floor(x)

    @staticmethod
    def ceil(x):
        return math.ceil(x)
