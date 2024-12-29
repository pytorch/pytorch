"""Ground types for various mathematical domains in SymPy. """

import builtins
from sympy.external.gmpy import GROUND_TYPES, factorial, sqrt, is_square, sqrtrem

PythonInteger = builtins.int
PythonReal = builtins.float
PythonComplex = builtins.complex

from .pythonrational import PythonRational

from sympy.core.intfunc import (
    igcdex as python_gcdex,
    igcd2 as python_gcd,
    ilcm as python_lcm,
)

from sympy.core.numbers import (Float as SymPyReal, Integer as SymPyInteger, Rational as SymPyRational)


class _GMPYInteger:
    def __init__(self, obj):
        pass

class _GMPYRational:
    def __init__(self, obj):
        pass


if GROUND_TYPES == 'gmpy':

    from gmpy2 import (
        mpz as GMPYInteger,
        mpq as GMPYRational,
        numer as gmpy_numer,
        denom as gmpy_denom,
        gcdext as gmpy_gcdex,
        gcd as gmpy_gcd,
        lcm as gmpy_lcm,
        qdiv as gmpy_qdiv,
    )
    gcdex = gmpy_gcdex
    gcd = gmpy_gcd
    lcm = gmpy_lcm

elif GROUND_TYPES == 'flint':

    from flint import fmpz as _fmpz

    GMPYInteger = _GMPYInteger
    GMPYRational = _GMPYRational
    gmpy_numer = None
    gmpy_denom = None
    gmpy_gcdex = None
    gmpy_gcd = None
    gmpy_lcm = None
    gmpy_qdiv = None

    def gcd(a, b):
        return a.gcd(b)

    def gcdex(a, b):
        x, y, g = python_gcdex(a, b)
        return _fmpz(x), _fmpz(y), _fmpz(g)

    def lcm(a, b):
        return a.lcm(b)

else:
    GMPYInteger = _GMPYInteger
    GMPYRational = _GMPYRational
    gmpy_numer = None
    gmpy_denom = None
    gmpy_gcdex = None
    gmpy_gcd = None
    gmpy_lcm = None
    gmpy_qdiv = None
    gcdex = python_gcdex
    gcd = python_gcd
    lcm = python_lcm


__all__ = [
    'PythonInteger', 'PythonReal', 'PythonComplex',

    'PythonRational',

    'python_gcdex', 'python_gcd', 'python_lcm',

    'SymPyReal', 'SymPyInteger', 'SymPyRational',

    'GMPYInteger', 'GMPYRational', 'gmpy_numer',
    'gmpy_denom', 'gmpy_gcdex', 'gmpy_gcd', 'gmpy_lcm',
    'gmpy_qdiv',

    'factorial', 'sqrt', 'is_square', 'sqrtrem',

    'GMPYInteger', 'GMPYRational',
]
