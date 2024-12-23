"""Implementation of mathematical domains. """

__all__ = [
    'Domain', 'FiniteField', 'IntegerRing', 'RationalField', 'RealField',
    'ComplexField', 'AlgebraicField', 'PolynomialRing', 'FractionField',
    'ExpressionDomain', 'PythonRational',

    'GF', 'FF', 'ZZ', 'QQ', 'ZZ_I', 'QQ_I', 'RR', 'CC', 'EX', 'EXRAW',
]

from .domain import Domain
from .finitefield import FiniteField, FF, GF
from .integerring import IntegerRing, ZZ
from .rationalfield import RationalField, QQ
from .algebraicfield import AlgebraicField
from .gaussiandomains import ZZ_I, QQ_I
from .realfield import RealField, RR
from .complexfield import ComplexField, CC
from .polynomialring import PolynomialRing
from .fractionfield import FractionField
from .expressiondomain import ExpressionDomain, EX
from .expressionrawdomain import EXRAW
from .pythonrational import PythonRational


# This is imported purely for backwards compatibility because some parts of
# the codebase used to import this from here and it's possible that downstream
# does as well:
from sympy.external.gmpy import GROUND_TYPES  # noqa: F401

#
# The rest of these are obsolete and provided only for backwards
# compatibility:
#

from .pythonfinitefield import PythonFiniteField
from .gmpyfinitefield import GMPYFiniteField
from .pythonintegerring import PythonIntegerRing
from .gmpyintegerring import GMPYIntegerRing
from .pythonrationalfield import PythonRationalField
from .gmpyrationalfield import GMPYRationalField

FF_python = PythonFiniteField
FF_gmpy = GMPYFiniteField

ZZ_python = PythonIntegerRing
ZZ_gmpy = GMPYIntegerRing

QQ_python = PythonRationalField
QQ_gmpy = GMPYRationalField

__all__.extend((
    'PythonFiniteField', 'GMPYFiniteField', 'PythonIntegerRing',
    'GMPYIntegerRing', 'PythonRational', 'GMPYRationalField',

    'FF_python', 'FF_gmpy', 'ZZ_python', 'ZZ_gmpy', 'QQ_python', 'QQ_gmpy',
))
