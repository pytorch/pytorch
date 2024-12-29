from sympy.external import import_module
from sympy.plotting.intervalmath import (
    Abs, acos, acosh, And, asin, asinh, atan, atanh, ceil, cos, cosh,
    exp, floor, imax, imin, interval, log, log10, Or, sin, sinh, sqrt,
    tan, tanh,
)

np = import_module('numpy')
if not np:
    disabled = True


#requires Numpy. Hence included in interval_functions


def test_interval_pow():
    a = 2**interval(1, 2) == interval(2, 4)
    assert a == (True, True)
    a = interval(1, 2)**interval(1, 2) == interval(1, 4)
    assert a == (True, True)
    a = interval(-1, 1)**interval(0.5, 2)
    assert a.is_valid is None
    a = interval(-2, -1) ** interval(1, 2)
    assert a.is_valid is False
    a = interval(-2, -1) ** (1.0 / 2)
    assert a.is_valid is False
    a = interval(-1, 1)**(1.0 / 2)
    assert a.is_valid is None
    a = interval(-1, 1)**(1.0 / 3) == interval(-1, 1)
    assert a == (True, True)
    a = interval(-1, 1)**2 == interval(0, 1)
    assert a == (True, True)
    a = interval(-1, 1) ** (1.0 / 29) == interval(-1, 1)
    assert a == (True, True)
    a = -2**interval(1, 1) == interval(-2, -2)
    assert a == (True, True)

    a = interval(1, 2, is_valid=False)**2
    assert a.is_valid is False

    a = (-3)**interval(1, 2)
    assert a.is_valid is False
    a = (-4)**interval(0.5, 0.5)
    assert a.is_valid is False
    assert ((-3)**interval(1, 1) == interval(-3, -3)) == (True, True)

    a = interval(8, 64)**(2.0 / 3)
    assert abs(a.start - 4) < 1e-10  # eps
    assert abs(a.end - 16) < 1e-10
    a = interval(-8, 64)**(2.0 / 3)
    assert abs(a.start - 4) < 1e-10  # eps
    assert abs(a.end - 16) < 1e-10


def test_exp():
    a = exp(interval(-np.inf, 0))
    assert a.start == np.exp(-np.inf)
    assert a.end == np.exp(0)
    a = exp(interval(1, 2))
    assert a.start == np.exp(1)
    assert a.end == np.exp(2)
    a = exp(1)
    assert a.start == np.exp(1)
    assert a.end == np.exp(1)


def test_log():
    a = log(interval(1, 2))
    assert a.start == 0
    assert a.end == np.log(2)
    a = log(interval(-1, 1))
    assert a.is_valid is None
    a = log(interval(-3, -1))
    assert a.is_valid is False
    a = log(-3)
    assert a.is_valid is False
    a = log(2)
    assert a.start == np.log(2)
    assert a.end == np.log(2)


def test_log10():
    a = log10(interval(1, 2))
    assert a.start == 0
    assert a.end == np.log10(2)
    a = log10(interval(-1, 1))
    assert a.is_valid is None
    a = log10(interval(-3, -1))
    assert a.is_valid is False
    a = log10(-3)
    assert a.is_valid is False
    a = log10(2)
    assert a.start == np.log10(2)
    assert a.end == np.log10(2)


def test_atan():
    a = atan(interval(0, 1))
    assert a.start == np.arctan(0)
    assert a.end == np.arctan(1)
    a = atan(1)
    assert a.start == np.arctan(1)
    assert a.end == np.arctan(1)


def test_sin():
    a = sin(interval(0, np.pi / 4))
    assert a.start == np.sin(0)
    assert a.end == np.sin(np.pi / 4)

    a = sin(interval(-np.pi / 4, np.pi / 4))
    assert a.start == np.sin(-np.pi / 4)
    assert a.end == np.sin(np.pi / 4)

    a = sin(interval(np.pi / 4, 3 * np.pi / 4))
    assert a.start == np.sin(np.pi / 4)
    assert a.end == 1

    a = sin(interval(7 * np.pi / 6, 7 * np.pi / 4))
    assert a.start == -1
    assert a.end == np.sin(7 * np.pi / 6)

    a = sin(interval(0, 3 * np.pi))
    assert a.start == -1
    assert a.end == 1

    a = sin(interval(np.pi / 3, 7 * np.pi / 4))
    assert a.start == -1
    assert a.end == 1

    a = sin(np.pi / 4)
    assert a.start == np.sin(np.pi / 4)
    assert a.end == np.sin(np.pi / 4)

    a = sin(interval(1, 2, is_valid=False))
    assert a.is_valid is False


def test_cos():
    a = cos(interval(0, np.pi / 4))
    assert a.start == np.cos(np.pi / 4)
    assert a.end == 1

    a = cos(interval(-np.pi / 4, np.pi / 4))
    assert a.start == np.cos(-np.pi / 4)
    assert a.end == 1

    a = cos(interval(np.pi / 4, 3 * np.pi / 4))
    assert a.start == np.cos(3 * np.pi / 4)
    assert a.end == np.cos(np.pi / 4)

    a = cos(interval(3 * np.pi / 4, 5 * np.pi / 4))
    assert a.start == -1
    assert a.end == np.cos(3 * np.pi / 4)

    a = cos(interval(0, 3 * np.pi))
    assert a.start == -1
    assert a.end == 1

    a = cos(interval(- np.pi / 3, 5 * np.pi / 4))
    assert a.start == -1
    assert a.end == 1

    a = cos(interval(1, 2, is_valid=False))
    assert a.is_valid is False


def test_tan():
    a = tan(interval(0, np.pi / 4))
    assert a.start == 0
    # must match lib_interval definition of tan:
    assert a.end == np.sin(np.pi / 4)/np.cos(np.pi / 4)

    a = tan(interval(np.pi / 4, 3 * np.pi / 4))
    #discontinuity
    assert a.is_valid is None


def test_sqrt():
    a = sqrt(interval(1, 4))
    assert a.start == 1
    assert a.end == 2

    a = sqrt(interval(0.01, 1))
    assert a.start == np.sqrt(0.01)
    assert a.end == 1

    a = sqrt(interval(-1, 1))
    assert a.is_valid is None

    a = sqrt(interval(-3, -1))
    assert a.is_valid is False

    a = sqrt(4)
    assert (a == interval(2, 2)) == (True, True)

    a = sqrt(-3)
    assert a.is_valid is False


def test_imin():
    a = imin(interval(1, 3), interval(2, 5), interval(-1, 3))
    assert a.start == -1
    assert a.end == 3

    a = imin(-2, interval(1, 4))
    assert a.start == -2
    assert a.end == -2

    a = imin(5, interval(3, 4), interval(-2, 2, is_valid=False))
    assert a.start == 3
    assert a.end == 4


def test_imax():
    a = imax(interval(-2, 2), interval(2, 7), interval(-3, 9))
    assert a.start == 2
    assert a.end == 9

    a = imax(8, interval(1, 4))
    assert a.start == 8
    assert a.end == 8

    a = imax(interval(1, 2), interval(3, 4), interval(-2, 2, is_valid=False))
    assert a.start == 3
    assert a.end == 4


def test_sinh():
    a = sinh(interval(-1, 1))
    assert a.start == np.sinh(-1)
    assert a.end == np.sinh(1)

    a = sinh(1)
    assert a.start == np.sinh(1)
    assert a.end == np.sinh(1)


def test_cosh():
    a = cosh(interval(1, 2))
    assert a.start == np.cosh(1)
    assert a.end == np.cosh(2)
    a = cosh(interval(-2, -1))
    assert a.start == np.cosh(-1)
    assert a.end == np.cosh(-2)

    a = cosh(interval(-2, 1))
    assert a.start == 1
    assert a.end == np.cosh(-2)

    a = cosh(1)
    assert a.start == np.cosh(1)
    assert a.end == np.cosh(1)


def test_tanh():
    a = tanh(interval(-3, 3))
    assert a.start == np.tanh(-3)
    assert a.end == np.tanh(3)

    a = tanh(3)
    assert a.start == np.tanh(3)
    assert a.end == np.tanh(3)


def test_asin():
    a = asin(interval(-0.5, 0.5))
    assert a.start == np.arcsin(-0.5)
    assert a.end == np.arcsin(0.5)

    a = asin(interval(-1.5, 1.5))
    assert a.is_valid is None
    a = asin(interval(-2, -1.5))
    assert a.is_valid is False

    a = asin(interval(0, 2))
    assert a.is_valid is None

    a = asin(interval(2, 5))
    assert a.is_valid is False

    a = asin(0.5)
    assert a.start == np.arcsin(0.5)
    assert a.end == np.arcsin(0.5)

    a = asin(1.5)
    assert a.is_valid is False


def test_acos():
    a = acos(interval(-0.5, 0.5))
    assert a.start == np.arccos(0.5)
    assert a.end == np.arccos(-0.5)

    a = acos(interval(-1.5, 1.5))
    assert a.is_valid is None
    a = acos(interval(-2, -1.5))
    assert a.is_valid is False

    a = acos(interval(0, 2))
    assert a.is_valid is None

    a = acos(interval(2, 5))
    assert a.is_valid is False

    a = acos(0.5)
    assert a.start == np.arccos(0.5)
    assert a.end == np.arccos(0.5)

    a = acos(1.5)
    assert a.is_valid is False


def test_ceil():
    a = ceil(interval(0.2, 0.5))
    assert a.start == 1
    assert a.end == 1

    a = ceil(interval(0.5, 1.5))
    assert a.start == 1
    assert a.end == 2
    assert a.is_valid is None

    a = ceil(interval(-5, 5))
    assert a.is_valid is None

    a = ceil(5.4)
    assert a.start == 6
    assert a.end == 6


def test_floor():
    a = floor(interval(0.2, 0.5))
    assert a.start == 0
    assert a.end == 0

    a = floor(interval(0.5, 1.5))
    assert a.start == 0
    assert a.end == 1
    assert a.is_valid is None

    a = floor(interval(-5, 5))
    assert a.is_valid is None

    a = floor(5.4)
    assert a.start == 5
    assert a.end == 5


def test_asinh():
    a = asinh(interval(1, 2))
    assert a.start == np.arcsinh(1)
    assert a.end == np.arcsinh(2)

    a = asinh(0.5)
    assert a.start == np.arcsinh(0.5)
    assert a.end == np.arcsinh(0.5)


def test_acosh():
    a = acosh(interval(3, 5))
    assert a.start == np.arccosh(3)
    assert a.end == np.arccosh(5)

    a = acosh(interval(0, 3))
    assert a.is_valid is None
    a = acosh(interval(-3, 0.5))
    assert a.is_valid is False

    a = acosh(0.5)
    assert a.is_valid is False

    a = acosh(2)
    assert a.start == np.arccosh(2)
    assert a.end == np.arccosh(2)


def test_atanh():
    a = atanh(interval(-0.5, 0.5))
    assert a.start == np.arctanh(-0.5)
    assert a.end == np.arctanh(0.5)

    a = atanh(interval(0, 3))
    assert a.is_valid is None

    a = atanh(interval(-3, -2))
    assert a.is_valid is False

    a = atanh(0.5)
    assert a.start == np.arctanh(0.5)
    assert a.end == np.arctanh(0.5)

    a = atanh(1.5)
    assert a.is_valid is False


def test_Abs():
    assert (Abs(interval(-0.5, 0.5)) == interval(0, 0.5)) == (True, True)
    assert (Abs(interval(-3, -2)) == interval(2, 3)) == (True, True)
    assert (Abs(-3) == interval(3, 3)) == (True, True)


def test_And():
    args = [(True, True), (True, False), (True, None)]
    assert And(*args) == (True, False)

    args = [(False, True), (None, None), (True, True)]
    assert And(*args) == (False, None)


def test_Or():
    args = [(True, True), (True, False), (False, None)]
    assert Or(*args) == (True, True)
    args = [(None, None), (False, None), (False, False)]
    assert Or(*args) == (None, None)
