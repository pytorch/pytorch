# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_fractions.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase


# redirect import statements
import sys
import importlib.abc

redirect_imports = (
    "test.mapping_tests",
    "test.typinganndata",
    "test.test_grammar",
    "test.test_math",
    "test.test_iter",
    "test.typinganndata.ann_module",
)

class RedirectImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # Check if the import is the problematic one
        if fullname in redirect_imports:
            try:
                # Attempt to import the standalone module
                name = fullname.removeprefix("test.")
                r = importlib.import_module(name)
                # Redirect the module in sys.modules
                sys.modules[fullname] = r
                # Return a module spec from the found module
                return importlib.util.find_spec(name)
            except ImportError:
                return None
        return None

# Add the custom finder to sys.meta_path
sys.meta_path.insert(0, RedirectImportFinder())


# ======= END DYNAMO PATCH =======

"""Tests for Lib/fractions.py."""

import cmath
from decimal import Decimal
from test.support import requires_IEEE_754, adjust_int_max_str_digits
import math
import numbers
import operator
import fractions
import functools
import os
import sys
import typing
import unittest
from copy import copy, deepcopy
import pickle
from pickle import dumps, loads
F = fractions.Fraction

#locate file with float format test values
test_dir = os.path.dirname(__file__) or os.curdir
format_testfile = os.path.join(test_dir, 'mathdata', 'formatfloat_testcases.txt')

class DummyFloat(object):
    """Dummy float class for testing comparisons with Fractions"""

    def __init__(self, value):
        if not isinstance(value, float):
            raise TypeError("DummyFloat can only be initialized from float")
        self.value = value

    def _richcmp(self, other, op):
        if isinstance(other, numbers.Rational):
            return op(F.from_float(self.value), other)
        elif isinstance(other, DummyFloat):
            return op(self.value, other.value)
        else:
            return NotImplemented

    def __eq__(self, other): return self._richcmp(other, operator.eq)
    def __le__(self, other): return self._richcmp(other, operator.le)
    def __lt__(self, other): return self._richcmp(other, operator.lt)
    def __ge__(self, other): return self._richcmp(other, operator.ge)
    def __gt__(self, other): return self._richcmp(other, operator.gt)

    # shouldn't be calling __float__ at all when doing comparisons
    def __float__(self):
        assert False, "__float__ should not be invoked for comparisons"

    # same goes for subtraction
    def __sub__(self, other):
        assert False, "__sub__ should not be invoked for comparisons"
    __rsub__ = __sub__


class DummyRational(object):
    """Test comparison of Fraction with a naive rational implementation."""

    def __init__(self, num, den):
        g = math.gcd(num, den)
        self.num = num // g
        self.den = den // g

    def __eq__(self, other):
        if isinstance(other, fractions.Fraction):
            return (self.num == other._numerator and
                    self.den == other._denominator)
        else:
            return NotImplemented

    def __lt__(self, other):
        return(self.num * other._denominator < self.den * other._numerator)

    def __gt__(self, other):
        return(self.num * other._denominator > self.den * other._numerator)

    def __le__(self, other):
        return(self.num * other._denominator <= self.den * other._numerator)

    def __ge__(self, other):
        return(self.num * other._denominator >= self.den * other._numerator)

    # this class is for testing comparisons; conversion to float
    # should never be used for a comparison, since it loses accuracy
    def __float__(self):
        assert False, "__float__ should not be invoked"

class DummyFraction(fractions.Fraction):
    """Dummy Fraction subclass for copy and deepcopy testing."""


def _components(r):
    return (r.numerator, r.denominator)

def typed_approx_eq(a, b):
    return type(a) == type(b) and (a == b or math.isclose(a, b))

class Symbolic:
    """Simple non-numeric class for testing mixed arithmetic.
    It is not Integral, Rational, Real or Complex, and cannot be converted
    to int, float or complex. but it supports some arithmetic operations.
    """
    def __init__(self, value):
        self.value = value
    def __mul__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(f'{self} * {other}')
    def __rmul__(self, other):
        return self.__class__(f'{other} * {self}')
    def __truediv__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(f'{self} / {other}')
    def __rtruediv__(self, other):
        return self.__class__(f'{other} / {self}')
    def __mod__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(f'{self} % {other}')
    def __rmod__(self, other):
        return self.__class__(f'{other} % {self}')
    def __pow__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(f'{self} ** {other}')
    def __rpow__(self, other):
        return self.__class__(f'{other} ** {self}')
    def __eq__(self, other):
        if other.__class__ != self.__class__:
            return NotImplemented
        return self.value == other.value
    def __str__(self):
        return f'{self.value}'
    def __repr__(self):
        return f'{self.__class__.__name__}({self.value!r})'

class SymbolicReal(Symbolic):
    pass
numbers.Real.register(SymbolicReal)

class SymbolicComplex(Symbolic):
    pass
numbers.Complex.register(SymbolicComplex)

class Rat:
    """Simple Rational class for testing mixed arithmetic."""
    def __init__(self, n, d):
        self.numerator = n
        self.denominator = d
    def __mul__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(self.numerator * other.numerator,
                              self.denominator * other.denominator)
    def __rmul__(self, other):
        return self.__class__(other.numerator * self.numerator,
                              other.denominator * self.denominator)
    def __truediv__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(self.numerator * other.denominator,
                              self.denominator * other.numerator)
    def __rtruediv__(self, other):
        return self.__class__(other.numerator * self.denominator,
                              other.denominator * self.numerator)
    def __mod__(self, other):
        if isinstance(other, F):
            return NotImplemented
        d = self.denominator * other.numerator
        return self.__class__(self.numerator * other.denominator % d, d)
    def __rmod__(self, other):
        d = other.denominator * self.numerator
        return self.__class__(other.numerator * self.denominator % d, d)

        return self.__class__(other.numerator / self.numerator,
                              other.denominator / self.denominator)
    def __pow__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(self.numerator ** other,
                              self.denominator ** other)
    def __float__(self):
        return self.numerator / self.denominator
    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return NotImplemented
        return (typed_approx_eq(self.numerator, other.numerator) and
                typed_approx_eq(self.denominator, other.denominator))
    def __repr__(self):
        return f'{self.__class__.__name__}({self.numerator!r}, {self.denominator!r})'
numbers.Rational.register(Rat)

class Root:
    """Simple Real class for testing mixed arithmetic."""
    def __init__(self, v, n=F(2)):
        self.base = v
        self.degree = n
    def __mul__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(self.base * other**self.degree, self.degree)
    def __rmul__(self, other):
        return self.__class__(other**self.degree * self.base, self.degree)
    def __truediv__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(self.base / other**self.degree, self.degree)
    def __rtruediv__(self, other):
        return self.__class__(other**self.degree / self.base, self.degree)
    def __pow__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(self.base, self.degree / other)
    def __float__(self):
        return float(self.base) ** (1 / float(self.degree))
    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return NotImplemented
        return typed_approx_eq(self.base, other.base) and typed_approx_eq(self.degree, other.degree)
    def __repr__(self):
        return f'{self.__class__.__name__}({self.base!r}, {self.degree!r})'
numbers.Real.register(Root)

class Polar:
    """Simple Complex class for testing mixed arithmetic."""
    def __init__(self, r, phi):
        self.r = r
        self.phi = phi
    def __mul__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(self.r * other, self.phi)
    def __rmul__(self, other):
        return self.__class__(other * self.r, self.phi)
    def __truediv__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(self.r / other, self.phi)
    def __rtruediv__(self, other):
        return self.__class__(other / self.r, -self.phi)
    def __pow__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(self.r ** other, self.phi * other)
    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return NotImplemented
        return typed_approx_eq(self.r, other.r) and typed_approx_eq(self.phi, other.phi)
    def __repr__(self):
        return f'{self.__class__.__name__}({self.r!r}, {self.phi!r})'
numbers.Complex.register(Polar)

class Rect:
    """Other simple Complex class for testing mixed arithmetic."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __mul__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(self.x * other, self.y * other)
    def __rmul__(self, other):
        return self.__class__(other * self.x, other * self.y)
    def __truediv__(self, other):
        if isinstance(other, F):
            return NotImplemented
        return self.__class__(self.x / other, self.y / other)
    def __rtruediv__(self, other):
        r = self.x * self.x + self.y * self.y
        return self.__class__(other * (self.x / r), other * (self.y / r))
    def __rpow__(self, other):
        return Polar(other ** self.x, math.log(other) * self.y)
    def __complex__(self):
        return complex(self.x, self.y)
    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return NotImplemented
        return typed_approx_eq(self.x, other.x) and typed_approx_eq(self.y, other.y)
    def __repr__(self):
        return f'{self.__class__.__name__}({self.x!r}, {self.y!r})'
numbers.Complex.register(Rect)

class RectComplex(Rect, complex):
    pass

class FractionTest(__TestCase):

    def assertTypedEquals(self, expected, actual):
        """Asserts that both the types and values are the same."""
        self.assertEqual(type(expected), type(actual))
        self.assertEqual(expected, actual)

    def assertTypedTupleEquals(self, expected, actual):
        """Asserts that both the types and values in the tuples are the same."""
        self.assertTupleEqual(expected, actual)
        self.assertListEqual(list(map(type, expected)), list(map(type, actual)))

    def assertRaisesMessage(self, exc_type, message,
                            callable, *args, **kwargs):
        """Asserts that callable(*args, **kwargs) raises exc_type(message)."""
        try:
            callable(*args, **kwargs)
        except exc_type as e:
            self.assertEqual(message, str(e))
        else:
            self.fail("%s not raised" % exc_type.__name__)

    def testInit(self):
        self.assertEqual((0, 1), _components(F()))
        self.assertEqual((7, 1), _components(F(7)))
        self.assertEqual((7, 3), _components(F(F(7, 3))))

        self.assertEqual((-1, 1), _components(F(-1, 1)))
        self.assertEqual((-1, 1), _components(F(1, -1)))
        self.assertEqual((1, 1), _components(F(-2, -2)))
        self.assertEqual((1, 2), _components(F(5, 10)))
        self.assertEqual((7, 15), _components(F(7, 15)))
        self.assertEqual((10**23, 1), _components(F(10**23)))

        self.assertEqual((3, 77), _components(F(F(3, 7), 11)))
        self.assertEqual((-9, 5), _components(F(2, F(-10, 9))))
        self.assertEqual((2486, 2485), _components(F(F(22, 7), F(355, 113))))

        self.assertRaisesMessage(ZeroDivisionError, "Fraction(12, 0)",
                                 F, 12, 0)
        self.assertRaises(TypeError, F, 1.5 + 3j)

        self.assertRaises(TypeError, F, "3/2", 3)
        self.assertRaises(TypeError, F, 3, 0j)
        self.assertRaises(TypeError, F, 3, 1j)
        self.assertRaises(TypeError, F, 1, 2, 3)

    @requires_IEEE_754
    def testInitFromFloat(self):
        self.assertEqual((5, 2), _components(F(2.5)))
        self.assertEqual((0, 1), _components(F(-0.0)))
        self.assertEqual((3602879701896397, 36028797018963968),
                         _components(F(0.1)))
        # bug 16469: error types should be consistent with float -> int
        self.assertRaises(ValueError, F, float('nan'))
        self.assertRaises(OverflowError, F, float('inf'))
        self.assertRaises(OverflowError, F, float('-inf'))

    def testInitFromDecimal(self):
        self.assertEqual((11, 10),
                         _components(F(Decimal('1.1'))))
        self.assertEqual((7, 200),
                         _components(F(Decimal('3.5e-2'))))
        self.assertEqual((0, 1),
                         _components(F(Decimal('.000e20'))))
        # bug 16469: error types should be consistent with decimal -> int
        self.assertRaises(ValueError, F, Decimal('nan'))
        self.assertRaises(ValueError, F, Decimal('snan'))
        self.assertRaises(OverflowError, F, Decimal('inf'))
        self.assertRaises(OverflowError, F, Decimal('-inf'))

    def testFromString(self):
        self.assertEqual((5, 1), _components(F("5")))
        self.assertEqual((5, 1), _components(F("005")))
        self.assertEqual((3, 2), _components(F("3/2")))
        self.assertEqual((3, 2), _components(F("3 / 2")))
        self.assertEqual((3, 2), _components(F(" \n  +3/2")))
        self.assertEqual((-3, 2), _components(F("-3/2  ")))
        self.assertEqual((13, 2), _components(F("    0013/002 \n  ")))
        self.assertEqual((16, 5), _components(F(" 3.2 ")))
        self.assertEqual((16, 5), _components(F("003.2")))
        self.assertEqual((-16, 5), _components(F(" -3.2 ")))
        self.assertEqual((-3, 1), _components(F(" -3. ")))
        self.assertEqual((3, 5), _components(F(" .6 ")))
        self.assertEqual((1, 3125), _components(F("32.e-5")))
        self.assertEqual((1000000, 1), _components(F("1E+06")))
        self.assertEqual((-12300, 1), _components(F("-1.23e4")))
        self.assertEqual((0, 1), _components(F(" .0e+0\t")))
        self.assertEqual((0, 1), _components(F("-0.000e0")))
        self.assertEqual((123, 1), _components(F("1_2_3")))
        self.assertEqual((41, 107), _components(F("1_2_3/3_2_1")))
        self.assertEqual((6283, 2000), _components(F("3.14_15")))
        self.assertEqual((6283, 2*10**13), _components(F("3.14_15e-1_0")))
        self.assertEqual((101, 100), _components(F("1.01")))
        self.assertEqual((101, 100), _components(F("1.0_1")))

        self.assertRaisesMessage(
            ZeroDivisionError, "Fraction(3, 0)",
            F, "3/0")

        def check_invalid(s):
            msg = "Invalid literal for Fraction: " + repr(s)
            self.assertRaisesMessage(ValueError, msg, F, s)

        check_invalid("3/")
        check_invalid("/2")
        # Denominators don't need a sign.
        check_invalid("3/+2")
        check_invalid("3/-2")
        # Imitate float's parsing.
        check_invalid("+ 3/2")
        check_invalid("- 3/2")
        # Avoid treating '.' as a regex special character.
        check_invalid("3a2")
        # Don't accept combinations of decimals and rationals.
        check_invalid("3/7.2")
        check_invalid("3.2/7")
        # No space around dot.
        check_invalid("3 .2")
        check_invalid("3. 2")
        # No space around e.
        check_invalid("3.2 e1")
        check_invalid("3.2e 1")
        # Fractional part don't need a sign.
        check_invalid("3.+2")
        check_invalid("3.-2")
        # Only accept base 10.
        check_invalid("0x10")
        check_invalid("0x10/1")
        check_invalid("1/0x10")
        check_invalid("0x10.")
        check_invalid("0x10.1")
        check_invalid("1.0x10")
        check_invalid("1.0e0x10")
        # Only accept decimal digits.
        check_invalid("³")
        check_invalid("³/2")
        check_invalid("3/²")
        check_invalid("³.2")
        check_invalid("3.²")
        check_invalid("3.2e²")
        check_invalid("¼")
        # Allow 3. and .3, but not .
        check_invalid(".")
        check_invalid("_")
        check_invalid("_1")
        check_invalid("1__2")
        check_invalid("/_")
        check_invalid("1_/")
        check_invalid("_1/")
        check_invalid("1__2/")
        check_invalid("1/_")
        check_invalid("1/_1")
        check_invalid("1/1__2")
        check_invalid("1._111")
        check_invalid("1.1__1")
        check_invalid("1.1e+_1")
        check_invalid("1.1e+1__1")
        check_invalid("123.dd")
        check_invalid("123.5_dd")
        check_invalid("dd.5")
        check_invalid("7_dd")
        check_invalid("1/dd")
        check_invalid("1/123_dd")
        check_invalid("789edd")
        check_invalid("789e2_dd")
        # Test catastrophic backtracking.
        val = "9"*50 + "_"
        check_invalid(val)
        check_invalid("1/" + val)
        check_invalid("1." + val)
        check_invalid("." + val)
        check_invalid("1.1+e" + val)
        check_invalid("1.1e" + val)

    def test_limit_int(self):
        maxdigits = 5000
        with adjust_int_max_str_digits(maxdigits):
            msg = 'Exceeds the limit'
            val = '1' * maxdigits
            num = (10**maxdigits - 1)//9
            self.assertEqual((num, 1), _components(F(val)))
            self.assertRaisesRegex(ValueError, msg, F, val + '1')
            self.assertEqual((num, 2), _components(F(val + '/2')))
            self.assertRaisesRegex(ValueError, msg, F, val + '1/2')
            self.assertEqual((1, num), _components(F('1/' + val)))
            self.assertRaisesRegex(ValueError, msg, F, '1/1' + val)
            self.assertEqual(((10**(maxdigits+1) - 1)//9, 10**maxdigits),
                             _components(F('1.' + val)))
            self.assertRaisesRegex(ValueError, msg, F, '1.1' + val)
            self.assertEqual((num, 10**maxdigits), _components(F('.' + val)))
            self.assertRaisesRegex(ValueError, msg, F, '.1' + val)
            self.assertRaisesRegex(ValueError, msg, F, '1.1e1' + val)
            self.assertEqual((11, 10), _components(F('1.1e' + '0' * maxdigits)))
            self.assertRaisesRegex(ValueError, msg, F, '1.1e' + '0' * (maxdigits+1))

    def testImmutable(self):
        r = F(7, 3)
        r.__init__(2, 15)
        self.assertEqual((7, 3), _components(r))

        self.assertRaises(AttributeError, setattr, r, 'numerator', 12)
        self.assertRaises(AttributeError, setattr, r, 'denominator', 6)
        self.assertEqual((7, 3), _components(r))

        # But if you _really_ need to:
        r._numerator = 4
        r._denominator = 2
        self.assertEqual((4, 2), _components(r))
        # Which breaks some important operations:
        self.assertNotEqual(F(4, 2), r)

    def testFromFloat(self):
        self.assertRaises(TypeError, F.from_float, 3+4j)
        self.assertEqual((10, 1), _components(F.from_float(10)))
        bigint = 1234567890123456789
        self.assertEqual((bigint, 1), _components(F.from_float(bigint)))
        self.assertEqual((0, 1), _components(F.from_float(-0.0)))
        self.assertEqual((10, 1), _components(F.from_float(10.0)))
        self.assertEqual((-5, 2), _components(F.from_float(-2.5)))
        self.assertEqual((99999999999999991611392, 1),
                         _components(F.from_float(1e23)))
        self.assertEqual(float(10**23), float(F.from_float(1e23)))
        self.assertEqual((3602879701896397, 1125899906842624),
                         _components(F.from_float(3.2)))
        self.assertEqual(3.2, float(F.from_float(3.2)))

        inf = 1e1000
        nan = inf - inf
        # bug 16469: error types should be consistent with float -> int
        self.assertRaisesMessage(
            OverflowError, "cannot convert Infinity to integer ratio",
            F.from_float, inf)
        self.assertRaisesMessage(
            OverflowError, "cannot convert Infinity to integer ratio",
            F.from_float, -inf)
        self.assertRaisesMessage(
            ValueError, "cannot convert NaN to integer ratio",
            F.from_float, nan)

    def testFromDecimal(self):
        self.assertRaises(TypeError, F.from_decimal, 3+4j)
        self.assertEqual(F(10, 1), F.from_decimal(10))
        self.assertEqual(F(0), F.from_decimal(Decimal("-0")))
        self.assertEqual(F(5, 10), F.from_decimal(Decimal("0.5")))
        self.assertEqual(F(5, 1000), F.from_decimal(Decimal("5e-3")))
        self.assertEqual(F(5000), F.from_decimal(Decimal("5e3")))
        self.assertEqual(1 - F(1, 10**30),
                         F.from_decimal(Decimal("0." + "9" * 30)))

        # bug 16469: error types should be consistent with decimal -> int
        self.assertRaisesMessage(
            OverflowError, "cannot convert Infinity to integer ratio",
            F.from_decimal, Decimal("inf"))
        self.assertRaisesMessage(
            OverflowError, "cannot convert Infinity to integer ratio",
            F.from_decimal, Decimal("-inf"))
        self.assertRaisesMessage(
            ValueError, "cannot convert NaN to integer ratio",
            F.from_decimal, Decimal("nan"))
        self.assertRaisesMessage(
            ValueError, "cannot convert NaN to integer ratio",
            F.from_decimal, Decimal("snan"))

    def test_is_integer(self):
        self.assertTrue(F(1, 1).is_integer())
        self.assertTrue(F(-1, 1).is_integer())
        self.assertTrue(F(1, -1).is_integer())
        self.assertTrue(F(2, 2).is_integer())
        self.assertTrue(F(-2, 2).is_integer())
        self.assertTrue(F(2, -2).is_integer())

        self.assertFalse(F(1, 2).is_integer())
        self.assertFalse(F(-1, 2).is_integer())
        self.assertFalse(F(1, -2).is_integer())
        self.assertFalse(F(-1, -2).is_integer())

    def test_as_integer_ratio(self):
        self.assertEqual(F(4, 6).as_integer_ratio(), (2, 3))
        self.assertEqual(F(-4, 6).as_integer_ratio(), (-2, 3))
        self.assertEqual(F(4, -6).as_integer_ratio(), (-2, 3))
        self.assertEqual(F(0, 6).as_integer_ratio(), (0, 1))

    def testLimitDenominator(self):
        rpi = F('3.1415926535897932')
        self.assertEqual(rpi.limit_denominator(10000), F(355, 113))
        self.assertEqual(-rpi.limit_denominator(10000), F(-355, 113))
        self.assertEqual(rpi.limit_denominator(113), F(355, 113))
        self.assertEqual(rpi.limit_denominator(112), F(333, 106))
        self.assertEqual(F(201, 200).limit_denominator(100), F(1))
        self.assertEqual(F(201, 200).limit_denominator(101), F(102, 101))
        self.assertEqual(F(0).limit_denominator(10000), F(0))
        for i in (0, -1):
            self.assertRaisesMessage(
                ValueError, "max_denominator should be at least 1",
                F(1).limit_denominator, i)

    def testConversions(self):
        self.assertTypedEquals(-1, math.trunc(F(-11, 10)))
        self.assertTypedEquals(1, math.trunc(F(11, 10)))
        self.assertTypedEquals(-2, math.floor(F(-11, 10)))
        self.assertTypedEquals(-1, math.ceil(F(-11, 10)))
        self.assertTypedEquals(-1, math.ceil(F(-10, 10)))
        self.assertTypedEquals(-1, int(F(-11, 10)))
        self.assertTypedEquals(0, round(F(-1, 10)))
        self.assertTypedEquals(0, round(F(-5, 10)))
        self.assertTypedEquals(-2, round(F(-15, 10)))
        self.assertTypedEquals(-1, round(F(-7, 10)))

        self.assertEqual(False, bool(F(0, 1)))
        self.assertEqual(True, bool(F(3, 2)))
        self.assertTypedEquals(0.1, float(F(1, 10)))

        # Check that __float__ isn't implemented by converting the
        # numerator and denominator to float before dividing.
        self.assertRaises(OverflowError, float, int('2'*400+'7'))
        self.assertAlmostEqual(2.0/3,
                               float(F(int('2'*400+'7'), int('3'*400+'1'))))

        self.assertTypedEquals(0.1+0j, complex(F(1,10)))

    def testSupportsInt(self):
        # See bpo-44547.
        f = F(3, 2)
        self.assertIsInstance(f, typing.SupportsInt)
        self.assertEqual(int(f), 1)
        self.assertEqual(type(int(f)), int)

    def testIntGuaranteesIntReturn(self):
        # Check that int(some_fraction) gives a result of exact type `int`
        # even if the fraction is using some other Integral type for its
        # numerator and denominator.

        class CustomInt(int):
            """
            Subclass of int with just enough machinery to convince the Fraction
            constructor to produce something with CustomInt numerator and
            denominator.
            """

            @property
            def numerator(self):
                return self

            @property
            def denominator(self):
                return CustomInt(1)

            def __mul__(self, other):
                return CustomInt(int(self) * int(other))

            def __floordiv__(self, other):
                return CustomInt(int(self) // int(other))

        f = F(CustomInt(13), CustomInt(5))

        self.assertIsInstance(f.numerator, CustomInt)
        self.assertIsInstance(f.denominator, CustomInt)
        self.assertIsInstance(f, typing.SupportsInt)
        self.assertEqual(int(f), 2)
        self.assertEqual(type(int(f)), int)

    def testBoolGuarateesBoolReturn(self):
        # Ensure that __bool__ is used on numerator which guarantees a bool
        # return.  See also bpo-39274.
        @functools.total_ordering
        class CustomValue:
            denominator = 1

            def __init__(self, value):
                self.value = value

            def __bool__(self):
                return bool(self.value)

            @property
            def numerator(self):
                # required to preserve `self` during instantiation
                return self

            def __eq__(self, other):
                raise AssertionError("Avoid comparisons in Fraction.__bool__")

            __lt__ = __eq__

        # We did not implement all abstract methods, so register:
        numbers.Rational.register(CustomValue)

        numerator = CustomValue(1)
        r = F(numerator)
        # ensure the numerator was not lost during instantiation:
        self.assertIs(r.numerator, numerator)
        self.assertIs(bool(r), True)

        numerator = CustomValue(0)
        r = F(numerator)
        self.assertIs(bool(r), False)

    def testRound(self):
        self.assertTypedEquals(F(-200), round(F(-150), -2))
        self.assertTypedEquals(F(-200), round(F(-250), -2))
        self.assertTypedEquals(F(30), round(F(26), -1))
        self.assertTypedEquals(F(-2, 10), round(F(-15, 100), 1))
        self.assertTypedEquals(F(-2, 10), round(F(-25, 100), 1))

    def testArithmetic(self):
        self.assertEqual(F(1, 2), F(1, 10) + F(2, 5))
        self.assertEqual(F(-3, 10), F(1, 10) - F(2, 5))
        self.assertEqual(F(1, 25), F(1, 10) * F(2, 5))
        self.assertEqual(F(5, 6), F(2, 3) * F(5, 4))
        self.assertEqual(F(1, 4), F(1, 10) / F(2, 5))
        self.assertEqual(F(-15, 8), F(3, 4) / F(-2, 5))
        self.assertRaises(ZeroDivisionError, operator.truediv, F(1), F(0))
        self.assertTypedEquals(2, F(9, 10) // F(2, 5))
        self.assertTypedEquals(10**23, F(10**23, 1) // F(1))
        self.assertEqual(F(5, 6), F(7, 3) % F(3, 2))
        self.assertEqual(F(2, 3), F(-7, 3) % F(3, 2))
        self.assertEqual((F(1), F(5, 6)), divmod(F(7, 3), F(3, 2)))
        self.assertEqual((F(-2), F(2, 3)), divmod(F(-7, 3), F(3, 2)))
        self.assertEqual(F(8, 27), F(2, 3) ** F(3))
        self.assertEqual(F(27, 8), F(2, 3) ** F(-3))
        self.assertTypedEquals(2.0, F(4) ** F(1, 2))
        self.assertEqual(F(1, 1), +F(1, 1))
        z = pow(F(-1), F(1, 2))
        self.assertAlmostEqual(z.real, 0)
        self.assertEqual(z.imag, 1)
        # Regression test for #27539.
        p = F(-1, 2) ** 0
        self.assertEqual(p, F(1, 1))
        self.assertEqual(p.numerator, 1)
        self.assertEqual(p.denominator, 1)
        p = F(-1, 2) ** -1
        self.assertEqual(p, F(-2, 1))
        self.assertEqual(p.numerator, -2)
        self.assertEqual(p.denominator, 1)
        p = F(-1, 2) ** -2
        self.assertEqual(p, F(4, 1))
        self.assertEqual(p.numerator, 4)
        self.assertEqual(p.denominator, 1)

    def testLargeArithmetic(self):
        self.assertTypedEquals(
            F(10101010100808080808080808101010101010000000000000000,
              1010101010101010101010101011111111101010101010101010101010101),
            F(10**35+1, 10**27+1) % F(10**27+1, 10**35-1)
        )
        self.assertTypedEquals(
            F(7, 1901475900342344102245054808064),
            F(-2**100, 3) % F(5, 2**100)
        )
        self.assertTypedTupleEquals(
            (9999999999999999,
             F(10101010100808080808080808101010101010000000000000000,
               1010101010101010101010101011111111101010101010101010101010101)),
            divmod(F(10**35+1, 10**27+1), F(10**27+1, 10**35-1))
        )
        self.assertTypedEquals(
            -2 ** 200 // 15,
            F(-2**100, 3) // F(5, 2**100)
        )
        self.assertTypedEquals(
            1,
            F(5, 2**100) // F(3, 2**100)
        )
        self.assertTypedEquals(
            (1, F(2, 2**100)),
            divmod(F(5, 2**100), F(3, 2**100))
        )
        self.assertTypedTupleEquals(
            (-2 ** 200 // 15,
             F(7, 1901475900342344102245054808064)),
            divmod(F(-2**100, 3), F(5, 2**100))
        )

    def testMixedArithmetic(self):
        self.assertTypedEquals(F(11, 10), F(1, 10) + 1)
        self.assertTypedEquals(1.1, F(1, 10) + 1.0)
        self.assertTypedEquals(1.1 + 0j, F(1, 10) + (1.0 + 0j))
        self.assertTypedEquals(F(11, 10), 1 + F(1, 10))
        self.assertTypedEquals(1.1, 1.0 + F(1, 10))
        self.assertTypedEquals(1.1 + 0j, (1.0 + 0j) + F(1, 10))

        self.assertTypedEquals(F(-9, 10), F(1, 10) - 1)
        self.assertTypedEquals(-0.9, F(1, 10) - 1.0)
        self.assertTypedEquals(-0.9 + 0j, F(1, 10) - (1.0 + 0j))
        self.assertTypedEquals(F(9, 10), 1 - F(1, 10))
        self.assertTypedEquals(0.9, 1.0 - F(1, 10))
        self.assertTypedEquals(0.9 + 0j, (1.0 + 0j) - F(1, 10))

    def testMixedMultiplication(self):
        self.assertTypedEquals(F(1, 10), F(1, 10) * 1)
        self.assertTypedEquals(0.1, F(1, 10) * 1.0)
        self.assertTypedEquals(0.1 + 0j, F(1, 10) * (1.0 + 0j))
        self.assertTypedEquals(F(1, 10), 1 * F(1, 10))
        self.assertTypedEquals(0.1, 1.0 * F(1, 10))
        self.assertTypedEquals(0.1 + 0j, (1.0 + 0j) * F(1, 10))

        self.assertTypedEquals(F(3, 2) * DummyFraction(5, 3), F(5, 2))
        self.assertTypedEquals(DummyFraction(5, 3) * F(3, 2), F(5, 2))
        self.assertTypedEquals(F(3, 2) * Rat(5, 3), Rat(15, 6))
        self.assertTypedEquals(Rat(5, 3) * F(3, 2), F(5, 2))

        self.assertTypedEquals(F(3, 2) * Root(4), Root(F(9, 1)))
        self.assertTypedEquals(Root(4) * F(3, 2), 3.0)
        self.assertEqual(F(3, 2) * SymbolicReal('X'), SymbolicReal('3/2 * X'))
        self.assertRaises(TypeError, operator.mul, SymbolicReal('X'), F(3, 2))

        self.assertTypedEquals(F(3, 2) * Polar(4, 2), Polar(F(6, 1), 2))
        self.assertTypedEquals(F(3, 2) * Polar(4.0, 2), Polar(6.0, 2))
        self.assertTypedEquals(F(3, 2) * Rect(4, 3), Rect(F(6, 1), F(9, 2)))
        self.assertTypedEquals(F(3, 2) * RectComplex(4, 3), RectComplex(6.0+0j, 4.5+0j))
        self.assertRaises(TypeError, operator.mul, Polar(4, 2), F(3, 2))
        self.assertTypedEquals(Rect(4, 3) * F(3, 2), 6.0 + 4.5j)
        self.assertEqual(F(3, 2) * SymbolicComplex('X'), SymbolicComplex('3/2 * X'))
        self.assertRaises(TypeError, operator.mul, SymbolicComplex('X'), F(3, 2))

        self.assertEqual(F(3, 2) * Symbolic('X'), Symbolic('3/2 * X'))
        self.assertRaises(TypeError, operator.mul, Symbolic('X'), F(3, 2))

    def testMixedDivision(self):
        self.assertTypedEquals(F(1, 10), F(1, 10) / 1)
        self.assertTypedEquals(0.1, F(1, 10) / 1.0)
        self.assertTypedEquals(0.1 + 0j, F(1, 10) / (1.0 + 0j))
        self.assertTypedEquals(F(10, 1), 1 / F(1, 10))
        self.assertTypedEquals(10.0, 1.0 / F(1, 10))
        self.assertTypedEquals(10.0 + 0j, (1.0 + 0j) / F(1, 10))

        self.assertTypedEquals(F(3, 2) / DummyFraction(3, 5), F(5, 2))
        self.assertTypedEquals(DummyFraction(5, 3) / F(2, 3), F(5, 2))
        self.assertTypedEquals(F(3, 2) / Rat(3, 5), Rat(15, 6))
        self.assertTypedEquals(Rat(5, 3) / F(2, 3), F(5, 2))

        self.assertTypedEquals(F(2, 3) / Root(4), Root(F(1, 9)))
        self.assertTypedEquals(Root(4) / F(2, 3), 3.0)
        self.assertEqual(F(3, 2) / SymbolicReal('X'), SymbolicReal('3/2 / X'))
        self.assertRaises(TypeError, operator.truediv, SymbolicReal('X'), F(3, 2))

        self.assertTypedEquals(F(3, 2) / Polar(4, 2), Polar(F(3, 8), -2))
        self.assertTypedEquals(F(3, 2) / Polar(4.0, 2), Polar(0.375, -2))
        self.assertTypedEquals(F(3, 2) / Rect(4, 3), Rect(0.24, 0.18))
        self.assertRaises(TypeError, operator.truediv, Polar(4, 2), F(2, 3))
        self.assertTypedEquals(Rect(4, 3) / F(2, 3), 6.0 + 4.5j)
        self.assertEqual(F(3, 2) / SymbolicComplex('X'), SymbolicComplex('3/2 / X'))
        self.assertRaises(TypeError, operator.truediv, SymbolicComplex('X'), F(3, 2))

        self.assertEqual(F(3, 2) / Symbolic('X'), Symbolic('3/2 / X'))
        self.assertRaises(TypeError, operator.truediv, Symbolic('X'), F(2, 3))

    def testMixedIntegerDivision(self):
        self.assertTypedEquals(0, F(1, 10) // 1)
        self.assertTypedEquals(0.0, F(1, 10) // 1.0)
        self.assertTypedEquals(10, 1 // F(1, 10))
        self.assertTypedEquals(10**23, 10**22 // F(1, 10))
        self.assertTypedEquals(1.0 // 0.1, 1.0 // F(1, 10))

        self.assertTypedEquals(F(1, 10), F(1, 10) % 1)
        self.assertTypedEquals(0.1, F(1, 10) % 1.0)
        self.assertTypedEquals(F(0, 1), 1 % F(1, 10))
        self.assertTypedEquals(1.0 % 0.1, 1.0 % F(1, 10))
        self.assertTypedEquals(0.1, F(1, 10) % float('inf'))
        self.assertTypedEquals(float('-inf'), F(1, 10) % float('-inf'))
        self.assertTypedEquals(float('inf'), F(-1, 10) % float('inf'))
        self.assertTypedEquals(-0.1, F(-1, 10) % float('-inf'))

        self.assertTypedTupleEquals((0, F(1, 10)), divmod(F(1, 10), 1))
        self.assertTypedTupleEquals(divmod(0.1, 1.0), divmod(F(1, 10), 1.0))
        self.assertTypedTupleEquals((10, F(0)), divmod(1, F(1, 10)))
        self.assertTypedTupleEquals(divmod(1.0, 0.1), divmod(1.0, F(1, 10)))
        self.assertTypedTupleEquals(divmod(0.1, float('inf')), divmod(F(1, 10), float('inf')))
        self.assertTypedTupleEquals(divmod(0.1, float('-inf')), divmod(F(1, 10), float('-inf')))
        self.assertTypedTupleEquals(divmod(-0.1, float('inf')), divmod(F(-1, 10), float('inf')))
        self.assertTypedTupleEquals(divmod(-0.1, float('-inf')), divmod(F(-1, 10), float('-inf')))

        self.assertTypedEquals(F(3, 2) % DummyFraction(3, 5), F(3, 10))
        self.assertTypedEquals(DummyFraction(5, 3) % F(2, 3), F(1, 3))
        self.assertTypedEquals(F(3, 2) % Rat(3, 5), Rat(3, 6))
        self.assertTypedEquals(Rat(5, 3) % F(2, 3), F(1, 3))

        self.assertRaises(TypeError, operator.mod, F(2, 3), Root(4))
        self.assertTypedEquals(Root(4) % F(3, 2), 0.5)
        self.assertEqual(F(3, 2) % SymbolicReal('X'), SymbolicReal('3/2 % X'))
        self.assertRaises(TypeError, operator.mod, SymbolicReal('X'), F(3, 2))

        self.assertRaises(TypeError, operator.mod, F(3, 2), Polar(4, 2))
        self.assertRaises(TypeError, operator.mod, F(3, 2), RectComplex(4, 3))
        self.assertRaises(TypeError, operator.mod, Rect(4, 3), F(2, 3))
        self.assertEqual(F(3, 2) % SymbolicComplex('X'), SymbolicComplex('3/2 % X'))
        self.assertRaises(TypeError, operator.mod, SymbolicComplex('X'), F(3, 2))

        self.assertEqual(F(3, 2) % Symbolic('X'), Symbolic('3/2 % X'))
        self.assertRaises(TypeError, operator.mod, Symbolic('X'), F(2, 3))

    def testMixedPower(self):
        # ** has more interesting conversion rules.
        self.assertTypedEquals(F(100, 1), F(1, 10) ** -2)
        self.assertTypedEquals(F(100, 1), F(10, 1) ** 2)
        self.assertTypedEquals(0.1, F(1, 10) ** 1.0)
        self.assertTypedEquals(0.1 + 0j, F(1, 10) ** (1.0 + 0j))
        self.assertTypedEquals(4 , 2 ** F(2, 1))
        z = pow(-1, F(1, 2))
        self.assertAlmostEqual(0, z.real)
        self.assertEqual(1, z.imag)
        self.assertTypedEquals(F(1, 4) , 2 ** F(-2, 1))
        self.assertTypedEquals(2.0 , 4 ** F(1, 2))
        self.assertTypedEquals(0.25, 2.0 ** F(-2, 1))
        self.assertTypedEquals(1.0 + 0j, (1.0 + 0j) ** F(1, 10))
        self.assertRaises(ZeroDivisionError, operator.pow,
                          F(0, 1), -2)

        self.assertTypedEquals(F(3, 2) ** Rat(3, 1), F(27, 8))
        self.assertTypedEquals(F(3, 2) ** Rat(-3, 1), F(8, 27))
        self.assertTypedEquals(F(-3, 2) ** Rat(-3, 1), F(-8, 27))
        self.assertTypedEquals(F(9, 4) ** Rat(3, 2), 3.375)
        self.assertIsInstance(F(4, 9) ** Rat(-3, 2), float)
        self.assertAlmostEqual(F(4, 9) ** Rat(-3, 2), 3.375)
        self.assertAlmostEqual(F(-4, 9) ** Rat(-3, 2), 3.375j)
        self.assertTypedEquals(Rat(9, 4) ** F(3, 2), 3.375)
        self.assertTypedEquals(Rat(3, 2) ** F(3, 1), Rat(27, 8))
        self.assertTypedEquals(Rat(3, 2) ** F(-3, 1), F(8, 27))
        self.assertIsInstance(Rat(4, 9) ** F(-3, 2), float)
        self.assertAlmostEqual(Rat(4, 9) ** F(-3, 2), 3.375)

        self.assertTypedEquals(Root(4) ** F(2, 3), Root(4, 3.0))
        self.assertTypedEquals(Root(4) ** F(2, 1), Root(4, F(1)))
        self.assertTypedEquals(Root(4) ** F(-2, 1), Root(4, -F(1)))
        self.assertTypedEquals(Root(4) ** F(-2, 3), Root(4, -3.0))
        self.assertEqual(F(3, 2) ** SymbolicReal('X'), SymbolicReal('3/2 ** X'))
        self.assertEqual(SymbolicReal('X') ** F(3, 2), SymbolicReal('X ** 1.5'))

        self.assertTypedEquals(F(3, 2) ** Rect(2, 0), Polar(F(9,4), 0.0))
        self.assertTypedEquals(F(1, 1) ** Rect(2, 3), Polar(F(1), 0.0))
        self.assertTypedEquals(F(3, 2) ** RectComplex(2, 0), Polar(2.25, 0.0))
        self.assertTypedEquals(F(1, 1) ** RectComplex(2, 3), Polar(1.0, 0.0))
        self.assertTypedEquals(Polar(4, 2) ** F(3, 2), Polar(8.0, 3.0))
        self.assertTypedEquals(Polar(4, 2) ** F(3, 1), Polar(64, 6))
        self.assertTypedEquals(Polar(4, 2) ** F(-3, 1), Polar(0.015625, -6))
        self.assertTypedEquals(Polar(4, 2) ** F(-3, 2), Polar(0.125, -3.0))
        self.assertEqual(F(3, 2) ** SymbolicComplex('X'), SymbolicComplex('3/2 ** X'))
        self.assertEqual(SymbolicComplex('X') ** F(3, 2), SymbolicComplex('X ** 1.5'))

        self.assertEqual(F(3, 2) ** Symbolic('X'), Symbolic('3/2 ** X'))
        self.assertEqual(Symbolic('X') ** F(3, 2), Symbolic('X ** 1.5'))

    def testMixingWithDecimal(self):
        # Decimal refuses mixed arithmetic (but not mixed comparisons)
        self.assertRaises(TypeError, operator.add,
                          F(3,11), Decimal('3.1415926'))
        self.assertRaises(TypeError, operator.add,
                          Decimal('3.1415926'), F(3,11))

    def testComparisons(self):
        self.assertTrue(F(1, 2) < F(2, 3))
        self.assertFalse(F(1, 2) < F(1, 2))
        self.assertTrue(F(1, 2) <= F(2, 3))
        self.assertTrue(F(1, 2) <= F(1, 2))
        self.assertFalse(F(2, 3) <= F(1, 2))
        self.assertTrue(F(1, 2) == F(1, 2))
        self.assertFalse(F(1, 2) == F(1, 3))
        self.assertFalse(F(1, 2) != F(1, 2))
        self.assertTrue(F(1, 2) != F(1, 3))

    def testComparisonsDummyRational(self):
        self.assertTrue(F(1, 2) == DummyRational(1, 2))
        self.assertTrue(DummyRational(1, 2) == F(1, 2))
        self.assertFalse(F(1, 2) == DummyRational(3, 4))
        self.assertFalse(DummyRational(3, 4) == F(1, 2))

        self.assertTrue(F(1, 2) < DummyRational(3, 4))
        self.assertFalse(F(1, 2) < DummyRational(1, 2))
        self.assertFalse(F(1, 2) < DummyRational(1, 7))
        self.assertFalse(F(1, 2) > DummyRational(3, 4))
        self.assertFalse(F(1, 2) > DummyRational(1, 2))
        self.assertTrue(F(1, 2) > DummyRational(1, 7))
        self.assertTrue(F(1, 2) <= DummyRational(3, 4))
        self.assertTrue(F(1, 2) <= DummyRational(1, 2))
        self.assertFalse(F(1, 2) <= DummyRational(1, 7))
        self.assertFalse(F(1, 2) >= DummyRational(3, 4))
        self.assertTrue(F(1, 2) >= DummyRational(1, 2))
        self.assertTrue(F(1, 2) >= DummyRational(1, 7))

        self.assertTrue(DummyRational(1, 2) < F(3, 4))
        self.assertFalse(DummyRational(1, 2) < F(1, 2))
        self.assertFalse(DummyRational(1, 2) < F(1, 7))
        self.assertFalse(DummyRational(1, 2) > F(3, 4))
        self.assertFalse(DummyRational(1, 2) > F(1, 2))
        self.assertTrue(DummyRational(1, 2) > F(1, 7))
        self.assertTrue(DummyRational(1, 2) <= F(3, 4))
        self.assertTrue(DummyRational(1, 2) <= F(1, 2))
        self.assertFalse(DummyRational(1, 2) <= F(1, 7))
        self.assertFalse(DummyRational(1, 2) >= F(3, 4))
        self.assertTrue(DummyRational(1, 2) >= F(1, 2))
        self.assertTrue(DummyRational(1, 2) >= F(1, 7))

    def testComparisonsDummyFloat(self):
        x = DummyFloat(1./3.)
        y = F(1, 3)
        self.assertTrue(x != y)
        self.assertTrue(x < y or x > y)
        self.assertFalse(x == y)
        self.assertFalse(x <= y and x >= y)
        self.assertTrue(y != x)
        self.assertTrue(y < x or y > x)
        self.assertFalse(y == x)
        self.assertFalse(y <= x and y >= x)

    def testMixedLess(self):
        self.assertTrue(2 < F(5, 2))
        self.assertFalse(2 < F(4, 2))
        self.assertTrue(F(5, 2) < 3)
        self.assertFalse(F(4, 2) < 2)

        self.assertTrue(F(1, 2) < 0.6)
        self.assertFalse(F(1, 2) < 0.4)
        self.assertTrue(0.4 < F(1, 2))
        self.assertFalse(0.5 < F(1, 2))

        self.assertFalse(float('inf') < F(1, 2))
        self.assertTrue(float('-inf') < F(0, 10))
        self.assertFalse(float('nan') < F(-3, 7))
        self.assertTrue(F(1, 2) < float('inf'))
        self.assertFalse(F(17, 12) < float('-inf'))
        self.assertFalse(F(144, -89) < float('nan'))

    def testMixedLessEqual(self):
        self.assertTrue(0.5 <= F(1, 2))
        self.assertFalse(0.6 <= F(1, 2))
        self.assertTrue(F(1, 2) <= 0.5)
        self.assertFalse(F(1, 2) <= 0.4)
        self.assertTrue(2 <= F(4, 2))
        self.assertFalse(2 <= F(3, 2))
        self.assertTrue(F(4, 2) <= 2)
        self.assertFalse(F(5, 2) <= 2)

        self.assertFalse(float('inf') <= F(1, 2))
        self.assertTrue(float('-inf') <= F(0, 10))
        self.assertFalse(float('nan') <= F(-3, 7))
        self.assertTrue(F(1, 2) <= float('inf'))
        self.assertFalse(F(17, 12) <= float('-inf'))
        self.assertFalse(F(144, -89) <= float('nan'))

    def testBigFloatComparisons(self):
        # Because 10**23 can't be represented exactly as a float:
        self.assertFalse(F(10**23) == float(10**23))
        # The first test demonstrates why these are important.
        self.assertFalse(1e23 < float(F(math.trunc(1e23) + 1)))
        self.assertTrue(1e23 < F(math.trunc(1e23) + 1))
        self.assertFalse(1e23 <= F(math.trunc(1e23) - 1))
        self.assertTrue(1e23 > F(math.trunc(1e23) - 1))
        self.assertFalse(1e23 >= F(math.trunc(1e23) + 1))

    def testBigComplexComparisons(self):
        self.assertFalse(F(10**23) == complex(10**23))
        self.assertRaises(TypeError, operator.gt, F(10**23), complex(10**23))
        self.assertRaises(TypeError, operator.le, F(10**23), complex(10**23))

        x = F(3, 8)
        z = complex(0.375, 0.0)
        w = complex(0.375, 0.2)
        self.assertTrue(x == z)
        self.assertFalse(x != z)
        self.assertFalse(x == w)
        self.assertTrue(x != w)
        for op in operator.lt, operator.le, operator.gt, operator.ge:
            self.assertRaises(TypeError, op, x, z)
            self.assertRaises(TypeError, op, z, x)
            self.assertRaises(TypeError, op, x, w)
            self.assertRaises(TypeError, op, w, x)

    def testMixedEqual(self):
        self.assertTrue(0.5 == F(1, 2))
        self.assertFalse(0.6 == F(1, 2))
        self.assertTrue(F(1, 2) == 0.5)
        self.assertFalse(F(1, 2) == 0.4)
        self.assertTrue(2 == F(4, 2))
        self.assertFalse(2 == F(3, 2))
        self.assertTrue(F(4, 2) == 2)
        self.assertFalse(F(5, 2) == 2)
        self.assertFalse(F(5, 2) == float('nan'))
        self.assertFalse(float('nan') == F(3, 7))
        self.assertFalse(F(5, 2) == float('inf'))
        self.assertFalse(float('-inf') == F(2, 5))

    def testStringification(self):
        self.assertEqual("Fraction(7, 3)", repr(F(7, 3)))
        self.assertEqual("Fraction(6283185307, 2000000000)",
                         repr(F('3.1415926535')))
        self.assertEqual("Fraction(-1, 100000000000000000000)",
                         repr(F(1, -10**20)))
        self.assertEqual("7/3", str(F(7, 3)))
        self.assertEqual("7", str(F(7, 1)))

    def testHash(self):
        hmod = sys.hash_info.modulus
        hinf = sys.hash_info.inf
        self.assertEqual(hash(2.5), hash(F(5, 2)))
        self.assertEqual(hash(10**50), hash(F(10**50)))
        self.assertNotEqual(hash(float(10**23)), hash(F(10**23)))
        self.assertEqual(hinf, hash(F(1, hmod)))
        # Check that __hash__ produces the same value as hash(), for
        # consistency with int and Decimal.  (See issue #10356.)
        self.assertEqual(hash(F(-1)), F(-1).__hash__())

    def testApproximatePi(self):
        # Algorithm borrowed from
        # http://docs.python.org/lib/decimal-recipes.html
        three = F(3)
        lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
        while abs(s - lasts) > F(1, 10**9):
            lasts = s
            n, na = n+na, na+8
            d, da = d+da, da+32
            t = (t * n) / d
            s += t
        self.assertAlmostEqual(math.pi, s)

    def testApproximateCos1(self):
        # Algorithm borrowed from
        # http://docs.python.org/lib/decimal-recipes.html
        x = F(1)
        i, lasts, s, fact, num, sign = 0, 0, F(1), 1, 1, 1
        while abs(s - lasts) > F(1, 10**9):
            lasts = s
            i += 2
            fact *= i * (i-1)
            num *= x * x
            sign *= -1
            s += num / fact * sign
        self.assertAlmostEqual(math.cos(1), s)

    def test_copy_deepcopy_pickle(self):
        r = F(13, 7)
        dr = DummyFraction(13, 7)
        for proto in range(0, pickle.HIGHEST_PROTOCOL + 1):
            self.assertEqual(r, loads(dumps(r, proto)))
        self.assertEqual(id(r), id(copy(r)))
        self.assertEqual(id(r), id(deepcopy(r)))
        self.assertNotEqual(id(dr), id(copy(dr)))
        self.assertNotEqual(id(dr), id(deepcopy(dr)))
        self.assertTypedEquals(dr, copy(dr))
        self.assertTypedEquals(dr, deepcopy(dr))

    def test_slots(self):
        # Issue 4998
        r = F(13, 7)
        self.assertRaises(AttributeError, setattr, r, 'a', 10)

    def test_int_subclass(self):
        class myint(int):
            def __mul__(self, other):
                return type(self)(int(self) * int(other))
            def __floordiv__(self, other):
                return type(self)(int(self) // int(other))
            def __mod__(self, other):
                x = type(self)(int(self) % int(other))
                return x
            @property
            def numerator(self):
                return type(self)(int(self))
            @property
            def denominator(self):
                return type(self)(1)

        f = fractions.Fraction(myint(1 * 3), myint(2 * 3))
        self.assertEqual(f.numerator, 1)
        self.assertEqual(f.denominator, 2)
        self.assertEqual(type(f.numerator), myint)
        self.assertEqual(type(f.denominator), myint)

    def test_format_no_presentation_type(self):
        # Triples (fraction, specification, expected_result).
        testcases = [
            # Explicit sign handling
            (F(2, 3), '+', '+2/3'),
            (F(-2, 3), '+', '-2/3'),
            (F(3), '+', '+3'),
            (F(-3), '+', '-3'),
            (F(2, 3), ' ', ' 2/3'),
            (F(-2, 3), ' ', '-2/3'),
            (F(3), ' ', ' 3'),
            (F(-3), ' ', '-3'),
            (F(2, 3), '-', '2/3'),
            (F(-2, 3), '-', '-2/3'),
            (F(3), '-', '3'),
            (F(-3), '-', '-3'),
            # Padding
            (F(0), '5', '    0'),
            (F(2, 3), '5', '  2/3'),
            (F(-2, 3), '5', ' -2/3'),
            (F(2, 3), '0', '2/3'),
            (F(2, 3), '1', '2/3'),
            (F(2, 3), '2', '2/3'),
            # Alignment
            (F(2, 3), '<5', '2/3  '),
            (F(2, 3), '>5', '  2/3'),
            (F(2, 3), '^5', ' 2/3 '),
            (F(2, 3), '=5', '  2/3'),
            (F(-2, 3), '<5', '-2/3 '),
            (F(-2, 3), '>5', ' -2/3'),
            (F(-2, 3), '^5', '-2/3 '),
            (F(-2, 3), '=5', '- 2/3'),
            # Fill
            (F(2, 3), 'X>5', 'XX2/3'),
            (F(-2, 3), '.<5', '-2/3.'),
            (F(-2, 3), '\n^6', '\n-2/3\n'),
            # Thousands separators
            (F(1234, 5679), ',', '1,234/5,679'),
            (F(-1234, 5679), '_', '-1_234/5_679'),
            (F(1234567), '_', '1_234_567'),
            (F(-1234567), ',', '-1,234,567'),
            # Alternate form forces a slash in the output
            (F(123), '#', '123/1'),
            (F(-123), '#', '-123/1'),
            (F(0), '#', '0/1'),
        ]
        for fraction, spec, expected in testcases:
            with self.subTest(fraction=fraction, spec=spec):
                self.assertEqual(format(fraction, spec), expected)

    def test_format_e_presentation_type(self):
        # Triples (fraction, specification, expected_result)
        testcases = [
            (F(2, 3), '.6e', '6.666667e-01'),
            (F(3, 2), '.6e', '1.500000e+00'),
            (F(2, 13), '.6e', '1.538462e-01'),
            (F(2, 23), '.6e', '8.695652e-02'),
            (F(2, 33), '.6e', '6.060606e-02'),
            (F(13, 2), '.6e', '6.500000e+00'),
            (F(20, 2), '.6e', '1.000000e+01'),
            (F(23, 2), '.6e', '1.150000e+01'),
            (F(33, 2), '.6e', '1.650000e+01'),
            (F(2, 3), '.6e', '6.666667e-01'),
            (F(3, 2), '.6e', '1.500000e+00'),
            # Zero
            (F(0), '.3e', '0.000e+00'),
            # Powers of 10, to exercise the log10 boundary logic
            (F(1, 1000), '.3e', '1.000e-03'),
            (F(1, 100), '.3e', '1.000e-02'),
            (F(1, 10), '.3e', '1.000e-01'),
            (F(1, 1), '.3e', '1.000e+00'),
            (F(10), '.3e', '1.000e+01'),
            (F(100), '.3e', '1.000e+02'),
            (F(1000), '.3e', '1.000e+03'),
            # Boundary where we round up to the next power of 10
            (F('99.999994999999'), '.6e', '9.999999e+01'),
            (F('99.999995'), '.6e', '1.000000e+02'),
            (F('99.999995000001'), '.6e', '1.000000e+02'),
            # Negatives
            (F(-2, 3), '.6e', '-6.666667e-01'),
            (F(-3, 2), '.6e', '-1.500000e+00'),
            (F(-100), '.6e', '-1.000000e+02'),
            # Large and small
            (F('1e1000'), '.3e', '1.000e+1000'),
            (F('1e-1000'), '.3e', '1.000e-1000'),
            # Using 'E' instead of 'e' should give us a capital 'E'
            (F(2, 3), '.6E', '6.666667E-01'),
            # Tiny precision
            (F(2, 3), '.1e', '6.7e-01'),
            (F('0.995'), '.0e', '1e+00'),
            # Default precision is 6
            (F(22, 7), 'e', '3.142857e+00'),
            # Alternate form forces a decimal point
            (F('0.995'), '#.0e', '1.e+00'),
            # Check that padding takes the exponent into account.
            (F(22, 7), '11.6e', '3.142857e+00'),
            (F(22, 7), '12.6e', '3.142857e+00'),
            (F(22, 7), '13.6e', ' 3.142857e+00'),
            # Thousands separators
            (F('1234567.123456'), ',.5e', '1.23457e+06'),
            (F('123.123456'), '012_.2e', '0_001.23e+02'),
            # z flag is legal, but never makes a difference to the output
            (F(-1, 7**100), 'z.6e', '-3.091690e-85'),
        ]
        for fraction, spec, expected in testcases:
            with self.subTest(fraction=fraction, spec=spec):
                self.assertEqual(format(fraction, spec), expected)

    def test_format_f_presentation_type(self):
        # Triples (fraction, specification, expected_result)
        testcases = [
            # Simple .f formatting
            (F(0, 1), '.2f', '0.00'),
            (F(1, 3), '.2f', '0.33'),
            (F(2, 3), '.2f', '0.67'),
            (F(4, 3), '.2f', '1.33'),
            (F(1, 8), '.2f', '0.12'),
            (F(3, 8), '.2f', '0.38'),
            (F(1, 13), '.2f', '0.08'),
            (F(1, 199), '.2f', '0.01'),
            (F(1, 200), '.2f', '0.00'),
            (F(22, 7), '.5f', '3.14286'),
            (F('399024789'), '.2f', '399024789.00'),
            # Large precision (more than float can provide)
            (F(104348, 33215), '.50f',
             '3.14159265392142104470871594159265392142104470871594'),
            # Precision defaults to 6 if not given
            (F(22, 7), 'f', '3.142857'),
            (F(0), 'f', '0.000000'),
            (F(-22, 7), 'f', '-3.142857'),
            # Round-ties-to-even checks
            (F('1.225'), '.2f', '1.22'),
            (F('1.2250000001'), '.2f', '1.23'),
            (F('1.2349999999'), '.2f', '1.23'),
            (F('1.235'), '.2f', '1.24'),
            (F('1.245'), '.2f', '1.24'),
            (F('1.2450000001'), '.2f', '1.25'),
            (F('1.2549999999'), '.2f', '1.25'),
            (F('1.255'), '.2f', '1.26'),
            (F('-1.225'), '.2f', '-1.22'),
            (F('-1.2250000001'), '.2f', '-1.23'),
            (F('-1.2349999999'), '.2f', '-1.23'),
            (F('-1.235'), '.2f', '-1.24'),
            (F('-1.245'), '.2f', '-1.24'),
            (F('-1.2450000001'), '.2f', '-1.25'),
            (F('-1.2549999999'), '.2f', '-1.25'),
            (F('-1.255'), '.2f', '-1.26'),
            # Negatives and sign handling
            (F(2, 3), '.2f', '0.67'),
            (F(2, 3), '-.2f', '0.67'),
            (F(2, 3), '+.2f', '+0.67'),
            (F(2, 3), ' .2f', ' 0.67'),
            (F(-2, 3), '.2f', '-0.67'),
            (F(-2, 3), '-.2f', '-0.67'),
            (F(-2, 3), '+.2f', '-0.67'),
            (F(-2, 3), ' .2f', '-0.67'),
            # Formatting to zero places
            (F(1, 2), '.0f', '0'),
            (F(-1, 2), '.0f', '-0'),
            (F(22, 7), '.0f', '3'),
            (F(-22, 7), '.0f', '-3'),
            # Formatting to zero places, alternate form
            (F(1, 2), '#.0f', '0.'),
            (F(-1, 2), '#.0f', '-0.'),
            (F(22, 7), '#.0f', '3.'),
            (F(-22, 7), '#.0f', '-3.'),
            # z flag for suppressing negative zeros
            (F('-0.001'), 'z.2f', '0.00'),
            (F('-0.001'), '-z.2f', '0.00'),
            (F('-0.001'), '+z.2f', '+0.00'),
            (F('-0.001'), ' z.2f', ' 0.00'),
            (F('0.001'), 'z.2f', '0.00'),
            (F('0.001'), '-z.2f', '0.00'),
            (F('0.001'), '+z.2f', '+0.00'),
            (F('0.001'), ' z.2f', ' 0.00'),
            # Specifying a minimum width
            (F(2, 3), '6.2f', '  0.67'),
            (F(12345), '6.2f', '12345.00'),
            (F(12345), '12f', '12345.000000'),
            # Fill and alignment
            (F(2, 3), '>6.2f', '  0.67'),
            (F(2, 3), '<6.2f', '0.67  '),
            (F(2, 3), '^3.2f', '0.67'),
            (F(2, 3), '^4.2f', '0.67'),
            (F(2, 3), '^5.2f', '0.67 '),
            (F(2, 3), '^6.2f', ' 0.67 '),
            (F(2, 3), '^7.2f', ' 0.67  '),
            (F(2, 3), '^8.2f', '  0.67  '),
            # '=' alignment
            (F(-2, 3), '=+8.2f', '-   0.67'),
            (F(2, 3), '=+8.2f', '+   0.67'),
            # Fill character
            (F(-2, 3), 'X>3.2f', '-0.67'),
            (F(-2, 3), 'X>7.2f', 'XX-0.67'),
            (F(-2, 3), 'X<7.2f', '-0.67XX'),
            (F(-2, 3), 'X^7.2f', 'X-0.67X'),
            (F(-2, 3), 'X=7.2f', '-XX0.67'),
            (F(-2, 3), ' >7.2f', '  -0.67'),
            # Corner cases: weird fill characters
            (F(-2, 3), '\x00>7.2f', '\x00\x00-0.67'),
            (F(-2, 3), '\n>7.2f', '\n\n-0.67'),
            (F(-2, 3), '\t>7.2f', '\t\t-0.67'),
            (F(-2, 3), '>>7.2f', '>>-0.67'),
            (F(-2, 3), '<>7.2f', '<<-0.67'),
            (F(-2, 3), '→>7.2f', '→→-0.67'),
            # Zero-padding
            (F(-2, 3), '07.2f', '-000.67'),
            (F(-2, 3), '-07.2f', '-000.67'),
            (F(2, 3), '+07.2f', '+000.67'),
            (F(2, 3), ' 07.2f', ' 000.67'),
            # An isolated zero is a minimum width, not a zero-pad flag.
            # So unlike zero-padding, it's legal in combination with alignment.
            (F(2, 3), '0.2f', '0.67'),
            (F(2, 3), '>0.2f', '0.67'),
            (F(2, 3), '<0.2f', '0.67'),
            (F(2, 3), '^0.2f', '0.67'),
            (F(2, 3), '=0.2f', '0.67'),
            # Corner case: zero-padding _and_ a zero minimum width.
            (F(2, 3), '00.2f', '0.67'),
            # Thousands separator (only affects portion before the point)
            (F(2, 3), ',.2f', '0.67'),
            (F(2, 3), ',.7f', '0.6666667'),
            (F('123456.789'), ',.2f', '123,456.79'),
            (F('1234567'), ',.2f', '1,234,567.00'),
            (F('12345678'), ',.2f', '12,345,678.00'),
            (F('12345678'), ',f', '12,345,678.000000'),
            # Underscore as thousands separator
            (F(2, 3), '_.2f', '0.67'),
            (F(2, 3), '_.7f', '0.6666667'),
            (F('123456.789'), '_.2f', '123_456.79'),
            (F('1234567'), '_.2f', '1_234_567.00'),
            (F('12345678'), '_.2f', '12_345_678.00'),
            # Thousands and zero-padding
            (F('1234.5678'), '07,.2f', '1,234.57'),
            (F('1234.5678'), '08,.2f', '1,234.57'),
            (F('1234.5678'), '09,.2f', '01,234.57'),
            (F('1234.5678'), '010,.2f', '001,234.57'),
            (F('1234.5678'), '011,.2f', '0,001,234.57'),
            (F('1234.5678'), '012,.2f', '0,001,234.57'),
            (F('1234.5678'), '013,.2f', '00,001,234.57'),
            (F('1234.5678'), '014,.2f', '000,001,234.57'),
            (F('1234.5678'), '015,.2f', '0,000,001,234.57'),
            (F('1234.5678'), '016,.2f', '0,000,001,234.57'),
            (F('-1234.5678'), '07,.2f', '-1,234.57'),
            (F('-1234.5678'), '08,.2f', '-1,234.57'),
            (F('-1234.5678'), '09,.2f', '-1,234.57'),
            (F('-1234.5678'), '010,.2f', '-01,234.57'),
            (F('-1234.5678'), '011,.2f', '-001,234.57'),
            (F('-1234.5678'), '012,.2f', '-0,001,234.57'),
            (F('-1234.5678'), '013,.2f', '-0,001,234.57'),
            (F('-1234.5678'), '014,.2f', '-00,001,234.57'),
            (F('-1234.5678'), '015,.2f', '-000,001,234.57'),
            (F('-1234.5678'), '016,.2f', '-0,000,001,234.57'),
            # Corner case: no decimal point
            (F('-1234.5678'), '06,.0f', '-1,235'),
            (F('-1234.5678'), '07,.0f', '-01,235'),
            (F('-1234.5678'), '08,.0f', '-001,235'),
            (F('-1234.5678'), '09,.0f', '-0,001,235'),
            # Corner-case - zero-padding specified through fill and align
            # instead of the zero-pad character - in this case, treat '0' as a
            # regular fill character and don't attempt to insert commas into
            # the filled portion. This differs from the int and float
            # behaviour.
            (F('1234.5678'), '0=12,.2f', '00001,234.57'),
            # Corner case where it's not clear whether the '0' indicates zero
            # padding or gives the minimum width, but there's still an obvious
            # answer to give. We want this to work in case the minimum width
            # is being inserted programmatically: spec = f'{width}.2f'.
            (F('12.34'), '0.2f', '12.34'),
            (F('12.34'), 'X>0.2f', '12.34'),
            # 'F' should work identically to 'f'
            (F(22, 7), '.5F', '3.14286'),
            # %-specifier
            (F(22, 7), '.2%', '314.29%'),
            (F(1, 7), '.2%', '14.29%'),
            (F(1, 70), '.2%', '1.43%'),
            (F(1, 700), '.2%', '0.14%'),
            (F(1, 7000), '.2%', '0.01%'),
            (F(1, 70000), '.2%', '0.00%'),
            (F(1, 7), '.0%', '14%'),
            (F(1, 7), '#.0%', '14.%'),
            (F(100, 7), ',.2%', '1,428.57%'),
            (F(22, 7), '7.2%', '314.29%'),
            (F(22, 7), '8.2%', ' 314.29%'),
            (F(22, 7), '08.2%', '0314.29%'),
            # Test cases from #67790 and discuss.python.org Ideas thread.
            (F(1, 3), '.2f', '0.33'),
            (F(1, 8), '.2f', '0.12'),
            (F(3, 8), '.2f', '0.38'),
            (F(2545, 1000), '.2f', '2.54'),
            (F(2549, 1000), '.2f', '2.55'),
            (F(2635, 1000), '.2f', '2.64'),
            (F(1, 100), '.1f', '0.0'),
            (F(49, 1000), '.1f', '0.0'),
            (F(51, 1000), '.1f', '0.1'),
            (F(149, 1000), '.1f', '0.1'),
            (F(151, 1000), '.1f', '0.2'),
        ]
        for fraction, spec, expected in testcases:
            with self.subTest(fraction=fraction, spec=spec):
                self.assertEqual(format(fraction, spec), expected)

    def test_format_g_presentation_type(self):
        # Triples (fraction, specification, expected_result)
        testcases = [
            (F('0.000012345678'), '.6g', '1.23457e-05'),
            (F('0.00012345678'), '.6g', '0.000123457'),
            (F('0.0012345678'), '.6g', '0.00123457'),
            (F('0.012345678'), '.6g', '0.0123457'),
            (F('0.12345678'), '.6g', '0.123457'),
            (F('1.2345678'), '.6g', '1.23457'),
            (F('12.345678'), '.6g', '12.3457'),
            (F('123.45678'), '.6g', '123.457'),
            (F('1234.5678'), '.6g', '1234.57'),
            (F('12345.678'), '.6g', '12345.7'),
            (F('123456.78'), '.6g', '123457'),
            (F('1234567.8'), '.6g', '1.23457e+06'),
            # Rounding up cases
            (F('9.99999e+2'), '.4g', '1000'),
            (F('9.99999e-8'), '.4g', '1e-07'),
            (F('9.99999e+8'), '.4g', '1e+09'),
            # Check round-ties-to-even behaviour
            (F('-0.115'), '.2g', '-0.12'),
            (F('-0.125'), '.2g', '-0.12'),
            (F('-0.135'), '.2g', '-0.14'),
            (F('-0.145'), '.2g', '-0.14'),
            (F('0.115'), '.2g', '0.12'),
            (F('0.125'), '.2g', '0.12'),
            (F('0.135'), '.2g', '0.14'),
            (F('0.145'), '.2g', '0.14'),
            # Trailing zeros and decimal point suppressed by default ...
            (F(0), '.6g', '0'),
            (F('123.400'), '.6g', '123.4'),
            (F('123.000'), '.6g', '123'),
            (F('120.000'), '.6g', '120'),
            (F('12000000'), '.6g', '1.2e+07'),
            # ... but not when alternate form is in effect
            (F(0), '#.6g', '0.00000'),
            (F('123.400'), '#.6g', '123.400'),
            (F('123.000'), '#.6g', '123.000'),
            (F('120.000'), '#.6g', '120.000'),
            (F('12000000'), '#.6g', '1.20000e+07'),
            # 'G' format (uses 'E' instead of 'e' for the exponent indicator)
            (F('123.45678'), '.6G', '123.457'),
            (F('1234567.8'), '.6G', '1.23457E+06'),
            # Default precision is 6 significant figures
            (F('3.1415926535'), 'g', '3.14159'),
            # Precision 0 is treated the same as precision 1.
            (F('0.000031415'), '.0g', '3e-05'),
            (F('0.00031415'), '.0g', '0.0003'),
            (F('0.31415'), '.0g', '0.3'),
            (F('3.1415'), '.0g', '3'),
            (F('3.1415'), '#.0g', '3.'),
            (F('31.415'), '.0g', '3e+01'),
            (F('31.415'), '#.0g', '3.e+01'),
            (F('0.000031415'), '.1g', '3e-05'),
            (F('0.00031415'), '.1g', '0.0003'),
            (F('0.31415'), '.1g', '0.3'),
            (F('3.1415'), '.1g', '3'),
            (F('3.1415'), '#.1g', '3.'),
            (F('31.415'), '.1g', '3e+01'),
            # Thousands separator
            (F(2**64), '_.25g', '18_446_744_073_709_551_616'),
            # As with 'e' format, z flag is legal, but has no effect
            (F(-1, 7**100), 'zg', '-3.09169e-85'),
        ]
        for fraction, spec, expected in testcases:
            with self.subTest(fraction=fraction, spec=spec):
                self.assertEqual(format(fraction, spec), expected)

    def test_invalid_formats(self):
        fraction = F(2, 3)
        with self.assertRaises(TypeError):
            format(fraction, None)

        invalid_specs = [
            'Q6f',  # regression test
            # illegal to use fill or alignment when zero padding
            'X>010f',
            'X<010f',
            'X^010f',
            'X=010f',
            '0>010f',
            '0<010f',
            '0^010f',
            '0=010f',
            '>010f',
            '<010f',
            '^010f',
            '=010e',
            '=010f',
            '=010g',
            '=010%',
            '>00.2f',
            '>00f',
            # Too many zeros - minimum width should not have leading zeros
            '006f',
            # Leading zeros in precision
            '.010f',
            '.02f',
            '.000f',
            # Missing precision
            '.e',
            '.f',
            '.g',
            '.%',
            # Z instead of z for negative zero suppression
            'Z.2f'
            # z flag not supported for general formatting
            'z',
            # zero padding not supported for general formatting
            '05',
        ]
        for spec in invalid_specs:
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    format(fraction, spec)

    @requires_IEEE_754
    def test_float_format_testfile(self):
        with open(format_testfile, encoding="utf-8") as testfile:
            for line in testfile:
                if line.startswith('--'):
                    continue
                line = line.strip()
                if not line:
                    continue

                lhs, rhs = map(str.strip, line.split('->'))
                fmt, arg = lhs.split()
                if fmt == '%r':
                    continue
                fmt2 = fmt[1:]
                with self.subTest(fmt=fmt, arg=arg):
                    f = F(float(arg))
                    self.assertEqual(format(f, fmt2), rhs)
                    if f:  # skip negative zero
                        self.assertEqual(format(-f, fmt2), '-' + rhs)
                    f = F(arg)
                    self.assertEqual(float(format(f, fmt2)), float(rhs))
                    self.assertEqual(float(format(-f, fmt2)), float('-' + rhs))

    def test_complex_handling(self):
        # See issue gh-102840 for more details.

        a = F(1, 2)
        b = 1j
        message = "unsupported operand type(s) for %s: '%s' and '%s'"
        # test forward
        self.assertRaisesMessage(TypeError,
                                 message % ("%", "Fraction", "complex"),
                                 operator.mod, a, b)
        self.assertRaisesMessage(TypeError,
                                 message % ("//", "Fraction", "complex"),
                                 operator.floordiv, a, b)
        self.assertRaisesMessage(TypeError,
                                 message % ("divmod()", "Fraction", "complex"),
                                 divmod, a, b)
        # test reverse
        self.assertRaisesMessage(TypeError,
                                 message % ("%", "complex", "Fraction"),
                                 operator.mod, b, a)
        self.assertRaisesMessage(TypeError,
                                 message % ("//", "complex", "Fraction"),
                                 operator.floordiv, b, a)
        self.assertRaisesMessage(TypeError,
                                 message % ("divmod()", "complex", "Fraction"),
                                 divmod, b, a)


if __name__ == "__main__":
    run_tests()
