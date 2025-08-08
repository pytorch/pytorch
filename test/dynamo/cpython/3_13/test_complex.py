# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_complex.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    xfailIfTorchDynamo,
)

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

import unittest
import sys
from test import support
from test.support.testcase import ComplexesAreIdenticalMixin
from random import random
from math import isnan, copysign
import math
import operator

VALID_UNDERSCORE_LITERALS = [
    '0_0_0',
    '4_2',
    '1_0000_0000',
    '0b1001_0100',
    '0xffff_ffff',
    '0o5_7_7',
    '1_00_00.5',
    '1_00_00.5e5',
    '1_00_00e5_1',
    '1e1_0',
    '.1_4',
    '.1_4e1',
    '0b_0',
    '0x_f',
    '0o_5',
    '1_00_00j',
    '1_00_00.5j',
    '1_00_00e5_1j',
    '.1_4j',
    '(1_2.5+3_3j)',
    '(.5_6j)',
]
INVALID_UNDERSCORE_LITERALS = [
    # Trailing underscores:
    '0_',
    '42_',
    '1.4j_',
    '0x_',
    '0b1_',
    '0xf_',
    '0o5_',
    '0 if 1_Else 1',
    # Underscores in the base selector:
    '0_b0',
    '0_xf',
    '0_o5',
    # Old-style octal, still disallowed:
    '0_7',
    '09_99',
    # Multiple consecutive underscores:
    '4_______2',
    '0.1__4',
    '0.1__4j',
    '0b1001__0100',
    '0xffff__ffff',
    '0x___',
    '0o5__77',
    '1e1__0',
    '1e1__0j',
    # Underscore right before a dot:
    '1_.4',
    '1_.4j',
    # Underscore right after a dot:
    '1._4',
    '1._4j',
    '._5',
    '._5j',
    # Underscore right after a sign:
    '1.0e+_1',
    '1.0e+_1j',
    # Underscore right before j:
    '1.4_j',
    '1.4e5_j',
    # Underscore right before e:
    '1_e1',
    '1.4_e1',
    '1.4_e1j',
    # Underscore right after e:
    '1e_1',
    '1.4e_1',
    '1.4e_1j',
    # Complex cases with parens:
    '(1+1.5_j_)',
    '(1+1.5_j)',
]

INF = float("inf")
NAN = float("nan")
DBL_MAX = sys.float_info.max
# These tests ensure that complex math does the right thing

ZERO_DIVISION = (
    (1+1j, 0+0j),
    (1+1j, 0.0),
    (1+1j, 0),
    (1.0, 0+0j),
    (1, 0+0j),
)

class WithIndex:
    def __init__(self, value):
        self.value = value
    def __index__(self):
        return self.value

class WithFloat:
    def __init__(self, value):
        self.value = value
    def __float__(self):
        return self.value

class ComplexSubclass(complex):
    pass

class WithComplex:
    def __init__(self, value):
        self.value = value
    def __complex__(self):
        return self.value

class ComplexTest(__TestCase):

    def assertFloatIdentical(self, x, y):
        """Fail unless floats x and y are identical, in the sense that:
        (1) both x and y are nans, or
        (2) both x and y are infinities, with the same sign, or
        (3) both x and y are zeros, with the same sign, or
        (4) x and y are both finite and nonzero, and x == y

        """
        msg = 'floats {!r} and {!r} are not identical'

        if math.isnan(x) or math.isnan(y):
            if math.isnan(x) and math.isnan(y):
                return
        elif x == y:
            if x != 0.0:
                return
            # both zero; check that signs match
            elif math.copysign(1.0, x) == math.copysign(1.0, y):
                return
            else:
                msg += ': zeros have different signs'
        self.fail(msg.format(x, y))

    def assertComplexesAreIdentical(self, x, y):
        """Fail unless complex numbers x and y have equal values and signs.

        In particular, if x and y both have real (or imaginary) part
        zero, but the zeros have different signs, this test will fail.

        """
        self.assertFloatIdentical(x.real, y.real)
        self.assertFloatIdentical(x.imag, y.imag)

    def assertAlmostEqual(self, a, b):
        if isinstance(a, complex):
            if isinstance(b, complex):
                unittest.TestCase.assertAlmostEqual(self, a.real, b.real)
                unittest.TestCase.assertAlmostEqual(self, a.imag, b.imag)
            else:
                unittest.TestCase.assertAlmostEqual(self, a.real, b)
                unittest.TestCase.assertAlmostEqual(self, a.imag, 0.)
        else:
            if isinstance(b, complex):
                unittest.TestCase.assertAlmostEqual(self, a, b.real)
                unittest.TestCase.assertAlmostEqual(self, 0., b.imag)
            else:
                unittest.TestCase.assertAlmostEqual(self, a, b)

    def assertCloseAbs(self, x, y, eps=1e-9):
        """Return true iff floats x and y "are close"."""
        # put the one with larger magnitude second
        if abs(x) > abs(y):
            x, y = y, x
        if y == 0:
            return abs(x) < eps
        if x == 0:
            return abs(y) < eps
        # check that relative difference < eps
        self.assertTrue(abs((x-y)/y) < eps)

    def assertFloatsAreIdentical(self, x, y):
        """assert that floats x and y are identical, in the sense that:
        (1) both x and y are nans, or
        (2) both x and y are infinities, with the same sign, or
        (3) both x and y are zeros, with the same sign, or
        (4) x and y are both finite and nonzero, and x == y

        """
        msg = 'floats {!r} and {!r} are not identical'

        if isnan(x) or isnan(y):
            if isnan(x) and isnan(y):
                return
        elif x == y:
            if x != 0.0:
                return
            # both zero; check that signs match
            elif copysign(1.0, x) == copysign(1.0, y):
                return
            else:
                msg += ': zeros have different signs'
        self.fail(msg.format(x, y))

    def assertClose(self, x, y, eps=1e-9):
        """Return true iff complexes x and y "are close"."""
        self.assertCloseAbs(x.real, y.real, eps)
        self.assertCloseAbs(x.imag, y.imag, eps)

    def check_div(self, x, y):
        """Compute complex z=x*y, and check that z/x==y and z/y==x."""
        z = x * y
        if x != 0:
            q = z / x
            self.assertClose(q, y)
            q = z.__truediv__(x)
            self.assertClose(q, y)
        if y != 0:
            q = z / y
            self.assertClose(q, x)
            q = z.__truediv__(y)
            self.assertClose(q, x)

    def test_truediv(self):
        simple_real = [float(i) for i in range(-5, 6)]
        simple_complex = [complex(x, y) for x in simple_real for y in simple_real]
        for x in simple_complex:
            for y in simple_complex:
                self.check_div(x, y)

        # A naive complex division algorithm (such as in 2.0) is very prone to
        # nonsense errors for these (overflows and underflows).
        self.check_div(complex(1e200, 1e200), 1+0j)
        self.check_div(complex(1e-200, 1e-200), 1+0j)

        # Just for fun.
        for i in range(100):
            self.check_div(complex(random(), random()),
                           complex(random(), random()))

        self.assertAlmostEqual(complex.__truediv__(2+0j, 1+1j), 1-1j)
        self.assertRaises(TypeError, operator.truediv, 1j, None)
        self.assertRaises(TypeError, operator.truediv, None, 1j)

        for denom_real, denom_imag in [(0, NAN), (NAN, 0), (NAN, NAN)]:
            z = complex(0, 0) / complex(denom_real, denom_imag)
            self.assertTrue(isnan(z.real))
            self.assertTrue(isnan(z.imag))

    def test_truediv_zero_division(self):
        for a, b in ZERO_DIVISION:
            with self.assertRaises(ZeroDivisionError):
                a / b

    def test_floordiv(self):
        with self.assertRaises(TypeError):
            (1+1j) // (1+0j)
        with self.assertRaises(TypeError):
            (1+1j) // 1.0
        with self.assertRaises(TypeError):
            (1+1j) // 1
        with self.assertRaises(TypeError):
            1.0 // (1+0j)
        with self.assertRaises(TypeError):
            1 // (1+0j)

    def test_floordiv_zero_division(self):
        for a, b in ZERO_DIVISION:
            with self.assertRaises(TypeError):
                a // b

    def test_richcompare(self):
        self.assertIs(complex.__eq__(1+1j, 1<<10000), False)
        self.assertIs(complex.__lt__(1+1j, None), NotImplemented)
        self.assertIs(complex.__eq__(1+1j, None), NotImplemented)
        self.assertIs(complex.__eq__(1+1j, 1+1j), True)
        self.assertIs(complex.__eq__(1+1j, 2+2j), False)
        self.assertIs(complex.__ne__(1+1j, 1+1j), False)
        self.assertIs(complex.__ne__(1+1j, 2+2j), True)
        for i in range(1, 100):
            f = i / 100.0
            self.assertIs(complex.__eq__(f+0j, f), True)
            self.assertIs(complex.__ne__(f+0j, f), False)
            self.assertIs(complex.__eq__(complex(f, f), f), False)
            self.assertIs(complex.__ne__(complex(f, f), f), True)
        self.assertIs(complex.__lt__(1+1j, 2+2j), NotImplemented)
        self.assertIs(complex.__le__(1+1j, 2+2j), NotImplemented)
        self.assertIs(complex.__gt__(1+1j, 2+2j), NotImplemented)
        self.assertIs(complex.__ge__(1+1j, 2+2j), NotImplemented)
        self.assertRaises(TypeError, operator.lt, 1+1j, 2+2j)
        self.assertRaises(TypeError, operator.le, 1+1j, 2+2j)
        self.assertRaises(TypeError, operator.gt, 1+1j, 2+2j)
        self.assertRaises(TypeError, operator.ge, 1+1j, 2+2j)
        self.assertIs(operator.eq(1+1j, 1+1j), True)
        self.assertIs(operator.eq(1+1j, 2+2j), False)
        self.assertIs(operator.ne(1+1j, 1+1j), False)
        self.assertIs(operator.ne(1+1j, 2+2j), True)
        self.assertIs(operator.eq(1+1j, 2.0), False)

    def test_richcompare_boundaries(self):
        def check(n, deltas, is_equal, imag = 0.0):
            for delta in deltas:
                i = n + delta
                z = complex(i, imag)
                self.assertIs(complex.__eq__(z, i), is_equal(delta))
                self.assertIs(complex.__ne__(z, i), not is_equal(delta))
        # For IEEE-754 doubles the following should hold:
        #    x in [2 ** (52 + i), 2 ** (53 + i + 1)] -> x mod 2 ** i == 0
        # where the interval is representable, of course.
        for i in range(1, 10):
            pow = 52 + i
            mult = 2 ** i
            check(2 ** pow, range(1, 101), lambda delta: delta % mult == 0)
            check(2 ** pow, range(1, 101), lambda delta: False, float(i))
        check(2 ** 53, range(-100, 0), lambda delta: True)

    def test_add(self):
        self.assertEqual(1j + int(+1), complex(+1, 1))
        self.assertEqual(1j + int(-1), complex(-1, 1))
        self.assertRaises(OverflowError, operator.add, 1j, 10**1000)
        self.assertRaises(TypeError, operator.add, 1j, None)
        self.assertRaises(TypeError, operator.add, None, 1j)

    def test_sub(self):
        self.assertEqual(1j - int(+1), complex(-1, 1))
        self.assertEqual(1j - int(-1), complex(1, 1))
        self.assertRaises(OverflowError, operator.sub, 1j, 10**1000)
        self.assertRaises(TypeError, operator.sub, 1j, None)
        self.assertRaises(TypeError, operator.sub, None, 1j)

    def test_mul(self):
        self.assertEqual(1j * int(20), complex(0, 20))
        self.assertEqual(1j * int(-1), complex(0, -1))
        self.assertRaises(OverflowError, operator.mul, 1j, 10**1000)
        self.assertRaises(TypeError, operator.mul, 1j, None)
        self.assertRaises(TypeError, operator.mul, None, 1j)

    def test_mod(self):
        # % is no longer supported on complex numbers
        with self.assertRaises(TypeError):
            (1+1j) % (1+0j)
        with self.assertRaises(TypeError):
            (1+1j) % 1.0
        with self.assertRaises(TypeError):
            (1+1j) % 1
        with self.assertRaises(TypeError):
            1.0 % (1+0j)
        with self.assertRaises(TypeError):
            1 % (1+0j)

    def test_mod_zero_division(self):
        for a, b in ZERO_DIVISION:
            with self.assertRaises(TypeError):
                a % b

    def test_divmod(self):
        self.assertRaises(TypeError, divmod, 1+1j, 1+0j)
        self.assertRaises(TypeError, divmod, 1+1j, 1.0)
        self.assertRaises(TypeError, divmod, 1+1j, 1)
        self.assertRaises(TypeError, divmod, 1.0, 1+0j)
        self.assertRaises(TypeError, divmod, 1, 1+0j)

    def test_divmod_zero_division(self):
        for a, b in ZERO_DIVISION:
            self.assertRaises(TypeError, divmod, a, b)

    def test_pow(self):
        self.assertAlmostEqual(pow(1+1j, 0+0j), 1.0)
        self.assertAlmostEqual(pow(0+0j, 2+0j), 0.0)
        self.assertEqual(pow(0+0j, 2000+0j), 0.0)
        self.assertEqual(pow(0, 0+0j), 1.0)
        self.assertEqual(pow(-1, 0+0j), 1.0)
        self.assertRaises(ZeroDivisionError, pow, 0+0j, 1j)
        self.assertRaises(ZeroDivisionError, pow, 0+0j, -1000)
        self.assertAlmostEqual(pow(1j, -1), 1/1j)
        self.assertAlmostEqual(pow(1j, 200), 1)
        self.assertRaises(ValueError, pow, 1+1j, 1+1j, 1+1j)
        self.assertRaises(OverflowError, pow, 1e200+1j, 1e200+1j)
        self.assertRaises(TypeError, pow, 1j, None)
        self.assertRaises(TypeError, pow, None, 1j)
        self.assertAlmostEqual(pow(1j, 0.5), 0.7071067811865476+0.7071067811865475j)

        a = 3.33+4.43j
        self.assertEqual(a ** 0j, 1)
        self.assertEqual(a ** 0.+0.j, 1)

        self.assertEqual(3j ** 0j, 1)
        self.assertEqual(3j ** 0, 1)

        try:
            0j ** a
        except ZeroDivisionError:
            pass
        else:
            self.fail("should fail 0.0 to negative or complex power")

        try:
            0j ** (3-2j)
        except ZeroDivisionError:
            pass
        else:
            self.fail("should fail 0.0 to negative or complex power")

        # The following is used to exercise certain code paths
        self.assertEqual(a ** 105, a ** 105)
        self.assertEqual(a ** -105, a ** -105)
        self.assertEqual(a ** -30, a ** -30)

        self.assertEqual(0.0j ** 0, 1)

        b = 5.1+2.3j
        self.assertRaises(ValueError, pow, a, b, 0)

        # Check some boundary conditions; some of these used to invoke
        # undefined behaviour (https://bugs.python.org/issue44698). We're
        # not actually checking the results of these operations, just making
        # sure they don't crash (for example when using clang's
        # UndefinedBehaviourSanitizer).
        values = (sys.maxsize, sys.maxsize+1, sys.maxsize-1,
                  -sys.maxsize, -sys.maxsize+1, -sys.maxsize+1)
        for real in values:
            for imag in values:
                with self.subTest(real=real, imag=imag):
                    c = complex(real, imag)
                    try:
                        c ** real
                    except OverflowError:
                        pass
                    try:
                        c ** c
                    except OverflowError:
                        pass

        # gh-113841: possible undefined division by 0 in _Py_c_pow()
        x, y = 9j, 33j**3
        with self.assertRaises(OverflowError):
            x**y

    def test_pow_with_small_integer_exponents(self):
        # Check that small integer exponents are handled identically
        # regardless of their type.
        values = [
            complex(5.0, 12.0),
            complex(5.0e100, 12.0e100),
            complex(-4.0, INF),
            complex(INF, 0.0),
        ]
        exponents = [-19, -5, -3, -2, -1, 0, 1, 2, 3, 5, 19]
        for value in values:
            for exponent in exponents:
                with self.subTest(value=value, exponent=exponent):
                    try:
                        int_pow = value**exponent
                    except OverflowError:
                        int_pow = "overflow"
                    try:
                        float_pow = value**float(exponent)
                    except OverflowError:
                        float_pow = "overflow"
                    try:
                        complex_pow = value**complex(exponent)
                    except OverflowError:
                        complex_pow = "overflow"
                    self.assertEqual(str(float_pow), str(int_pow))
                    self.assertEqual(str(complex_pow), str(int_pow))

    def test_boolcontext(self):
        for i in range(100):
            self.assertTrue(complex(random() + 1e-6, random() + 1e-6))
        self.assertTrue(not complex(0.0, 0.0))
        self.assertTrue(1j)

    def test_conjugate(self):
        self.assertClose(complex(5.3, 9.8).conjugate(), 5.3-9.8j)

    def test_constructor(self):
        def check(z, x, y):
            self.assertIs(type(z), complex)
            self.assertFloatsAreIdentical(z.real, x)
            self.assertFloatsAreIdentical(z.imag, y)

        check(complex(),  0.0, 0.0)
        check(complex(10), 10.0, 0.0)
        check(complex(4.25), 4.25, 0.0)
        check(complex(4.25+0j), 4.25, 0.0)
        check(complex(4.25+0.5j), 4.25, 0.5)
        check(complex(ComplexSubclass(4.25+0.5j)), 4.25, 0.5)
        check(complex(WithComplex(4.25+0.5j)), 4.25, 0.5)

        check(complex(1, 10), 1.0, 10.0)
        check(complex(1, 10.0), 1.0, 10.0)
        check(complex(1, 4.25), 1.0, 4.25)
        check(complex(1.0, 10), 1.0, 10.0)
        check(complex(4.25, 10), 4.25, 10.0)
        check(complex(1.0, 10.0), 1.0, 10.0)
        check(complex(4.25, 0.5), 4.25, 0.5)

        check(complex(4.25+0j, 0), 4.25, 0.0)
        check(complex(ComplexSubclass(4.25+0j), 0), 4.25, 0.0)
        check(complex(WithComplex(4.25+0j), 0), 4.25, 0.0)
        check(complex(4.25j, 0), 0.0, 4.25)
        check(complex(0j, 4.25), 0.0, 4.25)
        check(complex(0, 4.25+0j), 0.0, 4.25)
        check(complex(0, ComplexSubclass(4.25+0j)), 0.0, 4.25)
        with self.assertRaisesRegex(TypeError,
                "second argument must be a number, not 'WithComplex'"):
            complex(0, WithComplex(4.25+0j))
        check(complex(0.0, 4.25j), -4.25, 0.0)
        check(complex(4.25+0j, 0j), 4.25, 0.0)
        check(complex(4.25j, 0j), 0.0, 4.25)
        check(complex(0j, 4.25+0j), 0.0, 4.25)
        check(complex(0j, 4.25j), -4.25, 0.0)

        check(complex(real=4.25), 4.25, 0.0)
        check(complex(real=4.25+0j), 4.25, 0.0)
        check(complex(real=4.25+1.5j), 4.25, 1.5)
        check(complex(imag=1.5), 0.0, 1.5)
        check(complex(real=4.25, imag=1.5), 4.25, 1.5)
        check(complex(4.25, imag=1.5), 4.25, 1.5)

        # check that the sign of a zero in the real or imaginary part
        # is preserved when constructing from two floats.
        for x in 1.0, -1.0:
            for y in 0.0, -0.0:
                check(complex(x, y), x, y)
                check(complex(y, x), y, x)

        c = complex(4.25, 1.5)
        self.assertIs(complex(c), c)
        c2 = ComplexSubclass(c)
        self.assertEqual(c2, c)
        self.assertIs(type(c2), ComplexSubclass)
        del c, c2

        self.assertRaisesRegex(TypeError,
            "first argument must be a string or a number, not 'dict'",
            complex, {})
        self.assertRaisesRegex(TypeError,
            "first argument must be a string or a number, not 'NoneType'",
            complex, None)
        self.assertRaisesRegex(TypeError,
            "first argument must be a string or a number, not 'dict'",
            complex, {1:2}, 0)
        self.assertRaisesRegex(TypeError,
            "can't take second arg if first is a string",
            complex, '1', 0)
        self.assertRaisesRegex(TypeError,
            "second argument must be a number, not 'dict'",
            complex, 0, {1:2})
        self.assertRaisesRegex(TypeError,
                "second arg can't be a string",
            complex, 0, '1')

        self.assertRaises(TypeError, complex, WithComplex(1.5))
        self.assertRaises(TypeError, complex, WithComplex(1))
        self.assertRaises(TypeError, complex, WithComplex(None))
        self.assertRaises(TypeError, complex, WithComplex(4.25+0j), object())
        self.assertRaises(TypeError, complex, WithComplex(1.5), object())
        self.assertRaises(TypeError, complex, WithComplex(1), object())
        self.assertRaises(TypeError, complex, WithComplex(None), object())

        class EvilExc(Exception):
            pass

        class evilcomplex:
            def __complex__(self):
                raise EvilExc

        self.assertRaises(EvilExc, complex, evilcomplex())

        check(complex(WithFloat(4.25)), 4.25, 0.0)
        check(complex(WithFloat(4.25), 1.5), 4.25, 1.5)
        check(complex(1.5, WithFloat(4.25)), 1.5, 4.25)
        self.assertRaises(TypeError, complex, WithFloat(42))
        self.assertRaises(TypeError, complex, WithFloat(42), 1.5)
        self.assertRaises(TypeError, complex, 1.5, WithFloat(42))
        self.assertRaises(TypeError, complex, WithFloat(None))
        self.assertRaises(TypeError, complex, WithFloat(None), 1.5)
        self.assertRaises(TypeError, complex, 1.5, WithFloat(None))

        check(complex(WithIndex(42)), 42.0, 0.0)
        check(complex(WithIndex(42), 1.5), 42.0, 1.5)
        check(complex(1.5, WithIndex(42)), 1.5, 42.0)
        self.assertRaises(OverflowError, complex, WithIndex(2**2000))
        self.assertRaises(OverflowError, complex, WithIndex(2**2000), 1.5)
        self.assertRaises(OverflowError, complex, 1.5, WithIndex(2**2000))
        self.assertRaises(TypeError, complex, WithIndex(None))
        self.assertRaises(TypeError, complex, WithIndex(None), 1.5)
        self.assertRaises(TypeError, complex, 1.5, WithIndex(None))

        class MyInt:
            def __int__(self):
                return 42

        self.assertRaises(TypeError, complex, MyInt())
        self.assertRaises(TypeError, complex, MyInt(), 1.5)
        self.assertRaises(TypeError, complex, 1.5, MyInt())

        class complex0(complex):
            """Test usage of __complex__() when inheriting from 'complex'"""
            def __complex__(self):
                return 42j

        class complex1(complex):
            """Test usage of __complex__() with a __new__() method"""
            def __new__(self, value=0j):
                return complex.__new__(self, 2*value)
            def __complex__(self):
                return self

        class complex2(complex):
            """Make sure that __complex__() calls fail if anything other than a
            complex is returned"""
            def __complex__(self):
                return None

        check(complex(complex0(1j)), 0.0, 42.0)
        with self.assertWarns(DeprecationWarning):
            check(complex(complex1(1j)), 0.0, 2.0)
        self.assertRaises(TypeError, complex, complex2(1j))

    def test___complex__(self):
        z = 3 + 4j
        self.assertEqual(z.__complex__(), z)
        self.assertEqual(type(z.__complex__()), complex)

        z = ComplexSubclass(3 + 4j)
        self.assertEqual(z.__complex__(), 3 + 4j)
        self.assertEqual(type(z.__complex__()), complex)

    @support.requires_IEEE_754
    def test_constructor_special_numbers(self):
        for x in 0.0, -0.0, INF, -INF, NAN:
            for y in 0.0, -0.0, INF, -INF, NAN:
                with self.subTest(x=x, y=y):
                    z = complex(x, y)
                    self.assertFloatsAreIdentical(z.real, x)
                    self.assertFloatsAreIdentical(z.imag, y)
                    z = ComplexSubclass(x, y)
                    self.assertIs(type(z), ComplexSubclass)
                    self.assertFloatsAreIdentical(z.real, x)
                    self.assertFloatsAreIdentical(z.imag, y)
                    z = complex(ComplexSubclass(x, y))
                    self.assertIs(type(z), complex)
                    self.assertFloatsAreIdentical(z.real, x)
                    self.assertFloatsAreIdentical(z.imag, y)
                    z = ComplexSubclass(complex(x, y))
                    self.assertIs(type(z), ComplexSubclass)
                    self.assertFloatsAreIdentical(z.real, x)
                    self.assertFloatsAreIdentical(z.imag, y)

    def test_constructor_from_string(self):
        def check(z, x, y):
            self.assertIs(type(z), complex)
            self.assertFloatsAreIdentical(z.real, x)
            self.assertFloatsAreIdentical(z.imag, y)

        check(complex("1"), 1.0, 0.0)
        check(complex("1j"), 0.0, 1.0)
        check(complex("-1"), -1.0, 0.0)
        check(complex("+1"), 1.0, 0.0)
        check(complex("1+2j"), 1.0, 2.0)
        check(complex("(1+2j)"), 1.0, 2.0)
        check(complex("(1.5+4.25j)"), 1.5, 4.25)
        check(complex("4.25+1J"), 4.25, 1.0)
        check(complex(" ( +4.25-6J )"), 4.25, -6.0)
        check(complex(" ( +4.25-J )"), 4.25, -1.0)
        check(complex(" ( +4.25+j )"), 4.25, 1.0)
        check(complex("J"), 0.0, 1.0)
        check(complex("( j )"), 0.0, 1.0)
        check(complex("+J"), 0.0, 1.0)
        check(complex("( -j)"), 0.0, -1.0)
        check(complex('1-1j'), 1.0, -1.0)
        check(complex('1J'), 0.0, 1.0)

        check(complex('1e-500'), 0.0, 0.0)
        check(complex('-1e-500j'), 0.0, -0.0)
        check(complex('1e-500+1e-500j'), 0.0, 0.0)
        check(complex('-1e-500+1e-500j'), -0.0, 0.0)
        check(complex('1e-500-1e-500j'), 0.0, -0.0)
        check(complex('-1e-500-1e-500j'), -0.0, -0.0)

        # SF bug 543840:  complex(string) accepts strings with \0
        # Fixed in 2.3.
        self.assertRaises(ValueError, complex, '1+1j\0j')
        self.assertRaises(ValueError, complex, "")
        self.assertRaises(ValueError, complex, "\0")
        self.assertRaises(ValueError, complex, "3\09")
        self.assertRaises(ValueError, complex, "1+")
        self.assertRaises(ValueError, complex, "1+1j+1j")
        self.assertRaises(ValueError, complex, "--")
        self.assertRaises(ValueError, complex, "(1+2j")
        self.assertRaises(ValueError, complex, "1+2j)")
        self.assertRaises(ValueError, complex, "1+(2j)")
        self.assertRaises(ValueError, complex, "(1+2j)123")
        self.assertRaises(ValueError, complex, "x")
        self.assertRaises(ValueError, complex, "1j+2")
        self.assertRaises(ValueError, complex, "1e1ej")
        self.assertRaises(ValueError, complex, "1e++1ej")
        self.assertRaises(ValueError, complex, ")1+2j(")
        # the following three are accepted by Python 2.6
        self.assertRaises(ValueError, complex, "1..1j")
        self.assertRaises(ValueError, complex, "1.11.1j")
        self.assertRaises(ValueError, complex, "1e1.1j")

        # check that complex accepts long unicode strings
        self.assertIs(type(complex("1"*500)), complex)
        # check whitespace processing
        self.assertEqual(complex('\N{EM SPACE}(\N{EN SPACE}1+1j ) '), 1+1j)
        # Invalid unicode string
        # See bpo-34087
        self.assertRaises(ValueError, complex, '\u3053\u3093\u306b\u3061\u306f')

    def test_constructor_negative_nans_from_string(self):
        self.assertEqual(copysign(1., complex("-nan").real), -1.)
        self.assertEqual(copysign(1., complex("-nanj").imag), -1.)
        self.assertEqual(copysign(1., complex("-nan-nanj").real), -1.)
        self.assertEqual(copysign(1., complex("-nan-nanj").imag), -1.)

    def test_underscores(self):
        # check underscores
        for lit in VALID_UNDERSCORE_LITERALS:
            if not any(ch in lit for ch in 'xXoObB'):
                self.assertEqual(complex(lit), eval(lit))
                self.assertEqual(complex(lit), complex(lit.replace('_', '')))
        for lit in INVALID_UNDERSCORE_LITERALS:
            if lit in ('0_7', '09_99'):  # octals are not recognized here
                continue
            if not any(ch in lit for ch in 'xXoObB'):
                self.assertRaises(ValueError, complex, lit)

    def test_hash(self):
        for x in range(-30, 30):
            self.assertEqual(hash(x), hash(complex(x, 0)))
            x /= 3.0    # now check against floating-point
            self.assertEqual(hash(x), hash(complex(x, 0.)))

        self.assertNotEqual(hash(2000005 - 1j), -1)

    def test_abs(self):
        nums = [complex(x/3., y/7.) for x in range(-9,9) for y in range(-9,9)]
        for num in nums:
            self.assertAlmostEqual((num.real**2 + num.imag**2)  ** 0.5, abs(num))

        self.assertRaises(OverflowError, abs, complex(DBL_MAX, DBL_MAX))

    def test_repr_str(self):
        def test(v, expected, test_fn=self.assertEqual):
            test_fn(repr(v), expected)
            test_fn(str(v), expected)

        test(1+6j, '(1+6j)')
        test(1-6j, '(1-6j)')

        test(-(1+0j), '(-1+-0j)', test_fn=self.assertNotEqual)

        test(complex(1., INF), "(1+infj)")
        test(complex(1., -INF), "(1-infj)")
        test(complex(INF, 1), "(inf+1j)")
        test(complex(-INF, INF), "(-inf+infj)")
        test(complex(NAN, 1), "(nan+1j)")
        test(complex(1, NAN), "(1+nanj)")
        test(complex(NAN, NAN), "(nan+nanj)")
        test(complex(-NAN, -NAN), "(nan+nanj)")

        test(complex(0, INF), "infj")
        test(complex(0, -INF), "-infj")
        test(complex(0, NAN), "nanj")

        self.assertEqual(1-6j,complex(repr(1-6j)))
        self.assertEqual(1+6j,complex(repr(1+6j)))
        self.assertEqual(-6j,complex(repr(-6j)))
        self.assertEqual(6j,complex(repr(6j)))

    @support.requires_IEEE_754
    def test_negative_zero_repr_str(self):
        def test(v, expected, test_fn=self.assertEqual):
            test_fn(repr(v), expected)
            test_fn(str(v), expected)

        test(complex(0., 1.),   "1j")
        test(complex(-0., 1.),  "(-0+1j)")
        test(complex(0., -1.),  "-1j")
        test(complex(-0., -1.), "(-0-1j)")

        test(complex(0., 0.),   "0j")
        test(complex(0., -0.),  "-0j")
        test(complex(-0., 0.),  "(-0+0j)")
        test(complex(-0., -0.), "(-0-0j)")

    def test_pos(self):
        self.assertEqual(+(1+6j), 1+6j)
        self.assertEqual(+ComplexSubclass(1, 6), 1+6j)
        self.assertIs(type(+ComplexSubclass(1, 6)), complex)

    def test_neg(self):
        self.assertEqual(-(1+6j), -1-6j)

    def test_getnewargs(self):
        self.assertEqual((1+2j).__getnewargs__(), (1.0, 2.0))
        self.assertEqual((1-2j).__getnewargs__(), (1.0, -2.0))
        self.assertEqual((2j).__getnewargs__(), (0.0, 2.0))
        self.assertEqual((-0j).__getnewargs__(), (0.0, -0.0))
        self.assertEqual(complex(0, INF).__getnewargs__(), (0.0, INF))
        self.assertEqual(complex(INF, 0).__getnewargs__(), (INF, 0.0))

    @support.requires_IEEE_754
    def test_plus_minus_0j(self):
        # test that -0j and 0j literals are not identified
        z1, z2 = 0j, -0j
        self.assertFloatsAreIdentical(z1.imag, 0.0)
        self.assertFloatsAreIdentical(z2.imag, -0.0)

    @support.requires_IEEE_754
    def test_negated_imaginary_literal(self):
        z0 = -0j
        z1 = -7j
        z2 = -1e1000j
        # Note: In versions of Python < 3.2, a negated imaginary literal
        # accidentally ended up with real part 0.0 instead of -0.0, thanks to a
        # modification during CST -> AST translation (see issue #9011).  That's
        # fixed in Python 3.2.
        self.assertFloatsAreIdentical(z0.real, -0.0)
        self.assertFloatsAreIdentical(z0.imag, -0.0)
        self.assertFloatsAreIdentical(z1.real, -0.0)
        self.assertFloatsAreIdentical(z1.imag, -7.0)
        self.assertFloatsAreIdentical(z2.real, -0.0)
        self.assertFloatsAreIdentical(z2.imag, -INF)

    @support.requires_IEEE_754
    def test_overflow(self):
        self.assertEqual(complex("1e500"), complex(INF, 0.0))
        self.assertEqual(complex("-1e500j"), complex(0.0, -INF))
        self.assertEqual(complex("-1e500+1.8e308j"), complex(-INF, INF))

    @support.requires_IEEE_754
    def test_repr_roundtrip(self):
        vals = [0.0, 1e-500, 1e-315, 1e-200, 0.0123, 3.1415, 1e50, INF, NAN]
        vals += [-v for v in vals]

        # complex(repr(z)) should recover z exactly, even for complex
        # numbers involving an infinity, nan, or negative zero
        for x in vals:
            for y in vals:
                z = complex(x, y)
                roundtrip = complex(repr(z))
                self.assertComplexesAreIdentical(z, roundtrip)

        # if we predefine some constants, then eval(repr(z)) should
        # also work, except that it might change the sign of zeros
        inf, nan = float('inf'), float('nan')
        infj, nanj = complex(0.0, inf), complex(0.0, nan)
        for x in vals:
            for y in vals:
                z = complex(x, y)
                roundtrip = eval(repr(z))
                # adding 0.0 has no effect beside changing -0.0 to 0.0
                self.assertFloatsAreIdentical(0.0 + z.real,
                                              0.0 + roundtrip.real)
                self.assertFloatsAreIdentical(0.0 + z.imag,
                                              0.0 + roundtrip.imag)

    def test_format(self):
        # empty format string is same as str()
        self.assertEqual(format(1+3j, ''), str(1+3j))
        self.assertEqual(format(1.5+3.5j, ''), str(1.5+3.5j))
        self.assertEqual(format(3j, ''), str(3j))
        self.assertEqual(format(3.2j, ''), str(3.2j))
        self.assertEqual(format(3+0j, ''), str(3+0j))
        self.assertEqual(format(3.2+0j, ''), str(3.2+0j))

        # empty presentation type should still be analogous to str,
        # even when format string is nonempty (issue #5920).
        self.assertEqual(format(3.2+0j, '-'), str(3.2+0j))
        self.assertEqual(format(3.2+0j, '<'), str(3.2+0j))
        z = 4/7. - 100j/7.
        self.assertEqual(format(z, ''), str(z))
        self.assertEqual(format(z, '-'), str(z))
        self.assertEqual(format(z, '<'), str(z))
        self.assertEqual(format(z, '10'), str(z))
        z = complex(0.0, 3.0)
        self.assertEqual(format(z, ''), str(z))
        self.assertEqual(format(z, '-'), str(z))
        self.assertEqual(format(z, '<'), str(z))
        self.assertEqual(format(z, '2'), str(z))
        z = complex(-0.0, 2.0)
        self.assertEqual(format(z, ''), str(z))
        self.assertEqual(format(z, '-'), str(z))
        self.assertEqual(format(z, '<'), str(z))
        self.assertEqual(format(z, '3'), str(z))

        self.assertEqual(format(1+3j, 'g'), '1+3j')
        self.assertEqual(format(3j, 'g'), '0+3j')
        self.assertEqual(format(1.5+3.5j, 'g'), '1.5+3.5j')

        self.assertEqual(format(1.5+3.5j, '+g'), '+1.5+3.5j')
        self.assertEqual(format(1.5-3.5j, '+g'), '+1.5-3.5j')
        self.assertEqual(format(1.5-3.5j, '-g'), '1.5-3.5j')
        self.assertEqual(format(1.5+3.5j, ' g'), ' 1.5+3.5j')
        self.assertEqual(format(1.5-3.5j, ' g'), ' 1.5-3.5j')
        self.assertEqual(format(-1.5+3.5j, ' g'), '-1.5+3.5j')
        self.assertEqual(format(-1.5-3.5j, ' g'), '-1.5-3.5j')

        self.assertEqual(format(-1.5-3.5e-20j, 'g'), '-1.5-3.5e-20j')
        self.assertEqual(format(-1.5-3.5j, 'f'), '-1.500000-3.500000j')
        self.assertEqual(format(-1.5-3.5j, 'F'), '-1.500000-3.500000j')
        self.assertEqual(format(-1.5-3.5j, 'e'), '-1.500000e+00-3.500000e+00j')
        self.assertEqual(format(-1.5-3.5j, '.2e'), '-1.50e+00-3.50e+00j')
        self.assertEqual(format(-1.5-3.5j, '.2E'), '-1.50E+00-3.50E+00j')
        self.assertEqual(format(-1.5e10-3.5e5j, '.2G'), '-1.5E+10-3.5E+05j')

        self.assertEqual(format(1.5+3j, '<20g'),  '1.5+3j              ')
        self.assertEqual(format(1.5+3j, '*<20g'), '1.5+3j**************')
        self.assertEqual(format(1.5+3j, '>20g'),  '              1.5+3j')
        self.assertEqual(format(1.5+3j, '^20g'),  '       1.5+3j       ')
        self.assertEqual(format(1.5+3j, '<20'),   '(1.5+3j)            ')
        self.assertEqual(format(1.5+3j, '>20'),   '            (1.5+3j)')
        self.assertEqual(format(1.5+3j, '^20'),   '      (1.5+3j)      ')
        self.assertEqual(format(1.123-3.123j, '^20.2'), '     (1.1-3.1j)     ')

        self.assertEqual(format(1.5+3j, '20.2f'), '          1.50+3.00j')
        self.assertEqual(format(1.5+3j, '>20.2f'), '          1.50+3.00j')
        self.assertEqual(format(1.5+3j, '<20.2f'), '1.50+3.00j          ')
        self.assertEqual(format(1.5e20+3j, '<20.2f'), '150000000000000000000.00+3.00j')
        self.assertEqual(format(1.5e20+3j, '>40.2f'), '          150000000000000000000.00+3.00j')
        self.assertEqual(format(1.5e20+3j, '^40,.2f'), '  150,000,000,000,000,000,000.00+3.00j  ')
        self.assertEqual(format(1.5e21+3j, '^40,.2f'), ' 1,500,000,000,000,000,000,000.00+3.00j ')
        self.assertEqual(format(1.5e21+3000j, ',.2f'), '1,500,000,000,000,000,000,000.00+3,000.00j')

        # Issue 7094: Alternate formatting (specified by #)
        self.assertEqual(format(1+1j, '.0e'), '1e+00+1e+00j')
        self.assertEqual(format(1+1j, '#.0e'), '1.e+00+1.e+00j')
        self.assertEqual(format(1+1j, '.0f'), '1+1j')
        self.assertEqual(format(1+1j, '#.0f'), '1.+1.j')
        self.assertEqual(format(1.1+1.1j, 'g'), '1.1+1.1j')
        self.assertEqual(format(1.1+1.1j, '#g'), '1.10000+1.10000j')

        # Alternate doesn't make a difference for these, they format the same with or without it
        self.assertEqual(format(1+1j, '.1e'),  '1.0e+00+1.0e+00j')
        self.assertEqual(format(1+1j, '#.1e'), '1.0e+00+1.0e+00j')
        self.assertEqual(format(1+1j, '.1f'),  '1.0+1.0j')
        self.assertEqual(format(1+1j, '#.1f'), '1.0+1.0j')

        # Misc. other alternate tests
        self.assertEqual(format((-1.5+0.5j), '#f'), '-1.500000+0.500000j')
        self.assertEqual(format((-1.5+0.5j), '#.0f'), '-2.+0.j')
        self.assertEqual(format((-1.5+0.5j), '#e'), '-1.500000e+00+5.000000e-01j')
        self.assertEqual(format((-1.5+0.5j), '#.0e'), '-2.e+00+5.e-01j')
        self.assertEqual(format((-1.5+0.5j), '#g'), '-1.50000+0.500000j')
        self.assertEqual(format((-1.5+0.5j), '.0g'), '-2+0.5j')
        self.assertEqual(format((-1.5+0.5j), '#.0g'), '-2.+0.5j')

        # zero padding is invalid
        self.assertRaises(ValueError, (1.5+0.5j).__format__, '010f')

        # '=' alignment is invalid
        self.assertRaises(ValueError, (1.5+3j).__format__, '=20')

        # integer presentation types are an error
        for t in 'bcdoxX':
            self.assertRaises(ValueError, (1.5+0.5j).__format__, t)

        # make sure everything works in ''.format()
        self.assertEqual('*{0:.3f}*'.format(3.14159+2.71828j), '*3.142+2.718j*')

        # issue 3382
        self.assertEqual(format(complex(NAN, NAN), 'f'), 'nan+nanj')
        self.assertEqual(format(complex(1, NAN), 'f'), '1.000000+nanj')
        self.assertEqual(format(complex(NAN, 1), 'f'), 'nan+1.000000j')
        self.assertEqual(format(complex(NAN, -1), 'f'), 'nan-1.000000j')
        self.assertEqual(format(complex(NAN, NAN), 'F'), 'NAN+NANj')
        self.assertEqual(format(complex(1, NAN), 'F'), '1.000000+NANj')
        self.assertEqual(format(complex(NAN, 1), 'F'), 'NAN+1.000000j')
        self.assertEqual(format(complex(NAN, -1), 'F'), 'NAN-1.000000j')
        self.assertEqual(format(complex(INF, INF), 'f'), 'inf+infj')
        self.assertEqual(format(complex(1, INF), 'f'), '1.000000+infj')
        self.assertEqual(format(complex(INF, 1), 'f'), 'inf+1.000000j')
        self.assertEqual(format(complex(INF, -1), 'f'), 'inf-1.000000j')
        self.assertEqual(format(complex(INF, INF), 'F'), 'INF+INFj')
        self.assertEqual(format(complex(1, INF), 'F'), '1.000000+INFj')
        self.assertEqual(format(complex(INF, 1), 'F'), 'INF+1.000000j')
        self.assertEqual(format(complex(INF, -1), 'F'), 'INF-1.000000j')


if __name__ == "__main__":
    run_tests()
