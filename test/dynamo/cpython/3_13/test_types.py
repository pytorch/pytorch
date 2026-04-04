# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_types.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# Dummy decorator for CPython test compatibility
def no_rerun(reason):
    def decorator(func):
        return func
    return decorator

# ======= END DYNAMO PATCH =======

# Python test set -- part 6, built-in types

from test.support import (
    run_with_locale, is_apple_mobile, cpython_only,
    iter_builtin_types, iter_slot_wrappers,
    MISSING_C_DOCSTRINGS,
)
from test.support.script_helper import assert_python_ok
from test.support.import_helper import import_fresh_module

import collections.abc
from collections import namedtuple, UserDict
import copy
import _datetime
import gc
import inspect
import pickle
import locale
import sys
import textwrap
import types
import unittest.mock
import weakref
import typing


T = typing.TypeVar("T")

class Example:
    pass

class Forward: ...

def clear_typing_caches():
    for f in typing._cleanups:
        f()


class TypesTests(CPythonTestCase):

    def test_truth_values(self):
        if None: self.fail('None is true instead of false')
        if 0: self.fail('0 is true instead of false')
        if 0.0: self.fail('0.0 is true instead of false')
        if '': self.fail('\'\' is true instead of false')
        if not 1: self.fail('1 is false instead of true')
        if not 1.0: self.fail('1.0 is false instead of true')
        if not 'x': self.fail('\'x\' is false instead of true')
        if not {'x': 1}: self.fail('{\'x\': 1} is false instead of true')
        def f(): pass
        with torch._dynamo.error_on_graph_break(False):
            class C: pass
            x = C()
        if not f: self.fail('f is false instead of true')
        if not C: self.fail('C is false instead of true')
        if not sys: self.fail('sys is false instead of true')
        if not x: self.fail('x is false instead of true')

    def test_boolean_ops(self):
        if 0 or 0: self.fail('0 or 0 is true instead of false')
        if 1 and 1: pass
        else: self.fail('1 and 1 is false instead of true')
        if not 1: self.fail('not 1 is true instead of false')

    def test_comparisons(self):
        if 0 < 1 <= 1 == 1 >= 1 > 0 != 1: pass
        else: self.fail('int comparisons failed')
        if 0.0 < 1.0 <= 1.0 == 1.0 >= 1.0 > 0.0 != 1.0: pass
        else: self.fail('float comparisons failed')
        if '' < 'a' <= 'a' == 'a' < 'abc' < 'abd' < 'b': pass
        else: self.fail('string comparisons failed')
        if None is None: pass
        else: self.fail('identity test failed')

    def test_float_constructor(self):
        self.assertRaises(ValueError, float, '')
        self.assertRaises(ValueError, float, '5\0')
        self.assertRaises(ValueError, float, '5_5\0')

    def test_zero_division(self):
        try: 5.0 / 0.0
        except ZeroDivisionError: pass
        else: self.fail("5.0 / 0.0 didn't raise ZeroDivisionError")

        try: 5.0 // 0.0
        except ZeroDivisionError: pass
        else: self.fail("5.0 // 0.0 didn't raise ZeroDivisionError")

        try: 5.0 % 0.0
        except ZeroDivisionError: pass
        else: self.fail("5.0 % 0.0 didn't raise ZeroDivisionError")

        try: 5 / 0
        except ZeroDivisionError: pass
        else: self.fail("5 / 0 didn't raise ZeroDivisionError")

        try: 5 // 0
        except ZeroDivisionError: pass
        else: self.fail("5 // 0 didn't raise ZeroDivisionError")

        try: 5 % 0
        except ZeroDivisionError: pass
        else: self.fail("5 % 0 didn't raise ZeroDivisionError")

    def test_numeric_types(self):
        if 0 != 0.0 or 1 != 1.0 or -1 != -1.0:
            self.fail('int/float value not equal')
        # calling built-in types without argument must return 0
        if int() != 0: self.fail('int() does not return 0')
        if float() != 0.0: self.fail('float() does not return 0.0')
        if int(1.9) == 1 == int(1.1) and int(-1.1) == -1 == int(-1.9): pass
        else: self.fail('int() does not round properly')
        if float(1) == 1.0 and float(-1) == -1.0 and float(0) == 0.0: pass
        else: self.fail('float() does not work properly')

    def test_float_to_string(self):
        def test(f, result):
            self.assertEqual(f.__format__('e'), result)
            self.assertEqual('%e' % f, result)

        # test all 2 digit exponents, both with __format__ and with
        #  '%' formatting
        for i in range(-99, 100):
            test(float('1.5e'+str(i)), '1.500000e{0:+03d}'.format(i))

        # test some 3 digit exponents
        self.assertEqual(1.5e100.__format__('e'), '1.500000e+100')
        self.assertEqual('%e' % 1.5e100, '1.500000e+100')

        self.assertEqual(1.5e101.__format__('e'), '1.500000e+101')
        self.assertEqual('%e' % 1.5e101, '1.500000e+101')

        self.assertEqual(1.5e-100.__format__('e'), '1.500000e-100')
        self.assertEqual('%e' % 1.5e-100, '1.500000e-100')

        self.assertEqual(1.5e-101.__format__('e'), '1.500000e-101')
        self.assertEqual('%e' % 1.5e-101, '1.500000e-101')

        self.assertEqual('%g' % 1.0, '1')
        self.assertEqual('%#g' % 1.0, '1.00000')

    def test_normal_integers(self):
        # Ensure the first 256 integers are shared
        a = 256
        b = 128*2
        if a is not b: self.fail('256 is not shared')
        if 12 + 24 != 36: self.fail('int op')
        if 12 + (-24) != -12: self.fail('int op')
        if (-12) + 24 != 12: self.fail('int op')
        if (-12) + (-24) != -36: self.fail('int op')
        if not 12 < 24: self.fail('int op')
        if not -24 < -12: self.fail('int op')
        # Test for a particular bug in integer multiply
        xsize, ysize, zsize = 238, 356, 4
        if not (xsize*ysize*zsize == zsize*xsize*ysize == 338912):
            self.fail('int mul commutativity')
        # And another.
        m = -sys.maxsize - 1
        for divisor in 1, 2, 4, 8, 16, 32:
            j = m // divisor
            prod = divisor * j
            if prod != m:
                self.fail("%r * %r == %r != %r" % (divisor, j, prod, m))
            if type(prod) is not int:
                self.fail("expected type(prod) to be int, not %r" %
                                   type(prod))
        # Check for unified integral type
        for divisor in 1, 2, 4, 8, 16, 32:
            j = m // divisor - 1
            prod = divisor * j
            if type(prod) is not int:
                self.fail("expected type(%r) to be int, not %r" %
                                   (prod, type(prod)))
        # Check for unified integral type
        m = sys.maxsize
        for divisor in 1, 2, 4, 8, 16, 32:
            j = m // divisor + 1
            prod = divisor * j
            if type(prod) is not int:
                self.fail("expected type(%r) to be int, not %r" %
                                   (prod, type(prod)))

        x = sys.maxsize
        self.assertIsInstance(x + 1, int,
                              "(sys.maxsize + 1) should have returned int")
        self.assertIsInstance(-x - 1, int,
                              "(-sys.maxsize - 1) should have returned int")
        self.assertIsInstance(-x - 2, int,
                              "(-sys.maxsize - 2) should have returned int")

        try: 5 << -5
        except ValueError: pass
        else: self.fail('int negative shift <<')

        try: 5 >> -5
        except ValueError: pass
        else: self.fail('int negative shift >>')

    def test_floats(self):
        if 12.0 + 24.0 != 36.0: self.fail('float op')
        if 12.0 + (-24.0) != -12.0: self.fail('float op')
        if (-12.0) + 24.0 != 12.0: self.fail('float op')
        if (-12.0) + (-24.0) != -36.0: self.fail('float op')
        if not 12.0 < 24.0: self.fail('float op')
        if not -24.0 < -12.0: self.fail('float op')

    def test_strings(self):
        if len('') != 0: self.fail('len(\'\')')
        if len('a') != 1: self.fail('len(\'a\')')
        if len('abcdef') != 6: self.fail('len(\'abcdef\')')
        if 'xyz' + 'abcde' != 'xyzabcde': self.fail('string concatenation')
        if 'xyz'*3 != 'xyzxyzxyz': self.fail('string repetition *3')
        if 0*'abcde' != '': self.fail('string repetition 0*')
        if min('abc') != 'a' or max('abc') != 'c': self.fail('min/max string')
        if 'a' in 'abc' and 'b' in 'abc' and 'c' in 'abc' and 'd' not in 'abc': pass
        else: self.fail('in/not in string')
        x = 'x'*103
        if '%s!'%x != x+'!': self.fail('nasty string formatting bug')

        #extended slices for strings
        a = '0123456789'
        self.assertEqual(a[::], a)
        self.assertEqual(a[::2], '02468')
        self.assertEqual(a[1::2], '13579')
        self.assertEqual(a[::-1],'9876543210')
        self.assertEqual(a[::-2], '97531')
        self.assertEqual(a[3::-2], '31')
        self.assertEqual(a[-100:100:], a)
        self.assertEqual(a[100:-100:-1], a[::-1])
        self.assertEqual(a[-100:100:2], '02468')

    def test_type_function(self):
        self.assertRaises(TypeError, type, 1, 2)
        self.assertRaises(TypeError, type, 1, 2, 3, 4)

    def test_int__format__(self):
        def test(i, format_spec, result):
            # just make sure we have the unified type for integers
            self.assertIs(type(i), int)
            self.assertIs(type(format_spec), str)
            self.assertEqual(i.__format__(format_spec), result)

        test(123456789, 'd', '123456789')
        test(123456789, 'd', '123456789')

        test(1, 'c', '\01')

        # sign and aligning are interdependent
        test(1, "-", '1')
        test(-1, "-", '-1')
        test(1, "-3", '  1')
        test(-1, "-3", ' -1')
        test(1, "+3", ' +1')
        test(-1, "+3", ' -1')
        test(1, " 3", '  1')
        test(-1, " 3", ' -1')
        test(1, " ", ' 1')
        test(-1, " ", '-1')

        # hex
        test(3, "x", "3")
        test(3, "X", "3")
        test(1234, "x", "4d2")
        test(-1234, "x", "-4d2")
        test(1234, "8x", "     4d2")
        test(-1234, "8x", "    -4d2")
        test(1234, "x", "4d2")
        test(-1234, "x", "-4d2")
        test(-3, "x", "-3")
        test(-3, "X", "-3")
        test(int('be', 16), "x", "be")
        test(int('be', 16), "X", "BE")
        test(-int('be', 16), "x", "-be")
        test(-int('be', 16), "X", "-BE")

        # octal
        test(3, "o", "3")
        test(-3, "o", "-3")
        test(65, "o", "101")
        test(-65, "o", "-101")
        test(1234, "o", "2322")
        test(-1234, "o", "-2322")
        test(1234, "-o", "2322")
        test(-1234, "-o", "-2322")
        test(1234, " o", " 2322")
        test(-1234, " o", "-2322")
        test(1234, "+o", "+2322")
        test(-1234, "+o", "-2322")

        # binary
        test(3, "b", "11")
        test(-3, "b", "-11")
        test(1234, "b", "10011010010")
        test(-1234, "b", "-10011010010")
        test(1234, "-b", "10011010010")
        test(-1234, "-b", "-10011010010")
        test(1234, " b", " 10011010010")
        test(-1234, " b", "-10011010010")
        test(1234, "+b", "+10011010010")
        test(-1234, "+b", "-10011010010")

        # alternate (#) formatting
        test(0, "#b", '0b0')
        test(0, "-#b", '0b0')
        test(1, "-#b", '0b1')
        test(-1, "-#b", '-0b1')
        test(-1, "-#5b", ' -0b1')
        test(1, "+#5b", ' +0b1')
        test(100, "+#b", '+0b1100100')
        test(100, "#012b", '0b0001100100')
        test(-100, "#012b", '-0b001100100')

        test(0, "#o", '0o0')
        test(0, "-#o", '0o0')
        test(1, "-#o", '0o1')
        test(-1, "-#o", '-0o1')
        test(-1, "-#5o", ' -0o1')
        test(1, "+#5o", ' +0o1')
        test(100, "+#o", '+0o144')
        test(100, "#012o", '0o0000000144')
        test(-100, "#012o", '-0o000000144')

        test(0, "#x", '0x0')
        test(0, "-#x", '0x0')
        test(1, "-#x", '0x1')
        test(-1, "-#x", '-0x1')
        test(-1, "-#5x", ' -0x1')
        test(1, "+#5x", ' +0x1')
        test(100, "+#x", '+0x64')
        test(100, "#012x", '0x0000000064')
        test(-100, "#012x", '-0x000000064')
        test(123456, "#012x", '0x000001e240')
        test(-123456, "#012x", '-0x00001e240')

        test(0, "#X", '0X0')
        test(0, "-#X", '0X0')
        test(1, "-#X", '0X1')
        test(-1, "-#X", '-0X1')
        test(-1, "-#5X", ' -0X1')
        test(1, "+#5X", ' +0X1')
        test(100, "+#X", '+0X64')
        test(100, "#012X", '0X0000000064')
        test(-100, "#012X", '-0X000000064')
        test(123456, "#012X", '0X000001E240')
        test(-123456, "#012X", '-0X00001E240')

        test(123, ',', '123')
        test(-123, ',', '-123')
        test(1234, ',', '1,234')
        test(-1234, ',', '-1,234')
        test(123456, ',', '123,456')
        test(-123456, ',', '-123,456')
        test(1234567, ',', '1,234,567')
        test(-1234567, ',', '-1,234,567')

        # issue 5782, commas with no specifier type
        test(1234, '010,', '00,001,234')

        # Unified type for integers
        test(10**100, 'd', '1' + '0' * 100)
        test(10**100+100, 'd', '1' + '0' * 97 + '100')

        # make sure these are errors

        # precision disallowed
        self.assertRaises(ValueError, 3 .__format__, "1.3")
        # sign not allowed with 'c'
        self.assertRaises(ValueError, 3 .__format__, "+c")
        # format spec must be string
        self.assertRaises(TypeError, 3 .__format__, None)
        self.assertRaises(TypeError, 3 .__format__, 0)
        # can't have ',' with 'n'
        self.assertRaises(ValueError, 3 .__format__, ",n")
        # can't have ',' with 'c'
        self.assertRaises(ValueError, 3 .__format__, ",c")
        # can't have '#' with 'c'
        self.assertRaises(ValueError, 3 .__format__, "#c")

        # ensure that only int and float type specifiers work
        for format_spec in ([chr(x) for x in range(ord('a'), ord('z')+1)] +
                            [chr(x) for x in range(ord('A'), ord('Z')+1)]):
            if not format_spec in 'bcdoxXeEfFgGn%':
                self.assertRaises(ValueError, 0 .__format__, format_spec)
                self.assertRaises(ValueError, 1 .__format__, format_spec)
                self.assertRaises(ValueError, (-1) .__format__, format_spec)

        # ensure that float type specifiers work; format converts
        #  the int to a float
        for format_spec in 'eEfFgG%':
            for value in [0, 1, -1, 100, -100, 1234567890, -1234567890]:
                self.assertEqual(value.__format__(format_spec),
                                 float(value).__format__(format_spec))

        # Issue 6902
        test(123456, "0<20", '12345600000000000000')
        test(123456, "1<20", '12345611111111111111')
        test(123456, "*<20", '123456**************')
        test(123456, "0>20", '00000000000000123456')
        test(123456, "1>20", '11111111111111123456')
        test(123456, "*>20", '**************123456')
        test(123456, "0=20", '00000000000000123456')
        test(123456, "1=20", '11111111111111123456')
        test(123456, "*=20", '**************123456')

    @run_with_locale('LC_NUMERIC', 'en_US.UTF8', '')
    def test_float__format__locale(self):
        # test locale support for __format__ code 'n'

        for i in range(-10, 10):
            x = 1234567890.0 * (10.0 ** i)
            self.assertEqual(locale.format_string('%g', x, grouping=True), format(x, 'n'))
            self.assertEqual(locale.format_string('%.10g', x, grouping=True), format(x, '.10n'))

    @run_with_locale('LC_NUMERIC', 'en_US.UTF8', '')
    def test_int__format__locale(self):
        # test locale support for __format__ code 'n' for integers

        x = 123456789012345678901234567890
        for i in range(0, 30):
            self.assertEqual(locale.format_string('%d', x, grouping=True), format(x, 'n'))

            # move to the next integer to test
            x = x // 10

        rfmt = ">20n"
        lfmt = "<20n"
        cfmt = "^20n"
        for x in (1234, 12345, 123456, 1234567, 12345678, 123456789, 1234567890, 12345678900):
            self.assertEqual(len(format(0, rfmt)), len(format(x, rfmt)))
            self.assertEqual(len(format(0, lfmt)), len(format(x, lfmt)))
            self.assertEqual(len(format(0, cfmt)), len(format(x, cfmt)))

    def test_float__format__(self):
        def test(f, format_spec, result):
            self.assertEqual(f.__format__(format_spec), result)
            self.assertEqual(format(f, format_spec), result)

        test(0.0, 'f', '0.000000')

        # the default is 'g', except for empty format spec
        test(0.0, '', '0.0')
        test(0.01, '', '0.01')
        test(0.01, 'g', '0.01')

        # test for issue 3411
        test(1.23, '1', '1.23')
        test(-1.23, '1', '-1.23')
        test(1.23, '1g', '1.23')
        test(-1.23, '1g', '-1.23')

        test( 1.0, ' g', ' 1')
        test(-1.0, ' g', '-1')
        test( 1.0, '+g', '+1')
        test(-1.0, '+g', '-1')
        test(1.1234e200, 'g', '1.1234e+200')
        test(1.1234e200, 'G', '1.1234E+200')


        test(1.0, 'f', '1.000000')

        test(-1.0, 'f', '-1.000000')

        test( 1.0, ' f', ' 1.000000')
        test(-1.0, ' f', '-1.000000')
        test( 1.0, '+f', '+1.000000')
        test(-1.0, '+f', '-1.000000')

        # Python versions <= 3.0 switched from 'f' to 'g' formatting for
        # values larger than 1e50.  No longer.
        f = 1.1234e90
        for fmt in 'f', 'F':
            # don't do a direct equality check, since on some
            # platforms only the first few digits of dtoa
            # will be reliable
            result = f.__format__(fmt)
            self.assertEqual(len(result), 98)
            self.assertEqual(result[-7], '.')
            self.assertIn(result[:12], ('112340000000', '112339999999'))
        f = 1.1234e200
        for fmt in 'f', 'F':
            result = f.__format__(fmt)
            self.assertEqual(len(result), 208)
            self.assertEqual(result[-7], '.')
            self.assertIn(result[:12], ('112340000000', '112339999999'))


        test( 1.0, 'e', '1.000000e+00')
        test(-1.0, 'e', '-1.000000e+00')
        test( 1.0, 'E', '1.000000E+00')
        test(-1.0, 'E', '-1.000000E+00')
        test(1.1234e20, 'e', '1.123400e+20')
        test(1.1234e20, 'E', '1.123400E+20')

        # No format code means use g, but must have a decimal
        # and a number after the decimal.  This is tricky, because
        # a totally empty format specifier means something else.
        # So, just use a sign flag
        test(1e200, '+g', '+1e+200')
        test(1e200, '+', '+1e+200')

        test(1.1e200, '+g', '+1.1e+200')
        test(1.1e200, '+', '+1.1e+200')

        # 0 padding
        test(1234., '010f', '1234.000000')
        test(1234., '011f', '1234.000000')
        test(1234., '012f', '01234.000000')
        test(-1234., '011f', '-1234.000000')
        test(-1234., '012f', '-1234.000000')
        test(-1234., '013f', '-01234.000000')
        test(-1234.12341234, '013f', '-01234.123412')
        test(-123456.12341234, '011.2f', '-0123456.12')

        # issue 5782, commas with no specifier type
        test(1.2, '010,.2', '0,000,001.2')

        # 0 padding with commas
        test(1234., '011,f', '1,234.000000')
        test(1234., '012,f', '1,234.000000')
        test(1234., '013,f', '01,234.000000')
        test(-1234., '012,f', '-1,234.000000')
        test(-1234., '013,f', '-1,234.000000')
        test(-1234., '014,f', '-01,234.000000')
        test(-12345., '015,f', '-012,345.000000')
        test(-123456., '016,f', '-0,123,456.000000')
        test(-123456., '017,f', '-0,123,456.000000')
        test(-123456.12341234, '017,f', '-0,123,456.123412')
        test(-123456.12341234, '013,.2f', '-0,123,456.12')

        # % formatting
        test(-1.0, '%', '-100.000000%')

        # format spec must be string
        self.assertRaises(TypeError, 3.0.__format__, None)
        self.assertRaises(TypeError, 3.0.__format__, 0)

        # confirm format options expected to fail on floats, such as integer
        # presentation types
        for format_spec in 'sbcdoxX':
            self.assertRaises(ValueError, format, 0.0, format_spec)
            self.assertRaises(ValueError, format, 1.0, format_spec)
            self.assertRaises(ValueError, format, -1.0, format_spec)
            self.assertRaises(ValueError, format, 1e100, format_spec)
            self.assertRaises(ValueError, format, -1e100, format_spec)
            self.assertRaises(ValueError, format, 1e-100, format_spec)
            self.assertRaises(ValueError, format, -1e-100, format_spec)

        # Alternate float formatting
        test(1.0, '.0e', '1e+00')
        test(1.0, '#.0e', '1.e+00')
        test(1.0, '.0f', '1')
        test(1.0, '#.0f', '1.')
        test(1.1, 'g', '1.1')
        test(1.1, '#g', '1.10000')
        test(1.0, '.0%', '100%')
        test(1.0, '#.0%', '100.%')

        # Issue 7094: Alternate formatting (specified by #)
        test(1.0, '0e',  '1.000000e+00')
        test(1.0, '#0e', '1.000000e+00')
        test(1.0, '0f',  '1.000000' )
        test(1.0, '#0f', '1.000000')
        test(1.0, '.1e',  '1.0e+00')
        test(1.0, '#.1e', '1.0e+00')
        test(1.0, '.1f',  '1.0')
        test(1.0, '#.1f', '1.0')
        test(1.0, '.1%',  '100.0%')
        test(1.0, '#.1%', '100.0%')

        # Issue 6902
        test(12345.6, "0<20", '12345.60000000000000')
        test(12345.6, "1<20", '12345.61111111111111')
        test(12345.6, "*<20", '12345.6*************')
        test(12345.6, "0>20", '000000000000012345.6')
        test(12345.6, "1>20", '111111111111112345.6')
        test(12345.6, "*>20", '*************12345.6')
        test(12345.6, "0=20", '000000000000012345.6')
        test(12345.6, "1=20", '111111111111112345.6')
        test(12345.6, "*=20", '*************12345.6')

    def test_format_spec_errors(self):
        # int, float, and string all share the same format spec
        # mini-language parser.

        # Check that we can't ask for too many digits. This is
        # probably a CPython specific test. It tries to put the width
        # into a C long.
        self.assertRaises(ValueError, format, 0, '1'*10000 + 'd')

        # Similar with the precision.
        self.assertRaises(ValueError, format, 0, '.' + '1'*10000 + 'd')

        # And may as well test both.
        self.assertRaises(ValueError, format, 0, '1'*1000 + '.' + '1'*10000 + 'd')

        # Make sure commas aren't allowed with various type codes
        for code in 'xXobns':
            self.assertRaises(ValueError, format, 0, ',' + code)

    def test_internal_sizes(self):
        self.assertGreater(object.__basicsize__, 0)
        self.assertGreater(tuple.__itemsize__, 0)

    def test_slot_wrapper_types(self):
        self.assertIsInstance(object.__init__, types.WrapperDescriptorType)
        self.assertIsInstance(object.__str__, types.WrapperDescriptorType)
        self.assertIsInstance(object.__lt__, types.WrapperDescriptorType)
        self.assertIsInstance(int.__lt__, types.WrapperDescriptorType)

    @unittest.skipIf(MISSING_C_DOCSTRINGS,
                     "Signature information for builtins requires docstrings")
    def test_dunder_get_signature(self):
        sig = inspect.signature(object.__init__.__get__)
        self.assertEqual(list(sig.parameters), ["instance", "owner"])
        # gh-93021: Second parameter is optional
        self.assertIs(sig.parameters["owner"].default, None)

    def test_method_wrapper_types(self):
        self.assertIsInstance(object().__init__, types.MethodWrapperType)
        self.assertIsInstance(object().__str__, types.MethodWrapperType)
        self.assertIsInstance(object().__lt__, types.MethodWrapperType)
        self.assertIsInstance((42).__lt__, types.MethodWrapperType)

    def test_method_descriptor_types(self):
        self.assertIsInstance(str.join, types.MethodDescriptorType)
        self.assertIsInstance(list.append, types.MethodDescriptorType)
        self.assertIsInstance(''.join, types.BuiltinMethodType)
        self.assertIsInstance([].append, types.BuiltinMethodType)

        self.assertIsInstance(int.__dict__['from_bytes'], types.ClassMethodDescriptorType)
        self.assertIsInstance(int.from_bytes, types.BuiltinMethodType)
        self.assertIsInstance(int.__new__, types.BuiltinMethodType)

    def test_method_descriptor_crash(self):
        # gh-132747: The default __get__() implementation in C was unable
        # to handle a second argument of None when called from Python
        import _io
        import io
        import _queue

        to_check = [
            # (method, instance)
            (_io._TextIOBase.read, io.StringIO()),
            (_queue.SimpleQueue.put, _queue.SimpleQueue()),
            (str.capitalize, "nobody expects the spanish inquisition")
        ]

        for method, instance in to_check:
            with self.subTest(method=method, instance=instance):
                bound = method.__get__(instance)
                self.assertIsInstance(bound, types.BuiltinMethodType)

    def test_ellipsis_type(self):
        self.assertIsInstance(Ellipsis, types.EllipsisType)

    def test_notimplemented_type(self):
        self.assertIsInstance(NotImplemented, types.NotImplementedType)

    def test_none_type(self):
        self.assertIsInstance(None, types.NoneType)

    def test_traceback_and_frame_types(self):
        try:
            raise OSError
        except OSError as e:
            exc = e
        self.assertIsInstance(exc.__traceback__, types.TracebackType)
        self.assertIsInstance(exc.__traceback__.tb_frame, types.FrameType)

    def test_capsule_type(self):
        self.assertIsInstance(_datetime.datetime_CAPI, types.CapsuleType)

    def test_call_unbound_crash(self):
        # GH-131998: The specialized instruction would get tricked into dereferencing
        # a bound "self" that didn't exist if subsequently called unbound.
        code = """if True:

        def call(part):
            [] + ([] + [])
            part.pop()

        for _ in range(3):
            call(['a'])
        try:
            call(list)
        except TypeError:
            pass
        """
        assert_python_ok("-c", code)


class UnionTests(CPythonTestCase):

    def test_or_types_operator(self):
        self.assertEqual(int | str, typing.Union[int, str])
        self.assertNotEqual(int | list, typing.Union[int, str])
        self.assertEqual(str | int, typing.Union[int, str])
        self.assertEqual(int | None, typing.Union[int, None])
        self.assertEqual(None | int, typing.Union[int, None])
        self.assertEqual(int | type(None), int | None)
        self.assertEqual(type(None) | int, None | int)
        self.assertEqual(int | str | list, typing.Union[int, str, list])
        self.assertEqual(int | (str | list), typing.Union[int, str, list])
        self.assertEqual(str | (int | list), typing.Union[int, str, list])
        self.assertEqual(typing.List | typing.Tuple, typing.Union[typing.List, typing.Tuple])
        self.assertEqual(typing.List[int] | typing.Tuple[int], typing.Union[typing.List[int], typing.Tuple[int]])
        self.assertEqual(typing.List[int] | None, typing.Union[typing.List[int], None])
        self.assertEqual(None | typing.List[int], typing.Union[None, typing.List[int]])
        self.assertEqual(str | float | int | complex | int, (int | str) | (float | complex))
        self.assertEqual(typing.Union[str, int, typing.List[int]], str | int | typing.List[int])
        self.assertIs(int | int, int)
        self.assertEqual(
            BaseException |
            bool |
            bytes |
            complex |
            float |
            int |
            list |
            map |
            set,
            typing.Union[
                BaseException,
                bool,
                bytes,
                complex,
                float,
                int,
                list,
                map,
                set,
            ])
        with self.assertRaises(TypeError):
            int | 3
        with self.assertRaises(TypeError):
            3 | int
        with self.assertRaises(TypeError):
            Example() | int
        x = int | str
        self.assertEqual(x, int | str)
        self.assertEqual(x, str | int)
        self.assertNotEqual(x, {})  # should not raise exception
        with self.assertRaises(TypeError):
            x < x
        with self.assertRaises(TypeError):
            x <= x
        y = typing.Union[str, int]
        with self.assertRaises(TypeError):
            x < y
        y = int | bool
        with self.assertRaises(TypeError):
            x < y
        # Check that we don't crash if typing.Union does not have a tuple in __args__
        y = typing.Union[str, int]
        y.__args__ = [str, int]
        self.assertEqual(x, y)

    def test_hash(self):
        self.assertEqual(hash(int | str), hash(str | int))
        self.assertEqual(hash(int | str), hash(typing.Union[int, str]))

    def test_union_of_unhashable(self):
        with torch._dynamo.error_on_graph_break(False):
            class UnhashableMeta(type):
                __hash__ = None

        with torch._dynamo.error_on_graph_break(False):
            class A(metaclass=UnhashableMeta): ...
        with torch._dynamo.error_on_graph_break(False):
            class B(metaclass=UnhashableMeta): ...

        self.assertEqual((A | B).__args__, (A, B))
        union1 = A | B
        with self.assertRaises(TypeError):
            hash(union1)

        union2 = int | B
        with self.assertRaises(TypeError):
            hash(union2)

        union3 = A | int
        with self.assertRaises(TypeError):
            hash(union3)

    def test_instancecheck_and_subclasscheck(self):
        for x in (int | str, typing.Union[int, str]):
            with self.subTest(x=x):
                self.assertIsInstance(1, x)
                self.assertIsInstance(True, x)
                self.assertIsInstance('a', x)
                self.assertNotIsInstance(None, x)
                self.assertTrue(issubclass(int, x))
                self.assertTrue(issubclass(bool, x))
                self.assertTrue(issubclass(str, x))
                self.assertFalse(issubclass(type(None), x))

        for x in (int | None, typing.Union[int, None]):
            with self.subTest(x=x):
                self.assertIsInstance(None, x)
                self.assertTrue(issubclass(type(None), x))

        for x in (
            int | collections.abc.Mapping,
            typing.Union[int, collections.abc.Mapping],
        ):
            with self.subTest(x=x):
                self.assertIsInstance({}, x)
                self.assertNotIsInstance((), x)
                self.assertTrue(issubclass(dict, x))
                self.assertFalse(issubclass(list, x))

    def test_instancecheck_and_subclasscheck_order(self):
        T = typing.TypeVar('T')

        will_resolve = (
            int | T,
            typing.Union[int, T],
        )
        for x in will_resolve:
            with self.subTest(x=x):
                self.assertIsInstance(1, x)
                self.assertTrue(issubclass(int, x))

        wont_resolve = (
            T | int,
            typing.Union[T, int],
        )
        for x in wont_resolve:
            with self.subTest(x=x):
                with self.assertRaises(TypeError):
                    issubclass(int, x)
                with self.assertRaises(TypeError):
                    isinstance(1, x)

        for x in (*will_resolve, *wont_resolve):
            with self.subTest(x=x):
                with self.assertRaises(TypeError):
                    issubclass(object, x)
                with self.assertRaises(TypeError):
                    isinstance(object(), x)

    def test_bad_instancecheck(self):
        with torch._dynamo.error_on_graph_break(False):
            class BadMeta(type):
                def __instancecheck__(cls, inst):
                    1/0
        x = int | BadMeta('A', (), {})
        self.assertTrue(isinstance(1, x))
        self.assertRaises(ZeroDivisionError, isinstance, [], x)

    def test_bad_subclasscheck(self):
        with torch._dynamo.error_on_graph_break(False):
            class BadMeta(type):
                def __subclasscheck__(cls, sub):
                    1/0
        x = int | BadMeta('A', (), {})
        self.assertTrue(issubclass(int, x))
        self.assertRaises(ZeroDivisionError, issubclass, list, x)

    def test_or_type_operator_with_TypeVar(self):
        TV = typing.TypeVar('T')
        self.assertEqual(TV | str, typing.Union[TV, str])
        self.assertEqual(str | TV, typing.Union[str, TV])
        self.assertIs((int | TV)[int], int)
        self.assertIs((TV | int)[int], int)

    def test_union_args(self):
        def check(arg, expected):
            clear_typing_caches()
            self.assertEqual(arg.__args__, expected)

        check(int | str, (int, str))
        check((int | str) | list, (int, str, list))
        check(int | (str | list), (int, str, list))
        check((int | str) | int, (int, str))
        check(int | (str | int), (int, str))
        check((int | str) | (str | int), (int, str))
        check(typing.Union[int, str] | list, (int, str, list))
        check(int | typing.Union[str, list], (int, str, list))
        check((int | str) | (list | int), (int, str, list))
        check((int | str) | typing.Union[list, int], (int, str, list))
        check(typing.Union[int, str] | (list | int), (int, str, list))
        check((str | int) | (int | list), (str, int, list))
        check((str | int) | typing.Union[int, list], (str, int, list))
        check(typing.Union[str, int] | (int | list), (str, int, list))
        check(int | type(None), (int, type(None)))
        check(type(None) | int, (type(None), int))

        args = (int, list[int], typing.List[int],
                typing.Tuple[int, int], typing.Callable[[int], int],
                typing.Hashable, typing.TypeVar('T'))
        for x in args:
            with self.subTest(x):
                check(x | None, (x, type(None)))
                check(None | x, (type(None), x))

    def test_union_parameter_chaining(self):
        T = typing.TypeVar("T")
        S = typing.TypeVar("S")

        self.assertEqual((float | list[T])[int], float | list[int])
        self.assertEqual(list[int | list[T]].__parameters__, (T,))
        self.assertEqual(list[int | list[T]][str], list[int | list[str]])
        self.assertEqual((list[T] | list[S]).__parameters__, (T, S))
        self.assertEqual((list[T] | list[S])[int, T], list[int] | list[T])
        self.assertEqual((list[T] | list[S])[int, int], list[int])

    def test_union_parameter_substitution(self):
        def eq(actual, expected, typed=True):
            self.assertEqual(actual, expected)
            if typed:
                self.assertIs(type(actual), type(expected))

        T = typing.TypeVar('T')
        S = typing.TypeVar('S')
        NT = typing.NewType('NT', str)
        x = int | T | bytes

        eq(x[str], int | str | bytes, typed=False)
        eq(x[list[int]], int | list[int] | bytes, typed=False)
        eq(x[typing.List], int | typing.List | bytes)
        eq(x[typing.List[int]], int | typing.List[int] | bytes)
        eq(x[typing.Hashable], int | typing.Hashable | bytes)
        eq(x[collections.abc.Hashable],
           int | collections.abc.Hashable | bytes, typed=False)
        eq(x[typing.Callable[[int], str]],
           int | typing.Callable[[int], str] | bytes)
        eq(x[collections.abc.Callable[[int], str]],
           int | collections.abc.Callable[[int], str] | bytes, typed=False)
        eq(x[typing.Tuple[int, str]], int | typing.Tuple[int, str] | bytes)
        eq(x[typing.Literal['none']], int | typing.Literal['none'] | bytes)
        eq(x[str | list], int | str | list | bytes, typed=False)
        eq(x[typing.Union[str, list]], typing.Union[int, str, list, bytes])
        eq(x[str | int], int | str | bytes, typed=False)
        eq(x[typing.Union[str, int]], typing.Union[int, str, bytes])
        eq(x[NT], int | NT | bytes)
        eq(x[S], int | S | bytes)

    def test_union_pickle(self):
        orig = list[T] | int
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            s = pickle.dumps(orig, proto)
            loaded = pickle.loads(s)
            self.assertEqual(loaded, orig)
            self.assertEqual(loaded.__args__, orig.__args__)
            self.assertEqual(loaded.__parameters__, orig.__parameters__)

    def test_union_copy(self):
        orig = list[T] | int
        for copied in (copy.copy(orig), copy.deepcopy(orig)):
            self.assertEqual(copied, orig)
            self.assertEqual(copied.__args__, orig.__args__)
            self.assertEqual(copied.__parameters__, orig.__parameters__)

    def test_union_parameter_substitution_errors(self):
        T = typing.TypeVar("T")
        x = int | T
        with self.assertRaises(TypeError):
            x[int, str]

    def test_or_type_operator_with_forward(self):
        T = typing.TypeVar('T')
        ForwardAfter = T | 'Forward'
        ForwardBefore = 'Forward' | T
        def forward_after(x: ForwardAfter[int]) -> None: ...
        def forward_before(x: ForwardBefore[int]) -> None: ...
        self.assertEqual(typing.get_args(typing.get_type_hints(forward_after)['x']),
                         (int, Forward))
        self.assertEqual(typing.get_args(typing.get_type_hints(forward_before)['x']),
                         (int, Forward))

    def test_or_type_operator_with_Protocol(self):
        with torch._dynamo.error_on_graph_break(False):
            class Proto(typing.Protocol):
                def meth(self) -> int:
                    ...
        self.assertEqual(Proto | str, typing.Union[Proto, str])

    def test_or_type_operator_with_Alias(self):
        self.assertEqual(list | str, typing.Union[list, str])
        self.assertEqual(typing.List | str, typing.Union[typing.List, str])

    def test_or_type_operator_with_NamedTuple(self):
        NT = namedtuple('A', ['B', 'C', 'D'])
        self.assertEqual(NT | str, typing.Union[NT, str])

    def test_or_type_operator_with_TypedDict(self):
        with torch._dynamo.error_on_graph_break(False):
            class Point2D(typing.TypedDict):
                x: int
                y: int
                label: str
        self.assertEqual(Point2D | str, typing.Union[Point2D, str])

    def test_or_type_operator_with_NewType(self):
        UserId = typing.NewType('UserId', int)
        self.assertEqual(UserId | str, typing.Union[UserId, str])

    def test_or_type_operator_with_IO(self):
        self.assertEqual(typing.IO | str, typing.Union[typing.IO, str])

    def test_or_type_operator_with_SpecialForm(self):
        self.assertEqual(typing.Any | str, typing.Union[typing.Any, str])
        self.assertEqual(typing.NoReturn | str, typing.Union[typing.NoReturn, str])
        self.assertEqual(typing.Optional[int] | str, typing.Union[typing.Optional[int], str])
        self.assertEqual(typing.Optional[int] | str, typing.Union[int, str, None])
        self.assertEqual(typing.Union[int, bool] | str, typing.Union[int, bool, str])

    def test_or_type_operator_with_Literal(self):
        Literal = typing.Literal
        self.assertEqual((Literal[1] | Literal[2]).__args__,
                         (Literal[1], Literal[2]))

        self.assertEqual((Literal[0] | Literal[False]).__args__,
                         (Literal[0], Literal[False]))
        self.assertEqual((Literal[1] | Literal[True]).__args__,
                         (Literal[1], Literal[True]))

        self.assertEqual(Literal[1] | Literal[1], Literal[1])
        self.assertEqual(Literal['a'] | Literal['a'], Literal['a'])

        import enum
        with torch._dynamo.error_on_graph_break(False):
            class Ints(enum.IntEnum):
                A = 0
                B = 1

        self.assertEqual(Literal[Ints.A] | Literal[Ints.A], Literal[Ints.A])
        self.assertEqual(Literal[Ints.B] | Literal[Ints.B], Literal[Ints.B])

        self.assertEqual((Literal[Ints.B] | Literal[Ints.A]).__args__,
                         (Literal[Ints.B], Literal[Ints.A]))

        self.assertEqual((Literal[0] | Literal[Ints.A]).__args__,
                         (Literal[0], Literal[Ints.A]))
        self.assertEqual((Literal[1] | Literal[Ints.B]).__args__,
                         (Literal[1], Literal[Ints.B]))

    def test_or_type_repr(self):
        self.assertEqual(repr(int | str), "int | str")
        self.assertEqual(repr((int | str) | list), "int | str | list")
        self.assertEqual(repr(int | (str | list)), "int | str | list")
        self.assertEqual(repr(int | None), "int | None")
        self.assertEqual(repr(int | type(None)), "int | None")
        self.assertEqual(repr(int | typing.GenericAlias(list, int)), "int | list[int]")

    def test_or_type_operator_with_genericalias(self):
        a = list[int]
        b = list[str]
        c = dict[float, str]
        with torch._dynamo.error_on_graph_break(False):
            class SubClass(types.GenericAlias): ...
        d = SubClass(list, float)
        # equivalence with typing.Union
        self.assertEqual(a | b | c | d, typing.Union[a, b, c, d])
        # de-duplicate
        self.assertEqual(a | c | b | b | a | c | d | d, a | b | c | d)
        # order shouldn't matter
        self.assertEqual(a | b | d, b | a | d)
        self.assertEqual(repr(a | b | c | d),
                         "list[int] | list[str] | dict[float, str] | list[float]")

        with torch._dynamo.error_on_graph_break(False):
            class BadType(type):
                def __eq__(self, other):
                    return 1 / 0

        bt = BadType('bt', (), {})
        # Comparison should fail and errors should propagate out for bad types.
        with self.assertRaises(ZeroDivisionError):
            list[int] | list[bt]

        union_ga = (list[str] | int, collections.abc.Callable[..., str] | int,
                    d | int)
        # Raise error when isinstance(type, genericalias | type)
        for type_ in union_ga:
            with self.subTest(f"check isinstance/issubclass is invalid for {type_}"):
                with self.assertRaises(TypeError):
                    isinstance(1, type_)
                with self.assertRaises(TypeError):
                    issubclass(int, type_)

    def test_or_type_operator_with_bad_module(self):
        with torch._dynamo.error_on_graph_break(False):
            class BadMeta(type):
                __qualname__ = 'TypeVar'
                @property
                def __module__(self):
                    1 / 0
        TypeVar = BadMeta('TypeVar', (), {})
        _SpecialForm = BadMeta('_SpecialForm', (), {})
        # Crashes in Issue44483
        with self.assertRaises((TypeError, ZeroDivisionError)):
            str | TypeVar()
        with self.assertRaises((TypeError, ZeroDivisionError)):
            str | _SpecialForm()

    @cpython_only
    def test_or_type_operator_reference_cycle(self):
        if not hasattr(sys, 'gettotalrefcount'):
            self.skipTest('Cannot get total reference count.')
        gc.collect()
        before = sys.gettotalrefcount()
        for _ in range(30):
            T = typing.TypeVar('T')
            U = int | list[T]
            T.blah = U
            del T
            del U
        gc.collect()
        leeway = 15
        self.assertLessEqual(sys.gettotalrefcount() - before, leeway,
                             msg='Check for union reference leak.')


class MappingProxyTests(CPythonTestCase):
    mappingproxy = types.MappingProxyType

    def test_constructor(self):
        with torch._dynamo.error_on_graph_break(False):
            class userdict(dict):
                pass

        mapping = {'x': 1, 'y': 2}
        self.assertEqual(self.mappingproxy(mapping), mapping)
        mapping = userdict(x=1, y=2)
        self.assertEqual(self.mappingproxy(mapping), mapping)
        mapping = collections.ChainMap({'x': 1}, {'y': 2})
        self.assertEqual(self.mappingproxy(mapping), mapping)

        self.assertRaises(TypeError, self.mappingproxy, 10)
        self.assertRaises(TypeError, self.mappingproxy, ("a", "tuple"))
        self.assertRaises(TypeError, self.mappingproxy, ["a", "list"])

    def test_methods(self):
        attrs = set(dir(self.mappingproxy({}))) - set(dir(object()))
        self.assertEqual(attrs, {
             '__contains__',
             '__getitem__',
             '__class_getitem__',
             '__ior__',
             '__iter__',
             '__len__',
             '__or__',
             '__reversed__',
             '__ror__',
             'copy',
             'get',
             'items',
             'keys',
             'values',
        })

    def test_get(self):
        view = self.mappingproxy({'a': 'A', 'b': 'B'})
        self.assertEqual(view['a'], 'A')
        self.assertEqual(view['b'], 'B')
        self.assertRaises(KeyError, view.__getitem__, 'xxx')
        self.assertEqual(view.get('a'), 'A')
        self.assertIsNone(view.get('xxx'))
        self.assertEqual(view.get('xxx', 42), 42)

    def test_missing(self):
        with torch._dynamo.error_on_graph_break(False):
            class dictmissing(dict):
                def __missing__(self, key):
                    return "missing=%s" % key

        view = self.mappingproxy(dictmissing(x=1))
        self.assertEqual(view['x'], 1)
        self.assertEqual(view['y'], 'missing=y')
        self.assertEqual(view.get('x'), 1)
        self.assertEqual(view.get('y'), None)
        self.assertEqual(view.get('y', 42), 42)
        self.assertTrue('x' in view)
        self.assertFalse('y' in view)

    def test_customdict(self):
        with torch._dynamo.error_on_graph_break(False):
            class customdict(dict):
                def __contains__(self, key):
                    if key == 'magic':
                        return True
                    else:
                        return dict.__contains__(self, key)

                def __iter__(self):
                    return iter(('iter',))

                def __len__(self):
                    return 500

                def copy(self):
                    return 'copy'

                def keys(self):
                    return 'keys'

                def items(self):
                    return 'items'

                def values(self):
                    return 'values'

                def __getitem__(self, key):
                    return "getitem=%s" % dict.__getitem__(self, key)

                def get(self, key, default=None):
                    return "get=%s" % dict.get(self, key, 'default=%r' % default)

        custom = customdict({'key': 'value'})
        view = self.mappingproxy(custom)
        self.assertTrue('key' in view)
        self.assertTrue('magic' in view)
        self.assertFalse('xxx' in view)
        self.assertEqual(view['key'], 'getitem=value')
        self.assertRaises(KeyError, view.__getitem__, 'xxx')
        self.assertEqual(tuple(view), ('iter',))
        self.assertEqual(len(view), 500)
        self.assertEqual(view.copy(), 'copy')
        self.assertEqual(view.get('key'), 'get=value')
        self.assertEqual(view.get('xxx'), 'get=default=None')
        self.assertEqual(view.items(), 'items')
        self.assertEqual(view.keys(), 'keys')
        self.assertEqual(view.values(), 'values')

    def test_chainmap(self):
        d1 = {'x': 1}
        d2 = {'y': 2}
        mapping = collections.ChainMap(d1, d2)
        view = self.mappingproxy(mapping)
        self.assertTrue('x' in view)
        self.assertTrue('y' in view)
        self.assertFalse('z' in view)
        self.assertEqual(view['x'], 1)
        self.assertEqual(view['y'], 2)
        self.assertRaises(KeyError, view.__getitem__, 'z')
        self.assertEqual(tuple(sorted(view)), ('x', 'y'))
        self.assertEqual(len(view), 2)
        copy = view.copy()
        self.assertIsNot(copy, mapping)
        self.assertIsInstance(copy, collections.ChainMap)
        self.assertEqual(copy, mapping)
        self.assertEqual(view.get('x'), 1)
        self.assertEqual(view.get('y'), 2)
        self.assertIsNone(view.get('z'))
        self.assertEqual(tuple(sorted(view.items())), (('x', 1), ('y', 2)))
        self.assertEqual(tuple(sorted(view.keys())), ('x', 'y'))
        self.assertEqual(tuple(sorted(view.values())), (1, 2))

    def test_contains(self):
        view = self.mappingproxy(dict.fromkeys('abc'))
        self.assertTrue('a' in view)
        self.assertTrue('b' in view)
        self.assertTrue('c' in view)
        self.assertFalse('xxx' in view)

    def test_views(self):
        mapping = {}
        view = self.mappingproxy(mapping)
        keys = view.keys()
        values = view.values()
        items = view.items()
        self.assertEqual(list(keys), [])
        self.assertEqual(list(values), [])
        self.assertEqual(list(items), [])
        mapping['key'] = 'value'
        self.assertEqual(list(keys), ['key'])
        self.assertEqual(list(values), ['value'])
        self.assertEqual(list(items), [('key', 'value')])

    def test_len(self):
        for expected in range(6):
            data = dict.fromkeys('abcde'[:expected])
            self.assertEqual(len(data), expected)
            view = self.mappingproxy(data)
            self.assertEqual(len(view), expected)

    def test_iterators(self):
        keys = ('x', 'y')
        values = (1, 2)
        items = tuple(zip(keys, values))
        view = self.mappingproxy(dict(items))
        self.assertEqual(set(view), set(keys))
        self.assertEqual(set(view.keys()), set(keys))
        self.assertEqual(set(view.values()), set(values))
        self.assertEqual(set(view.items()), set(items))

    def test_reversed(self):
        d = {'a': 1, 'b': 2, 'foo': 0, 'c': 3, 'd': 4}
        mp = self.mappingproxy(d)
        del d['foo']
        r = reversed(mp)
        self.assertEqual(list(r), list('dcba'))
        self.assertRaises(StopIteration, next, r)

    def test_copy(self):
        original = {'key1': 27, 'key2': 51, 'key3': 93}
        view = self.mappingproxy(original)
        copy = view.copy()
        self.assertEqual(type(copy), dict)
        self.assertEqual(copy, original)
        original['key1'] = 70
        self.assertEqual(view['key1'], 70)
        self.assertEqual(copy['key1'], 27)

    def test_union(self):
        mapping = {'a': 0, 'b': 1, 'c': 2}
        view = self.mappingproxy(mapping)
        with self.assertRaises(TypeError):
            view | [('r', 2), ('d', 2)]
        with self.assertRaises(TypeError):
            [('r', 2), ('d', 2)] | view
        with self.assertRaises(TypeError):
            view |= [('r', 2), ('d', 2)]
        other = {'c': 3, 'p': 0}
        self.assertDictEqual(view | other, {'a': 0, 'b': 1, 'c': 3, 'p': 0})
        self.assertDictEqual(other | view, {'c': 2, 'p': 0, 'a': 0, 'b': 1})
        self.assertEqual(view, {'a': 0, 'b': 1, 'c': 2})
        self.assertDictEqual(mapping, {'a': 0, 'b': 1, 'c': 2})
        self.assertDictEqual(other, {'c': 3, 'p': 0})

    def test_hash(self):
        with torch._dynamo.error_on_graph_break(False):
            class HashableDict(dict):
                def __hash__(self):
                    return 3844817361
        view = self.mappingproxy({'a': 1, 'b': 2})
        self.assertRaises(TypeError, hash, view)
        mapping = HashableDict({'a': 1, 'b': 2})
        view = self.mappingproxy(mapping)
        self.assertEqual(hash(view), hash(mapping))


class ClassCreationTests(CPythonTestCase):

    class Meta(type):
        def __init__(cls, name, bases, ns, **kw):
            super().__init__(name, bases, ns)
        @staticmethod
        def __new__(mcls, name, bases, ns, **kw):
            return super().__new__(mcls, name, bases, ns)
        @classmethod
        def __prepare__(mcls, name, bases, **kw):
            ns = super().__prepare__(name, bases)
            ns["y"] = 1
            ns.update(kw)
            return ns

    def test_new_class_basics(self):
        C = types.new_class("C")
        self.assertEqual(C.__name__, "C")
        self.assertEqual(C.__bases__, (object,))

    def test_new_class_subclass(self):
        C = types.new_class("C", (int,))
        self.assertTrue(issubclass(C, int))

    def test_new_class_meta(self):
        Meta = self.Meta
        settings = {"metaclass": Meta, "z": 2}
        # We do this twice to make sure the passed in dict isn't mutated
        for i in range(2):
            C = types.new_class("C" + str(i), (), settings)
            self.assertIsInstance(C, Meta)
            self.assertEqual(C.y, 1)
            self.assertEqual(C.z, 2)

    def test_new_class_exec_body(self):
        Meta = self.Meta
        def func(ns):
            ns["x"] = 0
        C = types.new_class("C", (), {"metaclass": Meta, "z": 2}, func)
        self.assertIsInstance(C, Meta)
        self.assertEqual(C.x, 0)
        self.assertEqual(C.y, 1)
        self.assertEqual(C.z, 2)

    def test_new_class_metaclass_keywords(self):
        #Test that keywords are passed to the metaclass:
        def meta_func(name, bases, ns, **kw):
            return name, bases, ns, kw
        res = types.new_class("X",
                              (int, object),
                              dict(metaclass=meta_func, x=0))
        self.assertEqual(res, ("X", (int, object), {}, {"x": 0}))

    def test_new_class_defaults(self):
        # Test defaults/keywords:
        C = types.new_class("C", (), {}, None)
        self.assertEqual(C.__name__, "C")
        self.assertEqual(C.__bases__, (object,))

    def test_new_class_meta_with_base(self):
        Meta = self.Meta
        def func(ns):
            ns["x"] = 0
        C = types.new_class(name="C",
                            bases=(int,),
                            kwds=dict(metaclass=Meta, z=2),
                            exec_body=func)
        self.assertTrue(issubclass(C, int))
        self.assertIsInstance(C, Meta)
        self.assertEqual(C.x, 0)
        self.assertEqual(C.y, 1)
        self.assertEqual(C.z, 2)

    def test_new_class_with_mro_entry(self):
        with torch._dynamo.error_on_graph_break(False):
            class A: pass
        with torch._dynamo.error_on_graph_break(False):
            class C:
                def __mro_entries__(self, bases):
                    return (A,)
        c = C()
        D = types.new_class('D', (c,), {})
        self.assertEqual(D.__bases__, (A,))
        self.assertEqual(D.__orig_bases__, (c,))
        self.assertEqual(D.__mro__, (D, A, object))

    def test_new_class_with_mro_entry_genericalias(self):
        L1 = types.new_class('L1', (typing.List[int],), {})
        self.assertEqual(L1.__bases__, (list, typing.Generic))
        self.assertEqual(L1.__orig_bases__, (typing.List[int],))
        self.assertEqual(L1.__mro__, (L1, list, typing.Generic, object))

        L2 = types.new_class('L2', (list[int],), {})
        self.assertEqual(L2.__bases__, (list,))
        self.assertEqual(L2.__orig_bases__, (list[int],))
        self.assertEqual(L2.__mro__, (L2, list, object))

    def test_new_class_with_mro_entry_none(self):
        with torch._dynamo.error_on_graph_break(False):
            class A: pass
            class B: pass
            class C:
                def __mro_entries__(self, bases):
                    return ()
        c = C()
        D = types.new_class('D', (A, c, B), {})
        self.assertEqual(D.__bases__, (A, B))
        self.assertEqual(D.__orig_bases__, (A, c, B))
        self.assertEqual(D.__mro__, (D, A, B, object))

    def test_new_class_with_mro_entry_error(self):
        with torch._dynamo.error_on_graph_break(False):
            class A: pass
            class C:
                def __mro_entries__(self, bases):
                    return A
        c = C()
        with self.assertRaises(TypeError):
            types.new_class('D', (c,), {})

    def test_new_class_with_mro_entry_multiple(self):
        with torch._dynamo.error_on_graph_break(False):
            class A1: pass
            class A2: pass
            class B1: pass
            class B2: pass
            class A:
                def __mro_entries__(self, bases):
                    return (A1, A2)
            class B:
                def __mro_entries__(self, bases):
                    return (B1, B2)
        D = types.new_class('D', (A(), B()), {})
        self.assertEqual(D.__bases__, (A1, A2, B1, B2))

    def test_new_class_with_mro_entry_multiple_2(self):
        with torch._dynamo.error_on_graph_break(False):
            class A1: pass
            class A2: pass
            class A3: pass
            class B1: pass
            class B2: pass
            class A:
                def __mro_entries__(self, bases):
                    return (A1, A2, A3)
            class B:
                def __mro_entries__(self, bases):
                    return (B1, B2)
            class C: pass
        D = types.new_class('D', (A(), C, B()), {})
        self.assertEqual(D.__bases__, (A1, A2, A3, C, B1, B2))

    def test_get_original_bases(self):
        T = typing.TypeVar('T')
        with torch._dynamo.error_on_graph_break(False):
            class A: pass
            class B(typing.Generic[T]): pass
            class C(B[int]): pass
            class D(B[str], float): pass

        self.assertEqual(types.get_original_bases(A), (object,))
        self.assertEqual(types.get_original_bases(B), (typing.Generic[T],))
        self.assertEqual(types.get_original_bases(C), (B[int],))
        self.assertEqual(types.get_original_bases(int), (object,))
        self.assertEqual(types.get_original_bases(D), (B[str], float))

        with torch._dynamo.error_on_graph_break(False):
            class E(list[T]): pass
            class F(list[int]): pass

        self.assertEqual(types.get_original_bases(E), (list[T],))
        self.assertEqual(types.get_original_bases(F), (list[int],))

        with torch._dynamo.error_on_graph_break(False):
            class FirstBase(typing.Generic[T]): pass
            class SecondBase(typing.Generic[T]): pass
            class First(FirstBase[int]): pass
            class Second(SecondBase[int]): pass
            class G(First, Second): pass
        self.assertEqual(types.get_original_bases(G), (First, Second))

        with torch._dynamo.error_on_graph_break(False):
            class First_(typing.Generic[T]): pass
            class Second_(typing.Generic[T]): pass
            class H(First_, Second_): pass
        self.assertEqual(types.get_original_bases(H), (First_, Second_))

        with torch._dynamo.error_on_graph_break(False):
            class ClassBasedNamedTuple(typing.NamedTuple):
                x: int

            class GenericNamedTuple(typing.NamedTuple, typing.Generic[T]):
                x: T

        CallBasedNamedTuple = typing.NamedTuple("CallBasedNamedTuple", [("x", int)])

        self.assertIs(
            types.get_original_bases(ClassBasedNamedTuple)[0], typing.NamedTuple
        )
        self.assertEqual(
            types.get_original_bases(GenericNamedTuple),
            (typing.NamedTuple, typing.Generic[T])
        )
        self.assertIs(
            types.get_original_bases(CallBasedNamedTuple)[0], typing.NamedTuple
        )

        with torch._dynamo.error_on_graph_break(False):
            class ClassBasedTypedDict(typing.TypedDict):
                x: int

            class GenericTypedDict(typing.TypedDict, typing.Generic[T]):
                x: T

        CallBasedTypedDict = typing.TypedDict("CallBasedTypedDict", {"x": int})

        self.assertIs(
            types.get_original_bases(ClassBasedTypedDict)[0],
            typing.TypedDict
        )
        self.assertEqual(
            types.get_original_bases(GenericTypedDict),
            (typing.TypedDict, typing.Generic[T])
        )
        self.assertIs(
            types.get_original_bases(CallBasedTypedDict)[0],
            typing.TypedDict
        )

        with self.assertRaisesRegex(TypeError, "Expected an instance of type"):
            types.get_original_bases(object())

    # Many of the following tests are derived from test_descr.py
    def test_prepare_class(self):
        # Basic test of metaclass derivation
        expected_ns = {}
        with torch._dynamo.error_on_graph_break(False):
            class A(type):
                def __new__(*args, **kwargs):
                    return type.__new__(*args, **kwargs)

                def __prepare__(*args):
                    return expected_ns

        B = types.new_class("B", (object,))
        C = types.new_class("C", (object,), {"metaclass": A})

        # The most derived metaclass of D is A rather than type.
        meta, ns, kwds = types.prepare_class("D", (B, C), {"metaclass": type})
        self.assertIs(meta, A)
        self.assertIs(ns, expected_ns)
        self.assertEqual(len(kwds), 0)

    def test_bad___prepare__(self):
        # __prepare__() must return a mapping.
        with torch._dynamo.error_on_graph_break(False):
            class BadMeta(type):
                @classmethod
                def __prepare__(*args):
                    return None
        with self.assertRaisesRegex(TypeError,
                                    r'^BadMeta\.__prepare__\(\) must '
                                    r'return a mapping, not NoneType$'):
            class Foo(metaclass=BadMeta):
                pass
        # Also test the case in which the metaclass is not a type.
        with torch._dynamo.error_on_graph_break(False):
            class BadMeta:
                @classmethod
                def __prepare__(*args):
                    return None
        with self.assertRaisesRegex(TypeError,
                                    r'^<metaclass>\.__prepare__\(\) must '
                                    r'return a mapping, not NoneType$'):
            class Bar(metaclass=BadMeta()):
                pass

    def test_resolve_bases(self):
        with torch._dynamo.error_on_graph_break(False):
            class A: pass
            class B: pass
            class C:
                def __mro_entries__(self, bases):
                    if A in bases:
                        return ()
                    return (A,)
        c = C()
        self.assertEqual(types.resolve_bases(()), ())
        self.assertEqual(types.resolve_bases((c,)), (A,))
        self.assertEqual(types.resolve_bases((C,)), (C,))
        self.assertEqual(types.resolve_bases((A, C)), (A, C))
        self.assertEqual(types.resolve_bases((c, A)), (A,))
        self.assertEqual(types.resolve_bases((A, c)), (A,))
        x = (A,)
        y = (C,)
        z = (A, C)
        t = (A, C, B)
        for bases in [x, y, z, t]:
            self.assertIs(types.resolve_bases(bases), bases)

    def test_resolve_bases_with_mro_entry(self):
        self.assertEqual(types.resolve_bases((typing.List[int],)),
                         (list, typing.Generic))
        self.assertEqual(types.resolve_bases((list[int],)), (list,))

    def test_metaclass_derivation(self):
        # issue1294232: correct metaclass calculation
        new_calls = []  # to check the order of __new__ calls
        with torch._dynamo.error_on_graph_break(False):
            class AMeta(type):
                def __new__(mcls, name, bases, ns):
                    new_calls.append('AMeta')
                    return super().__new__(mcls, name, bases, ns)
                @classmethod
                def __prepare__(mcls, name, bases):
                    return {}

            class BMeta(AMeta):
                def __new__(mcls, name, bases, ns):
                    new_calls.append('BMeta')
                    return super().__new__(mcls, name, bases, ns)
                @classmethod
                def __prepare__(mcls, name, bases):
                    ns = super().__prepare__(name, bases)
                    ns['BMeta_was_here'] = True
                    return ns

        A = types.new_class("A", (), {"metaclass": AMeta})
        self.assertEqual(new_calls, ['AMeta'])
        new_calls.clear()

        B = types.new_class("B", (), {"metaclass": BMeta})
        # BMeta.__new__ calls AMeta.__new__ with super:
        self.assertEqual(new_calls, ['BMeta', 'AMeta'])
        new_calls.clear()

        C = types.new_class("C", (A, B))
        # The most derived metaclass is BMeta:
        self.assertEqual(new_calls, ['BMeta', 'AMeta'])
        new_calls.clear()
        # BMeta.__prepare__ should've been called:
        self.assertIn('BMeta_was_here', C.__dict__)

        # The order of the bases shouldn't matter:
        C2 = types.new_class("C2", (B, A))
        self.assertEqual(new_calls, ['BMeta', 'AMeta'])
        new_calls.clear()
        self.assertIn('BMeta_was_here', C2.__dict__)

        # Check correct metaclass calculation when a metaclass is declared:
        D = types.new_class("D", (C,), {"metaclass": type})
        self.assertEqual(new_calls, ['BMeta', 'AMeta'])
        new_calls.clear()
        self.assertIn('BMeta_was_here', D.__dict__)

        E = types.new_class("E", (C,), {"metaclass": AMeta})
        self.assertEqual(new_calls, ['BMeta', 'AMeta'])
        new_calls.clear()
        self.assertIn('BMeta_was_here', E.__dict__)

    def test_metaclass_override_function(self):
        # Special case: the given metaclass isn't a class,
        # so there is no metaclass calculation.
        with torch._dynamo.error_on_graph_break(False):
            class A(metaclass=self.Meta):
                pass

        marker = object()
        def func(*args, **kwargs):
            return marker

        X = types.new_class("X", (), {"metaclass": func})
        Y = types.new_class("Y", (object,), {"metaclass": func})
        Z = types.new_class("Z", (A,), {"metaclass": func})
        self.assertIs(marker, X)
        self.assertIs(marker, Y)
        self.assertIs(marker, Z)

    def test_metaclass_override_callable(self):
        # The given metaclass is a class,
        # but not a descendant of type.
        new_calls = []  # to check the order of __new__ calls
        prepare_calls = []  # to track __prepare__ calls
        with torch._dynamo.error_on_graph_break(False):
            class ANotMeta:
                def __new__(mcls, *args, **kwargs):
                    new_calls.append('ANotMeta')
                    return super().__new__(mcls)
                @classmethod
                def __prepare__(mcls, name, bases):
                    prepare_calls.append('ANotMeta')
                    return {}

            class BNotMeta(ANotMeta):
                def __new__(mcls, *args, **kwargs):
                    new_calls.append('BNotMeta')
                    return super().__new__(mcls)
                @classmethod
                def __prepare__(mcls, name, bases):
                    prepare_calls.append('BNotMeta')
                    return super().__prepare__(name, bases)

        A = types.new_class("A", (), {"metaclass": ANotMeta})
        self.assertIs(ANotMeta, type(A))
        self.assertEqual(prepare_calls, ['ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['ANotMeta'])
        new_calls.clear()

        B = types.new_class("B", (), {"metaclass": BNotMeta})
        self.assertIs(BNotMeta, type(B))
        self.assertEqual(prepare_calls, ['BNotMeta', 'ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['BNotMeta', 'ANotMeta'])
        new_calls.clear()

        C = types.new_class("C", (A, B))
        self.assertIs(BNotMeta, type(C))
        self.assertEqual(prepare_calls, ['BNotMeta', 'ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['BNotMeta', 'ANotMeta'])
        new_calls.clear()

        C2 = types.new_class("C2", (B, A))
        self.assertIs(BNotMeta, type(C2))
        self.assertEqual(prepare_calls, ['BNotMeta', 'ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['BNotMeta', 'ANotMeta'])
        new_calls.clear()

        # This is a TypeError, because of a metaclass conflict:
        # BNotMeta is neither a subclass, nor a superclass of type
        with self.assertRaises(TypeError):
            D = types.new_class("D", (C,), {"metaclass": type})

        E = types.new_class("E", (C,), {"metaclass": ANotMeta})
        self.assertIs(BNotMeta, type(E))
        self.assertEqual(prepare_calls, ['BNotMeta', 'ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['BNotMeta', 'ANotMeta'])
        new_calls.clear()

        F = types.new_class("F", (object(), C))
        self.assertIs(BNotMeta, type(F))
        self.assertEqual(prepare_calls, ['BNotMeta', 'ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['BNotMeta', 'ANotMeta'])
        new_calls.clear()

        F2 = types.new_class("F2", (C, object()))
        self.assertIs(BNotMeta, type(F2))
        self.assertEqual(prepare_calls, ['BNotMeta', 'ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['BNotMeta', 'ANotMeta'])
        new_calls.clear()

        # TypeError: BNotMeta is neither a
        # subclass, nor a superclass of int
        with self.assertRaises(TypeError):
            X = types.new_class("X", (C, int()))
        with self.assertRaises(TypeError):
            X = types.new_class("X", (int(), C))

    def test_one_argument_type(self):
        expected_message = 'type.__new__() takes exactly 3 arguments (1 given)'

        # Only type itself can use the one-argument form (#27157)
        self.assertIs(type(5), int)

        with torch._dynamo.error_on_graph_break(False):
            class M(type):
                pass
        with self.assertRaises(TypeError) as cm:
            M(5)
        self.assertEqual(str(cm.exception), expected_message)

        with torch._dynamo.error_on_graph_break(False):
            class N(type, metaclass=M):
                pass
        with self.assertRaises(TypeError) as cm:
            N(5)
        self.assertEqual(str(cm.exception), expected_message)

    def test_metaclass_new_error(self):
        # bpo-44232: The C function type_new() must properly report the
        # exception when a metaclass constructor raises an exception and the
        # winner class is not the metaclass.
        with torch._dynamo.error_on_graph_break(False):
            class ModelBase(type):
                def __new__(cls, name, bases, attrs):
                    super_new = super().__new__
                    new_class = super_new(cls, name, bases, {})
                    if name != "Model":
                        raise RuntimeWarning(f"{name=}")
                    return new_class

        with torch._dynamo.error_on_graph_break(False):
            class Model(metaclass=ModelBase):
                pass

        with self.assertRaises(RuntimeWarning):
            type("SouthPonies", (Model,), {})

    def test_tuple_subclass_as_bases(self):
        # gh-132176: it used to crash on using
        # tuple subclass for as base classes.
        with torch._dynamo.error_on_graph_break(False):
            class TupleSubclass(tuple): pass

        typ = type("typ", TupleSubclass((int, object)), {})
        self.assertEqual(typ.__bases__, (int, object))
        self.assertEqual(type(typ.__bases__), TupleSubclass)


class SimpleNamespaceTests(CPythonTestCase):

    def test_constructor(self):
        def check(ns, expected):
            self.assertEqual(len(ns.__dict__), len(expected))
            self.assertEqual(vars(ns), expected)
            # check order
            self.assertEqual(list(vars(ns).items()), list(expected.items()))
            for name in expected:
                self.assertEqual(getattr(ns, name), expected[name])

        check(types.SimpleNamespace(), {})
        check(types.SimpleNamespace(x=1, y=2), {'x': 1, 'y': 2})
        check(types.SimpleNamespace(**dict(x=1, y=2)), {'x': 1, 'y': 2})
        check(types.SimpleNamespace({'x': 1, 'y': 2}, x=4, z=3),
              {'x': 4, 'y': 2, 'z': 3})
        check(types.SimpleNamespace([['x', 1], ['y', 2]], x=4, z=3),
              {'x': 4, 'y': 2, 'z': 3})
        check(types.SimpleNamespace(UserDict({'x': 1, 'y': 2}), x=4, z=3),
              {'x': 4, 'y': 2, 'z': 3})
        check(types.SimpleNamespace({'x': 1, 'y': 2}), {'x': 1, 'y': 2})
        check(types.SimpleNamespace([['x', 1], ['y', 2]]), {'x': 1, 'y': 2})
        check(types.SimpleNamespace([], x=4, z=3), {'x': 4, 'z': 3})
        check(types.SimpleNamespace({}, x=4, z=3), {'x': 4, 'z': 3})
        check(types.SimpleNamespace([]), {})
        check(types.SimpleNamespace({}), {})

        with self.assertRaises(TypeError):
            types.SimpleNamespace([], [])  # too many positional arguments
        with self.assertRaises(TypeError):
            types.SimpleNamespace(1)  # not a mapping or iterable
        with self.assertRaises(TypeError):
            types.SimpleNamespace([1])  # non-iterable
        with self.assertRaises(ValueError):
            types.SimpleNamespace([['x']])  # not a pair
        with self.assertRaises(ValueError):
            types.SimpleNamespace([['x', 'y', 'z']])
        with self.assertRaises(TypeError):
            types.SimpleNamespace(**{1: 2})  # non-string key
        with self.assertRaises(TypeError):
            types.SimpleNamespace({1: 2})
        with self.assertRaises(TypeError):
            types.SimpleNamespace([[1, 2]])
        with self.assertRaises(TypeError):
            types.SimpleNamespace(UserDict({1: 2}))
        with self.assertRaises(TypeError):
            types.SimpleNamespace([[[], 2]])  # non-hashable key

    def test_unbound(self):
        ns1 = vars(types.SimpleNamespace())
        ns2 = vars(types.SimpleNamespace(x=1, y=2))

        self.assertEqual(ns1, {})
        self.assertEqual(ns2, {'y': 2, 'x': 1})

    def test_underlying_dict(self):
        ns1 = types.SimpleNamespace()
        ns2 = types.SimpleNamespace(x=1, y=2)
        ns3 = types.SimpleNamespace(a=True, b=False)
        mapping = ns3.__dict__
        del ns3

        self.assertEqual(ns1.__dict__, {})
        self.assertEqual(ns2.__dict__, {'y': 2, 'x': 1})
        self.assertEqual(mapping, dict(a=True, b=False))

    def test_attrget(self):
        ns = types.SimpleNamespace(x=1, y=2, w=3)

        self.assertEqual(ns.x, 1)
        self.assertEqual(ns.y, 2)
        self.assertEqual(ns.w, 3)
        with self.assertRaises(AttributeError):
            ns.z

    def test_attrset(self):
        ns1 = types.SimpleNamespace()
        ns2 = types.SimpleNamespace(x=1, y=2, w=3)
        ns1.a = 'spam'
        ns1.b = 'ham'
        ns2.z = 4
        ns2.theta = None

        self.assertEqual(ns1.__dict__, dict(a='spam', b='ham'))
        self.assertEqual(ns2.__dict__, dict(x=1, y=2, w=3, z=4, theta=None))

    def test_attrdel(self):
        ns1 = types.SimpleNamespace()
        ns2 = types.SimpleNamespace(x=1, y=2, w=3)

        with self.assertRaises(AttributeError):
            del ns1.spam
        with self.assertRaises(AttributeError):
            del ns2.spam

        del ns2.y
        self.assertEqual(vars(ns2), dict(w=3, x=1))
        ns2.y = 'spam'
        self.assertEqual(vars(ns2), dict(w=3, x=1, y='spam'))
        del ns2.y
        self.assertEqual(vars(ns2), dict(w=3, x=1))

        ns1.spam = 5
        self.assertEqual(vars(ns1), dict(spam=5))
        del ns1.spam
        self.assertEqual(vars(ns1), {})

    def test_repr(self):
        ns1 = types.SimpleNamespace(x=1, y=2, w=3)
        ns2 = types.SimpleNamespace()
        ns2.x = "spam"
        ns2._y = 5
        name = "namespace"

        self.assertEqual(repr(ns1), "{name}(x=1, y=2, w=3)".format(name=name))
        self.assertEqual(repr(ns2), "{name}(x='spam', _y=5)".format(name=name))

    def test_equal(self):
        ns1 = types.SimpleNamespace(x=1)
        ns2 = types.SimpleNamespace()
        ns2.x = 1

        self.assertEqual(types.SimpleNamespace(), types.SimpleNamespace())
        self.assertEqual(ns1, ns2)
        self.assertNotEqual(ns2, types.SimpleNamespace())

    def test_nested(self):
        ns1 = types.SimpleNamespace(a=1, b=2)
        ns2 = types.SimpleNamespace()
        ns3 = types.SimpleNamespace(x=ns1)
        ns2.spam = ns1
        ns2.ham = '?'
        ns2.spam = ns3

        self.assertEqual(vars(ns1), dict(a=1, b=2))
        self.assertEqual(vars(ns2), dict(spam=ns3, ham='?'))
        self.assertEqual(ns2.spam, ns3)
        self.assertEqual(vars(ns3), dict(x=ns1))
        self.assertEqual(ns3.x.a, 1)

    def test_recursive(self):
        ns1 = types.SimpleNamespace(c='cookie')
        ns2 = types.SimpleNamespace()
        ns3 = types.SimpleNamespace(x=1)
        ns1.spam = ns1
        ns2.spam = ns3
        ns3.spam = ns2

        self.assertEqual(ns1.spam, ns1)
        self.assertEqual(ns1.spam.spam, ns1)
        self.assertEqual(ns1.spam.spam, ns1.spam)
        self.assertEqual(ns2.spam, ns3)
        self.assertEqual(ns3.spam, ns2)
        self.assertEqual(ns2.spam.spam, ns2)

    def test_recursive_repr(self):
        ns1 = types.SimpleNamespace(c='cookie')
        ns2 = types.SimpleNamespace()
        ns3 = types.SimpleNamespace(x=1)
        ns1.spam = ns1
        ns2.spam = ns3
        ns3.spam = ns2
        name = "namespace"
        repr1 = "{name}(c='cookie', spam={name}(...))".format(name=name)
        repr2 = "{name}(spam={name}(x=1, spam={name}(...)))".format(name=name)

        self.assertEqual(repr(ns1), repr1)
        self.assertEqual(repr(ns2), repr2)

    def test_as_dict(self):
        ns = types.SimpleNamespace(spam='spamspamspam')

        with self.assertRaises(TypeError):
            len(ns)
        with self.assertRaises(TypeError):
            iter(ns)
        with self.assertRaises(TypeError):
            'spam' in ns
        with self.assertRaises(TypeError):
            ns['spam']

    def test_subclass(self):
        with torch._dynamo.error_on_graph_break(False):
            class Spam(types.SimpleNamespace):
                pass

        spam = Spam(ham=8, eggs=9)

        self.assertIs(type(spam), Spam)
        self.assertEqual(vars(spam), {'ham': 8, 'eggs': 9})

    def test_pickle(self):
        ns = types.SimpleNamespace(breakfast="spam", lunch="spam")

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            pname = "protocol {}".format(protocol)
            try:
                ns_pickled = pickle.dumps(ns, protocol)
            except TypeError as e:
                raise TypeError(pname) from e
            ns_roundtrip = pickle.loads(ns_pickled)

            self.assertEqual(ns, ns_roundtrip, pname)

    def test_replace(self):
        ns = types.SimpleNamespace(x=11, y=22)

        ns2 = copy.replace(ns)
        self.assertEqual(ns2, ns)
        self.assertIsNot(ns2, ns)
        self.assertIs(type(ns2), types.SimpleNamespace)
        self.assertEqual(vars(ns2), {'x': 11, 'y': 22})
        ns2.x = 3
        self.assertEqual(ns.x, 11)
        ns.x = 4
        self.assertEqual(ns2.x, 3)

        self.assertEqual(vars(copy.replace(ns, x=1)), {'x': 1, 'y': 22})
        self.assertEqual(vars(copy.replace(ns, y=2)), {'x': 4, 'y': 2})
        self.assertEqual(vars(copy.replace(ns, x=1, y=2)), {'x': 1, 'y': 2})

    def test_replace_subclass(self):
        with torch._dynamo.error_on_graph_break(False):
            class Spam(types.SimpleNamespace):
                pass

        spam = Spam(ham=8, eggs=9)
        spam2 = copy.replace(spam, ham=5)

        self.assertIs(type(spam2), Spam)
        self.assertEqual(vars(spam2), {'ham': 5, 'eggs': 9})

    def test_fake_namespace_compare(self):
        # Issue #24257: Incorrect use of PyObject_IsInstance() caused
        # SystemError.
        with torch._dynamo.error_on_graph_break(False):
            class FakeSimpleNamespace(str):
                __class__ = types.SimpleNamespace
        self.assertFalse(types.SimpleNamespace() == FakeSimpleNamespace())
        self.assertTrue(types.SimpleNamespace() != FakeSimpleNamespace())
        with self.assertRaises(TypeError):
            types.SimpleNamespace() < FakeSimpleNamespace()
        with self.assertRaises(TypeError):
            types.SimpleNamespace() <= FakeSimpleNamespace()
        with self.assertRaises(TypeError):
            types.SimpleNamespace() > FakeSimpleNamespace()
        with self.assertRaises(TypeError):
            types.SimpleNamespace() >= FakeSimpleNamespace()


class CoroutineTests(CPythonTestCase):
    def test_wrong_args(self):
        samples = [None, 1, object()]
        for sample in samples:
            with self.assertRaisesRegex(TypeError,
                                        'types.coroutine.*expects a callable'):
                types.coroutine(sample)

    def test_non_gen_values(self):
        @types.coroutine
        def foo():
            return 'spam'
        self.assertEqual(foo(), 'spam')

        with torch._dynamo.error_on_graph_break(False):
            class Awaitable:
                def __await__(self):
                    return ()
        aw = Awaitable()
        @types.coroutine
        def foo():
            return aw
        self.assertIs(aw, foo())

        # decorate foo second time
        foo = types.coroutine(foo)
        self.assertIs(aw, foo())

    def test_async_def(self):
        # Test that types.coroutine passes 'async def' coroutines
        # without modification

        async def foo(): pass
        foo_code = foo.__code__
        foo_flags = foo.__code__.co_flags
        decorated_foo = types.coroutine(foo)
        self.assertIs(foo, decorated_foo)
        self.assertEqual(foo.__code__.co_flags, foo_flags)
        self.assertIs(decorated_foo.__code__, foo_code)

        foo_coro = foo()
        def bar(): return foo_coro
        for _ in range(2):
            bar = types.coroutine(bar)
            coro = bar()
            self.assertIs(foo_coro, coro)
            self.assertEqual(coro.cr_code.co_flags, foo_flags)
            coro.close()

    def test_duck_coro(self):
        with torch._dynamo.error_on_graph_break(False):
            class CoroLike:
                def send(self): pass
                def throw(self): pass
                def close(self): pass
                def __await__(self): return self

        coro = CoroLike()
        @types.coroutine
        def foo():
            return coro
        self.assertIs(foo(), coro)
        self.assertIs(foo().__await__(), coro)

    def test_duck_corogen(self):
        with torch._dynamo.error_on_graph_break(False):
            class CoroGenLike:
                def send(self): pass
                def throw(self): pass
                def close(self): pass
                def __await__(self): return self
                def __iter__(self): return self
                def __next__(self): pass

        coro = CoroGenLike()
        @types.coroutine
        def foo():
            return coro
        self.assertIs(foo(), coro)
        self.assertIs(foo().__await__(), coro)

    def test_duck_gen(self):
        with torch._dynamo.error_on_graph_break(False):
            class GenLike:
                def send(self): pass
                def throw(self): pass
                def close(self): pass
                def __iter__(self): pass
                def __next__(self): pass

        # Setup generator mock object
        gen = unittest.mock.MagicMock(GenLike)
        gen.__iter__ = lambda gen: gen
        gen.__name__ = 'gen'
        gen.__qualname__ = 'test.gen'
        self.assertIsInstance(gen, collections.abc.Generator)
        self.assertIs(gen, iter(gen))

        @types.coroutine
        def foo(): return gen

        wrapper = foo()
        self.assertIsInstance(wrapper, types._GeneratorWrapper)
        self.assertIs(wrapper.__await__(), wrapper)
        # Wrapper proxies duck generators completely:
        self.assertIs(iter(wrapper), wrapper)

        self.assertIsInstance(wrapper, collections.abc.Coroutine)
        self.assertIsInstance(wrapper, collections.abc.Awaitable)

        self.assertIs(wrapper.__qualname__, gen.__qualname__)
        self.assertIs(wrapper.__name__, gen.__name__)

        # Test AttributeErrors
        for name in {'gi_running', 'gi_frame', 'gi_code', 'gi_yieldfrom',
                     'cr_running', 'cr_frame', 'cr_code', 'cr_await'}:
            with self.assertRaises(AttributeError):
                getattr(wrapper, name)

        # Test attributes pass-through
        gen.gi_running = object()
        gen.gi_frame = object()
        gen.gi_code = object()
        gen.gi_yieldfrom = object()
        self.assertIs(wrapper.gi_running, gen.gi_running)
        self.assertIs(wrapper.gi_frame, gen.gi_frame)
        self.assertIs(wrapper.gi_code, gen.gi_code)
        self.assertIs(wrapper.gi_yieldfrom, gen.gi_yieldfrom)
        self.assertIs(wrapper.cr_running, gen.gi_running)
        self.assertIs(wrapper.cr_frame, gen.gi_frame)
        self.assertIs(wrapper.cr_code, gen.gi_code)
        self.assertIs(wrapper.cr_await, gen.gi_yieldfrom)

        wrapper.close()
        gen.close.assert_called_once_with()

        wrapper.send(1)
        gen.send.assert_called_once_with(1)
        gen.reset_mock()

        next(wrapper)
        gen.__next__.assert_called_once_with()
        gen.reset_mock()

        wrapper.throw(1, 2, 3)
        gen.throw.assert_called_once_with(1, 2, 3)
        gen.reset_mock()

        wrapper.throw(1, 2)
        gen.throw.assert_called_once_with(1, 2)
        gen.reset_mock()

        wrapper.throw(1)
        gen.throw.assert_called_once_with(1)
        gen.reset_mock()

        # Test exceptions propagation
        error = Exception()
        gen.throw.side_effect = error
        try:
            wrapper.throw(1)
        except Exception as ex:
            self.assertIs(ex, error)
        else:
            self.fail('wrapper did not propagate an exception')

        # Test invalid args
        gen.reset_mock()
        with self.assertRaises(TypeError):
            wrapper.throw()
        self.assertFalse(gen.throw.called)
        with self.assertRaises(TypeError):
            wrapper.close(1)
        self.assertFalse(gen.close.called)
        with self.assertRaises(TypeError):
            wrapper.send()
        self.assertFalse(gen.send.called)

        # Test that we do not double wrap
        @types.coroutine
        def bar(): return wrapper
        self.assertIs(wrapper, bar())

        # Test weakrefs support
        ref = weakref.ref(wrapper)
        self.assertIs(ref(), wrapper)

    def test_duck_functional_gen(self):
        with torch._dynamo.error_on_graph_break(False):
            class Generator:
                """Emulates the following generator (very clumsy):

                  def gen(fut):
                      result = yield fut
                      return result * 2
                """
                def __init__(self, fut):
                    self._i = 0
                    self._fut = fut
                def __iter__(self):
                    return self
                def __next__(self):
                    return self.send(None)
                def send(self, v):
                    try:
                        if self._i == 0:
                            assert v is None
                            return self._fut
                        if self._i == 1:
                            raise StopIteration(v * 2)
                        if self._i > 1:
                            raise StopIteration
                    finally:
                        self._i += 1
                def throw(self, tp, *exc):
                    self._i = 100
                    if tp is not GeneratorExit:
                        raise tp
                def close(self):
                    self.throw(GeneratorExit)

        @types.coroutine
        def foo(): return Generator('spam')

        wrapper = foo()
        self.assertIsInstance(wrapper, types._GeneratorWrapper)

        async def corofunc():
            return await foo() + 100
        coro = corofunc()

        self.assertEqual(coro.send(None), 'spam')
        try:
            coro.send(20)
        except StopIteration as ex:
            self.assertEqual(ex.args[0], 140)
        else:
            self.fail('StopIteration was expected')

    def test_gen(self):
        def gen_func():
            yield 1
            return (yield 2)
        gen = gen_func()
        @types.coroutine
        def foo(): return gen
        wrapper = foo()
        self.assertIsInstance(wrapper, types._GeneratorWrapper)
        self.assertIs(wrapper.__await__(), gen)

        for name in ('__name__', '__qualname__', 'gi_code',
                     'gi_running', 'gi_frame'):
            self.assertIs(getattr(foo(), name),
                          getattr(gen, name))
        self.assertIs(foo().cr_code, gen.gi_code)

        self.assertEqual(next(wrapper), 1)
        self.assertEqual(wrapper.send(None), 2)
        with self.assertRaisesRegex(StopIteration, 'spam'):
            wrapper.send('spam')

        gen = gen_func()
        wrapper = foo()
        wrapper.send(None)
        with self.assertRaisesRegex(Exception, 'ham'):
            wrapper.throw(Exception('ham'))

        # decorate foo second time
        foo = types.coroutine(foo)
        self.assertIs(foo().__await__(), gen)

    def test_returning_itercoro(self):
        @types.coroutine
        def gen():
            yield

        gencoro = gen()

        @types.coroutine
        def foo():
            return gencoro

        self.assertIs(foo(), gencoro)

        # decorate foo second time
        foo = types.coroutine(foo)
        self.assertIs(foo(), gencoro)

    def test_genfunc(self):
        def gen(): yield
        self.assertIs(types.coroutine(gen), gen)
        self.assertIs(types.coroutine(types.coroutine(gen)), gen)

        self.assertTrue(gen.__code__.co_flags & inspect.CO_ITERABLE_COROUTINE)
        self.assertFalse(gen.__code__.co_flags & inspect.CO_COROUTINE)

        g = gen()
        self.assertTrue(g.gi_code.co_flags & inspect.CO_ITERABLE_COROUTINE)
        self.assertFalse(g.gi_code.co_flags & inspect.CO_COROUTINE)

        self.assertIs(types.coroutine(gen), gen)

    def test_wrapper_object(self):
        def gen():
            yield
        @types.coroutine
        def coro():
            return gen()

        wrapper = coro()
        self.assertIn('GeneratorWrapper', repr(wrapper))
        self.assertEqual(repr(wrapper), str(wrapper))
        self.assertTrue(set(dir(wrapper)).issuperset({
            '__await__', '__iter__', '__next__', 'cr_code', 'cr_running',
            'cr_frame', 'gi_code', 'gi_frame', 'gi_running', 'send',
            'close', 'throw'}))


class FunctionTests(CPythonTestCase):
    def test_function_type_defaults(self):
        def ex(a, /, b, *, c):
            return a + b + c

        func = types.FunctionType(
            ex.__code__, {}, "func", (1, 2), None, {'c': 3},
        )

        self.assertEqual(func(), 6)
        self.assertEqual(func.__defaults__, (1, 2))
        self.assertEqual(func.__kwdefaults__, {'c': 3})

        func = types.FunctionType(
            ex.__code__, {}, "func", None, None, None,
        )
        self.assertEqual(func.__defaults__, None)
        self.assertEqual(func.__kwdefaults__, None)

    def test_function_type_wrong_defaults(self):
        def ex(a, /, b, *, c):
            return a + b + c

        with self.assertRaisesRegex(TypeError, 'arg 4'):
            types.FunctionType(
                ex.__code__, {}, "func", 1, None, {'c': 3},
            )
        with self.assertRaisesRegex(TypeError, 'arg 6'):
            types.FunctionType(
                ex.__code__, {}, "func", None, None, 3,
            )


class SubinterpreterTests(CPythonTestCase):

    @classmethod
    def setUpClass(cls):
        global interpreters
        try:
            from test.support import interpreters
        except ModuleNotFoundError:
            raise unittest.SkipTest('subinterpreters required')
        import test.support.interpreters.channels
        super().setUpClass()

    @cpython_only
    @no_rerun('channels (and queues) might have a refleak; see gh-122199')
    def test_static_types_inherited_slots(self):
        rch, sch = interpreters.channels.create()

        slots = []
        script = ''
        for cls in iter_builtin_types():
            for slot, own in iter_slot_wrappers(cls):
                slots.append((cls, slot, own))
                script += textwrap.dedent(f"""
                    text = repr({cls.__name__}.{slot})
                    sch.send_nowait(({cls.__name__!r}, {slot!r}, text))
                    """)

        exec(script)
        all_expected = []
        for cls, slot, _ in slots:
            result = rch.recv()
            assert result == (cls.__name__, slot, result[-1]), (cls, slot, result)
            all_expected.append(result)

        interp = interpreters.create()
        interp.exec('from test.support import interpreters')
        interp.prepare_main(sch=sch)
        interp.exec(script)

        for i, (cls, slot, _) in enumerate(slots):
            with self.subTest(cls=cls, slot=slot):
                expected = all_expected[i]
                result = rch.recv()
                self.assertEqual(result, expected)


if __name__ == '__main__':
    run_tests()
