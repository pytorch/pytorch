# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    TEST_WITH_TORCHDYNAMO,
    run_tests,
    skipIfTorchDynamo,
)

if TEST_WITH_TORCHDYNAMO:
    __TestCase = CPythonTestCase
else:
    __TestCase = unittest.TestCase

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

import sys
import time

import unittest
from unittest import mock
from test import support
from test_grammar import VALID_UNDERSCORE_LITERALS, INVALID_UNDERSCORE_LITERALS

try:
    import _pylong
except ImportError:
    _pylong = None

L = [
        ('0', 0),
        ('1', 1),
        ('9', 9),
        ('10', 10),
        ('99', 99),
        ('100', 100),
        ('314', 314),
        (' 314', 314),
        ('314 ', 314),
        ('  \t\t  314  \t\t  ', 314),
        (repr(sys.maxsize), sys.maxsize),
        ('  1x', ValueError),
        ('  1  ', 1),
        ('  1\02  ', ValueError),
        ('', ValueError),
        (' ', ValueError),
        ('  \t\t  ', ValueError),
        ("\u0200", ValueError)
]

class IntSubclass(int):
    pass

class IntTestCases(__TestCase):

    def test_basic(self):
        self.assertEqual(int(314), 314)
        self.assertEqual(int(3.14), 3)
        # Check that conversion from float truncates towards zero
        self.assertEqual(int(-3.14), -3)
        self.assertEqual(int(3.9), 3)
        self.assertEqual(int(-3.9), -3)
        self.assertEqual(int(3.5), 3)
        self.assertEqual(int(-3.5), -3)
        self.assertEqual(int("-3"), -3)
        self.assertEqual(int(" -3 "), -3)
        self.assertEqual(int("\N{EM SPACE}-3\N{EN SPACE}"), -3)
        # Different base:
        self.assertEqual(int("10",16), 16)
        # Test conversion from strings and various anomalies
        for s, v in L:
            for sign in "", "+", "-":
                for prefix in "", " ", "\t", "  \t\t  ":
                    ss = prefix + sign + s
                    vv = v
                    if sign == "-" and v is not ValueError:
                        vv = -v
                    try:
                        self.assertEqual(int(ss), vv)
                    except ValueError:
                        pass

        s = repr(-1-sys.maxsize)
        x = int(s)
        self.assertEqual(x+1, -sys.maxsize)
        self.assertIsInstance(x, int)
        # should return int
        self.assertEqual(int(s[1:]), sys.maxsize+1)

        # should return int
        x = int(1e100)
        self.assertIsInstance(x, int)
        x = int(-1e100)
        self.assertIsInstance(x, int)


        # SF bug 434186:  0x80000000/2 != 0x80000000>>1.
        # Worked by accident in Windows release build, but failed in debug build.
        # Failed in all Linux builds.
        x = -1-sys.maxsize
        self.assertEqual(x >> 1, x//2)

        x = int('1' * 600)
        self.assertIsInstance(x, int)


        self.assertRaises(TypeError, int, 1, 12)
        self.assertRaises(TypeError, int, "10", 2, 1)

        self.assertEqual(int('0o123', 0), 83)
        self.assertEqual(int('0x123', 16), 291)

        # Bug 1679: "0x" is not a valid hex literal
        self.assertRaises(ValueError, int, "0x", 16)
        self.assertRaises(ValueError, int, "0x", 0)

        self.assertRaises(ValueError, int, "0o", 8)
        self.assertRaises(ValueError, int, "0o", 0)

        self.assertRaises(ValueError, int, "0b", 2)
        self.assertRaises(ValueError, int, "0b", 0)

        # SF bug 1334662: int(string, base) wrong answers
        # Various representations of 2**32 evaluated to 0
        # rather than 2**32 in previous versions

        self.assertEqual(int('100000000000000000000000000000000', 2), 4294967296)
        self.assertEqual(int('102002022201221111211', 3), 4294967296)
        self.assertEqual(int('10000000000000000', 4), 4294967296)
        self.assertEqual(int('32244002423141', 5), 4294967296)
        self.assertEqual(int('1550104015504', 6), 4294967296)
        self.assertEqual(int('211301422354', 7), 4294967296)
        self.assertEqual(int('40000000000', 8), 4294967296)
        self.assertEqual(int('12068657454', 9), 4294967296)
        self.assertEqual(int('4294967296', 10), 4294967296)
        self.assertEqual(int('1904440554', 11), 4294967296)
        self.assertEqual(int('9ba461594', 12), 4294967296)
        self.assertEqual(int('535a79889', 13), 4294967296)
        self.assertEqual(int('2ca5b7464', 14), 4294967296)
        self.assertEqual(int('1a20dcd81', 15), 4294967296)
        self.assertEqual(int('100000000', 16), 4294967296)
        self.assertEqual(int('a7ffda91', 17), 4294967296)
        self.assertEqual(int('704he7g4', 18), 4294967296)
        self.assertEqual(int('4f5aff66', 19), 4294967296)
        self.assertEqual(int('3723ai4g', 20), 4294967296)
        self.assertEqual(int('281d55i4', 21), 4294967296)
        self.assertEqual(int('1fj8b184', 22), 4294967296)
        self.assertEqual(int('1606k7ic', 23), 4294967296)
        self.assertEqual(int('mb994ag', 24), 4294967296)
        self.assertEqual(int('hek2mgl', 25), 4294967296)
        self.assertEqual(int('dnchbnm', 26), 4294967296)
        self.assertEqual(int('b28jpdm', 27), 4294967296)
        self.assertEqual(int('8pfgih4', 28), 4294967296)
        self.assertEqual(int('76beigg', 29), 4294967296)
        self.assertEqual(int('5qmcpqg', 30), 4294967296)
        self.assertEqual(int('4q0jto4', 31), 4294967296)
        self.assertEqual(int('4000000', 32), 4294967296)
        self.assertEqual(int('3aokq94', 33), 4294967296)
        self.assertEqual(int('2qhxjli', 34), 4294967296)
        self.assertEqual(int('2br45qb', 35), 4294967296)
        self.assertEqual(int('1z141z4', 36), 4294967296)

        # tests with base 0
        # this fails on 3.0, but in 2.x the old octal syntax is allowed
        self.assertEqual(int(' 0o123  ', 0), 83)
        self.assertEqual(int(' 0o123  ', 0), 83)
        self.assertEqual(int('000', 0), 0)
        self.assertEqual(int('0o123', 0), 83)
        self.assertEqual(int('0x123', 0), 291)
        self.assertEqual(int('0b100', 0), 4)
        self.assertEqual(int(' 0O123   ', 0), 83)
        self.assertEqual(int(' 0X123  ', 0), 291)
        self.assertEqual(int(' 0B100 ', 0), 4)
        with self.assertRaises(ValueError):
            int('010', 0)

        # without base still base 10
        self.assertEqual(int('0123'), 123)
        self.assertEqual(int('0123', 10), 123)

        # tests with prefix and base != 0
        self.assertEqual(int('0x123', 16), 291)
        self.assertEqual(int('0o123', 8), 83)
        self.assertEqual(int('0b100', 2), 4)
        self.assertEqual(int('0X123', 16), 291)
        self.assertEqual(int('0O123', 8), 83)
        self.assertEqual(int('0B100', 2), 4)

        # the code has special checks for the first character after the
        #  type prefix
        self.assertRaises(ValueError, int, '0b2', 2)
        self.assertRaises(ValueError, int, '0b02', 2)
        self.assertRaises(ValueError, int, '0B2', 2)
        self.assertRaises(ValueError, int, '0B02', 2)
        self.assertRaises(ValueError, int, '0o8', 8)
        self.assertRaises(ValueError, int, '0o08', 8)
        self.assertRaises(ValueError, int, '0O8', 8)
        self.assertRaises(ValueError, int, '0O08', 8)
        self.assertRaises(ValueError, int, '0xg', 16)
        self.assertRaises(ValueError, int, '0x0g', 16)
        self.assertRaises(ValueError, int, '0Xg', 16)
        self.assertRaises(ValueError, int, '0X0g', 16)

        # SF bug 1334662: int(string, base) wrong answers
        # Checks for proper evaluation of 2**32 + 1
        self.assertEqual(int('100000000000000000000000000000001', 2), 4294967297)
        self.assertEqual(int('102002022201221111212', 3), 4294967297)
        self.assertEqual(int('10000000000000001', 4), 4294967297)
        self.assertEqual(int('32244002423142', 5), 4294967297)
        self.assertEqual(int('1550104015505', 6), 4294967297)
        self.assertEqual(int('211301422355', 7), 4294967297)
        self.assertEqual(int('40000000001', 8), 4294967297)
        self.assertEqual(int('12068657455', 9), 4294967297)
        self.assertEqual(int('4294967297', 10), 4294967297)
        self.assertEqual(int('1904440555', 11), 4294967297)
        self.assertEqual(int('9ba461595', 12), 4294967297)
        self.assertEqual(int('535a7988a', 13), 4294967297)
        self.assertEqual(int('2ca5b7465', 14), 4294967297)
        self.assertEqual(int('1a20dcd82', 15), 4294967297)
        self.assertEqual(int('100000001', 16), 4294967297)
        self.assertEqual(int('a7ffda92', 17), 4294967297)
        self.assertEqual(int('704he7g5', 18), 4294967297)
        self.assertEqual(int('4f5aff67', 19), 4294967297)
        self.assertEqual(int('3723ai4h', 20), 4294967297)
        self.assertEqual(int('281d55i5', 21), 4294967297)
        self.assertEqual(int('1fj8b185', 22), 4294967297)
        self.assertEqual(int('1606k7id', 23), 4294967297)
        self.assertEqual(int('mb994ah', 24), 4294967297)
        self.assertEqual(int('hek2mgm', 25), 4294967297)
        self.assertEqual(int('dnchbnn', 26), 4294967297)
        self.assertEqual(int('b28jpdn', 27), 4294967297)
        self.assertEqual(int('8pfgih5', 28), 4294967297)
        self.assertEqual(int('76beigh', 29), 4294967297)
        self.assertEqual(int('5qmcpqh', 30), 4294967297)
        self.assertEqual(int('4q0jto5', 31), 4294967297)
        self.assertEqual(int('4000001', 32), 4294967297)
        self.assertEqual(int('3aokq95', 33), 4294967297)
        self.assertEqual(int('2qhxjlj', 34), 4294967297)
        self.assertEqual(int('2br45qc', 35), 4294967297)
        self.assertEqual(int('1z141z5', 36), 4294967297)

    def test_invalid_signs(self):
        with self.assertRaises(ValueError):
            int('+')
        with self.assertRaises(ValueError):
            int('-')
        with self.assertRaises(ValueError):
            int('- 1')
        with self.assertRaises(ValueError):
            int('+ 1')
        with self.assertRaises(ValueError):
            int(' + 1 ')

    def test_unicode(self):
        self.assertEqual(int("१२३४५६७८९०1234567890"), 12345678901234567890)
        self.assertEqual(int('١٢٣٤٥٦٧٨٩٠'), 1234567890)
        self.assertEqual(int("१२३४५६७८९०1234567890", 0), 12345678901234567890)
        self.assertEqual(int('١٢٣٤٥٦٧٨٩٠', 0), 1234567890)

    def test_underscores(self):
        for lit in VALID_UNDERSCORE_LITERALS:
            if any(ch in lit for ch in '.eEjJ'):
                continue
            self.assertEqual(int(lit, 0), eval(lit))
            self.assertEqual(int(lit, 0), int(lit.replace('_', ''), 0))
        for lit in INVALID_UNDERSCORE_LITERALS:
            if any(ch in lit for ch in '.eEjJ'):
                continue
            self.assertRaises(ValueError, int, lit, 0)
        # Additional test cases with bases != 0, only for the constructor:
        self.assertEqual(int("1_00", 3), 9)
        self.assertEqual(int("0_100"), 100)  # not valid as a literal!
        self.assertEqual(int(b"1_00"), 100)  # byte underscore
        self.assertRaises(ValueError, int, "_100")
        self.assertRaises(ValueError, int, "+_100")
        self.assertRaises(ValueError, int, "1__00")
        self.assertRaises(ValueError, int, "100_")

    @support.cpython_only
    def test_small_ints(self):
        # Bug #3236: Return small longs from PyLong_FromString
        self.assertIs(int('10'), 10)
        self.assertIs(int('-1'), -1)
        self.assertIs(int(b'10'), 10)
        self.assertIs(int(b'-1'), -1)

    def test_no_args(self):
        self.assertEqual(int(), 0)

    def test_keyword_args(self):
        # Test invoking int() using keyword arguments.
        self.assertEqual(int('100', base=2), 4)
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            int(x=1.2)
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            int(x='100', base=2)
        self.assertRaises(TypeError, int, base=10)
        self.assertRaises(TypeError, int, base=0)

    def test_int_base_limits(self):
        """Testing the supported limits of the int() base parameter."""
        self.assertEqual(int('0', 5), 0)
        with self.assertRaises(ValueError):
            int('0', 1)
        with self.assertRaises(ValueError):
            int('0', 37)
        with self.assertRaises(ValueError):
            int('0', -909)  # An old magic value base from Python 2.
        with self.assertRaises(ValueError):
            int('0', base=0-(2**234))
        with self.assertRaises(ValueError):
            int('0', base=2**234)
        # Bases 2 through 36 are supported.
        for base in range(2,37):
            self.assertEqual(int('0', base=base), 0)

    def test_int_base_bad_types(self):
        """Not integer types are not valid bases; issue16772."""
        with self.assertRaises(TypeError):
            int('0', 5.5)
        with self.assertRaises(TypeError):
            int('0', 5.0)

    def test_int_base_indexable(self):
        class MyIndexable(object):
            def __init__(self, value):
                self.value = value
            def __index__(self):
                return self.value

        # Check out of range bases.
        for base in 2**100, -2**100, 1, 37:
            with self.assertRaises(ValueError):
                int('43', base)

        # Check in-range bases.
        self.assertEqual(int('101', base=MyIndexable(2)), 5)
        self.assertEqual(int('101', base=MyIndexable(10)), 101)
        self.assertEqual(int('101', base=MyIndexable(36)), 1 + 36**2)

    def test_non_numeric_input_types(self):
        # Test possible non-numeric types for the argument x, including
        # subclasses of the explicitly documented accepted types.
        class CustomStr(str): pass
        class CustomBytes(bytes): pass
        class CustomByteArray(bytearray): pass

        factories = [
            bytes,
            bytearray,
            lambda b: CustomStr(b.decode()),
            CustomBytes,
            CustomByteArray,
            memoryview,
        ]
        try:
            from array import array
        except ImportError:
            pass
        else:
            factories.append(lambda b: array('B', b))

        for f in factories:
            x = f(b'100')
            with self.subTest(type(x)):
                self.assertEqual(int(x), 100)
                if isinstance(x, (str, bytes, bytearray)):
                    self.assertEqual(int(x, 2), 4)
                else:
                    msg = "can't convert non-string"
                    with self.assertRaisesRegex(TypeError, msg):
                        int(x, 2)
                with self.assertRaisesRegex(ValueError, 'invalid literal'):
                    int(f(b'A' * 0x10))

    def test_int_memoryview(self):
        self.assertEqual(int(memoryview(b'123')[1:3]), 23)
        self.assertEqual(int(memoryview(b'123\x00')[1:3]), 23)
        self.assertEqual(int(memoryview(b'123 ')[1:3]), 23)
        self.assertEqual(int(memoryview(b'123A')[1:3]), 23)
        self.assertEqual(int(memoryview(b'1234')[1:3]), 23)

    def test_string_float(self):
        self.assertRaises(ValueError, int, '1.2')

    def test_intconversion(self):
        # Test __int__()
        class ClassicMissingMethods:
            pass
        self.assertRaises(TypeError, int, ClassicMissingMethods())

        class MissingMethods(object):
            pass
        self.assertRaises(TypeError, int, MissingMethods())

        class Foo0:
            def __int__(self):
                return 42

        self.assertEqual(int(Foo0()), 42)

        class Classic:
            pass
        for base in (object, Classic):
            class IntOverridesTrunc(base):
                def __int__(self):
                    return 42
                def __trunc__(self):
                    return -12
            self.assertEqual(int(IntOverridesTrunc()), 42)

            class JustTrunc(base):
                def __trunc__(self):
                    return 42
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(int(JustTrunc()), 42)

            class ExceptionalTrunc(base):
                def __trunc__(self):
                    1 / 0
            with self.assertRaises(ZeroDivisionError), \
                 self.assertWarns(DeprecationWarning):
                int(ExceptionalTrunc())

            for trunc_result_base in (object, Classic):
                class Index(trunc_result_base):
                    def __index__(self):
                        return 42

                class TruncReturnsNonInt(base):
                    def __trunc__(self):
                        return Index()
                with self.assertWarns(DeprecationWarning):
                    self.assertEqual(int(TruncReturnsNonInt()), 42)

                class Intable(trunc_result_base):
                    def __int__(self):
                        return 42

                class TruncReturnsNonIndex(base):
                    def __trunc__(self):
                        return Intable()
                with self.assertWarns(DeprecationWarning):
                    self.assertEqual(int(TruncReturnsNonInt()), 42)

                class NonIntegral(trunc_result_base):
                    def __trunc__(self):
                        # Check that we avoid infinite recursion.
                        return NonIntegral()

                class TruncReturnsNonIntegral(base):
                    def __trunc__(self):
                        return NonIntegral()
                try:
                    with self.assertWarns(DeprecationWarning):
                        int(TruncReturnsNonIntegral())
                except TypeError as e:
                    self.assertEqual(str(e),
                                      "__trunc__ returned non-Integral"
                                      " (type NonIntegral)")
                else:
                    self.fail("Failed to raise TypeError with %s" %
                              ((base, trunc_result_base),))

                # Regression test for bugs.python.org/issue16060.
                class BadInt(trunc_result_base):
                    def __int__(self):
                        return 42.0

                class TruncReturnsBadInt(base):
                    def __trunc__(self):
                        return BadInt()

                with self.assertRaises(TypeError), \
                     self.assertWarns(DeprecationWarning):
                    int(TruncReturnsBadInt())

    def test_int_subclass_with_index(self):
        class MyIndex(int):
            def __index__(self):
                return 42

        class BadIndex(int):
            def __index__(self):
                return 42.0

        my_int = MyIndex(7)
        self.assertEqual(my_int, 7)
        self.assertEqual(int(my_int), 7)

        self.assertEqual(int(BadIndex()), 0)

    def test_int_subclass_with_int(self):
        class MyInt(int):
            def __int__(self):
                return 42

        class BadInt(int):
            def __int__(self):
                return 42.0

        my_int = MyInt(7)
        self.assertEqual(my_int, 7)
        self.assertEqual(int(my_int), 42)

        my_int = BadInt(7)
        self.assertEqual(my_int, 7)
        self.assertRaises(TypeError, int, my_int)

    def test_int_returns_int_subclass(self):
        class BadIndex:
            def __index__(self):
                return True

        class BadIndex2(int):
            def __index__(self):
                return True

        class BadInt:
            def __int__(self):
                return True

        class BadInt2(int):
            def __int__(self):
                return True

        class TruncReturnsBadIndex:
            def __trunc__(self):
                return BadIndex()

        class TruncReturnsBadInt:
            def __trunc__(self):
                return BadInt()

        class TruncReturnsIntSubclass:
            def __trunc__(self):
                return True

        bad_int = BadIndex()
        with self.assertWarns(DeprecationWarning):
            n = int(bad_int)
        self.assertEqual(n, 1)
        self.assertIs(type(n), int)

        bad_int = BadIndex2()
        n = int(bad_int)
        self.assertEqual(n, 0)
        self.assertIs(type(n), int)

        bad_int = BadInt()
        with self.assertWarns(DeprecationWarning):
            n = int(bad_int)
        self.assertEqual(n, 1)
        self.assertIs(type(n), int)

        bad_int = BadInt2()
        with self.assertWarns(DeprecationWarning):
            n = int(bad_int)
        self.assertEqual(n, 1)
        self.assertIs(type(n), int)

        bad_int = TruncReturnsBadIndex()
        with self.assertWarns(DeprecationWarning):
            n = int(bad_int)
        self.assertEqual(n, 1)
        self.assertIs(type(n), int)

        bad_int = TruncReturnsBadInt()
        with self.assertWarns(DeprecationWarning):
            self.assertRaises(TypeError, int, bad_int)

        good_int = TruncReturnsIntSubclass()
        with self.assertWarns(DeprecationWarning):
            n = int(good_int)
        self.assertEqual(n, 1)
        self.assertIs(type(n), int)
        with self.assertWarns(DeprecationWarning):
            n = IntSubclass(good_int)
        self.assertEqual(n, 1)
        self.assertIs(type(n), IntSubclass)

    def test_error_message(self):
        def check(s, base=None):
            with self.assertRaises(ValueError,
                                   msg="int(%r, %r)" % (s, base)) as cm:
                if base is None:
                    int(s)
                else:
                    int(s, base)
            self.assertEqual(cm.exception.args[0],
                "invalid literal for int() with base %d: %r" %
                (10 if base is None else base, s))

        check('\xbd')
        check('123\xbd')
        check('  123 456  ')

        check('123\x00')
        # SF bug 1545497: embedded NULs were not detected with explicit base
        check('123\x00', 10)
        check('123\x00 245', 20)
        check('123\x00 245', 16)
        check('123\x00245', 20)
        check('123\x00245', 16)
        # byte string with embedded NUL
        check(b'123\x00')
        check(b'123\x00', 10)
        # non-UTF-8 byte string
        check(b'123\xbd')
        check(b'123\xbd', 10)
        # lone surrogate in Unicode string
        check('123\ud800')
        check('123\ud800', 10)

    def test_issue31619(self):
        self.assertEqual(int('1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1_0_1', 2),
                         0b1010101010101010101010101010101)
        self.assertEqual(int('1_2_3_4_5_6_7_0_1_2_3', 8), 0o12345670123)
        self.assertEqual(int('1_2_3_4_5_6_7_8_9', 16), 0x123456789)
        self.assertEqual(int('1_2_3_4_5_6_7', 32), 1144132807)


class IntStrDigitLimitsTests(__TestCase):

    int_class = int  # Override this in subclasses to reuse the suite.

    def setUp(self):
        super().setUp()
        self._previous_limit = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(2048)

    def tearDown(self):
        sys.set_int_max_str_digits(self._previous_limit)
        super().tearDown()

    def test_disabled_limit(self):
        self.assertGreater(sys.get_int_max_str_digits(), 0)
        self.assertLess(sys.get_int_max_str_digits(), 20_000)
        with support.adjust_int_max_str_digits(0):
            self.assertEqual(sys.get_int_max_str_digits(), 0)
            i = self.int_class('1' * 20_000)
            str(i)
        self.assertGreater(sys.get_int_max_str_digits(), 0)

    def test_max_str_digits_edge_cases(self):
        """Ignore the +/- sign and space padding."""
        int_class = self.int_class
        maxdigits = sys.get_int_max_str_digits()

        int_class('1' * maxdigits)
        int_class(' ' + '1' * maxdigits)
        int_class('1' * maxdigits + ' ')
        int_class('+' + '1' * maxdigits)
        int_class('-' + '1' * maxdigits)
        self.assertEqual(len(str(10 ** (maxdigits - 1))), maxdigits)

    def check(self, i, base=None):
        with self.assertRaises(ValueError):
            if base is None:
                self.int_class(i)
            else:
                self.int_class(i, base)

    def test_max_str_digits(self):
        maxdigits = sys.get_int_max_str_digits()

        self.check('1' * (maxdigits + 1))
        self.check(' ' + '1' * (maxdigits + 1))
        self.check('1' * (maxdigits + 1) + ' ')
        self.check('+' + '1' * (maxdigits + 1))
        self.check('-' + '1' * (maxdigits + 1))
        self.check('1' * (maxdigits + 1))

        i = 10 ** maxdigits
        with self.assertRaises(ValueError):
            str(i)

    def test_denial_of_service_prevented_int_to_str(self):
        """Regression test: ensure we fail before performing O(N**2) work."""
        maxdigits = sys.get_int_max_str_digits()
        assert maxdigits < 50_000, maxdigits  # A test prerequisite.

        huge_int = int(f'0x{"c"*65_000}', base=16)  # 78268 decimal digits.
        digits = 78_268
        with (
                support.adjust_int_max_str_digits(digits),
                support.CPUStopwatch() as sw_convert):
            huge_decimal = str(huge_int)
        self.assertEqual(len(huge_decimal), digits)
        # Ensuring that we chose a slow enough conversion to measure.
        # It takes 0.1 seconds on a Zen based cloud VM in an opt build.
        # Some OSes have a low res 1/64s timer, skip if hard to measure.
        if sw_convert.seconds < sw_convert.clock_info.resolution * 2:
            raise unittest.SkipTest('"slow" conversion took only '
                                    f'{sw_convert.seconds} seconds.')

        # We test with the limit almost at the size needed to check performance.
        # The performant limit check is slightly fuzzy, give it a some room.
        with support.adjust_int_max_str_digits(int(.995 * digits)):
            with (
                    self.assertRaises(ValueError) as err,
                    support.CPUStopwatch() as sw_fail_huge):
                str(huge_int)
        self.assertIn('conversion', str(err.exception))
        self.assertLessEqual(sw_fail_huge.seconds, sw_convert.seconds/2)

        # Now we test that a conversion that would take 30x as long also fails
        # in a similarly fast fashion.
        extra_huge_int = int(f'0x{"c"*500_000}', base=16)  # 602060 digits.
        with (
                self.assertRaises(ValueError) as err,
                support.CPUStopwatch() as sw_fail_extra_huge):
            # If not limited, 8 seconds said Zen based cloud VM.
            str(extra_huge_int)
        self.assertIn('conversion', str(err.exception))
        self.assertLess(sw_fail_extra_huge.seconds, sw_convert.seconds/2)

    def test_denial_of_service_prevented_str_to_int(self):
        """Regression test: ensure we fail before performing O(N**2) work."""
        maxdigits = sys.get_int_max_str_digits()
        assert maxdigits < 100_000, maxdigits  # A test prerequisite.

        digits = 133700
        huge = '8'*digits
        with (
                support.adjust_int_max_str_digits(digits),
                support.CPUStopwatch() as sw_convert):
            int(huge)
        # Ensuring that we chose a slow enough conversion to measure.
        # It takes 0.1 seconds on a Zen based cloud VM in an opt build.
        # Some OSes have a low res 1/64s timer, skip if hard to measure.
        if sw_convert.seconds < sw_convert.clock_info.resolution * 2:
            raise unittest.SkipTest('"slow" conversion took only '
                                    f'{sw_convert.seconds} seconds.')

        with support.adjust_int_max_str_digits(digits - 1):
            with (
                    self.assertRaises(ValueError) as err,
                    support.CPUStopwatch() as sw_fail_huge):
                int(huge)
        self.assertIn('conversion', str(err.exception))
        self.assertLessEqual(sw_fail_huge.seconds, sw_convert.seconds/2)

        # Now we test that a conversion that would take 30x as long also fails
        # in a similarly fast fashion.
        extra_huge = '7'*1_200_000
        with (
                self.assertRaises(ValueError) as err,
                support.CPUStopwatch() as sw_fail_extra_huge):
            # If not limited, 8 seconds in the Zen based cloud VM.
            int(extra_huge)
        self.assertIn('conversion', str(err.exception))
        self.assertLessEqual(sw_fail_extra_huge.seconds, sw_convert.seconds/2)

    def test_power_of_two_bases_unlimited(self):
        """The limit does not apply to power of 2 bases."""
        maxdigits = sys.get_int_max_str_digits()

        for base in (2, 4, 8, 16, 32):
            with self.subTest(base=base):
                self.int_class('1' * (maxdigits + 1), base)
                assert maxdigits < 100_000
                self.int_class('1' * 100_000, base)

    def test_underscores_ignored(self):
        maxdigits = sys.get_int_max_str_digits()

        triples = maxdigits // 3
        s = '111' * triples
        s_ = '1_11' * triples
        self.int_class(s)  # succeeds
        self.int_class(s_)  # succeeds
        self.check(f'{s}111')
        self.check(f'{s_}_111')

    def test_sign_not_counted(self):
        int_class = self.int_class
        max_digits = sys.get_int_max_str_digits()
        s = '5' * max_digits
        i = int_class(s)
        pos_i = int_class(f'+{s}')
        assert i == pos_i
        neg_i = int_class(f'-{s}')
        assert -pos_i == neg_i
        str(pos_i)
        str(neg_i)

    def _other_base_helper(self, base):
        int_class = self.int_class
        max_digits = sys.get_int_max_str_digits()
        s = '2' * max_digits
        i = int_class(s, base)
        if base > 10:
            with self.assertRaises(ValueError):
                str(i)
        elif base < 10:
            str(i)
        with self.assertRaises(ValueError) as err:
            int_class(f'{s}1', base)

    def test_int_from_other_bases(self):
        base = 3
        with self.subTest(base=base):
            self._other_base_helper(base)
        base = 36
        with self.subTest(base=base):
            self._other_base_helper(base)

    def test_int_max_str_digits_is_per_interpreter(self):
        # Changing the limit in one interpreter does not change others.
        code = """if 1:
        # Subinterpreters maintain and enforce their own limit
        import sys
        sys.set_int_max_str_digits(2323)
        try:
            int('3'*3333)
        except ValueError:
            pass
        else:
            raise AssertionError('Expected a int max str digits ValueError.')
        """
        with support.adjust_int_max_str_digits(4000):
            before_value = sys.get_int_max_str_digits()
            self.assertEqual(support.run_in_subinterp(code), 0,
                             'subinterp code failure, check stderr.')
            after_value = sys.get_int_max_str_digits()
            self.assertEqual(before_value, after_value)


class IntSubclassStrDigitLimitsTests(IntStrDigitLimitsTests):
    int_class = IntSubclass


class PyLongModuleTests(__TestCase):
    # Tests of the functions in _pylong.py.  Those get used when the
    # number of digits in the input values are large enough.

    def setUp(self):
        super().setUp()
        self._previous_limit = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(0)

    def tearDown(self):
        sys.set_int_max_str_digits(self._previous_limit)
        super().tearDown()

    def _test_pylong_int_to_decimal(self, n, suffix):
        s = str(n)
        self.assertEqual(s[-10:], suffix)
        s2 = str(-n)
        self.assertEqual(s2, '-' + s)
        s3 = '%d' % n
        self.assertEqual(s3, s)
        s4 = b'%d' % n
        self.assertEqual(s4, s.encode('ascii'))

    def test_pylong_int_to_decimal(self):
        self._test_pylong_int_to_decimal((1 << 100_000), '9883109376')
        self._test_pylong_int_to_decimal((1 << 100_000) - 1, '9883109375')
        self._test_pylong_int_to_decimal(10**30_000, '0000000000')
        self._test_pylong_int_to_decimal(10**30_000 - 1, '9999999999')
        self._test_pylong_int_to_decimal(3**60_000, '9313200001')

    @support.requires_resource('cpu')
    def test_pylong_int_to_decimal_2(self):
        self._test_pylong_int_to_decimal(2**1_000_000, '2747109376')
        self._test_pylong_int_to_decimal(10**300_000, '0000000000')
        self._test_pylong_int_to_decimal(3**600_000, '3132000001')

    def test_pylong_int_divmod(self):
        n = (1 << 100_000)
        a, b = divmod(n*3 + 1, n)
        assert a == 3 and b == 1

    def test_pylong_str_to_int(self):
        v1 = 1 << 100_000
        s = str(v1)
        v2 = int(s)
        assert v1 == v2
        v3 = int(' -' + s)
        assert -v1 == v3
        v4 = int(' +' + s + ' ')
        assert v1 == v4
        with self.assertRaises(ValueError) as err:
            int(s + 'z')
        with self.assertRaises(ValueError) as err:
            int(s + '_')
        with self.assertRaises(ValueError) as err:
            int('_' + s)

    @support.cpython_only  # tests implementation details of CPython.
    @unittest.skipUnless(_pylong, "_pylong module required")
    @mock.patch.object(_pylong, "int_to_decimal_string")
    def test_pylong_misbehavior_error_path_to_str(
            self, mock_int_to_str):
        with support.adjust_int_max_str_digits(20_000):
            big_value = int('7'*19_999)
            mock_int_to_str.return_value = None  # not a str
            with self.assertRaises(TypeError) as ctx:
                str(big_value)
            self.assertIn('_pylong.int_to_decimal_string did not',
                          str(ctx.exception))
            mock_int_to_str.side_effect = RuntimeError("testABC")
            with self.assertRaises(RuntimeError):
                str(big_value)

    @support.cpython_only  # tests implementation details of CPython.
    @unittest.skipUnless(_pylong, "_pylong module required")
    @mock.patch.object(_pylong, "int_from_string")
    def test_pylong_misbehavior_error_path_from_str(
            self, mock_int_from_str):
        big_value = '7'*19_999
        with support.adjust_int_max_str_digits(20_000):
            mock_int_from_str.return_value = b'not an int'
            with self.assertRaises(TypeError) as ctx:
                int(big_value)
            self.assertIn('_pylong.int_from_string did not',
                          str(ctx.exception))

            mock_int_from_str.side_effect = RuntimeError("test123")
            with self.assertRaises(RuntimeError):
                int(big_value)

    def test_pylong_roundtrip(self):
        from random import randrange, getrandbits
        bits = 5000
        while bits <= 1_000_000:
            bits += randrange(-100, 101) # break bitlength patterns
            hibit = 1 << (bits - 1)
            n = hibit | getrandbits(bits - 1)
            assert n.bit_length() == bits
            sn = str(n)
            self.assertFalse(sn.startswith('0'))
            self.assertEqual(n, int(sn))
            bits <<= 1

if __name__ == "__main__":
    if TEST_WITH_TORCHDYNAMO:
        run_tests()
    else:
        unittest.main()
