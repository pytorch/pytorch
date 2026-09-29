# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_long.py

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


import unittest
from test import support

import sys

import random
import math
import array

# SHIFT should match the value in longintrepr.h for best testing.
SHIFT = sys.int_info.bits_per_digit
BASE = 2 ** SHIFT
MASK = BASE - 1
KARATSUBA_CUTOFF = 70   # from longobject.c

# Max number of base BASE digits to use in test cases.  Doubling
# this will more than double the runtime.
MAXDIGITS = 15

# build some special values
special = [0, 1, 2, BASE, BASE >> 1, 0x5555555555555555, 0xaaaaaaaaaaaaaaaa]
#  some solid strings of one bits
p2 = 4  # 0 and 1 already added
for i in range(2*SHIFT):
    special.append(p2 - 1)
    p2 = p2 << 1
del p2
# add complements & negations
special += [~x for x in special] + [-x for x in special]

DBL_MAX = sys.float_info.max
DBL_MAX_EXP = sys.float_info.max_exp
DBL_MIN_EXP = sys.float_info.min_exp
DBL_MANT_DIG = sys.float_info.mant_dig
DBL_MIN_OVERFLOW = 2**DBL_MAX_EXP - 2**(DBL_MAX_EXP - DBL_MANT_DIG - 1)


# Pure Python version of correctly-rounded integer-to-float conversion.
def int_to_float(n):
    """
    Correctly-rounded integer-to-float conversion.

    """
    # Constants, depending only on the floating-point format in use.
    # We use an extra 2 bits of precision for rounding purposes.
    PRECISION = sys.float_info.mant_dig + 2
    SHIFT_MAX = sys.float_info.max_exp - PRECISION
    Q_MAX = 1 << PRECISION
    ROUND_HALF_TO_EVEN_CORRECTION = [0, -1, -2, 1, 0, -1, 2, 1]

    # Reduce to the case where n is positive.
    if n == 0:
        return 0.0
    elif n < 0:
        return -int_to_float(-n)

    # Convert n to a 'floating-point' number q * 2**shift, where q is an
    # integer with 'PRECISION' significant bits.  When shifting n to create q,
    # the least significant bit of q is treated as 'sticky'.  That is, the
    # least significant bit of q is set if either the corresponding bit of n
    # was already set, or any one of the bits of n lost in the shift was set.
    shift = n.bit_length() - PRECISION
    q = n << -shift if shift < 0 else (n >> shift) | bool(n & ~(-1 << shift))

    # Round half to even (actually rounds to the nearest multiple of 4,
    # rounding ties to a multiple of 8).
    q += ROUND_HALF_TO_EVEN_CORRECTION[q & 7]

    # Detect overflow.
    if shift + (q == Q_MAX) > SHIFT_MAX:
        raise OverflowError("integer too large to convert to float")

    # Checks: q is exactly representable, and q**2**shift doesn't overflow.
    assert q % 4 == 0 and q // 4 <= 2**(sys.float_info.mant_dig)
    assert q * 2**shift <= sys.float_info.max

    # Some circularity here, since float(q) is doing an int-to-float
    # conversion.  But here q is of bounded size, and is exactly representable
    # as a float.  In a low-level C-like language, this operation would be a
    # simple cast (e.g., from unsigned long long to double).
    return math.ldexp(float(q), shift)


# pure Python version of correctly-rounded true division
def truediv(a, b):
    """Correctly-rounded true division for integers."""
    negative = a^b < 0
    a, b = abs(a), abs(b)

    # exceptions:  division by zero, overflow
    if not b:
        raise ZeroDivisionError("division by zero")
    if a >= DBL_MIN_OVERFLOW * b:
        raise OverflowError("int/int too large to represent as a float")

   # find integer d satisfying 2**(d - 1) <= a/b < 2**d
    d = a.bit_length() - b.bit_length()
    if d >= 0 and a >= 2**d * b or d < 0 and a * 2**-d >= b:
        d += 1

    # compute 2**-exp * a / b for suitable exp
    exp = max(d, DBL_MIN_EXP) - DBL_MANT_DIG
    a, b = a << max(-exp, 0), b << max(exp, 0)
    q, r = divmod(a, b)

    # round-half-to-even: fractional part is r/b, which is > 0.5 iff
    # 2*r > b, and == 0.5 iff 2*r == b.
    if 2*r > b or 2*r == b and q % 2 == 1:
        q += 1

    result = math.ldexp(q, exp)
    return -result if negative else result


class LongTest(__TestCase):

    # Get quasi-random long consisting of ndigits digits (in base BASE).
    # quasi == the most-significant digit will not be 0, and the number
    # is constructed to contain long strings of 0 and 1 bits.  These are
    # more likely than random bits to provoke digit-boundary errors.
    # The sign of the number is also random.

    def getran(self, ndigits):
        self.assertGreater(ndigits, 0)
        nbits_hi = ndigits * SHIFT
        nbits_lo = nbits_hi - SHIFT + 1
        answer = 0
        nbits = 0
        r = int(random.random() * (SHIFT * 2)) | 1  # force 1 bits to start
        while nbits < nbits_lo:
            bits = (r >> 1) + 1
            bits = min(bits, nbits_hi - nbits)
            self.assertTrue(1 <= bits <= SHIFT)
            nbits = nbits + bits
            answer = answer << bits
            if r & 1:
                answer = answer | ((1 << bits) - 1)
            r = int(random.random() * (SHIFT * 2))
        self.assertTrue(nbits_lo <= nbits <= nbits_hi)
        if random.random() < 0.5:
            answer = -answer
        return answer

    # Get random long consisting of ndigits random digits (relative to base
    # BASE).  The sign bit is also random.

    def getran2(ndigits):
        answer = 0
        for i in range(ndigits):
            answer = (answer << SHIFT) | random.randint(0, MASK)
        if random.random() < 0.5:
            answer = -answer
        return answer

    def check_division(self, x, y):
        eq = self.assertEqual
        with self.subTest(x=x, y=y):
            q, r = divmod(x, y)
            q2, r2 = x//y, x%y
            pab, pba = x*y, y*x
            eq(pab, pba, "multiplication does not commute")
            eq(q, q2, "divmod returns different quotient than /")
            eq(r, r2, "divmod returns different mod than %")
            eq(x, q*y + r, "x != q*y + r after divmod")
            if y > 0:
                self.assertTrue(0 <= r < y, "bad mod from divmod")
            else:
                self.assertTrue(y < r <= 0, "bad mod from divmod")

    def test_division(self):
        digits = list(range(1, MAXDIGITS+1)) + list(range(KARATSUBA_CUTOFF,
                                                      KARATSUBA_CUTOFF + 14))
        digits.append(KARATSUBA_CUTOFF * 3)
        for lenx in digits:
            x = self.getran(lenx)
            for leny in digits:
                y = self.getran(leny) or 1
                self.check_division(x, y)

        # specific numbers chosen to exercise corner cases of the
        # current long division implementation

        # 30-bit cases involving a quotient digit estimate of BASE+1
        self.check_division(1231948412290879395966702881,
                            1147341367131428698)
        self.check_division(815427756481275430342312021515587883,
                       707270836069027745)
        self.check_division(627976073697012820849443363563599041,
                       643588798496057020)
        self.check_division(1115141373653752303710932756325578065,
                       1038556335171453937726882627)
        # 30-bit cases that require the post-subtraction correction step
        self.check_division(922498905405436751940989320930368494,
                       949985870686786135626943396)
        self.check_division(768235853328091167204009652174031844,
                       1091555541180371554426545266)

        # 15-bit cases involving a quotient digit estimate of BASE+1
        self.check_division(20172188947443, 615611397)
        self.check_division(1020908530270155025, 950795710)
        self.check_division(128589565723112408, 736393718)
        self.check_division(609919780285761575, 18613274546784)
        # 15-bit cases that require the post-subtraction correction step
        self.check_division(710031681576388032, 26769404391308)
        self.check_division(1933622614268221, 30212853348836)



    def test_karatsuba(self):
        digits = list(range(1, 5)) + list(range(KARATSUBA_CUTOFF,
                                                KARATSUBA_CUTOFF + 10))
        digits.extend([KARATSUBA_CUTOFF * 10, KARATSUBA_CUTOFF * 100])

        bits = [digit * SHIFT for digit in digits]

        # Test products of long strings of 1 bits -- (2**x-1)*(2**y-1) ==
        # 2**(x+y) - 2**x - 2**y + 1, so the proper result is easy to check.
        for abits in bits:
            a = (1 << abits) - 1
            for bbits in bits:
                if bbits < abits:
                    continue
                with self.subTest(abits=abits, bbits=bbits):
                    b = (1 << bbits) - 1
                    x = a * b
                    y = ((1 << (abits + bbits)) -
                         (1 << abits) -
                         (1 << bbits) +
                         1)
                    self.assertEqual(x, y)

    def check_bitop_identities_1(self, x):
        eq = self.assertEqual
        with self.subTest(x=x):
            eq(x & 0, 0)
            eq(x | 0, x)
            eq(x ^ 0, x)
            eq(x & -1, x)
            eq(x | -1, -1)
            eq(x ^ -1, ~x)
            eq(x, ~~x)
            eq(x & x, x)
            eq(x | x, x)
            eq(x ^ x, 0)
            eq(x & ~x, 0)
            eq(x | ~x, -1)
            eq(x ^ ~x, -1)
            eq(-x, 1 + ~x)
            eq(-x, ~(x-1))
        for n in range(2*SHIFT):
            p2 = 2 ** n
            with self.subTest(x=x, n=n, p2=p2):
                eq(x << n >> n, x)
                eq(x // p2, x >> n)
                eq(x * p2, x << n)
                eq(x & -p2, x >> n << n)
                eq(x & -p2, x & ~(p2 - 1))

    def check_bitop_identities_2(self, x, y):
        eq = self.assertEqual
        with self.subTest(x=x, y=y):
            eq(x & y, y & x)
            eq(x | y, y | x)
            eq(x ^ y, y ^ x)
            eq(x ^ y ^ x, y)
            eq(x & y, ~(~x | ~y))
            eq(x | y, ~(~x & ~y))
            eq(x ^ y, (x | y) & ~(x & y))
            eq(x ^ y, (x & ~y) | (~x & y))
            eq(x ^ y, (x | y) & (~x | ~y))

    def check_bitop_identities_3(self, x, y, z):
        eq = self.assertEqual
        with self.subTest(x=x, y=y, z=z):
            eq((x & y) & z, x & (y & z))
            eq((x | y) | z, x | (y | z))
            eq((x ^ y) ^ z, x ^ (y ^ z))
            eq(x & (y | z), (x & y) | (x & z))
            eq(x | (y & z), (x | y) & (x | z))

    def test_bitop_identities(self):
        for x in special:
            self.check_bitop_identities_1(x)
        digits = range(1, MAXDIGITS+1)
        for lenx in digits:
            x = self.getran(lenx)
            self.check_bitop_identities_1(x)
            for leny in digits:
                y = self.getran(leny)
                self.check_bitop_identities_2(x, y)
                self.check_bitop_identities_3(x, y, self.getran((lenx + leny)//2))

    def slow_format(self, x, base):
        digits = []
        sign = 0
        if x < 0:
            sign, x = 1, -x
        while x:
            x, r = divmod(x, base)
            digits.append(int(r))
        digits.reverse()
        digits = digits or [0]
        return '-'[:sign] + \
               {2: '0b', 8: '0o', 10: '', 16: '0x'}[base] + \
               "".join("0123456789abcdef"[i] for i in digits)

    def check_format_1(self, x):
        for base, mapper in (2, bin), (8, oct), (10, str), (10, repr), (16, hex):
            got = mapper(x)
            with self.subTest(x=x, mapper=mapper.__name__):
                expected = self.slow_format(x, base)
                self.assertEqual(got, expected)
            with self.subTest(got=got):
                self.assertEqual(int(got, 0), x)

    def test_format(self):
        for x in special:
            self.check_format_1(x)
        for i in range(10):
            for lenx in range(1, MAXDIGITS+1):
                x = self.getran(lenx)
                self.check_format_1(x)

    def test_long(self):
        # Check conversions from string
        LL = [
                ('1' + '0'*20, 10**20),
                ('1' + '0'*100, 10**100)
        ]
        for s, v in LL:
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

        # trailing L should no longer be accepted...
        self.assertRaises(ValueError, int, '123L')
        self.assertRaises(ValueError, int, '123l')
        self.assertRaises(ValueError, int, '0L')
        self.assertRaises(ValueError, int, '-37L')
        self.assertRaises(ValueError, int, '0x32L', 16)
        self.assertRaises(ValueError, int, '1L', 21)
        # ... but it's just a normal digit if base >= 22
        self.assertEqual(int('1L', 22), 43)

        # tests with base 0
        self.assertEqual(int('000', 0), 0)
        self.assertEqual(int('0o123', 0), 83)
        self.assertEqual(int('0x123', 0), 291)
        self.assertEqual(int('0b100', 0), 4)
        self.assertEqual(int(' 0O123   ', 0), 83)
        self.assertEqual(int(' 0X123  ', 0), 291)
        self.assertEqual(int(' 0B100 ', 0), 4)
        self.assertEqual(int('0', 0), 0)
        self.assertEqual(int('+0', 0), 0)
        self.assertEqual(int('-0', 0), 0)
        self.assertEqual(int('00', 0), 0)
        self.assertRaises(ValueError, int, '08', 0)
        self.assertRaises(ValueError, int, '-012395', 0)

        # invalid bases
        invalid_bases = [-909,
                          2**31-1, 2**31, -2**31, -2**31-1,
                          2**63-1, 2**63, -2**63, -2**63-1,
                          2**100, -2**100,
                          ]
        for base in invalid_bases:
            self.assertRaises(ValueError, int, '42', base)

        # Invalid unicode string
        # See bpo-34087
        self.assertRaises(ValueError, int, '\u3053\u3093\u306b\u3061\u306f')


    def test_conversion(self):

        class JustLong:
            # test that __long__ no longer used in 3.x
            def __long__(self):
                return 42
        self.assertRaises(TypeError, int, JustLong())

        class LongTrunc:
            # __long__ should be ignored in 3.x
            def __long__(self):
                return 42
            def __trunc__(self):
                return 1729
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(int(LongTrunc()), 1729)

    def check_float_conversion(self, n):
        # Check that int -> float conversion behaviour matches
        # that of the pure Python version above.
        try:
            actual = float(n)
        except OverflowError:
            actual = 'overflow'

        try:
            expected = int_to_float(n)
        except OverflowError:
            expected = 'overflow'

        msg = ("Error in conversion of integer {} to float.  "
               "Got {}, expected {}.".format(n, actual, expected))
        self.assertEqual(actual, expected, msg)

    @support.requires_IEEE_754
    def test_float_conversion(self):

        exact_values = [0, 1, 2,
                         2**53-3,
                         2**53-2,
                         2**53-1,
                         2**53,
                         2**53+2,
                         2**54-4,
                         2**54-2,
                         2**54,
                         2**54+4]
        for x in exact_values:
            self.assertEqual(float(x), x)
            self.assertEqual(float(-x), -x)

        # test round-half-even
        for x, y in [(1, 0), (2, 2), (3, 4), (4, 4), (5, 4), (6, 6), (7, 8)]:
            for p in range(15):
                self.assertEqual(int(float(2**p*(2**53+x))), 2**p*(2**53+y))

        for x, y in [(0, 0), (1, 0), (2, 0), (3, 4), (4, 4), (5, 4), (6, 8),
                     (7, 8), (8, 8), (9, 8), (10, 8), (11, 12), (12, 12),
                     (13, 12), (14, 16), (15, 16)]:
            for p in range(15):
                self.assertEqual(int(float(2**p*(2**54+x))), 2**p*(2**54+y))

        # behaviour near extremes of floating-point range
        int_dbl_max = int(DBL_MAX)
        top_power = 2**DBL_MAX_EXP
        halfway = (int_dbl_max + top_power)//2
        self.assertEqual(float(int_dbl_max), DBL_MAX)
        self.assertEqual(float(int_dbl_max+1), DBL_MAX)
        self.assertEqual(float(halfway-1), DBL_MAX)
        self.assertRaises(OverflowError, float, halfway)
        self.assertEqual(float(1-halfway), -DBL_MAX)
        self.assertRaises(OverflowError, float, -halfway)
        self.assertRaises(OverflowError, float, top_power-1)
        self.assertRaises(OverflowError, float, top_power)
        self.assertRaises(OverflowError, float, top_power+1)
        self.assertRaises(OverflowError, float, 2*top_power-1)
        self.assertRaises(OverflowError, float, 2*top_power)
        self.assertRaises(OverflowError, float, top_power*top_power)

        for p in range(100):
            x = 2**p * (2**53 + 1) + 1
            y = 2**p * (2**53 + 2)
            self.assertEqual(int(float(x)), y)

            x = 2**p * (2**53 + 1)
            y = 2**p * 2**53
            self.assertEqual(int(float(x)), y)

        # Compare builtin float conversion with pure Python int_to_float
        # function above.
        test_values = [
            int_dbl_max-1, int_dbl_max, int_dbl_max+1,
            halfway-1, halfway, halfway + 1,
            top_power-1, top_power, top_power+1,
            2*top_power-1, 2*top_power, top_power*top_power,
        ]
        test_values.extend(exact_values)
        for p in range(-4, 8):
            for x in range(-128, 128):
                test_values.append(2**(p+53) + x)
        for value in test_values:
            self.check_float_conversion(value)
            self.check_float_conversion(-value)

    def test_float_overflow(self):
        for x in -2.0, -1.0, 0.0, 1.0, 2.0:
            self.assertEqual(float(int(x)), x)

        shuge = '12345' * 120
        huge = 1 << 30000
        mhuge = -huge
        namespace = {'huge': huge, 'mhuge': mhuge, 'shuge': shuge, 'math': math}
        for test in ["float(huge)", "float(mhuge)",
                     "complex(huge)", "complex(mhuge)",
                     "complex(huge, 1)", "complex(mhuge, 1)",
                     "complex(1, huge)", "complex(1, mhuge)",
                     "1. + huge", "huge + 1.", "1. + mhuge", "mhuge + 1.",
                     "1. - huge", "huge - 1.", "1. - mhuge", "mhuge - 1.",
                     "1. * huge", "huge * 1.", "1. * mhuge", "mhuge * 1.",
                     "1. // huge", "huge // 1.", "1. // mhuge", "mhuge // 1.",
                     "1. / huge", "huge / 1.", "1. / mhuge", "mhuge / 1.",
                     "1. ** huge", "huge ** 1.", "1. ** mhuge", "mhuge ** 1.",
                     "math.sin(huge)", "math.sin(mhuge)",
                     "math.sqrt(huge)", "math.sqrt(mhuge)", # should do better
                     # math.floor() of an int returns an int now
                     ##"math.floor(huge)", "math.floor(mhuge)",
                     ]:

            self.assertRaises(OverflowError, eval, test, namespace)

        # XXX Perhaps float(shuge) can raise OverflowError on some box?
        # The comparison should not.
        self.assertNotEqual(float(shuge), int(shuge),
            "float(shuge) should not equal int(shuge)")

    def test_logs(self):
        LOG10E = math.log10(math.e)

        for exp in list(range(10)) + [100, 1000, 10000]:
            value = 10 ** exp
            log10 = math.log10(value)
            self.assertAlmostEqual(log10, exp)

            # log10(value) == exp, so log(value) == log10(value)/log10(e) ==
            # exp/LOG10E
            expected = exp / LOG10E
            log = math.log(value)
            self.assertAlmostEqual(log, expected)

        for bad in -(1 << 10000), -2, 0:
            self.assertRaises(ValueError, math.log, bad)
            self.assertRaises(ValueError, math.log10, bad)

    def test_mixed_compares(self):
        eq = self.assertEqual

        # We're mostly concerned with that mixing floats and ints does the
        # right stuff, even when ints are too large to fit in a float.
        # The safest way to check the results is to use an entirely different
        # method, which we do here via a skeletal rational class (which
        # represents all Python ints and floats exactly).
        class Rat:
            def __init__(self, value):
                if isinstance(value, int):
                    self.n = value
                    self.d = 1
                elif isinstance(value, float):
                    # Convert to exact rational equivalent.
                    f, e = math.frexp(abs(value))
                    assert f == 0 or 0.5 <= f < 1.0
                    # |value| = f * 2**e exactly

                    # Suck up CHUNK bits at a time; 28 is enough so that we suck
                    # up all bits in 2 iterations for all known binary double-
                    # precision formats, and small enough to fit in an int.
                    CHUNK = 28
                    top = 0
                    # invariant: |value| = (top + f) * 2**e exactly
                    while f:
                        f = math.ldexp(f, CHUNK)
                        digit = int(f)
                        assert digit >> CHUNK == 0
                        top = (top << CHUNK) | digit
                        f -= digit
                        assert 0.0 <= f < 1.0
                        e -= CHUNK

                    # Now |value| = top * 2**e exactly.
                    if e >= 0:
                        n = top << e
                        d = 1
                    else:
                        n = top
                        d = 1 << -e
                    if value < 0:
                        n = -n
                    self.n = n
                    self.d = d
                    assert float(n) / float(d) == value
                else:
                    raise TypeError("can't deal with %r" % value)

            def _cmp__(self, other):
                if not isinstance(other, Rat):
                    other = Rat(other)
                x, y = self.n * other.d, self.d * other.n
                return (x > y) - (x < y)
            def __eq__(self, other):
                return self._cmp__(other) == 0
            def __ge__(self, other):
                return self._cmp__(other) >= 0
            def __gt__(self, other):
                return self._cmp__(other) > 0
            def __le__(self, other):
                return self._cmp__(other) <= 0
            def __lt__(self, other):
                return self._cmp__(other) < 0

        cases = [0, 0.001, 0.99, 1.0, 1.5, 1e20, 1e200]
        # 2**48 is an important boundary in the internals.  2**53 is an
        # important boundary for IEEE double precision.
        for t in 2.0**48, 2.0**50, 2.0**53:
            cases.extend([t - 1.0, t - 0.3, t, t + 0.3, t + 1.0,
                          int(t-1), int(t), int(t+1)])
        cases.extend([0, 1, 2, sys.maxsize, float(sys.maxsize)])
        # 1 << 20000 should exceed all double formats.  int(1e200) is to
        # check that we get equality with 1e200 above.
        t = int(1e200)
        cases.extend([0, 1, 2, 1 << 20000, t-1, t, t+1])
        cases.extend([-x for x in cases])
        for x in cases:
            Rx = Rat(x)
            for y in cases:
                Ry = Rat(y)
                Rcmp = (Rx > Ry) - (Rx < Ry)
                with self.subTest(x=x, y=y, Rcmp=Rcmp):
                    xycmp = (x > y) - (x < y)
                    eq(Rcmp, xycmp)
                    eq(x == y, Rcmp == 0)
                    eq(x != y, Rcmp != 0)
                    eq(x < y, Rcmp < 0)
                    eq(x <= y, Rcmp <= 0)
                    eq(x > y, Rcmp > 0)
                    eq(x >= y, Rcmp >= 0)

    def test__format__(self):
        self.assertEqual(format(123456789, 'd'), '123456789')
        self.assertEqual(format(123456789, 'd'), '123456789')
        self.assertEqual(format(123456789, ','), '123,456,789')
        self.assertEqual(format(123456789, '_'), '123_456_789')

        # sign and aligning are interdependent
        self.assertEqual(format(1, "-"), '1')
        self.assertEqual(format(-1, "-"), '-1')
        self.assertEqual(format(1, "-3"), '  1')
        self.assertEqual(format(-1, "-3"), ' -1')
        self.assertEqual(format(1, "+3"), ' +1')
        self.assertEqual(format(-1, "+3"), ' -1')
        self.assertEqual(format(1, " 3"), '  1')
        self.assertEqual(format(-1, " 3"), ' -1')
        self.assertEqual(format(1, " "), ' 1')
        self.assertEqual(format(-1, " "), '-1')

        # hex
        self.assertEqual(format(3, "x"), "3")
        self.assertEqual(format(3, "X"), "3")
        self.assertEqual(format(1234, "x"), "4d2")
        self.assertEqual(format(-1234, "x"), "-4d2")
        self.assertEqual(format(1234, "8x"), "     4d2")
        self.assertEqual(format(-1234, "8x"), "    -4d2")
        self.assertEqual(format(1234, "x"), "4d2")
        self.assertEqual(format(-1234, "x"), "-4d2")
        self.assertEqual(format(-3, "x"), "-3")
        self.assertEqual(format(-3, "X"), "-3")
        self.assertEqual(format(int('be', 16), "x"), "be")
        self.assertEqual(format(int('be', 16), "X"), "BE")
        self.assertEqual(format(-int('be', 16), "x"), "-be")
        self.assertEqual(format(-int('be', 16), "X"), "-BE")
        self.assertRaises(ValueError, format, 1234567890, ',x')
        self.assertEqual(format(1234567890, '_x'), '4996_02d2')
        self.assertEqual(format(1234567890, '_X'), '4996_02D2')

        # octal
        self.assertEqual(format(3, "o"), "3")
        self.assertEqual(format(-3, "o"), "-3")
        self.assertEqual(format(1234, "o"), "2322")
        self.assertEqual(format(-1234, "o"), "-2322")
        self.assertEqual(format(1234, "-o"), "2322")
        self.assertEqual(format(-1234, "-o"), "-2322")
        self.assertEqual(format(1234, " o"), " 2322")
        self.assertEqual(format(-1234, " o"), "-2322")
        self.assertEqual(format(1234, "+o"), "+2322")
        self.assertEqual(format(-1234, "+o"), "-2322")
        self.assertRaises(ValueError, format, 1234567890, ',o')
        self.assertEqual(format(1234567890, '_o'), '111_4540_1322')

        # binary
        self.assertEqual(format(3, "b"), "11")
        self.assertEqual(format(-3, "b"), "-11")
        self.assertEqual(format(1234, "b"), "10011010010")
        self.assertEqual(format(-1234, "b"), "-10011010010")
        self.assertEqual(format(1234, "-b"), "10011010010")
        self.assertEqual(format(-1234, "-b"), "-10011010010")
        self.assertEqual(format(1234, " b"), " 10011010010")
        self.assertEqual(format(-1234, " b"), "-10011010010")
        self.assertEqual(format(1234, "+b"), "+10011010010")
        self.assertEqual(format(-1234, "+b"), "-10011010010")
        self.assertRaises(ValueError, format, 1234567890, ',b')
        self.assertEqual(format(12345, '_b'), '11_0000_0011_1001')

        # make sure these are errors
        self.assertRaises(ValueError, format, 3, "1.3")  # precision disallowed
        self.assertRaises(ValueError, format, 3, "_c")   # underscore,
        self.assertRaises(ValueError, format, 3, ",c")   # comma, and
        self.assertRaises(ValueError, format, 3, "+c")   # sign not allowed
                                                         # with 'c'

        self.assertRaisesRegex(ValueError, 'Cannot specify both', format, 3, '_,')
        self.assertRaisesRegex(ValueError, 'Cannot specify both', format, 3, ',_')
        self.assertRaisesRegex(ValueError, 'Cannot specify both', format, 3, '_,d')
        self.assertRaisesRegex(ValueError, 'Cannot specify both', format, 3, ',_d')

        self.assertRaisesRegex(ValueError, "Cannot specify ',' with 's'", format, 3, ',s')
        self.assertRaisesRegex(ValueError, "Cannot specify '_' with 's'", format, 3, '_s')

        # ensure that only int and float type specifiers work
        for format_spec in ([chr(x) for x in range(ord('a'), ord('z')+1)] +
                            [chr(x) for x in range(ord('A'), ord('Z')+1)]):
            if not format_spec in 'bcdoxXeEfFgGn%':
                self.assertRaises(ValueError, format, 0, format_spec)
                self.assertRaises(ValueError, format, 1, format_spec)
                self.assertRaises(ValueError, format, -1, format_spec)
                self.assertRaises(ValueError, format, 2**100, format_spec)
                self.assertRaises(ValueError, format, -(2**100), format_spec)

        # ensure that float type specifiers work; format converts
        #  the int to a float
        for format_spec in 'eEfFgG%':
            for value in [0, 1, -1, 100, -100, 1234567890, -1234567890]:
                self.assertEqual(format(value, format_spec),
                                 format(float(value), format_spec))

    def test_nan_inf(self):
        self.assertRaises(OverflowError, int, float('inf'))
        self.assertRaises(OverflowError, int, float('-inf'))
        self.assertRaises(ValueError, int, float('nan'))

    def test_mod_division(self):
        with self.assertRaises(ZeroDivisionError):
            _ = 1 % 0

        self.assertEqual(13 % 10, 3)
        self.assertEqual(-13 % 10, 7)
        self.assertEqual(13 % -10, -7)
        self.assertEqual(-13 % -10, -3)

        self.assertEqual(12 % 4, 0)
        self.assertEqual(-12 % 4, 0)
        self.assertEqual(12 % -4, 0)
        self.assertEqual(-12 % -4, 0)

    def test_true_division(self):
        huge = 1 << 40000
        mhuge = -huge
        self.assertEqual(huge / huge, 1.0)
        self.assertEqual(mhuge / mhuge, 1.0)
        self.assertEqual(huge / mhuge, -1.0)
        self.assertEqual(mhuge / huge, -1.0)
        self.assertEqual(1 / huge, 0.0)
        self.assertEqual(1 / huge, 0.0)
        self.assertEqual(1 / mhuge, 0.0)
        self.assertEqual(1 / mhuge, 0.0)
        self.assertEqual((666 * huge + (huge >> 1)) / huge, 666.5)
        self.assertEqual((666 * mhuge + (mhuge >> 1)) / mhuge, 666.5)
        self.assertEqual((666 * huge + (huge >> 1)) / mhuge, -666.5)
        self.assertEqual((666 * mhuge + (mhuge >> 1)) / huge, -666.5)
        self.assertEqual(huge / (huge << 1), 0.5)
        self.assertEqual((1000000 * huge) / huge, 1000000)

        namespace = {'huge': huge, 'mhuge': mhuge}

        for overflow in ["float(huge)", "float(mhuge)",
                         "huge / 1", "huge / 2", "huge / -1", "huge / -2",
                         "mhuge / 100", "mhuge / 200"]:
            self.assertRaises(OverflowError, eval, overflow, namespace)

        for underflow in ["1 / huge", "2 / huge", "-1 / huge", "-2 / huge",
                         "100 / mhuge", "200 / mhuge"]:
            result = eval(underflow, namespace)
            self.assertEqual(result, 0.0,
                             "expected underflow to 0 from %r" % underflow)

        for zero in ["huge / 0", "mhuge / 0"]:
            self.assertRaises(ZeroDivisionError, eval, zero, namespace)

    def test_floordiv(self):
        with self.assertRaises(ZeroDivisionError):
            _ = 1 // 0

        self.assertEqual(2 // 3, 0)
        self.assertEqual(2 // -3, -1)
        self.assertEqual(-2 // 3, -1)
        self.assertEqual(-2 // -3, 0)

        self.assertEqual(-11 // -3, 3)
        self.assertEqual(-11 // 3, -4)
        self.assertEqual(11 // -3, -4)
        self.assertEqual(11 // 3, 3)

        self.assertEqual(-12 // -3, 4)
        self.assertEqual(-12 // 3, -4)
        self.assertEqual(12 // -3, -4)
        self.assertEqual(12 // 3, 4)

    def check_truediv(self, a, b, skip_small=True):
        """Verify that the result of a/b is correctly rounded, by
        comparing it with a pure Python implementation of correctly
        rounded division.  b should be nonzero."""

        # skip check for small a and b: in this case, the current
        # implementation converts the arguments to float directly and
        # then applies a float division.  This can give doubly-rounded
        # results on x87-using machines (particularly 32-bit Linux).
        if skip_small and max(abs(a), abs(b)) < 2**DBL_MANT_DIG:
            return

        try:
            # use repr so that we can distinguish between -0.0 and 0.0
            expected = repr(truediv(a, b))
        except OverflowError:
            expected = 'overflow'
        except ZeroDivisionError:
            expected = 'zerodivision'

        try:
            got = repr(a / b)
        except OverflowError:
            got = 'overflow'
        except ZeroDivisionError:
            got = 'zerodivision'

        self.assertEqual(expected, got, "Incorrectly rounded division {}/{}: "
                         "expected {}, got {}".format(a, b, expected, got))

    @support.requires_IEEE_754
    def test_correctly_rounded_true_division(self):
        # more stringent tests than those above, checking that the
        # result of true division of ints is always correctly rounded.
        # This test should probably be considered CPython-specific.

        # Exercise all the code paths not involving Gb-sized ints.
        # ... divisions involving zero
        self.check_truediv(123, 0)
        self.check_truediv(-456, 0)
        self.check_truediv(0, 3)
        self.check_truediv(0, -3)
        self.check_truediv(0, 0)
        # ... overflow or underflow by large margin
        self.check_truediv(671 * 12345 * 2**DBL_MAX_EXP, 12345)
        self.check_truediv(12345, 345678 * 2**(DBL_MANT_DIG - DBL_MIN_EXP))
        # ... a much larger or smaller than b
        self.check_truediv(12345*2**100, 98765)
        self.check_truediv(12345*2**30, 98765*7**81)
        # ... a / b near a boundary: one of 1, 2**DBL_MANT_DIG, 2**DBL_MIN_EXP,
        #                 2**DBL_MAX_EXP, 2**(DBL_MIN_EXP-DBL_MANT_DIG)
        bases = (0, DBL_MANT_DIG, DBL_MIN_EXP,
                 DBL_MAX_EXP, DBL_MIN_EXP - DBL_MANT_DIG)
        for base in bases:
            for exp in range(base - 15, base + 15):
                self.check_truediv(75312*2**max(exp, 0), 69187*2**max(-exp, 0))
                self.check_truediv(69187*2**max(exp, 0), 75312*2**max(-exp, 0))

        # overflow corner case
        for m in [1, 2, 7, 17, 12345, 7**100,
                  -1, -2, -5, -23, -67891, -41**50]:
            for n in range(-10, 10):
                self.check_truediv(m*DBL_MIN_OVERFLOW + n, m)
                self.check_truediv(m*DBL_MIN_OVERFLOW + n, -m)

        # check detection of inexactness in shifting stage
        for n in range(250):
            # (2**DBL_MANT_DIG+1)/(2**DBL_MANT_DIG) lies halfway
            # between two representable floats, and would usually be
            # rounded down under round-half-to-even.  The tiniest of
            # additions to the numerator should cause it to be rounded
            # up instead.
            self.check_truediv((2**DBL_MANT_DIG + 1)*12345*2**200 + 2**n,
                           2**DBL_MANT_DIG*12345)

        # 1/2731 is one of the smallest division cases that's subject
        # to double rounding on IEEE 754 machines working internally with
        # 64-bit precision.  On such machines, the next check would fail,
        # were it not explicitly skipped in check_truediv.
        self.check_truediv(1, 2731)

        # a particularly bad case for the old algorithm:  gives an
        # error of close to 3.5 ulps.
        self.check_truediv(295147931372582273023, 295147932265116303360)
        for i in range(1000):
            self.check_truediv(10**(i+1), 10**i)
            self.check_truediv(10**i, 10**(i+1))

        # test round-half-to-even behaviour, normal result
        for m in [1, 2, 4, 7, 8, 16, 17, 32, 12345, 7**100,
                  -1, -2, -5, -23, -67891, -41**50]:
            for n in range(-10, 10):
                self.check_truediv(2**DBL_MANT_DIG*m + n, m)

        # test round-half-to-even, subnormal result
        for n in range(-20, 20):
            self.check_truediv(n, 2**1076)

        # largeish random divisions: a/b where |a| <= |b| <=
        # 2*|a|; |ans| is between 0.5 and 1.0, so error should
        # always be bounded by 2**-54 with equality possible only
        # if the least significant bit of q=ans*2**53 is zero.
        for M in [10**10, 10**100, 10**1000]:
            for i in range(1000):
                a = random.randrange(1, M)
                b = random.randrange(a, 2*a+1)
                self.check_truediv(a, b)
                self.check_truediv(-a, b)
                self.check_truediv(a, -b)
                self.check_truediv(-a, -b)

        # and some (genuinely) random tests
        for _ in range(10000):
            a_bits = random.randrange(1000)
            b_bits = random.randrange(1, 1000)
            x = random.randrange(2**a_bits)
            y = random.randrange(1, 2**b_bits)
            self.check_truediv(x, y)
            self.check_truediv(x, -y)
            self.check_truediv(-x, y)
            self.check_truediv(-x, -y)

    def test_negative_shift_count(self):
        with self.assertRaises(ValueError):
            42 << -3
        with self.assertRaises(ValueError):
            42 << -(1 << 1000)
        with self.assertRaises(ValueError):
            42 >> -3
        with self.assertRaises(ValueError):
            42 >> -(1 << 1000)

    def test_lshift_of_zero(self):
        self.assertEqual(0 << 0, 0)
        self.assertEqual(0 << 10, 0)
        with self.assertRaises(ValueError):
            0 << -1
        self.assertEqual(0 << (1 << 1000), 0)
        with self.assertRaises(ValueError):
            0 << -(1 << 1000)

    @support.cpython_only
    def test_huge_lshift_of_zero(self):
        # Shouldn't try to allocate memory for a huge shift. See issue #27870.
        # Other implementations may have a different boundary for overflow,
        # or not raise at all.
        self.assertEqual(0 << sys.maxsize, 0)
        self.assertEqual(0 << (sys.maxsize + 1), 0)

    @support.cpython_only
    @support.bigmemtest(sys.maxsize + 1000, memuse=2/15 * 2, dry_run=False)
    def test_huge_lshift(self, size):
        self.assertEqual(1 << (sys.maxsize + 1000), 1 << 1000 << sys.maxsize)

    def test_huge_rshift(self):
        huge_shift = 1 << 1000
        self.assertEqual(42 >> huge_shift, 0)
        self.assertEqual((-42) >> huge_shift, -1)
        self.assertEqual(1123 >> huge_shift, 0)
        self.assertEqual((-1123) >> huge_shift, -1)
        self.assertEqual(2**128 >> huge_shift, 0)
        self.assertEqual(-2**128 >> huge_shift, -1)

    @support.cpython_only
    @support.bigmemtest(sys.maxsize + 500, memuse=2/15, dry_run=False)
    def test_huge_rshift_of_huge(self, size):
        huge = ((1 << 500) + 11) << sys.maxsize
        self.assertEqual(huge >> (sys.maxsize + 1), (1 << 499) + 5)
        self.assertEqual(huge >> (sys.maxsize + 1000), 0)

    def test_small_rshift(self):
        self.assertEqual(42 >> 1, 21)
        self.assertEqual((-42) >> 1, -21)
        self.assertEqual(43 >> 1, 21)
        self.assertEqual((-43) >> 1, -22)

        self.assertEqual(1122 >> 1, 561)
        self.assertEqual((-1122) >> 1, -561)
        self.assertEqual(1123 >> 1, 561)
        self.assertEqual((-1123) >> 1, -562)

        self.assertEqual(2**128 >> 1, 2**127)
        self.assertEqual(-2**128 >> 1, -2**127)
        self.assertEqual((2**128 + 1) >> 1, 2**127)
        self.assertEqual(-(2**128 + 1) >> 1, -2**127 - 1)

    def test_medium_rshift(self):
        self.assertEqual(42 >> 9, 0)
        self.assertEqual((-42) >> 9, -1)
        self.assertEqual(1122 >> 9, 2)
        self.assertEqual((-1122) >> 9, -3)
        self.assertEqual(2**128 >> 9, 2**119)
        self.assertEqual(-2**128 >> 9, -2**119)
        # Exercise corner case of the current algorithm, where the result of
        # shifting a two-limb int by the limb size still has two limbs.
        self.assertEqual((1 - BASE*BASE) >> SHIFT, -BASE)
        self.assertEqual((BASE - 1 - BASE*BASE) >> SHIFT, -BASE)

    def test_big_rshift(self):
        self.assertEqual(42 >> 32, 0)
        self.assertEqual((-42) >> 32, -1)
        self.assertEqual(1122 >> 32, 0)
        self.assertEqual((-1122) >> 32, -1)
        self.assertEqual(2**128 >> 32, 2**96)
        self.assertEqual(-2**128 >> 32, -2**96)

    def test_small_lshift(self):
        self.assertEqual(42 << 1, 84)
        self.assertEqual((-42) << 1, -84)
        self.assertEqual(561 << 1, 1122)
        self.assertEqual((-561) << 1, -1122)
        self.assertEqual(2**127 << 1, 2**128)
        self.assertEqual(-2**127 << 1, -2**128)

    def test_medium_lshift(self):
        self.assertEqual(42 << 9, 21504)
        self.assertEqual((-42) << 9, -21504)
        self.assertEqual(1122 << 9, 574464)
        self.assertEqual((-1122) << 9, -574464)

    def test_big_lshift(self):
        self.assertEqual(42 << 32, 42 * 2**32)
        self.assertEqual((-42) << 32, -42 * 2**32)
        self.assertEqual(1122 << 32, 1122 * 2**32)
        self.assertEqual((-1122) << 32, -1122 * 2**32)
        self.assertEqual(2**128 << 32, 2**160)
        self.assertEqual(-2**128 << 32, -2**160)

    @support.cpython_only
    def test_small_ints_in_huge_calculation(self):
        a = 2 ** 100
        b = -a + 1
        c = a + 1
        self.assertIs(a + b, 1)
        self.assertIs(c - a, 1)

    @support.cpython_only
    def test_pow_uses_cached_small_ints(self):
        self.assertIs(pow(10, 3, 998), 2)
        self.assertIs(10 ** 3 % 998, 2)
        a, p, m = 10, 3, 998
        self.assertIs(a ** p % m, 2)

        self.assertIs(pow(2, 31, 2 ** 31 - 1), 1)
        self.assertIs(2 ** 31 % (2 ** 31 - 1), 1)
        a, p, m = 2, 31, 2 ** 31 - 1
        self.assertIs(a ** p % m, 1)

        self.assertIs(pow(2, 100, 2**100 - 3), 3)
        self.assertIs(2 ** 100 % (2 ** 100 - 3), 3)
        a, p, m = 2, 100, 2**100 - 3
        self.assertIs(a ** p % m, 3)

    @support.cpython_only
    def test_divmod_uses_cached_small_ints(self):
        big = 10 ** 100

        self.assertIs((big + 1) % big, 1)
        self.assertIs((big + 1) // big, 1)
        self.assertIs(big // (big // 2), 2)
        self.assertIs(big // (big // -4), -4)

        q, r = divmod(2 * big + 3, big)
        self.assertIs(q, 2)
        self.assertIs(r, 3)

        q, r = divmod(-4 * big + 100, big)
        self.assertIs(q, -4)
        self.assertIs(r, 100)

        q, r = divmod(3 * (-big) - 1, -big)
        self.assertIs(q, 3)
        self.assertIs(r, -1)

        q, r = divmod(3 * big - 1, -big)
        self.assertIs(q, -3)
        self.assertIs(r, -1)

    def test_small_ints(self):
        for i in range(-5, 257):
            self.assertIs(i, i + 0)
            self.assertIs(i, i * 1)
            self.assertIs(i, i - 0)
            self.assertIs(i, i // 1)
            self.assertIs(i, i & -1)
            self.assertIs(i, i | 0)
            self.assertIs(i, i ^ 0)
            self.assertIs(i, ~~i)
            self.assertIs(i, i**1)
            self.assertIs(i, int(str(i)))
            self.assertIs(i, i<<2>>2, str(i))
        # corner cases
        i = 1 << 70
        self.assertIs(i - i, 0)
        self.assertIs(0 * i, 0)

    def test_bit_length(self):
        tiny = 1e-10
        for x in range(-65000, 65000):
            k = x.bit_length()
            # Check equivalence with Python version
            self.assertEqual(k, len(bin(x).lstrip('-0b')))
            # Behaviour as specified in the docs
            if x != 0:
                self.assertTrue(2**(k-1) <= abs(x) < 2**k)
            else:
                self.assertEqual(k, 0)
            # Alternative definition: x.bit_length() == 1 + floor(log_2(x))
            if x != 0:
                # When x is an exact power of 2, numeric errors can
                # cause floor(log(x)/log(2)) to be one too small; for
                # small x this can be fixed by adding a small quantity
                # to the quotient before taking the floor.
                self.assertEqual(k, 1 + math.floor(
                        math.log(abs(x))/math.log(2) + tiny))

        self.assertEqual((0).bit_length(), 0)
        self.assertEqual((1).bit_length(), 1)
        self.assertEqual((-1).bit_length(), 1)
        self.assertEqual((2).bit_length(), 2)
        self.assertEqual((-2).bit_length(), 2)
        for i in [2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 234]:
            a = 2**i
            self.assertEqual((a-1).bit_length(), i)
            self.assertEqual((1-a).bit_length(), i)
            self.assertEqual((a).bit_length(), i+1)
            self.assertEqual((-a).bit_length(), i+1)
            self.assertEqual((a+1).bit_length(), i+1)
            self.assertEqual((-a-1).bit_length(), i+1)

    def test_bit_count(self):
        for a in range(-1000, 1000):
            self.assertEqual(a.bit_count(), bin(a).count("1"))

        for exp in [10, 17, 63, 64, 65, 1009, 70234, 1234567]:
            a = 2**exp
            self.assertEqual(a.bit_count(), 1)
            self.assertEqual((a - 1).bit_count(), exp)
            self.assertEqual((a ^ 63).bit_count(), 7)
            self.assertEqual(((a - 1) ^ 510).bit_count(), exp - 8)

    def test_round(self):
        # check round-half-even algorithm. For round to nearest ten;
        # rounding map is invariant under adding multiples of 20
        test_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0,
                     6:10, 7:10, 8:10, 9:10, 10:10, 11:10, 12:10, 13:10, 14:10,
                     15:20, 16:20, 17:20, 18:20, 19:20}
        for offset in range(-520, 520, 20):
            for k, v in test_dict.items():
                got = round(k+offset, -1)
                expected = v+offset
                self.assertEqual(got, expected)
                self.assertIs(type(got), int)

        # larger second argument
        self.assertEqual(round(-150, -2), -200)
        self.assertEqual(round(-149, -2), -100)
        self.assertEqual(round(-51, -2), -100)
        self.assertEqual(round(-50, -2), 0)
        self.assertEqual(round(-49, -2), 0)
        self.assertEqual(round(-1, -2), 0)
        self.assertEqual(round(0, -2), 0)
        self.assertEqual(round(1, -2), 0)
        self.assertEqual(round(49, -2), 0)
        self.assertEqual(round(50, -2), 0)
        self.assertEqual(round(51, -2), 100)
        self.assertEqual(round(149, -2), 100)
        self.assertEqual(round(150, -2), 200)
        self.assertEqual(round(250, -2), 200)
        self.assertEqual(round(251, -2), 300)
        self.assertEqual(round(172500, -3), 172000)
        self.assertEqual(round(173500, -3), 174000)
        self.assertEqual(round(31415926535, -1), 31415926540)
        self.assertEqual(round(31415926535, -2), 31415926500)
        self.assertEqual(round(31415926535, -3), 31415927000)
        self.assertEqual(round(31415926535, -4), 31415930000)
        self.assertEqual(round(31415926535, -5), 31415900000)
        self.assertEqual(round(31415926535, -6), 31416000000)
        self.assertEqual(round(31415926535, -7), 31420000000)
        self.assertEqual(round(31415926535, -8), 31400000000)
        self.assertEqual(round(31415926535, -9), 31000000000)
        self.assertEqual(round(31415926535, -10), 30000000000)
        self.assertEqual(round(31415926535, -11), 0)
        self.assertEqual(round(31415926535, -12), 0)
        self.assertEqual(round(31415926535, -999), 0)

        # should get correct results even for huge inputs
        for k in range(10, 100):
            got = round(10**k + 324678, -3)
            expect = 10**k + 325000
            self.assertEqual(got, expect)
            self.assertIs(type(got), int)

        # nonnegative second argument: round(x, n) should just return x
        for n in range(5):
            for i in range(100):
                x = random.randrange(-10000, 10000)
                got = round(x, n)
                self.assertEqual(got, x)
                self.assertIs(type(got), int)
        for huge_n in 2**31-1, 2**31, 2**63-1, 2**63, 2**100, 10**100:
            self.assertEqual(round(8979323, huge_n), 8979323)

        # omitted second argument
        for i in range(100):
            x = random.randrange(-10000, 10000)
            got = round(x)
            self.assertEqual(got, x)
            self.assertIs(type(got), int)

        # bad second argument
        bad_exponents = ('brian', 2.0, 0j)
        for e in bad_exponents:
            self.assertRaises(TypeError, round, 3, e)

    def test_to_bytes(self):
        def check(tests, byteorder, signed=False):
            def equivalent_python(n, length, byteorder, signed=False):
                if byteorder == 'little':
                    order = range(length)
                elif byteorder == 'big':
                    order = reversed(range(length))
                return bytes((n >> i*8) & 0xff for i in order)

            for test, expected in tests.items():
                try:
                    self.assertEqual(
                        test.to_bytes(len(expected), byteorder, signed=signed),
                        expected)
                except Exception as err:
                    raise AssertionError(
                        "failed to convert {} with byteorder={} and signed={}"
                        .format(test, byteorder, signed)) from err

                # Test for all default arguments.
                if len(expected) == 1 and byteorder == 'big' and not signed:
                    try:
                        self.assertEqual(test.to_bytes(), expected)
                    except Exception as err:
                        raise AssertionError(
                            "failed to convert {} with default arguments"
                            .format(test)) from err

                try:
                    self.assertEqual(
                        equivalent_python(
                            test, len(expected), byteorder, signed=signed),
                        expected
                    )
                except Exception as err:
                    raise AssertionError(
                        "Code equivalent from docs is not equivalent for "
                        "conversion of {0} with byteorder byteorder={1} and "
                        "signed={2}".format(test, byteorder, signed)) from err

        # Convert integers to signed big-endian byte arrays.
        tests1 = {
            0: b'\x00',
            1: b'\x01',
            -1: b'\xff',
            -127: b'\x81',
            -128: b'\x80',
            -129: b'\xff\x7f',
            127: b'\x7f',
            129: b'\x00\x81',
            -255: b'\xff\x01',
            -256: b'\xff\x00',
            255: b'\x00\xff',
            256: b'\x01\x00',
            32767: b'\x7f\xff',
            -32768: b'\xff\x80\x00',
            65535: b'\x00\xff\xff',
            -65536: b'\xff\x00\x00',
            -8388608: b'\x80\x00\x00'
        }
        check(tests1, 'big', signed=True)

        # Convert integers to signed little-endian byte arrays.
        tests2 = {
            0: b'\x00',
            1: b'\x01',
            -1: b'\xff',
            -127: b'\x81',
            -128: b'\x80',
            -129: b'\x7f\xff',
            127: b'\x7f',
            129: b'\x81\x00',
            -255: b'\x01\xff',
            -256: b'\x00\xff',
            255: b'\xff\x00',
            256: b'\x00\x01',
            32767: b'\xff\x7f',
            -32768: b'\x00\x80',
            65535: b'\xff\xff\x00',
            -65536: b'\x00\x00\xff',
            -8388608: b'\x00\x00\x80'
        }
        check(tests2, 'little', signed=True)

        # Convert integers to unsigned big-endian byte arrays.
        tests3 = {
            0: b'\x00',
            1: b'\x01',
            127: b'\x7f',
            128: b'\x80',
            255: b'\xff',
            256: b'\x01\x00',
            32767: b'\x7f\xff',
            32768: b'\x80\x00',
            65535: b'\xff\xff',
            65536: b'\x01\x00\x00'
        }
        check(tests3, 'big', signed=False)

        # Convert integers to unsigned little-endian byte arrays.
        tests4 = {
            0: b'\x00',
            1: b'\x01',
            127: b'\x7f',
            128: b'\x80',
            255: b'\xff',
            256: b'\x00\x01',
            32767: b'\xff\x7f',
            32768: b'\x00\x80',
            65535: b'\xff\xff',
            65536: b'\x00\x00\x01'
        }
        check(tests4, 'little', signed=False)

        self.assertRaises(OverflowError, (256).to_bytes, 1, 'big', signed=False)
        self.assertRaises(OverflowError, (256).to_bytes, 1, 'big', signed=True)
        self.assertRaises(OverflowError, (256).to_bytes, 1, 'little', signed=False)
        self.assertRaises(OverflowError, (256).to_bytes, 1, 'little', signed=True)
        self.assertRaises(OverflowError, (-1).to_bytes, 2, 'big', signed=False)
        self.assertRaises(OverflowError, (-1).to_bytes, 2, 'little', signed=False)
        self.assertEqual((0).to_bytes(0, 'big'), b'')
        self.assertEqual((1).to_bytes(5, 'big'), b'\x00\x00\x00\x00\x01')
        self.assertEqual((0).to_bytes(5, 'big'), b'\x00\x00\x00\x00\x00')
        self.assertEqual((-1).to_bytes(5, 'big', signed=True),
                         b'\xff\xff\xff\xff\xff')
        self.assertRaises(OverflowError, (1).to_bytes, 0, 'big')

        # gh-98783
        class SubStr(str):
            pass
        self.assertEqual((0).to_bytes(1, SubStr('big')), b'\x00')
        self.assertEqual((0).to_bytes(0, SubStr('little')), b'')

    def test_from_bytes(self):
        def check(tests, byteorder, signed=False):
            def equivalent_python(byte_array, byteorder, signed=False):
                if byteorder == 'little':
                    little_ordered = list(byte_array)
                elif byteorder == 'big':
                    little_ordered = list(reversed(byte_array))

                n = sum(b << i*8 for i, b in enumerate(little_ordered))
                if signed and little_ordered and (little_ordered[-1] & 0x80):
                    n -= 1 << 8*len(little_ordered)

                return n

            for test, expected in tests.items():
                try:
                    self.assertEqual(
                        int.from_bytes(test, byteorder, signed=signed),
                        expected)
                except Exception as err:
                    raise AssertionError(
                        "failed to convert {} with byteorder={!r} and signed={}"
                        .format(test, byteorder, signed)) from err

                # Test for all default arguments.
                if byteorder == 'big' and not signed:
                    try:
                        self.assertEqual(
                            int.from_bytes(test),
                            expected)
                    except Exception as err:
                        raise AssertionError(
                            "failed to convert {} with default arguments"
                            .format(test)) from err

                try:
                    self.assertEqual(
                        equivalent_python(test, byteorder, signed=signed),
                        expected
                    )
                except Exception as err:
                    raise AssertionError(
                        "Code equivalent from docs is not equivalent for "
                        "conversion of {0} with byteorder={1!r} and signed={2}"
                        .format(test, byteorder, signed)) from err

        # Convert signed big-endian byte arrays to integers.
        tests1 = {
            b'': 0,
            b'\x00': 0,
            b'\x00\x00': 0,
            b'\x01': 1,
            b'\x00\x01': 1,
            b'\xff': -1,
            b'\xff\xff': -1,
            b'\x81': -127,
            b'\x80': -128,
            b'\xff\x7f': -129,
            b'\x7f': 127,
            b'\x00\x81': 129,
            b'\xff\x01': -255,
            b'\xff\x00': -256,
            b'\x00\xff': 255,
            b'\x01\x00': 256,
            b'\x7f\xff': 32767,
            b'\x80\x00': -32768,
            b'\x00\xff\xff': 65535,
            b'\xff\x00\x00': -65536,
            b'\x80\x00\x00': -8388608
        }
        check(tests1, 'big', signed=True)

        # Convert signed little-endian byte arrays to integers.
        tests2 = {
            b'': 0,
            b'\x00': 0,
            b'\x00\x00': 0,
            b'\x01': 1,
            b'\x00\x01': 256,
            b'\xff': -1,
            b'\xff\xff': -1,
            b'\x81': -127,
            b'\x80': -128,
            b'\x7f\xff': -129,
            b'\x7f': 127,
            b'\x81\x00': 129,
            b'\x01\xff': -255,
            b'\x00\xff': -256,
            b'\xff\x00': 255,
            b'\x00\x01': 256,
            b'\xff\x7f': 32767,
            b'\x00\x80': -32768,
            b'\xff\xff\x00': 65535,
            b'\x00\x00\xff': -65536,
            b'\x00\x00\x80': -8388608
        }
        check(tests2, 'little', signed=True)

        # Convert unsigned big-endian byte arrays to integers.
        tests3 = {
            b'': 0,
            b'\x00': 0,
            b'\x01': 1,
            b'\x7f': 127,
            b'\x80': 128,
            b'\xff': 255,
            b'\x01\x00': 256,
            b'\x7f\xff': 32767,
            b'\x80\x00': 32768,
            b'\xff\xff': 65535,
            b'\x01\x00\x00': 65536,
        }
        check(tests3, 'big', signed=False)

        # Convert integers to unsigned little-endian byte arrays.
        tests4 = {
            b'': 0,
            b'\x00': 0,
            b'\x01': 1,
            b'\x7f': 127,
            b'\x80': 128,
            b'\xff': 255,
            b'\x00\x01': 256,
            b'\xff\x7f': 32767,
            b'\x00\x80': 32768,
            b'\xff\xff': 65535,
            b'\x00\x00\x01': 65536,
        }
        check(tests4, 'little', signed=False)

        class myint(int):
            pass

        self.assertIs(type(myint.from_bytes(b'\x00', 'big')), myint)
        self.assertEqual(myint.from_bytes(b'\x01', 'big'), 1)
        self.assertIs(
            type(myint.from_bytes(b'\x00', 'big', signed=False)), myint)
        self.assertEqual(myint.from_bytes(b'\x01', 'big', signed=False), 1)
        self.assertIs(type(myint.from_bytes(b'\x00', 'little')), myint)
        self.assertEqual(myint.from_bytes(b'\x01', 'little'), 1)
        self.assertIs(type(myint.from_bytes(
            b'\x00', 'little', signed=False)), myint)
        self.assertEqual(myint.from_bytes(b'\x01', 'little', signed=False), 1)
        self.assertEqual(
            int.from_bytes([255, 0, 0], 'big', signed=True), -65536)
        self.assertEqual(
            int.from_bytes((255, 0, 0), 'big', signed=True), -65536)
        self.assertEqual(int.from_bytes(
            bytearray(b'\xff\x00\x00'), 'big', signed=True), -65536)
        self.assertEqual(int.from_bytes(
            bytearray(b'\xff\x00\x00'), 'big', signed=True), -65536)
        self.assertEqual(int.from_bytes(
            array.array('B', b'\xff\x00\x00'), 'big', signed=True), -65536)
        self.assertEqual(int.from_bytes(
            memoryview(b'\xff\x00\x00'), 'big', signed=True), -65536)
        self.assertRaises(ValueError, int.from_bytes, [256], 'big')
        self.assertRaises(ValueError, int.from_bytes, [0], 'big\x00')
        self.assertRaises(ValueError, int.from_bytes, [0], 'little\x00')
        self.assertRaises(TypeError, int.from_bytes, "", 'big')
        self.assertRaises(TypeError, int.from_bytes, "\x00", 'big')
        self.assertRaises(TypeError, int.from_bytes, 0, 'big')
        self.assertRaises(TypeError, int.from_bytes, 0, 'big', True)
        self.assertRaises(TypeError, myint.from_bytes, "", 'big')
        self.assertRaises(TypeError, myint.from_bytes, "\x00", 'big')
        self.assertRaises(TypeError, myint.from_bytes, 0, 'big')
        self.assertRaises(TypeError, int.from_bytes, 0, 'big', True)

        class myint2(int):
            def __new__(cls, value):
                return int.__new__(cls, value + 1)

        i = myint2.from_bytes(b'\x01', 'big')
        self.assertIs(type(i), myint2)
        self.assertEqual(i, 2)

        class myint3(int):
            def __init__(self, value):
                self.foo = 'bar'

        i = myint3.from_bytes(b'\x01', 'big')
        self.assertIs(type(i), myint3)
        self.assertEqual(i, 1)
        self.assertEqual(getattr(i, 'foo', 'none'), 'bar')

        class ValidBytes:
            def __bytes__(self):
                return b'\x01'
        class InvalidBytes:
            def __bytes__(self):
                return 'abc'
        class MissingBytes: ...
        class RaisingBytes:
            def __bytes__(self):
                1 / 0

        self.assertEqual(int.from_bytes(ValidBytes()), 1)
        self.assertRaises(TypeError, int.from_bytes, InvalidBytes())
        self.assertRaises(TypeError, int.from_bytes, MissingBytes())
        self.assertRaises(ZeroDivisionError, int.from_bytes, RaisingBytes())

        # gh-98783
        class SubStr(str):
            pass
        self.assertEqual(int.from_bytes(b'', SubStr('big')), 0)
        self.assertEqual(int.from_bytes(b'\x00', SubStr('little')), 0)

    @support.cpython_only
    def test_from_bytes_small(self):
        # bpo-46361
        for i in range(-5, 257):
            b = i.to_bytes(2, signed=True)
            self.assertIs(int.from_bytes(b, signed=True), i)

    def test_is_integer(self):
        self.assertTrue((-1).is_integer())
        self.assertTrue((0).is_integer())
        self.assertTrue((1).is_integer())

    def test_access_to_nonexistent_digit_0(self):
        # http://bugs.python.org/issue14630: A bug in _PyLong_Copy meant that
        # ob_digit[0] was being incorrectly accessed for instances of a
        # subclass of int, with value 0.
        class Integer(int):
            def __new__(cls, value=0):
                self = int.__new__(cls, value)
                self.foo = 'foo'
                return self

        integers = [Integer(0) for i in range(1000)]
        for n in map(int, integers):
            self.assertEqual(n, 0)

    def test_shift_bool(self):
        # Issue #21422: ensure that bool << int and bool >> int return int
        for value in (True, False):
            for shift in (0, 2):
                self.assertEqual(type(value << shift), int)
                self.assertEqual(type(value >> shift), int)

    def test_as_integer_ratio(self):
        class myint(int):
            pass
        tests = [10, 0, -10, 1, sys.maxsize + 1, True, False, myint(42)]
        for value in tests:
            numerator, denominator = value.as_integer_ratio()
            self.assertEqual((numerator, denominator), (int(value), 1))
            self.assertEqual(type(numerator), int)
            self.assertEqual(type(denominator), int)

    def test_square(self):
        # Multiplication makes a special case of multiplying an int with
        # itself, using a special, faster algorithm. This test is mostly
        # to ensure that no asserts in the implementation trigger, in
        # cases with a maximal amount of carries.
        for bitlen in range(1, 400):
            n = (1 << bitlen) - 1 # solid string of 1 bits
            with self.subTest(bitlen=bitlen, n=n):
                # (2**i - 1)**2 = 2**(2*i) - 2*2**i + 1
                self.assertEqual(n**2,
                    (1 << (2 * bitlen)) - (1 << (bitlen + 1)) + 1)

    def test___sizeof__(self):
        self.assertEqual(int.__itemsize__, sys.int_info.sizeof_digit)

        # Pairs (test_value, number of allocated digits)
        test_values = [
            # We always allocate space for at least one digit, even for
            # a value of zero; sys.getsizeof should reflect that.
            (0, 1),
            (1, 1),
            (-1, 1),
            (BASE-1, 1),
            (1-BASE, 1),
            (BASE, 2),
            (-BASE, 2),
            (BASE*BASE - 1, 2),
            (BASE*BASE, 3),
        ]

        for value, ndigits in test_values:
            with self.subTest(value):
                self.assertEqual(
                    value.__sizeof__(),
                    int.__basicsize__ + int.__itemsize__ * ndigits
                )

        # Same test for a subclass of int.
        class MyInt(int):
            pass

        self.assertEqual(MyInt.__itemsize__, sys.int_info.sizeof_digit)

        for value, ndigits in test_values:
            with self.subTest(value):
                self.assertEqual(
                    MyInt(value).__sizeof__(),
                    MyInt.__basicsize__ + MyInt.__itemsize__ * ndigits
                )

        # GH-117195 -- This shouldn't crash
        object.__sizeof__(1)

if __name__ == "__main__":
    run_tests()
