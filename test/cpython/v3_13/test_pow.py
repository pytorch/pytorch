# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_pow.py

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


import math
import unittest

class PowTest(__TestCase):

    def powtest(self, type):
        if type != float:
            for i in range(-1000, 1000):
                self.assertEqual(pow(type(i), 0), 1)
                self.assertEqual(pow(type(i), 1), type(i))
                self.assertEqual(pow(type(0), 1), type(0))
                self.assertEqual(pow(type(1), 1), type(1))

            for i in range(-100, 100):
                self.assertEqual(pow(type(i), 3), i*i*i)

            pow2 = 1
            for i in range(0, 31):
                self.assertEqual(pow(2, i), pow2)
                if i != 30 : pow2 = pow2*2

            for i in list(range(-10, 0)) + list(range(1, 10)):
                ii = type(i)
                inv = pow(ii, -1) # inverse of ii
                for jj in range(-10, 0):
                    self.assertAlmostEqual(pow(ii, jj), pow(inv, -jj))

        for othertype in int, float:
            for i in range(1, 100):
                zero = type(0)
                exp = -othertype(i/10.0)
                if exp == 0:
                    continue
                self.assertRaises(ZeroDivisionError, pow, zero, exp)

        il, ih = -20, 20
        jl, jh = -5,   5
        kl, kh = -10, 10
        asseq = self.assertEqual
        if type == float:
            il = 1
            asseq = self.assertAlmostEqual
        elif type == int:
            jl = 0
        elif type == int:
            jl, jh = 0, 15
        for i in range(il, ih+1):
            for j in range(jl, jh+1):
                for k in range(kl, kh+1):
                    if k != 0:
                        if type == float or j < 0:
                            self.assertRaises(TypeError, pow, type(i), j, k)
                            continue
                        asseq(
                            pow(type(i),j,k),
                            pow(type(i),j)% type(k)
                        )

    def test_powint(self):
        self.powtest(int)

    def test_powfloat(self):
        self.powtest(float)

    def test_other(self):
        # Other tests-- not very systematic
        self.assertEqual(pow(3,3) % 8, pow(3,3,8))
        self.assertEqual(pow(3,3) % -8, pow(3,3,-8))
        self.assertEqual(pow(3,2) % -2, pow(3,2,-2))
        self.assertEqual(pow(-3,3) % 8, pow(-3,3,8))
        self.assertEqual(pow(-3,3) % -8, pow(-3,3,-8))
        self.assertEqual(pow(5,2) % -8, pow(5,2,-8))

        self.assertEqual(pow(3,3) % 8, pow(3,3,8))
        self.assertEqual(pow(3,3) % -8, pow(3,3,-8))
        self.assertEqual(pow(3,2) % -2, pow(3,2,-2))
        self.assertEqual(pow(-3,3) % 8, pow(-3,3,8))
        self.assertEqual(pow(-3,3) % -8, pow(-3,3,-8))
        self.assertEqual(pow(5,2) % -8, pow(5,2,-8))

        for i in range(-10, 11):
            for j in range(0, 6):
                for k in range(-7, 11):
                    if j >= 0 and k != 0:
                        self.assertEqual(
                            pow(i,j) % k,
                            pow(i,j,k)
                        )
                    if j >= 0 and k != 0:
                        self.assertEqual(
                            pow(int(i),j) % k,
                            pow(int(i),j,k)
                        )

    def test_big_exp(self):
        import random
        self.assertEqual(pow(2, 50000), 1 << 50000)
        # Randomized modular tests, checking the identities
        #  a**(b1 + b2) == a**b1 * a**b2
        #  a**(b1 * b2) == (a**b1)**b2
        prime = 1000000000039 # for speed, relatively small prime modulus
        for i in range(10):
            a = random.randrange(1000, 1000000)
            bpower = random.randrange(1000, 50000)
            b = random.randrange(1 << (bpower - 1), 1 << bpower)
            b1 = random.randrange(1, b)
            b2 = b - b1
            got1 = pow(a, b, prime)
            got2 = pow(a, b1, prime) * pow(a, b2, prime) % prime
            if got1 != got2:
                self.fail(f"{a=:x} {b1=:x} {b2=:x} {got1=:x} {got2=:x}")
            got3 = pow(a, b1 * b2, prime)
            got4 = pow(pow(a, b1, prime), b2, prime)
            if got3 != got4:
                self.fail(f"{a=:x} {b1=:x} {b2=:x} {got3=:x} {got4=:x}")

    def test_bug643260(self):
        class TestRpow:
            def __rpow__(self, other):
                return None
        None ** TestRpow() # Won't fail when __rpow__ invoked.  SF bug #643260.

    def test_bug705231(self):
        # -1.0 raised to an integer should never blow up.  It did if the
        # platform pow() was buggy, and Python didn't worm around it.
        eq = self.assertEqual
        a = -1.0
        # The next two tests can still fail if the platform floor()
        # function doesn't treat all large inputs as integers
        # test_math should also fail if that is happening
        eq(pow(a, 1.23e167), 1.0)
        eq(pow(a, -1.23e167), 1.0)
        for b in range(-10, 11):
            eq(pow(a, float(b)), b & 1 and -1.0 or 1.0)
        for n in range(0, 100):
            fiveto = float(5 ** n)
            # For small n, fiveto will be odd.  Eventually we run out of
            # mantissa bits, though, and thereafer fiveto will be even.
            expected = fiveto % 2.0 and -1.0 or 1.0
            eq(pow(a, fiveto), expected)
            eq(pow(a, -fiveto), expected)
        eq(expected, 1.0)   # else we didn't push fiveto to evenness

    def test_negative_exponent(self):
        for a in range(-50, 50):
            for m in range(-50, 50):
                with self.subTest(a=a, m=m):
                    if m != 0 and math.gcd(a, m) == 1:
                        # Exponent -1 should give an inverse, with the
                        # same sign as m.
                        inv = pow(a, -1, m)
                        self.assertEqual(inv, inv % m)
                        self.assertEqual((inv * a - 1) % m, 0)

                        # Larger exponents
                        self.assertEqual(pow(a, -2, m), pow(inv, 2, m))
                        self.assertEqual(pow(a, -3, m), pow(inv, 3, m))
                        self.assertEqual(pow(a, -1001, m), pow(inv, 1001, m))

                    else:
                        with self.assertRaises(ValueError):
                            pow(a, -1, m)
                        with self.assertRaises(ValueError):
                            pow(a, -2, m)
                        with self.assertRaises(ValueError):
                            pow(a, -1001, m)


if __name__ == "__main__":
    run_tests()
