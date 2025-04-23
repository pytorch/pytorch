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

"""Test correct treatment of hex/oct constants.

This is complex because of changes due to PEP 237.
"""

import unittest

class TestHexOctBin(__TestCase):

    def test_hex_baseline(self):
        # A few upper/lowercase tests
        self.assertEqual(0x0, 0X0)
        self.assertEqual(0x1, 0X1)
        self.assertEqual(0x123456789abcdef, 0X123456789abcdef)
        # Baseline tests
        self.assertEqual(0x0, 0)
        self.assertEqual(0x10, 16)
        self.assertEqual(0x7fffffff, 2147483647)
        self.assertEqual(0x7fffffffffffffff, 9223372036854775807)
        # Ditto with a minus sign and parentheses
        self.assertEqual(-(0x0), 0)
        self.assertEqual(-(0x10), -16)
        self.assertEqual(-(0x7fffffff), -2147483647)
        self.assertEqual(-(0x7fffffffffffffff), -9223372036854775807)
        # Ditto with a minus sign and NO parentheses
        self.assertEqual(-0x0, 0)
        self.assertEqual(-0x10, -16)
        self.assertEqual(-0x7fffffff, -2147483647)
        self.assertEqual(-0x7fffffffffffffff, -9223372036854775807)

    def test_hex_unsigned(self):
        # Positive constants
        self.assertEqual(0x80000000, 2147483648)
        self.assertEqual(0xffffffff, 4294967295)
        # Ditto with a minus sign and parentheses
        self.assertEqual(-(0x80000000), -2147483648)
        self.assertEqual(-(0xffffffff), -4294967295)
        # Ditto with a minus sign and NO parentheses
        # This failed in Python 2.2 through 2.2.2 and in 2.3a1
        self.assertEqual(-0x80000000, -2147483648)
        self.assertEqual(-0xffffffff, -4294967295)

        # Positive constants
        self.assertEqual(0x8000000000000000, 9223372036854775808)
        self.assertEqual(0xffffffffffffffff, 18446744073709551615)
        # Ditto with a minus sign and parentheses
        self.assertEqual(-(0x8000000000000000), -9223372036854775808)
        self.assertEqual(-(0xffffffffffffffff), -18446744073709551615)
        # Ditto with a minus sign and NO parentheses
        # This failed in Python 2.2 through 2.2.2 and in 2.3a1
        self.assertEqual(-0x8000000000000000, -9223372036854775808)
        self.assertEqual(-0xffffffffffffffff, -18446744073709551615)

    def test_oct_baseline(self):
        # A few upper/lowercase tests
        self.assertEqual(0o0, 0O0)
        self.assertEqual(0o1, 0O1)
        self.assertEqual(0o1234567, 0O1234567)
        # Baseline tests
        self.assertEqual(0o0, 0)
        self.assertEqual(0o20, 16)
        self.assertEqual(0o17777777777, 2147483647)
        self.assertEqual(0o777777777777777777777, 9223372036854775807)
        # Ditto with a minus sign and parentheses
        self.assertEqual(-(0o0), 0)
        self.assertEqual(-(0o20), -16)
        self.assertEqual(-(0o17777777777), -2147483647)
        self.assertEqual(-(0o777777777777777777777), -9223372036854775807)
        # Ditto with a minus sign and NO parentheses
        self.assertEqual(-0o0, 0)
        self.assertEqual(-0o20, -16)
        self.assertEqual(-0o17777777777, -2147483647)
        self.assertEqual(-0o777777777777777777777, -9223372036854775807)

    def test_oct_unsigned(self):
        # Positive constants
        self.assertEqual(0o20000000000, 2147483648)
        self.assertEqual(0o37777777777, 4294967295)
        # Ditto with a minus sign and parentheses
        self.assertEqual(-(0o20000000000), -2147483648)
        self.assertEqual(-(0o37777777777), -4294967295)
        # Ditto with a minus sign and NO parentheses
        # This failed in Python 2.2 through 2.2.2 and in 2.3a1
        self.assertEqual(-0o20000000000, -2147483648)
        self.assertEqual(-0o37777777777, -4294967295)

        # Positive constants
        self.assertEqual(0o1000000000000000000000, 9223372036854775808)
        self.assertEqual(0o1777777777777777777777, 18446744073709551615)
        # Ditto with a minus sign and parentheses
        self.assertEqual(-(0o1000000000000000000000), -9223372036854775808)
        self.assertEqual(-(0o1777777777777777777777), -18446744073709551615)
        # Ditto with a minus sign and NO parentheses
        # This failed in Python 2.2 through 2.2.2 and in 2.3a1
        self.assertEqual(-0o1000000000000000000000, -9223372036854775808)
        self.assertEqual(-0o1777777777777777777777, -18446744073709551615)

    def test_bin_baseline(self):
        # A few upper/lowercase tests
        self.assertEqual(0b0, 0B0)
        self.assertEqual(0b1, 0B1)
        self.assertEqual(0b10101010101, 0B10101010101)
        # Baseline tests
        self.assertEqual(0b0, 0)
        self.assertEqual(0b10000, 16)
        self.assertEqual(0b1111111111111111111111111111111, 2147483647)
        self.assertEqual(0b111111111111111111111111111111111111111111111111111111111111111, 9223372036854775807)
        # Ditto with a minus sign and parentheses
        self.assertEqual(-(0b0), 0)
        self.assertEqual(-(0b10000), -16)
        self.assertEqual(-(0b1111111111111111111111111111111), -2147483647)
        self.assertEqual(-(0b111111111111111111111111111111111111111111111111111111111111111), -9223372036854775807)
        # Ditto with a minus sign and NO parentheses
        self.assertEqual(-0b0, 0)
        self.assertEqual(-0b10000, -16)
        self.assertEqual(-0b1111111111111111111111111111111, -2147483647)
        self.assertEqual(-0b111111111111111111111111111111111111111111111111111111111111111, -9223372036854775807)

    def test_bin_unsigned(self):
        # Positive constants
        self.assertEqual(0b10000000000000000000000000000000, 2147483648)
        self.assertEqual(0b11111111111111111111111111111111, 4294967295)
        # Ditto with a minus sign and parentheses
        self.assertEqual(-(0b10000000000000000000000000000000), -2147483648)
        self.assertEqual(-(0b11111111111111111111111111111111), -4294967295)
        # Ditto with a minus sign and NO parentheses
        # This failed in Python 2.2 through 2.2.2 and in 2.3a1
        self.assertEqual(-0b10000000000000000000000000000000, -2147483648)
        self.assertEqual(-0b11111111111111111111111111111111, -4294967295)

        # Positive constants
        self.assertEqual(0b1000000000000000000000000000000000000000000000000000000000000000, 9223372036854775808)
        self.assertEqual(0b1111111111111111111111111111111111111111111111111111111111111111, 18446744073709551615)
        # Ditto with a minus sign and parentheses
        self.assertEqual(-(0b1000000000000000000000000000000000000000000000000000000000000000), -9223372036854775808)
        self.assertEqual(-(0b1111111111111111111111111111111111111111111111111111111111111111), -18446744073709551615)
        # Ditto with a minus sign and NO parentheses
        # This failed in Python 2.2 through 2.2.2 and in 2.3a1
        self.assertEqual(-0b1000000000000000000000000000000000000000000000000000000000000000, -9223372036854775808)
        self.assertEqual(-0b1111111111111111111111111111111111111111111111111111111111111111, -18446744073709551615)

if __name__ == "__main__":
    if TEST_WITH_TORCHDYNAMO:
        run_tests()
    else:
        unittest.main()
