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

# UserString is a wrapper around the native builtin string type.
# UserString instances should behave similar to builtin string objects.

import unittest
import string_tests

from collections import UserString

class UserStringTest(
    string_tests.StringLikeTest,
    __TestCase
    ):

    type2test = UserString

    # Overwrite the three testing methods, because UserString
    # can't cope with arguments propagated to UserString
    # (and we don't test with subclasses)
    def checkequal(self, result, object, methodname, *args, **kwargs):
        result = self.fixtype(result)
        object = self.fixtype(object)
        # we don't fix the arguments, because UserString can't cope with it
        realresult = getattr(object, methodname)(*args, **kwargs)
        self.assertEqual(
            result,
            realresult
        )

    def checkraises(self, exc, obj, methodname, *args, expected_msg=None):
        obj = self.fixtype(obj)
        # we don't fix the arguments, because UserString can't cope with it
        with self.assertRaises(exc) as cm:
            getattr(obj, methodname)(*args)
        self.assertNotEqual(str(cm.exception), '')
        if expected_msg is not None:
            self.assertEqual(str(cm.exception), expected_msg)

    def checkcall(self, object, methodname, *args):
        object = self.fixtype(object)
        # we don't fix the arguments, because UserString can't cope with it
        getattr(object, methodname)(*args)

    def test_rmod(self):
        class ustr2(UserString):
            pass

        class ustr3(ustr2):
            def __rmod__(self, other):
                return super().__rmod__(other)

        fmt2 = ustr2('value is %s')
        str3 = ustr3('TEST')
        self.assertEqual(fmt2 % str3, 'value is TEST')

    def test_encode_default_args(self):
        self.checkequal(b'hello', 'hello', 'encode')
        # Check that encoding defaults to utf-8
        self.checkequal(b'\xf0\xa3\x91\x96', '\U00023456', 'encode')
        # Check that errors defaults to 'strict'
        self.checkraises(UnicodeError, '\ud800', 'encode')

    def test_encode_explicit_none_args(self):
        self.checkequal(b'hello', 'hello', 'encode', None, None)
        # Check that encoding defaults to utf-8
        self.checkequal(b'\xf0\xa3\x91\x96', '\U00023456', 'encode', None, None)
        # Check that errors defaults to 'strict'
        self.checkraises(UnicodeError, '\ud800', 'encode', None, None)


if __name__ == "__main__":
    if TEST_WITH_TORCHDYNAMO:
        run_tests()
    else:
        unittest.main()
