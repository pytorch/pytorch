# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_defaultdict.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
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


"""Unit tests for collections.defaultdict."""

import copy
import pickle
import unittest

from collections import defaultdict

def foobar():
    return list

class TestDefaultDict(__TestCase):

    def test_basic(self):
        d1 = defaultdict()
        self.assertEqual(d1.default_factory, None)
        d1.default_factory = list
        d1[12].append(42)
        self.assertEqual(d1, {12: [42]})
        d1[12].append(24)
        self.assertEqual(d1, {12: [42, 24]})
        d1[13]
        d1[14]
        self.assertEqual(d1, {12: [42, 24], 13: [], 14: []})
        self.assertTrue(d1[12] is not d1[13] is not d1[14])
        d2 = defaultdict(list, foo=1, bar=2)
        self.assertEqual(d2.default_factory, list)
        self.assertEqual(d2, {"foo": 1, "bar": 2})
        self.assertEqual(d2["foo"], 1)
        self.assertEqual(d2["bar"], 2)
        self.assertEqual(d2[42], [])
        self.assertIn("foo", d2)
        self.assertIn("foo", d2.keys())
        self.assertIn("bar", d2)
        self.assertIn("bar", d2.keys())
        self.assertIn(42, d2)
        self.assertIn(42, d2.keys())
        self.assertNotIn(12, d2)
        self.assertNotIn(12, d2.keys())
        d2.default_factory = None
        self.assertEqual(d2.default_factory, None)
        try:
            d2[15]
        except KeyError as err:
            self.assertEqual(err.args, (15,))
        else:
            self.fail("d2[15] didn't raise KeyError")
        self.assertRaises(TypeError, defaultdict, 1)

    def test_missing(self):
        d1 = defaultdict()
        self.assertRaises(KeyError, d1.__missing__, 42)
        d1.default_factory = list
        self.assertEqual(d1.__missing__(42), [])

    def test_repr(self):
        d1 = defaultdict()
        self.assertEqual(d1.default_factory, None)
        self.assertEqual(repr(d1), "defaultdict(None, {})")
        self.assertEqual(eval(repr(d1)), d1)
        d1[11] = 41
        self.assertEqual(repr(d1), "defaultdict(None, {11: 41})")
        d2 = defaultdict(int)
        self.assertEqual(d2.default_factory, int)
        d2[12] = 42
        self.assertEqual(repr(d2), "defaultdict(<class 'int'>, {12: 42})")
        def foo(): return 43
        d3 = defaultdict(foo)
        self.assertTrue(d3.default_factory is foo)
        d3[13]
        self.assertEqual(repr(d3), "defaultdict(%s, {13: 43})" % repr(foo))

    def test_copy(self):
        d1 = defaultdict()
        d2 = d1.copy()
        self.assertEqual(type(d2), defaultdict)
        self.assertEqual(d2.default_factory, None)
        self.assertEqual(d2, {})
        d1.default_factory = list
        d3 = d1.copy()
        self.assertEqual(type(d3), defaultdict)
        self.assertEqual(d3.default_factory, list)
        self.assertEqual(d3, {})
        d1[42]
        d4 = d1.copy()
        self.assertEqual(type(d4), defaultdict)
        self.assertEqual(d4.default_factory, list)
        self.assertEqual(d4, {42: []})
        d4[12]
        self.assertEqual(d4, {42: [], 12: []})

        # Issue 6637: Copy fails for empty default dict
        d = defaultdict()
        d['a'] = 42
        e = d.copy()
        self.assertEqual(e['a'], 42)

    def test_shallow_copy(self):
        d1 = defaultdict(foobar, {1: 1})
        d2 = copy.copy(d1)
        self.assertEqual(d2.default_factory, foobar)
        self.assertEqual(d2, d1)
        d1.default_factory = list
        d2 = copy.copy(d1)
        self.assertEqual(d2.default_factory, list)
        self.assertEqual(d2, d1)

    def test_deep_copy(self):
        d1 = defaultdict(foobar, {1: [1]})
        d2 = copy.deepcopy(d1)
        self.assertEqual(d2.default_factory, foobar)
        self.assertEqual(d2, d1)
        self.assertTrue(d1[1] is not d2[1])
        d1.default_factory = list
        d2 = copy.deepcopy(d1)
        self.assertEqual(d2.default_factory, list)
        self.assertEqual(d2, d1)

    def test_keyerror_without_factory(self):
        d1 = defaultdict()
        try:
            d1[(1,)]
        except KeyError as err:
            self.assertEqual(err.args[0], (1,))
        else:
            self.fail("expected KeyError")

    def test_recursive_repr(self):
        # Issue2045: stack overflow when default_factory is a bound method
        with torch._dynamo.set_fullgraph(fullgraph=False):
            class sub(defaultdict):
                def __init__(self):
                    self.default_factory = self._factory
                def _factory(self):
                    return []
        d = sub()
        self.assertRegex(repr(d),
            r"sub\(<bound method .*sub\._factory "
            r"of sub\(\.\.\., \{\}\)>, \{\}\)")

    def test_callable_arg(self):
        self.assertRaises(TypeError, defaultdict, {})

    def test_pickling(self):
        d = defaultdict(int)
        d[1]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            s = pickle.dumps(d, proto)
            o = pickle.loads(s)
            self.assertEqual(d, o)

    def test_union(self):
        i = defaultdict(int, {1: 1, 2: 2})
        s = defaultdict(str, {0: "zero", 1: "one"})

        i_s = i | s
        self.assertIs(i_s.default_factory, int)
        self.assertDictEqual(i_s, {1: "one", 2: 2, 0: "zero"})
        self.assertEqual(list(i_s), [1, 2, 0])

        s_i = s | i
        self.assertIs(s_i.default_factory, str)
        self.assertDictEqual(s_i, {0: "zero", 1: 1, 2: 2})
        self.assertEqual(list(s_i), [0, 1, 2])

        i_ds = i | dict(s)
        self.assertIs(i_ds.default_factory, int)
        self.assertDictEqual(i_ds, {1: "one", 2: 2, 0: "zero"})
        self.assertEqual(list(i_ds), [1, 2, 0])

        ds_i = dict(s) | i
        self.assertIs(ds_i.default_factory, int)
        self.assertDictEqual(ds_i, {0: "zero", 1: 1, 2: 2})
        self.assertEqual(list(ds_i), [0, 1, 2])

        with self.assertRaises(TypeError):
            i | list(s.items())
        with self.assertRaises(TypeError):
            list(s.items()) | i

        # We inherit a fine |= from dict, so just a few sanity checks here:
        i |= list(s.items())
        self.assertIs(i.default_factory, int)
        self.assertDictEqual(i, {1: "one", 2: 2, 0: "zero"})
        self.assertEqual(list(i), [1, 2, 0])

        with self.assertRaises(TypeError):
            i |= None

if __name__ == "__main__":
    run_tests()
