# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_bisect.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import sys
import unittest
from test.support import import_helper
from collections import UserList


py_bisect = import_helper.import_fresh_module('bisect', blocked=['_bisect'])
c_bisect = import_helper.import_fresh_module('bisect', fresh=['_bisect'])

class Range(object):
    """A trivial range()-like object that has an insert() method."""
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.last_insert = None

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, idx):
        n = self.stop - self.start
        if idx < 0:
            idx += n
        if idx >= n:
            raise IndexError(idx)
        return self.start + idx

    def insert(self, idx, item):
        self.last_insert = idx, item


class _TestBisect:
    def setUp(self):
        super().setUp()
        self.precomputedCases = [
            (self.module.bisect_right, [], 1, 0),
            (self.module.bisect_right, [1], 0, 0),
            (self.module.bisect_right, [1], 1, 1),
            (self.module.bisect_right, [1], 2, 1),
            (self.module.bisect_right, [1, 1], 0, 0),
            (self.module.bisect_right, [1, 1], 1, 2),
            (self.module.bisect_right, [1, 1], 2, 2),
            (self.module.bisect_right, [1, 1, 1], 0, 0),
            (self.module.bisect_right, [1, 1, 1], 1, 3),
            (self.module.bisect_right, [1, 1, 1], 2, 3),
            (self.module.bisect_right, [1, 1, 1, 1], 0, 0),
            (self.module.bisect_right, [1, 1, 1, 1], 1, 4),
            (self.module.bisect_right, [1, 1, 1, 1], 2, 4),
            (self.module.bisect_right, [1, 2], 0, 0),
            (self.module.bisect_right, [1, 2], 1, 1),
            (self.module.bisect_right, [1, 2], 1.5, 1),
            (self.module.bisect_right, [1, 2], 2, 2),
            (self.module.bisect_right, [1, 2], 3, 2),
            (self.module.bisect_right, [1, 1, 2, 2], 0, 0),
            (self.module.bisect_right, [1, 1, 2, 2], 1, 2),
            (self.module.bisect_right, [1, 1, 2, 2], 1.5, 2),
            (self.module.bisect_right, [1, 1, 2, 2], 2, 4),
            (self.module.bisect_right, [1, 1, 2, 2], 3, 4),
            (self.module.bisect_right, [1, 2, 3], 0, 0),
            (self.module.bisect_right, [1, 2, 3], 1, 1),
            (self.module.bisect_right, [1, 2, 3], 1.5, 1),
            (self.module.bisect_right, [1, 2, 3], 2, 2),
            (self.module.bisect_right, [1, 2, 3], 2.5, 2),
            (self.module.bisect_right, [1, 2, 3], 3, 3),
            (self.module.bisect_right, [1, 2, 3], 4, 3),
            (self.module.bisect_right, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 0, 0),
            (self.module.bisect_right, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 1, 1),
            (self.module.bisect_right, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 1.5, 1),
            (self.module.bisect_right, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 2, 3),
            (self.module.bisect_right, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 2.5, 3),
            (self.module.bisect_right, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 3, 6),
            (self.module.bisect_right, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 3.5, 6),
            (self.module.bisect_right, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 4, 10),
            (self.module.bisect_right, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 5, 10),

            (self.module.bisect_left, [], 1, 0),
            (self.module.bisect_left, [1], 0, 0),
            (self.module.bisect_left, [1], 1, 0),
            (self.module.bisect_left, [1], 2, 1),
            (self.module.bisect_left, [1, 1], 0, 0),
            (self.module.bisect_left, [1, 1], 1, 0),
            (self.module.bisect_left, [1, 1], 2, 2),
            (self.module.bisect_left, [1, 1, 1], 0, 0),
            (self.module.bisect_left, [1, 1, 1], 1, 0),
            (self.module.bisect_left, [1, 1, 1], 2, 3),
            (self.module.bisect_left, [1, 1, 1, 1], 0, 0),
            (self.module.bisect_left, [1, 1, 1, 1], 1, 0),
            (self.module.bisect_left, [1, 1, 1, 1], 2, 4),
            (self.module.bisect_left, [1, 2], 0, 0),
            (self.module.bisect_left, [1, 2], 1, 0),
            (self.module.bisect_left, [1, 2], 1.5, 1),
            (self.module.bisect_left, [1, 2], 2, 1),
            (self.module.bisect_left, [1, 2], 3, 2),
            (self.module.bisect_left, [1, 1, 2, 2], 0, 0),
            (self.module.bisect_left, [1, 1, 2, 2], 1, 0),
            (self.module.bisect_left, [1, 1, 2, 2], 1.5, 2),
            (self.module.bisect_left, [1, 1, 2, 2], 2, 2),
            (self.module.bisect_left, [1, 1, 2, 2], 3, 4),
            (self.module.bisect_left, [1, 2, 3], 0, 0),
            (self.module.bisect_left, [1, 2, 3], 1, 0),
            (self.module.bisect_left, [1, 2, 3], 1.5, 1),
            (self.module.bisect_left, [1, 2, 3], 2, 1),
            (self.module.bisect_left, [1, 2, 3], 2.5, 2),
            (self.module.bisect_left, [1, 2, 3], 3, 2),
            (self.module.bisect_left, [1, 2, 3], 4, 3),
            (self.module.bisect_left, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 0, 0),
            (self.module.bisect_left, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 1, 0),
            (self.module.bisect_left, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 1.5, 1),
            (self.module.bisect_left, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 2, 1),
            (self.module.bisect_left, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 2.5, 3),
            (self.module.bisect_left, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 3, 3),
            (self.module.bisect_left, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 3.5, 6),
            (self.module.bisect_left, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 4, 6),
            (self.module.bisect_left, [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 5, 10)
        ]

    def test_precomputed(self):
        for func, data, elem, expected in self.precomputedCases:
            self.assertEqual(func(data, elem), expected)
            self.assertEqual(func(UserList(data), elem), expected)

    def test_negative_lo(self):
        # Issue 3301
        mod = self.module
        self.assertRaises(ValueError, mod.bisect_left, [1, 2, 3], 5, -1, 3)
        self.assertRaises(ValueError, mod.bisect_right, [1, 2, 3], 5, -1, 3)
        self.assertRaises(ValueError, mod.insort_left, [1, 2, 3], 5, -1, 3)
        self.assertRaises(ValueError, mod.insort_right, [1, 2, 3], 5, -1, 3)

    def test_large_range(self):
        # Issue 13496
        mod = self.module
        n = sys.maxsize
        data = range(n-1)
        self.assertEqual(mod.bisect_left(data, n-3), n-3)
        self.assertEqual(mod.bisect_right(data, n-3), n-2)
        self.assertEqual(mod.bisect_left(data, n-3, n-10, n), n-3)
        self.assertEqual(mod.bisect_right(data, n-3, n-10, n), n-2)

    def test_large_pyrange(self):
        # Same as above, but without C-imposed limits on range() parameters
        mod = self.module
        n = sys.maxsize
        data = Range(0, n-1)
        self.assertEqual(mod.bisect_left(data, n-3), n-3)
        self.assertEqual(mod.bisect_right(data, n-3), n-2)
        self.assertEqual(mod.bisect_left(data, n-3, n-10, n), n-3)
        self.assertEqual(mod.bisect_right(data, n-3, n-10, n), n-2)
        x = n - 100
        mod.insort_left(data, x, x - 50, x + 50)
        self.assertEqual(data.last_insert, (x, x))
        x = n - 200
        mod.insort_right(data, x, x - 50, x + 50)
        self.assertEqual(data.last_insert, (x + 1, x))

    def test_random(self, n=25):
        from random import randrange
        for i in range(n):
            data = [randrange(0, n, 2) for j in range(i)]
            data.sort()
            elem = randrange(-1, n+1)
            ip = self.module.bisect_left(data, elem)
            if ip < len(data):
                self.assertTrue(elem <= data[ip])
            if ip > 0:
                self.assertTrue(data[ip-1] < elem)
            ip = self.module.bisect_right(data, elem)
            if ip < len(data):
                self.assertTrue(elem < data[ip])
            if ip > 0:
                self.assertTrue(data[ip-1] <= elem)

    def test_optionalSlicing(self):
        for func, data, elem, expected in self.precomputedCases:
            for lo in range(4):
                lo = min(len(data), lo)
                for hi in range(3,8):
                    hi = min(len(data), hi)
                    ip = func(data, elem, lo, hi)
                    self.assertTrue(lo <= ip <= hi)
                    if func is self.module.bisect_left and ip < hi:
                        self.assertTrue(elem <= data[ip])
                    if func is self.module.bisect_left and ip > lo:
                        self.assertTrue(data[ip-1] < elem)
                    if func is self.module.bisect_right and ip < hi:
                        self.assertTrue(elem < data[ip])
                    if func is self.module.bisect_right and ip > lo:
                        self.assertTrue(data[ip-1] <= elem)
                    self.assertEqual(ip, max(lo, min(hi, expected)))

    def test_backcompatibility(self):
        self.assertEqual(self.module.bisect, self.module.bisect_right)

    def test_keyword_args(self):
        data = [10, 20, 30, 40, 50]
        self.assertEqual(self.module.bisect_left(a=data, x=25, lo=1, hi=3), 2)
        self.assertEqual(self.module.bisect_right(a=data, x=25, lo=1, hi=3), 2)
        self.assertEqual(self.module.bisect(a=data, x=25, lo=1, hi=3), 2)
        self.module.insort_left(a=data, x=25, lo=1, hi=3)
        self.module.insort_right(a=data, x=25, lo=1, hi=3)
        self.module.insort(a=data, x=25, lo=1, hi=3)
        self.assertEqual(data, [10, 20, 25, 25, 25, 30, 40, 50])

    def test_lookups_with_key_function(self):
        mod = self.module

        # Invariant: Index with a keyfunc on an array
        # should match the index on an array where
        # key function has already been applied.

        keyfunc = abs
        arr = sorted([2, -4, 6, 8, -10], key=keyfunc)
        precomputed_arr = list(map(keyfunc, arr))
        for x in precomputed_arr:
            self.assertEqual(
                mod.bisect_left(arr, x, key=keyfunc),
                mod.bisect_left(precomputed_arr, x)
            )
            self.assertEqual(
                mod.bisect_right(arr, x, key=keyfunc),
                mod.bisect_right(precomputed_arr, x)
            )

        keyfunc = str.casefold
        arr = sorted('aBcDeEfgHhiIiij', key=keyfunc)
        precomputed_arr = list(map(keyfunc, arr))
        for x in precomputed_arr:
            self.assertEqual(
                mod.bisect_left(arr, x, key=keyfunc),
                mod.bisect_left(precomputed_arr, x)
            )
            self.assertEqual(
                mod.bisect_right(arr, x, key=keyfunc),
                mod.bisect_right(precomputed_arr, x)
            )

    def test_insort(self):
        from random import shuffle
        mod = self.module

        # Invariant:  As random elements are inserted in
        # a target list, the targetlist remains sorted.
        keyfunc = abs
        data = list(range(-10, 11)) + list(range(-20, 20, 2))
        shuffle(data)
        target = []
        for x in data:
            mod.insort_left(target, x, key=keyfunc)
            self.assertEqual(
                sorted(target, key=keyfunc),
                target
            )
        target = []
        for x in data:
            mod.insort_right(target, x, key=keyfunc)
            self.assertEqual(
                sorted(target, key=keyfunc),
                target
            )

    def test_insort_keynotNone(self):
        x = []
        y = {"a": 2, "b": 1}
        for f in (self.module.insort_left, self.module.insort_right):
            self.assertRaises(TypeError, f, x, y, key = "b")

    def test_lt_returns_non_bool(self):
        class A:
            def __init__(self, val):
                self.val = val
            def __lt__(self, other):
                return "nonempty" if self.val < other.val else ""

        data = [A(i) for i in range(100)]
        i1 = self.module.bisect_left(data, A(33))
        i2 = self.module.bisect_right(data, A(33))
        self.assertEqual(i1, 33)
        self.assertEqual(i2, 34)

    def test_lt_returns_notimplemented(self):
        class A:
            def __init__(self, val):
                self.val = val
            def __lt__(self, other):
                return NotImplemented
            def __gt__(self, other):
                return self.val > other.val

        data = [A(i) for i in range(100)]
        i1 = self.module.bisect_left(data, A(40))
        i2 = self.module.bisect_right(data, A(40))
        self.assertEqual(i1, 40)
        self.assertEqual(i2, 41)

class TestBisectPython(_TestBisect, CPythonTestCase):
    module = py_bisect

class TestBisectC(_TestBisect, CPythonTestCase):
    module = c_bisect

#==============================================================================

class _TestInsort:
    def test_vsBuiltinSort(self, n=500):
        from random import choice
        for insorted in (list(), UserList()):
            for i in range(n):
                digit = choice("0123456789")
                if digit in "02468":
                    f = self.module.insort_left
                else:
                    f = self.module.insort_right
                f(insorted, digit)
            self.assertEqual(sorted(insorted), insorted)

    def test_backcompatibility(self):
        self.assertEqual(self.module.insort, self.module.insort_right)

    def test_listDerived(self):
        class List(list):
            data = []
            def insert(self, index, item):
                self.data.insert(index, item)

        lst = List()
        self.module.insort_left(lst, 10)
        self.module.insort_right(lst, 5)
        self.assertEqual([5, 10], lst.data)

class TestInsortPython(_TestInsort, CPythonTestCase):
    module = py_bisect

class TestInsortC(_TestInsort, CPythonTestCase):
    module = c_bisect

#==============================================================================

class LenOnly:
    "Dummy sequence class defining __len__ but not __getitem__."
    def __len__(self):
        return 10

class GetOnly:
    "Dummy sequence class defining __getitem__ but not __len__."
    def __getitem__(self, ndx):
        return 10

class CmpErr:
    "Dummy element that always raises an error during comparison"
    def __lt__(self, other):
        raise ZeroDivisionError
    __gt__ = __lt__
    __le__ = __lt__
    __ge__ = __lt__
    __eq__ = __lt__
    __ne__ = __lt__

class _TestErrorHandling:
    def test_non_sequence(self):
        for f in (self.module.bisect_left, self.module.bisect_right,
                  self.module.insort_left, self.module.insort_right):
            self.assertRaises(TypeError, f, 10, 10)

    def test_len_only(self):
        for f in (self.module.bisect_left, self.module.bisect_right,
                  self.module.insort_left, self.module.insort_right):
            self.assertRaises(TypeError, f, LenOnly(), 10)

    def test_get_only(self):
        for f in (self.module.bisect_left, self.module.bisect_right,
                  self.module.insort_left, self.module.insort_right):
            self.assertRaises(TypeError, f, GetOnly(), 10)

    def test_cmp_err(self):
        seq = [CmpErr(), CmpErr(), CmpErr()]
        for f in (self.module.bisect_left, self.module.bisect_right,
                  self.module.insort_left, self.module.insort_right):
            self.assertRaises(ZeroDivisionError, f, seq, 10)

    def test_arg_parsing(self):
        for f in (self.module.bisect_left, self.module.bisect_right,
                  self.module.insort_left, self.module.insort_right):
            self.assertRaises(TypeError, f, 10)

class TestErrorHandlingPython(_TestErrorHandling, CPythonTestCase):
    module = py_bisect

class TestErrorHandlingC(_TestErrorHandling, CPythonTestCase):
    module = c_bisect

#==============================================================================

class _TestDocExample:
    def test_grades(self):
        def grade(score, breakpoints=[60, 70, 80, 90], grades='FDCBA'):
            i = self.module.bisect(breakpoints, score)
            return grades[i]

        result = [grade(score) for score in [33, 99, 77, 70, 89, 90, 100]]
        self.assertEqual(result, ['F', 'A', 'C', 'C', 'B', 'A', 'A'])

    def test_colors(self):
        data = [('red', 5), ('blue', 1), ('yellow', 8), ('black', 0)]
        data.sort(key=lambda r: r[1])
        keys = [r[1] for r in data]
        bisect_left = self.module.bisect_left
        self.assertEqual(data[bisect_left(keys, 0)], ('black', 0))
        self.assertEqual(data[bisect_left(keys, 1)], ('blue', 1))
        self.assertEqual(data[bisect_left(keys, 5)], ('red', 5))
        self.assertEqual(data[bisect_left(keys, 8)], ('yellow', 8))

class TestDocExamplePython(_TestDocExample, CPythonTestCase):
    module = py_bisect

class TestDocExampleC(_TestDocExample, CPythonTestCase):
    module = c_bisect

#------------------------------------------------------------------------------

if __name__ == "__main__":
    run_tests()
