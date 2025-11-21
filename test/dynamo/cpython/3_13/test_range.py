# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_range.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

# Python test set -- built-in functions

import unittest
import sys
import pickle
import itertools
from test.support import ALWAYS_EQ

# pure Python implementations (3 args only), for comparison
def pyrange(start, stop, step):
    if (start - stop) // step < 0:
        # replace stop with next element in the sequence of integers
        # that are congruent to start modulo step.
        stop += (start - stop) % step
        while start != stop:
            yield start
            start += step

def pyrange_reversed(start, stop, step):
    stop += (start - stop) % step
    return pyrange(stop - step, start - step, -step)


class RangeTest(__TestCase):
    def assert_iterators_equal(self, xs, ys, test_id, limit=None):
        # check that an iterator xs matches the expected results ys,
        # up to a given limit.
        if limit is not None:
            xs = itertools.islice(xs, limit)
            ys = itertools.islice(ys, limit)
        sentinel = object()
        pairs = itertools.zip_longest(xs, ys, fillvalue=sentinel)
        for i, (x, y) in enumerate(pairs):
            if x == y:
                continue
            elif x == sentinel:
                self.fail('{}: iterator ended unexpectedly '
                          'at position {}; expected {}'.format(test_id, i, y))
            elif y == sentinel:
                self.fail('{}: unexpected excess element {} at '
                          'position {}'.format(test_id, x, i))
            else:
                self.fail('{}: wrong element at position {}; '
                          'expected {}, got {}'.format(test_id, i, y, x))

    def test_range(self):
        self.assertEqual(list(range(3)), [0, 1, 2])
        self.assertEqual(list(range(1, 5)), [1, 2, 3, 4])
        self.assertEqual(list(range(0)), [])
        self.assertEqual(list(range(-3)), [])
        self.assertEqual(list(range(1, 10, 3)), [1, 4, 7])
        self.assertEqual(list(range(5, -5, -3)), [5, 2, -1, -4])

        a = 10
        b = 100
        c = 50

        self.assertEqual(list(range(a, a+2)), [a, a+1])
        self.assertEqual(list(range(a+2, a, -1)), [a+2, a+1])
        self.assertEqual(list(range(a+4, a, -2)), [a+4, a+2])

        seq = list(range(a, b, c))
        self.assertIn(a, seq)
        self.assertNotIn(b, seq)
        self.assertEqual(len(seq), 2)

        seq = list(range(b, a, -c))
        self.assertIn(b, seq)
        self.assertNotIn(a, seq)
        self.assertEqual(len(seq), 2)

        seq = list(range(-a, -b, -c))
        self.assertIn(-a, seq)
        self.assertNotIn(-b, seq)
        self.assertEqual(len(seq), 2)

        self.assertEqual(len(range(0, sys.maxsize, sys.maxsize-1)), 2)

        r = range(-sys.maxsize, sys.maxsize, 2)
        self.assertEqual(len(r), sys.maxsize)

    def test_range_constructor_error_messages(self):
        with self.assertRaisesRegex(
                TypeError,
                "range expected at least 1 argument, got 0"
        ):
            range()

        with self.assertRaisesRegex(
                TypeError,
                "range expected at most 3 arguments, got 6"
        ):
            range(1, 2, 3, 4, 5, 6)

    def test_large_operands(self):
        x = range(10**20, 10**20+10, 3)
        self.assertEqual(len(x), 4)
        self.assertEqual(len(list(x)), 4)

        x = range(10**20+10, 10**20, 3)
        self.assertEqual(len(x), 0)
        self.assertEqual(len(list(x)), 0)
        self.assertFalse(x)

        x = range(10**20, 10**20+10, -3)
        self.assertEqual(len(x), 0)
        self.assertEqual(len(list(x)), 0)
        self.assertFalse(x)

        x = range(10**20+10, 10**20, -3)
        self.assertEqual(len(x), 4)
        self.assertEqual(len(list(x)), 4)
        self.assertTrue(x)

        # Now test range() with longs
        for x in [range(-2**100),
                  range(0, -2**100),
                  range(0, 2**100, -1)]:
            self.assertEqual(list(x), [])
            self.assertFalse(x)

        a = int(10 * sys.maxsize)
        b = int(100 * sys.maxsize)
        c = int(50 * sys.maxsize)

        self.assertEqual(list(range(a, a+2)), [a, a+1])
        self.assertEqual(list(range(a+2, a, -1)), [a+2, a+1])
        self.assertEqual(list(range(a+4, a, -2)), [a+4, a+2])

        seq = list(range(a, b, c))
        self.assertIn(a, seq)
        self.assertNotIn(b, seq)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], a)
        self.assertEqual(seq[-1], a+c)

        seq = list(range(b, a, -c))
        self.assertIn(b, seq)
        self.assertNotIn(a, seq)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], b)
        self.assertEqual(seq[-1], b-c)

        seq = list(range(-a, -b, -c))
        self.assertIn(-a, seq)
        self.assertNotIn(-b, seq)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], -a)
        self.assertEqual(seq[-1], -a-c)

    def test_large_range(self):
        # Check long ranges (len > sys.maxsize)
        # len() is expected to fail due to limitations of the __len__ protocol
        def _range_len(x):
            try:
                length = len(x)
            except OverflowError:
                step = x[1] - x[0]
                length = 1 + ((x[-1] - x[0]) // step)
            return length

        a = -sys.maxsize
        b = sys.maxsize
        expected_len = b - a
        x = range(a, b)
        self.assertIn(a, x)
        self.assertNotIn(b, x)
        self.assertRaises(OverflowError, len, x)
        self.assertTrue(x)
        self.assertEqual(_range_len(x), expected_len)
        self.assertEqual(x[0], a)
        idx = sys.maxsize+1
        self.assertEqual(x[idx], a+idx)
        self.assertEqual(x[idx:idx+1][0], a+idx)
        with self.assertRaises(IndexError):
            x[-expected_len-1]
        with self.assertRaises(IndexError):
            x[expected_len]

        a = 0
        b = 2 * sys.maxsize
        expected_len = b - a
        x = range(a, b)
        self.assertIn(a, x)
        self.assertNotIn(b, x)
        self.assertRaises(OverflowError, len, x)
        self.assertTrue(x)
        self.assertEqual(_range_len(x), expected_len)
        self.assertEqual(x[0], a)
        idx = sys.maxsize+1
        self.assertEqual(x[idx], a+idx)
        self.assertEqual(x[idx:idx+1][0], a+idx)
        with self.assertRaises(IndexError):
            x[-expected_len-1]
        with self.assertRaises(IndexError):
            x[expected_len]

        a = 0
        b = sys.maxsize**10
        c = 2*sys.maxsize
        expected_len = 1 + (b - a) // c
        x = range(a, b, c)
        self.assertIn(a, x)
        self.assertNotIn(b, x)
        self.assertRaises(OverflowError, len, x)
        self.assertTrue(x)
        self.assertEqual(_range_len(x), expected_len)
        self.assertEqual(x[0], a)
        idx = sys.maxsize+1
        self.assertEqual(x[idx], a+(idx*c))
        self.assertEqual(x[idx:idx+1][0], a+(idx*c))
        with self.assertRaises(IndexError):
            x[-expected_len-1]
        with self.assertRaises(IndexError):
            x[expected_len]

        a = sys.maxsize**10
        b = 0
        c = -2*sys.maxsize
        expected_len = 1 + (b - a) // c
        x = range(a, b, c)
        self.assertIn(a, x)
        self.assertNotIn(b, x)
        self.assertRaises(OverflowError, len, x)
        self.assertTrue(x)
        self.assertEqual(_range_len(x), expected_len)
        self.assertEqual(x[0], a)
        idx = sys.maxsize+1
        self.assertEqual(x[idx], a+(idx*c))
        self.assertEqual(x[idx:idx+1][0], a+(idx*c))
        with self.assertRaises(IndexError):
            x[-expected_len-1]
        with self.assertRaises(IndexError):
            x[expected_len]

    def test_invalid_invocation(self):
        self.assertRaises(TypeError, range)
        self.assertRaises(TypeError, range, 1, 2, 3, 4)
        self.assertRaises(ValueError, range, 1, 2, 0)
        a = int(10 * sys.maxsize)
        self.assertRaises(ValueError, range, a, a + 1, int(0))
        self.assertRaises(TypeError, range, 1., 1., 1.)
        self.assertRaises(TypeError, range, 1e100, 1e101, 1e101)
        self.assertRaises(TypeError, range, 0, "spam")
        self.assertRaises(TypeError, range, 0, 42, "spam")
        # Exercise various combinations of bad arguments, to check
        # refcounting logic
        self.assertRaises(TypeError, range, 0.0)
        self.assertRaises(TypeError, range, 0, 0.0)
        self.assertRaises(TypeError, range, 0.0, 0)
        self.assertRaises(TypeError, range, 0.0, 0.0)
        self.assertRaises(TypeError, range, 0, 0, 1.0)
        self.assertRaises(TypeError, range, 0, 0.0, 1)
        self.assertRaises(TypeError, range, 0, 0.0, 1.0)
        self.assertRaises(TypeError, range, 0.0, 0, 1)
        self.assertRaises(TypeError, range, 0.0, 0, 1.0)
        self.assertRaises(TypeError, range, 0.0, 0.0, 1)
        self.assertRaises(TypeError, range, 0.0, 0.0, 1.0)

    def test_index(self):
        u = range(2)
        self.assertEqual(u.index(0), 0)
        self.assertEqual(u.index(1), 1)
        self.assertRaises(ValueError, u.index, 2)

        u = range(-2, 3)
        self.assertEqual(u.count(0), 1)
        self.assertEqual(u.index(0), 2)
        self.assertRaises(TypeError, u.index)

        class BadExc(Exception):
            pass

        class BadCmp:
            def __eq__(self, other):
                if other == 2:
                    raise BadExc()
                return False

        a = range(4)
        self.assertRaises(BadExc, a.index, BadCmp())

        a = range(-2, 3)
        self.assertEqual(a.index(0), 2)
        self.assertEqual(range(1, 10, 3).index(4), 1)
        self.assertEqual(range(1, -10, -3).index(-5), 2)

        self.assertEqual(range(10**20).index(1), 1)
        self.assertEqual(range(10**20).index(10**20 - 1), 10**20 - 1)

        self.assertRaises(ValueError, range(1, 2**100, 2).index, 2**87)
        self.assertEqual(range(1, 2**100, 2).index(2**87+1), 2**86)

        self.assertEqual(range(10).index(ALWAYS_EQ), 0)

    def test_user_index_method(self):
        bignum = 2*sys.maxsize
        smallnum = 42

        # User-defined class with an __index__ method
        class I:
            def __init__(self, n):
                self.n = int(n)
            def __index__(self):
                return self.n
        self.assertEqual(list(range(I(bignum), I(bignum + 1))), [bignum])
        self.assertEqual(list(range(I(smallnum), I(smallnum + 1))), [smallnum])

        # User-defined class with a failing __index__ method
        class IX:
            def __index__(self):
                raise RuntimeError
        self.assertRaises(RuntimeError, range, IX())

        # User-defined class with an invalid __index__ method
        class IN:
            def __index__(self):
                return "not a number"

        self.assertRaises(TypeError, range, IN())

        # Test use of user-defined classes in slice indices.
        self.assertEqual(range(10)[:I(5)], range(5))

        with self.assertRaises(RuntimeError):
            range(0, 10)[:IX()]

        with self.assertRaises(TypeError):
            range(0, 10)[:IN()]

    def test_count(self):
        self.assertEqual(range(3).count(-1), 0)
        self.assertEqual(range(3).count(0), 1)
        self.assertEqual(range(3).count(1), 1)
        self.assertEqual(range(3).count(2), 1)
        self.assertEqual(range(3).count(3), 0)
        self.assertIs(type(range(3).count(-1)), int)
        self.assertIs(type(range(3).count(1)), int)
        self.assertEqual(range(10**20).count(1), 1)
        self.assertEqual(range(10**20).count(10**20), 0)
        self.assertEqual(range(3).index(1), 1)
        self.assertEqual(range(1, 2**100, 2).count(2**87), 0)
        self.assertEqual(range(1, 2**100, 2).count(2**87+1), 1)

        # self.assertEqual(range(10).count(ALWAYS_EQ), 10)

        self.assertEqual(len(range(sys.maxsize, sys.maxsize+10)), 10)

    def test_repr(self):
        self.assertEqual(repr(range(1)), 'range(0, 1)')
        self.assertEqual(repr(range(1, 2)), 'range(1, 2)')
        self.assertEqual(repr(range(1, 2, 3)), 'range(1, 2, 3)')

    def test_pickling(self):
        testcases = [(13,), (0, 11), (-22, 10), (20, 3, -1),
                     (13, 21, 3), (-2, 2, 2), (2**65, 2**65+2)]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            for t in testcases:
                with self.subTest(proto=proto, test=t):
                    r = range(*t)
                    self.assertEqual(list(pickle.loads(pickle.dumps(r, proto))),
                                     list(r))

    def test_iterator_pickling(self):
        testcases = [(13,), (0, 11), (-22, 10), (20, 3, -1), (13, 21, 3),
                     (-2, 2, 2)]
        for M in 2**31, 2**63:
            testcases += [
                (M-3, M-1), (4*M, 4*M+2),
                (M-2, M-1, 2), (-M+1, -M, -2),
                (1, 2, M-1), (-1, -2, -M),
                (1, M-1, M-1), (-1, -M, -M),
            ]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            for t in testcases:
                with self.subTest(proto=proto, t=t):
                    it = itorg = iter(range(*t))
                    data = list(range(*t))

                    d = pickle.dumps(it, proto)
                    it = pickle.loads(d)
                    self.assertEqual(type(itorg), type(it))
                    self.assertEqual(list(it), data)

                    it = pickle.loads(d)
                    try:
                        next(it)
                    except StopIteration:
                        continue
                    d = pickle.dumps(it, proto)
                    it = pickle.loads(d)
                    self.assertEqual(list(it), data[1:])

    @skipIfTorchDynamo("infinite loop")
    def test_iterator_pickling_overflowing_index(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                it = iter(range(2**32 + 2))
                it.__setstate__(2**32 + 1)  # undocumented way to advance an iterator
                d = pickle.dumps(it, proto)
                it = pickle.loads(d)
                self.assertEqual(next(it), 2**32 + 1)

    def test_exhausted_iterator_pickling(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            r = range(2**65, 2**65+2)
            i = iter(r)
            while True:
                r = next(i)
                if r == 2**65+1:
                    break
            d = pickle.dumps(i, proto)
            i2 = pickle.loads(d)
            self.assertEqual(list(i), [])
            self.assertEqual(list(i2), [])

    def test_large_exhausted_iterator_pickling(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            r = range(20)
            i = iter(r)
            while True:
                r = next(i)
                if r == 19:
                    break
            d = pickle.dumps(i, proto)
            i2 = pickle.loads(d)
            self.assertEqual(list(i), [])
            self.assertEqual(list(i2), [])

    def test_iterator_unpickle_compat(self):
        testcases = [
            b'c__builtin__\niter\n(c__builtin__\nxrange\n(I10\nI20\nI2\ntRtRI2\nb.',
            b'c__builtin__\niter\n(c__builtin__\nxrange\n(K\nK\x14K\x02tRtRK\x02b.',
            b'\x80\x02c__builtin__\niter\nc__builtin__\nxrange\nK\nK\x14K\x02\x87R\x85RK\x02b.',
            b'\x80\x03cbuiltins\niter\ncbuiltins\nrange\nK\nK\x14K\x02\x87R\x85RK\x02b.',
            b'\x80\x04\x951\x00\x00\x00\x00\x00\x00\x00\x8c\x08builtins\x8c\x04iter\x93\x8c\x08builtins\x8c\x05range\x93K\nK\x14K\x02\x87R\x85RK\x02b.',

            b'c__builtin__\niter\n(c__builtin__\nxrange\n(L-36893488147419103232L\nI20\nI2\ntRtRL18446744073709551623L\nb.',
            b'c__builtin__\niter\n(c__builtin__\nxrange\n(L-36893488147419103232L\nK\x14K\x02tRtRL18446744073709551623L\nb.',
            b'\x80\x02c__builtin__\niter\nc__builtin__\nxrange\n\x8a\t\x00\x00\x00\x00\x00\x00\x00\x00\xfeK\x14K\x02\x87R\x85R\x8a\t\x07\x00\x00\x00\x00\x00\x00\x00\x01b.',
            b'\x80\x03cbuiltins\niter\ncbuiltins\nrange\n\x8a\t\x00\x00\x00\x00\x00\x00\x00\x00\xfeK\x14K\x02\x87R\x85R\x8a\t\x07\x00\x00\x00\x00\x00\x00\x00\x01b.',
            b'\x80\x04\x95C\x00\x00\x00\x00\x00\x00\x00\x8c\x08builtins\x8c\x04iter\x93\x8c\x08builtins\x8c\x05range\x93\x8a\t\x00\x00\x00\x00\x00\x00\x00\x00\xfeK\x14K\x02\x87R\x85R\x8a\t\x07\x00\x00\x00\x00\x00\x00\x00\x01b.',
        ]
        for t in testcases:
            it = pickle.loads(t)
            self.assertEqual(list(it), [14, 16, 18])

    def test_iterator_setstate(self):
        it = iter(range(10, 20, 2))
        it.__setstate__(2)
        self.assertEqual(list(it), [14, 16, 18])
        it = reversed(range(10, 20, 2))
        it.__setstate__(3)
        self.assertEqual(list(it), [12, 10])
        it = iter(range(-2**65, 20, 2))
        it.__setstate__(2**64 + 7)
        self.assertEqual(list(it), [14, 16, 18])
        it = reversed(range(10, 2**65, 2))
        it.__setstate__(2**64 - 7)
        self.assertEqual(list(it), [12, 10])

    def test_odd_bug(self):
        # This used to raise a "SystemError: NULL result without error"
        # because the range validation step was eating the exception
        # before NULL was returned.
        with self.assertRaises(TypeError):
            range([], 1, -1)

    def test_types(self):
        # Non-integer objects *equal* to any of the range's items are supposed
        # to be contained in the range.
        self.assertIn(1.0, range(3))
        self.assertIn(True, range(3))
        self.assertIn(1+0j, range(3))

        self.assertIn(ALWAYS_EQ, range(3))

        # Objects are never coerced into other types for comparison.
        class C2:
            def __int__(self): return 1
            def __index__(self): return 1
        self.assertNotIn(C2(), range(3))
        # ..except if explicitly told so.
        self.assertIn(int(C2()), range(3))

        # Check that the range.__contains__ optimization is only
        # used for ints, not for instances of subclasses of int.
        class C3(int):
            def __eq__(self, other): return True
        self.assertIn(C3(11), range(10))
        self.assertIn(C3(11), list(range(10)))

    def test_strided_limits(self):
        r = range(0, 101, 2)
        self.assertIn(0, r)
        self.assertNotIn(1, r)
        self.assertIn(2, r)
        self.assertNotIn(99, r)
        self.assertIn(100, r)
        self.assertNotIn(101, r)

        r = range(0, -20, -1)
        self.assertIn(0, r)
        self.assertIn(-1, r)
        self.assertIn(-19, r)
        self.assertNotIn(-20, r)

        r = range(0, -20, -2)
        self.assertIn(-18, r)
        self.assertNotIn(-19, r)
        self.assertNotIn(-20, r)

    def test_empty(self):
        r = range(0)
        self.assertNotIn(0, r)
        self.assertNotIn(1, r)

        r = range(0, -10)
        self.assertNotIn(0, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(1, r)

    def test_range_iterators(self):
        # exercise 'fast' iterators, that use a rangeiterobject internally.
        # see issue 7298
        limits = [base + jiggle
                  for M in (2**32, 2**64)
                  for base in (-M, -M//2, 0, M//2, M)
                  for jiggle in (-2, -1, 0, 1, 2)]
        test_ranges = [(start, end, step)
                       for start in limits
                       for end in limits
                       for step in (-2**63, -2**31, -2, -1, 1, 2)]
        test_ranges += [(-2**63, 2**63-2, 1)] # regression test for gh-100810

        for start, end, step in test_ranges:
            iter1 = range(start, end, step)
            iter2 = pyrange(start, end, step)
            test_id = "range({}, {}, {})".format(start, end, step)
            # check first 100 entries
            self.assert_iterators_equal(iter1, iter2, test_id, limit=100)

            iter1 = reversed(range(start, end, step))
            iter2 = pyrange_reversed(start, end, step)
            test_id = "reversed(range({}, {}, {}))".format(start, end, step)
            self.assert_iterators_equal(iter1, iter2, test_id, limit=100)

    def test_range_iterators_invocation(self):
        # verify range iterators instances cannot be created by
        # calling their type
        rangeiter_type = type(iter(range(0)))
        self.assertRaises(TypeError, rangeiter_type, 1, 3, 1)
        long_rangeiter_type = type(iter(range(1 << 1000)))
        self.assertRaises(TypeError, long_rangeiter_type, 1, 3, 1)

    def test_slice(self):
        def check(start, stop, step=None):
            i = slice(start, stop, step)
            self.assertEqual(list(r[i]), list(r)[i])
            self.assertEqual(len(r[i]), len(list(r)[i]))
        for r in [range(10),
                  range(0),
                  range(1, 9, 3),
                  range(8, 0, -3),
                  range(sys.maxsize+1, sys.maxsize+10),
                  ]:
            check(0, 2)
            check(0, 20)
            check(1, 2)
            check(20, 30)
            check(-30, -20)
            check(-1, 100, 2)
            check(0, -1)
            check(-1, -3, -1)

    def test_contains(self):
        r = range(10)
        self.assertIn(0, r)
        self.assertIn(1, r)
        self.assertIn(5.0, r)
        self.assertNotIn(5.1, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(10, r)
        self.assertNotIn("", r)
        r = range(9, -1, -1)
        self.assertIn(0, r)
        self.assertIn(1, r)
        self.assertIn(5.0, r)
        self.assertNotIn(5.1, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(10, r)
        self.assertNotIn("", r)
        r = range(0, 10, 2)
        self.assertIn(0, r)
        self.assertNotIn(1, r)
        self.assertNotIn(5.0, r)
        self.assertNotIn(5.1, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(10, r)
        self.assertNotIn("", r)
        r = range(9, -1, -2)
        self.assertNotIn(0, r)
        self.assertIn(1, r)
        self.assertIn(5.0, r)
        self.assertNotIn(5.1, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(10, r)
        self.assertNotIn("", r)

    def test_reverse_iteration(self):
        for r in [range(10),
                  range(0),
                  range(1, 9, 3),
                  range(8, 0, -3),
                  range(sys.maxsize+1, sys.maxsize+10),
                  ]:
            self.assertEqual(list(reversed(r)), list(r)[::-1])

    def test_issue11845(self):
        r = range(*slice(1, 18, 2).indices(20))
        values = {None, 0, 1, -1, 2, -2, 5, -5, 19, -19,
                  20, -20, 21, -21, 30, -30, 99, -99}
        for i in values:
            for j in values:
                for k in values - {0}:
                    r[i:j:k]

    def test_comparison(self):
        test_ranges = [range(0), range(0, -1), range(1, 1, 3),
                       range(1), range(5, 6), range(5, 6, 2),
                       range(5, 7, 2), range(2), range(0, 4, 2),
                       range(0, 5, 2), range(0, 6, 2)]
        test_tuples = list(map(tuple, test_ranges))

        # Check that equality of ranges matches equality of the corresponding
        # tuples for each pair from the test lists above.
        ranges_eq = [a == b for a in test_ranges for b in test_ranges]
        tuples_eq = [a == b for a in test_tuples for b in test_tuples]
        self.assertEqual(ranges_eq, tuples_eq)

        # Check that != correctly gives the logical negation of ==
        ranges_ne = [a != b for a in test_ranges for b in test_ranges]
        self.assertEqual(ranges_ne, [not x for x in ranges_eq])

        # Ranges are unequal to other types (even sequence types)
        self.assertIs(range(0) == (), False)
        # self.assertIs(() == range(0), False)
        self.assertIs(range(2) == [0, 1], False)

        # Huge integers aren't a problem.
        self.assertEqual(range(0, 2**100 - 1, 2),
                         range(0, 2**100, 2))
        self.assertNotEqual(range(0, 2**100, 2),
                            range(0, 2**100 + 1, 2))
        self.assertEqual(range(2**200, 2**201 - 2**99, 2**100),
                         range(2**200, 2**201, 2**100))
        self.assertNotEqual(range(2**200, 2**201, 2**100),
                            range(2**200, 2**201 + 1, 2**100))

        # Order comparisons are not implemented for ranges.
        with self.assertRaises(TypeError):
            range(0) < range(0)
        with self.assertRaises(TypeError):
            range(0) > range(0)
        with self.assertRaises(TypeError):
            range(0) <= range(0)
        with self.assertRaises(TypeError):
            range(0) >= range(0)


    def test_attributes(self):
        # test the start, stop and step attributes of range objects
        self.assert_attrs(range(0), 0, 0, 1)
        self.assert_attrs(range(10), 0, 10, 1)
        self.assert_attrs(range(-10), 0, -10, 1)
        self.assert_attrs(range(0, 10, 1), 0, 10, 1)
        self.assert_attrs(range(0, 10, 3), 0, 10, 3)
        self.assert_attrs(range(10, 0, -1), 10, 0, -1)
        self.assert_attrs(range(10, 0, -3), 10, 0, -3)
        self.assert_attrs(range(True), 0, 1, 1)
        self.assert_attrs(range(False, True), 0, 1, 1)
        self.assert_attrs(range(False, True, True), 0, 1, 1)

    def assert_attrs(self, rangeobj, start, stop, step):
        self.assertEqual(rangeobj.start, start)
        self.assertEqual(rangeobj.stop, stop)
        self.assertEqual(rangeobj.step, step)
        self.assertIs(type(rangeobj.start), int)
        self.assertIs(type(rangeobj.stop), int)
        self.assertIs(type(rangeobj.step), int)


if __name__ == "__main__":
    run_tests()
