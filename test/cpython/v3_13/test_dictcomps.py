# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_dictcomps.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import traceback
import unittest

from test.support import BrokenIter

# For scope testing.
g = "Global variable"


class DictComprehensionTest(CPythonTestCase):

    def test_basics(self):
        expected = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17,
                    8: 18, 9: 19}
        actual = {k: k + 10 for k in range(10)}
        self.assertEqual(actual, expected)

        expected = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
        actual = {k: v for k in range(10) for v in range(10) if k == v}
        self.assertEqual(actual, expected)

    def test_scope_isolation(self):
        k = "Local Variable"

        expected = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None,
                    6: None, 7: None, 8: None, 9: None}
        actual = {k: None for k in range(10)}
        self.assertEqual(actual, expected)
        self.assertEqual(k, "Local Variable")

        expected = {9: 1, 18: 2, 19: 2, 27: 3, 28: 3, 29: 3, 36: 4, 37: 4,
                    38: 4, 39: 4, 45: 5, 46: 5, 47: 5, 48: 5, 49: 5, 54: 6,
                    55: 6, 56: 6, 57: 6, 58: 6, 59: 6, 63: 7, 64: 7, 65: 7,
                    66: 7, 67: 7, 68: 7, 69: 7, 72: 8, 73: 8, 74: 8, 75: 8,
                    76: 8, 77: 8, 78: 8, 79: 8, 81: 9, 82: 9, 83: 9, 84: 9,
                    85: 9, 86: 9, 87: 9, 88: 9, 89: 9}
        actual = {k: v for v in range(10) for k in range(v * 9, v * 10)}
        self.assertEqual(k, "Local Variable")
        self.assertEqual(actual, expected)

    def test_scope_isolation_from_global(self):
        expected = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None,
                    6: None, 7: None, 8: None, 9: None}
        actual = {g: None for g in range(10)}
        self.assertEqual(actual, expected)
        self.assertEqual(g, "Global variable")

        expected = {9: 1, 18: 2, 19: 2, 27: 3, 28: 3, 29: 3, 36: 4, 37: 4,
                    38: 4, 39: 4, 45: 5, 46: 5, 47: 5, 48: 5, 49: 5, 54: 6,
                    55: 6, 56: 6, 57: 6, 58: 6, 59: 6, 63: 7, 64: 7, 65: 7,
                    66: 7, 67: 7, 68: 7, 69: 7, 72: 8, 73: 8, 74: 8, 75: 8,
                    76: 8, 77: 8, 78: 8, 79: 8, 81: 9, 82: 9, 83: 9, 84: 9,
                    85: 9, 86: 9, 87: 9, 88: 9, 89: 9}
        actual = {g: v for v in range(10) for g in range(v * 9, v * 10)}
        self.assertEqual(g, "Global variable")
        self.assertEqual(actual, expected)

    def test_global_visibility(self):
        expected = {0: 'Global variable', 1: 'Global variable',
                    2: 'Global variable', 3: 'Global variable',
                    4: 'Global variable', 5: 'Global variable',
                    6: 'Global variable', 7: 'Global variable',
                    8: 'Global variable', 9: 'Global variable'}
        actual = {k: g for k in range(10)}
        self.assertEqual(actual, expected)

    def test_local_visibility(self):
        v = "Local variable"
        expected = {0: 'Local variable', 1: 'Local variable',
                    2: 'Local variable', 3: 'Local variable',
                    4: 'Local variable', 5: 'Local variable',
                    6: 'Local variable', 7: 'Local variable',
                    8: 'Local variable', 9: 'Local variable'}
        actual = {k: v for k in range(10)}
        self.assertEqual(actual, expected)
        self.assertEqual(v, "Local variable")

    def test_illegal_assignment(self):
        with self.assertRaisesRegex(SyntaxError, "cannot assign"):
            compile("{x: y for y, x in ((1, 2), (3, 4))} = 5", "<test>",
                    "exec")

        with self.assertRaisesRegex(SyntaxError, "illegal expression"):
            compile("{x: y for y, x in ((1, 2), (3, 4))} += 5", "<test>",
                    "exec")

    def test_evaluation_order(self):
        expected = {
            'H': 'W',
            'e': 'o',
            'l': 'l',
            'o': 'd',
        }

        expected_calls = [
            ('key', 'H'), ('value', 'W'),
            ('key', 'e'), ('value', 'o'),
            ('key', 'l'), ('value', 'r'),
            ('key', 'l'), ('value', 'l'),
            ('key', 'o'), ('value', 'd'),
        ]

        actual_calls = []

        def add_call(pos, value):
            actual_calls.append((pos, value))
            return value

        actual = {
            add_call('key', k): add_call('value', v)
            for k, v in zip('Hello', 'World')
        }

        self.assertEqual(actual, expected)
        self.assertEqual(actual_calls, expected_calls)

    def test_assignment_idiom_in_comprehensions(self):
        expected = {1: 1, 2: 4, 3: 9, 4: 16}
        actual = {j: j*j for i in range(4) for j in [i+1]}
        self.assertEqual(actual, expected)
        expected = {3: 2, 5: 6, 7: 12, 9: 20}
        actual = {j+k: j*k for i in range(4) for j in [i+1] for k in [j+1]}
        self.assertEqual(actual, expected)
        expected = {3: 2, 5: 6, 7: 12, 9: 20}
        actual = {j+k: j*k for i in range(4)  for j, k in [(i+1, i+2)]}
        self.assertEqual(actual, expected)

    def test_star_expression(self):
        expected = {0: 0, 1: 1, 2: 4, 3: 9}
        self.assertEqual({i: i*i for i in [*range(4)]}, expected)
        self.assertEqual({i: i*i for i in (*range(4),)}, expected)

    def test_exception_locations(self):
        # The location of an exception raised from __init__ or
        # __next__ should should be the iterator expression
        def init_raises():
            try:
                {x:x for x in BrokenIter(init_raises=True)}
            except Exception as e:
                return e

        def next_raises():
            try:
                {x:x for x in BrokenIter(next_raises=True)}
            except Exception as e:
                return e

        def iter_raises():
            try:
                {x:x for x in BrokenIter(iter_raises=True)}
            except Exception as e:
                return e

        for func, expected in [(init_raises, "BrokenIter(init_raises=True)"),
                               (next_raises, "BrokenIter(next_raises=True)"),
                               (iter_raises, "BrokenIter(iter_raises=True)"),
                              ]:
            with self.subTest(func):
                exc = func()
                f = traceback.extract_tb(exc.__traceback__)[0]
                indent = 16
                co = func.__code__
                self.assertEqual(f.lineno, co.co_firstlineno + 2)
                self.assertEqual(f.end_lineno, co.co_firstlineno + 2)
                self.assertEqual(f.line[f.colno - indent : f.end_colno - indent],
                                 expected)


if __name__ == "__main__":
    run_tests()
