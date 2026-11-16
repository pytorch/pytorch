# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_contains.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

from collections import deque
import unittest
from test.support import NEVER_EQ


class base_set:
    def __init__(self, el):
        self.el = el

class myset(base_set):
    def __contains__(self, el):
        return self.el == el

class seq(base_set):
    def __getitem__(self, n):
        return [self.el][n]

class TestContains(CPythonTestCase):
    def test_common_tests(self):
        a = base_set(1)
        b = myset(1)
        c = seq(1)
        self.assertIn(1, b)
        self.assertNotIn(0, b)
        self.assertIn(1, c)
        self.assertNotIn(0, c)
        self.assertRaises(TypeError, lambda: 1 in a)
        self.assertRaises(TypeError, lambda: 1 not in a)

        # test char in string
        self.assertIn('c', 'abc')
        self.assertNotIn('d', 'abc')

        self.assertIn('', '')
        self.assertIn('', 'abc')

        self.assertRaises(TypeError, lambda: None in 'abc')

    def test_builtin_sequence_types(self):
        # a collection of tests on builtin sequence types
        a = range(10)
        for i in a:
            self.assertIn(i, a)
        self.assertNotIn(16, a)
        self.assertNotIn(a, a)

        a = tuple(a)
        for i in a:
            self.assertIn(i, a)
        self.assertNotIn(16, a)
        self.assertNotIn(a, a)

        class Deviant1:
            """Behaves strangely when compared

            This class is designed to make sure that the contains code
            works when the list is modified during the check.
            """
            aList = list(range(15))
            def __eq__(self, other):
                if other == 12:
                    self.aList.remove(12)
                    self.aList.remove(13)
                    self.aList.remove(14)
                return 0

        self.assertNotIn(Deviant1(), Deviant1.aList)

    def test_nonreflexive(self):
        # containment and equality tests involving elements that are
        # not necessarily equal to themselves

        values = float('nan'), 1, None, 'abc', NEVER_EQ
        constructors = list, tuple, dict.fromkeys, set, frozenset, deque
        for constructor in constructors:
            container = constructor(values)
            for elem in container:
                self.assertIn(elem, container)
            self.assertTrue(container == constructor(values))
            self.assertTrue(container == container)

    def test_block_fallback(self):
        # blocking fallback with __contains__ = None
        class ByContains(object):
            def __contains__(self, other):
                return False
        c = ByContains()
        class BlockContains(ByContains):
            """Is not a container

            This class is a perfectly good iterable (as tested by
            list(bc)), as well as inheriting from a perfectly good
            container, but __contains__ = None prevents the usual
            fallback to iteration in the container protocol. That
            is, normally, 0 in bc would fall back to the equivalent
            of any(x==0 for x in bc), but here it's blocked from
            doing so.
            """
            def __iter__(self):
                while False:
                    yield None
            __contains__ = None
        bc = BlockContains()
        self.assertFalse(0 in c)
        self.assertFalse(0 in list(bc))
        self.assertRaises(TypeError, lambda: 0 in bc)

if __name__ == '__main__':
    run_tests()
