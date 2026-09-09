# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_iterlen.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

""" Test Iterator Length Transparency

Some functions or methods which accept general iterable arguments have
optional, more efficient code paths if they know how many items to expect.
For instance, map(func, iterable), will pre-allocate the exact amount of
space required whenever the iterable can report its length.

The desired invariant is:  len(it)==len(list(it)).

A complication is that an iterable and iterator can be the same object. To
maintain the invariant, an iterator needs to dynamically update its length.
For instance, an iterable such as range(10) always reports its length as ten,
but it=iter(range(10)) starts at ten, and then goes to nine after next(it).
Having this capability means that map() can ignore the distinction between
map(func, iterable) and map(func, iter(iterable)).

When the iterable is immutable, the implementation can straight-forwardly
report the original length minus the cumulative number of calls to next().
This is the case for tuples, range objects, and itertools.repeat().

Some containers become temporarily immutable during iteration.  This includes
dicts, sets, and collections.deque.  Their implementation is equally simple
though they need to permanently set their length to zero whenever there is
an attempt to iterate after a length mutation.

The situation slightly more involved whenever an object allows length mutation
during iteration.  Lists and sequence iterators are dynamically updatable.
So, if a list is extended during iteration, the iterator will continue through
the new items.  If it shrinks to a point before the most recent iteration,
then no further items are available and the length is reported at zero.

Reversed objects can also be wrapped around mutable objects; however, any
appends after the current position are ignored.  Any other approach leads
to confusion and possibly returning the same item more than once.

The iterators not listed above, such as enumerate and the other itertools,
are not length transparent because they have no way to distinguish between
iterables that report static length and iterators whose length changes with
each call (i.e. the difference between enumerate('abc') and
enumerate(iter('abc')).

"""

import unittest
from itertools import repeat
from collections import deque
from operator import length_hint

n = 10


class __TestInvariantWithoutMutations:

    def test_invariant(self):
        it = self.it
        for i in reversed(range(1, n+1)):
            self.assertEqual(length_hint(it), i)
            next(it)
        self.assertEqual(length_hint(it), 0)
        self.assertRaises(StopIteration, next, it)
        self.assertEqual(length_hint(it), 0)

class __TestTemporarilyImmutable(__TestInvariantWithoutMutations):

    def test_immutable_during_iteration(self):
        # objects such as deques, sets, and dictionaries enforce
        # length immutability  during iteration

        it = self.it
        self.assertEqual(length_hint(it), n)
        next(it)
        self.assertEqual(length_hint(it), n-1)
        self.mutate()
        self.assertRaises(RuntimeError, next, it)
        self.assertEqual(length_hint(it), 0)

## ------- Concrete Type Tests -------

class TestRepeat(__TestInvariantWithoutMutations, CPythonTestCase):

    def setUp(self):
        super().setUp()
        self.it = repeat(None, n)

class TestXrange(__TestInvariantWithoutMutations, CPythonTestCase):

    def setUp(self):
        super().setUp()
        self.it = iter(range(n))

class TestXrangeCustomReversed(__TestInvariantWithoutMutations, CPythonTestCase):

    def setUp(self):
        super().setUp()
        self.it = reversed(range(n))

class TestTuple(__TestInvariantWithoutMutations, CPythonTestCase):

    def setUp(self):
        super().setUp()
        self.it = iter(tuple(range(n)))

## ------- Types that should not be mutated during iteration -------

class TestDeque(__TestTemporarilyImmutable, CPythonTestCase):

    def setUp(self):
        super().setUp()
        d = deque(range(n))
        self.it = iter(d)
        self.mutate = d.pop

class TestDequeReversed(__TestTemporarilyImmutable, CPythonTestCase):

    def setUp(self):
        super().setUp()
        d = deque(range(n))
        self.it = reversed(d)
        self.mutate = d.pop

class TestDictKeys(__TestTemporarilyImmutable, CPythonTestCase):

    def setUp(self):
        super().setUp()
        d = dict.fromkeys(range(n))
        self.it = iter(d)
        self.mutate = d.popitem

class TestDictItems(__TestTemporarilyImmutable, CPythonTestCase):

    def setUp(self):
        super().setUp()
        d = dict.fromkeys(range(n))
        self.it = iter(d.items())
        self.mutate = d.popitem

class TestDictValues(__TestTemporarilyImmutable, CPythonTestCase):

    def setUp(self):
        super().setUp()
        d = dict.fromkeys(range(n))
        self.it = iter(d.values())
        self.mutate = d.popitem

class TestSet(__TestTemporarilyImmutable, CPythonTestCase):

    def setUp(self):
        super().setUp()
        d = set(range(n))
        self.it = iter(d)
        self.mutate = d.pop

## ------- Types that can mutate during iteration -------

class TestList(__TestInvariantWithoutMutations, CPythonTestCase):

    def setUp(self):
        super().setUp()
        self.it = iter(range(n))

    def test_mutation(self):
        d = list(range(n))
        it = iter(d)
        next(it)
        next(it)
        self.assertEqual(length_hint(it), n - 2)
        d.append(n)
        self.assertEqual(length_hint(it), n - 1)  # grow with append
        d[1:] = []
        self.assertEqual(length_hint(it), 0)
        self.assertEqual(list(it), [])
        d.extend(range(20))
        self.assertEqual(length_hint(it), 0)


class TestListReversed(__TestInvariantWithoutMutations, CPythonTestCase):

    def setUp(self):
        super().setUp()
        self.it = reversed(range(n))

    def test_mutation(self):
        d = list(range(n))
        it = reversed(d)
        next(it)
        next(it)
        self.assertEqual(length_hint(it), n - 2)
        d.append(n)
        self.assertEqual(length_hint(it), n - 2)  # ignore append
        d[1:] = []
        self.assertEqual(length_hint(it), 0)
        self.assertEqual(list(it), [])  # confirm invariant
        d.extend(range(20))
        self.assertEqual(length_hint(it), 0)

## -- Check to make sure exceptions are not suppressed by __length_hint__()


class BadLen(object):
    def __iter__(self):
        return iter(range(10))

    def __len__(self):
        raise RuntimeError('hello')


class BadLengthHint(object):
    def __iter__(self):
        return iter(range(10))

    def __length_hint__(self):
        raise RuntimeError('hello')


class NoneLengthHint(object):
    def __iter__(self):
        return iter(range(10))

    def __length_hint__(self):
        return NotImplemented


class TestLengthHintExceptions(CPythonTestCase):

    def test_issue1242657(self):
        self.assertRaises(RuntimeError, list, BadLen())
        self.assertRaises(RuntimeError, list, BadLengthHint())
        self.assertRaises(RuntimeError, [].extend, BadLen())
        self.assertRaises(RuntimeError, [].extend, BadLengthHint())
        b = bytearray(range(10))
        self.assertRaises(RuntimeError, b.extend, BadLen())
        self.assertRaises(RuntimeError, b.extend, BadLengthHint())

    def test_invalid_hint(self):
        # Make sure an invalid result doesn't muck-up the works
        self.assertEqual(list(NoneLengthHint()), list(range(10)))


if __name__ == "__main__":
    run_tests()
