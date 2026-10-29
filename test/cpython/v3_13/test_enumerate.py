# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_enumerate.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import unittest
import operator
import sys
import pickle
import gc

from test import support

class G:
    'Sequence using __getitem__'
    def __init__(self, seqn):
        self.seqn = seqn
    def __getitem__(self, i):
        return self.seqn[i]

class I:
    'Sequence using iterator protocol'
    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i >= len(self.seqn): raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v

class Ig:
    'Sequence using iterator protocol defined with a generator'
    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        for val in self.seqn:
            yield val

class X:
    'Missing __getitem__ and __iter__'
    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0
    def __next__(self):
        if self.i >= len(self.seqn): raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v

class E:
    'Test propagation of exceptions'
    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        3 // 0

class N:
    'Iterator missing __next__()'
    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        return self

class PickleTest:
    # Helper to check picklability
    def check_pickle(self, itorg, seq):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            d = pickle.dumps(itorg, proto)
            it = pickle.loads(d)
            self.assertEqual(type(itorg), type(it))
            self.assertEqual(list(it), seq)

            it = pickle.loads(d)
            try:
                next(it)
            except StopIteration:
                self.assertFalse(seq[1:])
                continue
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            self.assertEqual(list(it), seq[1:])

class EnumerateTestCase(CPythonTestCase, PickleTest):

    enum = enumerate
    seq, res = 'abc', [(0,'a'), (1,'b'), (2,'c')]

    def test_basicfunction(self):
        self.assertEqual(type(self.enum(self.seq)), self.enum)
        e = self.enum(self.seq)
        self.assertEqual(iter(e), e)
        self.assertEqual(list(self.enum(self.seq)), self.res)
        self.enum.__doc__

    def test_pickle(self):
        self.check_pickle(self.enum(self.seq), self.res)

    def test_getitemseqn(self):
        self.assertEqual(list(self.enum(G(self.seq))), self.res)
        e = self.enum(G(''))
        self.assertRaises(StopIteration, next, e)

    def test_iteratorseqn(self):
        self.assertEqual(list(self.enum(I(self.seq))), self.res)
        e = self.enum(I(''))
        self.assertRaises(StopIteration, next, e)

    def test_iteratorgenerator(self):
        self.assertEqual(list(self.enum(Ig(self.seq))), self.res)
        e = self.enum(Ig(''))
        self.assertRaises(StopIteration, next, e)

    def test_noniterable(self):
        self.assertRaises(TypeError, self.enum, X(self.seq))

    def test_illformediterable(self):
        self.assertRaises(TypeError, self.enum, N(self.seq))

    def test_exception_propagation(self):
        self.assertRaises(ZeroDivisionError, list, self.enum(E(self.seq)))

    def test_argumentcheck(self):
        self.assertRaises(TypeError, self.enum) # no arguments
        self.assertRaises(TypeError, self.enum, 1) # wrong type (not iterable)
        self.assertRaises(TypeError, self.enum, 'abc', 'a') # wrong type
        self.assertRaises(TypeError, self.enum, 'abc', 2, 3) # too many arguments

    def test_kwargs(self):
        self.assertEqual(list(self.enum(iterable=Ig(self.seq))), self.res)
        expected = list(self.enum(Ig(self.seq), 0))
        self.assertEqual(list(self.enum(iterable=Ig(self.seq), start=0)),
                         expected)
        self.assertEqual(list(self.enum(start=0, iterable=Ig(self.seq))),
                         expected)
        self.assertRaises(TypeError, self.enum, iterable=[], x=3)
        self.assertRaises(TypeError, self.enum, start=0, x=3)
        self.assertRaises(TypeError, self.enum, x=0, y=3)
        self.assertRaises(TypeError, self.enum, x=0)

    @support.cpython_only
    def test_tuple_reuse(self):
        # Tests an implementation detail where tuple is reused
        # whenever nothing else holds a reference to it
        self.assertEqual(len(set(map(id, list(enumerate(self.seq))))), len(self.seq))
        self.assertEqual(len(set(map(id, enumerate(self.seq)))), min(1,len(self.seq)))

    @support.cpython_only
    def test_enumerate_result_gc(self):
        # bpo-42536: enumerate's tuple-reuse speed trick breaks the GC's
        # assumptions about what can be untracked. Make sure we re-track result
        # tuples whenever we reuse them.
        it = self.enum([[]])
        gc.collect()
        # That GC collection probably untracked the recycled internal result
        # tuple, which is initialized to (None, None). Make sure it's re-tracked
        # when it's mutated and returned from __next__:
        self.assertTrue(gc.is_tracked(next(it)))

class MyEnum(enumerate):
    pass

class SubclassTestCase(EnumerateTestCase):

    enum = MyEnum

class TestEmpty(EnumerateTestCase):

    seq, res = '', []

class TestBig(EnumerateTestCase):

    seq = range(10,20000,2)
    res = list(zip(range(20000), seq))

class TestReversed(CPythonTestCase, PickleTest):

    def test_simple(self):
        class A:
            def __getitem__(self, i):
                if i < 5:
                    return str(i)
                raise StopIteration
            def __len__(self):
                return 5
        for data in ('abc', range(5), tuple(enumerate('abc')), A(),
                    range(1,17,5), dict.fromkeys('abcde')):
            self.assertEqual(list(data)[::-1], list(reversed(data)))
        # don't allow keyword arguments
        self.assertRaises(TypeError, reversed, [], a=1)

    def test_range_optimization(self):
        x = range(1)
        self.assertEqual(type(reversed(x)), type(iter(x)))

    def test_len(self):
        for s in ('hello', tuple('hello'), list('hello'), range(5)):
            self.assertEqual(operator.length_hint(reversed(s)), len(s))
            r = reversed(s)
            list(r)
            self.assertEqual(operator.length_hint(r), 0)
        class SeqWithWeirdLen:
            called = False
            def __len__(self):
                if not self.called:
                    self.called = True
                    return 10
                raise ZeroDivisionError
            def __getitem__(self, index):
                return index
        r = reversed(SeqWithWeirdLen())
        self.assertRaises(ZeroDivisionError, operator.length_hint, r)


    def test_gc(self):
        class Seq:
            def __len__(self):
                return 10
            def __getitem__(self, index):
                return index
        s = Seq()
        r = reversed(s)
        s.r = r

    def test_args(self):
        self.assertRaises(TypeError, reversed)
        self.assertRaises(TypeError, reversed, [], 'extra')

    @unittest.skipUnless(hasattr(sys, 'getrefcount'), 'test needs sys.getrefcount()')
    def test_bug1229429(self):
        # this bug was never in reversed, it was in
        # PyObject_CallMethod, and reversed_new calls that sometimes.
        def f():
            pass
        r = f.__reversed__ = object()
        rc = sys.getrefcount(r)
        for i in range(10):
            try:
                reversed(f)
            except TypeError:
                pass
            else:
                self.fail("non-callable __reversed__ didn't raise!")
        self.assertEqual(rc, sys.getrefcount(r))

    def test_objmethods(self):
        # Objects must have __len__() and __getitem__() implemented.
        class NoLen(object):
            def __getitem__(self, i): return 1
        nl = NoLen()
        self.assertRaises(TypeError, reversed, nl)

        class NoGetItem(object):
            def __len__(self): return 2
        ngi = NoGetItem()
        self.assertRaises(TypeError, reversed, ngi)

        class Blocked(object):
            def __getitem__(self, i): return 1
            def __len__(self): return 2
            __reversed__ = None
        b = Blocked()
        self.assertRaises(TypeError, reversed, b)

    def test_pickle(self):
        for data in 'abc', range(5), tuple(enumerate('abc')), range(1,17,5):
            self.check_pickle(reversed(data), list(data)[::-1])


class EnumerateStartTestCase(EnumerateTestCase):

    def test_basicfunction(self):
        e = self.enum(self.seq)
        self.assertEqual(iter(e), e)
        self.assertEqual(list(self.enum(self.seq)), self.res)


class TestStart(EnumerateStartTestCase):
    def enum(self, iterable, start=11):
        return enumerate(iterable, start=start)

    seq, res = 'abc', [(11, 'a'), (12, 'b'), (13, 'c')]


class TestLongStart(EnumerateStartTestCase):
    def enum(self, iterable, start=sys.maxsize + 1):
        return enumerate(iterable, start=start)

    seq, res = 'abc', [(sys.maxsize+1,'a'), (sys.maxsize+2,'b'),
                       (sys.maxsize+3,'c')]


if __name__ == "__main__":
    run_tests()
