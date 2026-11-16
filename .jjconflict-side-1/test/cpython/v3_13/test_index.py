# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_index.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import unittest
from test import support
import operator
maxsize = support.MAX_Py_ssize_t

class newstyle:
    def __index__(self):
        return self.ind

class TrapInt(int):
    def __index__(self):
        return int(self)

class BaseTestCase(CPythonTestCase):
    def setUp(self):
        super().setUp()
        self.o = newstyle()
        self.n = newstyle()

    def test_basic(self):
        self.o.ind = -2
        self.n.ind = 2
        self.assertEqual(operator.index(self.o), -2)
        self.assertEqual(operator.index(self.n), 2)

    def test_slice(self):
        self.o.ind = 1
        self.n.ind = 2
        slc = slice(self.o, self.o, self.o)
        check_slc = slice(1, 1, 1)
        self.assertEqual(slc.indices(self.o), check_slc.indices(1))
        slc = slice(self.n, self.n, self.n)
        check_slc = slice(2, 2, 2)
        self.assertEqual(slc.indices(self.n), check_slc.indices(2))

    def test_wrappers(self):
        self.o.ind = 4
        self.n.ind = 5
        self.assertEqual(6 .__index__(), 6)
        self.assertEqual(-7 .__index__(), -7)
        self.assertEqual(self.o.__index__(), 4)
        self.assertEqual(self.n.__index__(), 5)
        self.assertEqual(True.__index__(), 1)
        self.assertEqual(False.__index__(), 0)

    def test_subclasses(self):
        r = list(range(10))
        self.assertEqual(r[TrapInt(5):TrapInt(10)], r[5:10])
        self.assertEqual(slice(TrapInt()).indices(0), (0,0,1))

    def test_error(self):
        self.o.ind = 'dumb'
        self.n.ind = 'bad'
        self.assertRaises(TypeError, operator.index, self.o)
        self.assertRaises(TypeError, operator.index, self.n)
        self.assertRaises(TypeError, slice(self.o).indices, 0)
        self.assertRaises(TypeError, slice(self.n).indices, 0)

    def test_int_subclass_with_index(self):
        # __index__ should be used when computing indices, even for int
        # subclasses.  See issue #17576.
        class MyInt(int):
            def __index__(self):
                return int(str(self)) + 1

        my_int = MyInt(7)
        direct_index = my_int.__index__()
        operator_index = operator.index(my_int)
        self.assertEqual(direct_index, 8)
        self.assertEqual(operator_index, 7)
        # Both results should be of exact type int.
        self.assertIs(type(direct_index), int)
        #self.assertIs(type(operator_index), int)

    def test_index_returns_int_subclass(self):
        class BadInt:
            def __index__(self):
                return True

        class BadInt2(int):
            def __index__(self):
                return True

        bad_int = BadInt()
        with self.assertWarns(DeprecationWarning):
            n = operator.index(bad_int)
        self.assertEqual(n, 1)

        bad_int = BadInt2()
        n = operator.index(bad_int)
        self.assertEqual(n, 0)


class SeqTestCase:
    # This test case isn't run directly. It just defines common tests
    # to the different sequence types below
    def setUp(self):
        super().setUp()
        self.o = newstyle()
        self.n = newstyle()
        self.o2 = newstyle()
        self.n2 = newstyle()

    def test_index(self):
        self.o.ind = -2
        self.n.ind = 2
        self.assertEqual(self.seq[self.n], self.seq[2])
        self.assertEqual(self.seq[self.o], self.seq[-2])

    def test_slice(self):
        self.o.ind = 1
        self.o2.ind = 3
        self.n.ind = 2
        self.n2.ind = 4
        self.assertEqual(self.seq[self.o:self.o2], self.seq[1:3])
        self.assertEqual(self.seq[self.n:self.n2], self.seq[2:4])

    def test_slice_bug7532(self):
        seqlen = len(self.seq)
        self.o.ind = int(seqlen * 1.5)
        self.n.ind = seqlen + 2
        self.assertEqual(self.seq[self.o:], self.seq[0:0])
        self.assertEqual(self.seq[:self.o], self.seq)
        self.assertEqual(self.seq[self.n:], self.seq[0:0])
        self.assertEqual(self.seq[:self.n], self.seq)
        self.o2.ind = -seqlen - 2
        self.n2.ind = -int(seqlen * 1.5)
        self.assertEqual(self.seq[self.o2:], self.seq)
        self.assertEqual(self.seq[:self.o2], self.seq[0:0])
        self.assertEqual(self.seq[self.n2:], self.seq)
        self.assertEqual(self.seq[:self.n2], self.seq[0:0])

    def test_repeat(self):
        self.o.ind = 3
        self.n.ind = 2
        self.assertEqual(self.seq * self.o, self.seq * 3)
        self.assertEqual(self.seq * self.n, self.seq * 2)
        self.assertEqual(self.o * self.seq, self.seq * 3)
        self.assertEqual(self.n * self.seq, self.seq * 2)

    def test_wrappers(self):
        self.o.ind = 4
        self.n.ind = 5
        self.assertEqual(self.seq.__getitem__(self.o), self.seq[4])
        self.assertEqual(self.seq.__mul__(self.o), self.seq * 4)
        self.assertEqual(self.seq.__rmul__(self.o), self.seq * 4)
        self.assertEqual(self.seq.__getitem__(self.n), self.seq[5])
        self.assertEqual(self.seq.__mul__(self.n), self.seq * 5)
        self.assertEqual(self.seq.__rmul__(self.n), self.seq * 5)

    def test_subclasses(self):
        self.assertEqual(self.seq[TrapInt()], self.seq[0])

    def test_error(self):
        self.o.ind = 'dumb'
        self.n.ind = 'bad'
        indexobj = lambda x, obj: obj.seq[x]
        self.assertRaises(TypeError, indexobj, self.o, self)
        self.assertRaises(TypeError, indexobj, self.n, self)
        sliceobj = lambda x, obj: obj.seq[x:]
        self.assertRaises(TypeError, sliceobj, self.o, self)
        self.assertRaises(TypeError, sliceobj, self.n, self)


class ListTestCase(SeqTestCase, CPythonTestCase):
    seq = [0,10,20,30,40,50]

    def test_setdelitem(self):
        self.o.ind = -2
        self.n.ind = 2
        lst = list('ab!cdefghi!j')
        del lst[self.o]
        del lst[self.n]
        lst[self.o] = 'X'
        lst[self.n] = 'Y'
        self.assertEqual(lst, list('abYdefghXj'))

        lst = [5, 6, 7, 8, 9, 10, 11]
        lst.__setitem__(self.n, "here")
        self.assertEqual(lst, [5, 6, "here", 8, 9, 10, 11])
        lst.__delitem__(self.n)
        self.assertEqual(lst, [5, 6, 8, 9, 10, 11])

    def test_inplace_repeat(self):
        self.o.ind = 2
        self.n.ind = 3
        lst = [6, 4]
        lst *= self.o
        self.assertEqual(lst, [6, 4, 6, 4])
        lst *= self.n
        self.assertEqual(lst, [6, 4, 6, 4] * 3)

        lst = [5, 6, 7, 8, 9, 11]
        l2 = lst.__imul__(self.n)
        self.assertIs(l2, lst)
        self.assertEqual(lst, [5, 6, 7, 8, 9, 11] * 3)


class NewSeq:

    def __init__(self, iterable):
        self._list = list(iterable)

    def __repr__(self):
        return repr(self._list)

    def __eq__(self, other):
        return self._list == other

    def __len__(self):
        return len(self._list)

    def __mul__(self, n):
        return self.__class__(self._list*n)
    __rmul__ = __mul__

    def __getitem__(self, index):
        return self._list[index]


class TupleTestCase(SeqTestCase, CPythonTestCase):
    seq = (0,10,20,30,40,50)

class ByteArrayTestCase(SeqTestCase, CPythonTestCase):
    seq = bytearray(b"this is a test")

class BytesTestCase(SeqTestCase, CPythonTestCase):
    seq = b"this is a test"

class StringTestCase(SeqTestCase, CPythonTestCase):
    seq = "this is a test"

class NewSeqTestCase(SeqTestCase, CPythonTestCase):
    seq = NewSeq((0,10,20,30,40,50))



class RangeTestCase(CPythonTestCase):

    def test_range(self):
        n = newstyle()
        n.ind = 5
        self.assertEqual(range(1, 20)[n], 6)
        self.assertEqual(range(1, 20).__getitem__(n), 6)


class OverflowTestCase(CPythonTestCase):

    def setUp(self):
        super().setUp()
        self.pos = 2**100
        self.neg = -self.pos

    def test_large_longs(self):
        self.assertEqual(self.pos.__index__(), self.pos)
        self.assertEqual(self.neg.__index__(), self.neg)

    def test_getitem(self):
        class GetItem:
            def __len__(self):
                assert False, "__len__ should not be invoked"
            def __getitem__(self, key):
                return key
        x = GetItem()
        self.assertEqual(x[self.pos], self.pos)
        self.assertEqual(x[self.neg], self.neg)
        self.assertEqual(x[self.neg:self.pos].indices(maxsize),
                         (0, maxsize, 1))
        self.assertEqual(x[self.neg:self.pos:1].indices(maxsize),
                         (0, maxsize, 1))

    def test_sequence_repeat(self):
        self.assertRaises(OverflowError, lambda: "a" * self.pos)
        self.assertRaises(OverflowError, lambda: "a" * self.neg)


if __name__ == "__main__":
    run_tests()
