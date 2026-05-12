# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_deque.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    skipIfTorchDynamo,
    run_tests,
)


# ======= END DYNAMO PATCH =======

from collections import deque
import doctest
import unittest
from test import support
import seq_tests
import gc
import weakref
import copy
import pickle
import random
import struct

BIG = 100000

def fail():
    raise SyntaxError
    yield 1

class BadCmp:
    def __eq__(self, other):
        raise RuntimeError

class MutateCmp:
    def __init__(self, deque, result):
        self.deque = deque
        self.result = result
    def __eq__(self, other):
        self.deque.clear()
        return self.result

class TestBasic(CPythonTestCase):

    def test_basics(self):
        d = deque(range(-5125, -5000))
        d.__init__(range(200))
        for i in range(200, 400):
            d.append(i)
        for i in reversed(range(-200, 0)):
            d.appendleft(i)
        self.assertEqual(list(d), list(range(-200, 400)))
        self.assertEqual(len(d), 600)

        left = [d.popleft() for i in range(250)]
        self.assertEqual(left, list(range(-200, 50)))
        self.assertEqual(list(d), list(range(50, 400)))

        right = [d.pop() for i in range(250)]
        right.reverse()
        self.assertEqual(right, list(range(150, 400)))
        self.assertEqual(list(d), list(range(50, 150)))

    def test_maxlen(self):
        self.assertRaises(ValueError, deque, 'abc', -1)
        self.assertRaises(ValueError, deque, 'abc', -2)
        it = iter(range(10))
        d = deque(it, maxlen=3)
        self.assertEqual(list(it), [])
        self.assertEqual(repr(d), 'deque([7, 8, 9], maxlen=3)')
        self.assertEqual(list(d), [7, 8, 9])
        self.assertEqual(d, deque(range(10), 3))
        d.append(10)
        self.assertEqual(list(d), [8, 9, 10])
        d.appendleft(7)
        self.assertEqual(list(d), [7, 8, 9])
        d.extend([10, 11])
        self.assertEqual(list(d), [9, 10, 11])
        d.extendleft([8, 7])
        self.assertEqual(list(d), [7, 8, 9])
        d = deque(range(200), maxlen=10)
        d.append(d)
        self.assertEqual(repr(d)[-30:], ', 198, 199, [...]], maxlen=10)')
        d = deque(range(10), maxlen=None)
        self.assertEqual(repr(d), 'deque([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])')

    def test_maxlen_zero(self):
        it = iter(range(100))
        deque(it, maxlen=0)
        self.assertEqual(list(it), [])

        it = iter(range(100))
        d = deque(maxlen=0)
        d.extend(it)
        self.assertEqual(list(it), [])

        it = iter(range(100))
        d = deque(maxlen=0)
        d.extendleft(it)
        self.assertEqual(list(it), [])

    def test_maxlen_attribute(self):
        self.assertEqual(deque().maxlen, None)
        self.assertEqual(deque('abc').maxlen, None)
        self.assertEqual(deque('abc', maxlen=4).maxlen, 4)
        self.assertEqual(deque('abc', maxlen=2).maxlen, 2)
        self.assertEqual(deque('abc', maxlen=0).maxlen, 0)
        with self.assertRaises(AttributeError):
            d = deque('abc')
            d.maxlen = 10

    def test_count(self):
        for s in ('', 'abracadabra', 'simsalabim'*500+'abc'):
            s = list(s)
            d = deque(s)
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                self.assertEqual(s.count(letter), d.count(letter), (s, d, letter))
        self.assertRaises(TypeError, d.count)       # too few args
        self.assertRaises(TypeError, d.count, 1, 2) # too many args
        class BadCompare:
            def __eq__(self, other):
                raise ArithmeticError
        d = deque([1, 2, BadCompare(), 3])
        self.assertRaises(ArithmeticError, d.count, 2)
        d = deque([1, 2, 3])
        self.assertRaises(ArithmeticError, d.count, BadCompare())
        class MutatingCompare:
            def __eq__(self, other):
                self.d.pop()
                return True
        m = MutatingCompare()
        d = deque([1, 2, 3, m, 4, 5])
        m.d = d
        self.assertRaises(RuntimeError, d.count, 3)

        # test issue11004
        # block advance failed after rotation aligned elements on right side of block
        d = deque([None]*16)
        for i in range(len(d)):
            d.rotate(-1)
        d.rotate(1)
        self.assertEqual(d.count(1), 0)
        self.assertEqual(d.count(None), 16)

    def test_comparisons(self):
        d = deque('xabc')
        d.popleft()
        for e in [d, deque('abc'), deque('ab'), deque(), list(d)]:
            self.assertEqual(d==e, type(d)==type(e) and list(d)==list(e))
            self.assertEqual(d!=e, not(type(d)==type(e) and list(d)==list(e)))

        args = map(deque, ('', 'a', 'b', 'ab', 'ba', 'abc', 'xba', 'xabc', 'cba'))
        for x in args:
            for y in args:
                self.assertEqual(x == y, list(x) == list(y), (x,y))
                self.assertEqual(x != y, list(x) != list(y), (x,y))
                self.assertEqual(x <  y, list(x) <  list(y), (x,y))
                self.assertEqual(x <= y, list(x) <= list(y), (x,y))
                self.assertEqual(x >  y, list(x) >  list(y), (x,y))
                self.assertEqual(x >= y, list(x) >= list(y), (x,y))

    def test_contains(self):
        n = 200

        d = deque(range(n))
        for i in range(n):
            self.assertTrue(i in d)
        self.assertTrue((n+1) not in d)

        # Test detection of mutation during iteration
        d = deque(range(n))
        d[n//2] = MutateCmp(d, False)
        with self.assertRaises(RuntimeError):
            n in d

        # Test detection of comparison exceptions
        d = deque(range(n))
        d[n//2] = BadCmp()
        with self.assertRaises(RuntimeError):
            n in d

    def test_contains_count_index_stop_crashes(self):
        class A:
            def __eq__(self, other):
                d.clear()
                return NotImplemented
        d = deque([A(), A()])
        with self.assertRaises(RuntimeError):
            _ = 3 in d
        d = deque([A(), A()])
        with self.assertRaises(RuntimeError):
            _ = d.count(3)

        d = deque([A()])
        with self.assertRaises(RuntimeError):
            d.index(0)

    def test_extend(self):
        d = deque('a')
        self.assertRaises(TypeError, d.extend, 1)
        d.extend('bcd')
        self.assertEqual(list(d), list('abcd'))
        d.extend(d)
        self.assertEqual(list(d), list('abcdabcd'))

    def test_add(self):
        d = deque()
        e = deque('abc')
        f = deque('def')
        self.assertEqual(d + d, deque())
        self.assertEqual(e + f, deque('abcdef'))
        self.assertEqual(e + e, deque('abcabc'))
        self.assertEqual(e + d, deque('abc'))
        self.assertEqual(d + e, deque('abc'))
        self.assertIsNot(d + d, deque())
        self.assertIsNot(e + d, deque('abc'))
        self.assertIsNot(d + e, deque('abc'))

        g = deque('abcdef', maxlen=4)
        h = deque('gh')
        self.assertEqual(g + h, deque('efgh'))

        with self.assertRaises(TypeError):
            deque('abc') + 'def'

    def test_iadd(self):
        d = deque('a')
        d += 'bcd'
        self.assertEqual(list(d), list('abcd'))
        d += d
        self.assertEqual(list(d), list('abcdabcd'))

    def test_extendleft(self):
        d = deque('a')
        self.assertRaises(TypeError, d.extendleft, 1)
        d.extendleft('bcd')
        self.assertEqual(list(d), list(reversed('abcd')))
        d.extendleft(d)
        self.assertEqual(list(d), list('abcddcba'))
        d = deque()
        d.extendleft(range(1000))
        self.assertEqual(list(d), list(reversed(range(1000))))
        self.assertRaises(SyntaxError, d.extendleft, fail())

    def test_getitem(self):
        n = 200
        d = deque(range(n))
        l = list(range(n))
        for i in range(n):
            d.popleft()
            l.pop(0)
            if random.random() < 0.5:
                d.append(i)
                l.append(i)
            for j in range(1-len(l), len(l)):
                assert d[j] == l[j]

        d = deque('superman')
        self.assertEqual(d[0], 's')
        self.assertEqual(d[-1], 'n')
        d = deque()
        self.assertRaises(IndexError, d.__getitem__, 0)
        self.assertRaises(IndexError, d.__getitem__, -1)

    def test_index(self):
        for n in 1, 2, 30, 40, 200:

            d = deque(range(n))
            for i in range(n):
                self.assertEqual(d.index(i), i)

            with self.assertRaises(ValueError):
                d.index(n+1)

            # Test detection of mutation during iteration
            d = deque(range(n))
            d[n//2] = MutateCmp(d, False)
            with self.assertRaises(RuntimeError):
                d.index(n)

            # Test detection of comparison exceptions
            d = deque(range(n))
            d[n//2] = BadCmp()
            with self.assertRaises(RuntimeError):
                d.index(n)

        # Test start and stop arguments behavior matches list.index()
        elements = 'ABCDEFGHI'
        nonelement = 'Z'
        d = deque(elements * 2)
        s = list(elements * 2)
        for start in range(-5 - len(s)*2, 5 + len(s) * 2):
            for stop in range(-5 - len(s)*2, 5 + len(s) * 2):
                for element in elements + 'Z':
                    try:
                        target = s.index(element, start, stop)
                    except ValueError:
                        with self.assertRaises(ValueError):
                            d.index(element, start, stop)
                    else:
                        self.assertEqual(d.index(element, start, stop), target)

        # Test large start argument
        d = deque(range(0, 10000, 10))
        for step in range(100):
            i = d.index(8500, 700)
            self.assertEqual(d[i], 8500)
            # Repeat test with a different internal offset
            d.rotate()

    def test_index_bug_24913(self):
        d = deque('A' * 3)
        with self.assertRaises(ValueError):
            i = d.index("Hello world", 0, 4)

    def test_insert(self):
        # Test to make sure insert behaves like lists
        elements = 'ABCDEFGHI'
        for i in range(-5 - len(elements)*2, 5 + len(elements) * 2):
            d = deque('ABCDEFGHI')
            s = list('ABCDEFGHI')
            d.insert(i, 'Z')
            s.insert(i, 'Z')
            self.assertEqual(list(d), s)

    def test_insert_bug_26194(self):
        data = 'ABC'
        d = deque(data, maxlen=len(data))
        with self.assertRaises(IndexError):
            d.insert(2, None)

        elements = 'ABCDEFGHI'
        for i in range(-len(elements), len(elements)):
            d = deque(elements, maxlen=len(elements)+1)
            d.insert(i, 'Z')
            if i >= 0:
                self.assertEqual(d[i], 'Z')
            else:
                self.assertEqual(d[i-1], 'Z')

    def test_imul(self):
        for n in (-10, -1, 0, 1, 2, 10, 1000):
            d = deque()
            d *= n
            self.assertEqual(d, deque())
            self.assertIsNone(d.maxlen)

        for n in (-10, -1, 0, 1, 2, 10, 1000):
            d = deque('a')
            d *= n
            self.assertEqual(d, deque('a' * n))
            self.assertIsNone(d.maxlen)

        for n in (-10, -1, 0, 1, 2, 10, 499, 500, 501, 1000):
            d = deque('a', 500)
            d *= n
            self.assertEqual(d, deque('a' * min(n, 500)))
            self.assertEqual(d.maxlen, 500)

        for n in (-10, -1, 0, 1, 2, 10, 1000):
            d = deque('abcdef')
            d *= n
            self.assertEqual(d, deque('abcdef' * n))
            self.assertIsNone(d.maxlen)

        for n in (-10, -1, 0, 1, 2, 10, 499, 500, 501, 1000):
            d = deque('abcdef', 500)
            d *= n
            self.assertEqual(d, deque(('abcdef' * n)[-500:]))
            self.assertEqual(d.maxlen, 500)

    def test_mul(self):
        d = deque('abc')
        self.assertEqual(d * -5, deque())
        self.assertEqual(d * 0, deque())
        self.assertEqual(d * 1, deque('abc'))
        self.assertEqual(d * 2, deque('abcabc'))
        self.assertEqual(d * 3, deque('abcabcabc'))
        self.assertIsNot(d * 1, d)

        self.assertEqual(deque() * 0, deque())
        self.assertEqual(deque() * 1, deque())
        self.assertEqual(deque() * 5, deque())

        self.assertEqual(-5 * d, deque())
        self.assertEqual(0 * d, deque())
        self.assertEqual(1 * d, deque('abc'))
        self.assertEqual(2 * d, deque('abcabc'))
        self.assertEqual(3 * d, deque('abcabcabc'))

        d = deque('abc', maxlen=5)
        self.assertEqual(d * -5, deque())
        self.assertEqual(d * 0, deque())
        self.assertEqual(d * 1, deque('abc'))
        self.assertEqual(d * 2, deque('bcabc'))
        self.assertEqual(d * 30, deque('bcabc'))

    def test_setitem(self):
        n = 200
        d = deque(range(n))
        for i in range(n):
            d[i] = 10 * i
        self.assertEqual(list(d), [10*i for i in range(n)])
        l = list(d)
        for i in range(1-n, 0, -1):
            d[i] = 7*i
            l[i] = 7*i
        self.assertEqual(list(d), l)

    def test_delitem(self):
        n = 500         # O(n**2) test, don't make this too big
        d = deque(range(n))
        self.assertRaises(IndexError, d.__delitem__, -n-1)
        self.assertRaises(IndexError, d.__delitem__, n)
        for i in range(n):
            self.assertEqual(len(d), n-i)
            j = random.randrange(-len(d), len(d))
            val = d[j]
            self.assertIn(val, d)
            del d[j]
            self.assertNotIn(val, d)
        self.assertEqual(len(d), 0)

    def test_reverse(self):
        n = 500         # O(n**2) test, don't make this too big
        data = [random.random() for i in range(n)]
        for i in range(n):
            d = deque(data[:i])
            r = d.reverse()
            self.assertEqual(list(d), list(reversed(data[:i])))
            self.assertIs(r, None)
            d.reverse()
            self.assertEqual(list(d), data[:i])
        self.assertRaises(TypeError, d.reverse, 1)          # Arity is zero

    def test_rotate(self):
        s = tuple('abcde')
        n = len(s)

        d = deque(s)
        d.rotate(1)             # verify rot(1)
        self.assertEqual(''.join(d), 'eabcd')

        d = deque(s)
        d.rotate(-1)            # verify rot(-1)
        self.assertEqual(''.join(d), 'bcdea')
        d.rotate()              # check default to 1
        self.assertEqual(tuple(d), s)

        for i in range(n*3):
            d = deque(s)
            e = deque(d)
            d.rotate(i)         # check vs. rot(1) n times
            for j in range(i):
                e.rotate(1)
            self.assertEqual(tuple(d), tuple(e))
            d.rotate(-i)        # check that it works in reverse
            self.assertEqual(tuple(d), s)
            e.rotate(n-i)       # check that it wraps forward
            self.assertEqual(tuple(e), s)

        for i in range(n*3):
            d = deque(s)
            e = deque(d)
            d.rotate(-i)
            for j in range(i):
                e.rotate(-1)    # check vs. rot(-1) n times
            self.assertEqual(tuple(d), tuple(e))
            d.rotate(i)         # check that it works in reverse
            self.assertEqual(tuple(d), s)
            e.rotate(i-n)       # check that it wraps backaround
            self.assertEqual(tuple(e), s)

        d = deque(s)
        e = deque(s)
        e.rotate(BIG+17)        # verify on long series of rotates
        dr = d.rotate
        for i in range(BIG+17):
            dr()
        self.assertEqual(tuple(d), tuple(e))

        self.assertRaises(TypeError, d.rotate, 'x')   # Wrong arg type
        self.assertRaises(TypeError, d.rotate, 1, 10) # Too many args

        d = deque()
        d.rotate()              # rotate an empty deque
        self.assertEqual(d, deque())

    def test_len(self):
        d = deque('ab')
        self.assertEqual(len(d), 2)
        d.popleft()
        self.assertEqual(len(d), 1)
        d.pop()
        self.assertEqual(len(d), 0)
        self.assertRaises(IndexError, d.pop)
        self.assertEqual(len(d), 0)
        d.append('c')
        self.assertEqual(len(d), 1)
        d.appendleft('d')
        self.assertEqual(len(d), 2)
        d.clear()
        self.assertEqual(len(d), 0)

    def test_underflow(self):
        d = deque()
        self.assertRaises(IndexError, d.pop)
        self.assertRaises(IndexError, d.popleft)

    def test_clear(self):
        d = deque(range(100))
        self.assertEqual(len(d), 100)
        d.clear()
        self.assertEqual(len(d), 0)
        self.assertEqual(list(d), [])
        d.clear()               # clear an empty deque
        self.assertEqual(list(d), [])

    def test_remove(self):
        d = deque('abcdefghcij')
        d.remove('c')
        self.assertEqual(d, deque('abdefghcij'))
        d.remove('c')
        self.assertEqual(d, deque('abdefghij'))
        self.assertRaises(ValueError, d.remove, 'c')
        self.assertEqual(d, deque('abdefghij'))

        # Handle comparison errors
        d = deque(['a', 'b', BadCmp(), 'c'])
        e = deque(d)
        self.assertRaises(RuntimeError, d.remove, 'c')
        for x, y in zip(d, e):
            # verify that original order and values are retained.
            self.assertTrue(x is y)

        # Handle evil mutator
        for match in (True, False):
            d = deque(['ab'])
            d.extend([MutateCmp(d, match), 'c'])
            self.assertRaises(IndexError, d.remove, 'c')
            self.assertEqual(d, deque())

    def test_repr(self):
        d = deque(range(200))
        e = eval(repr(d))
        self.assertEqual(list(d), list(e))
        d.append(d)
        self.assertEqual(repr(d)[-20:], '7, 198, 199, [...]])')

    def test_init(self):
        self.assertRaises(TypeError, deque, 'abc', 2, 3)
        self.assertRaises(TypeError, deque, 1)

    def test_hash(self):
        self.assertRaises(TypeError, hash, deque('abc'))

    @skipIfTorchDynamo("slow Test")
    def test_long_steadystate_queue_popleft(self):
        for size in (0, 1, 2, 100, 1000):
            d = deque(range(size))
            append, pop = d.append, d.popleft
            for i in range(size, BIG):
                append(i)
                x = pop()
                if x != i - size:
                    self.assertEqual(x, i-size)
            self.assertEqual(list(d), list(range(BIG-size, BIG)))

    @skipIfTorchDynamo("slow Test")
    def test_long_steadystate_queue_popright(self):
        for size in (0, 1, 2, 100, 1000):
            d = deque(reversed(range(size)))
            append, pop = d.appendleft, d.pop
            for i in range(size, BIG):
                append(i)
                x = pop()
                if x != i - size:
                    self.assertEqual(x, i-size)
            self.assertEqual(list(reversed(list(d))),
                             list(range(BIG-size, BIG)))

    @skipIfTorchDynamo("slow Test")
    def test_big_queue_popleft(self):
        pass
        d = deque()
        append, pop = d.append, d.popleft
        for i in range(BIG):
            append(i)
        for i in range(BIG):
            x = pop()
            if x != i:
                self.assertEqual(x, i)

    @skipIfTorchDynamo("slow Test")
    def test_big_queue_popright(self):
        d = deque()
        append, pop = d.appendleft, d.pop
        for i in range(BIG):
            append(i)
        for i in range(BIG):
            x = pop()
            if x != i:
                self.assertEqual(x, i)

    @skipIfTorchDynamo("slow Test")
    def test_big_stack_right(self):
        d = deque()
        append, pop = d.append, d.pop
        for i in range(BIG):
            append(i)
        for i in reversed(range(BIG)):
            x = pop()
            if x != i:
                self.assertEqual(x, i)
        self.assertEqual(len(d), 0)

    @skipIfTorchDynamo("slow Test")
    def test_big_stack_left(self):
        d = deque()
        append, pop = d.appendleft, d.popleft
        for i in range(BIG):
            append(i)
        for i in reversed(range(BIG)):
            x = pop()
            if x != i:
                self.assertEqual(x, i)
        self.assertEqual(len(d), 0)

    def test_roundtrip_iter_init(self):
        d = deque(range(200))
        e = deque(d)
        self.assertNotEqual(id(d), id(e))
        self.assertEqual(list(d), list(e))

    def test_pickle(self):
        for d in deque(range(200)), deque(range(200), 100):
            for i in range(pickle.HIGHEST_PROTOCOL + 1):
                s = pickle.dumps(d, i)
                e = pickle.loads(s)
                self.assertNotEqual(id(e), id(d))
                self.assertEqual(list(e), list(d))
                self.assertEqual(e.maxlen, d.maxlen)

    def test_pickle_recursive(self):
        for d in deque('abc'), deque('abc', 3):
            d.append(d)
            for i in range(pickle.HIGHEST_PROTOCOL + 1):
                e = pickle.loads(pickle.dumps(d, i))
                self.assertNotEqual(id(e), id(d))
                self.assertEqual(id(e[-1]), id(e))
                self.assertEqual(e.maxlen, d.maxlen)

    def test_iterator_pickle(self):
        orig = deque(range(200))
        data = [i*1.01 for i in orig]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # initial iterator
            itorg = iter(orig)
            dump = pickle.dumps((itorg, orig), proto)
            it, d = pickle.loads(dump)
            for i, x in enumerate(data):
                d[i] = x
            self.assertEqual(type(it), type(itorg))
            self.assertEqual(list(it), data)

            # running iterator
            next(itorg)
            dump = pickle.dumps((itorg, orig), proto)
            it, d = pickle.loads(dump)
            for i, x in enumerate(data):
                d[i] = x
            self.assertEqual(type(it), type(itorg))
            self.assertEqual(list(it), data[1:])

            # empty iterator
            for i in range(1, len(data)):
                next(itorg)
            dump = pickle.dumps((itorg, orig), proto)
            it, d = pickle.loads(dump)
            for i, x in enumerate(data):
                d[i] = x
            self.assertEqual(type(it), type(itorg))
            self.assertEqual(list(it), [])

            # exhausted iterator
            self.assertRaises(StopIteration, next, itorg)
            dump = pickle.dumps((itorg, orig), proto)
            it, d = pickle.loads(dump)
            for i, x in enumerate(data):
                d[i] = x
            self.assertEqual(type(it), type(itorg))
            self.assertEqual(list(it), [])

    def test_deepcopy(self):
        mut = [10]
        d = deque([mut])
        e = copy.deepcopy(d)
        self.assertEqual(list(d), list(e))
        mut[0] = 11
        self.assertNotEqual(id(d), id(e))
        self.assertNotEqual(list(d), list(e))

    def test_copy(self):
        mut = [10]
        d = deque([mut])
        e = copy.copy(d)
        self.assertEqual(list(d), list(e))
        mut[0] = 11
        self.assertNotEqual(id(d), id(e))
        self.assertEqual(list(d), list(e))

        for i in range(5):
            for maxlen in range(-1, 6):
                s = [random.random() for j in range(i)]
                d = deque(s) if maxlen == -1 else deque(s, maxlen)
                e = d.copy()
                self.assertEqual(d, e)
                self.assertEqual(d.maxlen, e.maxlen)
                self.assertTrue(all(x is y for x, y in zip(d, e)))

    def test_copy_method(self):
        mut = [10]
        d = deque([mut])
        e = d.copy()
        self.assertEqual(list(d), list(e))
        mut[0] = 11
        self.assertNotEqual(id(d), id(e))
        self.assertEqual(list(d), list(e))

    def test_reversed(self):
        for s in ('abcd', range(2000)):
            self.assertEqual(list(reversed(deque(s))), list(reversed(s)))

    def test_reversed_new(self):
        klass = type(reversed(deque()))
        for s in ('abcd', range(2000)):
            self.assertEqual(list(klass(deque(s))), list(reversed(s)))

    def test_gc_doesnt_blowup(self):
        import gc
        # This used to assert-fail in deque_traverse() under a debug
        # build, or run wild with a NULL pointer in a release build.
        d = deque()
        for i in range(100):
            d.append(1)
            gc.collect()

    def test_container_iterator(self):
        # Bug #3680: tp_traverse was not implemented for deque iterator objects
        class C(object):
            pass
        for i in range(2):
            obj = C()
            ref = weakref.ref(obj)
            if i == 0:
                container = deque([obj, 1])
            else:
                container = reversed(deque([obj, 1]))
            obj.x = iter(container)
            del obj, container
            gc.collect()
            self.assertTrue(ref() is None, "Cycle was not collected")

    check_sizeof = support.check_sizeof

    @support.cpython_only
    def test_sizeof(self):
        MAXFREEBLOCKS = 16
        BLOCKLEN = 64
        basesize = support.calcvobjsize('2P5n%dPP' % MAXFREEBLOCKS)
        blocksize = struct.calcsize('P%dPP' % BLOCKLEN)
        self.assertEqual(object.__sizeof__(deque()), basesize)
        check = self.check_sizeof
        check(deque(), basesize + blocksize)
        check(deque('a'), basesize + blocksize)
        check(deque('a' * (BLOCKLEN - 1)), basesize + blocksize)
        check(deque('a' * BLOCKLEN), basesize + 2 * blocksize)
        check(deque('a' * (42 * BLOCKLEN)), basesize + 43 * blocksize)

class TestVariousIteratorArgs(CPythonTestCase):

    def test_constructor(self):
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (seq_tests.Sequence, seq_tests.IterFunc,
                      seq_tests.IterGen, seq_tests.IterFuncStop,
                      seq_tests.itermulti, seq_tests.iterfunc):
                self.assertEqual(list(deque(g(s))), list(g(s)))
            self.assertRaises(TypeError, deque, seq_tests.IterNextOnly(s))
            self.assertRaises(TypeError, deque, seq_tests.IterNoNext(s))
            self.assertRaises(ZeroDivisionError, deque, seq_tests.IterGenExc(s))

    def test_iter_with_altered_data(self):
        d = deque('abcdefg')
        it = iter(d)
        d.pop()
        self.assertRaises(RuntimeError, next, it)

    def test_runtime_error_on_empty_deque(self):
        d = deque()
        it = iter(d)
        d.append(10)
        self.assertRaises(RuntimeError, next, it)

class Deque(deque):
    pass

class DequeWithSlots(deque):
    __slots__ = ('x', 'y', '__dict__')

class DequeWithBadIter(deque):
    def __iter__(self):
        raise TypeError

class TestSubclass(CPythonTestCase):

    def test_basics(self):
        d = Deque(range(25))
        d.__init__(range(200))
        for i in range(200, 400):
            d.append(i)
        for i in reversed(range(-200, 0)):
            d.appendleft(i)
        self.assertEqual(list(d), list(range(-200, 400)))
        self.assertEqual(len(d), 600)

        left = [d.popleft() for i in range(250)]
        self.assertEqual(left, list(range(-200, 50)))
        self.assertEqual(list(d), list(range(50, 400)))

        right = [d.pop() for i in range(250)]
        right.reverse()
        self.assertEqual(right, list(range(150, 400)))
        self.assertEqual(list(d), list(range(50, 150)))

        d.clear()
        self.assertEqual(len(d), 0)

    def test_copy_pickle(self):
        for cls in Deque, DequeWithSlots:
            for d in cls('abc'), cls('abcde', maxlen=4):
                d.x = ['x']
                d.z = ['z']

                e = d.__copy__()
                self.assertEqual(type(d), type(e))
                self.assertEqual(list(d), list(e))

                e = cls(d)
                self.assertEqual(type(d), type(e))
                self.assertEqual(list(d), list(e))

                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    s = pickle.dumps(d, proto)
                    e = pickle.loads(s)
                    self.assertNotEqual(id(d), id(e))
                    self.assertEqual(type(d), type(e))
                    self.assertEqual(list(d), list(e))
                    self.assertEqual(e.x, d.x)
                    self.assertEqual(e.z, d.z)
                    self.assertFalse(hasattr(e, 'y'))

    def test_pickle_recursive(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            for d in Deque('abc'), Deque('abc', 3):
                d.append(d)

                e = pickle.loads(pickle.dumps(d, proto))
                self.assertNotEqual(id(e), id(d))
                self.assertEqual(type(e), type(d))
                self.assertEqual(e.maxlen, d.maxlen)
                dd = d.pop()
                ee = e.pop()
                self.assertEqual(id(ee), id(e))
                self.assertEqual(e, d)

                d.x = d
                e = pickle.loads(pickle.dumps(d, proto))
                self.assertEqual(id(e.x), id(e))

            for d in DequeWithBadIter('abc'), DequeWithBadIter('abc', 2):
                self.assertRaises(TypeError, pickle.dumps, d, proto)

    def test_weakref(self):
        d = deque('gallahad')
        p = weakref.proxy(d)
        self.assertEqual(str(p), str(d))
        d = None
        support.gc_collect()  # For PyPy or other GCs.
        self.assertRaises(ReferenceError, str, p)

    def test_strange_subclass(self):
        class X(deque):
            def __iter__(self):
                return iter([])
        d1 = X([1,2,3])
        d2 = X([4,5,6])
        d1 == d2   # not clear if this is supposed to be True or False,
                   # but it used to give a SystemError

    @support.cpython_only
    def test_bug_31608(self):
        # The interpreter used to crash in specific cases where a deque
        # subclass returned a non-deque.
        class X(deque):
            pass
        d = X()
        def bad___new__(cls, *args, **kwargs):
            return [42]
        X.__new__ = bad___new__
        with self.assertRaises(TypeError):
            d * 42  # shouldn't crash
        with self.assertRaises(TypeError):
            d + deque([1, 2, 3])  # shouldn't crash


class SubclassWithKwargs(deque):
    def __init__(self, newarg=1):
        deque.__init__(self)

class TestSubclassWithKwargs(CPythonTestCase):
    def test_subclass_with_kwargs(self):
        # SF bug #1486663 -- this used to erroneously raise a TypeError
        SubclassWithKwargs(newarg=1)

class TestSequence(seq_tests.CommonTest):
    type2test = deque

    def test_getitem(self):
        # For now, bypass tests that require slicing
        pass

    def test_getslice(self):
        # For now, bypass tests that require slicing
        pass

    def test_subscript(self):
        # For now, bypass tests that require slicing
        pass

    def test_free_after_iterating(self):
        # For now, bypass tests that require slicing
        self.skipTest("Exhausted deque iterator doesn't free a deque")

#==============================================================================

libreftest = """
Example from the Library Reference:  Doc/lib/libcollections.tex

>>> from collections import deque
>>> d = deque('ghi')                 # make a new deque with three items
>>> for elem in d:                   # iterate over the deque's elements
...     print(elem.upper())
G
H
I
>>> d.append('j')                    # add a new entry to the right side
>>> d.appendleft('f')                # add a new entry to the left side
>>> d                                # show the representation of the deque
deque(['f', 'g', 'h', 'i', 'j'])
>>> d.pop()                          # return and remove the rightmost item
'j'
>>> d.popleft()                      # return and remove the leftmost item
'f'
>>> list(d)                          # list the contents of the deque
['g', 'h', 'i']
>>> d[0]                             # peek at leftmost item
'g'
>>> d[-1]                            # peek at rightmost item
'i'
>>> list(reversed(d))                # list the contents of a deque in reverse
['i', 'h', 'g']
>>> 'h' in d                         # search the deque
True
>>> d.extend('jkl')                  # add multiple elements at once
>>> d
deque(['g', 'h', 'i', 'j', 'k', 'l'])
>>> d.rotate(1)                      # right rotation
>>> d
deque(['l', 'g', 'h', 'i', 'j', 'k'])
>>> d.rotate(-1)                     # left rotation
>>> d
deque(['g', 'h', 'i', 'j', 'k', 'l'])
>>> deque(reversed(d))               # make a new deque in reverse order
deque(['l', 'k', 'j', 'i', 'h', 'g'])
>>> d.clear()                        # empty the deque
>>> d.pop()                          # cannot pop from an empty deque
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in -toplevel-
    d.pop()
IndexError: pop from an empty deque

>>> d.extendleft('abc')              # extendleft() reverses the input order
>>> d
deque(['c', 'b', 'a'])



>>> def delete_nth(d, n):
...     d.rotate(-n)
...     d.popleft()
...     d.rotate(n)
...
>>> d = deque('abcdef')
>>> delete_nth(d, 2)   # remove the entry at d[2]
>>> d
deque(['a', 'b', 'd', 'e', 'f'])



>>> def roundrobin(*iterables):
...     pending = deque(iter(i) for i in iterables)
...     while pending:
...         task = pending.popleft()
...         try:
...             yield next(task)
...         except StopIteration:
...             continue
...         pending.append(task)
...

>>> for value in roundrobin('abc', 'd', 'efgh'):
...     print(value)
...
a
d
e
b
f
c
g
h


>>> def maketree(iterable):
...     d = deque(iterable)
...     while len(d) > 1:
...         pair = [d.popleft(), d.popleft()]
...         d.append(pair)
...     return list(d)
...
>>> print(maketree('abcdefgh'))
[[[['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']]]]

"""


#==============================================================================

__test__ = {'libreftest' : libreftest}


if __name__ == "__main__":
    run_tests()
