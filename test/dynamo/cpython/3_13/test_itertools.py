# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_itertools.py

import torch
import torch._dynamo.test_case
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    slowTest,
)

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

import doctest
import unittest
import itertools
from test import support
from test.support import threading_helper, script_helper
from itertools import *
import weakref
from decimal import Decimal
from fractions import Fraction
import operator
import random
import copy
import pickle
from functools import reduce
import sys
import struct
import threading
import gc
import warnings

def pickle_deprecated(testfunc):
    """ Run the test three times.
    First, verify that a Deprecation Warning is raised.
    Second, run normally but with DeprecationWarnings temporarily disabled.
    Third, run with warnings promoted to errors.
    """
    def inner(self):
        with self.assertWarns(DeprecationWarning):
            testfunc(self)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            testfunc(self)
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=DeprecationWarning)
            with self.assertRaises((DeprecationWarning, AssertionError, SystemError)):
                testfunc(self)

    return inner

maxsize = support.MAX_Py_ssize_t
minsize = -maxsize-1

@torch._dynamo.disable
def choice(*args):
    return random.choice(*args)

@torch._dynamo.disable
def randrange(*args):
    return random.randrange(*args)

def lzip(*args):
    return list(zip(*args))

def onearg(x):
    'Test function of one argument'
    return 2*x

def errfunc(*args):
    'Test function that raises an error'
    raise ValueError

def gen3():
    'Non-restartable source sequence'
    for i in (0, 1, 2):
        yield i

def isEven(x):
    'Test predicate'
    return x%2==0

def isOdd(x):
    'Test predicate'
    return x%2==1

def tupleize(*args):
    return args

def irange(n):
    for i in range(n):
        yield i

class StopNow:
    'Class emulating an empty iterable.'
    def __iter__(self):
        return self
    def __next__(self):
        raise StopIteration

def take(n, seq):
    'Convenience function for partially consuming a long of infinite iterable'
    return list(islice(seq, n))

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def fact(n):
    'Factorial'
    return prod(range(1, n+1))

# root level methods for pickling ability
def _testR(r):
    return r[0]

def _testR2(r):
    return r[2]

def underten(x):
    return x<10

picklecopiers = [lambda s, proto=proto: pickle.loads(pickle.dumps(s, proto))
                 for proto in range(pickle.HIGHEST_PROTOCOL + 1)]

class TestBasicOps(__TestCase):

    def pickletest(self, protocol, it, stop=4, take=1, compare=None):
        """Test that an iterator is the same after pickling, also when part-consumed"""
        def expand(it, i=0):
            # Recursively expand iterables, within sensible bounds
            if i > 10:
                raise RuntimeError("infinite recursion encountered")
            if isinstance(it, str):
                return it
            try:
                l = list(islice(it, stop))
            except TypeError:
                return it # can't expand it
            return [expand(e, i+1) for e in l]

        # Test the initial copy against the original
        dump = pickle.dumps(it, protocol)
        i2 = pickle.loads(dump)
        self.assertEqual(type(it), type(i2))
        a, b = expand(it), expand(i2)
        self.assertEqual(a, b)
        if compare:
            c = expand(compare)
            self.assertEqual(a, c)

        # Take from the copy, and create another copy and compare them.
        i3 = pickle.loads(dump)
        took = 0
        try:
            for i in range(take):
                next(i3)
                took += 1
        except StopIteration:
            pass #in case there is less data than 'take'
        dump = pickle.dumps(i3, protocol)
        i4 = pickle.loads(dump)
        a, b = expand(i3), expand(i4)
        self.assertEqual(a, b)
        if compare:
            c = expand(compare[took:])
            self.assertEqual(a, c);

    @pickle_deprecated
    def test_accumulate(self):
        self.assertEqual(list(accumulate(range(10))),               # one positional arg
                          [0, 1, 3, 6, 10, 15, 21, 28, 36, 45])
        self.assertEqual(list(accumulate(iterable=range(10))),      # kw arg
                          [0, 1, 3, 6, 10, 15, 21, 28, 36, 45])
        for typ in int, complex, Decimal, Fraction:                 # multiple types
            self.assertEqual(
                list(accumulate(map(typ, range(10)))),
                list(map(typ, [0, 1, 3, 6, 10, 15, 21, 28, 36, 45])))
        self.assertEqual(list(accumulate('abc')), ['a', 'ab', 'abc'])   # works with non-numeric
        self.assertEqual(list(accumulate([])), [])                  # empty iterable
        self.assertEqual(list(accumulate([7])), [7])                # iterable of length one
        self.assertRaises(TypeError, accumulate, range(10), 5, 6)   # too many args
        self.assertRaises(TypeError, accumulate)                    # too few args
        self.assertRaises(TypeError, accumulate, x=range(10))       # unexpected kwd arg
        self.assertRaises(TypeError, list, accumulate([1, []]))     # args that don't add

        s = [2, 8, 9, 5, 7, 0, 3, 4, 1, 6]
        self.assertEqual(list(accumulate(s, min)),
                         [2, 2, 2, 2, 2, 0, 0, 0, 0, 0])
        self.assertEqual(list(accumulate(s, max)),
                         [2, 8, 9, 9, 9, 9, 9, 9, 9, 9])
        self.assertEqual(list(accumulate(s, operator.mul)),
                         [2, 16, 144, 720, 5040, 0, 0, 0, 0, 0])
        with self.assertRaises(TypeError):
            list(accumulate(s, chr))                                # unary-operation
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, accumulate(range(10)))           # test pickling
            self.pickletest(proto, accumulate(range(10), initial=7))
        self.assertEqual(list(accumulate([10, 5, 1], initial=None)), [10, 15, 16])
        self.assertEqual(list(accumulate([10, 5, 1], initial=100)), [100, 110, 115, 116])
        self.assertEqual(list(accumulate([], initial=100)), [100])
        with self.assertRaises(TypeError):
            list(accumulate([10, 20], 100))

    def test_batched(self):
        self.assertEqual(list(batched('ABCDEFG', 3)),
                             [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)])
        self.assertEqual(list(batched('ABCDEFG', 2)),
                             [('A', 'B'), ('C', 'D'), ('E', 'F'), ('G',)])
        self.assertEqual(list(batched('ABCDEFG', 1)),
                            [('A',), ('B',), ('C',), ('D',), ('E',), ('F',), ('G',)])
        self.assertEqual(list(batched('ABCDEF', 2, strict=True)),
                             [('A', 'B'), ('C', 'D'), ('E', 'F')])

        with self.assertRaises(ValueError):         # Incomplete batch when strict
            list(batched('ABCDEFG', 3, strict=True))
        with self.assertRaises(TypeError):          # Too few arguments
            list(batched('ABCDEFG'))
        with self.assertRaises(TypeError):
            list(batched('ABCDEFG', 3, None))       # Too many arguments
        with self.assertRaises(TypeError):
            list(batched(None, 3))                  # Non-iterable input
        with self.assertRaises(TypeError):
            list(batched('ABCDEFG', 'hello'))       # n is a string
        with self.assertRaises(ValueError):
            list(batched('ABCDEFG', 0))             # n is zero
        with self.assertRaises(ValueError):
            list(batched('ABCDEFG', -1))            # n is negative

        data = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for n in range(1, 6):
            for i in range(len(data)):
                s = data[:i]
                batches = list(batched(s, n))
                with self.subTest(s=s, n=n, batches=batches):
                    # Order is preserved and no data is lost
                    self.assertEqual(''.join(chain(*batches)), s)
                    # Each batch is an exact tuple
                    self.assertTrue(all(type(batch) is tuple for batch in batches))
                    # All but the last batch is of size n
                    if batches:
                        last_batch = batches.pop()
                        self.assertTrue(all(len(batch) == n for batch in batches))
                        self.assertTrue(len(last_batch) <= n)
                        batches.append(last_batch)

    def test_chain(self):

        def chain2(*iterables):
            'Pure python version in the docs'
            for it in iterables:
                for element in it:
                    yield element

        for c in (chain, chain2):
            self.assertEqual(list(c('abc', 'def')), list('abcdef'))
            self.assertEqual(list(c('abc')), list('abc'))
            self.assertEqual(list(c('')), [])
            self.assertEqual(take(4, c('abc', 'def')), list('abcd'))
            self.assertRaises(TypeError, list,c(2, 3))

    def test_chain_from_iterable(self):
        self.assertEqual(list(chain.from_iterable(['abc', 'def'])), list('abcdef'))
        self.assertEqual(list(chain.from_iterable(['abc'])), list('abc'))
        self.assertEqual(list(chain.from_iterable([''])), [])
        self.assertEqual(take(4, chain.from_iterable(['abc', 'def'])), list('abcd'))
        self.assertRaises(TypeError, list, chain.from_iterable([2, 3]))
        self.assertEqual(list(islice(chain.from_iterable(repeat(range(5))), 2)), [0, 1])

    @pickle_deprecated
    def test_chain_reducible(self):
        for oper in [copy.deepcopy] + picklecopiers:
            it = chain('abc', 'def')
            self.assertEqual(list(oper(it)), list('abcdef'))
            self.assertEqual(next(it), 'a')
            self.assertEqual(list(oper(it)), list('bcdef'))

            self.assertEqual(list(oper(chain(''))), [])
            self.assertEqual(take(4, oper(chain('abc', 'def'))), list('abcd'))
            self.assertRaises(TypeError, list, oper(chain(2, 3)))
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, chain('abc', 'def'), compare=list('abcdef'))

    @pickle_deprecated
    def test_chain_setstate(self):
        self.assertRaises(TypeError, chain().__setstate__, ())
        self.assertRaises(TypeError, chain().__setstate__, [])
        self.assertRaises(TypeError, chain().__setstate__, 0)
        self.assertRaises(TypeError, chain().__setstate__, ([],))
        self.assertRaises(TypeError, chain().__setstate__, (iter([]), []))
        it = chain()
        it.__setstate__((iter(['abc', 'def']),))
        self.assertEqual(list(it), ['a', 'b', 'c', 'd', 'e', 'f'])
        it = chain()
        it.__setstate__((iter(['abc', 'def']), iter(['ghi'])))
        self.assertEqual(list(it), ['ghi', 'a', 'b', 'c', 'd', 'e', 'f'])

    @pickle_deprecated
    def test_combinations(self):
        self.assertRaises(TypeError, combinations, 'abc')       # missing r argument
        self.assertRaises(TypeError, combinations, 'abc', 2, 1) # too many arguments
        self.assertRaises(TypeError, combinations, None)        # pool is not iterable
        self.assertRaises(ValueError, combinations, 'abc', -2)  # r is negative

        for op in [lambda a:a] + picklecopiers:
            self.assertEqual(list(op(combinations('abc', 32))), [])     # r > n

            self.assertEqual(list(op(combinations('ABCD', 2))),
                             [('A','B'), ('A','C'), ('A','D'), ('B','C'), ('B','D'), ('C','D')])
            testIntermediate = combinations('ABCD', 2)
            next(testIntermediate)
            self.assertEqual(list(op(testIntermediate)),
                             [('A','C'), ('A','D'), ('B','C'), ('B','D'), ('C','D')])

            self.assertEqual(list(op(combinations(range(4), 3))),
                             [(0,1,2), (0,1,3), (0,2,3), (1,2,3)])
            testIntermediate = combinations(range(4), 3)
            next(testIntermediate)
            self.assertEqual(list(op(testIntermediate)),
                             [(0,1,3), (0,2,3), (1,2,3)])

        def combinations1(iterable, r):
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            if r > n:
                return
            indices = list(range(r))
            yield tuple(pool[i] for i in indices)
            while 1:
                for i in reversed(range(r)):
                    if indices[i] != i + n - r:
                        break
                else:
                    return
                indices[i] += 1
                for j in range(i+1, r):
                    indices[j] = indices[j-1] + 1
                yield tuple(pool[i] for i in indices)

        def combinations2(iterable, r):
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            for indices in permutations(range(n), r):
                if sorted(indices) == list(indices):
                    yield tuple(pool[i] for i in indices)

        def combinations3(iterable, r):
            'Pure python version from cwr()'
            pool = tuple(iterable)
            n = len(pool)
            for indices in combinations_with_replacement(range(n), r):
                if len(set(indices)) == r:
                    yield tuple(pool[i] for i in indices)

        for n in range(7):
            values = [5*x-12 for x in range(n)]
            for r in range(n+2):
                result = list(combinations(values, r))
                self.assertEqual(len(result), 0 if r>n else fact(n) / fact(r) / fact(n-r)) # right number of combs
                self.assertEqual(len(result), len(set(result)))         # no repeats
                self.assertEqual(result, sorted(result))                # lexicographic order
                for c in result:
                    self.assertEqual(len(c), r)                         # r-length combinations
                    self.assertEqual(len(set(c)), r)                    # no duplicate elements
                    self.assertEqual(list(c), sorted(c))                # keep original ordering
                    self.assertTrue(all(e in values for e in c))           # elements taken from input iterable
                    self.assertEqual(list(c),
                                     [e for e in values if e in c])      # comb is a subsequence of the input iterable
                self.assertEqual(result, list(combinations1(values, r))) # matches first pure python version
                self.assertEqual(result, list(combinations2(values, r))) # matches second pure python version
                self.assertEqual(result, list(combinations3(values, r))) # matches second pure python version

                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.pickletest(proto, combinations(values, r))      # test pickling

    @support.bigaddrspacetest
    def test_combinations_overflow(self):
        with self.assertRaises((OverflowError, MemoryError)):
            combinations("AA", 2**29)

        # Test implementation detail:  tuple re-use
    @support.impl_detail("tuple reuse is specific to CPython")
    def test_combinations_tuple_reuse(self):
        self.assertEqual(len(set(map(id, combinations('abcde', 3)))), 1)
        self.assertNotEqual(len(set(map(id, list(combinations('abcde', 3))))), 1)

    @pickle_deprecated
    def test_combinations_with_replacement(self):
        cwr = combinations_with_replacement
        self.assertRaises(TypeError, cwr, 'abc')       # missing r argument
        self.assertRaises(TypeError, cwr, 'abc', 2, 1) # too many arguments
        self.assertRaises(TypeError, cwr, None)        # pool is not iterable
        self.assertRaises(ValueError, cwr, 'abc', -2)  # r is negative

        for op in [lambda a:a] + picklecopiers:
            self.assertEqual(list(op(cwr('ABC', 2))),
                             [('A','A'), ('A','B'), ('A','C'), ('B','B'), ('B','C'), ('C','C')])
            testIntermediate = cwr('ABC', 2)
            next(testIntermediate)
            self.assertEqual(list(op(testIntermediate)),
                             [('A','B'), ('A','C'), ('B','B'), ('B','C'), ('C','C')])


        def cwr1(iterable, r):
            'Pure python version shown in the docs'
            # number items returned:  (n+r-1)! / r! / (n-1)! when n>0
            pool = tuple(iterable)
            n = len(pool)
            if not n and r:
                return
            indices = [0] * r
            yield tuple(pool[i] for i in indices)
            while 1:
                for i in reversed(range(r)):
                    if indices[i] != n - 1:
                        break
                else:
                    return
                indices[i:] = [indices[i] + 1] * (r - i)
                yield tuple(pool[i] for i in indices)

        def cwr2(iterable, r):
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            for indices in product(range(n), repeat=r):
                if sorted(indices) == list(indices):
                    yield tuple(pool[i] for i in indices)

        def numcombs(n, r):
            if not n:
                return 0 if r else 1
            return fact(n+r-1) / fact(r)/ fact(n-1)

        for n in range(7):
            values = [5*x-12 for x in range(n)]
            for r in range(n+2):
                result = list(cwr(values, r))

                self.assertEqual(len(result), numcombs(n, r))           # right number of combs
                self.assertEqual(len(result), len(set(result)))         # no repeats
                self.assertEqual(result, sorted(result))                # lexicographic order

                regular_combs = list(combinations(values, r))           # compare to combs without replacement
                if n == 0 or r <= 1:
                    self.assertEqual(result, regular_combs)            # cases that should be identical
                else:
                    self.assertTrue(set(result) >= set(regular_combs))     # rest should be supersets of regular combs

                for c in result:
                    self.assertEqual(len(c), r)                         # r-length combinations
                    noruns = [k for k,v in groupby(c)]                  # combo without consecutive repeats
                    self.assertEqual(len(noruns), len(set(noruns)))     # no repeats other than consecutive
                    self.assertEqual(list(c), sorted(c))                # keep original ordering
                    self.assertTrue(all(e in values for e in c))           # elements taken from input iterable
                    self.assertEqual(noruns,
                                     [e for e in values if e in c])     # comb is a subsequence of the input iterable
                self.assertEqual(result, list(cwr1(values, r)))         # matches first pure python version
                self.assertEqual(result, list(cwr2(values, r)))         # matches second pure python version

                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.pickletest(proto, cwr(values,r))               # test pickling

    @support.bigaddrspacetest
    def test_combinations_with_replacement_overflow(self):
        with self.assertRaises((OverflowError, MemoryError)):
            combinations_with_replacement("AA", 2**30)

        # Test implementation detail:  tuple re-use
    @support.impl_detail("tuple reuse is specific to CPython")
    def test_combinations_with_replacement_tuple_reuse(self):
        cwr = combinations_with_replacement
        self.assertEqual(len(set(map(id, cwr('abcde', 3)))), 1)
        self.assertNotEqual(len(set(map(id, list(cwr('abcde', 3))))), 1)

    def test_permutations(self):
        self.assertEqual(list(permutations('abc', 32)), [])     # r > n
        self.assertEqual(list(permutations(range(3), 2)),
                                           [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)])

        def permutations1(iterable, r=None):
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            r = n if r is None else r
            if r > n:
                return
            indices = list(range(n))
            cycles = list(range(n-r+1, n+1))[::-1]
            yield tuple(pool[i] for i in indices[:r])
            while n:
                for i in reversed(range(r)):
                    cycles[i] -= 1
                    if cycles[i] == 0:
                        indices[i:] = indices[i+1:] + indices[i:i+1]
                        cycles[i] = n - i
                    else:
                        j = cycles[i]
                        indices[i], indices[-j] = indices[-j], indices[i]
                        yield tuple(pool[i] for i in indices[:r])
                        break
                else:
                    return

        def permutations2(iterable, r=None):
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            r = n if r is None else r
            for indices in product(range(n), repeat=r):
                if len(set(indices)) == r:
                    yield tuple(pool[i] for i in indices)

        for n in range(5):
            values = [5*x-12 for x in range(n)]
            for r in range(n+2):
                result = list(permutations(values, r))
                self.assertEqual(len(result), 0 if r>n else fact(n) / fact(n-r))      # right number of perms
                self.assertEqual(len(result), len(set(result)))         # no repeats
                self.assertEqual(result, sorted(result))                # lexicographic order
                for p in result:
                    self.assertEqual(len(p), r)                         # r-length permutations
                    self.assertEqual(len(set(p)), r)                    # no duplicate elements
                    self.assertTrue(all(e in values for e in p))           # elements taken from input iterable
                self.assertEqual(result, list(permutations1(values, r))) # matches first pure python version
                self.assertEqual(result, list(permutations2(values, r))) # matches second pure python version
                if r == n:
                    self.assertEqual(result, list(permutations(values, None))) # test r as None
                    self.assertEqual(result, list(permutations(values)))       # test default r

    @support.bigaddrspacetest
    def test_permutations_overflow(self):
        with self.assertRaises((OverflowError, MemoryError)):
            permutations("A", 2**30)

    @support.impl_detail("tuple reuse is specific to CPython")
    def test_permutations_tuple_reuse(self):
        self.assertEqual(len(set(map(id, permutations('abcde', 3)))), 1)
        self.assertNotEqual(len(set(map(id, list(permutations('abcde', 3))))), 1)

    def test_combinatorics(self):
        # Test relationships between product(), permutations(),
        # combinations() and combinations_with_replacement().

        for n in range(6):
            s = 'ABCDEFG'[:n]
            for r in range(8):
                prod = list(product(s, repeat=r))
                cwr = list(combinations_with_replacement(s, r))
                perm = list(permutations(s, r))
                comb = list(combinations(s, r))

                # Check size
                self.assertEqual(len(prod), n**r)
                self.assertEqual(len(cwr), (fact(n+r-1) / fact(r)/ fact(n-1)) if n else (not r))
                self.assertEqual(len(perm), 0 if r>n else fact(n) / fact(n-r))
                self.assertEqual(len(comb), 0 if r>n else fact(n) / fact(r) / fact(n-r))

                # Check lexicographic order without repeated tuples
                self.assertEqual(prod, sorted(set(prod)))
                self.assertEqual(cwr, sorted(set(cwr)))
                self.assertEqual(perm, sorted(set(perm)))
                self.assertEqual(comb, sorted(set(comb)))

                # Check interrelationships
                self.assertEqual(cwr, [t for t in prod if sorted(t)==list(t)]) # cwr: prods which are sorted
                self.assertEqual(perm, [t for t in prod if len(set(t))==r])    # perm: prods with no dups
                self.assertEqual(comb, [t for t in perm if sorted(t)==list(t)]) # comb: perms that are sorted
                self.assertEqual(comb, [t for t in cwr if len(set(t))==r])      # comb: cwrs without dups
                self.assertEqual(comb, list(filter(set(cwr).__contains__, perm)))     # comb: perm that is a cwr
                self.assertEqual(comb, list(filter(set(perm).__contains__, cwr)))     # comb: cwr that is a perm
                self.assertEqual(comb, sorted(set(cwr) & set(perm)))            # comb: both a cwr and a perm

    @pickle_deprecated
    def test_compress(self):
        self.assertEqual(list(compress(data='ABCDEF', selectors=[1,0,1,0,1,1])), list('ACEF'))
        self.assertEqual(list(compress('ABCDEF', [1,0,1,0,1,1])), list('ACEF'))
        self.assertEqual(list(compress('ABCDEF', [0,0,0,0,0,0])), list(''))
        self.assertEqual(list(compress('ABCDEF', [1,1,1,1,1,1])), list('ABCDEF'))
        self.assertEqual(list(compress('ABCDEF', [1,0,1])), list('AC'))
        self.assertEqual(list(compress('ABC', [0,1,1,1,1,1])), list('BC'))
        n = 10000
        data = chain.from_iterable(repeat(range(6), n))
        selectors = chain.from_iterable(repeat((0, 1)))
        self.assertEqual(list(compress(data, selectors)), [1,3,5] * n)
        self.assertRaises(TypeError, compress, None, range(6))      # 1st arg not iterable
        self.assertRaises(TypeError, compress, range(6), None)      # 2nd arg not iterable
        self.assertRaises(TypeError, compress, range(6))            # too few args
        self.assertRaises(TypeError, compress, range(6), None)      # too many args

        # check copy, deepcopy, pickle
        for op in [lambda a:copy.copy(a), lambda a:copy.deepcopy(a)] + picklecopiers:
            for data, selectors, result1, result2 in [
                ('ABCDEF', [1,0,1,0,1,1], 'ACEF', 'CEF'),
                ('ABCDEF', [0,0,0,0,0,0], '', ''),
                ('ABCDEF', [1,1,1,1,1,1], 'ABCDEF', 'BCDEF'),
                ('ABCDEF', [1,0,1], 'AC', 'C'),
                ('ABC', [0,1,1,1,1,1], 'BC', 'C'),
                ]:

                self.assertEqual(list(op(compress(data=data, selectors=selectors))), list(result1))
                self.assertEqual(list(op(compress(data, selectors))), list(result1))
                testIntermediate = compress(data, selectors)
                if result1:
                    next(testIntermediate)
                    self.assertEqual(list(op(testIntermediate)), list(result2))

    @pickle_deprecated
    def test_count(self):
        self.assertEqual(lzip('abc',count()), [('a', 0), ('b', 1), ('c', 2)])
        self.assertEqual(lzip('abc',count(3)), [('a', 3), ('b', 4), ('c', 5)])
        self.assertEqual(take(2, lzip('abc',count(3))), [('a', 3), ('b', 4)])
        self.assertEqual(take(2, zip('abc',count(-1))), [('a', -1), ('b', 0)])
        self.assertEqual(take(2, zip('abc',count(-3))), [('a', -3), ('b', -2)])
        self.assertRaises(TypeError, count, 2, 3, 4)
        self.assertRaises(TypeError, count, 'a')
        self.assertEqual(take(3, count(maxsize)),
                        [maxsize, maxsize + 1, maxsize + 2])
        self.assertEqual(take(10, count(maxsize-5)),
                         list(range(maxsize-5, maxsize+5)))
        self.assertEqual(take(10, count(-maxsize-5)),
                         list(range(-maxsize-5, -maxsize+5)))
        self.assertEqual(take(3, count(3.25)), [3.25, 4.25, 5.25])
        self.assertEqual(take(3, count(3.25-4j)), [3.25-4j, 4.25-4j, 5.25-4j])
        self.assertEqual(take(3, count(Decimal('1.1'))),
                         [Decimal('1.1'), Decimal('2.1'), Decimal('3.1')])
        self.assertEqual(take(3, count(Fraction(2, 3))),
                         [Fraction(2, 3), Fraction(5, 3), Fraction(8, 3)])
        BIGINT = 1<<1000
        self.assertEqual(take(3, count(BIGINT)), [BIGINT, BIGINT+1, BIGINT+2])
        c = count(3)
        self.assertEqual(repr(c), 'count(3)')
        next(c)
        self.assertEqual(repr(c), 'count(4)')
        c = count(-9)
        self.assertEqual(repr(c), 'count(-9)')
        next(c)
        self.assertEqual(next(c), -8)
        self.assertEqual(repr(count(10.25)), 'count(10.25)')
        self.assertEqual(repr(count(10.0)), 'count(10.0)')

        self.assertEqual(repr(count(maxsize)), f'count({maxsize})')
        c = count(maxsize - 1)
        self.assertEqual(repr(c), f'count({maxsize - 1})')
        next(c)  # c is now at masize
        self.assertEqual(repr(c), f'count({maxsize})')
        next(c)
        self.assertEqual(repr(c), f'count({maxsize + 1})')

        self.assertEqual(type(next(count(10.0))), float)
        for i in (-sys.maxsize-5, -sys.maxsize+5 ,-10, -1, 0, 10, sys.maxsize-5, sys.maxsize+5):
            # Test repr
            r1 = repr(count(i))
            r2 = 'count(%r)'.__mod__(i)
            self.assertEqual(r1, r2)

        # check copy, deepcopy, pickle
        for value in -3, 3, maxsize-5, maxsize+5:
            c = count(value)
            self.assertEqual(next(copy.copy(c)), value)
            self.assertEqual(next(copy.deepcopy(c)), value)
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                self.pickletest(proto, count(value))

        #check proper internal error handling for large "step' sizes
        count(1, maxsize+5); sys.exc_info()

    @pickle_deprecated
    def test_count_with_step(self):
        self.assertEqual(lzip('abc',count(2,3)), [('a', 2), ('b', 5), ('c', 8)])
        self.assertEqual(lzip('abc',count(start=2,step=3)),
                         [('a', 2), ('b', 5), ('c', 8)])
        self.assertEqual(lzip('abc',count(step=-1)),
                         [('a', 0), ('b', -1), ('c', -2)])
        self.assertRaises(TypeError, count, 'a', 'b')
        self.assertEqual(lzip('abc',count(2,0)), [('a', 2), ('b', 2), ('c', 2)])
        self.assertEqual(lzip('abc',count(2,1)), [('a', 2), ('b', 3), ('c', 4)])
        self.assertEqual(lzip('abc',count(2,3)), [('a', 2), ('b', 5), ('c', 8)])
        self.assertEqual(take(20, count(maxsize-15, 3)), take(20, range(maxsize-15, maxsize+100, 3)))
        self.assertEqual(take(20, count(-maxsize-15, 3)), take(20, range(-maxsize-15,-maxsize+100, 3)))
        self.assertEqual(take(3, count(10, maxsize+5)),
                         list(range(10, 10+3*(maxsize+5), maxsize+5)))
        self.assertEqual(take(3, count(maxsize, 2)),
                         [maxsize, maxsize + 2, maxsize + 4])
        self.assertEqual(take(3, count(maxsize, maxsize)),
                         [maxsize, 2 * maxsize, 3 * maxsize])
        self.assertEqual(take(3, count(-maxsize, maxsize)),
                        [-maxsize, 0, maxsize])
        self.assertEqual(take(3, count(2, 1.25)), [2, 3.25, 4.5])
        self.assertEqual(take(3, count(2, 3.25-4j)), [2, 5.25-4j, 8.5-8j])
        self.assertEqual(take(3, count(Decimal('1.1'), Decimal('.1'))),
                         [Decimal('1.1'), Decimal('1.2'), Decimal('1.3')])
        self.assertEqual(take(3, count(Fraction(2,3), Fraction(1,7))),
                         [Fraction(2,3), Fraction(17,21), Fraction(20,21)])
        BIGINT = 1<<1000
        self.assertEqual(take(3, count(step=BIGINT)), [0, BIGINT, 2*BIGINT])
        self.assertEqual(repr(take(3, count(10, 2.5))), repr([10, 12.5, 15.0]))
        c = count(3, 5)
        self.assertEqual(repr(c), 'count(3, 5)')
        next(c)
        self.assertEqual(repr(c), 'count(8, 5)')
        c = count(-9, 0)
        self.assertEqual(repr(c), 'count(-9, 0)')
        next(c)
        self.assertEqual(repr(c), 'count(-9, 0)')
        c = count(-9, -3)
        self.assertEqual(repr(c), 'count(-9, -3)')
        next(c)
        self.assertEqual(repr(c), 'count(-12, -3)')
        self.assertEqual(repr(c), 'count(-12, -3)')
        self.assertEqual(repr(count(10.5, 1.25)), 'count(10.5, 1.25)')
        self.assertEqual(repr(count(10.5, 1)), 'count(10.5)')           # suppress step=1 when it's an int
        self.assertEqual(repr(count(10.5, 1.00)), 'count(10.5, 1.0)')   # do show float values lilke 1.0
        self.assertEqual(repr(count(10, 1.00)), 'count(10, 1.0)')
        c = count(10, 1.0)
        self.assertEqual(type(next(c)), int)
        self.assertEqual(type(next(c)), float)
        for i in (-sys.maxsize-5, -sys.maxsize+5 ,-10, -1, 0, 10, sys.maxsize-5, sys.maxsize+5):
            for j in  (-sys.maxsize-5, -sys.maxsize+5 ,-10, -1, 0, 1, 10, sys.maxsize-5, sys.maxsize+5):
                # Test repr
                r1 = repr(count(i, j))
                if j == 1:
                    r2 = ('count(%r)' % i)
                else:
                    r2 = ('count(%r, %r)' % (i, j))
                self.assertEqual(r1, r2)
                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.pickletest(proto, count(i, j))

        c = count(maxsize -2, 2)
        self.assertEqual(repr(c), f'count({maxsize - 2}, 2)')
        next(c)  # c is now at masize
        self.assertEqual(repr(c), f'count({maxsize}, 2)')
        next(c)
        self.assertEqual(repr(c), f'count({maxsize + 2}, 2)')

        c = count(maxsize + 1, -1)
        self.assertEqual(repr(c), f'count({maxsize + 1}, -1)')
        next(c)  # c is now at masize
        self.assertEqual(repr(c), f'count({maxsize}, -1)')
        next(c)
        self.assertEqual(repr(c), f'count({maxsize - 1}, -1)')

    @threading_helper.requires_working_threading()
    def test_count_threading(self, step=1):
        # this test verifies multithreading consistency, which is
        # mostly for testing builds without GIL, but nice to test anyway
        count_to = 10_000
        num_threads = 10
        c = count(step=step)
        def counting_thread():
            for i in range(count_to):
                next(c)
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=counting_thread)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        self.assertEqual(next(c), count_to * num_threads * step)

    def test_count_with_step_threading(self):
        self.test_count_threading(step=5)

    def test_cycle(self):
        self.assertEqual(take(10, cycle('abc')), list('abcabcabca'))
        self.assertEqual(list(cycle('')), [])
        # self.assertRaises(TypeError, cycle)
        self.assertRaises(TypeError, cycle, 5)
        self.assertEqual(list(islice(cycle(gen3()),10)), [0,1,2,0,1,2,0,1,2,0])

    @pickle_deprecated
    def test_cycle_copy_pickle(self):
        # check copy, deepcopy, pickle
        c = cycle('abc')
        self.assertEqual(next(c), 'a')
        #simple copy currently not supported, because __reduce__ returns
        #an internal iterator
        #self.assertEqual(take(10, copy.copy(c)), list('bcabcabcab'))
        self.assertEqual(take(10, copy.deepcopy(c)), list('bcabcabcab'))
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertEqual(take(10, pickle.loads(pickle.dumps(c, proto))),
                             list('bcabcabcab'))
            next(c)
            self.assertEqual(take(10, pickle.loads(pickle.dumps(c, proto))),
                             list('cabcabcabc'))
            next(c)
            next(c)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, cycle('abc'))

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # test with partial consumed input iterable
            it = iter('abcde')
            c = cycle(it)
            _ = [next(c) for i in range(2)]      # consume 2 of 5 inputs
            p = pickle.dumps(c, proto)
            d = pickle.loads(p)                  # rebuild the cycle object
            self.assertEqual(take(20, d), list('cdeabcdeabcdeabcdeab'))

            # test with completely consumed input iterable
            it = iter('abcde')
            c = cycle(it)
            _ = [next(c) for i in range(7)]      # consume 7 of 5 inputs
            p = pickle.dumps(c, proto)
            d = pickle.loads(p)                  # rebuild the cycle object
            self.assertEqual(take(20, d), list('cdeabcdeabcdeabcdeab'))

    @pickle_deprecated
    def test_cycle_unpickle_compat(self):
        testcases = [
            b'citertools\ncycle\n(c__builtin__\niter\n((lI1\naI2\naI3\natRI1\nbtR((lI1\naI0\ntb.',
            b'citertools\ncycle\n(c__builtin__\niter\n(](K\x01K\x02K\x03etRK\x01btR(]K\x01aK\x00tb.',
            b'\x80\x02citertools\ncycle\nc__builtin__\niter\n](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01aK\x00\x86b.',
            b'\x80\x03citertools\ncycle\ncbuiltins\niter\n](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01aK\x00\x86b.',
            b'\x80\x04\x95=\x00\x00\x00\x00\x00\x00\x00\x8c\titertools\x8c\x05cycle\x93\x8c\x08builtins\x8c\x04iter\x93](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01aK\x00\x86b.',

            b'citertools\ncycle\n(c__builtin__\niter\n((lp0\nI1\naI2\naI3\natRI1\nbtR(g0\nI1\ntb.',
            b'citertools\ncycle\n(c__builtin__\niter\n(]q\x00(K\x01K\x02K\x03etRK\x01btR(h\x00K\x01tb.',
            b'\x80\x02citertools\ncycle\nc__builtin__\niter\n]q\x00(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00K\x01\x86b.',
            b'\x80\x03citertools\ncycle\ncbuiltins\niter\n]q\x00(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00K\x01\x86b.',
            b'\x80\x04\x95<\x00\x00\x00\x00\x00\x00\x00\x8c\titertools\x8c\x05cycle\x93\x8c\x08builtins\x8c\x04iter\x93]\x94(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00K\x01\x86b.',

            b'citertools\ncycle\n(c__builtin__\niter\n((lI1\naI2\naI3\natRI1\nbtR((lI1\naI00\ntb.',
            b'citertools\ncycle\n(c__builtin__\niter\n(](K\x01K\x02K\x03etRK\x01btR(]K\x01aI00\ntb.',
            b'\x80\x02citertools\ncycle\nc__builtin__\niter\n](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01a\x89\x86b.',
            b'\x80\x03citertools\ncycle\ncbuiltins\niter\n](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01a\x89\x86b.',
            b'\x80\x04\x95<\x00\x00\x00\x00\x00\x00\x00\x8c\titertools\x8c\x05cycle\x93\x8c\x08builtins\x8c\x04iter\x93](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01a\x89\x86b.',

            b'citertools\ncycle\n(c__builtin__\niter\n((lp0\nI1\naI2\naI3\natRI1\nbtR(g0\nI01\ntb.',
            b'citertools\ncycle\n(c__builtin__\niter\n(]q\x00(K\x01K\x02K\x03etRK\x01btR(h\x00I01\ntb.',
            b'\x80\x02citertools\ncycle\nc__builtin__\niter\n]q\x00(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00\x88\x86b.',
            b'\x80\x03citertools\ncycle\ncbuiltins\niter\n]q\x00(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00\x88\x86b.',
            b'\x80\x04\x95;\x00\x00\x00\x00\x00\x00\x00\x8c\titertools\x8c\x05cycle\x93\x8c\x08builtins\x8c\x04iter\x93]\x94(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00\x88\x86b.',
        ]
        assert len(testcases) == 20
        for t in testcases:
            it = pickle.loads(t)
            self.assertEqual(take(10, it), [2, 3, 1, 2, 3, 1, 2, 3, 1, 2])

    @pickle_deprecated
    def test_cycle_setstate(self):
        # Verify both modes for restoring state

        # Mode 0 is efficient.  It uses an incompletely consumed input
        # iterator to build a cycle object and then passes in state with
        # a list of previously consumed values.  There is no data
        # overlap between the two.
        c = cycle('defg')
        c.__setstate__((list('abc'), 0))
        self.assertEqual(take(20, c), list('defgabcdefgabcdefgab'))

        # Mode 1 is inefficient.  It starts with a cycle object built
        # from an iterator over the remaining elements in a partial
        # cycle and then passes in state with all of the previously
        # seen values (this overlaps values included in the iterator).
        c = cycle('defg')
        c.__setstate__((list('abcdefg'), 1))
        self.assertEqual(take(20, c), list('defgabcdefgabcdefgab'))

        # The first argument to setstate needs to be a tuple
        with self.assertRaises(TypeError):
            cycle('defg').__setstate__([list('abcdefg'), 0])

        # The first argument in the setstate tuple must be a list
        with self.assertRaises(TypeError):
            c = cycle('defg')
            c.__setstate__((tuple('defg'), 0))
        take(20, c)

        # The second argument in the setstate tuple must be an int
        with self.assertRaises(TypeError):
            cycle('defg').__setstate__((list('abcdefg'), 'x'))

        self.assertRaises(TypeError, cycle('').__setstate__, ())
        self.assertRaises(TypeError, cycle('').__setstate__, ([],))

    @pickle_deprecated
    def test_groupby(self):
        # Check whether it accepts arguments correctly
        self.assertEqual([], list(groupby([])))
        self.assertEqual([], list(groupby([], key=id)))
        self.assertRaises(TypeError, list, groupby('abc', []))
        self.assertRaises(TypeError, groupby, None)
        self.assertRaises(TypeError, groupby, 'abc', lambda x:x, 10)

        # Check normal input
        s = [(0, 10, 20), (0, 11,21), (0,12,21), (1,13,21), (1,14,22),
             (2,15,22), (3,16,23), (3,17,23)]
        dup = []
        for k, g in groupby(s, lambda r:r[0]):
            for elem in g:
                self.assertEqual(k, elem[0])
                dup.append(elem)
        self.assertEqual(s, dup)

        # Check normal pickled
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            dup = []
            for k, g in pickle.loads(pickle.dumps(groupby(s, _testR), proto)):
                for elem in g:
                    self.assertEqual(k, elem[0])
                    dup.append(elem)
            self.assertEqual(s, dup)

        # Check nested case
        dup = []
        for k, g in groupby(s, _testR):
            for ik, ig in groupby(g, _testR2):
                for elem in ig:
                    self.assertEqual(k, elem[0])
                    self.assertEqual(ik, elem[2])
                    dup.append(elem)
        self.assertEqual(s, dup)

        # Check nested and pickled
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            dup = []
            for k, g in pickle.loads(pickle.dumps(groupby(s, _testR), proto)):
                for ik, ig in pickle.loads(pickle.dumps(groupby(g, _testR2), proto)):
                    for elem in ig:
                        self.assertEqual(k, elem[0])
                        self.assertEqual(ik, elem[2])
                        dup.append(elem)
            self.assertEqual(s, dup)


        # Check case where inner iterator is not used
        keys = [k for k, g in groupby(s, _testR)]
        expectedkeys = set([r[0] for r in s])
        self.assertEqual(set(keys), expectedkeys)
        self.assertEqual(len(keys), len(expectedkeys))

        # Check case where inner iterator is used after advancing the groupby
        # iterator
        s = list(zip('AABBBAAAA', range(9)))
        it = groupby(s, _testR)
        _, g1 = next(it)
        _, g2 = next(it)
        _, g3 = next(it)
        self.assertEqual(list(g1), [])
        self.assertEqual(list(g2), [])
        self.assertEqual(next(g3), ('A', 5))
        list(it)  # exhaust the groupby iterator
        self.assertEqual(list(g3), [])

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            it = groupby(s, _testR)
            _, g = next(it)
            next(it)
            next(it)
            self.assertEqual(list(pickle.loads(pickle.dumps(g, proto))), [])

        # Exercise pipes and filters style
        s = 'abracadabra'
        # sort s | uniq
        r = [k for k, g in groupby(sorted(s))]
        self.assertEqual(r, ['a', 'b', 'c', 'd', 'r'])
        # sort s | uniq -d
        r = [k for k, g in groupby(sorted(s)) if list(islice(g,1,2))]
        self.assertEqual(r, ['a', 'b', 'r'])
        # sort s | uniq -c
        r = [(len(list(g)), k) for k, g in groupby(sorted(s))]
        self.assertEqual(r, [(5, 'a'), (2, 'b'), (1, 'c'), (1, 'd'), (2, 'r')])
        # sort s | uniq -c | sort -rn | head -3
        r = sorted([(len(list(g)) , k) for k, g in groupby(sorted(s))], reverse=True)[:3]
        self.assertEqual(r, [(5, 'a'), (2, 'r'), (2, 'b')])

        # iter.__next__ failure
        class ExpectedError(Exception):
            pass
        def delayed_raise(n=0):
            for i in range(n):
                yield 'yo'
            raise ExpectedError
        def gulp(iterable, keyp=None, func=list):
            return [func(g) for k, g in groupby(iterable, keyp)]

        # iter.__next__ failure on outer object
        self.assertRaises(ExpectedError, gulp, delayed_raise(0))
        # iter.__next__ failure on inner object
        self.assertRaises(ExpectedError, gulp, delayed_raise(1))

        # __eq__ failure
        class DummyCmp:
            def __eq__(self, dst):
                raise ExpectedError
        s = [DummyCmp(), DummyCmp(), None]

        # __eq__ failure on outer object
        self.assertRaises(ExpectedError, gulp, s, func=id)
        # __eq__ failure on inner object
        self.assertRaises(ExpectedError, gulp, s)

        # keyfunc failure
        def keyfunc(obj):
            if keyfunc.skip > 0:
                keyfunc.skip -= 1
                return obj
            else:
                raise ExpectedError

        # keyfunc failure on outer object
        keyfunc.skip = 0
        self.assertRaises(ExpectedError, gulp, [None], keyfunc)
        keyfunc.skip = 1
        self.assertRaises(ExpectedError, gulp, [None, None], keyfunc)

    def test_filter(self):
        self.assertEqual(list(filter(isEven, range(6))), [0,2,4])
        self.assertEqual(list(filter(None, [0,1,0,2,0])), [1,2])
        self.assertEqual(list(filter(bool, [0,1,0,2,0])), [1,2])
        self.assertEqual(take(4, filter(isEven, count())), [0,2,4,6])
        # these tests raise dynamo exceptions, not TypeError
        # self.assertRaises(TypeError, filter)
        # self.assertRaises(TypeError, filter, lambda x:x)
        # self.assertRaises(TypeError, filter, lambda x:x, range(6), 7)
        # self.assertRaises(TypeError, filter, isEven, 3)
        # dynamo raises Unsupported in this case
        # self.assertRaises(TypeError, next, filter(range(6), range(6)))

        # check copy, deepcopy, pickle
        # ans = [0,2,4]

        # c = filter(isEven, range(6))
        # self.assertEqual(list(copy.copy(c)), ans)
        # c = filter(isEven, range(6))
        # self.assertEqual(list(copy.deepcopy(c)), ans)
        # for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        #     c = filter(isEven, range(6))
        #     self.assertEqual(list(pickle.loads(pickle.dumps(c, proto))), ans)
        #     next(c)
        #     self.assertEqual(list(pickle.loads(pickle.dumps(c, proto))), ans[1:])
        # for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        #     c = filter(isEven, range(6))
        #     self.pickletest(proto, c)

    def test_filterfalse(self):
        self.assertEqual(list(filterfalse(isEven, range(6))), [1,3,5])
        self.assertEqual(list(filterfalse(None, [0,1,0,2,0])), [0,0,0])
        self.assertEqual(list(filterfalse(bool, [0,1,0,2,0])), [0,0,0])
        self.assertEqual(take(4, filterfalse(isEven, count())), [1,3,5,7])
        self.assertRaises(TypeError, filterfalse)
        self.assertRaises(TypeError, filterfalse, lambda x:x)
        self.assertRaises(TypeError, filterfalse, lambda x:x, range(6), 7)
        self.assertRaises(TypeError, filterfalse, isEven, 3)
        with torch._dynamo.error_on_graph_break(False):
            self.assertRaises(TypeError, next, filterfalse(range(6), range(6)))
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                self.pickletest(proto, filterfalse(isEven, range(6)))

    def test_zip(self):
        # XXX This is rather silly now that builtin zip() calls zip()...
        ans = [(x,y) for x, y in zip('abc',count())]
        self.assertEqual(ans, [('a', 0), ('b', 1), ('c', 2)])
        self.assertEqual(list(zip('abc', range(6))), lzip('abc', range(6)))
        self.assertEqual(list(zip('abcdef', range(3))), lzip('abcdef', range(3)))
        self.assertEqual(take(3,zip('abcdef', count())), lzip('abcdef', range(3)))
        self.assertEqual(list(zip('abcdef')), lzip('abcdef'))
        self.assertEqual(list(zip()), lzip())
        # self.assertRaises(TypeError, zip, 3)
        # self.assertRaises(TypeError, zip, range(3), 3)
        self.assertEqual([tuple(list(pair)) for pair in zip('abc', 'def')],
                         lzip('abc', 'def'))
        self.assertEqual([pair for pair in zip('abc', 'def')],
                         lzip('abc', 'def'))

    @support.impl_detail("tuple reuse is specific to CPython")
    @pickle_deprecated
    def test_zip_tuple_reuse(self):
        ids = list(map(id, zip('abc', 'def')))
        self.assertEqual(min(ids), max(ids))
        ids = list(map(id, list(zip('abc', 'def'))))
        self.assertEqual(len(dict.fromkeys(ids)), len(ids))

        # check copy, deepcopy, pickle
        ans = [(x,y) for x, y in copy.copy(zip('abc',count()))]
        self.assertEqual(ans, [('a', 0), ('b', 1), ('c', 2)])

        ans = [(x,y) for x, y in copy.deepcopy(zip('abc',count()))]
        self.assertEqual(ans, [('a', 0), ('b', 1), ('c', 2)])

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            ans = [(x,y) for x, y in pickle.loads(pickle.dumps(zip('abc',count()), proto))]
            self.assertEqual(ans, [('a', 0), ('b', 1), ('c', 2)])

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            testIntermediate = zip('abc',count())
            next(testIntermediate)
            ans = [(x,y) for x, y in pickle.loads(pickle.dumps(testIntermediate, proto))]
            self.assertEqual(ans, [('b', 1), ('c', 2)])

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, zip('abc', count()))

    def test_ziplongest(self):
        for args in [
                ['abc', range(6)],
                [range(6), 'abc'],
                [range(1000), range(2000,2100), range(3000,3050)],
                [range(1000), range(0), range(3000,3050), range(1200), range(1500)],
                [range(1000), range(0), range(3000,3050), range(1200), range(1500), range(0)],
            ]:
            target = [tuple([arg[i] if i < len(arg) else None for arg in args])
                      for i in range(max(map(len, args)))]
            self.assertEqual(list(zip_longest(*args)), target)
            self.assertEqual(list(zip_longest(*args, **{})), target)
            target = [tuple((e is None and 'X' or e) for e in t) for t in target]   # Replace None fills with 'X'
            self.assertEqual(list(zip_longest(*args, **dict(fillvalue='X'))), target)

        self.assertEqual(take(3,zip_longest('abcdef', count())), list(zip('abcdef', range(3)))) # take 3 from infinite input

        self.assertEqual(list(zip_longest()), list(zip()))
        self.assertEqual(list(zip_longest([])), list(zip([])))
        self.assertEqual(list(zip_longest('abcdef')), list(zip('abcdef')))

        self.assertEqual(list(zip_longest('abc', 'defg', **{})),
                         list(zip(list('abc')+[None], 'defg'))) # empty keyword dict
        # self.assertRaises(TypeError, zip_longest, 3)
        # self.assertRaises(TypeError, zip_longest, range(3), 3)

        # for stmt in [
        #     "zip_longest('abc', fv=1)",
        #     "zip_longest('abc', fillvalue=1, bogus_keyword=None)",
        # ]:
        #     try:
        #         eval(stmt, globals(), locals())
        #     except TypeError:
        #         pass
        #     else:
        #         self.fail('Did not raise Type in:  ' + stmt)

        self.assertEqual([tuple(list(pair)) for pair in zip_longest('abc', 'def')],
                         list(zip('abc', 'def')))
        self.assertEqual([pair for pair in zip_longest('abc', 'def')],
                         list(zip('abc', 'def')))

    @support.impl_detail("tuple reuse is specific to CPython")
    def test_zip_longest_tuple_reuse(self):
        ids = list(map(id, zip_longest('abc', 'def')))
        self.assertEqual(min(ids), max(ids))
        ids = list(map(id, list(zip_longest('abc', 'def'))))
        self.assertEqual(len(dict.fromkeys(ids)), len(ids))

    @pickle_deprecated
    def test_zip_longest_pickling(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, zip_longest("abc", "def"))
            self.pickletest(proto, zip_longest("abc", "defgh"))
            self.pickletest(proto, zip_longest("abc", "defgh", fillvalue=1))
            self.pickletest(proto, zip_longest("", "defgh"))

    def test_zip_longest_bad_iterable(self):
        exception = TypeError()

        class BadIterable:
            def __iter__(self):
                raise exception

        with self.assertRaises(TypeError) as cm:
            zip_longest(BadIterable())

        self.assertIs(cm.exception, exception)

    def test_bug_7244(self):

        class Repeater:
            # this class is similar to itertools.repeat
            def __init__(self, o, t, e):
                self.o = o
                self.t = int(t)
                self.e = e
            def __iter__(self): # its iterator is itself
                return self
            def __next__(self):
                if self.t > 0:
                    self.t -= 1
                    return self.o
                else:
                    raise self.e

        # Formerly this code in would fail in debug mode
        # with Undetected Error and Stop Iteration
        r1 = Repeater(1, 3, StopIteration)
        r2 = Repeater(2, 4, StopIteration)
        def run(r1, r2):
            result = []
            for i, j in zip_longest(r1, r2, fillvalue=0):
                with support.captured_output('stdout'):
                    print((i, j))
                result.append((i, j))
            return result
        self.assertEqual(run(r1, r2), [(1,2), (1,2), (1,2), (0,2)])

        # Formerly, the RuntimeError would be lost
        # and StopIteration would stop as expected
        r1 = Repeater(1, 3, RuntimeError)
        r2 = Repeater(2, 4, StopIteration)
        it = zip_longest(r1, r2, fillvalue=0)
        self.assertEqual(next(it), (1, 2))
        self.assertEqual(next(it), (1, 2))
        self.assertEqual(next(it), (1, 2))
        self.assertRaises(RuntimeError, next, it)

    def test_pairwise(self):
        self.assertEqual(list(pairwise('')), [])
        self.assertEqual(list(pairwise('a')), [])
        self.assertEqual(list(pairwise('ab')),
                              [('a', 'b')]),
        self.assertEqual(list(pairwise('abcde')),
                              [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])
        self.assertEqual(list(pairwise(range(10_000))),
                         list(zip(range(10_000), range(1, 10_000))))

        with self.assertRaises(TypeError):
            pairwise()                                      # too few arguments
        with self.assertRaises(TypeError):
            pairwise('abc', 10)                             # too many arguments
        with self.assertRaises(TypeError):
            pairwise(iterable='abc')                        # keyword arguments
        with self.assertRaises(TypeError):
            pairwise(None)                                  # non-iterable argument

    def test_pairwise_reenter(self):
        def check(reenter_at, expected):
            class I:
                count = 0
                def __iter__(self):
                    return self
                def __next__(self):
                    self.count +=1
                    if self.count in reenter_at:
                        return next(it)
                    return [self.count]  # new object

            it = pairwise(I())
            for item in expected:
                self.assertEqual(next(it), item)

        check({1}, [
            (([2], [3]), [4]),
            ([4], [5]),
        ])
        check({2}, [
            ([1], ([1], [3])),
            (([1], [3]), [4]),
            ([4], [5]),
        ])
        check({3}, [
            ([1], [2]),
            ([2], ([2], [4])),
            (([2], [4]), [5]),
            ([5], [6]),
        ])
        check({1, 2}, [
            ((([3], [4]), [5]), [6]),
            ([6], [7]),
        ])
        check({1, 3}, [
            (([2], ([2], [4])), [5]),
            ([5], [6]),
        ])
        check({1, 4}, [
            (([2], [3]), (([2], [3]), [5])),
            ((([2], [3]), [5]), [6]),
            ([6], [7]),
        ])
        check({2, 3}, [
            ([1], ([1], ([1], [4]))),
            (([1], ([1], [4])), [5]),
            ([5], [6]),
        ])

    def test_pairwise_reenter2(self):
        def check(maxcount, expected):
            class I:
                count = 0
                def __iter__(self):
                    return self
                def __next__(self):
                    if self.count >= maxcount:
                        raise StopIteration
                    self.count +=1
                    if self.count == 1:
                        return next(it, None)
                    return [self.count]  # new object

            it = pairwise(I())
            self.assertEqual(list(it), expected)

        check(1, [])
        check(2, [])
        check(3, [])
        check(4, [(([2], [3]), [4])])

    def test_product(self):
        for args, result in [
            ([], [()]),                     # zero iterables
            (['ab'], [('a',), ('b',)]),     # one iterable
            ([range(2), range(3)], [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]),     # two iterables
            ([range(0), range(2), range(3)], []),           # first iterable with zero length
            ([range(2), range(0), range(3)], []),           # middle iterable with zero length
            ([range(2), range(3), range(0)], []),           # last iterable with zero length
            ]:
            self.assertEqual(list(product(*args)), result)
            for r in range(4):
                self.assertEqual(list(product(*(args*r))),
                                 list(product(*args, **dict(repeat=r))))
        self.assertEqual(len(list(product(*[range(7)]*6))), 7**6)

        def product1(*args, **kwds):
            pools = list(map(tuple, args)) * kwds.get('repeat', 1)
            n = len(pools)
            if n == 0:
                yield ()
                return
            if any(len(pool) == 0 for pool in pools):
                return
            indices = [0] * n
            yield tuple(pool[i] for pool, i in zip(pools, indices))
            while 1:
                for i in reversed(range(n)):  # right to left
                    if indices[i] == len(pools[i]) - 1:
                        continue
                    indices[i] += 1
                    for j in range(i+1, n):
                        indices[j] = 0
                    yield tuple(pool[i] for pool, i in zip(pools, indices))
                    break
                else:
                    return

        def product2(*iterables, repeat=1):
            'Pure python version used in docs'
            if repeat < 0:
                raise ValueError('repeat argument cannot be negative')
            pools = [tuple(pool) for pool in iterables] * repeat

            result = [[]]
            for pool in pools:
                result = [x+[y] for x in result for y in pool]

            for prod in result:
                yield tuple(prod)

        argtypes = ['', 'abc', '', range(0), range(4), dict(a=1, b=2, c=3),
                    set('abcdefg'), range(11), tuple(range(13))]
        for i in range(100):
            with torch._dynamo.error_on_graph_break(False):
                args = [choice(argtypes) for j in range(randrange(5))]
            expected_len = prod(map(len, args))
            self.assertEqual(len(list(product(*args))), expected_len)
            self.assertEqual(list(product(*args)), list(product1(*args)))
            self.assertEqual(list(product(*args)), list(product2(*args)))
            args = map(iter, args)
            self.assertEqual(len(list(product(*args))), expected_len)

    @support.bigaddrspacetest
    def test_product_overflow(self):
        with self.assertRaises((OverflowError, MemoryError)):
            product(*(['ab']*2**5), repeat=2**25)

    @support.impl_detail("tuple reuse is specific to CPython")
    def test_product_tuple_reuse(self):
        self.assertEqual(len(set(map(id, product('abc', 'def')))), 1)
        self.assertNotEqual(len(set(map(id, list(product('abc', 'def'))))), 1)

    @pickle_deprecated
    def test_product_pickling(self):
        # check copy, deepcopy, pickle
        for args, result in [
            ([], [()]),                     # zero iterables
            (['ab'], [('a',), ('b',)]),     # one iterable
            ([range(2), range(3)], [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]),     # two iterables
            ([range(0), range(2), range(3)], []),           # first iterable with zero length
            ([range(2), range(0), range(3)], []),           # middle iterable with zero length
            ([range(2), range(3), range(0)], []),           # last iterable with zero length
            ]:
            self.assertEqual(list(copy.copy(product(*args))), result)
            self.assertEqual(list(copy.deepcopy(product(*args))), result)
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                self.pickletest(proto, product(*args))

    @pickle_deprecated
    def test_product_issue_25021(self):
        # test that indices are properly clamped to the length of the tuples
        p = product((1, 2),(3,))
        p.__setstate__((0, 0x1000))  # will access tuple element 1 if not clamped
        self.assertEqual(next(p), (2, 3))
        # test that empty tuple in the list will result in an immediate StopIteration
        p = product((1, 2), (), (3,))
        p.__setstate__((0, 0, 0x1000))  # will access tuple element 1 if not clamped
        self.assertRaises(StopIteration, next, p)

    @pickle_deprecated
    def test_repeat(self):
        self.assertEqual(list(repeat(object='a', times=3)), ['a', 'a', 'a'])
        self.assertEqual(lzip(range(3),repeat('a')),
                         [(0, 'a'), (1, 'a'), (2, 'a')])
        self.assertEqual(list(repeat('a', 3)), ['a', 'a', 'a'])
        self.assertEqual(take(3, repeat('a')), ['a', 'a', 'a'])
        self.assertEqual(list(repeat('a', 0)), [])
        self.assertEqual(list(repeat('a', -3)), [])
        self.assertRaises(TypeError, repeat)
        self.assertRaises(TypeError, repeat, None, 3, 4)
        self.assertRaises(TypeError, repeat, None, 'a')
        r = repeat(1+0j)
        self.assertEqual(repr(r), 'repeat((1+0j))')
        r = repeat(1+0j, 5)
        self.assertEqual(repr(r), 'repeat((1+0j), 5)')
        list(r)
        self.assertEqual(repr(r), 'repeat((1+0j), 0)')

        # check copy, deepcopy, pickle
        c = repeat(object='a', times=10)
        self.assertEqual(next(c), 'a')
        self.assertEqual(take(2, copy.copy(c)), list('a' * 2))
        self.assertEqual(take(2, copy.deepcopy(c)), list('a' * 2))
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, repeat(object='a', times=10))

    def test_repeat_with_negative_times(self):
        self.assertEqual(repr(repeat('a', -1)), "repeat('a', 0)")
        self.assertEqual(repr(repeat('a', -2)), "repeat('a', 0)")
        self.assertEqual(repr(repeat('a', times=-1)), "repeat('a', 0)")
        self.assertEqual(repr(repeat('a', times=-2)), "repeat('a', 0)")

    @pickle_deprecated
    def test_map(self):
        self.assertEqual(list(map(operator.pow, range(3), range(1,7))),
                         [0**1, 1**2, 2**3])
        self.assertEqual(list(map(tupleize, 'abc', range(5))),
                         [('a',0),('b',1),('c',2)])
        self.assertEqual(list(map(tupleize, 'abc', count())),
                         [('a',0),('b',1),('c',2)])
        self.assertEqual(take(2,map(tupleize, 'abc', count())),
                         [('a',0),('b',1)])
        self.assertEqual(list(map(operator.pow, [])), [])
        self.assertRaises(TypeError, map)
        self.assertRaises(TypeError, list, map(None, range(3), range(3)))
        self.assertRaises(TypeError, map, operator.neg)
        self.assertRaises(TypeError, next, map(10, range(5)))
        self.assertRaises(ValueError, next, map(errfunc, [4], [5]))
        self.assertRaises(TypeError, next, map(onearg, [4], [5]))

        # check copy, deepcopy, pickle
        ans = [('a',0),('b',1),('c',2)]

        c = map(tupleize, 'abc', count())
        self.assertEqual(list(copy.copy(c)), ans)

        c = map(tupleize, 'abc', count())
        self.assertEqual(list(copy.deepcopy(c)), ans)

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            c = map(tupleize, 'abc', count())
            self.pickletest(proto, c)

    @pickle_deprecated
    def test_starmap(self):
        self.assertEqual(list(starmap(operator.pow, zip(range(3), range(1,7)))),
                         [0**1, 1**2, 2**3])
        self.assertEqual(take(3, starmap(operator.pow, zip(count(), count(1)))),
                         [0**1, 1**2, 2**3])
        self.assertEqual(list(starmap(operator.pow, [])), [])
        self.assertEqual(list(starmap(operator.pow, [iter([4,5])])), [4**5])
        self.assertRaises(TypeError, list, starmap(operator.pow, [None]))
        self.assertRaises(TypeError, starmap)
        self.assertRaises(TypeError, starmap, operator.pow, [(4,5)], 'extra')
        self.assertRaises(TypeError, next, starmap(10, [(4,5)]))
        self.assertRaises(ValueError, next, starmap(errfunc, [(4,5)]))
        self.assertRaises(TypeError, next, starmap(onearg, [(4,5)]))

        # check copy, deepcopy, pickle
        ans = [0**1, 1**2, 2**3]

        c = starmap(operator.pow, zip(range(3), range(1,7)))
        self.assertEqual(list(copy.copy(c)), ans)

        c = starmap(operator.pow, zip(range(3), range(1,7)))
        self.assertEqual(list(copy.deepcopy(c)), ans)

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            c = starmap(operator.pow, zip(range(3), range(1,7)))
            self.pickletest(proto, c)

    @pickle_deprecated
    def test_islice(self):
        for args in [          # islice(args) should agree with range(args)
                (10, 20, 3),
                (10, 3, 20),
                (10, 20),
                (10, 10),
                (10, 3),
                (20,)
                ]:
            self.assertEqual(list(islice(range(100), *args)),
                             list(range(*args)))

        for args, tgtargs in [  # Stop when seqn is exhausted
                ((10, 110, 3), ((10, 100, 3))),
                ((10, 110), ((10, 100))),
                ((110,), (100,))
                ]:
            self.assertEqual(list(islice(range(100), *args)),
                             list(range(*tgtargs)))

        # Test stop=None
        self.assertEqual(list(islice(range(10), None)), list(range(10)))
        self.assertEqual(list(islice(range(10), None, None)), list(range(10)))
        self.assertEqual(list(islice(range(10), None, None, None)), list(range(10)))
        self.assertEqual(list(islice(range(10), 2, None)), list(range(2, 10)))
        self.assertEqual(list(islice(range(10), 1, None, 2)), list(range(1, 10, 2)))

        # Test number of items consumed     SF #1171417
        it = iter(range(10))
        self.assertEqual(list(islice(it, 3)), list(range(3)))
        self.assertEqual(list(it), list(range(3, 10)))

        it = iter(range(10))
        self.assertEqual(list(islice(it, 3, 3)), [])
        self.assertEqual(list(it), list(range(3, 10)))

        # Test invalid arguments
        ra = range(10)
        self.assertRaises(TypeError, islice, ra)
        self.assertRaises(TypeError, islice, ra, 1, 2, 3, 4)
        self.assertRaises(ValueError, islice, ra, -5, 10, 1)
        self.assertRaises(ValueError, islice, ra, 1, -5, -1)
        self.assertRaises(ValueError, islice, ra, 1, 10, -1)
        self.assertRaises(ValueError, islice, ra, 1, 10, 0)
        self.assertRaises(ValueError, islice, ra, 'a')
        self.assertRaises(ValueError, islice, ra, 'a', 1)
        self.assertRaises(ValueError, islice, ra, 1, 'a')
        self.assertRaises(ValueError, islice, ra, 'a', 1, 1)
        self.assertRaises(ValueError, islice, ra, 1, 'a', 1)
        self.assertEqual(len(list(islice(count(), 1, 10, maxsize))), 1)

        # Issue #10323:  Less islice in a predictable state
        c = count()
        self.assertEqual(list(islice(c, 1, 3, 50)), [1])
        self.assertEqual(next(c), 3)

        # check copy, deepcopy, pickle
        for args in [          # islice(args) should agree with range(args)
                (10, 20, 3),
                (10, 3, 20),
                (10, 20),
                (10, 3),
                (20,)
                ]:
            self.assertEqual(list(copy.copy(islice(range(100), *args))),
                             list(range(*args)))
            self.assertEqual(list(copy.deepcopy(islice(range(100), *args))),
                             list(range(*args)))
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                self.pickletest(proto, islice(range(100), *args))

        # Issue #21321: check source iterator is not referenced
        # from islice() after the latter has been exhausted
        it = (x for x in (1, 2))
        wr = weakref.ref(it)
        it = islice(it, 1)
        self.assertIsNotNone(wr())
        list(it) # exhaust the iterator
        support.gc_collect()
        self.assertIsNone(wr())

        # Issue #30537: islice can accept integer-like objects as
        # arguments
        class IntLike(object):
            def __init__(self, val):
                self.val = val
            def __index__(self):
                return self.val
        self.assertEqual(list(islice(range(100), IntLike(10))), list(range(10)))
        self.assertEqual(list(islice(range(100), IntLike(10), IntLike(50))),
                         list(range(10, 50)))
        self.assertEqual(list(islice(range(100), IntLike(10), IntLike(50), IntLike(5))),
                         list(range(10,50,5)))

    @pickle_deprecated
    def test_takewhile(self):
        data = [1, 3, 5, 20, 2, 4, 6, 8]
        self.assertEqual(list(takewhile(underten, data)), [1, 3, 5])
        self.assertEqual(list(takewhile(underten, [])), [])
        self.assertRaises(TypeError, takewhile)
        self.assertRaises(TypeError, takewhile, operator.pow)
        self.assertRaises(TypeError, takewhile, operator.pow, [(4,5)], 'extra')
        self.assertRaises(TypeError, next, takewhile(10, [(4,5)]))
        self.assertRaises(ValueError, next, takewhile(errfunc, [(4,5)]))
        t = takewhile(bool, [1, 1, 1, 0, 0, 0])
        self.assertEqual(list(t), [1, 1, 1])
        self.assertRaises(StopIteration, next, t)

        # check copy, deepcopy, pickle
        self.assertEqual(list(copy.copy(takewhile(underten, data))), [1, 3, 5])
        self.assertEqual(list(copy.deepcopy(takewhile(underten, data))),
                        [1, 3, 5])
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, takewhile(underten, data))

    @pickle_deprecated
    def test_dropwhile(self):
        data = [1, 3, 5, 20, 2, 4, 6, 8]
        self.assertEqual(list(dropwhile(underten, data)), [20, 2, 4, 6, 8])
        self.assertEqual(list(dropwhile(underten, [])), [])
        self.assertRaises(TypeError, dropwhile)
        self.assertRaises(TypeError, dropwhile, operator.pow)
        self.assertRaises(TypeError, dropwhile, operator.pow, [(4,5)], 'extra')
        self.assertRaises(TypeError, next, dropwhile(10, [(4,5)]))
        self.assertRaises(ValueError, next, dropwhile(errfunc, [(4,5)]))

        # check copy, deepcopy, pickle
        self.assertEqual(list(copy.copy(dropwhile(underten, data))), [20, 2, 4, 6, 8])
        self.assertEqual(list(copy.deepcopy(dropwhile(underten, data))),
                        [20, 2, 4, 6, 8])
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, dropwhile(underten, data))

    @pickle_deprecated
    def test_tee(self):
        n = 200

        a, b = tee([])        # test empty iterator
        self.assertEqual(list(a), [])
        self.assertEqual(list(b), [])

        a, b = tee(irange(n)) # test 100% interleaved
        self.assertEqual(lzip(a,b), lzip(range(n), range(n)))

        a, b = tee(irange(n)) # test 0% interleaved
        self.assertEqual(list(a), list(range(n)))
        self.assertEqual(list(b), list(range(n)))

        a, b = tee(irange(n)) # test dealloc of leading iterator
        for i in range(100):
            self.assertEqual(next(a), i)
        del a
        self.assertEqual(list(b), list(range(n)))

        a, b = tee(irange(n)) # test dealloc of trailing iterator
        for i in range(100):
            self.assertEqual(next(a), i)
        del b
        self.assertEqual(list(a), list(range(100, n)))

        for j in range(5):   # test randomly interleaved
            order = [0]*n + [1]*n
            random.shuffle(order)
            lists = ([], [])
            its = tee(irange(n))
            for i in order:
                value = next(its[i])
                lists[i].append(value)
            self.assertEqual(lists[0], list(range(n)))
            self.assertEqual(lists[1], list(range(n)))

        # test argument format checking
        self.assertRaises(TypeError, tee)
        self.assertRaises(TypeError, tee, 3)
        self.assertRaises(TypeError, tee, [1,2], 'x')
        self.assertRaises(TypeError, tee, [1,2], 3, 'x')

        # tee object should be instantiable
        a, b = tee('abc')
        c = type(a)('def')
        self.assertEqual(list(c), list('def'))

        # test long-lagged and multi-way split
        a, b, c = tee(range(2000), 3)
        for i in range(100):
            self.assertEqual(next(a), i)
        self.assertEqual(list(b), list(range(2000)))
        self.assertEqual([next(c), next(c)], list(range(2)))
        self.assertEqual(list(a), list(range(100,2000)))
        self.assertEqual(list(c), list(range(2,2000)))

        # test values of n
        self.assertRaises(TypeError, tee, 'abc', 'invalid')
        self.assertRaises(ValueError, tee, [], -1)
        for n in range(5):
            result = tee('abc', n)
            self.assertEqual(type(result), tuple)
            self.assertEqual(len(result), n)
            self.assertEqual([list(x) for x in result], [list('abc')]*n)

        # tee objects are independent (see bug gh-123884)
        a, b = tee('abc')
        c, d = tee(a)
        e, f = tee(c)
        self.assertTrue(len({a, b, c, d, e, f}) == 6)

        # test tee_new
        t1, t2 = tee('abc')
        tnew = type(t1)
        self.assertRaises(TypeError, tnew)
        self.assertRaises(TypeError, tnew, 10)
        t3 = tnew(t1)
        self.assertTrue(list(t1) == list(t2) == list(t3) == list('abc'))

        # test that tee objects are weak referenceable
        a, b = tee(range(10))
        p = weakref.proxy(a)
        self.assertEqual(getattr(p, '__class__'), type(b))
        del a
        support.gc_collect()  # For PyPy or other GCs.
        self.assertRaises(ReferenceError, getattr, p, '__class__')

        ans = list('abc')
        long_ans = list(range(10000))

        # check copy
        a, b = tee('abc')
        self.assertEqual(list(copy.copy(a)), ans)
        self.assertEqual(list(copy.copy(b)), ans)
        a, b = tee(list(range(10000)))
        self.assertEqual(list(copy.copy(a)), long_ans)
        self.assertEqual(list(copy.copy(b)), long_ans)

        # check partially consumed copy
        a, b = tee('abc')
        take(2, a)
        take(1, b)
        self.assertEqual(list(copy.copy(a)), ans[2:])
        self.assertEqual(list(copy.copy(b)), ans[1:])
        self.assertEqual(list(a), ans[2:])
        self.assertEqual(list(b), ans[1:])
        a, b = tee(range(10000))
        take(100, a)
        take(60, b)
        self.assertEqual(list(copy.copy(a)), long_ans[100:])
        self.assertEqual(list(copy.copy(b)), long_ans[60:])
        self.assertEqual(list(a), long_ans[100:])
        self.assertEqual(list(b), long_ans[60:])

        # check deepcopy
        a, b = tee('abc')
        self.assertEqual(list(copy.deepcopy(a)), ans)
        self.assertEqual(list(copy.deepcopy(b)), ans)
        self.assertEqual(list(a), ans)
        self.assertEqual(list(b), ans)
        a, b = tee(range(10000))
        self.assertEqual(list(copy.deepcopy(a)), long_ans)
        self.assertEqual(list(copy.deepcopy(b)), long_ans)
        self.assertEqual(list(a), long_ans)
        self.assertEqual(list(b), long_ans)

        # check partially consumed deepcopy
        a, b = tee('abc')
        take(2, a)
        take(1, b)
        self.assertEqual(list(copy.deepcopy(a)), ans[2:])
        self.assertEqual(list(copy.deepcopy(b)), ans[1:])
        self.assertEqual(list(a), ans[2:])
        self.assertEqual(list(b), ans[1:])
        a, b = tee(range(10000))
        take(100, a)
        take(60, b)
        self.assertEqual(list(copy.deepcopy(a)), long_ans[100:])
        self.assertEqual(list(copy.deepcopy(b)), long_ans[60:])
        self.assertEqual(list(a), long_ans[100:])
        self.assertEqual(list(b), long_ans[60:])

        # check pickle
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, iter(tee('abc')))
            a, b = tee('abc')
            self.pickletest(proto, a, compare=ans)
            self.pickletest(proto, b, compare=ans)

    def test_tee_dealloc_segfault(self):
        # gh-115874: segfaults when accessing module state in tp_dealloc.
        script = (
            "import typing, copyreg, itertools; "
            "copyreg.buggy_tee = itertools.tee(())"
        )
        script_helper.assert_python_ok("-c", script)

    # Issue 13454: Crash when deleting backward iterator from tee()
    @skipIfTorchDynamo("infinite loop in torch dynamo")
    def test_tee_del_backward(self):
        forward, backward = tee(repeat(None, 20000000))
        try:
            any(forward)  # exhaust the iterator
            del backward
        except:
            del forward, backward
            raise

    def test_tee_reenter(self):
        class I:
            first = True
            def __iter__(self):
                return self
            def __next__(self):
                first = self.first
                self.first = False
                if first:
                    return next(b)

        a, b = tee(I())
        with self.assertRaisesRegex(RuntimeError, "tee"):
            next(a)

    @threading_helper.requires_working_threading()
    def test_tee_concurrent(self):
        start = threading.Event()
        finish = threading.Event()
        class I:
            def __iter__(self):
                return self
            def __next__(self):
                start.set()
                finish.wait()

        a, b = tee(I())
        thread = threading.Thread(target=next, args=[a])
        thread.start()
        try:
            start.wait()
            with self.assertRaisesRegex(RuntimeError, "tee"):
                next(b)
        finally:
            finish.set()
            thread.join()

    def test_StopIteration(self):
        self.assertRaises(StopIteration, next, zip())

        for f in (chain, cycle, zip, groupby):
            self.assertRaises(StopIteration, next, f([]))
            self.assertRaises(StopIteration, next, f(StopNow()))

        self.assertRaises(StopIteration, next, islice([], None))
        self.assertRaises(StopIteration, next, islice(StopNow(), None))

        p, q = tee([])
        self.assertRaises(StopIteration, next, p)
        self.assertRaises(StopIteration, next, q)
        p, q = tee(StopNow())
        self.assertRaises(StopIteration, next, p)
        self.assertRaises(StopIteration, next, q)

        self.assertRaises(StopIteration, next, repeat(None, 0))

        for f in (filter, filterfalse, map, takewhile, dropwhile, starmap):
            self.assertRaises(StopIteration, next, f(lambda x:x, []))
            self.assertRaises(StopIteration, next, f(lambda x:x, StopNow()))

    @support.cpython_only
    def test_combinations_result_gc(self):
        # bpo-42536: combinations's tuple-reuse speed trick breaks the GC's
        # assumptions about what can be untracked. Make sure we re-track result
        # tuples whenever we reuse them.
        it = combinations([None, []], 1)
        next(it)
        gc.collect()
        # That GC collection probably untracked the recycled internal result
        # tuple, which has the value (None,). Make sure it's re-tracked when
        # it's mutated and returned from __next__:
        self.assertTrue(gc.is_tracked(next(it)))

    @support.cpython_only
    def test_combinations_with_replacement_result_gc(self):
        # Ditto for combinations_with_replacement.
        it = combinations_with_replacement([None, []], 1)
        next(it)
        gc.collect()
        self.assertTrue(gc.is_tracked(next(it)))

    @support.cpython_only
    def test_permutations_result_gc(self):
        # Ditto for permutations.
        it = permutations([None, []], 1)
        next(it)
        gc.collect()
        self.assertTrue(gc.is_tracked(next(it)))

    @support.cpython_only
    def test_product_result_gc(self):
        # Ditto for product.
        it = product([None, []])
        next(it)
        gc.collect()
        self.assertTrue(gc.is_tracked(next(it)))

    @support.cpython_only
    def test_zip_longest_result_gc(self):
        # Ditto for zip_longest.
        it = zip_longest([[]])
        gc.collect()
        self.assertTrue(gc.is_tracked(next(it)))

    @support.cpython_only
    def test_pairwise_result_gc(self):
        # Ditto for pairwise.
        it = pairwise([None, None])
        gc.collect()
        self.assertTrue(gc.is_tracked(next(it)))

    @support.cpython_only
    def test_immutable_types(self):
        from itertools import _grouper, _tee, _tee_dataobject
        dataset = (
            accumulate,
            batched,
            chain,
            combinations,
            combinations_with_replacement,
            compress,
            count,
            cycle,
            dropwhile,
            filterfalse,
            groupby,
            _grouper,
            islice,
            pairwise,
            permutations,
            product,
            repeat,
            starmap,
            takewhile,
            _tee,
            _tee_dataobject,
            zip_longest,
        )
        for tp in dataset:
            with self.subTest(tp=tp):
                with self.assertRaisesRegex(TypeError, "immutable"):
                    tp.foobar = 1


class TestExamples(__TestCase):

    def test_accumulate(self):
        self.assertEqual(list(accumulate([1,2,3,4,5])), [1, 3, 6, 10, 15])

    @pickle_deprecated
    def test_accumulate_reducible(self):
        # check copy, deepcopy, pickle
        data = [1, 2, 3, 4, 5]
        accumulated = [1, 3, 6, 10, 15]

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            it = accumulate(data)
            self.assertEqual(list(pickle.loads(pickle.dumps(it, proto))), accumulated[:])
            self.assertEqual(next(it), 1)
            self.assertEqual(list(pickle.loads(pickle.dumps(it, proto))), accumulated[1:])
        it = accumulate(data)
        self.assertEqual(next(it), 1)
        self.assertEqual(list(copy.deepcopy(it)), accumulated[1:])
        self.assertEqual(list(copy.copy(it)), accumulated[1:])

    @pickle_deprecated
    def test_accumulate_reducible_none(self):
        # Issue #25718: total is None
        it = accumulate([None, None, None], operator.is_)
        self.assertEqual(next(it), None)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            it_copy = pickle.loads(pickle.dumps(it, proto))
            self.assertEqual(list(it_copy), [True, False])
        self.assertEqual(list(copy.deepcopy(it)), [True, False])
        self.assertEqual(list(copy.copy(it)), [True, False])

    def test_chain(self):
        self.assertEqual(''.join(chain('ABC', 'DEF')), 'ABCDEF')

    def test_chain_from_iterable(self):
        self.assertEqual(''.join(chain.from_iterable(['ABC', 'DEF'])), 'ABCDEF')

    def test_combinations(self):
        self.assertEqual(list(combinations('ABCD', 2)),
                         [('A','B'), ('A','C'), ('A','D'), ('B','C'), ('B','D'), ('C','D')])
        self.assertEqual(list(combinations(range(4), 3)),
                         [(0,1,2), (0,1,3), (0,2,3), (1,2,3)])

    def test_combinations_with_replacement(self):
        self.assertEqual(list(combinations_with_replacement('ABC', 2)),
                         [('A','A'), ('A','B'), ('A','C'), ('B','B'), ('B','C'), ('C','C')])

    def test_compress(self):
        self.assertEqual(list(compress('ABCDEF', [1,0,1,0,1,1])), list('ACEF'))

    def test_count(self):
        self.assertEqual(list(islice(count(10), 5)), [10, 11, 12, 13, 14])

    def test_cycle(self):
        self.assertEqual(list(islice(cycle('ABCD'), 12)), list('ABCDABCDABCD'))

    def test_dropwhile(self):
        self.assertEqual(list(dropwhile(lambda x: x<5, [1,4,6,4,1])), [6,4,1])

    def test_groupby(self):
        self.assertEqual([k for k, g in groupby('AAAABBBCCDAABBB')],
                         list('ABCDAB'))
        self.assertEqual([(list(g)) for k, g in groupby('AAAABBBCCD')],
                         [list('AAAA'), list('BBB'), list('CC'), list('D')])

    def test_filter(self):
        self.assertEqual(list(filter(lambda x: x%2, range(10))), [1,3,5,7,9])

    def test_filterfalse(self):
        self.assertEqual(list(filterfalse(lambda x: x%2, range(10))), [0,2,4,6,8])

    def test_map(self):
        self.assertEqual(list(map(pow, (2,3,10), (5,2,3))), [32, 9, 1000])

    def test_islice(self):
        self.assertEqual(list(islice('ABCDEFG', 2)), list('AB'))
        self.assertEqual(list(islice('ABCDEFG', 2, 4)), list('CD'))
        self.assertEqual(list(islice('ABCDEFG', 2, None)), list('CDEFG'))
        self.assertEqual(list(islice('ABCDEFG', 0, None, 2)), list('ACEG'))

    def test_zip(self):
        self.assertEqual(list(zip('ABCD', 'xy')), [('A', 'x'), ('B', 'y')])

    def test_zip_longest(self):
        self.assertEqual(list(zip_longest('ABCD', 'xy', fillvalue='-')),
                         [('A', 'x'), ('B', 'y'), ('C', '-'), ('D', '-')])

    def test_permutations(self):
        self.assertEqual(list(permutations('ABCD', 2)),
                         list(map(tuple, 'AB AC AD BA BC BD CA CB CD DA DB DC'.split())))
        self.assertEqual(list(permutations(range(3))),
                         [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)])

    def test_product(self):
        self.assertEqual(list(product('ABCD', 'xy')),
                         list(map(tuple, 'Ax Ay Bx By Cx Cy Dx Dy'.split())))
        self.assertEqual(list(product(range(2), repeat=3)),
                        [(0,0,0), (0,0,1), (0,1,0), (0,1,1),
                         (1,0,0), (1,0,1), (1,1,0), (1,1,1)])

    def test_repeat(self):
        self.assertEqual(list(repeat(10, 3)), [10, 10, 10])

    def test_stapmap(self):
        self.assertEqual(list(starmap(pow, [(2,5), (3,2), (10,3)])),
                         [32, 9, 1000])

    def test_takewhile(self):
        self.assertEqual(list(takewhile(lambda x: x<5, [1,4,6,4,1])), [1,4])


class TestPurePythonRoughEquivalents(__TestCase):

    def test_batched_recipe(self):
        def batched_recipe(iterable, n):
            "Batch data into tuples of length n. The last batch may be shorter."
            # batched('ABCDEFG', 3) --> ABC DEF G
            if n < 1:
                raise ValueError('n must be at least one')
            it = iter(iterable)
            while batch := tuple(islice(it, n)):
                yield batch

        for iterable, n in product(
                ['', 'a', 'ab', 'abc', 'abcd', 'abcde', 'abcdef', 'abcdefg', None],
                [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None]):
            with self.subTest(iterable=iterable, n=n):
                try:
                    e1, r1 = None, list(batched(iterable, n))
                except Exception as e:
                    e1, r1 = type(e), None
                try:
                    e2, r2 = None, list(batched_recipe(iterable, n))
                except Exception as e:
                    e2, r2 = type(e), None
                self.assertEqual(r1, r2)
                self.assertEqual(e1, e2)

    @staticmethod
    def islice(iterable, *args):
        s = slice(*args)
        start, stop, step = s.start or 0, s.stop or sys.maxsize, s.step or 1
        it = iter(range(start, stop, step))
        try:
            nexti = next(it)
        except StopIteration:
            # Consume *iterable* up to the *start* position.
            for i, element in zip(range(start), iterable):
                pass
            return
        try:
            for i, element in enumerate(iterable):
                if i == nexti:
                    yield element
                    nexti = next(it)
        except StopIteration:
            # Consume to *stop*.
            for i, element in zip(range(i + 1, stop), iterable):
                pass

    @skipIfTorchDynamo("infinite loop in torch dynamo")
    def test_islice_recipe(self):
        self.assertEqual(list(self.islice('ABCDEFG', 2)), list('AB'))
        self.assertEqual(list(self.islice('ABCDEFG', 2, 4)), list('CD'))
        self.assertEqual(list(self.islice('ABCDEFG', 2, None)), list('CDEFG'))
        self.assertEqual(list(self.islice('ABCDEFG', 0, None, 2)), list('ACEG'))
        # Test items consumed.
        it = iter(range(10))
        self.assertEqual(list(self.islice(it, 3)), list(range(3)))
        self.assertEqual(list(it), list(range(3, 10)))
        it = iter(range(10))
        self.assertEqual(list(self.islice(it, 3, 3)), [])
        self.assertEqual(list(it), list(range(3, 10)))
        # Test that slice finishes in predictable state.
        c = count()
        self.assertEqual(list(self.islice(c, 1, 3, 50)), [1])
        self.assertEqual(next(c), 3)


    def test_tee_recipe(self):

        # Begin tee() recipe ###########################################

        def tee(iterable, n=2):
            if n < 0:
                raise ValueError
            if n == 0:
                return ()
            iterator = _tee(iterable)
            result = [iterator]
            for _ in range(n - 1):
                result.append(_tee(iterator))
            return tuple(result)

        class _tee:

            def __init__(self, iterable):
                it = iter(iterable)
                if isinstance(it, _tee):
                    self.iterator = it.iterator
                    self.link = it.link
                else:
                    self.iterator = it
                    self.link = [None, None]

            def __iter__(self):
                return self

            def __next__(self):
                link = self.link
                if link[1] is None:
                    link[0] = next(self.iterator)
                    link[1] = [None, None]
                value, self.link = link
                return value

        # End tee() recipe #############################################

        n = 200

        a, b = tee([])        # test empty iterator
        self.assertEqual(list(a), [])
        self.assertEqual(list(b), [])

        a, b = tee(irange(n)) # test 100% interleaved
        self.assertEqual(lzip(a,b), lzip(range(n), range(n)))

        a, b = tee(irange(n)) # test 0% interleaved
        self.assertEqual(list(a), list(range(n)))
        self.assertEqual(list(b), list(range(n)))

        a, b = tee(irange(n)) # test dealloc of leading iterator
        for i in range(100):
            self.assertEqual(next(a), i)
        del a
        self.assertEqual(list(b), list(range(n)))

        a, b = tee(irange(n)) # test dealloc of trailing iterator
        for i in range(100):
            self.assertEqual(next(a), i)
        del b
        self.assertEqual(list(a), list(range(100, n)))

        for j in range(5):   # test randomly interleaved
            order = [0]*n + [1]*n
            random.shuffle(order)
            lists = ([], [])
            its = tee(irange(n))
            for i in order:
                value = next(its[i])
                lists[i].append(value)
            self.assertEqual(lists[0], list(range(n)))
            self.assertEqual(lists[1], list(range(n)))

        # test argument format checking
        self.assertRaises(TypeError, tee)
        self.assertRaises(TypeError, tee, 3)
        self.assertRaises(TypeError, tee, [1,2], 'x')
        self.assertRaises(TypeError, tee, [1,2], 3, 'x')

        # tee object should be instantiable
        a, b = tee('abc')
        c = type(a)('def')
        self.assertEqual(list(c), list('def'))

        # test long-lagged and multi-way split
        a, b, c = tee(range(2000), 3)
        for i in range(100):
            self.assertEqual(next(a), i)
        self.assertEqual(list(b), list(range(2000)))
        self.assertEqual([next(c), next(c)], list(range(2)))
        self.assertEqual(list(a), list(range(100,2000)))
        self.assertEqual(list(c), list(range(2,2000)))

        # test invalid values of n
        self.assertRaises(TypeError, tee, 'abc', 'invalid')
        self.assertRaises(ValueError, tee, [], -1)

        for n in range(5):
            result = tee('abc', n)
            self.assertEqual(type(result), tuple)
            self.assertEqual(len(result), n)
            self.assertEqual([list(x) for x in result], [list('abc')]*n)

        # tee objects are independent (see bug gh-123884)
        a, b = tee('abc')
        c, d = tee(a)
        e, f = tee(c)
        self.assertTrue(len({a, b, c, d, e, f}) == 6)

        # test tee_new
        t1, t2 = tee('abc')
        tnew = type(t1)
        self.assertRaises(TypeError, tnew)
        self.assertRaises(TypeError, tnew, 10)
        t3 = tnew(t1)
        self.assertTrue(list(t1) == list(t2) == list(t3) == list('abc'))

        # test that tee objects are weak referenceable
        a, b = tee(range(10))
        p = weakref.proxy(a)
        self.assertEqual(getattr(p, '__class__'), type(b))
        del a
        gc.collect()  # For PyPy or other GCs.
        self.assertRaises(ReferenceError, getattr, p, '__class__')

        ans = list('abc')
        long_ans = list(range(10000))

        # Tests not applicable to the tee() recipe
        if False:
            # check copy
            a, b = tee('abc')
            self.assertEqual(list(copy.copy(a)), ans)
            self.assertEqual(list(copy.copy(b)), ans)
            a, b = tee(list(range(10000)))
            self.assertEqual(list(copy.copy(a)), long_ans)
            self.assertEqual(list(copy.copy(b)), long_ans)

            # check partially consumed copy
            a, b = tee('abc')
            take(2, a)
            take(1, b)
            self.assertEqual(list(copy.copy(a)), ans[2:])
            self.assertEqual(list(copy.copy(b)), ans[1:])
            self.assertEqual(list(a), ans[2:])
            self.assertEqual(list(b), ans[1:])
            a, b = tee(range(10000))
            take(100, a)
            take(60, b)
            self.assertEqual(list(copy.copy(a)), long_ans[100:])
            self.assertEqual(list(copy.copy(b)), long_ans[60:])
            self.assertEqual(list(a), long_ans[100:])
            self.assertEqual(list(b), long_ans[60:])

        # Issue 13454: Crash when deleting backward iterator from tee()
        forward, backward = tee(repeat(None, 2000)) # 20000000
        try:
            any(forward)  # exhaust the iterator
            del backward
        except:
            del forward, backward
            raise


class TestGC(__TestCase):

    def makecycle(self, iterator, container):
        container.append(iterator)
        next(iterator)
        del container, iterator

    def test_accumulate(self):
        a = []
        self.makecycle(accumulate([1,2,a,3]), a)

    def test_batched(self):
        a = []
        self.makecycle(batched([1,2,a,3], 2), a)

    def test_chain(self):
        a = []
        self.makecycle(chain(a), a)

    def test_chain_from_iterable(self):
        a = []
        self.makecycle(chain.from_iterable([a]), a)

    def test_combinations(self):
        a = []
        self.makecycle(combinations([1,2,a,3], 3), a)

    def test_combinations_with_replacement(self):
        a = []
        self.makecycle(combinations_with_replacement([1,2,a,3], 3), a)

    def test_compress(self):
        a = []
        self.makecycle(compress('ABCDEF', [1,0,1,0,1,0]), a)

    def test_count(self):
        a = []
        Int = type('Int', (int,), dict(x=a))
        self.makecycle(count(Int(0), Int(1)), a)

    def test_cycle(self):
        a = []
        self.makecycle(cycle([a]*2), a)

    def test_dropwhile(self):
        a = []
        self.makecycle(dropwhile(bool, [0, a, a]), a)

    def test_groupby(self):
        a = []
        self.makecycle(groupby([a]*2, lambda x:x), a)

    def test_issue2246(self):
        # Issue 2246 -- the _grouper iterator was not included in GC
        n = 10
        keyfunc = lambda x: x
        for i, j in groupby(range(n), key=keyfunc):
            keyfunc.__dict__.setdefault('x',[]).append(j)

    def test_filter(self):
        a = []
        self.makecycle(filter(lambda x:True, [a]*2), a)

    def test_filterfalse(self):
        a = []
        self.makecycle(filterfalse(lambda x:False, a), a)

    def test_zip(self):
        a = []
        self.makecycle(zip([a]*2, [a]*3), a)

    def test_zip_longest(self):
        a = []
        self.makecycle(zip_longest([a]*2, [a]*3), a)
        b = [a, None]
        self.makecycle(zip_longest([a]*2, [a]*3, fillvalue=b), a)

    def test_map(self):
        a = []
        self.makecycle(map(lambda x:x, [a]*2), a)

    def test_islice(self):
        a = []
        self.makecycle(islice([a]*2, None), a)

    def test_pairwise(self):
        a = []
        self.makecycle(pairwise([a]*5), a)

    def test_permutations(self):
        a = []
        self.makecycle(permutations([1,2,a,3], 3), a)

    def test_product(self):
        a = []
        self.makecycle(product([1,2,a,3], repeat=3), a)

    def test_repeat(self):
        a = []
        self.makecycle(repeat(a), a)

    def test_starmap(self):
        a = []
        self.makecycle(starmap(lambda *t: t, [(a,a)]*2), a)

    def test_takewhile(self):
        a = []
        self.makecycle(takewhile(bool, [1, 0, a, a]), a)

def R(seqn):
    'Regular generator'
    for i in seqn:
        yield i

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

class N:
    'Iterator missing __next__()'
    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        return self

class E:
    'Test propagation of exceptions'
    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        3 // 0

class E2:
    'Test propagation of exceptions after two iterations'
    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == 2:
            raise ZeroDivisionError
        v = self.seqn[self.i]
        self.i += 1
        return v

class S:
    'Test immediate stop'
    def __init__(self, seqn):
        pass
    def __iter__(self):
        return self
    def __next__(self):
        raise StopIteration

def L(seqn):
    'Test multiple tiers of iterators'
    return chain(map(lambda x:x, R(Ig(G(seqn)))))


class TestVariousIteratorArgs(__TestCase):

    def test_accumulate(self):
        s = [1,2,3,4,5]
        r = [1,3,6,10,15]
        n = len(s)
        for g in (G, I, Ig, L, R):
            self.assertEqual(list(accumulate(g(s))), r)
        self.assertEqual(list(accumulate(S(s))), [])
        self.assertRaises(TypeError, accumulate, X(s))
        self.assertRaises(TypeError, accumulate, N(s))
        self.assertRaises(ZeroDivisionError, list, accumulate(E(s)))

    def test_batched(self):
        s = 'abcde'
        r = [('a', 'b'), ('c', 'd'), ('e',)]
        n = 2
        for g in (G, I, Ig, L, R):
            with self.subTest(g=g):
                self.assertEqual(list(batched(g(s), n)), r)
        self.assertEqual(list(batched(S(s), 2)), [])
        self.assertRaises(TypeError, batched, X(s), 2)
        self.assertRaises(TypeError, batched, N(s), 2)
        self.assertRaises(ZeroDivisionError, list, batched(E(s), 2))
        self.assertRaises(ZeroDivisionError, list, batched(E2(s), 4))

    def test_chain(self):
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(chain(g(s))), list(g(s)))
                self.assertEqual(list(chain(g(s), g(s))), list(g(s))+list(g(s)))
            self.assertRaises(TypeError, list, chain(X(s)))
            self.assertRaises(TypeError, list, chain(N(s)))
            self.assertRaises(ZeroDivisionError, list, chain(E(s)))

    def test_compress(self):
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            n = len(s)
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(compress(g(s), repeat(1))), list(g(s)))
            self.assertRaises(TypeError, compress, X(s), repeat(1))
            self.assertRaises(TypeError, compress, N(s), repeat(1))
            self.assertRaises(ZeroDivisionError, list, compress(E(s), repeat(1)))

    def test_product(self):
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            self.assertRaises(TypeError, product, X(s))
            self.assertRaises(TypeError, product, N(s))
            self.assertRaises(ZeroDivisionError, product, E(s))

    def test_cycle(self):
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                tgtlen = len(s) * 3
                expected = list(g(s))*3
                actual = list(islice(cycle(g(s)), tgtlen))
                self.assertEqual(actual, expected)
            self.assertRaises(TypeError, cycle, X(s))
            self.assertRaises(TypeError, cycle, N(s))
            self.assertRaises(ZeroDivisionError, list, cycle(E(s)))

    def test_groupby(self):
        for s in (range(10), range(0), range(1000), (7,11), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual([k for k, sb in groupby(g(s))], list(g(s)))
            self.assertRaises(TypeError, groupby, X(s))
            self.assertRaises(TypeError, groupby, N(s))
            self.assertRaises(ZeroDivisionError, list, groupby(E(s)))

    def test_filter(self):
        for s in (range(10), range(0), range(1000), (7,11), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(filter(isEven, g(s))),
                                 [x for x in g(s) if isEven(x)])
            self.assertRaises(TypeError, filter, isEven, X(s))
            self.assertRaises(TypeError, filter, isEven, N(s))
            self.assertRaises(ZeroDivisionError, list, filter(isEven, E(s)))

    def test_filterfalse(self):
        for s in (range(10), range(0), range(1000), (7,11), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(filterfalse(isEven, g(s))),
                                 [x for x in g(s) if isOdd(x)])
            self.assertRaises(TypeError, filterfalse, isEven, X(s))
            self.assertRaises(TypeError, filterfalse, isEven, N(s))
            self.assertRaises(ZeroDivisionError, list, filterfalse(isEven, E(s)))

    def test_zip(self):
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(zip(g(s))), lzip(g(s)))
                self.assertEqual(list(zip(g(s), g(s))), lzip(g(s), g(s)))
            self.assertRaises(TypeError, zip, X(s))
            self.assertRaises(TypeError, zip, N(s))
            self.assertRaises(ZeroDivisionError, list, zip(E(s)))

    def test_ziplongest(self):
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(zip_longest(g(s))), list(zip(g(s))))
                self.assertEqual(list(zip_longest(g(s), g(s))), list(zip(g(s), g(s))))
            self.assertRaises(TypeError, zip_longest, X(s))
            self.assertRaises(TypeError, zip_longest, N(s))
            self.assertRaises(ZeroDivisionError, list, zip_longest(E(s)))

    def test_map(self):
        for s in (range(10), range(0), range(100), (7,11), range(20,50,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(map(onearg, g(s))),
                                 [onearg(x) for x in g(s)])
                self.assertEqual(list(map(operator.pow, g(s), g(s))),
                                 [x**x for x in g(s)])
            self.assertRaises(TypeError, map, onearg, X(s))
            self.assertRaises(TypeError, map, onearg, N(s))
            self.assertRaises(ZeroDivisionError, list, map(onearg, E(s)))

    def test_islice(self):
        for s in ("12345", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(islice(g(s),1,None,2)), list(g(s))[1::2])
            self.assertRaises(TypeError, islice, X(s), 10)
            self.assertRaises(TypeError, islice, N(s), 10)
            self.assertRaises(ZeroDivisionError, list, islice(E(s), 10))

    def test_pairwise(self):
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                seq = list(g(s))
                expected = list(zip(seq, seq[1:]))
                actual = list(pairwise(g(s)))
                self.assertEqual(actual, expected)
            self.assertRaises(TypeError, pairwise, X(s))
            self.assertRaises(TypeError, pairwise, N(s))
            self.assertRaises(ZeroDivisionError, list, pairwise(E(s)))

    def test_starmap(self):
        for s in (range(10), range(0), range(100), (7,11), range(20,50,5)):
            for g in (G, I, Ig, S, L, R):
                ss = lzip(s, s)
                self.assertEqual(list(starmap(operator.pow, g(ss))),
                                 [x**x for x in g(s)])
            self.assertRaises(TypeError, starmap, operator.pow, X(ss))
            self.assertRaises(TypeError, starmap, operator.pow, N(ss))
            self.assertRaises(ZeroDivisionError, list, starmap(operator.pow, E(ss)))

    def test_takewhile(self):
        for s in (range(10), range(0), range(1000), (7,11), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                tgt = []
                for elem in g(s):
                    if not isEven(elem): break
                    tgt.append(elem)
                self.assertEqual(list(takewhile(isEven, g(s))), tgt)
            self.assertRaises(TypeError, takewhile, isEven, X(s))
            self.assertRaises(TypeError, takewhile, isEven, N(s))
            self.assertRaises(ZeroDivisionError, list, takewhile(isEven, E(s)))

    def test_dropwhile(self):
        for s in (range(10), range(0), range(1000), (7,11), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                tgt = []
                for elem in g(s):
                    if not tgt and isOdd(elem): continue
                    tgt.append(elem)
                self.assertEqual(list(dropwhile(isOdd, g(s))), tgt)
            self.assertRaises(TypeError, dropwhile, isOdd, X(s))
            self.assertRaises(TypeError, dropwhile, isOdd, N(s))
            self.assertRaises(ZeroDivisionError, list, dropwhile(isOdd, E(s)))

    def test_tee(self):
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                it1, it2 = tee(g(s))
                self.assertEqual(list(it1), list(g(s)))
                self.assertEqual(list(it2), list(g(s)))
            self.assertRaises(TypeError, tee, X(s))
            self.assertRaises(TypeError, tee, N(s))
            self.assertRaises(ZeroDivisionError, list, tee(E(s))[0])

class LengthTransparency(__TestCase):

    def test_repeat(self):
        self.assertEqual(operator.length_hint(repeat(None, 50)), 50)
        self.assertEqual(operator.length_hint(repeat(None, 0)), 0)
        self.assertEqual(operator.length_hint(repeat(None), 12), 12)

    def test_repeat_with_negative_times(self):
        self.assertEqual(operator.length_hint(repeat(None, -1)), 0)
        self.assertEqual(operator.length_hint(repeat(None, -2)), 0)
        self.assertEqual(operator.length_hint(repeat(None, times=-1)), 0)
        self.assertEqual(operator.length_hint(repeat(None, times=-2)), 0)

class RegressionTests(__TestCase):

    def test_sf_793826(self):
        # Fix Armin Rigo's successful efforts to wreak havoc

        def mutatingtuple(tuple1, f, tuple2):
            # this builds a tuple t which is a copy of tuple1,
            # then calls f(t), then mutates t to be equal to tuple2
            # (needs len(tuple1) == len(tuple2)).
            def g(value, first=[1]):
                if first:
                    del first[:]
                    f(next(z))
                return value
            items = list(tuple2)
            items[1:1] = list(tuple1)
            gen = map(g, items)
            z = zip(*[gen]*len(tuple1))
            next(z)

        def f(t):
            global T
            T = t
            first[:] = list(T)

        first = []
        mutatingtuple((1,2,3), f, (4,5,6))
        second = list(T)
        self.assertEqual(first, second)


    def test_sf_950057(self):
        # Make sure that chain() and cycle() catch exceptions immediately
        # rather than when shifting between input sources

        def gen1():
            hist.append(0)
            yield 1
            hist.append(1)
            raise AssertionError
            hist.append(2)

        def gen2(x):
            hist.append(3)
            yield 2
            hist.append(4)

        hist = []
        self.assertRaises(AssertionError, list, chain(gen1(), gen2(False)))
        self.assertEqual(hist, [0,1])

        hist = []
        self.assertRaises(AssertionError, list, chain(gen1(), gen2(True)))
        self.assertEqual(hist, [0,1])

        hist = []
        self.assertRaises(AssertionError, list, cycle(gen1()))
        self.assertEqual(hist, [0,1])

    @support.skip_if_pgo_task
    @support.requires_resource('cpu')
    @slowTest
    def test_long_chain_of_empty_iterables(self):
        # Make sure itertools.chain doesn't run into recursion limits when
        # dealing with long chains of empty iterables. Even with a high
        # number this would probably only fail in Py_DEBUG mode.
        it = chain.from_iterable(() for unused in range(10000000))
        with self.assertRaises(StopIteration):
            next(it)

    def test_issue30347_1(self):
        def f(n):
            if n == 5:
                list(b)
            return n != 6
        for (k, b) in groupby(range(10), f):
            list(b)  # shouldn't crash

    def test_issue30347_2(self):
        class K:
            def __init__(self, v):
                pass
            def __eq__(self, other):
                nonlocal i
                i += 1
                if i == 1:
                    next(g, None)
                return True
        i = 0
        g = next(groupby(range(10), K))[1]
        for j in range(2):
            next(g, None)  # shouldn't crash


class SubclassWithKwargsTest(__TestCase):
    def test_keywords_in_subclass(self):
        # count is not subclassable...
        testcases = [
            (repeat, (1, 2), [1, 1]),
            (zip, ([1, 2], 'ab'), [(1, 'a'), (2, 'b')]),
            (filter, (None, [0, 1]), [1]),
            (filterfalse, (None, [0, 1]), [0]),
            (chain, ([1, 2], [3, 4]), [1, 2, 3]),
            (map, (str, [1, 2]), ['1', '2']),
            (starmap, (operator.pow, ((2, 3), (3, 2))), [8, 9]),
            (islice, ([1, 2, 3, 4], 1, 3), [2, 3]),
            (takewhile, (isEven, [2, 3, 4]), [2]),
            (dropwhile, (isEven, [2, 3, 4]), [3, 4]),
            (cycle, ([1, 2],), [1, 2, 1]),
            (compress, ('ABC', [1, 0, 1]), ['A', 'C']),
        ]
        for cls, args, result in testcases:
            with self.subTest(cls):
                class subclass(cls):
                    pass
                u = subclass(*args)
                self.assertIs(type(u), subclass)
                self.assertEqual(list(islice(u, 0, 3)), result)
                with self.assertRaises(TypeError):
                    subclass(*args, newarg=3)

        for cls, args, result in testcases:
            # Constructors of repeat, zip, compress accept keyword arguments.
            # Their subclasses need overriding __new__ to support new
            # keyword arguments.
            if cls in [repeat, zip, compress]:
                continue
            with self.subTest(cls):
                class subclass_with_init(cls):
                    def __init__(self, *args, newarg=None):
                        self.newarg = newarg
                u = subclass_with_init(*args, newarg=3)
                self.assertIs(type(u), subclass_with_init)
                self.assertEqual(list(islice(u, 0, 3)), result)
                self.assertEqual(u.newarg, 3)

        for cls, args, result in testcases:
            with self.subTest(cls):
                class subclass_with_new(cls):
                    def __new__(cls, *args, newarg=None):
                        self = super().__new__(cls, *args)
                        self.newarg = newarg
                        return self
                u = subclass_with_new(*args, newarg=3)
                self.assertIs(type(u), subclass_with_new)
                self.assertEqual(list(islice(u, 0, 3)), result)
                self.assertEqual(u.newarg, 3)


if __name__ == "__main__":
    run_tests()
