# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    TEST_WITH_TORCHDYNAMO,
    skipIfTorchDynamo,
    run_tests,
)

if TEST_WITH_TORCHDYNAMO:
    __TestCase = CPythonTestCase
else:
    __TestCase = unittest.TestCase

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

# Test iterators.

import sys
import unittest
from test.support import cpython_only
from test.support.os_helper import TESTFN, unlink
from test.support import check_free_after_iterating, ALWAYS_EQ, NEVER_EQ
import pickle
import collections.abc
import functools
import contextlib
import builtins

# Test result of triple loop (too big to inline)
TRIPLETS = [(0, 0, 0), (0, 0, 1), (0, 0, 2),
            (0, 1, 0), (0, 1, 1), (0, 1, 2),
            (0, 2, 0), (0, 2, 1), (0, 2, 2),

            (1, 0, 0), (1, 0, 1), (1, 0, 2),
            (1, 1, 0), (1, 1, 1), (1, 1, 2),
            (1, 2, 0), (1, 2, 1), (1, 2, 2),

            (2, 0, 0), (2, 0, 1), (2, 0, 2),
            (2, 1, 0), (2, 1, 1), (2, 1, 2),
            (2, 2, 0), (2, 2, 1), (2, 2, 2)]

# Helper classes

class BasicIterClass:
    def __init__(self, n):
        self.n = n
        self.i = 0
    def __next__(self):
        res = self.i
        if res >= self.n:
            raise StopIteration
        self.i = res + 1
        return res
    def __iter__(self):
        return self

class IteratingSequenceClass:
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return BasicIterClass(self.n)

class IteratorProxyClass:
    def __init__(self, i):
        self.i = i
    def __next__(self):
        return next(self.i)
    def __iter__(self):
        return self

class SequenceClass:
    def __init__(self, n):
        self.n = n
    def __getitem__(self, i):
        if 0 <= i < self.n:
            return i
        else:
            raise IndexError

class SequenceProxyClass:
    def __init__(self, s):
        self.s = s
    def __getitem__(self, i):
        return self.s[i]

class UnlimitedSequenceClass:
    def __getitem__(self, i):
        return i

class DefaultIterClass:
    pass

class NoIterClass:
    def __getitem__(self, i):
        return i
    __iter__ = None

class BadIterableClass:
    def __iter__(self):
        raise ZeroDivisionError

class CallableIterClass:
    def __init__(self):
        self.i = 0
    def __call__(self):
        i = self.i
        self.i = i + 1
        if i > 100:
            raise IndexError # Emergency stop
        return i

class EmptyIterClass:
    def __len__(self):
        return 0
    def __getitem__(self, i):
        raise StopIteration

# Main test suite

class TestCase(__TestCase):

    # Helper to check that an iterator returns a given sequence
    def check_iterator(self, it, seq, pickle=True):
        if pickle:
            self.check_pickle(it, seq)
        res = []
        while 1:
            try:
                val = next(it)
            except StopIteration:
                break
            res.append(val)
        self.assertEqual(res, seq)

    # Helper to check that a for loop generates a given sequence
    def check_for_loop(self, expr, seq, pickle=True):
        if pickle:
            self.check_pickle(iter(expr), seq)
        res = []
        for val in expr:
            res.append(val)
        self.assertEqual(res, seq)

    # Helper to check picklability
    def check_pickle(self, itorg, seq):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            d = pickle.dumps(itorg, proto)
            it = pickle.loads(d)
            # Cannot assert type equality because dict iterators unpickle as list
            # iterators.
            # self.assertEqual(type(itorg), type(it))
            self.assertTrue(isinstance(it, collections.abc.Iterator))
            self.assertEqual(list(it), seq)

            it = pickle.loads(d)
            try:
                next(it)
            except StopIteration:
                continue
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            self.assertEqual(list(it), seq[1:])

    # Test basic use of iter() function
    def test_iter_basic(self):
        self.check_iterator(iter(range(10)), list(range(10)))

    # Test that iter(iter(x)) is the same as iter(x)
    def test_iter_idempotency(self):
        seq = list(range(10))
        it = iter(seq)
        it2 = iter(it)
        self.assertTrue(it is it2)

    # Test that for loops over iterators work
    def test_iter_for_loop(self):
        self.check_for_loop(iter(range(10)), list(range(10)))

    # Test several independent iterators over the same list
    def test_iter_independence(self):
        seq = range(3)
        res = []
        for i in iter(seq):
            for j in iter(seq):
                for k in iter(seq):
                    res.append((i, j, k))
        self.assertEqual(res, TRIPLETS)

    # Test triple list comprehension using iterators
    def test_nested_comprehensions_iter(self):
        seq = range(3)
        res = [(i, j, k)
               for i in iter(seq) for j in iter(seq) for k in iter(seq)]
        self.assertEqual(res, TRIPLETS)

    # Test triple list comprehension without iterators
    def test_nested_comprehensions_for(self):
        seq = range(3)
        res = [(i, j, k) for i in seq for j in seq for k in seq]
        self.assertEqual(res, TRIPLETS)

    # Test a class with __iter__ in a for loop
    def test_iter_class_for(self):
        self.check_for_loop(IteratingSequenceClass(10), list(range(10)))

    # Test a class with __iter__ with explicit iter()
    def test_iter_class_iter(self):
        self.check_iterator(iter(IteratingSequenceClass(10)), list(range(10)))

    # Test for loop on a sequence class without __iter__
    def test_seq_class_for(self):
        self.check_for_loop(SequenceClass(10), list(range(10)))

    # Test iter() on a sequence class without __iter__
    def test_seq_class_iter(self):
        self.check_iterator(iter(SequenceClass(10)), list(range(10)))

    def test_mutating_seq_class_iter_pickle(self):
        orig = SequenceClass(5)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # initial iterator
            itorig = iter(orig)
            d = pickle.dumps((itorig, orig), proto)
            it, seq = pickle.loads(d)
            seq.n = 7
            self.assertIs(type(it), type(itorig))
            self.assertEqual(list(it), list(range(7)))

            # running iterator
            next(itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, seq = pickle.loads(d)
            seq.n = 7
            self.assertIs(type(it), type(itorig))
            self.assertEqual(list(it), list(range(1, 7)))

            # empty iterator
            for i in range(1, 5):
                next(itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, seq = pickle.loads(d)
            seq.n = 7
            self.assertIs(type(it), type(itorig))
            self.assertEqual(list(it), list(range(5, 7)))

            # exhausted iterator
            self.assertRaises(StopIteration, next, itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, seq = pickle.loads(d)
            seq.n = 7
            self.assertTrue(isinstance(it, collections.abc.Iterator))
            self.assertEqual(list(it), [])

    def test_mutating_seq_class_exhausted_iter(self):
        a = SequenceClass(5)
        exhit = iter(a)
        empit = iter(a)
        for x in exhit:  # exhaust the iterator
            next(empit)  # not exhausted
        a.n = 7
        self.assertEqual(list(exhit), [])
        self.assertEqual(list(empit), [5, 6])
        self.assertEqual(list(a), [0, 1, 2, 3, 4, 5, 6])

    def test_reduce_mutating_builtins_iter(self):
        # This is a reproducer of issue #101765
        # where iter `__reduce__` calls could lead to a segfault or SystemError
        # depending on the order of C argument evaluation, which is undefined

        # Backup builtins
        builtins_dict = builtins.__dict__
        orig = {"iter": iter, "reversed": reversed}

        def run(builtin_name, item, sentinel=None):
            it = iter(item) if sentinel is None else iter(item, sentinel)

            class CustomStr:
                def __init__(self, name, iterator):
                    self.name = name
                    self.iterator = iterator
                def __hash__(self):
                    return hash(self.name)
                def __eq__(self, other):
                    # Here we exhaust our iterator, possibly changing
                    # its `it_seq` pointer to NULL
                    # The `__reduce__` call should correctly get
                    # the pointers after this call
                    list(self.iterator)
                    return other == self.name

            # del is required here
            # to not prematurely call __eq__ from
            # the hash collision with the old key
            del builtins_dict[builtin_name]
            builtins_dict[CustomStr(builtin_name, it)] = orig[builtin_name]

            return it.__reduce__()

        types = [
            (EmptyIterClass(),),
            (bytes(8),),
            (bytearray(8),),
            ((1, 2, 3),),
            (lambda: 0, 0),
            (tuple[int],)  # GenericAlias
        ]

        try:
            run_iter = functools.partial(run, "iter")
            # The returned value of `__reduce__` should not only be valid
            # but also *empty*, as `it` was exhausted during `__eq__`
            # i.e "xyz" returns (iter, ("",))
            self.assertEqual(run_iter("xyz"), (orig["iter"], ("",)))
            self.assertEqual(run_iter([1, 2, 3]), (orig["iter"], ([],)))

            # _PyEval_GetBuiltin is also called for `reversed` in a branch of
            # listiter_reduce_general
            self.assertEqual(
                run("reversed", orig["reversed"](list(range(8)))),
                (reversed, ([],))
            )

            for case in types:
                self.assertEqual(run_iter(*case), (orig["iter"], ((),)))
        finally:
            # Restore original builtins
            for key, func in orig.items():
                # need to suppress KeyErrors in case
                # a failed test deletes the key without setting anything
                with contextlib.suppress(KeyError):
                    # del is required here
                    # to not invoke our custom __eq__ from
                    # the hash collision with the old key
                    del builtins_dict[key]
                builtins_dict[key] = func

    # Test a new_style class with __iter__ but no next() method
    def test_new_style_iter_class(self):
        class IterClass(object):
            def __iter__(self):
                return self
        self.assertRaises(TypeError, iter, IterClass())

    # Test two-argument iter() with callable instance
    def test_iter_callable(self):
        self.check_iterator(iter(CallableIterClass(), 10), list(range(10)), pickle=True)

    # Test two-argument iter() with function
    def test_iter_function(self):
        def spam(state=[0]):
            i = state[0]
            state[0] = i+1
            return i
        self.check_iterator(iter(spam, 10), list(range(10)), pickle=False)

    # Test two-argument iter() with function that raises StopIteration
    def test_iter_function_stop(self):
        def spam(state=[0]):
            i = state[0]
            if i == 10:
                raise StopIteration
            state[0] = i+1
            return i
        self.check_iterator(iter(spam, 20), list(range(10)), pickle=False)

    def test_iter_function_concealing_reentrant_exhaustion(self):
        # gh-101892: Test two-argument iter() with a function that
        # exhausts its associated iterator but forgets to either return
        # a sentinel value or raise StopIteration.
        HAS_MORE = 1
        NO_MORE = 2

        def exhaust(iterator):
            """Exhaust an iterator without raising StopIteration."""
            list(iterator)

        def spam():
            # Touching the iterator with exhaust() below will call
            # spam() once again so protect against recursion.
            if spam.is_recursive_call:
                return NO_MORE
            spam.is_recursive_call = True
            exhaust(spam.iterator)
            return HAS_MORE

        spam.is_recursive_call = False
        spam.iterator = iter(spam, NO_MORE)
        with self.assertRaises(StopIteration):
            next(spam.iterator)

    # Test exception propagation through function iterator
    def test_exception_function(self):
        def spam(state=[0]):
            i = state[0]
            state[0] = i+1
            if i == 10:
                raise RuntimeError
            return i
        res = []
        try:
            for x in iter(spam, 20):
                res.append(x)
        except RuntimeError:
            self.assertEqual(res, list(range(10)))
        else:
            self.fail("should have raised RuntimeError")

    # Test exception propagation through sequence iterator
    def test_exception_sequence(self):
        class MySequenceClass(SequenceClass):
            def __getitem__(self, i):
                if i == 10:
                    raise RuntimeError
                return SequenceClass.__getitem__(self, i)
        res = []
        try:
            for x in MySequenceClass(20):
                res.append(x)
        except RuntimeError:
            self.assertEqual(res, list(range(10)))
        else:
            self.fail("should have raised RuntimeError")

    # Test for StopIteration from __getitem__
    def test_stop_sequence(self):
        class MySequenceClass(SequenceClass):
            def __getitem__(self, i):
                if i == 10:
                    raise StopIteration
                return SequenceClass.__getitem__(self, i)
        self.check_for_loop(MySequenceClass(20), list(range(10)), pickle=False)

    # Test a big range
    def test_iter_big_range(self):
        self.check_for_loop(iter(range(10000)), list(range(10000)))

    # Test an empty list
    def test_iter_empty(self):
        self.check_for_loop(iter([]), [])

    # Test a tuple
    def test_iter_tuple(self):
        self.check_for_loop(iter((0,1,2,3,4,5,6,7,8,9)), list(range(10)))

    # Test a range
    def test_iter_range(self):
        self.check_for_loop(iter(range(10)), list(range(10)))

    # Test a string
    def test_iter_string(self):
        self.check_for_loop(iter("abcde"), ["a", "b", "c", "d", "e"])

    # Test a directory
    def test_iter_dict(self):
        dict = {}
        for i in range(10):
            dict[i] = None
        self.check_for_loop(dict, list(dict.keys()))

    # Test a file
    def test_iter_file(self):
        f = open(TESTFN, "w", encoding="utf-8")
        try:
            for i in range(5):
                f.write("%d\n" % i)
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            self.check_for_loop(f, ["0\n", "1\n", "2\n", "3\n", "4\n"], pickle=False)
            self.check_for_loop(f, [], pickle=False)
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test list()'s use of iterators.
    def test_builtin_list(self):
        self.assertEqual(list(SequenceClass(5)), list(range(5)))
        self.assertEqual(list(SequenceClass(0)), [])
        self.assertEqual(list(()), [])

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(list(d), list(d.keys()))

        self.assertRaises(TypeError, list, list)
        self.assertRaises(TypeError, list, 42)

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            for i in range(5):
                f.write("%d\n" % i)
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            self.assertEqual(list(f), ["0\n", "1\n", "2\n", "3\n", "4\n"])
            f.seek(0, 0)
            self.assertEqual(list(f),
                             ["0\n", "1\n", "2\n", "3\n", "4\n"])
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test tuples()'s use of iterators.
    def test_builtin_tuple(self):
        self.assertEqual(tuple(SequenceClass(5)), (0, 1, 2, 3, 4))
        self.assertEqual(tuple(SequenceClass(0)), ())
        self.assertEqual(tuple([]), ())
        self.assertEqual(tuple(()), ())
        self.assertEqual(tuple("abc"), ("a", "b", "c"))

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(tuple(d), tuple(d.keys()))

        self.assertRaises(TypeError, tuple, list)
        self.assertRaises(TypeError, tuple, 42)

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            for i in range(5):
                f.write("%d\n" % i)
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            self.assertEqual(tuple(f), ("0\n", "1\n", "2\n", "3\n", "4\n"))
            f.seek(0, 0)
            self.assertEqual(tuple(f),
                             ("0\n", "1\n", "2\n", "3\n", "4\n"))
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test filter()'s use of iterators.
    def test_builtin_filter(self):
        self.assertEqual(list(filter(None, SequenceClass(5))),
                         list(range(1, 5)))
        self.assertEqual(list(filter(None, SequenceClass(0))), [])
        self.assertEqual(list(filter(None, ())), [])
        self.assertEqual(list(filter(None, "abc")), ["a", "b", "c"])

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(list(filter(None, d)), list(d.keys()))

        self.assertRaises(TypeError, filter, None, list)
        self.assertRaises(TypeError, filter, None, 42)

        class Boolean:
            def __init__(self, truth):
                self.truth = truth
            def __bool__(self):
                return self.truth
        bTrue = Boolean(True)
        bFalse = Boolean(False)

        class Seq:
            def __init__(self, *args):
                self.vals = args
            def __iter__(self):
                class SeqIter:
                    def __init__(self, vals):
                        self.vals = vals
                        self.i = 0
                    def __iter__(self):
                        return self
                    def __next__(self):
                        i = self.i
                        self.i = i + 1
                        if i < len(self.vals):
                            return self.vals[i]
                        else:
                            raise StopIteration
                return SeqIter(self.vals)

        seq = Seq(*([bTrue, bFalse] * 25))
        self.assertEqual(list(filter(lambda x: not x, seq)), [bFalse]*25)
        self.assertEqual(list(filter(lambda x: not x, iter(seq))), [bFalse]*25)

    # Test max() and min()'s use of iterators.
    def test_builtin_max_min(self):
        self.assertEqual(max(SequenceClass(5)), 4)
        self.assertEqual(min(SequenceClass(5)), 0)
        self.assertEqual(max(8, -1), 8)
        self.assertEqual(min(8, -1), -1)

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(max(d), "two")
        self.assertEqual(min(d), "one")
        self.assertEqual(max(d.values()), 3)
        self.assertEqual(min(iter(d.values())), 1)

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            f.write("medium line\n")
            f.write("xtra large line\n")
            f.write("itty-bitty line\n")
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            self.assertEqual(min(f), "itty-bitty line\n")
            f.seek(0, 0)
            self.assertEqual(max(f), "xtra large line\n")
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test map()'s use of iterators.
    def test_builtin_map(self):
        self.assertEqual(list(map(lambda x: x+1, SequenceClass(5))),
                         list(range(1, 6)))

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(list(map(lambda k, d=d: (k, d[k]), d)),
                         list(d.items()))
        dkeys = list(d.keys())
        expected = [(i < len(d) and dkeys[i] or None,
                     i,
                     i < len(d) and dkeys[i] or None)
                    for i in range(3)]

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            for i in range(10):
                f.write("xy" * i + "\n") # line i has len 2*i+1
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            self.assertEqual(list(map(len, f)), list(range(1, 21, 2)))
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test zip()'s use of iterators.
    @skipIfTorchDynamo("infinite loop")
    def test_builtin_zip(self):
        self.assertEqual(list(zip()), [])
        self.assertEqual(list(zip(*[])), [])
        self.assertEqual(list(zip(*[(1, 2), 'ab'])), [(1, 'a'), (2, 'b')])

        self.assertRaises(TypeError, zip, None)
        self.assertRaises(TypeError, zip, range(10), 42)
        self.assertRaises(TypeError, zip, range(10), zip)

        self.assertEqual(list(zip(IteratingSequenceClass(3))),
                         [(0,), (1,), (2,)])
        self.assertEqual(list(zip(SequenceClass(3))),
                         [(0,), (1,), (2,)])

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(list(d.items()), list(zip(d, d.values())))

        # Generate all ints starting at constructor arg.
        class IntsFrom:
            def __init__(self, start):
                self.i = start

            def __iter__(self):
                return self

            def __next__(self):
                i = self.i
                self.i = i+1
                return i

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            f.write("a\n" "bbb\n" "cc\n")
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            self.assertEqual(list(zip(IntsFrom(0), f, IntsFrom(-100))),
                             [(0, "a\n", -100),
                              (1, "bbb\n", -99),
                              (2, "cc\n", -98)])
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

        self.assertEqual(list(zip(range(5))), [(i,) for i in range(5)])

        # Classes that lie about their lengths.
        class NoGuessLen5:
            def __getitem__(self, i):
                if i >= 5:
                    raise IndexError
                return i

        class Guess3Len5(NoGuessLen5):
            def __len__(self):
                return 3

        class Guess30Len5(NoGuessLen5):
            def __len__(self):
                return 30

        def lzip(*args):
            return list(zip(*args))

        self.assertEqual(len(Guess3Len5()), 3)
        self.assertEqual(len(Guess30Len5()), 30)
        self.assertEqual(lzip(NoGuessLen5()), lzip(range(5)))
        self.assertEqual(lzip(Guess3Len5()), lzip(range(5)))
        self.assertEqual(lzip(Guess30Len5()), lzip(range(5)))

        expected = [(i, i) for i in range(5)]
        for x in NoGuessLen5(), Guess3Len5(), Guess30Len5():
            for y in NoGuessLen5(), Guess3Len5(), Guess30Len5():
                self.assertEqual(lzip(x, y), expected)

    def test_unicode_join_endcase(self):

        # This class inserts a Unicode object into its argument's natural
        # iteration, in the 3rd position.
        class OhPhooey:
            def __init__(self, seq):
                self.it = iter(seq)
                self.i = 0

            def __iter__(self):
                return self

            def __next__(self):
                i = self.i
                self.i = i+1
                if i == 2:
                    return "fooled you!"
                return next(self.it)

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            f.write("a\n" + "b\n" + "c\n")
        finally:
            f.close()

        f = open(TESTFN, "r", encoding="utf-8")
        # Nasty:  string.join(s) can't know whether unicode.join() is needed
        # until it's seen all of s's elements.  But in this case, f's
        # iterator cannot be restarted.  So what we're testing here is
        # whether string.join() can manage to remember everything it's seen
        # and pass that on to unicode.join().
        try:
            got = " - ".join(OhPhooey(f))
            self.assertEqual(got, "a\n - b\n - fooled you! - c\n")
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test iterators with 'x in y' and 'x not in y'.
    def test_in_and_not_in(self):
        for sc5 in IteratingSequenceClass(5), SequenceClass(5):
            for i in range(5):
                self.assertIn(i, sc5)
            for i in "abc", -1, 5, 42.42, (3, 4), [], {1: 1}, 3-12j, sc5:
                self.assertNotIn(i, sc5)

        self.assertIn(ALWAYS_EQ, IteratorProxyClass(iter([1])))
        self.assertIn(ALWAYS_EQ, SequenceProxyClass([1]))
        self.assertNotIn(ALWAYS_EQ, IteratorProxyClass(iter([NEVER_EQ])))
        self.assertNotIn(ALWAYS_EQ, SequenceProxyClass([NEVER_EQ]))
        self.assertIn(NEVER_EQ, IteratorProxyClass(iter([ALWAYS_EQ])))
        self.assertIn(NEVER_EQ, SequenceProxyClass([ALWAYS_EQ]))

        self.assertRaises(TypeError, lambda: 3 in 12)
        self.assertRaises(TypeError, lambda: 3 not in map)
        self.assertRaises(ZeroDivisionError, lambda: 3 in BadIterableClass())

        d = {"one": 1, "two": 2, "three": 3, 1j: 2j}
        for k in d:
            self.assertIn(k, d)
            self.assertNotIn(k, d.values())
        for v in d.values():
            self.assertIn(v, d.values())
            self.assertNotIn(v, d)
        for k, v in d.items():
            self.assertIn((k, v), d.items())
            self.assertNotIn((v, k), d.items())

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            f.write("a\n" "b\n" "c\n")
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            for chunk in "abc":
                f.seek(0, 0)
                self.assertNotIn(chunk, f)
                f.seek(0, 0)
                self.assertIn((chunk + "\n"), f)
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test iterators with operator.countOf (PySequence_Count).
    def test_countOf(self):
        from operator import countOf
        self.assertEqual(countOf([1,2,2,3,2,5], 2), 3)
        self.assertEqual(countOf((1,2,2,3,2,5), 2), 3)
        self.assertEqual(countOf("122325", "2"), 3)
        self.assertEqual(countOf("122325", "6"), 0)

        self.assertRaises(TypeError, countOf, 42, 1)
        self.assertRaises(TypeError, countOf, countOf, countOf)

        d = {"one": 3, "two": 3, "three": 3, 1j: 2j}
        for k in d:
            self.assertEqual(countOf(d, k), 1)
        self.assertEqual(countOf(d.values(), 3), 3)
        self.assertEqual(countOf(d.values(), 2j), 1)
        self.assertEqual(countOf(d.values(), 1j), 0)

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            f.write("a\n" "b\n" "c\n" "b\n")
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            for letter, count in ("a", 1), ("b", 2), ("c", 1), ("d", 0):
                f.seek(0, 0)
                self.assertEqual(countOf(f, letter + "\n"), count)
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test iterators with operator.indexOf (PySequence_Index).
    def test_indexOf(self):
        from operator import indexOf
        self.assertEqual(indexOf([1,2,2,3,2,5], 1), 0)
        self.assertEqual(indexOf((1,2,2,3,2,5), 2), 1)
        self.assertEqual(indexOf((1,2,2,3,2,5), 3), 3)
        self.assertEqual(indexOf((1,2,2,3,2,5), 5), 5)
        self.assertRaises(ValueError, indexOf, (1,2,2,3,2,5), 0)
        self.assertRaises(ValueError, indexOf, (1,2,2,3,2,5), 6)

        self.assertEqual(indexOf("122325", "2"), 1)
        self.assertEqual(indexOf("122325", "5"), 5)
        self.assertRaises(ValueError, indexOf, "122325", "6")

        self.assertRaises(TypeError, indexOf, 42, 1)
        self.assertRaises(TypeError, indexOf, indexOf, indexOf)
        self.assertRaises(ZeroDivisionError, indexOf, BadIterableClass(), 1)

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            f.write("a\n" "b\n" "c\n" "d\n" "e\n")
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            fiter = iter(f)
            self.assertEqual(indexOf(fiter, "b\n"), 1)
            self.assertEqual(indexOf(fiter, "d\n"), 1)
            self.assertEqual(indexOf(fiter, "e\n"), 0)
            self.assertRaises(ValueError, indexOf, fiter, "a\n")
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

        iclass = IteratingSequenceClass(3)
        for i in range(3):
            self.assertEqual(indexOf(iclass, i), i)
        self.assertRaises(ValueError, indexOf, iclass, -1)

    # Test iterators with file.writelines().
    def test_writelines(self):
        f = open(TESTFN, "w", encoding="utf-8")

        try:
            self.assertRaises(TypeError, f.writelines, None)
            self.assertRaises(TypeError, f.writelines, 42)

            f.writelines(["1\n", "2\n"])
            f.writelines(("3\n", "4\n"))
            f.writelines({'5\n': None})
            f.writelines({})

            # Try a big chunk too.
            class Iterator:
                def __init__(self, start, finish):
                    self.start = start
                    self.finish = finish
                    self.i = self.start

                def __next__(self):
                    if self.i >= self.finish:
                        raise StopIteration
                    result = str(self.i) + '\n'
                    self.i += 1
                    return result

                def __iter__(self):
                    return self

            class Whatever:
                def __init__(self, start, finish):
                    self.start = start
                    self.finish = finish

                def __iter__(self):
                    return Iterator(self.start, self.finish)

            f.writelines(Whatever(6, 6+2000))
            f.close()

            f = open(TESTFN, encoding="utf-8")
            expected = [str(i) + "\n" for i in range(1, 2006)]
            self.assertEqual(list(f), expected)

        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass


    # Test iterators on RHS of unpacking assignments.
    def test_unpack_iter(self):
        a, b = 1, 2
        self.assertEqual((a, b), (1, 2))

        a, b, c = IteratingSequenceClass(3)
        self.assertEqual((a, b, c), (0, 1, 2))

        try:    # too many values
            a, b = IteratingSequenceClass(3)
        except ValueError:
            pass
        else:
            self.fail("should have raised ValueError")

        try:    # not enough values
            a, b, c = IteratingSequenceClass(2)
        except ValueError:
            pass
        else:
            self.fail("should have raised ValueError")

        try:    # not iterable
            a, b, c = len
        except TypeError:
            pass
        else:
            self.fail("should have raised TypeError")

        a, b, c = {1: 42, 2: 42, 3: 42}.values()
        self.assertEqual((a, b, c), (42, 42, 42))

        f = open(TESTFN, "w", encoding="utf-8")
        lines = ("a\n", "bb\n", "ccc\n")
        try:
            for line in lines:
                f.write(line)
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            a, b, c = f
            self.assertEqual((a, b, c), lines)
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

        (a, b), (c,) = IteratingSequenceClass(2), {42: 24}
        self.assertEqual((a, b, c), (0, 1, 42))


    @cpython_only
    def test_ref_counting_behavior(self):
        class C(object):
            count = 0
            def __new__(cls):
                cls.count += 1
                return object.__new__(cls)
            def __del__(self):
                cls = self.__class__
                assert cls.count > 0
                cls.count -= 1
        x = C()
        self.assertEqual(C.count, 1)
        del x
        self.assertEqual(C.count, 0)
        l = [C(), C(), C()]
        self.assertEqual(C.count, 3)
        try:
            a, b = iter(l)
        except ValueError:
            pass
        del l
        self.assertEqual(C.count, 0)


    # Make sure StopIteration is a "sink state".
    # This tests various things that weren't sink states in Python 2.2.1,
    # plus various things that always were fine.

    def test_sinkstate_list(self):
        # This used to fail
        a = list(range(5))
        b = iter(a)
        self.assertEqual(list(b), list(range(5)))
        a.extend(range(5, 10))
        self.assertEqual(list(b), [])

    def test_sinkstate_tuple(self):
        a = (0, 1, 2, 3, 4)
        b = iter(a)
        self.assertEqual(list(b), list(range(5)))
        self.assertEqual(list(b), [])

    def test_sinkstate_string(self):
        a = "abcde"
        b = iter(a)
        self.assertEqual(list(b), ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(list(b), [])

    def test_sinkstate_sequence(self):
        # This used to fail
        a = SequenceClass(5)
        b = iter(a)
        self.assertEqual(list(b), list(range(5)))
        a.n = 10
        self.assertEqual(list(b), [])

    def test_sinkstate_callable(self):
        # This used to fail
        def spam(state=[0]):
            i = state[0]
            state[0] = i+1
            if i == 10:
                raise AssertionError("shouldn't have gotten this far")
            return i
        b = iter(spam, 5)
        self.assertEqual(list(b), list(range(5)))
        self.assertEqual(list(b), [])

    def test_sinkstate_dict(self):
        # XXX For a more thorough test, see towards the end of:
        # http://mail.python.org/pipermail/python-dev/2002-July/026512.html
        a = {1:1, 2:2, 0:0, 4:4, 3:3}
        for b in iter(a), a.keys(), a.items(), a.values():
            b = iter(a)
            self.assertEqual(len(list(b)), 5)
            self.assertEqual(list(b), [])

    def test_sinkstate_yield(self):
        def gen():
            for i in range(5):
                yield i
        b = gen()
        self.assertEqual(list(b), list(range(5)))
        self.assertEqual(list(b), [])

    def test_sinkstate_range(self):
        a = range(5)
        b = iter(a)
        self.assertEqual(list(b), list(range(5)))
        self.assertEqual(list(b), [])

    def test_sinkstate_enumerate(self):
        a = range(5)
        e = enumerate(a)
        b = iter(e)
        self.assertEqual(list(b), list(zip(range(5), range(5))))
        self.assertEqual(list(b), [])

    def test_3720(self):
        # Avoid a crash, when an iterator deletes its next() method.
        class BadIterator(object):
            def __iter__(self):
                return self
            def __next__(self):
                del BadIterator.__next__
                return 1

        try:
            for i in BadIterator() :
                pass
        except TypeError:
            pass

    def test_extending_list_with_iterator_does_not_segfault(self):
        # The code to extend a list with an iterator has a fair
        # amount of nontrivial logic in terms of guessing how
        # much memory to allocate in advance, "stealing" refs,
        # and then shrinking at the end.  This is a basic smoke
        # test for that scenario.
        def gen():
            for i in range(500):
                yield i
        lst = [0] * 500
        for i in range(240):
            lst.pop(0)
        lst.extend(gen())
        self.assertEqual(len(lst), 760)

    @cpython_only
    def test_iter_overflow(self):
        # Test for the issue 22939
        it = iter(UnlimitedSequenceClass())
        # Manually set `it_index` to PY_SSIZE_T_MAX-2 without a loop
        it.__setstate__(sys.maxsize - 2)
        self.assertEqual(next(it), sys.maxsize - 2)
        self.assertEqual(next(it), sys.maxsize - 1)
        with self.assertRaises(OverflowError):
            next(it)
        # Check that Overflow error is always raised
        with self.assertRaises(OverflowError):
            next(it)

    def test_iter_neg_setstate(self):
        it = iter(UnlimitedSequenceClass())
        it.__setstate__(-42)
        self.assertEqual(next(it), 0)
        self.assertEqual(next(it), 1)

    def test_free_after_iterating(self):
        check_free_after_iterating(self, iter, SequenceClass, (0,))

    def test_error_iter(self):
        for typ in (DefaultIterClass, NoIterClass):
            self.assertRaises(TypeError, iter, typ())
        self.assertRaises(ZeroDivisionError, iter, BadIterableClass())


if __name__ == "__main__":
    if TEST_WITH_TORCHDYNAMO:
        run_tests()
    else:
        unittest.main()
