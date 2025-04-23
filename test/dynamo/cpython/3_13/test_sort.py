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

from test import support
import random
import unittest
from functools import cmp_to_key

verbose = support.verbose
nerrors = 0


def check(tag, expected, raw, compare=None):
    global nerrors

    if verbose:
        print("    checking", tag)

    orig = raw[:]   # save input in case of error
    if compare:
        raw.sort(key=cmp_to_key(compare))
    else:
        raw.sort()

    if len(expected) != len(raw):
        print("error in", tag)
        print("length mismatch;", len(expected), len(raw))
        print(expected)
        print(orig)
        print(raw)
        nerrors += 1
        return

    for i, good in enumerate(expected):
        maybe = raw[i]
        if good is not maybe:
            print("error in", tag)
            print("out of order at index", i, good, maybe)
            print(expected)
            print(orig)
            print(raw)
            nerrors += 1
            return

class TestBase(__TestCase):
    def testStressfully(self):
        # Try a variety of sizes at and around powers of 2, and at powers of 10.
        sizes = [0]
        for power in range(1, 10):
            n = 2 ** power
            sizes.extend(range(n-1, n+2))
        sizes.extend([10, 100, 1000])

        class Complains(object):
            maybe_complain = True

            def __init__(self, i):
                self.i = i

            def __lt__(self, other):
                if Complains.maybe_complain and random.random() < 0.001:
                    if verbose:
                        print("        complaining at", self, other)
                    raise RuntimeError
                return self.i < other.i

            def __repr__(self):
                return "Complains(%d)" % self.i

        class Stable(object):
            def __init__(self, key, i):
                self.key = key
                self.index = i

            def __lt__(self, other):
                return self.key < other.key

            def __repr__(self):
                return "Stable(%d, %d)" % (self.key, self.index)

        for n in sizes:
            x = list(range(n))
            if verbose:
                print("Testing size", n)

            s = x[:]
            check("identity", x, s)

            s = x[:]
            s.reverse()
            check("reversed", x, s)

            s = x[:]
            random.shuffle(s)
            check("random permutation", x, s)

            y = x[:]
            y.reverse()
            s = x[:]
            check("reversed via function", y, s, lambda a, b: (b>a)-(b<a))

            if verbose:
                print("    Checking against an insane comparison function.")
                print("        If the implementation isn't careful, this may segfault.")
            s = x[:]
            s.sort(key=cmp_to_key(lambda a, b:  int(random.random() * 3) - 1))
            check("an insane function left some permutation", x, s)

            if len(x) >= 2:
                def bad_key(x):
                    raise RuntimeError
                s = x[:]
                self.assertRaises(RuntimeError, s.sort, key=bad_key)

            x = [Complains(i) for i in x]
            s = x[:]
            random.shuffle(s)
            Complains.maybe_complain = True
            it_complained = False
            try:
                s.sort()
            except RuntimeError:
                it_complained = True
            if it_complained:
                Complains.maybe_complain = False
                check("exception during sort left some permutation", x, s)

            s = [Stable(random.randrange(10), i) for i in range(n)]
            augmented = [(e, e.index) for e in s]
            augmented.sort()    # forced stable because ties broken by index
            x = [e for e, i in augmented] # a stable sort of s
            check("stability", x, s)

    def test_small_stability(self):
        from itertools import product
        from operator import itemgetter

        # Exhaustively test stability across all lists of small lengths
        # and only a few distinct elements.
        # This can provoke edge cases that randomization is unlikely to find.
        # But it can grow very expensive quickly, so don't overdo it.
        NELTS = 3
        MAXSIZE = 9

        pick0 = itemgetter(0)
        for length in range(MAXSIZE + 1):
            # There are NELTS ** length distinct lists.
            for t in product(range(NELTS), repeat=length):
                xs = list(zip(t, range(length)))
                # Stability forced by index in each element.
                forced = sorted(xs)
                # Use key= to hide the index from compares.
                native = sorted(xs, key=pick0)
                self.assertEqual(forced, native)
#==============================================================================

class TestBugs(__TestCase):

    def test_bug453523(self):
        # bug 453523 -- list.sort() crasher.
        # If this fails, the most likely outcome is a core dump.
        # Mutations during a list sort should raise a ValueError.

        class C:
            def __lt__(self, other):
                if L and random.random() < 0.75:
                    L.pop()
                else:
                    L.append(3)
                return random.random() < 0.5

        L = [C() for i in range(50)]
        self.assertRaises(ValueError, L.sort)

    def test_undetected_mutation(self):
        # Python 2.4a1 did not always detect mutation
        memorywaster = []
        for i in range(20):
            def mutating_cmp(x, y):
                L.append(3)
                L.pop()
                return (x > y) - (x < y)
            L = [1,2]
            self.assertRaises(ValueError, L.sort, key=cmp_to_key(mutating_cmp))
            def mutating_cmp(x, y):
                L.append(3)
                del L[:]
                return (x > y) - (x < y)
            self.assertRaises(ValueError, L.sort, key=cmp_to_key(mutating_cmp))
            memorywaster = [memorywaster]

#==============================================================================

class TestDecorateSortUndecorate(__TestCase):

    def test_decorated(self):
        data = 'The quick Brown fox Jumped over The lazy Dog'.split()
        copy = data[:]
        random.shuffle(data)
        data.sort(key=str.lower)
        def my_cmp(x, y):
            xlower, ylower = x.lower(), y.lower()
            return (xlower > ylower) - (xlower < ylower)
        copy.sort(key=cmp_to_key(my_cmp))

    def test_baddecorator(self):
        data = 'The quick Brown fox Jumped over The lazy Dog'.split()
        self.assertRaises(TypeError, data.sort, key=lambda x,y: 0)

    def test_stability(self):
        data = [(random.randrange(100), i) for i in range(200)]
        copy = data[:]
        data.sort(key=lambda t: t[0])   # sort on the random first field
        copy.sort()                     # sort using both fields
        self.assertEqual(data, copy)    # should get the same result

    def test_key_with_exception(self):
        # Verify that the wrapper has been removed
        data = list(range(-2, 2))
        dup = data[:]
        self.assertRaises(ZeroDivisionError, data.sort, key=lambda x: 1/x)
        self.assertEqual(data, dup)

    def test_key_with_mutation(self):
        data = list(range(10))
        def k(x):
            del data[:]
            data[:] = range(20)
            return x
        self.assertRaises(ValueError, data.sort, key=k)

    def test_key_with_mutating_del(self):
        data = list(range(10))
        class SortKiller(object):
            def __init__(self, x):
                pass
            def __del__(self):
                del data[:]
                data[:] = range(20)
            def __lt__(self, other):
                return id(self) < id(other)
        self.assertRaises(ValueError, data.sort, key=SortKiller)

    def test_key_with_mutating_del_and_exception(self):
        data = list(range(10))
        ## dup = data[:]
        class SortKiller(object):
            def __init__(self, x):
                if x > 2:
                    raise RuntimeError
            def __del__(self):
                del data[:]
                data[:] = list(range(20))
        self.assertRaises(RuntimeError, data.sort, key=SortKiller)
        ## major honking subtlety: we *can't* do:
        ##
        ## self.assertEqual(data, dup)
        ##
        ## because there is a reference to a SortKiller in the
        ## traceback and by the time it dies we're outside the call to
        ## .sort() and so the list protection gimmicks are out of
        ## date (this cost some brain cells to figure out...).

    def test_reverse(self):
        data = list(range(100))
        random.shuffle(data)
        data.sort(reverse=True)
        self.assertEqual(data, list(range(99,-1,-1)))

    def test_reverse_stability(self):
        data = [(random.randrange(100), i) for i in range(200)]
        copy1 = data[:]
        copy2 = data[:]
        def my_cmp(x, y):
            x0, y0 = x[0], y[0]
            return (x0 > y0) - (x0 < y0)
        def my_cmp_reversed(x, y):
            x0, y0 = x[0], y[0]
            return (y0 > x0) - (y0 < x0)
        data.sort(key=cmp_to_key(my_cmp), reverse=True)
        copy1.sort(key=cmp_to_key(my_cmp_reversed))
        self.assertEqual(data, copy1)
        copy2.sort(key=lambda x: x[0], reverse=True)
        self.assertEqual(data, copy2)

#==============================================================================
def check_against_PyObject_RichCompareBool(self, L):
    ## The idea here is to exploit the fact that unsafe_tuple_compare uses
    ## PyObject_RichCompareBool for the second elements of tuples. So we have,
    ## for (most) L, sorted(L) == [y[1] for y in sorted([(0,x) for x in L])]
    ## This will work as long as __eq__ => not __lt__ for all the objects in L,
    ## which holds for all the types used below.
    ##
    ## Testing this way ensures that the optimized implementation remains consistent
    ## with the naive implementation, even if changes are made to any of the
    ## richcompares.
    ##
    ## This function tests sorting for three lists (it randomly shuffles each one):
    ##                        1. L
    ##                        2. [(x,) for x in L]
    ##                        3. [((x,),) for x in L]

    random.seed(0)
    random.shuffle(L)
    L_1 = L[:]
    L_2 = [(x,) for x in L]
    L_3 = [((x,),) for x in L]
    for L in [L_1, L_2, L_3]:
        optimized = sorted(L)
        reference = [y[1] for y in sorted([(0,x) for x in L])]
        for (opt, ref) in zip(optimized, reference):
            self.assertIs(opt, ref)
            #note: not assertEqual! We want to ensure *identical* behavior.

class TestOptimizedCompares(__TestCase):
    def test_safe_object_compare(self):
        heterogeneous_lists = [[0, 'foo'],
                               [0.0, 'foo'],
                               [('foo',), 'foo']]
        for L in heterogeneous_lists:
            self.assertRaises(TypeError, L.sort)
            self.assertRaises(TypeError, [(x,) for x in L].sort)
            self.assertRaises(TypeError, [((x,),) for x in L].sort)

        float_int_lists = [[1,1.1],
                           [1<<70,1.1],
                           [1.1,1],
                           [1.1,1<<70]]
        for L in float_int_lists:
            check_against_PyObject_RichCompareBool(self, L)

    def test_unsafe_object_compare(self):

        # This test is by ppperry. It ensures that unsafe_object_compare is
        # verifying ms->key_richcompare == tp->richcompare before comparing.

        class WackyComparator(int):
            def __lt__(self, other):
                elem.__class__ = WackyList2
                return int.__lt__(self, other)

        class WackyList1(list):
            pass

        class WackyList2(list):
            def __lt__(self, other):
                raise ValueError

        L = [WackyList1([WackyComparator(i), i]) for i in range(10)]
        elem = L[-1]
        with self.assertRaises(ValueError):
            L.sort()

        L = [WackyList1([WackyComparator(i), i]) for i in range(10)]
        elem = L[-1]
        with self.assertRaises(ValueError):
            [(x,) for x in L].sort()

        # The following test is also by ppperry. It ensures that
        # unsafe_object_compare handles Py_NotImplemented appropriately.
        class PointlessComparator:
            def __lt__(self, other):
                return NotImplemented
        L = [PointlessComparator(), PointlessComparator()]
        self.assertRaises(TypeError, L.sort)
        self.assertRaises(TypeError, [(x,) for x in L].sort)

        # The following tests go through various types that would trigger
        # ms->key_compare = unsafe_object_compare
        lists = [list(range(100)) + [(1<<70)],
                 [str(x) for x in range(100)] + ['\uffff'],
                 [bytes(x) for x in range(100)],
                 [cmp_to_key(lambda x,y: x<y)(x) for x in range(100)]]
        for L in lists:
            check_against_PyObject_RichCompareBool(self, L)

    def test_unsafe_latin_compare(self):
        check_against_PyObject_RichCompareBool(self, [str(x) for
                                                      x in range(100)])

    def test_unsafe_long_compare(self):
        check_against_PyObject_RichCompareBool(self, [x for
                                                      x in range(100)])

    def test_unsafe_float_compare(self):
        check_against_PyObject_RichCompareBool(self, [float(x) for
                                                      x in range(100)])

    def test_unsafe_tuple_compare(self):
        # This test was suggested by Tim Peters. It verifies that the tuple
        # comparison respects the current tuple compare semantics, which do not
        # guarantee that x < x <=> (x,) < (x,)
        #
        # Note that we don't have to put anything in tuples here, because
        # the check function does a tuple test automatically.

        check_against_PyObject_RichCompareBool(self, [float('nan')]*100)
        check_against_PyObject_RichCompareBool(self, [float('nan') for
                                                      _ in range(100)])

    def test_not_all_tuples(self):
        self.assertRaises(TypeError, [(1.0, 1.0), (False, "A"), 6].sort)
        self.assertRaises(TypeError, [('a', 1), (1, 'a')].sort)
        self.assertRaises(TypeError, [(1, 'a'), ('a', 1)].sort)

    def test_none_in_tuples(self):
        expected = [(None, 1), (None, 2)]
        actual = sorted([(None, 2), (None, 1)])
        self.assertEqual(actual, expected)

#==============================================================================

if __name__ == "__main__":
    if TEST_WITH_TORCHDYNAMO:
        run_tests()
    else:
        unittest.main()
