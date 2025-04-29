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

__TestCase = CPythonTestCase
# if TEST_WITH_TORCHDYNAMO:
#     __TestCase = CPythonTestCase
# else:
#     __TestCase = unittest.TestCase

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
import seq_tests
import unittest

import gc
import pickle

# For tuple hashes, we normally only run a test to ensure that we get
# the same results across platforms in a handful of cases.  If that's
# so, there's no real point to running more.  Set RUN_ALL_HASH_TESTS to
# run more anyway.  That's usually of real interest only when analyzing,
# or changing, the hash algorithm.  In which case it's usually also
# most useful to set JUST_SHOW_HASH_RESULTS, to see all the results
# instead of wrestling with test "failures".  See the bottom of the
# file for extensive notes on what we're testing here and why.
RUN_ALL_HASH_TESTS = False
JUST_SHOW_HASH_RESULTS = False # if RUN_ALL_HASH_TESTS, just display

class TupleTest(seq_tests.CommonTest):
    type2test = tuple

    def test_getitem_error(self):
        t = ()
        msg = "tuple indices must be integers or slices"
        with self.assertRaisesRegex(TypeError, msg):
            t['a']

    def test_constructors(self):
        super().test_constructors()
        # calling built-in types without argument must return empty
        self.assertEqual(tuple(), ())
        t0_3 = (0, 1, 2, 3)
        t0_3_bis = tuple(t0_3)
        self.assertTrue(t0_3 is t0_3_bis)
        self.assertEqual(tuple([]), ())
        self.assertEqual(tuple([0, 1, 2, 3]), (0, 1, 2, 3))
        self.assertEqual(tuple(''), ())
        self.assertEqual(tuple('spam'), ('s', 'p', 'a', 'm'))
        self.assertEqual(tuple(x for x in range(10) if x % 2),
                         (1, 3, 5, 7, 9))

    def test_keyword_args(self):
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            tuple(sequence=())

    def test_keywords_in_subclass(self):
        class subclass(tuple):
            pass
        u = subclass([1, 2])
        self.assertIs(type(u), subclass)
        self.assertEqual(list(u), [1, 2])
        with self.assertRaises(TypeError):
            subclass(sequence=())

        class subclass_with_init(tuple):
            def __init__(self, arg, newarg=None):
                self.newarg = newarg
        u = subclass_with_init([1, 2], newarg=3)
        self.assertIs(type(u), subclass_with_init)
        self.assertEqual(list(u), [1, 2])
        self.assertEqual(u.newarg, 3)

        class subclass_with_new(tuple):
            def __new__(cls, arg, newarg=None):
                self = super().__new__(cls, arg)
                self.newarg = newarg
                return self
        u = subclass_with_new([1, 2], newarg=3)
        self.assertIs(type(u), subclass_with_new)
        self.assertEqual(list(u), [1, 2])
        self.assertEqual(u.newarg, 3)

    def test_truth(self):
        super().test_truth()
        self.assertTrue(not ())
        self.assertTrue((42, ))

    def test_len(self):
        super().test_len()
        self.assertEqual(len(()), 0)
        self.assertEqual(len((0,)), 1)
        self.assertEqual(len((0, 1, 2)), 3)

    def test_iadd(self):
        super().test_iadd()
        u = (0, 1)
        u2 = u
        u += (2, 3)
        self.assertTrue(u is not u2)

    def test_imul(self):
        super().test_imul()
        u = (0, 1)
        u2 = u
        u *= 3
        self.assertTrue(u is not u2)

    def test_tupleresizebug(self):
        # Check that a specific bug in _PyTuple_Resize() is squashed.
        def f():
            for i in range(1000):
                yield i
        self.assertEqual(list(tuple(f())), list(range(1000)))

    # We expect tuples whose base components have deterministic hashes to
    # have deterministic hashes too - and, indeed, the same hashes across
    # platforms with hash codes of the same bit width.
    def test_hash_exact(self):
        def check_one_exact(t, e32, e64):
            got = hash(t)
            expected = e32 if support.NHASHBITS == 32 else e64
            if got != expected:
                msg = f"FAIL hash({t!r}) == {got} != {expected}"
                self.fail(msg)

        check_one_exact((), 750394483, 5740354900026072187)
        check_one_exact((0,), 1214856301, -8753497827991233192)
        check_one_exact((0, 0), -168982784, -8458139203682520985)
        check_one_exact((0.5,), 2077348973, -408149959306781352)
        check_one_exact((0.5, (), (-2, 3, (4, 6))), 714642271,
                        -1845940830829704396)

    # Various tests for hashing of tuples to check that we get few collisions.
    # Does something only if RUN_ALL_HASH_TESTS is true.
    #
    # Earlier versions of the tuple hash algorithm had massive collisions
    # reported at:
    # - https://bugs.python.org/issue942952
    # - https://bugs.python.org/issue34751
    def test_hash_optional(self):
        from itertools import product

        if not RUN_ALL_HASH_TESTS:
            return

        # If specified, `expected` is a 2-tuple of expected
        # (number_of_collisions, pileup) values, and the test fails if
        # those aren't the values we get.  Also if specified, the test
        # fails if z > `zlimit`.
        def tryone_inner(tag, nbins, hashes, expected=None, zlimit=None):
            from collections import Counter

            nballs = len(hashes)
            mean, sdev = support.collision_stats(nbins, nballs)
            c = Counter(hashes)
            collisions = nballs - len(c)
            z = (collisions - mean) / sdev
            pileup = max(c.values()) - 1
            del c
            got = (collisions, pileup)
            failed = False
            prefix = ""
            if zlimit is not None and z > zlimit:
                failed = True
                prefix = f"FAIL z > {zlimit}; "
            if expected is not None and got != expected:
                failed = True
                prefix += f"FAIL {got} != {expected}; "
            if failed or JUST_SHOW_HASH_RESULTS:
                msg = f"{prefix}{tag}; pileup {pileup:,} mean {mean:.1f} "
                msg += f"coll {collisions:,} z {z:+.1f}"
                if JUST_SHOW_HASH_RESULTS:
                    import sys
                    print(msg, file=sys.__stdout__)
                else:
                    self.fail(msg)

        def tryone(tag, xs,
                   native32=None, native64=None, hi32=None, lo32=None,
                   zlimit=None):
            NHASHBITS = support.NHASHBITS
            hashes = list(map(hash, xs))
            tryone_inner(tag + f"; {NHASHBITS}-bit hash codes",
                         1 << NHASHBITS,
                         hashes,
                         native32 if NHASHBITS == 32 else native64,
                         zlimit)

            if NHASHBITS > 32:
                shift = NHASHBITS - 32
                tryone_inner(tag + "; 32-bit upper hash codes",
                             1 << 32,
                             [h >> shift for h in hashes],
                             hi32,
                             zlimit)

                mask = (1 << 32) - 1
                tryone_inner(tag + "; 32-bit lower hash codes",
                             1 << 32,
                             [h & mask for h in hashes],
                             lo32,
                             zlimit)

        # Tuples of smallish positive integers are common - nice if we
        # get "better than random" for these.
        tryone("range(100) by 3", list(product(range(100), repeat=3)),
               (0, 0), (0, 0), (4, 1), (0, 0))

        # A previous hash had systematic problems when mixing integers of
        # similar magnitude but opposite sign, obscurely related to that
        # j ^ -2 == -j when j is odd.
        cands = list(range(-10, -1)) + list(range(9))

        # Note:  -1 is omitted because hash(-1) == hash(-2) == -2, and
        # there's nothing the tuple hash can do to avoid collisions
        # inherited from collisions in the tuple components' hashes.
        tryone("-10 .. 8 by 4", list(product(cands, repeat=4)),
               (0, 0), (0, 0), (0, 0), (0, 0))
        del cands

        # The hashes here are a weird mix of values where all the
        # variation is in the lowest bits and across a single high-order
        # bit - the middle bits are all zeroes. A decent hash has to
        # both propagate low bits to the left and high bits to the
        # right.  This is also complicated a bit in that there are
        # collisions among the hashes of the integers in L alone.
        L = [n << 60 for n in range(100)]
        tryone("0..99 << 60 by 3", list(product(L, repeat=3)),
               (0, 0), (0, 0), (0, 0), (324, 1))
        del L

        # Used to suffer a massive number of collisions.
        tryone("[-3, 3] by 18", list(product([-3, 3], repeat=18)),
               (7, 1), (0, 0), (7, 1), (6, 1))

        # And even worse.  hash(0.5) has only a single bit set, at the
        # high end. A decent hash needs to propagate high bits right.
        tryone("[0, 0.5] by 18", list(product([0, 0.5], repeat=18)),
               (5, 1), (0, 0), (9, 1), (12, 1))

        # Hashes of ints and floats are the same across platforms.
        # String hashes vary even on a single platform across runs, due
        # to hash randomization for strings.  So we can't say exactly
        # what this should do.  Instead we insist that the # of
        # collisions is no more than 4 sdevs above the theoretically
        # random mean.  Even if the tuple hash can't achieve that on its
        # own, the string hash is trying to be decently pseudo-random
        # (in all bit positions) on _its_ own.  We can at least test
        # that the tuple hash doesn't systematically ruin that.
        tryone("4-char tuples",
               list(product("abcdefghijklmnopqrstuvwxyz", repeat=4)),
               zlimit=4.0)

        # The "old tuple test".  See https://bugs.python.org/issue942952.
        # Ensures, for example, that the hash:
        #   is non-commutative
        #   spreads closely spaced values
        #   doesn't exhibit cancellation in tuples like (x,(x,y))
        N = 50
        base = list(range(N))
        xp = list(product(base, repeat=2))
        inps = base + list(product(base, xp)) + \
                     list(product(xp, base)) + xp + list(zip(base))
        tryone("old tuple test", inps,
               (2, 1), (0, 0), (52, 49), (7, 1))
        del base, xp, inps

        # The "new tuple test".  See https://bugs.python.org/issue34751.
        # Even more tortured nesting, and a mix of signed ints of very
        # small magnitude.
        n = 5
        A = [x for x in range(-n, n+1) if x != -1]
        B = A + [(a,) for a in A]
        L2 = list(product(A, repeat=2))
        L3 = L2 + list(product(A, repeat=3))
        L4 = L3 + list(product(A, repeat=4))
        # T = list of testcases. These consist of all (possibly nested
        # at most 2 levels deep) tuples containing at most 4 items from
        # the set A.
        T = A
        T += [(a,) for a in B + L4]
        T += product(L3, B)
        T += product(L2, repeat=2)
        T += product(B, L3)
        T += product(B, B, L2)
        T += product(B, L2, B)
        T += product(L2, B, B)
        T += product(B, repeat=4)
        assert len(T) == 345130
        tryone("new tuple test", T,
               (9, 1), (0, 0), (21, 5), (6, 1))

    def test_repr(self):
        l0 = tuple()
        l2 = (0, 1, 2)
        a0 = self.type2test(l0)
        a2 = self.type2test(l2)

        self.assertEqual(str(a0), repr(l0))
        self.assertEqual(str(a2), repr(l2))
        self.assertEqual(repr(a0), "()")
        self.assertEqual(repr(a2), "(0, 1, 2)")

    def _not_tracked(self, t):
        # Nested tuples can take several collections to untrack
        gc.collect()
        gc.collect()
        self.assertFalse(gc.is_tracked(t), t)

    def _tracked(self, t):
        self.assertTrue(gc.is_tracked(t), t)
        gc.collect()
        gc.collect()
        self.assertTrue(gc.is_tracked(t), t)

    @support.cpython_only
    def test_track_literals(self):
        # Test GC-optimization of tuple literals
        x, y, z = 1.5, "a", []

        self._not_tracked(())
        self._not_tracked((1,))
        self._not_tracked((1, 2))
        self._not_tracked((1, 2, "a"))
        self._not_tracked((1, 2, (None, True, False, ()), int))
        self._not_tracked((object(),))
        self._not_tracked(((1, x), y, (2, 3)))

        # Tuples with mutable elements are always tracked, even if those
        # elements are not tracked right now.
        self._tracked(([],))
        self._tracked(([1],))
        self._tracked(({},))
        self._tracked((set(),))
        self._tracked((x, y, z))

    def check_track_dynamic(self, tp, always_track):
        x, y, z = 1.5, "a", []

        check = self._tracked if always_track else self._not_tracked
        check(tp())
        check(tp([]))
        check(tp(set()))
        check(tp([1, x, y]))
        check(tp(obj for obj in [1, x, y]))
        check(tp(set([1, x, y])))
        check(tp(tuple([obj]) for obj in [1, x, y]))
        check(tuple(tp([obj]) for obj in [1, x, y]))

        self._tracked(tp([z]))
        self._tracked(tp([[x, y]]))
        self._tracked(tp([{x: y}]))
        self._tracked(tp(obj for obj in [x, y, z]))
        self._tracked(tp(tuple([obj]) for obj in [x, y, z]))
        self._tracked(tuple(tp([obj]) for obj in [x, y, z]))

    @support.cpython_only
    def test_track_dynamic(self):
        # Test GC-optimization of dynamically constructed tuples.
        self.check_track_dynamic(tuple, False)

    @support.cpython_only
    def test_track_subtypes(self):
        # Tuple subtypes must always be tracked
        class MyTuple(tuple):
            pass
        self.check_track_dynamic(MyTuple, True)

    @support.cpython_only
    def test_bug7466(self):
        # Trying to untrack an unfinished tuple could crash Python
        self._not_tracked(tuple(gc.collect() for i in range(101)))

    def test_repr_large(self):
        # Check the repr of large list objects
        def check(n):
            l = (0,) * n
            s = repr(l)
            self.assertEqual(s,
                '(' + ', '.join(['0'] * n) + ')')
        check(10)       # check our checking code
        check(1000000)

    def test_iterator_pickle(self):
        # Userlist iterators don't support pickling yet since
        # they are based on generators.
        data = self.type2test([4, 5, 6, 7])
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            itorg = iter(data)
            d = pickle.dumps(itorg, proto)
            it = pickle.loads(d)
            self.assertEqual(type(itorg), type(it))
            self.assertEqual(self.type2test(it), self.type2test(data))

            it = pickle.loads(d)
            next(it)
            d = pickle.dumps(it, proto)
            self.assertEqual(self.type2test(it), self.type2test(data)[1:])

    def test_reversed_pickle(self):
        data = self.type2test([4, 5, 6, 7])
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            itorg = reversed(data)
            d = pickle.dumps(itorg, proto)
            it = pickle.loads(d)
            self.assertEqual(type(itorg), type(it))
            self.assertEqual(self.type2test(it), self.type2test(reversed(data)))

            it = pickle.loads(d)
            next(it)
            d = pickle.dumps(it, proto)
            self.assertEqual(self.type2test(it), self.type2test(reversed(data))[1:])

    def test_no_comdat_folding(self):
        # Issue 8847: In the PGO build, the MSVC linker's COMDAT folding
        # optimization causes failures in code that relies on distinct
        # function addresses.
        class T(tuple): pass
        with self.assertRaises(TypeError):
            [3,] + T((1,2))

    def test_lexicographic_ordering(self):
        # Issue 21100
        a = self.type2test([1, 2])
        b = self.type2test([1, 2, 0])
        c = self.type2test([1, 3])
        self.assertLess(a, b)
        self.assertLess(b, c)

# Notes on testing hash codes.  The primary thing is that Python doesn't
# care about "random" hash codes.  To the contrary, we like them to be
# very regular when possible, so that the low-order bits are as evenly
# distributed as possible.  For integers this is easy: hash(i) == i for
# all not-huge i except i==-1.
#
# For tuples of mixed type there's really no hope of that, so we want
# "randomish" here instead.  But getting close to pseudo-random in all
# bit positions is more expensive than we've been willing to pay for.
#
# We can tolerate large deviations from random - what we don't want is
# catastrophic pileups on a relative handful of hash codes.  The dict
# and set lookup routines remain effective provided that full-width hash
# codes for not-equal objects are distinct.
#
# So we compute various statistics here based on what a "truly random"
# hash would do, but don't automate "pass or fail" based on those
# results.  Instead those are viewed as inputs to human judgment, and the
# automated tests merely ensure we get the _same_ results across
# platforms.  In fact, we normally don't bother to run them at all -
# set RUN_ALL_HASH_TESTS to force it.
#
# When global JUST_SHOW_HASH_RESULTS is True, the tuple hash statistics
# are just displayed to stdout.  A typical output line looks like:
#
# old tuple test; 32-bit upper hash codes; \
#             pileup 49 mean 7.4 coll 52 z +16.4
#
# "old tuple test" is just a string name for the test being run.
#
# "32-bit upper hash codes" means this was run under a 64-bit build and
# we've shifted away the lower 32 bits of the hash codes.
#
# "pileup" is 0 if there were no collisions across those hash codes.
# It's 1 less than the maximum number of times any single hash code was
# seen.  So in this case, there was (at least) one hash code that was
# seen 50 times:  that hash code "piled up" 49 more times than ideal.
#
# "mean" is the number of collisions a perfectly random hash function
# would have yielded, on average.
#
# "coll" is the number of collisions actually seen.
#
# "z" is "coll - mean" divided by the standard deviation of the number
# of collisions a perfectly random hash function would suffer.  A
# positive value is "worse than random", and negative value "better than
# random".  Anything of magnitude greater than 3 would be highly suspect
# for a hash function that claimed to be random.  It's essentially
# impossible that a truly random function would deliver a result 16.4
# sdevs "worse than random".
#
# But we don't care here!  That's why the test isn't coded to fail.
# Knowing something about how the high-order hash code bits behave
# provides insight, but is irrelevant to how the dict and set lookup
# code performs.  The low-order bits are much more important to that,
# and on the same test those did "just like random":
#
# old tuple test; 32-bit lower hash codes; \
#            pileup 1 mean 7.4 coll 7 z -0.2
#
# So there are always tradeoffs to consider.  For another:
#
# 0..99 << 60 by 3; 32-bit hash codes; \
#            pileup 0 mean 116.4 coll 0 z -10.8
#
# That was run under a 32-bit build, and is spectacularly "better than
# random".  On a 64-bit build the wider hash codes are fine too:
#
# 0..99 << 60 by 3; 64-bit hash codes; \
#             pileup 0 mean 0.0 coll 0 z -0.0
#
# but their lower 32 bits are poor:
#
# 0..99 << 60 by 3; 32-bit lower hash codes; \
#             pileup 1 mean 116.4 coll 324 z +19.2
#
# In a statistical sense that's waaaaay too many collisions, but (a) 324
# collisions out of a million hash codes isn't anywhere near being a
# real problem; and, (b) the worst pileup on a single hash code is a measly
# 1 extra.  It's a relatively poor case for the tuple hash, but still
# fine for practical use.
#
# This isn't, which is what Python 3.7.1 produced for the hashes of
# itertools.product([0, 0.5], repeat=18).  Even with a fat 64-bit
# hashcode, the highest pileup was over 16,000 - making a dict/set
# lookup on one of the colliding values thousands of times slower (on
# average) than we expect.
#
# [0, 0.5] by 18; 64-bit hash codes; \
#            pileup 16,383 mean 0.0 coll 262,128 z +6073641856.9
# [0, 0.5] by 18; 32-bit lower hash codes; \
#            pileup 262,143 mean 8.0 coll 262,143 z +92683.6

if __name__ == "__main__":
    run_tests()
