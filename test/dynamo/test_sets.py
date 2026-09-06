# Owner(s): ["module: dynamo"]

# TODO: move set tests from test_functions.py/test_misc.py to this file

import math
import unittest
from collections.abc import Iterable

import torch
import torch._dynamo.test_case
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import make_dynamo_test, munge_exc
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


class SetSubclass(set):
    pass


class FrozenstSubclass(frozenset):
    pass


class BadCmp:
    # Elements collide on hash (so __eq__ runs during set insertion/lookup),
    # and __eq__ raises so the error must propagate.  Mirrors CPython
    # test_set.py BadCmp.
    def __hash__(self):
        return 1

    def __eq__(self, other):
        raise RuntimeError


class _BaseSetTests(torch._dynamo.test_case.TestCase):
    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    def assertEqual(self, a, b):
        return self.assertTrue(a == b, lambda msg: f"{msg}\n{a} != {b}")

    def assertNotEqual(self, a, b):
        return self.assertTrue(a != b, lambda msg: f"{msg}\n{a} == {b}")


class CustomSetTests(_BaseSetTests):
    class CustomSet(set):
        def add(self, item):
            return super().add(item + 1)

        def contains(self, item):
            return True

    thetype = CustomSet

    @make_dynamo_test
    def test_custom_add(self):
        s = self.thetype([1, 2])
        s.add(3)
        self.assertTrue(s == {1, 2, 4})

    @make_dynamo_test
    def test_custom_contains(self):
        s = self.thetype([1, 2])
        self.assertTrue(s.contains(3))


class MiscTests(torch._dynamo.test_case.TestCase):
    def test_isdisjoint_with_generator(self):
        n = 0

        def gen():
            nonlocal n
            n += 1
            yield 1
            n += 2
            yield 2
            n += 3
            yield 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            nonlocal n
            s = {2, 4, 5}
            s.isdisjoint(gen())
            if n == 3:
                return x.sin()
            return x.cos()

        x = torch.randn(1)
        y = fn(x)
        self.assertEqual(y, x.sin())

    def test_set_iterator_length_hint(self):
        # setiter_len/dictiter_len: __length_hint__ returns the number of
        # not-yet-consumed elements and decreases as the iterator advances.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            results = []
            for obj in ({1, 2, 3}, {1: "a", 2: "b"}, {1: "a"}.values()):
                it = iter(obj)
                results.append(it.__length_hint__())
                next(it)
                results.append(it.__length_hint__())
            results.append(iter(set()).__length_hint__())
            return x.sin(), results

        x = torch.randn(1)
        y, results = fn(x)
        self.assertEqual(y, x.sin())
        self.assertEqual(results, [3, 2, 2, 1, 1, 0, 0])

    def test_id_arithmetic_in_custom_hash_fullgraph(self):
        # id() of an object created inside the compiled region is a fake,
        # compile-time-only int. Bitwise/arithmetic ops on it (a common way to
        # derive a __hash__, e.g. CPython test_set.test_subclass_with_custom_hash)
        # must not graph break and must stay compile-time. Covers set,
        # frozenset, and subclass bases whose __hash__ masks id(self).
        class HSet(set):
            def __hash__(self):
                return int((id(self) & 0x7FFFFFFF) ^ 3)

        class HFrozen(frozenset):
            def __hash__(self):
                return int((id(self) & 0x7FFFFFFF) ^ 3)

        class HSubclass(SetSubclass):
            def __hash__(self):
                return int((id(self) & 0x7FFFFFFF) ^ 3)

        for cls in (HSet, HFrozen, HSubclass):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                s = cls()
                f = set()
                f.add(s)
                present = s in f
                f.discard(s)
                return x + 1, present

            res, present = fn(torch.zeros(1))
            self.assertEqual(res, torch.ones(1))
            self.assertTrue(present)

    def test_id_arithmetic_ops_stay_compile_time(self):
        # A spread of int arithmetic/bitwise/unary ops on a fake id() value
        # all stay compile-time-only and can serve as a hash without breaking.
        class H(set):
            def __hash__(self):
                i = id(self)
                return int((i & 0xFFFF) | (i >> 4) ^ (~i) + (i * 2) - (i // 3))

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            f = {H()}
            return x + 1, len(f)

        res, n = fn(torch.zeros(1))
        self.assertEqual(res, torch.ones(1))
        self.assertEqual(n, 1)

    def test_do_not_rehash_dict_keys(self):
        # Building a set/frozenset (or subclass) from a dict must reuse the
        # dict's stored hashes instead of re-invoking __hash__, mirroring
        # CPython's set_update_internal fast path.  Also covers the explicit
        # base-slot call `int.__hash__(self)` inside the custom __hash__.
        class HashCountingInt(int):
            def __init__(self, *args):
                self.hash_count = 0

            def __hash__(self):
                self.hash_count += 1
                return int.__hash__(self)

        def run(thetype, n):
            d = dict.fromkeys(map(HashCountingInt, range(n)))
            counts = [sum(e.hash_count for e in d)]
            s = thetype(d)
            counts.append(sum(e.hash_count for e in d))
            s.difference(d)
            counts.append(sum(e.hash_count for e in d))
            dict.fromkeys(set(d))
            counts.append(sum(e.hash_count for e in d))
            dict.fromkeys(frozenset(d))
            counts.append(sum(e.hash_count for e in d))
            return counts, len(s)

        for thetype in (set, frozenset, SetSubclass, FrozenstSubclass):
            n = 10
            ref = run(thetype, n)
            res = torch.compile(run, backend="eager", fullgraph=True)(thetype, n)
            self.assertEqual(ref, res)
            # Every key hashed exactly once (during the initial fromkeys).
            self.assertEqual(ref[0], [n] * 5)


class TestSetGuards(LoggingTestCase):
    def test_set_with_function(self):
        s = {
            torch._C._set_grad_enabled,
            "hello",
            torch.amp._exit_autocast,
        }
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x, s):
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(2)
        y = fn(x, s)
        self.assertEqual(y, x.sin())
        self.assertEqual(cnts.frame_count, 1)

        s.remove(torch.amp._exit_autocast)
        s.add(torch._C._set_fwd_grad_enabled)
        y = fn(x, s)
        self.assertEqual(y, x.cos())
        self.assertEqual(cnts.frame_count, 2)

    @make_logging_test(recompiles=True)
    def test_in_guard(self, records):
        s = {
            "Dynamo",
            "Inductor",
            "PyTorch",
            torch.sin,
        }
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x, s):
            if "PyTorch" in s:
                return x.sin()
            return x.cos()

        x = torch.randn(2)
        y = fn(x, s)
        self.assertEqual(y, x.sin())
        self.assertEqual(cnts.frame_count, 1)

        s.remove("PyTorch")
        s.add("Cuda")
        y = fn(x, s)
        self.assertEqual(y, x.cos())
        self.assertEqual(cnts.frame_count, 2)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "set.__contains__")
        self.assertIn(
            """set.__contains__(s, 'PyTorch')""",
            munge_exc(record.getMessage()),
        )

    def test_set_with_tensors(self):
        s = {
            torch.ones(1),
            torch.tensor([1.0]),
            torch.zeros(1),
        }
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x, s):
            z = torch.zeros(1)
            for i in s:
                z += i
            return x + z

        x = torch.tensor([1.0])
        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: fn(x, s),
            """\
Attempted to wrap a set with tensors
  Explanation: Dynamo cannot trace sets of tensors. To get a stable ordering, Dynamo needs to convert the set into a list and the order might not be stable if the set contains tensors.
  Hint: Use a dictionary where the keys are tensors.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: Python set containing torch.Tensor elements

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0222.html

from user code:
   File "test_sets.py", line N, in fn
    for i in s:""",
        )

    def test_set_multiple_types(self):
        s = {
            "PyTorch",
            3.3,
            1j,
            math.nan,
        }
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x, s):
            if "PyTorch" in s:
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        y = fn(x, s)
        self.assertEqual(y, x.sin())
        self.assertEqual(cnts.frame_count, 1)

        s.remove("PyTorch")
        y = fn(x, s)
        self.assertEqual(y, x.cos())
        self.assertEqual(cnts.frame_count, 2)

    def test_set_recompile_on_key_pop(self):
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = CompileCounter()

        def fn(x, s):
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, s)
        opt_fn(x, s)
        self.assertEqual(res, fn(x, s))
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # Pop a value
        s.remove(torch.amp._exit_autocast)

        res = opt_fn(x, s)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, s))

    def test_set_recompile_on_key_change(self):
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = CompileCounter()

        def fn(x, s):
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, s)
        opt_fn(x, s)
        self.assertEqual(res, fn(x, s))
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # Pop a value
        s.remove(torch.amp._exit_autocast)
        # Add a different value
        s.add(torch._C._set_autograd_fallback_mode)

        res = opt_fn(x, s)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, s))

    @unittest.skip("random failures on Python 3.9")
    def test_set_guard_on_keys_change(self):
        # This test guarantee that we're not triggering any of the dict guards
        # on sets
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = CompileCounter()

        def fn(x, s):
            for e in s:
                x = x * len(str(e))
            return x

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(torch.randn(4), s)
        opt_fn(torch.randn(4), s)
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # pop and add the same item
        s.remove(torch.amp._exit_autocast)
        # It is not guaranteed that _exit_autocast will be in a specific order
        s.add(torch.amp._exit_autocast)

        x = torch.randn(4)
        res = opt_fn(x, s)
        # Check Dynamo don't recompile
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, fn(x, s))


class _FrozensetBase:
    # Frozenset methods
    # + copy
    # + difference
    # + intersection
    # + isdisjoint
    # + issubset
    # + issuperset
    # + symmetric_difference
    # + union
    # BinOps:
    # +, -, |, &, ^, <, >, <=, >=, ==, !=

    @make_dynamo_test
    def test_binop_sub(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p - p, self.thetype())
        self.assertEqual(p - q, self.thetype("ac"))
        self.assertEqual(q - p, self.thetype("ef"))
        self.assertRaises(TypeError, lambda: p - 1)
        self.assertEqual(self.thetype.__sub__(p, q), set("ac"))

    @make_dynamo_test
    def test_binop_or(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p | p, self.thetype("abc"))
        self.assertEqual(p | q, self.thetype("abcef"))
        self.assertEqual(self.thetype.__or__(p, q), set("abcef"))

    @make_dynamo_test
    def test_binop_and(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p & p, self.thetype("abc"))
        self.assertEqual(p & q, self.thetype("b"))
        self.assertEqual(self.thetype.__and__(p, q), set("b"))

    @make_dynamo_test
    def test_binop_xor(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p ^ p, self.thetype())
        self.assertEqual(p ^ q, self.thetype("acef"))
        self.assertEqual(self.thetype.__xor__(p, q), set("acef"))

    @make_dynamo_test
    def test_badcmp(self):
        # A comparison error during insertion/lookup must propagate as the
        # user exception.  For frozenset types hasattr(s, "add") is False, so
        # the mutating block is skipped (regression: exact frozenset used to
        # report hasattr(set, "add")).  Mirrors CPython test_set.py test_badcmp.
        s = self.thetype([BadCmp()])
        self.assertRaises(RuntimeError, self.thetype, [BadCmp(), BadCmp()])
        self.assertRaises(RuntimeError, s.__contains__, BadCmp())
        if hasattr(s, "add"):
            self.assertRaises(RuntimeError, s.add, BadCmp())
            self.assertRaises(RuntimeError, s.discard, BadCmp())
            self.assertRaises(RuntimeError, s.remove, BadCmp())

    @make_dynamo_test
    def test_cmp_eq(self):
        p = self.thetype("abc")
        self.assertEqual(p, p)
        for C in set, frozenset, SetSubclass:
            self.assertEqual(p, C("abc"))
            self.assertEqual(p, C(p))
        self.assertTrue(self.thetype.__eq__(p, p))

    @make_dynamo_test
    def test_cmp_ne(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertNotEqual(p, q)
        self.assertNotEqual(q, p)
        for C in set, frozenset, SetSubclass, dict.fromkeys, str, list, tuple:
            self.assertNotEqual(p, C("abe"))
        self.assertNotEqual(p, 1)
        self.assertTrue(self.thetype.__ne__(p, q))

    @make_dynamo_test
    def test_cmp_less_than(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertFalse(p < p)
        self.assertFalse(p < q)
        self.assertTrue(r < p)
        self.assertFalse(r < q)
        self.assertFalse(self.thetype.__lt__(p, p))

    @make_dynamo_test
    def test_cmp_greater_than(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertFalse(p > p)
        self.assertFalse(p > q)
        self.assertTrue(p > r)
        self.assertFalse(q > r)
        self.assertFalse(self.thetype.__gt__(p, p))

    @make_dynamo_test
    def test_cmp_less_than_or_equal(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertTrue(p <= p)
        self.assertFalse(p <= q)
        self.assertTrue(r <= p)
        self.assertFalse(r <= q)
        self.assertTrue(self.thetype.__le__(p, p))

    @make_dynamo_test
    def test_cmp_greater_than_or_equal(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertTrue(p >= p)
        self.assertFalse(p >= q)
        self.assertTrue(p >= r)
        self.assertFalse(q >= r)
        self.assertTrue(self.thetype.__ge__(p, p))

    @make_dynamo_test
    def test_copy(self):
        p = self.thetype("abc")
        q = p.copy()
        self.assertEqual(p, q)
        self.assertRaises(TypeError, p.copy, 1)
        self.assertEqual(self.thetype.copy(p), p)

    @make_dynamo_test
    def test_issubset(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        self.assertTrue(q.issubset(p))
        self.assertFalse(r.issubset(p))
        self.assertRaises(TypeError, p.issubset)
        self.assertRaises(TypeError, p.issubset, 1)
        self.assertRaises(TypeError, p.issubset, [[]])
        self.assertTrue(self.thetype.issubset(q, p))

    @make_dynamo_test
    def test_issuperset(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        self.assertTrue(p.issuperset(q))
        self.assertFalse(p.issuperset(r))
        self.assertRaises(TypeError, p.issuperset)
        self.assertRaises(TypeError, p.issuperset, 1)
        self.assertRaises(TypeError, p.issuperset, [[]])
        self.assertTrue(self.thetype.issuperset(p, q))

    @make_dynamo_test
    def test_constructor_iterable(self):
        p = self.thetype("abc")
        self.assertIsInstance(p, self.thetype)
        self.assertIsInstance(p, Iterable)

    @make_dynamo_test
    def test_new_or_init(self):
        # set/frozenset constructors reject extra positional args and any
        # keyword arguments; set().__init__ rejects keywords even with 0 args.
        self.assertRaises(TypeError, set, [], 2)
        self.assertRaises(TypeError, frozenset, [], 2)
        self.assertRaises(TypeError, set, a=1)
        self.assertRaises(TypeError, frozenset, a=1)
        self.assertRaises(TypeError, set().__init__, a=1)

    @make_dynamo_test
    def test_equality(self):
        a = self.thetype("abc")
        for typ in (self.thetype, set, frozenset):
            self.assertEqual(a, typ(a))
            self.assertTrue(a == typ(a))
            self.assertTrue(a.__eq__(typ(a)))
            self.assertTrue(self.thetype.__eq__(a, typ(a)))

    @make_dynamo_test
    def test_in_frozenset(self):
        item = self.thetype("abc")
        container = self.thetype([frozenset("abc")])
        self.assertIn(item, container)

    @make_dynamo_test
    def test_contains(self):
        s = self.thetype(["a", "b", "c"])
        self.assertIn("a", s)
        self.assertNotIn("d", s)
        self.assertTrue(s.__contains__("a"))
        self.assertTrue(self.thetype.__contains__(s, "b"))

    @make_dynamo_test
    def test_isdisjoint(self):
        x = self.thetype({"apple", "banana", "cherry"})
        y = self.thetype({"google", "microsoft", "apple"})
        z = self.thetype({"shoes", "flipflops", "sneakers"})
        self.assertFalse(x.isdisjoint(y))
        self.assertTrue(x.isdisjoint(z))
        self.assertRaises(TypeError, x.isdisjoint)
        self.assertRaises(TypeError, x.isdisjoint, 1)
        self.assertRaises(TypeError, x.isdisjoint, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertFalse(self.thetype.isdisjoint(p, q))

    @make_dynamo_test
    def test_intersection(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "apple"})
        intersection_set = set1.intersection(set2, set3)
        self.assertEqual(intersection_set, {"apple"})
        self.assertRaises(TypeError, set1.intersection, 1)
        self.assertRaises(TypeError, set1.intersection, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(self.thetype.intersection(p, q), {"b"})

    @make_dynamo_test
    def test_union(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        union_set = p.union(q, r)
        self.assertEqual(union_set, {"a", "b", "c", "e", "f"})
        self.assertRaises(TypeError, p.union, 1)
        self.assertRaises(TypeError, p.union, [[]])
        s = self.thetype.union(q, r)
        self.assertEqual(s, {"b", "c", "e", "f"})

    @make_dynamo_test
    def test_difference(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "sneakers"})
        difference_set = set1.difference(set2, set3)
        self.assertEqual(difference_set, {"banana", "cherry"})
        self.assertRaises(TypeError, set1.difference, 1)
        self.assertRaises(TypeError, set1.difference, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(self.thetype.difference(p, q), {"a", "c"})

    @make_dynamo_test
    def test_symmetric_difference(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        symmetric_diff_set = set1.difference(set2)
        self.assertEqual(symmetric_diff_set, {"banana", "cherry"})
        self.assertRaises(TypeError, set1.symmetric_difference)
        self.assertRaises(TypeError, set1.symmetric_difference, 1)
        self.assertRaises(TypeError, set1.symmetric_difference, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        symmetric_diff_set = self.thetype.symmetric_difference(p, q)
        self.assertEqual(symmetric_diff_set, {"a", "c", "e", "f"})

    @make_dynamo_test
    def test_to_frozenset(self):
        set1 = frozenset(self.thetype({"apple", "banana", "cherry"}))
        self.assertIsInstance(set1, frozenset)
        self.assertEqual(len(set1), 3)

    @make_dynamo_test
    def test_to_set(self):
        set1 = frozenset(self.thetype({"apple", "banana", "cherry"}))
        self.assertIsInstance(set1, frozenset)
        self.assertEqual(len(set1), 3)


class _SetBase(_FrozensetBase):
    # Set Methods
    # + add
    # + clear
    # - copy (inherited from frozenset)
    # - difference (inherited from frozenset)
    # + difference_update
    # + discard
    # - intersection (inherited from frozenset)
    # + intersection_update
    # - isdisjoint (inherited from frozenset)
    # - issubset (inherited from frozenset)
    # - issuperset (inherited from frozenset)
    # + pop
    # + remove
    # - symmetric_difference (inherited from frozenset)
    # + symmetric_difference_update
    # - union (inherited from frozenset)
    # + update

    @make_dynamo_test
    def test_add(self):
        p = self.thetype("abc")
        p.add("d")
        self.assertEqual(p, {"a", "b", "c", "d"})
        p.add("a")
        self.assertEqual(p, {"a", "b", "c", "d"})
        self.assertRaises(TypeError, p.add, ["ab"])
        self.assertRaises(TypeError, p.add)
        self.thetype.add(p, "e")
        self.assertEqual(p, {"a", "b", "c", "d", "e"})

    @make_dynamo_test
    def test_clear(self):
        p = self.thetype("abc")
        p.clear()
        self.assertEqual(p, set())
        p = self.thetype("abc")
        self.thetype.clear(p)
        self.assertEqual(len(p), 0)

    @make_dynamo_test
    def test_remove(self):
        p = self.thetype("abc")
        self.assertEqual(p.remove("a"), None)
        self.assertEqual(p, {"b", "c"})
        self.assertRaises(KeyError, p.remove, "a")
        p = self.thetype("abc")
        self.thetype.remove(p, "b")
        self.assertEqual(p, self.thetype({"a", "c"}))

    @make_dynamo_test
    def test_intersection_update(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "apple"})
        self.assertIsNone(set1.intersection_update(set2, set3))
        self.assertEqual(set1, {"apple"})
        self.assertRaises(TypeError, set1.intersection_update, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.thetype.intersection_update(p, q)
        self.assertEqual(p, {"b"})

    @make_dynamo_test
    def test_difference_update(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "sneakers"})
        self.assertIsNone(set1.difference_update(set2, set3))
        self.assertEqual(set1, {"banana", "cherry"})
        self.assertRaises(TypeError, set1.difference_update, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.thetype.difference_update(p, q)
        self.assertEqual(p, {"a", "c"})

    @make_dynamo_test
    def test_symmetric_difference_update(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        self.assertIsNone(set1.symmetric_difference_update(set2))
        self.assertEqual(set1, {"banana", "cherry", "google", "microsoft"})
        self.assertRaises(TypeError, set1.symmetric_difference_update)
        self.assertRaises(TypeError, set1.symmetric_difference_update, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.thetype.symmetric_difference_update(p, q)
        self.assertEqual(p, {"a", "c", "e", "f"})

    @make_dynamo_test
    def test_pop(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        e = set1.pop()
        self.assertNotIn(e, set1)
        s = self.thetype()
        self.assertRaises(KeyError, s.pop)
        p = self.thetype("a")
        self.assertEqual(self.thetype.pop(p), "a")

    @make_dynamo_test
    def test_update(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        p.update(q, r)
        self.assertEqual(p, {"a", "b", "c", "e", "f"})
        self.assertRaises(TypeError, p.update, [[]])
        self.thetype.update(q, r)
        self.assertEqual(q, {"b", "c", "e", "f"})

    @make_dynamo_test
    def test_discard(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set1.discard("banana")
        set2.discard("cherry")
        self.assertEqual(set1, {"apple", "cherry"})
        self.assertEqual(set2, {"google", "microsoft", "apple"})
        p = self.thetype("abc")
        self.thetype.discard(p, "a")
        self.assertEqual(p, {"b", "c"})

    @make_dynamo_test
    def test_remove_discard_unhashable(self):
        # remove/discard hash the key before the membership check, so an
        # unhashable element raises TypeError rather than KeyError (remove) or
        # silently succeeding (discard). Mirrors CPython set_discard_key.
        self.assertRaises(TypeError, self.thetype("abc").remove, [])
        self.assertRaises(TypeError, self.thetype("abc").discard, [])


class FrozensetTests(_FrozensetBase, _BaseSetTests):
    thetype = frozenset

    @make_dynamo_test
    def test_copy_preserves_identity(self):
        p = frozenset("abc")
        self.assertTrue(id(p) == id(p.copy()))
        self.assertTrue(id(p) == id(frozenset.copy(p)))


class _SetKeyCoercionMixin:
    # set/frozenset allow an (unhashable) set key for remove/discard by
    # coercing it to a frozenset for the lookup, mirroring the set-key
    # fallback in CPython set_remove_impl / set_discard_impl.
    @make_dynamo_test
    def test_remove_set_key(self):
        s = self.thetype([frozenset("ab")])
        s.remove(set("ab"))
        self.assertEqual(len(s), 0)
        self.assertRaises(KeyError, self.thetype([frozenset("ab")]).remove, set("z"))

    @make_dynamo_test
    def test_discard_set_key(self):
        s = self.thetype([frozenset("ab")])
        s.discard(set("ab"))
        self.assertEqual(len(s), 0)
        # A second discard of a missing key is a no-op, not an error.
        s.discard(set("ab"))
        self.assertEqual(len(s), 0)


class SetTests(_SetBase, _SetKeyCoercionMixin, _BaseSetTests):
    thetype = set

    def test_in_frozenset(self):
        super().test_in_frozenset()


class UserDefinedSetTests(_SetBase, _SetKeyCoercionMixin, _BaseSetTests):
    class CustomSet(set):
        pass

    thetype = CustomSet

    def test_in_frozenset(self):
        super().test_in_frozenset()

    def test_equality(self):
        super().test_equality()


class UserDefinedFrozensetTests(_FrozensetBase, _BaseSetTests):
    class CustomFrozenset(frozenset):
        pass

    thetype = CustomFrozenset

    @make_dynamo_test
    def test_copy_returns_base_frozenset(self):
        p = self.thetype("abc")
        result = p.copy()
        self.assertTrue(type(result) is frozenset)
        self.assertTrue(id(result) != id(p))

        result = frozenset.copy(p)
        self.assertTrue(type(result) is frozenset)
        self.assertTrue(id(result) != id(p))

    def test_in_frozenset(self):
        super().test_in_frozenset()


class OrderedSetTests(_SetBase, _BaseSetTests):
    from torch.utils._ordered_set import OrderedSet

    thetype = OrderedSet

    def test_in_frozenset(self):
        # We aren't equal w/ other sets due to ordering
        pass

    def test_equality(self):
        super().test_equality()

    @make_dynamo_test
    def test_maintains_order(self):
        # Test that OrderedSet maintains insertion order
        s = self.thetype(["c", "b", "a"])
        items = list(s)
        self.assertEqual(items, ["c", "b", "a"])

    @make_dynamo_test
    def test_intersection_maintains_order(self):
        # Test that intersection maintains order from first set
        s1 = self.thetype(["a", "b", "c", "d"])
        s2 = self.thetype(["d", "c", "b"])
        result = s1.intersection(s2)
        self.assertIsInstance(result, self.thetype)
        self.assertEqual(list(result), ["b", "c", "d"])

    @make_dynamo_test
    def test_union_maintains_order(self):
        # Test that union maintains order (first set order, then second set new items)
        s1 = self.thetype(["a", "b", "c"])
        s2 = self.thetype(["c", "d", "e"])
        result = s1.union(s2)
        self.assertIsInstance(result, self.thetype)
        self.assertEqual(list(result), ["a", "b", "c", "d", "e"])

    @make_dynamo_test
    def test_difference_maintains_order(self):
        # Test that difference maintains order from first set
        s1 = self.thetype(["a", "b", "c", "d"])
        s2 = self.thetype(["b", "d"])
        result = s1.difference(s2)
        self.assertIsInstance(result, self.thetype)
        self.assertEqual(list(result), ["a", "c"])

    @make_dynamo_test
    def test_symmetric_difference_maintains_order(self):
        # Test that symmetric_difference maintains order
        s1 = self.thetype(["a", "b", "c"])
        s2 = self.thetype(["c", "d", "e"])
        result = s1.symmetric_difference(s2)
        self.assertIsInstance(result, self.thetype)
        # Should have items from s1 not in s2, then items from s2 not in s1
        self.assertEqual(set(result), {"a", "b", "d", "e"})

    @make_dynamo_test
    def test_copy_preserves_type(self):
        # Test that copy returns an OrderedSet
        s = self.thetype(["a", "b", "c"])
        s_copy = s.copy()
        self.assertIsInstance(s_copy, self.thetype)
        self.assertEqual(list(s_copy), ["a", "b", "c"])

    @make_dynamo_test
    def test_binop_preserves_type(self):
        # Test that binary operations preserve OrderedSet type
        s1 = self.thetype(["a", "b", "c"])
        s2 = self.thetype(["b", "c", "d"])

        # Test |
        result = s1 | s2
        self.assertIsInstance(result, self.thetype)

        # Test &
        result = s1 & s2
        self.assertIsInstance(result, self.thetype)

        # Test -
        result = s1 - s2
        self.assertIsInstance(result, self.thetype)

        # Test ^
        result = s1 ^ s2
        self.assertIsInstance(result, self.thetype)

    @make_dynamo_test
    def test_construct_from_generator(self):
        s = self.thetype(x.upper() for x in ["a", "b", "c"])
        self.assertEqual(list(s), ["A", "B", "C"])

    @make_dynamo_test
    def test_construct_from_map(self):
        s = self.thetype(map(str, [1, 2, 3]))
        self.assertEqual(list(s), ["1", "2", "3"])

    @make_dynamo_test
    def test_construct_from_range(self):
        s = self.thetype(range(4))
        self.assertEqual(list(s), [0, 1, 2, 3])


class FrozensetHierarchyTests(_BaseSetTests):
    """frozenset must not be a subclass of set (CPython parity). Part of #192874."""

    def test_frozenset_not_a_set_variable(self):
        from torch._dynamo.variables.sets import (
            BaseSetVariable,
            FrozensetVariable,
            SetVariable,
        )

        self.assertFalse(issubclass(FrozensetVariable, SetVariable))
        self.assertTrue(issubclass(FrozensetVariable, BaseSetVariable))
        self.assertTrue(issubclass(SetVariable, BaseSetVariable))
        # Matches real CPython.
        self.assertFalse(issubclass(frozenset, set))

    def test_frozenset_has_no_mutating_methods(self):
        # Before the VT split, FrozensetVariable inherited SetVariable's
        # named methods via the tp_methods MRO-merge, so eight of the nine
        # below were reachable on a traced frozenset even though real
        # frozenset doesn't have them. `update` is the exception: it already
        # returned early on `not self.is_mutable()`, so it degraded correctly.
        for name, args in [
            ("add", (1,)),
            ("pop", ()),
            ("remove", (1,)),
            ("discard", (1,)),
            ("clear", ()),
            ("update", ({1},)),
            ("intersection_update", ({1},)),
            ("difference_update", ({1},)),
            ("symmetric_difference_update", ({1},)),
        ]:
            with self.subTest(name=name):
                self.assertFalse(hasattr(frozenset(), name))

                @torch.compile(backend="eager", fullgraph=True)
                def fn(fs, _name=name, _args=args):
                    return getattr(fs, _name)(*_args)

                with self.assertRaises(Unsupported):
                    fn(frozenset({1, 2, 3}))

    def test_frozenset_mutating_method_graph_breaks(self):
        # With fullgraph=False a frozenset mutation must graph break and let
        # eager raise the real AttributeError. Before the split it surfaced an
        # internal "mutation_type is None for FrozensetVariable() in
        # check_allowed_side_effect" AssertionError instead.
        @torch.compile(backend="eager")
        def fn(fs):
            fs.add(1)
            return fs

        with self.assertRaises(AttributeError):
            fn(frozenset({1, 2, 3}))

    @make_dynamo_test
    def test_frozenset_inplace_operators_rebind_not_mutate(self):
        """Not a regression test: this also passes before the split.

        Real frozenset defines no __ior__/__iand__/__ixor__/__isub__, so
        `fs |= other` rebinds the name to a new frozenset built by __or__.
        FrozensetVariable used to inherit SetVariable's nb_inplace_* handlers,
        but binary_iop1 gates on the real CPython type's nb slot bits (see
        object_protocol.binary_iop1) and frozenset sets none of them, so the
        inherited handlers were never reachable. Kept as a guard that the
        split does not change this.
        """
        fs = frozenset({1, 2, 3})
        other = frozenset({3, 4, 5})

        fs_or = fs
        fs_or |= other
        self.assertEqual(fs_or, frozenset({1, 2, 3, 4, 5}))

        fs_and = fs
        fs_and &= other
        self.assertEqual(fs_and, frozenset({3}))

        fs_xor = fs
        fs_xor ^= other
        self.assertEqual(fs_xor, frozenset({1, 2, 4, 5}))

    @make_dynamo_test
    def test_frozenset_readonly_ops_still_work(self):
        fs = frozenset({1, 2, 3})
        other = frozenset({3, 4, 5})
        self.assertEqual(fs | other, frozenset({1, 2, 3, 4, 5}))
        self.assertEqual(fs & other, frozenset({3}))
        self.assertEqual(fs - other, frozenset({1, 2}))
        self.assertEqual(fs ^ other, frozenset({1, 2, 4, 5}))
        self.assertEqual(fs.union(other), frozenset({1, 2, 3, 4, 5}))
        self.assertEqual(fs.intersection(other), frozenset({3}))
        self.assertEqual(fs.difference(other), frozenset({1, 2}))
        self.assertEqual(fs.symmetric_difference(other), frozenset({1, 2, 4, 5}))
        self.assertTrue(fs.isdisjoint(frozenset({7, 8})))
        self.assertTrue(frozenset({1, 2}).issubset(fs))
        self.assertTrue(fs.issuperset(frozenset({1, 2})))
        self.assertEqual(fs.copy(), fs)
        self.assertEqual(len(fs), 3)
        self.assertTrue(3 in fs)

    @make_dynamo_test
    def test_regular_set_mutations_unaffected(self):
        # The mutable-set path must be completely unaffected by the split.
        s = {1, 2, 3}
        s.add(4)
        s.discard(1)
        s |= {10}
        self.assertEqual(s, {2, 3, 4, 10})


class OrderedSetHierarchyTests(torch._dynamo.test_case.TestCase):
    """``OrderedSet`` is a ``MutableSet``, not a ``set``. Part of #192874.

    ``torch.utils._ordered_set.OrderedSet`` has ``set``'s whole named-method
    surface, so unlike the frozenset and dict_keys rows nothing is removed
    here. What differs is that insertion order is observable and that the
    in-place operators are ``collections.abc.MutableSet``'s.
    """

    # Deliberately not in hash order. A result that comes back as [1, 2, 3]
    # was rebuilt from a set, not from the OrderedSet's own order.
    BASE = [3, 1, 2]
    OTHER = [5, 4, 9]

    @staticmethod
    def _snapshot(value):
        # OrderedSet.__eq__ ignores order (it compares the backing dicts), so
        # order has to be checked through list().
        from torch.utils._ordered_set import OrderedSet

        if isinstance(value, OrderedSet):
            return ("OrderedSet", list(value))
        if isinstance(value, tuple):
            return tuple(OrderedSetHierarchyTests._snapshot(v) for v in value)
        return value

    def _assert_matches_eager(self, fn, *args, fullgraph=True):
        expected = self._snapshot(fn(*args))
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager", fullgraph=fullgraph)
        self.assertEqual(self._snapshot(compiled(*args)), expected)

    def test_ordered_set_is_not_a_set_variable(self):
        """Structural guard against the classes being merged again."""
        from collections.abc import MutableSet

        from torch._dynamo.variables.sets import (
            BaseSetVariable,
            OrderedSetVariable,
            SetVariable,
        )
        from torch.utils._ordered_set import OrderedSet

        self.assertFalse(issubclass(OrderedSetVariable, SetVariable))
        self.assertTrue(issubclass(OrderedSetVariable, BaseSetVariable))
        # Matches real CPython.
        self.assertFalse(issubclass(OrderedSet, set))
        self.assertTrue(issubclass(OrderedSet, MutableSet))

    def test_named_methods_match_eager(self):
        """All 17 named set methods trace and agree with eager, order included."""
        from torch.utils._ordered_set import OrderedSet

        base, other = list(self.BASE), list(self.OTHER)
        calls = {
            "isdisjoint": (other,),
            "intersection": ([2, 1, 9],),
            "union": (other,),
            "difference": ([1],),
            "symmetric_difference": (other,),
            "issubset": ([3, 1, 2, 9],),
            "issuperset": ([1],),
            "copy": (),
            "add": (9,),
            "pop": (),
            "remove": (1,),
            "discard": (1,),
            "clear": (),
            "update": (other,),
            "intersection_update": ([2, 1, 9],),
            "difference_update": ([1],),
            "symmetric_difference_update": (other,),
        }
        self.assertEqual(
            sorted(calls), sorted(n for n in dir(set) if not n.startswith("_"))
        )
        for name, args in calls.items():
            with self.subTest(method=name):

                def fn(_name=name, _args=args):
                    s = OrderedSet(base)
                    r = getattr(s, _name)(*_args)
                    return r, s

                self._assert_matches_eager(fn)

    def test_pop_is_lifo(self):
        """OrderedSet.pop is dict.popitem: the most recently inserted element."""
        from torch.utils._ordered_set import OrderedSet

        def fn():
            s = OrderedSet([3, 1, 2])
            a = s.pop()
            b = s.pop()
            return a, b, list(s)

        self.assertEqual(fn(), (2, 1, [3]))
        self._assert_matches_eager(fn)

    def test_reconstruction_preserves_insertion_order(self):
        """An OrderedSet leaving the compiled region keeps its order."""
        from torch.utils._ordered_set import OrderedSet

        def fn():
            s = OrderedSet([3, 1, 2])
            return (
                s,
                repr(s),
                s.copy(),
                OrderedSet(s),
                s | [9, 0],
                OrderedSet([7, 0, 3]),
            )

        self.assertEqual(
            self._snapshot(fn())[0:2],
            (("OrderedSet", [3, 1, 2]), "OrderedSet([3, 1, 2])"),
        )
        self._assert_matches_eager(fn)

    def test_inplace_operators_mutate_in_place(self):
        """MutableSet's in-place operators take any iterable and keep identity.

        set's slots return NotImplemented for a non-set operand, which falls
        back to the binary operator and rebinds the name to a new object.
        """
        from torch.utils._ordered_set import OrderedSet

        base, other = list(self.BASE), list(self.OTHER)

        def ior(ctor):
            s = OrderedSet(base)
            alias = s
            s |= ctor(other)
            return s is alias, s

        def iand(ctor):
            s = OrderedSet(base)
            alias = s
            s &= ctor([2, 1, 9])
            return s is alias, s

        def isub(ctor):
            s = OrderedSet(base)
            alias = s
            s -= ctor([1])
            return s is alias, s

        def ixor(ctor):
            s = OrderedSet(base)
            alias = s
            s ^= ctor(other)
            return s is alias, s

        for fn in (ior, iand, isub, ixor):
            for ctor in (OrderedSet, list, tuple):
                with self.subTest(op=fn.__name__, operand=ctor.__name__):
                    self.assertTrue(fn(ctor)[0])
                    self._assert_matches_eager(fn, ctor)

        # set operands as literals: CPython folds a literal of constants into
        # one frozenset constant, so eager and Dynamo iterate it in the same
        # order. set([5, 4, 9]) would not: Dynamo keeps the list's order.
        def with_set_operands():
            s = OrderedSet(base)
            alias = s
            s |= {4, 5, 9}
            a = s is alias
            s &= {1, 2, 4}
            b = s is alias
            s -= {1}
            c = s is alias
            s ^= {2, 7}
            d = s is alias
            return (a, b, c, d), s

        self.assertEqual(
            self._snapshot(with_set_operands()),
            ((True, True, True, True), ("OrderedSet", [4, 7])),
        )
        self._assert_matches_eager(with_set_operands)

    def test_binary_operators_match_eager(self):
        """Set's operators accept any iterable, on either side, and build an
        OrderedSet (Set._from_iterable) even when the OrderedSet is the right
        operand."""
        from torch.utils._ordered_set import OrderedSet

        base, other = list(self.BASE), list(self.OTHER)

        def forward(ctor):
            s = OrderedSet(base)
            o = ctor(other)
            return s | o, s & o, s - o, s ^ o

        def reflected(ctor):
            s = OrderedSet(base)
            o = ctor(other)
            return o | s, o & s, o - s, o ^ s

        for fn in (forward, reflected):
            for ctor in (OrderedSet, list, tuple):
                with self.subTest(op=fn.__name__, operand=ctor.__name__):
                    for r in fn(ctor):
                        self.assertIs(type(r), OrderedSet)
                    self._assert_matches_eager(fn, ctor)

        def reflected_from_set():
            s = OrderedSet(base)
            r = {5, 4, 9} - s
            return type(r), r

        self._assert_matches_eager(reflected_from_set)

    def test_unbound_class_method_calls(self):
        """OrderedSet.method(obj, ...) is dispatched only for an OrderedSet obj."""
        from torch.utils._ordered_set import OrderedSet

        base = list(self.BASE)

        def on_ordered_set():
            s = OrderedSet(base)
            OrderedSet.add(s, 9)
            popped = OrderedSet.pop(s)
            return OrderedSet.union(s, [4]), popped, s

        self._assert_matches_eager(on_ordered_set)

        # With a plain set as the receiver the pure-Python method bodies run
        # against a set and raise in eager. Before the split Dynamo routed
        # these to SetVariable and returned a value instead.
        def add_on_set():
            s = {3, 1, 2}
            OrderedSet.add(s, 9)
            return s

        def union_on_set():
            return OrderedSet.union({3, 1, 2}, [4])

        for fn, exc in ((add_on_set, AttributeError), (union_on_set, TypeError)):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(exc):
                    fn()
                torch._dynamo.reset()
                with self.assertRaises(exc):
                    torch.compile(fn, backend="eager", fullgraph=False)()
                torch._dynamo.reset()
                with self.assertRaises(Unsupported):
                    torch.compile(fn, backend="eager", fullgraph=True)()

    def test_unbound_calls_with_unknown_name_or_no_receiver(self):
        """Names set lacks, or a missing receiver, fall through instead of
        escaping as IndexError / InternalTorchDynamoError."""
        from torch.utils._ordered_set import OrderedSet

        def no_receiver():
            return OrderedSet.union()

        def private_name():
            return OrderedSet._wrap_iter_in_set(OrderedSet([3, 1, 2]), [1])

        for fn in (no_receiver, private_name):
            with self.subTest(fn=fn.__name__):
                # fullgraph=False: graph break, eager semantics (TypeError for
                # the first two, [2, 1, 3] for the third).
                torch._dynamo.reset()
                compiled = torch.compile(fn, backend="eager", fullgraph=False)
                try:
                    expected = fn()
                except TypeError as e:
                    with self.assertRaises(TypeError) as cm:
                        compiled()
                    self.assertEqual(str(cm.exception), str(e))
                else:
                    self.assertEqual(compiled(), expected)
                torch._dynamo.reset()
                with self.assertRaises(Unsupported):
                    torch.compile(fn, backend="eager", fullgraph=True)()

    def test_passed_in_ordered_set_reads(self):
        """Membership and iteration on an OrderedSet argument, and the guards
        that go with them.

        VariableBuilder guards the argument through ``_dict`` (DICT_KEYS_MATCH
        with key order). The per-key SET_CONTAINS guard evaluated
        ``set.__contains__`` on the OrderedSet and failed on the frame that
        created it, and registering the object's own source for key-order
        guarding hit ``Expected dict`` in the guard manager.
        """
        from torch._dynamo.testing import CompileCounter
        from torch.utils._ordered_set import OrderedSet

        def fn(x, s):
            out = []
            for v in s:  # the iteration protocol itself is under test
                out.append(v)  # noqa: PERF402
            return 2 in s, 9 in s, out, list(s), s.union([9]), x + 1

        cnt = CompileCounter()
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.ones(1)
        # Same content twice, then a different order, different contents and
        # an empty one: every distinct input must recompile and agree with
        # eager, including iteration order.
        inputs = [[3, 1, 2], [3, 1, 2], [2, 1, 3], [3, 1], [3, 1, 2, 9], []]
        for items in inputs:
            self.assertEqual(
                self._snapshot(compiled(x, OrderedSet(items))),
                self._snapshot(fn(x, OrderedSet(items))),
            )
        self.assertEqual(cnt.frame_count, 5)

    def test_passed_in_ordered_set_mutations_reach_caller(self):
        """Mutating an OrderedSet argument must mutate the caller's object.

        The builder registered a passed-in OrderedSet with
        ``track_object_existing`` (AttributeMutationExisting), which
        ``SideEffects.mutation`` never flags, so every mutation was traced
        and then dropped. A literal ``set`` argument goes through
        ``wrap_literal`` and ``track_mutable`` instead, which is why sets
        appeared to work.
        """
        from torch.utils._ordered_set import OrderedSet

        base = list(self.BASE)

        def add(s):
            s.add(9)

        def discard(s):
            s.discard(1)

        def remove(s):
            s.remove(1)

        def pop(s):
            return s.pop()

        def clear(s):
            s.clear()

        def update(s):
            s.update([9, 8])

        def intersection_update(s):
            s.intersection_update([2, 1, 9])

        def difference_update(s):
            s.difference_update([1])

        def symmetric_difference_update(s):
            s.symmetric_difference_update([1, 9])

        def ior(s):
            s |= [9, 8]

        def iand(s):
            s &= [2, 1, 9]

        def isub(s):
            s -= [1]

        def ixor(s):
            s ^= [1, 9]

        def unbound_add(s):
            OrderedSet.add(s, 9)

        def chain(s):
            s.add(9)
            s.discard(3)
            s |= [7]
            return s.pop()

        for mutate in (
            add,
            discard,
            remove,
            pop,
            clear,
            update,
            intersection_update,
            difference_update,
            symmetric_difference_update,
            ior,
            iand,
            isub,
            ixor,
            unbound_add,
            chain,
        ):
            for fullgraph in (True, False):
                with self.subTest(op=mutate.__name__, fullgraph=fullgraph):

                    def fn(x, s):
                        r = mutate(s)
                        return r, x + 1

                    eager_s = OrderedSet(base)
                    expected = fn(torch.ones(1), eager_s)
                    torch._dynamo.reset()
                    compiled_s = OrderedSet(base)
                    got = torch.compile(fn, backend="eager", fullgraph=fullgraph)(
                        torch.ones(1), compiled_s
                    )
                    self.assertEqual(self._snapshot(got), self._snapshot(expected))
                    self.assertEqual(list(compiled_s), list(eager_s))

    def test_passed_in_non_literal_set_mutation_reaches_caller(self):
        """Same builder branch as OrderedSet: a set whose elements are not all
        literals was also registered for attribute mutation and lost its
        mutations. A literal set never hit it."""

        def fn(x, s):
            s.add(9)
            s.discard(slice(0, 1))
            return x + 1

        for make in (lambda: {slice(0, 1), 3}, lambda: {3, 1, 2}):
            with self.subTest(literal=all(isinstance(v, int) for v in make())):
                eager_s = make()
                fn(torch.ones(1), eager_s)
                torch._dynamo.reset()
                compiled_s = make()
                torch.compile(fn, backend="eager", fullgraph=True)(
                    torch.ones(1), compiled_s
                )
                self.assertEqual(compiled_s, eager_s)

    def test_constructor_none_and_iterables(self):
        """OrderedSet(None) is the documented empty constructor."""
        from torch.utils._ordered_set import OrderedSet

        def fn():
            return (
                OrderedSet(None),
                OrderedSet(),
                OrderedSet(iterable=[3, 1]),
                OrderedSet("cab"),
                OrderedSet(range(3, 0, -1)),
                OrderedSet(v * 2 for v in [3, 1, 2]),
            )

        self._assert_matches_eager(fn)

    def test_dict_attribute_reads(self):
        """``s._dict`` exposes the backing dict so inlined OrderedSet methods
        that read it (e.g. ``elem in self._dict``) trace instead of graph
        breaking on an unmodeled attribute."""
        from torch.utils._ordered_set import OrderedSet

        def reads(x, s):
            return 3 in s._dict, 9 in s._dict, list(s._dict), len(s._dict), x + 1

        self._assert_matches_eager(reads, torch.ones(1), OrderedSet([3, 1, 2]))

    def test_reversed_matches_eager(self):
        """OrderedSet.__reversed__ gives reverse insertion order; set has none."""
        from torch.utils._ordered_set import OrderedSet

        def in_region():
            return list(reversed(OrderedSet([3, 1, 2])))

        def unbound():
            return list(OrderedSet.__reversed__(OrderedSet([3, 1, 2])))

        self.assertEqual(in_region(), [2, 1, 3])
        self._assert_matches_eager(in_region)
        self._assert_matches_eager(unbound)

        def passed_in(x, s):
            return list(reversed(s)), x + 1

        x = torch.ones(1)
        self.assertEqual(
            self._snapshot(passed_in(x, OrderedSet([3, 1, 2]))),
            self._snapshot(
                torch.compile(passed_in, backend="eager", fullgraph=True)(
                    x, OrderedSet([3, 1, 2])
                )
            ),
        )

    def test_is_unhashable(self):
        from torch.utils._ordered_set import OrderedSet

        def fn():
            return hash(OrderedSet([1]))

        with self.assertRaises(TypeError):
            fn()
        torch._dynamo.reset()
        with self.assertRaises(TypeError):
            torch.compile(fn, backend="eager", fullgraph=False)()

    def test_regular_set_behaviour_unaffected(self):
        def fn(s):
            s.add(4)
            s |= {10}
            return (
                s.union({3}),
                s.pop() in {1, 2, 4, 10},
                s == {1, 2},
                s.isdisjoint({9}),
            )

        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled({1, 2}), fn({1, 2}))


class DictKeySetHierarchyTests(torch._dynamo.test_case.TestCase):
    """`dict_keys` is not a `set`.

    A `dict_keys` passed into the compiled region is traced by
    `DictKeySetVariable`. Its CPython surface is the `collections.abc.Set` one,
    `isdisjoint` plus the operator slots, not `set`'s named methods.
    """

    # dir(set) - dir(dict.keys()), i.e. everything a dict_keys must not have.
    SET_ONLY_METHODS = (
        "union",
        "intersection",
        "difference",
        "symmetric_difference",
        "issubset",
        "issuperset",
        "copy",
        "add",
        "pop",
        "remove",
        "discard",
        "clear",
        "update",
        "intersection_update",
        "difference_update",
        "symmetric_difference_update",
    )

    @staticmethod
    def keys():
        return {1: 0, 2: 0}.keys()

    @staticmethod
    def caller(name):
        if name in ("copy", "clear", "pop"):
            return lambda k: getattr(k, name)()
        return lambda k: getattr(k, name)({1})

    def test_set_only_methods_raise_attribute_error(self):
        """Compiled behaviour matches eager: AttributeError, not a value."""
        for name in self.SET_ONLY_METHODS:
            with self.subTest(method=name):
                fn = self.caller(name)
                with self.assertRaises(AttributeError):
                    fn(self.keys())
                torch._dynamo.reset()
                compiled = torch.compile(fn, backend="eager", fullgraph=False)
                with self.assertRaises(AttributeError):
                    compiled(self.keys())

    def test_set_only_methods_are_untraceable(self):
        """Under fullgraph there is no graph break, so it must be Unsupported."""
        for name in self.SET_ONLY_METHODS:
            with self.subTest(method=name):
                torch._dynamo.reset()
                compiled = torch.compile(
                    self.caller(name), backend="eager", fullgraph=True
                )
                with self.assertRaises(Unsupported):
                    compiled(self.keys())

    def test_binary_ops_match_eager(self):
        """dictview_or and friends accept any iterable, unlike set's slots."""
        ops = {
            "or_set": lambda k: k | {3},
            "and_set": lambda k: k & {1},
            "sub_set": lambda k: k - {1},
            "xor_set": lambda k: k ^ {1},
            "or_list": lambda k: k | [3],
            "and_list": lambda k: k & [1],
            "sub_list": lambda k: k - [1],
            "xor_list": lambda k: k ^ [1],
            "reflected_or": lambda k: {3} | k,
        }
        for name, fn in ops.items():
            with self.subTest(op=name):
                torch._dynamo.reset()
                compiled = torch.compile(fn, backend="eager", fullgraph=True)
                self.assertEqual(compiled(self.keys()), fn(self.keys()))

    def test_binary_op_with_non_iterable_raises(self):
        fn = lambda k: k | 5  # noqa: E731
        with self.assertRaises(TypeError):
            fn(self.keys())
        torch._dynamo.reset()
        with self.assertRaises((TypeError, Unsupported)):
            torch.compile(fn, backend="eager", fullgraph=True)(self.keys())

    def test_comparisons_match_eager(self):
        cmps = {
            "eq_set": lambda k: k == {1, 2},
            "eq_frozenset": lambda k: k == frozenset({1, 2}),
            "eq_list": lambda k: k == [1, 2],
            "ne_set": lambda k: k != {1, 2},
            "le_set": lambda k: k <= {1, 2, 3},
            "lt_set": lambda k: k < {1, 2, 3},
            "ge_set": lambda k: k >= {1},
            "gt_set": lambda k: k > {1},
        }
        for name, fn in cmps.items():
            with self.subTest(cmp=name):
                torch._dynamo.reset()
                compiled = torch.compile(fn, backend="eager", fullgraph=True)
                self.assertEqual(compiled(self.keys()), fn(self.keys()))

    def test_ordering_against_non_set_raises(self):
        fn = lambda k: k <= [1, 2, 3]  # noqa: E731
        with self.assertRaises(TypeError):
            fn(self.keys())
        torch._dynamo.reset()
        with self.assertRaises((TypeError, Unsupported)):
            torch.compile(fn, backend="eager", fullgraph=True)(self.keys())

    def test_set_abc_surface_still_works(self):
        """isdisjoint, len, containment and iteration are unaffected."""
        ops = {
            "isdisjoint_set": lambda k: k.isdisjoint({9}),
            "isdisjoint_list": lambda k: k.isdisjoint([9]),
            "isdisjoint_false": lambda k: k.isdisjoint({1}),
            "len": lambda k: len(k),
            "contains": lambda k: 1 in k,
            "iter": lambda k: set(k),
            "mapping": lambda k: dict(k.mapping),
        }
        for name, fn in ops.items():
            with self.subTest(op=name):
                torch._dynamo.reset()
                compiled = torch.compile(fn, backend="eager", fullgraph=True)
                self.assertEqual(compiled(self.keys()), fn(self.keys()))

    def test_dict_view_operands_match_eager(self):
        """A view built inside the region is a DictKeysVariable, not this VT.

        Comparing against one must fall through to the reflected operation.
        """
        ops = {
            "eq_keys": lambda k: k == {1: 9, 2: 9}.keys(),
            "le_keys": lambda k: k <= {1: 9, 2: 9, 3: 0}.keys(),
            "or_keys": lambda k: k | {3: 0}.keys(),
            "and_keys": lambda k: k & {1: 0}.keys(),
            "eq_items": lambda k: k == {1: 9}.items(),
            "eq_dict": lambda k: k == {1: 0, 2: 0},
            "isdisjoint_keys": lambda k: k.isdisjoint({9: 0}.keys()),
        }
        for name, fn in ops.items():
            with self.subTest(op=name):
                torch._dynamo.reset()
                compiled = torch.compile(fn, backend="eager", fullgraph=True)
                self.assertEqual(compiled(self.keys()), fn(self.keys()))

    def test_dict_keys_is_not_a_set_variable(self):
        """Structural guard against the classes being merged again."""
        from torch._dynamo.variables.sets import (
            BaseSetVariable,
            DictKeySetVariable,
            SetVariable,
        )

        self.assertFalse(issubclass(DictKeySetVariable, SetVariable))
        self.assertTrue(issubclass(DictKeySetVariable, BaseSetVariable))

    def test_regular_set_behaviour_unaffected(self):
        def fn(s):
            return s.union({3}), s | {4}, s == {1, 2}, s.isdisjoint({9})

        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled({1, 2}), fn({1, 2}))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
