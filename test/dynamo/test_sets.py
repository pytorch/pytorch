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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
