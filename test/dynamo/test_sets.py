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


class _BaseSetTests(torch._dynamo.test_case.TestCase):
    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    def assertEqual(self, a, b):
        return self.assertTrue(a == b, f"{a} != {b}")

    def assertNotEqual(self, a, b):
        return self.assertTrue(a != b, f"{a} == {b}")


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
            nonlocal n  # noqa: F824
            s = {2, 4, 5}
            s.isdisjoint(gen())
            if n == 3:
                return x.sin()
            return x.cos()

        x = torch.randn(1)
        y = fn(x)
        self.assertEqual(y, x.sin())


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


from user code:
   File "test_sets.py", line N, in fn
    for i in s:""",  # noqa: B950
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
        container = self.thetype([frozenset("abc")])  # noqa: C405
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
        set.add(p, "e")
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


class FrozensetTests(_FrozensetBase, _BaseSetTests):
    thetype = frozenset


class SetTests(_SetBase, _BaseSetTests):
    thetype = set

    @unittest.expectedFailure
    def test_in_frozenset(self):
        super().test_in_frozenset()


class UserDefinedSetTests(_SetBase, _BaseSetTests):
    class CustomSet(set):
        pass

    thetype = CustomSet

    @unittest.expectedFailure
    def test_in_frozenset(self):
        super().test_in_frozenset()

    @unittest.expectedFailure
    def test_equality(self):
        super().test_in_frozenset()


class UserDefinedFrozensetTests(_FrozensetBase, _BaseSetTests):
    class CustomFrozenset(frozenset):
        pass

    thetype = CustomFrozenset

    @unittest.expectedFailure
    def test_in_frozenset(self):
        super().test_in_frozenset()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
