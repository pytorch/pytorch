# Owner(s): ["module: dynamo"]

# TODO: move set tests from test_functions.py/test_misc.py to this file

import unittest

import torch
import torch._dynamo.config
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


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
    # +, -, |, &, ^, <, >, <=, >=

    @make_dynamo_test
    def test_binop_sub(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p - p, self.thetype())
        self.assertEqual(p - q, self.thetype("ac"))
        self.assertEqual(q - p, self.thetype("ef"))
        self.assertRaises(TypeError, lambda: p - 1)

    @make_dynamo_test
    def test_binop_or(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p | p, self.thetype("abc"))
        self.assertEqual(p | q, self.thetype("abcef"))

    @make_dynamo_test
    def test_binop_and(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p & p, self.thetype("abc"))
        self.assertEqual(p & q, self.thetype("b"))

    @make_dynamo_test
    def test_binop_xor(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p ^ p, self.thetype())
        self.assertEqual(p ^ q, self.thetype("acef"))

    @make_dynamo_test
    def test_binop_less_than(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertFalse(p < p)
        self.assertFalse(p < q)
        self.assertTrue(r < p)
        self.assertFalse(r < q)

    @make_dynamo_test
    def test_binop_greater_than(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertFalse(p > p)
        self.assertFalse(p > q)
        self.assertTrue(p > r)
        self.assertFalse(q > r)

    @make_dynamo_test
    def test_binop_less_than_or_equal(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertTrue(p <= p)
        self.assertFalse(p <= q)
        self.assertTrue(r <= p)
        self.assertFalse(r <= q)

    @make_dynamo_test
    def test_binop_greater_than_or_equal(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertTrue(p >= p)
        self.assertFalse(p >= q)
        self.assertTrue(p >= r)
        self.assertFalse(q >= r)

    @make_dynamo_test
    def test_copy(self):
        p = self.thetype("abc")
        q = p.copy()
        self.assertEqual(p, q)

    @make_dynamo_test
    def test_issubset(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        self.assertTrue(q.issubset(p))
        self.assertFalse(r.issubset(p))

    @make_dynamo_test
    def test_issuperset(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        self.assertTrue(p.issuperset(q))
        self.assertFalse(p.issuperset(r))

    @make_dynamo_test
    def test_constructor_iterable(self):
        p = self.thetype("abc")
        self.assertIsInstance(p, self.thetype)

    @make_dynamo_test
    def test_equality(self):
        a = self.thetype("abc")
        for typ in (self.thetype, set, frozenset):
            self.assertEqual(a, typ(a))

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

    @make_dynamo_test
    def test_isdisjoint(self):
        x = self.thetype({"apple", "banana", "cherry"})
        y = self.thetype({"google", "microsoft", "apple"})
        z = self.thetype({"shoes", "flipflops", "sneakers"})
        self.assertFalse(x.isdisjoint(y))
        self.assertTrue(x.isdisjoint(z))

    @make_dynamo_test
    def test_intersection(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "apple"})
        intersection_set = set1.intersection(set2, set3)
        self.assertEqual(intersection_set, {"apple"})

    @make_dynamo_test
    def test_union(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        union_set = p.union(q, r)
        self.assertEqual(union_set, {"a", "b", "c", "e", "f"})

    @make_dynamo_test
    def test_difference(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "sneakers"})
        difference_set = set1.difference(set2, set3)
        self.assertEqual(difference_set, {"banana", "cherry"})

    @make_dynamo_test
    def test_symmetric_difference(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        symmetric_diff_set = set1.difference(set2)
        self.assertEqual(symmetric_diff_set, {"banana", "cherry"})
        self.assertRaises(TypeError, set1.symmetric_difference)

    @make_dynamo_test
    def test_to_frozenset(self):
        set1 = frozenset(self.thetype({"apple", "banana", "cherry"}))
        self.assertIsInstance(set1, frozenset)

    @make_dynamo_test
    def test_to_set(self):
        set1 = frozenset(self.thetype({"apple", "banana", "cherry"}))
        self.assertIsInstance(set1, frozenset)


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

    @make_dynamo_test
    def test_clear(self):
        p = self.thetype("abc")
        p.clear()
        self.assertEqual(p, set())

    @make_dynamo_test
    def test_remove(self):
        p = self.thetype("abc")
        self.assertEqual(p.remove("a"), None)
        self.assertEqual(p, {"b", "c"})
        self.assertRaises(KeyError, p.remove, "a")

    @make_dynamo_test
    def test_intersection_update(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "apple"})
        set1.intersection_update(set2, set3)
        self.assertEqual(set1, {"apple"})

    @make_dynamo_test
    def test_difference_update(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "sneakers"})
        set1.difference_update(set2, set3)
        self.assertEqual(set1, {"banana", "cherry"})

    @make_dynamo_test
    def test_symmetric_difference_update(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set1.difference(set2)
        self.assertEqual(set1, {"banana", "cherry", "apple"})
        self.assertRaises(TypeError, set1.symmetric_difference_update)

    @make_dynamo_test
    def test_pop(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        e = set1.pop()
        self.assertNotIn(e, set1)
        s = self.thetype()
        self.assertRaises(KeyError, s.pop)

    @make_dynamo_test
    def test_update(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        p.update(q, r)
        self.assertEqual(p, {"a", "b", "c", "e", "f"})

    @make_dynamo_test
    def test_discard(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set1.discard("banana")
        set2.discard("cherry")
        self.assertEqual(set1, {"apple", "cherry"})
        self.assertEqual(set2, {"google", "microsoft", "apple"})


class FrozensetTests(_FrozensetBase, _BaseSetTests):
    thetype = frozenset

    @unittest.expectedFailure
    def test_issuperset(self):
        super().test_issuperset()


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
    def test_binop_greater_than(self):
        super().test_binop_greater_than()

    @unittest.expectedFailure
    def test_binop_greater_than_or_equal(self):
        super().test_binop_greater_than_or_equal()

    @unittest.expectedFailure
    def test_binop_less_than(self):
        super().test_binop_less_than()

    @unittest.expectedFailure
    def test_binop_less_than_or_equal(self):
        super().test_binop_less_than_or_equal()

    @unittest.expectedFailure
    def test_binop_sub(self):
        super().test_binop_sub()

    @unittest.expectedFailure
    def test_in_frozenset(self):
        super().test_in_frozenset()

    @unittest.expectedFailure
    def test_equality(self):
        super().test_in_frozenset()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
