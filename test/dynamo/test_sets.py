# Owner(s): ["module: dynamo"]

# TODO: move set tests from test_functions.py/test_misc.py to this file

import unittest

import torch
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

    def assertNotEqual(self, a, b):
        return self.assertTrue(a != b, f"{a} == {b}")


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

    @unittest.expectedFailure
    @make_dynamo_test
    def test_binop_sub(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(self.thetype.__sub__(p, q), set("ac"))

    @make_dynamo_test
    def test_binop_or(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(self.thetype.__or__(p, q), set("abcef"))

    @unittest.expectedFailure
    @make_dynamo_test
    def test_binop_xor(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(self.thetype.__xor__(p, q), set("acef"))

    @make_dynamo_test
    def test_cmp_eq(self):
        p = self.thetype("abc")
        self.assertTrue(self.thetype.__eq__(p, p))

    @make_dynamo_test
    def test_cmp_ne(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertTrue(self.thetype.__ne__(p, q))

    @make_dynamo_test
    def test_cmp_less_than(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertFalse(self.thetype.__lt__(p, p))

    @make_dynamo_test
    def test_cmp_greater_than(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertFalse(self.thetype.__gt__(p, p))

    @make_dynamo_test
    def test_cmp_less_than_or_equal(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertTrue(self.thetype.__le__(p, p))

    @make_dynamo_test
    def test_cmp_greater_than_or_equal(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertTrue(self.thetype.__ge__(p, p))

    @make_dynamo_test
    def test_copy(self):
        p = self.thetype("abc")
        self.assertEqual(self.thetype.copy(p), p)

    @make_dynamo_test
    def test_issubset(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        self.assertTrue(self.thetype.issubset(q, p))

    @make_dynamo_test
    def test_issuperset(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        self.assertTrue(self.thetype.issuperset(p, q))

    @make_dynamo_test
    def test_equality(self):
        a = self.thetype("abc")
        for typ in (self.thetype, set, frozenset):
            self.assertTrue(self.thetype.__eq__(a, typ(a)))

    @make_dynamo_test
    def test_contains(self):
        s = self.thetype(["a", "b", "c"])
        self.assertTrue(self.thetype.__contains__(s, "b"))

    @make_dynamo_test
    def test_isdisjoint(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertFalse(self.thetype.isdisjoint(p, q))

    @make_dynamo_test
    def test_intersection(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(self.thetype.intersection(p, q), {"b"})

    @make_dynamo_test
    def test_union(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        s = self.thetype.union(q, r)
        self.assertEqual(s, {"b", "c", "e", "f"})

    @make_dynamo_test
    def test_difference(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(self.thetype.difference(p, q), {"a", "c"})

    @make_dynamo_test
    def test_symmetric_difference(self):
        p, q = map(self.thetype, ["abc", "bef"])
        symmetric_diff_set = self.thetype.symmetric_difference(p, q)
        self.assertEqual(symmetric_diff_set, {"a", "c", "e", "f"})


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
        self.thetype.add(p, "e")
        self.assertEqual(p, {"a", "b", "c", "e"})

    @make_dynamo_test
    def test_clear(self):
        p = self.thetype("abc")
        self.thetype.clear(p)
        self.assertEqual(len(p), 0)

    @make_dynamo_test
    def test_remove(self):
        p = self.thetype("abc")
        self.thetype.remove(p, "b")
        self.assertEqual(p, self.thetype({"a", "c"}))

    @make_dynamo_test
    def test_intersection_update(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.thetype.intersection_update(p, q)
        self.assertEqual(p, {"b"})

    @make_dynamo_test
    def test_difference_update(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.thetype.difference_update(p, q)
        self.assertEqual(p, {"a", "c"})

    @make_dynamo_test
    def test_symmetric_difference_update(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.thetype.symmetric_difference_update(p, q)
        self.assertEqual(p, {"a", "c", "e", "f"})

    @make_dynamo_test
    def test_pop(self):
        p = self.thetype("a")
        self.assertEqual(self.thetype.pop(p), "a")

    @make_dynamo_test
    def test_update(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        self.thetype.update(q, r)
        self.assertEqual(q, {"b", "c", "e", "f"})

    @make_dynamo_test
    def test_discard(self):
        p = self.thetype("abc")
        self.thetype.discard(p, "a")
        self.assertEqual(p, {"b", "c"})


class FrozensetTests(_FrozensetBase, _BaseSetTests):
    thetype = frozenset

    @unittest.expectedFailure
    def test_union(self):
        super().test_union()


class SetTests(_SetBase, _BaseSetTests):
    thetype = set

    @unittest.expectedFailure
    def test_in_frozenset(self):
        super().test_in_frozenset()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
