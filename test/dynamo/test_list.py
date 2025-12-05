# Owner(s): ["module: dynamo"]

# TODO: move set tests from test_functions.py/test_misc.py to this file


import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


lst = []


class TupleTests(torch._dynamo.test_case.TestCase):
    # Tuple methods
    # + count
    # + index
    # BinOps:
    # +, <, >, <=, >=, ==, !=
    # Dunder methods:
    # + __getitem__
    # + __contains__
    # + __delitem__

    thetype = tuple

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    def assertEqual(self, a, b):
        return self.assertTrue(a == b, f"{a} != {b}")

    def assertNotEqual(self, x, y, msg=None, *, atol=None, rtol=None, **kwargs):
        return self.assertTrue(x != y, f"{x} == {y}")

    @make_dynamo_test
    def test_count(self):
        p = self.thetype("abcab")
        self.assertEqual(p.count("a"), 2)
        self.assertEqual(p.count("ab"), 0)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.count)
        self.assertRaises(TypeError, p.count, 2, 3)

    @make_dynamo_test
    def test_index(self):
        p = self.thetype("abc")
        self.assertEqual(p.index("a"), 0)
        self.assertRaises(ValueError, p.index, "e")

        # Wrong number of arguments
        self.assertRaises(TypeError, p.index)

    @make_dynamo_test
    def test_binop_imul(self):
        p = self.thetype([1, 2, 3])
        r = p.__mul__(2)
        self.assertIsInstance(r, self.thetype)
        self.assertEqual(r, self.thetype([1, 2, 3, 1, 2, 3]))
        self.assertEqual(p, self.thetype([1, 2, 3]))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__mul__)

        # can only multiply list by an integer
        self.assertRaises(TypeError, p.__mul__, 2.2)

    @make_dynamo_test
    def test_binop_add(self):
        p, q = map(self.thetype, ["abc", "bcd"])
        self.assertIsInstance(p + q, self.thetype)
        self.assertEqual(p + q, self.thetype("abcbcd"))
        self.assertEqual(p.__add__(q), self.thetype("abcbcd"))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__add__)

        # can only concatenate items of the same type
        self.assertRaises(TypeError, p.__add__, dict.fromkeys(q))

    @make_dynamo_test
    def test_cmp_eq(self):
        p, q, r = map(self.thetype, ["ab", "abc", "ab"])
        self.assertTrue(p == p)
        self.assertTrue(p == r)
        self.assertEqual(p, p)
        self.assertEqual(p, r)
        self.assertNotEqual(p, q)
        self.assertTrue(p.__eq__(r))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__eq__)

    @make_dynamo_test
    def test_cmp_ne(self):
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(p != q)
        self.assertNotEqual(p, q)
        self.assertTrue(p.__ne__(q))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__ne__)

    @make_dynamo_test
    def test_cmp_less_than(self):
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(p < q)
        self.assertTrue(p.__lt__(q))
        self.assertFalse(q < p)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__lt__)

    @make_dynamo_test
    def test_cmp_greater_than(self):
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(q > p)
        self.assertTrue(q.__gt__(p))
        self.assertFalse(p > q)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__gt__)

    @make_dynamo_test
    def test_cmp_less_than_or_equal(self):
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(p <= q)
        self.assertTrue(p.__le__(q))
        self.assertFalse(q <= p)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__le__)

    @make_dynamo_test
    def test_cmp_greater_than_or_equal(self):
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(q >= p)
        self.assertTrue(q.__ge__(p))
        self.assertFalse(p >= q)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__ge__)

    @make_dynamo_test
    def test___getitem__(self):
        p = self.thetype("abc")
        self.assertEqual(p.__getitem__(2), "c")
        self.assertRaises(IndexError, p.__getitem__, 10)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__getitem__)
        self.assertRaises(TypeError, p.__getitem__, 1, 2)

    @make_dynamo_test
    def test___contains__(self):
        p = self.thetype("abc")
        self.assertTrue(p.__contains__("a"))
        self.assertIsInstance(p.__contains__("c"), bool)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__contains__)
        self.assertRaises(TypeError, p.__contains__, 1, 2)

    @make_dynamo_test
    def test___iter__(self):
        p = self.thetype([1])
        it = p.__iter__()
        self.assertEqual(next(it), 1)
        it = p.__iter__().__iter__()
        self.assertEqual(next(it), 1)


class ListTests(TupleTests):
    # List methods
    # + append
    # + copy
    # + clear
    # + extend
    # + insert
    # + pop
    # + remove
    # + reverse
    # + sort
    # BinOps:
    # +, <, >, <=, >=, ==, !=
    # Dunder methods:
    # + __setitem__
    # + __getitem__
    # + __contains__
    # + __delitem__

    thetype = list

    @make_dynamo_test
    def test_append(self):
        p = self.thetype("abc")
        self.assertIsNone(p.append("d"))
        self.assertEqual(p, ["a", "b", "c", "d"])

        # Wrong number of arguments
        self.assertRaises(TypeError, p.append)
        self.assertRaises(TypeError, p.append, 2, 3)

    @make_dynamo_test
    def test_copy(self):
        p = self.thetype("abc")
        self.assertEqual(p.copy(), p)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.copy, 1)

    @make_dynamo_test
    def test_clear(self):
        p = self.thetype("abc")
        self.assertIsNone(p.clear())
        self.assertEqual(p, [])
        self.assertEqual(len(p), 0)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.clear, 1)

    @make_dynamo_test
    def test_extend(self):
        p, q = map(self.thetype, ["ab", "cd"])
        self.assertIsNone(p.extend(q))
        self.assertEqual(p, self.thetype("abcd"))

        # extend needs an iterable
        self.assertRaises(TypeError, p.extend, 1)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.extend)
        self.assertRaises(TypeError, p.extend, 2, 3)

    @make_dynamo_test
    def test_insert(self):
        p = self.thetype("abc")
        self.assertIsNone(p.insert(1, "ef"))
        self.assertEqual(p, ["a", "ef", "b", "c"])

        # Wrong number of arguments
        self.assertRaises(TypeError, p.insert)
        self.assertRaises(TypeError, p.insert, 1)
        self.assertRaises(TypeError, p.insert, 1, 2, 3)

    @make_dynamo_test
    def test_pop(self):
        p = self.thetype("abcd")
        self.assertEqual(p.pop(), "d")
        self.assertEqual(p.pop(1), "b")
        self.assertRaises(IndexError, p.pop, 10)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.pop, 2, 3)

    @make_dynamo_test
    def test_remove(self):
        p = self.thetype("abad")
        self.assertIsNone(p.remove("a"))
        self.assertEqual(p, ["b", "a", "d"])
        self.assertRaises(ValueError, p.remove, "x")

        # Wrong number of arguments
        self.assertRaises(TypeError, p.remove)
        self.assertRaises(TypeError, p.remove, 2, 3)

    @make_dynamo_test
    def test_reverse(self):
        p = self.thetype("abcd")
        self.assertIsNone(p.reverse())
        self.assertEqual(p, self.thetype("dcba"))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.reverse, 1)

    @make_dynamo_test
    def test_sort(self):
        p = self.thetype("dbca")
        self.assertIsNone(p.sort())
        self.assertEqual(p, self.thetype("abcd"))

    @make_dynamo_test
    def test_binop_imul(self):
        p = self.thetype([1, 2, 3])
        r = p.__imul__(2)
        self.assertIsInstance(r, self.thetype)
        self.assertEqual(r, self.thetype([1, 2, 3, 1, 2, 3]))
        self.assertEqual(p, self.thetype([1, 2, 3, 1, 2, 3]))

        p = self.thetype("ab")
        p *= 2
        self.assertEqual(p, self.thetype("abab"))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__imul__)

        # can only multiply list by an integer
        self.assertRaises(TypeError, p.__imul__, 2.2)

    def test_binop_imul_global_list(self):
        global lst
        lst = self.thetype(["a", "b"])

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            global lst
            lst *= 2
            lst.__imul__(3)
            return x.sin()

        x = torch.tensor(1.0)
        self.assertEqual(fn(x), x.sin())
        self.assertEqual(lst, ["a", "b"] * 6)

    @make_dynamo_test
    def test_binop_iadd(self):
        p, q = map(self.thetype, ["abc", "bcd"])
        r = p.__iadd__(q)
        self.assertIsInstance(r, self.thetype)
        self.assertEqual(r, self.thetype("abcbcd"))
        self.assertEqual(p, self.thetype("abcbcd"))

        p = self.thetype("ab")
        p += "cd"
        self.assertEqual(p, self.thetype("abcd"))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__iadd__)

        # can only concatenate items of the same type
        self.assertRaises(TypeError, p.__add__, dict.fromkeys(q))

    def test_binop_iadd_global_list(self):
        global lst
        lst = self.thetype([])

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            global lst
            lst += ["a"]
            lst.__iadd__(["b"])
            return x.sin()

        x = torch.tensor(1.0)
        self.assertEqual(fn(x), x.sin())
        self.assertEqual(lst, ["a", "b"])

    def test_binop_delitem_global_list(self):
        global lst
        lst = self.thetype(["a", "b", "c"])

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            global lst
            del lst[1]
            return x.sin()

        x = torch.tensor(1.0)
        self.assertEqual(fn(x), x.sin())
        self.assertEqual(lst, ["a", "c"])

    @make_dynamo_test
    def test___setitem__(self):
        p = self.thetype("abc")
        self.assertIsNone(p.__setitem__(2, "a"))
        self.assertEqual(p, self.thetype("aba"))

        p[0:] = []
        self.assertEqual(p, [])

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__setitem__)
        self.assertRaises(TypeError, p.__setitem__, 1)
        self.assertRaises(TypeError, p.__setitem__, 1, 2, 3)

    @make_dynamo_test
    def test___delitem__(self):
        p = self.thetype("abcdef")
        self.assertIsNone(p.__delitem__(1))
        self.assertEqual(p, self.thetype("acdef"))

        self.assertIsNone(p.__delitem__(slice(1, 3)))
        self.assertEqual(p, self.thetype("aef"))

        # Slice step == 0
        self.assertRaises(ValueError, p.__delitem__, slice(1, 1, 0))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__delitem__)
        self.assertRaises(TypeError, p.__delitem__, 1.1)
        self.assertRaises(TypeError, p.__delitem__, 1, 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
