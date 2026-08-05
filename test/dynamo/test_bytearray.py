# Owner(s): ["module: dynamo"]
"""Tests for ByteArrayVariable: bytearray support in Dynamo.

Ported from CPython's Lib/test/test_bytes.py (BaseBytesTest class).
Only tests covering operations supported in PR 1 (read-only ops + wiring)
are included; mutation, methods, and conversions will follow in later PRs.
"""

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import make_dynamo_test, run_tests


class ByteArrayTest(torch._dynamo.test_case.TestCase):
    """bytearray-specific tests, ported from CPython ByteArrayTest."""

    type2test = bytearray

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    # -- Construction --

    @make_dynamo_test
    def test_basics(self):
        b = self.type2test()
        self.assertEqual(type(b), self.type2test)

    @make_dynamo_test
    def test_empty_sequence(self):
        b = self.type2test()
        self.assertEqual(len(b), 0)
        self.assertRaises(IndexError, lambda: b[0])
        self.assertRaises(IndexError, lambda: b[1])
        self.assertRaises(IndexError, lambda: b[-1])

    @make_dynamo_test
    def test_from_iterable(self):
        b = self.type2test(range(256))
        self.assertEqual(len(b), 256)
        self.assertEqual(list(b), list(range(256)))

    @make_dynamo_test
    def test_from_tuple(self):
        b = self.type2test(tuple(range(256)))
        self.assertEqual(len(b), 256)
        self.assertEqual(list(b), list(range(256)))

    @make_dynamo_test
    def test_from_list(self):
        b = self.type2test(list(range(256)))
        self.assertEqual(len(b), 256)
        self.assertEqual(list(b), list(range(256)))

    @make_dynamo_test
    def test_from_index(self):
        b = self.type2test([0, 1, 2, 255])
        self.assertEqual(list(b), [0, 1, 2, 255])

    @make_dynamo_test
    def test_from_int(self):
        b = self.type2test(0)
        self.assertEqual(b, self.type2test())
        b = self.type2test(10)
        self.assertEqual(b, self.type2test([0] * 10))

    # -- Error paths for constructor --

    @make_dynamo_test
    def test_constructor_negative_count(self):
        with self.assertRaises(ValueError):
            self.type2test(-1)

    @make_dynamo_test
    def test_constructor_value_out_of_range(self):
        with self.assertRaises(ValueError):
            self.type2test([256])

    @make_dynamo_test
    def test_constructor_string_without_encoding(self):
        with self.assertRaises(TypeError):
            self.type2test("abc")

    @make_dynamo_test
    def test_constructor_encoding_without_string(self):
        self.assertRaises(TypeError, self.type2test, 0, "ascii")
        self.assertRaises(TypeError, self.type2test, b"", "ascii")
        self.assertRaises(TypeError, self.type2test, encoding="ascii")
        self.assertRaises(TypeError, self.type2test, errors="ignore")

    def test_constructor_encoding_non_constant_source(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return bytearray(x, "utf-8")

        with self.assertRaises((TypeError, torch._dynamo.exc.Unsupported)):
            fn(torch.tensor(1))

    def test_constructor_encoding_non_constant_encoding(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(enc):
            return bytearray("hello", enc)

        with self.assertRaises((TypeError, torch._dynamo.exc.Unsupported)):
            fn(torch.tensor(1))

    # -- Comparison --

    @make_dynamo_test
    def test_compare(self):
        b1 = self.type2test([1, 2, 3])
        b2 = self.type2test([1, 2, 3])
        b3 = self.type2test([1, 3])

        self.assertEqual(b1, b2)
        self.assertTrue(b2 != b3)
        self.assertTrue(b1 <= b2)
        self.assertTrue(b1 <= b3)
        self.assertTrue(b1 < b3)
        self.assertTrue(b1 >= b2)
        self.assertTrue(b3 >= b2)
        self.assertTrue(b3 > b2)

        self.assertFalse(b1 != b2)
        self.assertFalse(b2 == b3)
        self.assertFalse(b1 > b2)
        self.assertFalse(b1 > b3)
        self.assertFalse(b1 >= b3)
        self.assertFalse(b1 < b2)
        self.assertFalse(b3 < b2)
        self.assertFalse(b3 <= b2)

    def test_compare_to_bytes(self):
        @torch.compile(backend="eager")
        def fn(ba, other):
            return ba == other, ba < other, ba <= other

        eq, lt, le = fn(bytearray(b"ab"), b"ab")
        self.assertTrue(eq)
        self.assertFalse(lt)
        self.assertTrue(le)

        eq, lt, le = fn(bytearray(b"ab"), b"b")
        self.assertFalse(eq)
        self.assertTrue(lt)
        self.assertTrue(le)

    def test_compare_to_str(self):
        import warnings

        @torch.compile(backend="eager")
        def fn(ba, s):
            return ba == s, ba != s

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BytesWarning)
            eq, ne = fn(bytearray(b"\0a\0b\0c"), "abc")
            self.assertFalse(eq)
            self.assertTrue(ne)

            eq, ne = fn(bytearray(), "")
            self.assertFalse(eq)
            self.assertTrue(ne)

    # -- Sequence ops --

    @make_dynamo_test
    def test_reversed(self):
        input_data = list(map(ord, "Hello"))
        b = self.type2test(input_data)
        output = list(reversed(b))
        input_data.reverse()
        self.assertEqual(output, input_data)

    @make_dynamo_test
    def test_getslice(self):
        b = self.type2test(b"Hello, world")
        self.assertEqual(b[:5], self.type2test(b"Hello"))
        self.assertEqual(b[1:5], self.type2test(b"ello"))
        self.assertEqual(b[5:7], self.type2test(b", "))
        self.assertEqual(b[7:], self.type2test(b"world"))
        self.assertEqual(b[7:12], self.type2test(b"world"))
        self.assertEqual(b[7:100], self.type2test(b"world"))
        self.assertEqual(b[:-7], self.type2test(b"Hello"))
        self.assertEqual(b[-11:-7], self.type2test(b"ello"))
        self.assertEqual(b[-7:-5], self.type2test(b", "))
        self.assertEqual(b[-5:], self.type2test(b"world"))
        self.assertEqual(b[-5:12], self.type2test(b"world"))
        self.assertEqual(b[-5:100], self.type2test(b"world"))
        self.assertEqual(b[-100:5], self.type2test(b"Hello"))

    @make_dynamo_test
    def test_extended_getslice(self):
        L = list(range(20))
        b = self.type2test(L)
        self.assertEqual(b[::2], self.type2test(L[::2]))
        self.assertEqual(b[1::2], self.type2test(L[1::2]))
        self.assertEqual(b[::-1], self.type2test(L[::-1]))
        self.assertEqual(b[3:10:3], self.type2test(L[3:10:3]))
        self.assertEqual(b[-1:-10:-1], self.type2test(L[-1:-10:-1]))
        self.assertEqual(b[0:0:1], self.type2test(L[0:0:1]))

    @make_dynamo_test
    def test_repeat(self):
        b = self.type2test(b"abc")
        self.assertEqual(b * 3, b"abcabcabc")
        self.assertEqual(b * 0, b"")
        self.assertEqual(b * -1, b"")
        self.assertRaises(TypeError, lambda: b * 3.14)
        self.assertRaises(TypeError, lambda: 3.14 * b)

    @make_dynamo_test
    def test_repeat_1char(self):
        self.assertEqual(self.type2test(b"x") * 100, self.type2test([ord("x")] * 100))

    @make_dynamo_test
    def test_contains(self):
        b = self.type2test(b"abc")
        self.assertIn(ord("a"), b)
        self.assertIn(int(ord("a")), b)
        self.assertNotIn(200, b)

    @make_dynamo_test
    def test_concat(self):
        b1 = self.type2test(b"abc")
        b2 = self.type2test(b"def")
        self.assertEqual(b1 + b2, b"abcdef")
        self.assertRaises(TypeError, lambda: b1 + "def")

    @make_dynamo_test
    def test_iter(self):
        b = self.type2test(b"abc")
        result = list(b)
        self.assertEqual(result, [97, 98, 99])

    @make_dynamo_test
    def test_iter_sum(self):
        b = self.type2test(b"abc")
        total = 0
        for byte_val in b:
            total += byte_val
        self.assertEqual(total, 97 + 98 + 99)

    @make_dynamo_test
    def test_truth(self):
        self.assertFalse(bool(self.type2test()))
        self.assertTrue(bool(self.type2test(b"a")))

    @make_dynamo_test
    def test_isinstance(self):
        b = self.type2test(b"abc")
        self.assertIsInstance(b, self.type2test)

    @make_dynamo_test
    def test_len(self):
        b = self.type2test(b"hello")
        self.assertEqual(len(b), 5)
        self.assertEqual(len(self.type2test()), 0)

    def test_repr(self):
        @torch.compile(backend="eager")
        def fn():
            return bytearray(b"abc")

        self.assertEqual(repr(fn()), "bytearray(b'abc')")

    def test_repr_empty(self):
        @torch.compile(backend="eager")
        def fn():
            return bytearray()

        self.assertEqual(repr(fn()), "bytearray(b'')")

    @make_dynamo_test
    def test_unhashable(self):
        self.assertRaises(TypeError, hash, self.type2test(b"abc"))

    # -- builder.py input path: pass bytearray as argument to compiled fn --

    def test_input_bytearray_basic(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            return len(ba)

        self.assertEqual(fn(bytearray(b"hello")), 5)

    def test_input_bytearray_index(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            return ba[0], ba[2]

        self.assertEqual(fn(bytearray(b"abc")), (97, 99))

    def test_input_bytearray_iter(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            return list(ba)

        self.assertEqual(fn(bytearray(b"abc")), [97, 98, 99])

    def test_input_bytearray_contains(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            return 97 in ba

        self.assertTrue(fn(bytearray(b"abc")))

    def test_input_bytearray_slice(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            return ba[1:3]

        self.assertEqual(fn(bytearray(b"abcd")), bytearray(b"bc"))

    def test_large_bytearray_constructor(self):
        n = 1 << 20

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return len(bytearray(n))

        self.assertEqual(fn(), n)

    # -- reconstruct: force a graph break with a live bytearray --

    def test_reconstruct_graph_break(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(ba):
            a = ba[0]
            torch._dynamo.graph_break()
            return a + ba[1]

        result = fn(bytearray(b"abc"))
        self.assertEqual(result, 97 + 98)


if __name__ == "__main__":
    run_tests()
