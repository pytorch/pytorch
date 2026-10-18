# Owner(s): ["module: dynamo"]
"""Tests for ByteArrayVariable: bytearray support in Dynamo.

Ported from CPython's Lib/test/test_bytes.py (BaseBytesTest class).
Only tests covering operations supported in PR 1 (read-only ops + wiring)
are included; mutation, methods, and conversions will follow in later PRs.
"""

import sys

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
    run_tests,
)


class Indexable:
    def __init__(self, value=0):
        self.value = value

    def __index__(self):
        return self.value


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

    @make_dynamo_test
    def test_mod(self):
        b = self.type2test(b"hello, %b!")
        orig = b
        b = b % b"world"
        self.assertEqual(b, self.type2test(b"hello, world!"))
        self.assertEqual(orig, self.type2test(b"hello, %b!"))
        self.assertIs(type(b), self.type2test)

        b = self.type2test(b"%s / 100 = %d%%")
        a = b % (b"seventy-nine", 79)
        self.assertEqual(a, self.type2test(b"seventy-nine / 100 = 79%"))
        self.assertIs(type(a), self.type2test)

    @make_dynamo_test
    def test_inplace_mod(self):
        b = self.type2test(b"%d")
        b %= 5
        self.assertEqual(b, self.type2test(b"5"))

    @make_dynamo_test
    def test_count(self):
        b = self.type2test(b"mississippi")
        self.assertEqual(b.count(b"i"), 4)
        self.assertEqual(b.count(b"i", 6), 2)

    @make_dynamo_test
    def test_rindex(self):
        b = self.type2test(b"abcabc")

        self.assertEqual(b.rindex(b"b"), 4)
        self.assertEqual(b.rindex(b"b", 0, 4), 1)

        with self.assertRaises(ValueError):
            b.rindex(b"x")
        with self.assertRaisesRegex(TypeError, "integer or bytes-like object"):
            b.rindex("b")

    @make_dynamo_test
    def test_split_rsplit(self):
        b = self.type2test(b"a,,b,c")
        parts = b.split(b",", maxsplit=2)
        self.assertEqual(
            parts,
            [self.type2test(b"a"), self.type2test(), self.type2test(b"b,c")],
        )
        self.assertIs(type(parts[0]), bytearray)
        self.assertEqual(
            b.rsplit(b",", maxsplit=2),
            [self.type2test(b"a,"), self.type2test(b"b"), self.type2test(b"c")],
        )
        self.assertEqual(
            b.split(self.type2test(b","), 1),
            [self.type2test(b"a"), self.type2test(b",b,c")],
        )
        whitespace = self.type2test(b"  a  b  ")
        self.assertEqual(whitespace.split(None, 0), [self.type2test(b"a  b  ")])
        self.assertEqual(whitespace.rsplit(None, 0), [self.type2test(b"  a  b")])
        self.assertRaisesRegex(ValueError, "^empty separator$", b.split, b"")
        self.assertRaisesRegex(
            TypeError, "^a bytes-like object is required, not 'str'$", b.rsplit, ","
        )
        self.assertRaisesRegex(
            OverflowError,
            "^Python int too large to convert to C ssize_t$",
            b.split,
            b",",
            sys.maxsize + 1,
        )
        self.assertRaisesRegex(
            OverflowError,
            "^Python int too large to convert to C ssize_t$",
            b.rsplit,
            b",",
            sys.maxsize + 1,
        )

    def test_input_bytearray_split_rsplit(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            return ba.split(b","), ba.rsplit(b",", 1)

        result = fn(bytearray(b"a,b,c"))
        self.assertEqual(
            result,
            (
                [bytearray(b"a"), bytearray(b"b"), bytearray(b"c")],
                [bytearray(b"a,b"), bytearray(b"c")],
            ),
        )
        for parts in result:
            for part in parts:
                self.assertIs(type(part), bytearray)

    def test_input_bytearray_split_rsplit_no_separator(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            return ba.split(b","), ba.rsplit(b",")

        ba = bytearray(b"abc")
        result = fn(ba)
        self.assertEqual(result, ([bytearray(b"abc")], [bytearray(b"abc")]))
        for parts in result:
            self.assertIs(type(parts[0]), bytearray)
            self.assertIsNot(parts[0], ba)
        self.assertIsNot(result[0][0], result[1][0])

    def test_input_bytearray_split_rsplit_after_mutation(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            ba += b",d"
            return ba.split(b","), ba.rsplit(b",", 1)

        ba = bytearray(b"a,b")
        self.assertEqual(
            fn(ba),
            (
                [bytearray(b"a"), bytearray(b"b"), bytearray(b"d")],
                [bytearray(b"a,b"), bytearray(b"d")],
            ),
        )
        self.assertEqual(ba, bytearray(b"a,b,d"))

    def test_input_bytearray_split_rsplit_mutate_results(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            results = ba.split(b","), ba.rsplit(b",")
            for parts in results:
                parts[0].append(ord("z"))
                parts[0] += b"y"
            return ba, results

        ba = bytearray(b"a,b")
        returned, results = fn(ba)
        self.assertIs(returned, ba)
        self.assertEqual(ba, bytearray(b"a,b"))
        for parts in results:
            self.assertEqual(parts, [bytearray(b"azy"), bytearray(b"b")])
            for part in parts:
                self.assertIs(type(part), bytearray)

    def test_input_bytearray_split_rsplit_nonconstant_separator(self):
        def fn(x, ba, sep):
            return x + 1, ba.split(sep), ba.rsplit(sep)

        args = torch.ones(1), bytearray(b"a,b,c"), memoryview(b",")
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Dynamo does not know how to trace method `split` of class `bytearray`",
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(*args)
        result = torch.compile(fn, backend="eager")(*args)
        self.assertEqual(
            result,
            (
                torch.tensor([2.0]),
                [bytearray(b"a"), bytearray(b"b"), bytearray(b"c")],
                [bytearray(b"a"), bytearray(b"b"), bytearray(b"c")],
            ),
        )
        for parts in result[1:]:
            for part in parts:
                self.assertIs(type(part), bytearray)

    @make_dynamo_test
    def test_hex(self):
        b = self.type2test(b"\x01\x02\xff")
        self.assertEqual(b.hex(), "0102ff")
        self.assertEqual(b.hex(":", 2), "01:02ff")
        self.assertEqual(b.hex("-", -2), "0102-ff")
        with self.assertRaises(OverflowError):
            b.hex("-", 1 << 100)

    @make_dynamo_test
    def test_decode(self):
        b = self.type2test(b"foo\xc3\xa9")
        self.assertEqual(b.decode(), "foo\xe9")
        self.assertEqual(b.decode("utf-8", "strict"), "foo\xe9")

        invalid = self.type2test(b"\xffabc")
        self.assertEqual(invalid.decode(errors="replace"), "\ufffdabc")

    @make_dynamo_test
    def test_upper(self):
        b = self.type2test(b"hello world")
        self.assertEqual(b.upper(), self.type2test(b"HELLO WORLD"))
        # Non-ASCII bytes are unchanged (ASCII-only case mapping)
        b2 = self.type2test(b"\xff\xfeabc")
        self.assertEqual(b2.upper(), self.type2test(b"\xff\xfeABC"))
        self.assertRaises(TypeError, lambda: b.upper(1))

    @make_dynamo_test
    def test_lower(self):
        b = self.type2test(b"HELLO WORLD")
        self.assertEqual(b.lower(), self.type2test(b"hello world"))
        b2 = self.type2test(b"\xff\xfeABC")
        self.assertEqual(b2.lower(), self.type2test(b"\xff\xfeabc"))
        self.assertRaises(TypeError, lambda: b.lower(1))

    @make_dynamo_test
    def test_title(self):
        b = self.type2test(b"hello world")
        self.assertEqual(b.title(), self.type2test(b"Hello World"))
        b2 = self.type2test(b"they're bill's friends")
        self.assertEqual(b2.title(), self.type2test(b"They'Re Bill'S Friends"))
        self.assertRaises(TypeError, lambda: b.title(1))

    def test_input_bytearray_upper_lower_title(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            return ba.upper(), ba.lower(), ba.title()

        self.assertEqual(
            fn(bytearray(b"hello world")),
            (
                bytearray(b"HELLO WORLD"),
                bytearray(b"hello world"),
                bytearray(b"Hello World"),
            ),
        )

    @make_dynamo_test
    def test_splitlines(self):
        b = self.type2test(b"a\r\nb\nc\rd")
        self.assertEqual(
            b.splitlines(),
            [
                self.type2test(b"a"),
                self.type2test(b"b"),
                self.type2test(b"c"),
                self.type2test(b"d"),
            ],
        )
        lines = b.splitlines(keepends=True)
        self.assertEqual(
            lines,
            [
                self.type2test(b"a\r\n"),
                self.type2test(b"b\n"),
                self.type2test(b"c\r"),
                self.type2test(b"d"),
            ],
        )
        for line in lines:
            self.assertIs(type(line), bytearray)

    @make_dynamo_test
    def test_partition_and_rpartition(self):
        b = self.type2test(b"a-b-c")
        self.assertEqual(
            b.partition(b"-"),
            (self.type2test(b"a"), self.type2test(b"-"), self.type2test(b"b-c")),
        )
        result = b.rpartition(b"-")
        self.assertEqual(
            result,
            (self.type2test(b"a-b"), self.type2test(b"-"), self.type2test(b"c")),
        )
        for part in result:
            self.assertIs(type(part), bytearray)
        self.assertEqual(
            self.type2test(b"abc").partition(b"-"),
            (self.type2test(b"abc"), self.type2test(), self.type2test()),
        )
        self.assertEqual(
            self.type2test(b"abc").rpartition(b"-"),
            (self.type2test(), self.type2test(), self.type2test(b"abc")),
        )

    @make_dynamo_test
    def test_partition_empty_separator_and_keyword(self):
        b = self.type2test(b"abc")
        for method in (b.partition, b.rpartition):
            with self.assertRaisesRegex(ValueError, "empty separator"):
                method(b"")
            with self.assertRaisesRegex(TypeError, "takes no keyword arguments"):
                method(sep=b"-")

    def test_input_bytearray_hex_decode(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            return ba.hex(), ba.decode("latin-1")

        self.assertEqual(
            fn(bytearray(b"abc")),
            ("616263", "abc"),
        )
        self.assertEqual(
            fn(bytearray(b"\xff\x00")),
            ("ff00", "\xff\x00"),
        )

    def test_input_bytearray_splitlines_partition_rpartition(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            return ba.splitlines(keepends=True), ba.partition(b"-"), ba.rpartition(b"-")

        result = fn(self.type2test(b"a-b\nc-d"))
        self.assertEqual(
            result,
            (
                [bytearray(b"a-b\n"), bytearray(b"c-d")],
                (bytearray(b"a"), bytearray(b"-"), bytearray(b"b\nc-d")),
                (bytearray(b"a-b\nc"), bytearray(b"-"), bytearray(b"d")),
            ),
        )
        for parts in result:
            for part in parts:
                self.assertIs(type(part), bytearray)

    @parametrize("method", ["splitlines", "partition", "rpartition"])
    @parametrize("data", [b"abc", b""])
    def test_input_bytearray_splitlines_partition_rpartition_no_separator(
        self, method, data
    ):
        args = () if method == "splitlines" else (b"-",)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            return getattr(ba, method)(*args)

        ba = self.type2test(data)
        result = fn(ba)
        self.assertEqual(result, getattr(ba, method)(*args))
        self.assertIs(type(result), list if method == "splitlines" else tuple)
        for i, part in enumerate(result):
            self.assertIs(type(part), bytearray)
            self.assertIsNot(part, ba)
            for other in result[:i]:
                self.assertIsNot(part, other)

    @parametrize("method", ["splitlines", "partition", "rpartition"])
    def test_input_bytearray_splitlines_partition_rpartition_after_mutation(
        self, method
    ):
        args = () if method == "splitlines" else (b"-",)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            ba += b"-c\nd"
            return getattr(ba, method)(*args)

        ba = self.type2test(b"a-b")
        expected = self.type2test(b"a-b-c\nd")
        self.assertEqual(fn(ba), getattr(expected, method)(*args))
        self.assertEqual(ba, expected)

    @parametrize("method", ["splitlines", "partition", "rpartition"])
    def test_input_bytearray_splitlines_partition_rpartition_mutate_results(
        self, method
    ):
        args = () if method == "splitlines" else (b"-",)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba):
            result = getattr(ba, method)(*args)
            part = result[0]
            part.append(ord("z"))
            part += b"y"
            return ba, result

        for _ in range(2):
            ba = self.type2test(b"abc")
            expected = getattr(ba, method)(*args)
            expected[0].extend(b"zy")
            returned, result = fn(ba)
            self.assertIs(returned, ba)
            self.assertEqual(ba, self.type2test(b"abc"))
            self.assertEqual(result, expected)
            for part in result:
                self.assertIs(type(part), bytearray)

    def test_input_bytearray_splitlines_partition_rpartition_nonconstant_args(self):
        def splitlines(x, ba, keepends):
            return x + 1, ba.splitlines(keepends=keepends)

        def partition(x, ba, sep):
            return x + 1, ba.partition(sep)

        def rpartition(x, ba, sep):
            return x + 1, ba.rpartition(sep)

        x = torch.ones(1)
        ba = self.type2test(b"a-b\nc-d")
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Dynamo does not know how to trace method `splitlines` of class `bytearray`",
        ):
            torch.compile(splitlines, backend="eager", fullgraph=True)(
                x, ba, torch.tensor(True)
            )
        self.assertEqual(
            torch.compile(splitlines, backend="eager")(x, ba, torch.tensor(True)),
            (torch.tensor([2.0]), [bytearray(b"a-b\n"), bytearray(b"c-d")]),
        )

        for method, expected in (
            (partition, (bytearray(b"a"), bytearray(b"-"), bytearray(b"b\nc-d"))),
            (rpartition, (bytearray(b"a-b\nc"), bytearray(b"-"), bytearray(b"d"))),
        ):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                f"Dynamo does not know how to trace method `{method.__name__}` of class `bytearray`",
            ):
                torch.compile(method, backend="eager", fullgraph=True)(
                    x, ba, memoryview(b"-")
                )
            self.assertEqual(
                torch.compile(method, backend="eager")(x, ba, memoryview(b"-")),
                (torch.tensor([2.0]), expected),
            )

    def test_inplace_concat_arg_mutation(self):
        # Regression: missing sq_inplace_concat fell back to constant-folding
        # operator.iadd on the live bytearray, then skip-to-eager ran += again.
        @torch.compile(backend="eager")
        def f(ba, x):
            y = x + 1
            ba += b"z"
            return bytes(ba), y

        b = bytearray(b"ab")
        out_bytes, out_y = f(b, torch.ones(1))
        self.assertEqual(out_bytes, b"abz")
        self.assertEqual(bytes(b), b"abz")
        self.assertEqual(out_y, torch.ones(1) + 1)

    @make_dynamo_test
    def test_inplace_concat(self):
        b = self.type2test(b"ab")
        b += b"z"
        self.assertEqual(b, self.type2test(b"abz"))
        b += self.type2test(b"cd")
        self.assertEqual(b, self.type2test(b"abzcd"))

    def test_inplace_repeat_arg_mutation(self):
        # Same class of bug as += : missing sq_inplace_repeat rebinds a local
        # copy and never updates the caller's bytearray.
        @torch.compile(backend="eager", fullgraph=True)
        def f(ba, x):
            ba *= 2
            return len(ba), x + 1

        b = bytearray(b"ab")
        out_len, out_y = f(b, torch.ones(1))
        self.assertEqual(out_len, 4)
        self.assertEqual(bytes(b), b"abab")
        self.assertEqual(out_y, torch.ones(1) + 1)

    @make_dynamo_test
    def test_inplace_repeat(self):
        b = self.type2test(b"ab")
        b *= 2
        self.assertEqual(b, self.type2test(b"abab"))
        b *= 0
        self.assertEqual(b, self.type2test())

    @make_dynamo_test
    def test_append(self):
        b = self.type2test(b"hell")
        b.append(ord("o"))
        self.assertEqual(b, b"hello")
        self.assertEqual(b.append(100), None)
        b.append(Indexable(ord("A")))
        self.assertEqual(b, b"hellodA")
        self.assertRaises(TypeError, lambda: b.append(b"o"))
        self.assertRaises(ValueError, lambda: b.append(256))
        self.assertRaises(ValueError, lambda: b.append(-1))
        self.assertEqual(b, b"hellodA")

    @make_dynamo_test
    def test_extend(self):
        a = self.type2test(b"hello")
        a.extend(a)
        self.assertEqual(a, b"hellohello")
        a = self.type2test()
        a.extend(map(int, b"ab"))
        a.extend(int(x) for x in b"cd")
        a.extend([101, Indexable(102)])
        self.assertEqual(a, b"abcdef")
        self.assertRaises(ValueError, a.extend, [0, 1, 2, 256])
        self.assertRaises(ValueError, a.extend, map(int, "X"))
        self.assertEqual(a, b"abcdef")
        self.assertRaises(TypeError, a.extend, "def")
        with self.assertRaisesRegex(TypeError, "can't extend bytearray with float"):
            a.extend(1.0)

    def test_extend_consumes_iterator(self):
        def f(ba):
            it = iter([1, 2])
            ba.extend(it)
            return next(it, "done")

        b, expected = bytearray(b"a"), bytearray(b"a")
        out = torch.compile(f, backend="eager", fullgraph=True)(b)
        self.assertEqual(out, f(expected))
        self.assertEqual(b, expected)

    def test_extend_stops_at_first_out_of_range_byte(self):
        def f(ba):
            seen = []

            def gen():
                for x in (1, 300, 2):
                    seen.append(x)
                    yield x
                raise RuntimeError("drained past the bad byte")

            try:
                ba.extend(gen())
            except ValueError as e:
                return str(e), seen
            return None, seen

        self.assertEqual(
            torch.compile(f, backend="eager", fullgraph=True)(bytearray()),
            f(bytearray()),
        )

    def test_extend_leaves_iterator_tail_after_bad_byte(self):
        def f(x):
            y = x + 1
            it = iter([1, 300, 2])
            dst = bytearray()
            try:
                dst.extend(it)
            except ValueError:
                pass
            return y, next(it, "done"), dst

        x = torch.ones(1)
        self.assertEqual(
            torch.compile(f, backend="eager", fullgraph=True)(x)[1:], f(x)[1:]
        )

    def test_extend_sees_list_growth_during_index(self):
        class Grow:
            def __init__(self, items):
                self.items = items

            def __index__(self):
                if len(self.items) < 3:
                    self.items.append(7)
                return 1

        def f(x):
            y = x + 1
            items = []
            items.append(Grow(items))
            dst = bytearray()
            dst.extend(items)
            return y, dst

        x = torch.ones(1)
        self.assertEqual(
            torch.compile(f, backend="eager", fullgraph=True)(x)[1], f(x)[1]
        )

    def test_extend_bytes_subclass_reads_the_buffer(self):
        # CPython reads the buffer before it looks at __iter__.
        class Raw(bytes):
            def __iter__(self):
                return iter((9,))

        class RawArray(bytearray):
            def __iter__(self):
                return iter((9,))

            def __len__(self):
                return 1

        def f(x, value):
            y = x + 1
            dst = bytearray(b"q")
            dst.extend(value)
            return y, dst

        x = torch.ones(1)
        for value in (Raw(b"ab"), RawArray(b"ab")):
            self.assertEqual(
                torch.compile(f, backend="eager")(x, value)[1], f(x, value)[1]
            )

    def test_extend_asks_for_a_length_hint_first(self):
        class Items:
            def __iter__(self):
                return iter((1,))

            def __length_hint__(self):
                raise ValueError("hint failed")

        def f(x, items):
            y = x + 1
            dst = bytearray()
            try:
                dst.extend(items)
            except ValueError as e:
                return y, str(e)
            return y, dst

        x = torch.ones(1)
        self.assertEqual(
            torch.compile(f, backend="eager", fullgraph=True)(x, Items())[1],
            f(x, Items())[1],
        )

    def test_append_extend_arg_mutation(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(ba, x):
            ba.append(ord("c"))
            ba.extend(b"de")
            return len(ba), x + 1

        b = bytearray(b"ab")
        out_len, out_y = f(b, torch.ones(1))
        self.assertEqual(out_len, 5)
        self.assertEqual(b, bytearray(b"abcde"))
        self.assertEqual(out_y, torch.ones(1) + 1)

    def test_append_graph_break(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(ba, x):
            x = x + 1
            ba.append(1)
            x = x * 2
            torch._dynamo.graph_break()
            x = x + 3
            ba.extend(ba)
            return x - 1

        b = bytearray(b"a")
        self.assertEqual(f(b, torch.ones(1)), torch.full((1,), 6.0))
        self.assertEqual(b, bytearray(b"a\x01a\x01"))
        # One frame on each side of the explicit graph break: append and extend
        # are traced, and their mutations are replayed on the input. Each
        # mutation has a tensor op on both sides, so a graph break at either
        # one would split its frame and add a third.
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 4)

    @make_dynamo_test
    def test_insert(self):
        b = self.type2test(b"msssspp")
        b.insert(1, ord("i"))
        b.insert(4, ord("i"))
        b.insert(-2, ord("i"))
        b.insert(1000, ord("i"))
        self.assertEqual(b, b"mississippi")
        b.insert(Indexable(0), Indexable(ord("A")))
        self.assertEqual(b, b"Amississippi")
        self.assertRaises(TypeError, lambda: b.insert(0, b"1"))
        self.assertRaises(TypeError, lambda: b.insert("0", 1))
        self.assertRaises(ValueError, lambda: b.insert(0, 256))
        self.assertRaises(TypeError, lambda: b.insert(0))
        self.assertRaises(OverflowError, lambda: b.insert(2**64, 1))
        self.assertEqual(b, b"Amississippi")

    @make_dynamo_test
    def test_pop(self):
        b = self.type2test(b"world")
        self.assertEqual(b.pop(), ord("d"))
        self.assertEqual(b.pop(0), ord("w"))
        self.assertEqual(b.pop(Indexable(-2)), ord("r"))
        self.assertEqual(b, b"ol")
        self.assertRaises(IndexError, lambda: b.pop(10))
        self.assertRaises(IndexError, lambda: self.type2test().pop())
        self.assertRaises(TypeError, lambda: b.pop("0"))
        self.assertRaises(OverflowError, lambda: b.pop(2**64))
        self.assertEqual(self.type2test(b"\xff").pop(), 0xFF)

    def test_insert_pop_arg_mutation(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(ba, x):
            ba.insert(0, ord("a"))
            return ba.pop(), x + 1

        b = bytearray(b"bc")
        out, out_y = f(b, torch.ones(1))
        self.assertEqual(out, ord("c"))
        self.assertEqual(b, bytearray(b"ab"))
        self.assertEqual(out_y, torch.ones(1) + 1)

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

    @make_dynamo_test
    def test_contains_out_of_range(self):
        b = bytearray(b"abc")
        with self.assertRaises(ValueError):
            _ = 300 in b

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

    def test_iterator_graph_break(self):
        @torch.compile(backend="eager")
        def fn(b):
            it = iter(b)
            x0 = next(it)
            torch._dynamo.graph_break()
            x1 = next(it)
            x2 = next(it)
            return x0, x1, x2

        self.assertEqual(fn(bytearray(b"abc")), (97, 98, 99))

    def test_reversed_iterator_graph_break(self):
        @torch.compile(backend="eager")
        def fn(b):
            it = reversed(b)
            x0 = next(it)
            torch._dynamo.graph_break()
            x1 = next(it)
            x2 = next(it)
            return x0, x1, x2

        self.assertEqual(fn(bytearray(b"abc")), (99, 98, 97))


instantiate_parametrized_tests(ByteArrayTest)


if __name__ == "__main__":
    run_tests()
