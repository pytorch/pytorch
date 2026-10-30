# Owner(s): ["module: dynamo"]
"""Tests for ByteArrayVariable: bytearray support in Dynamo.

Ported from CPython's Lib/test/test_bytes.py (BaseBytesTest class).
"""

import sys

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import CompileCounter
from torch._dynamo.test_case import run_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)


class Indexable:
    def __init__(self, value=0):
        self.value = value

    def __index__(self):
        return self.value


@instantiate_parametrized_tests
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

    def test_inplace_concat_arg_mutation(self):
        # Regression: missing sq_inplace_concat fell back to constant-folding
        # operator.iadd on the live bytearray, then skip-to-eager ran += again.
        @torch.compile(backend="eager", fullgraph=True)
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

    # -- Item assignment and deletion --

    @make_dynamo_test
    def test_setitem(self):
        b = self.type2test(b"abcd")
        b[0] = ord("A")
        b[-1] = ord("D")
        b[1] = Indexable(ord("B"))
        self.assertEqual(b, b"ABcD")
        self.assertRaises(IndexError, lambda: b.__setitem__(4, 101))
        self.assertRaises(IndexError, lambda: b.__setitem__(-5, 101))
        self.assertRaises(ValueError, lambda: b.__setitem__(0, 256))
        self.assertRaises(ValueError, lambda: b.__setitem__(0, -1))
        self.assertRaises(TypeError, lambda: b.__setitem__(0, "x"))
        self.assertRaises(TypeError, lambda: b.__setitem__("a", 101))
        # A rejected store leaves the buffer untouched.
        self.assertEqual(b, b"ABcD")

    @make_dynamo_test
    def test_delitem(self):
        b = self.type2test(b"abcd")
        del b[0]
        self.assertEqual(b, b"bcd")
        del b[-1]
        self.assertEqual(b, b"bc")
        del b[Indexable(0)]
        self.assertEqual(b, b"c")
        self.assertRaises(IndexError, lambda: b.__delitem__(1))
        self.assertRaises(IndexError, lambda: b.__delitem__(-2))
        self.assertRaises(TypeError, lambda: b.__delitem__("a"))
        self.assertEqual(b, b"c")

    @make_dynamo_test
    def test_setitem_augmented(self):
        b = self.type2test(b"abc")
        b[0] += 1
        b[-1] += 2
        self.assertEqual(b, b"bbe")

    def test_setitem_local(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            b = bytearray(b"abc")
            b[1] = ord("B")
            del b[0]
            return bytes(b)

        self.assertEqual(fn(), b"Bc")

    def test_setitem_arg_mutation(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def f(ba, i, v, x):
            ba[i] = v
            return x + len(ba)

        b = bytearray(b"abc")
        self.assertEqual(f(b, 1, ord("B"), torch.ones(1)), torch.full((1,), 4.0))
        self.assertEqual(b, bytearray(b"aBc"))
        # bytearray has no source, but wrap_bytearray guards on its contents;
        # the first store changes them, so the second call recompiles.
        self.assertEqual(f(b, -1, ord("Z"), torch.ones(1)), torch.full((1,), 4.0))
        self.assertEqual(b, bytearray(b"aBZ"))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)

    def test_delitem_arg_mutation(self):
        # A sourced bytearray cannot be resized under Dynamo: it may hold a
        # buffer export that replay cannot observe, so the frame stays eager.
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(ba, i, x):
            del ba[i]
            return x + len(ba)

        b = bytearray(b"abcd")
        self.assertEqual(f(b, 0, torch.ones(1)), torch.full((1,), 4.0))
        self.assertEqual(b, bytearray(b"bcd"))
        self.assertEqual(f(b, -1, torch.ones(1)), torch.full((1,), 3.0))
        self.assertEqual(b, bytearray(b"bc"))
        self.assertEqual(cnt.frame_count, 0)

    def test_delitem_local(self):
        def f(data, x):
            ba = bytearray(data)
            del ba[0]
            del ba[-1]
            return bytes(ba), x + 1

        self.assertEqual(
            torch.compile(f, backend="eager", fullgraph=True)(b"abcd", torch.ones(1)),
            (b"bc", torch.full((1,), 2.0)),
        )

    def test_setitem_delitem_single_frame(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(data, x):
            ba = bytearray(data)
            x = x + 1
            ba[0] = 65
            del ba[-1]
            return bytes(ba), x * 2

        self.assertEqual(f(b"abc", torch.ones(1)), (b"Ab", torch.full((1,), 4.0)))
        # Both mutations are traced, so the tensor ops on either side stay in
        # the same frame.
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_setitem_value_mutates_buffer(self):
        class Clear:
            def __init__(self, ba):
                self.ba = ba

            def __index__(self):
                del self.ba[:]
                return 65

        def f(x):
            ba = bytearray(b"abc")
            try:
                ba[-1] = Clear(ba)
            except IndexError as e:
                return bytes(ba), str(e), x + 1
            return bytes(ba), None, x + 1

        x = torch.ones(1)
        self.assertEqual(torch.compile(f, backend="eager", fullgraph=True)(x), f(x))

    def _error_from_traced(self, op):
        # The exception is caught inside the traced function, so the text has
        # to come from the tracer.
        def f(ba):
            try:
                op(ba)
            except Exception as e:
                return type(e).__name__, str(e)
            return None

        eager_buf = bytearray(b"abc")
        expected = f(eager_buf)
        buf = bytearray(b"abc")
        traced = torch.compile(f, backend="eager", fullgraph=True)(buf)
        self.assertEqual(traced, expected)
        # A rejected store leaves the buffer untouched on both paths.
        self.assertEqual(buf, eager_buf)
        return traced

    def test_setitem_error_order_and_message(self):
        # bytearray_ass_subscript validates the byte value before it range-checks
        # the index, and the exception text is CPython's.
        def set_bad_value_type(ba):
            ba[99] = "x"

        def set_out_of_range_value(ba):
            ba[99] = 300

        def set_out_of_range_index(ba):
            ba[99] = 1

        def del_out_of_range_index(ba):
            del ba[99]

        self.assertEqual(
            self._error_from_traced(set_bad_value_type),
            ("TypeError", "'str' object cannot be interpreted as an integer"),
        )
        self.assertEqual(
            self._error_from_traced(set_out_of_range_value),
            ("ValueError", "byte must be in range(0, 256)"),
        )
        self.assertEqual(
            self._error_from_traced(set_out_of_range_index),
            ("IndexError", "bytearray index out of range"),
        )
        self.assertEqual(
            self._error_from_traced(del_out_of_range_index),
            ("IndexError", "bytearray index out of range"),
        )

    def test_setitem_wraps_index_once(self):
        # The sequence slot wraps a negative index once. A second wrap would
        # turn -5 on a 3-byte buffer into index 1 and silently overwrite the
        # wrong byte.
        def set_far_negative(ba):
            try:
                ba[-5] = 1
            except IndexError as e:
                return str(e), bytes(ba)
            return None, bytes(ba)

        def del_far_negative(ba):
            try:
                del ba[-5]
            except IndexError as e:
                return str(e), bytes(ba)
            return None, bytes(ba)

        for fn in (set_far_negative, del_far_negative):
            eager_buf = bytearray(b"abc")
            expected = fn(eager_buf)
            self.assertEqual(expected, ("bytearray index out of range", b"abc"))

            buf = bytearray(b"abc")
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(buf), expected
            )
            self.assertEqual(buf, eager_buf)

    def test_setitem_entry_points(self):
        import operator

        def f(x):
            ba = bytearray(b"abc")
            operator.setitem(ba, 0, 65)
            ba.__setitem__(1, 66)
            operator.delitem(ba, -1)
            return bytes(ba), x + 1

        self.assertEqual(
            torch.compile(f, backend="eager", fullgraph=True)(torch.ones(1)),
            f(torch.ones(1)),
        )

    def test_setitem_same_bytearray_twice(self):
        # Both arguments alias one buffer, so the two stores must land in the
        # same tracking state and both be visible to the caller.
        def f(ba, alias, x):
            x = x + 1
            ba[0] = 65
            alias[1] = 66
            return bytes(ba), bytes(alias), x * 2

        eager_buf = bytearray(b"abc")
        expected = f(eager_buf, eager_buf, torch.ones(1))

        buf = bytearray(b"abc")
        self.assertEqual(
            torch.compile(f, backend="eager", fullgraph=True)(buf, buf, torch.ones(1)),
            expected,
        )
        self.assertEqual(buf, eager_buf)

    # -- Slice assignment and deletion --

    @make_dynamo_test
    def test_setslice(self):
        b = self.type2test(range(10))
        self.assertEqual(list(b), list(range(10)))

        b[0:5] = bytearray([1, 1, 1, 1, 1])
        self.assertEqual(b, bytearray([1, 1, 1, 1, 1, 5, 6, 7, 8, 9]))

        del b[0:-5]
        self.assertEqual(b, bytearray([5, 6, 7, 8, 9]))

        b[0:0] = bytearray([0, 1, 2, 3, 4])
        self.assertEqual(b, bytearray(range(10)))

        b[-7:-3] = bytearray([100, 101])
        self.assertEqual(b, bytearray([0, 1, 2, 100, 101, 7, 8, 9]))

        b[3:5] = [3, 4, 5, 6]
        self.assertEqual(b, bytearray(range(10)))

        b[3:0] = [42, 42, 42]
        self.assertEqual(b, bytearray([0, 1, 2, 42, 42, 42, 3, 4, 5, 6, 7, 8, 9]))

        b[3:] = b"foo"
        self.assertEqual(b, bytearray([0, 1, 2, 102, 111, 111]))

        # CPython's test_repeat_after_setslice: repeating the resized buffer.
        self.assertEqual(bytes(b * 3), bytes(b) * 3)

        # CPython assigns memoryview(b"foo") here; memoryview construction is
        # unsupported, so keep the same buffer contents with plain bytes.
        b[:3] = b"foo"
        self.assertEqual(b, bytearray([102, 111, 111, 102, 111, 111]))

        b[3:4] = []
        self.assertEqual(b, bytearray([102, 111, 111, 111, 111]))

        for elem in [5, -5, 0, int(10e20), "str", 2.3, ["a", "b"], [b"a", b"b"], [[]]]:
            with self.assertRaises(TypeError):
                b[3:4] = elem

        for elem in [[254, 255, 256], [-256, 9000]]:
            with self.assertRaises(ValueError):
                b[3:4] = elem

    @make_dynamo_test
    def test_setslice_extend(self):
        # Exercise the resizing logic (see CPython issue #19087).
        b = self.type2test(range(100))
        self.assertEqual(list(b), list(range(100)))
        del b[:10]
        self.assertEqual(list(b), list(range(10, 100)))
        b.extend(range(100, 110))
        self.assertEqual(list(b), list(range(10, 110)))

    @make_dynamo_test
    def test_setslice_trap(self):
        # Assigning self to a slice of self (the old Lambert Meertens trap).
        b = self.type2test(range(256))
        b[8:] = b
        self.assertEqual(b, bytearray(list(range(8)) + list(range(256))))

    def test_extended_set_del_slice(self):
        # Same index matrix and expected values as CPython's
        # test_extended_set_del_slice; the per-step comparisons stay inside the
        # compiled frame.
        indices = (
            0,
            None,
            1,
            3,
            19,
            300,
            1 << 333,
            sys.maxsize,
            -1,
            -2,
            -31,
            -300,
        )

        def f():
            failures = []
            for start in indices:
                for stop in indices:
                    # Skip invalid step 0.
                    for step in indices[1:]:
                        expected = list(range(255))
                        b = bytearray(expected)
                        # A slice of exactly the right length, but with
                        # different data, so a length mismatch would be caught.
                        data = expected[start:stop:step]
                        data.reverse()
                        expected[start:stop:step] = data
                        b[start:stop:step] = data
                        if b != bytearray(expected):
                            failures.append((start, stop, step, "assign"))

                        del expected[start:stop:step]
                        del b[start:stop:step]
                        if b != bytearray(expected):
                            failures.append((start, stop, step, "delete"))
            return failures

        self.assertEqual(torch.compile(f, backend="eager", fullgraph=True)(), [])

    def test_setslice_iterable_sources(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(ba):
            ba[0:1] = [65]
            ba[1:2] = (66,)
            ba[2:3] = range(67, 68)
            ba[3:4] = (x for x in [68])
            ba[4:5] = bytearray(b"E")
            return bytes(ba)

        self.assertEqual(f(bytearray(b"abcde")), b"ABCDE")

    def test_setslice_consumes_iterator(self):
        def f(x, values):
            ba = bytearray(b"abc")
            it = iter(values)
            try:
                ba[0:1] = it
            except ValueError as e:
                return bytes(ba), list(it), str(e), x + 1
            return bytes(ba), list(it), None, x + 1

        x = torch.ones(1)
        for values in ([65, 66], (65, 66), [65, 300, 66], [Indexable(65)]):
            with self.subTest(values=values):
                self.assertEqual(
                    torch.compile(f, backend="eager", fullgraph=True)(x, values),
                    f(x, values),
                )

    def test_setslice_rejects_number_before_iteration(self):
        class Number:
            def __index__(self):
                return 1

            def __iter__(self):
                raise RuntimeError("must not iterate")

        class IterableComplex(complex):
            def __iter__(self):
                return iter([65])

        def f(x, number):
            ba = bytearray(b"abc")
            try:
                ba[:] = number
            except TypeError as e:
                return str(e), bytes(ba), x + 1
            return None, bytes(ba), x + 1

        x = torch.ones(1)
        for number in (Number(), IterableComplex(1, 2)):
            with self.subTest(number=type(number).__name__):
                self.assertEqual(
                    torch.compile(f, backend="eager", fullgraph=True)(x, number),
                    f(x, number),
                )

    def test_input_mutation_preserves_buffer_error(self):
        def delete_item(ba):
            del ba[0]

        def grow_slice(ba):
            ba[0:1] = b"XY"

        def delete_extended(ba):
            del ba[::2]

        def shrink_slice(ba):
            ba[0:2] = b"X"

        for op in (
            delete_item,
            grow_slice,
            delete_extended,
            shrink_slice,
        ):
            with self.subTest(op=op.__name__):
                with self.assertRaisesRegex(
                    torch._dynamo.exc.Unsupported, "unobservable buffer export"
                ):
                    torch.compile(op, backend="eager", fullgraph=True)(
                        bytearray(b"abc")
                    )

            def f(ba, x):
                x = x + 1
                try:
                    op(ba)
                except BufferError as e:
                    return str(e), bytes(ba), x * 2
                return None, bytes(ba), x * 2

            for exported in (False, True):
                with self.subTest(op=op.__name__, exported=exported):
                    eager = bytearray(b"abc")
                    actual = bytearray(b"abc")
                    views = (memoryview(eager), memoryview(actual)) if exported else ()
                    x = torch.ones(1)
                    self.assertEqual(
                        torch.compile(f, backend="eager")(actual, x), f(eager, x)
                    )
                    self.assertEqual(actual, eager)
                    for view in views:
                        view.release()

    def test_setslice_reads_buffer_before_iterator(self):
        import array
        import ctypes

        class Array(array.array):
            def __iter__(self):
                return iter([90])

        class Buffer:
            def __buffer__(self, flags):
                return memoryview(b"AB")

            def __iter__(self):
                return iter([90])

        class CBuffer(ctypes.Array):
            _type_ = ctypes.c_char
            _length_ = 2

            def __iter__(self):
                return iter([90])

        def f(value, x):
            ba = bytearray(b"abc")
            ba[0:1] = value
            return bytes(ba), x + 1

        values = [Array("B", [65, 66]), CBuffer(b"A", b"B")]
        if sys.version_info >= (3, 12):
            values.append(Buffer())
        x = torch.ones(1)
        for value in values:
            with self.subTest(value=type(value).__name__):
                with self.assertRaisesRegex(
                    torch._dynamo.exc.Unsupported, "unsupported buffer"
                ):
                    torch.compile(f, backend="eager", fullgraph=True)(value, x)
                torch._dynamo.reset()
                self.assertEqual(
                    torch.compile(f, backend="eager")(value, x), f(value, x)
                )

    def test_setslice_indexable_bounds(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(ba):
            ba[Indexable(1) : Indexable(3)] = b"XY"
            return bytes(ba)

        self.assertEqual(f(bytearray(b"abcd")), b"aXYd")

    def test_setslice_changing_bounds(self):
        class Bound:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        def f(x, start):
            ba = bytearray(b"abc")
            ba[Bound(start) : 2] = b"Z"
            return bytes(ba), x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.ones(1)
        self.assertEqual(compiled(x, 0), f(x, 0))
        # Changing the bound exercises automatic integer generalization.
        self.assertEqual(compiled(x, 1), f(x, 1))

    @parametrize("rhs_kind", ["list", "bytes", "self", "other", "invalid"])
    def test_setslice_callback_reentry(self, rhs_kind):
        class Bound:
            def __init__(self):
                self.calls = 0

            def __index__(self):
                result = self.calls
                self.calls += 1
                return result

        def f(x):
            ba = bytearray(b"abc")
            bound = Bound()
            if rhs_kind == "list":
                value = [90]
            elif rhs_kind == "bytes":
                value = b"Z"
            elif rhs_kind == "self":
                value = ba
            elif rhs_kind == "other":
                value = bytearray(b"Z")
            else:
                value = 3
            try:
                ba[bound:2] = value
            except TypeError as e:
                return bytes(ba), bound.calls, str(e), x + 1
            return bytes(ba), bound.calls, None, x + 1

        x = torch.ones(1)
        self.assertEqual(torch.compile(f, backend="eager", fullgraph=True)(x), f(x))

    @parametrize("step", [0, 1])
    def test_setslice_callback_order(self, step):
        class Bound:
            def __init__(self, value, name, seen):
                self.value = value
                self.name = name
                self.seen = seen

            def __index__(self):
                self.seen.append(self.name)
                return self.value

        def f(x):
            ba = bytearray(b"abc")
            seen = []
            key = slice(
                Bound(0, "start", seen),
                Bound(2, "stop", seen),
                Bound(step, "step", seen),
            )
            try:
                ba[key] = [90]
            except ValueError as e:
                return bytes(ba), seen, str(e), x + 1
            return bytes(ba), seen, None, x + 1

        x = torch.ones(1)
        self.assertEqual(torch.compile(f, backend="eager", fullgraph=True)(x), f(x))

    def test_delslice(self):
        def f(ba):
            del ba[::2]
            del ba[-2:]
            return bytes(ba)

        eager_buf = bytearray(b"abcdefgh")
        expected = f(eager_buf)

        buf = bytearray(b"abcdefgh")
        cnt = CompileCounter()
        self.assertEqual(torch.compile(f, backend=cnt)(buf), expected)
        self.assertEqual(buf, eager_buf)
        self.assertEqual(cnt.frame_count, 0)

        def g(ba):
            del ba[:]
            return bytes(ba)

        buf = bytearray(b"abc")
        self.assertEqual(torch.compile(g, backend="eager")(buf), b"")
        self.assertEqual(buf, bytearray())

        def local(data, x):
            ba = bytearray(data)
            del ba[::2]
            return bytes(ba), x + 1

        self.assertEqual(
            torch.compile(local, backend="eager", fullgraph=True)(
                b"abcdefgh", torch.ones(1)
            ),
            (b"bdfh", torch.full((1,), 2.0)),
        )

    def test_setslice_arg_mutation(self):
        def f(ba):
            ba[1:2] = b"XY"
            return len(ba)

        cnt = CompileCounter()
        compiled = torch.compile(f, backend=cnt)

        eager_buf = bytearray(b"abc")
        expected = f(eager_buf)
        buf = bytearray(b"abc")
        self.assertEqual(compiled(buf), expected)
        self.assertEqual(buf, eager_buf)
        # Growing a sourced buffer falls back to eager.
        self.assertEqual(cnt.frame_count, 0)
        # A second call on the updated buffer still mutates the caller.
        expected = f(eager_buf)
        self.assertEqual(compiled(buf), expected)
        self.assertEqual(buf, eager_buf)

    def test_setslice_empty_extended_keeps_buffer_error(self):
        # bytearray_ass_subscript runs _canresize before the empty-slice early
        # return, so a live export raises BufferError even when nothing changes.
        # A traced copy has no exports, so a sourced bytearray must stay eager
        # here rather than silently dropping the error.
        def f(ba, x):
            x = x + 1
            try:
                del ba[0:0:2]
            except BufferError:
                return "caught", x
            return "ok", x

        compiled = torch.compile(f, backend="eager")
        buf = bytearray(b"abc")
        with memoryview(buf):
            self.assertEqual(
                compiled(buf, torch.ones(1)), ("caught", torch.full((1,), 2.0))
            )
        self.assertEqual(buf, bytearray(b"abc"))

        def g(ba, x):
            x = x + 1
            try:
                ba[0:0:2] = b""
            except BufferError:
                return "caught", x
            return "ok", x

        compiled_g = torch.compile(g, backend="eager")
        buf2 = bytearray(b"abc")
        with memoryview(buf2):
            self.assertEqual(
                compiled_g(buf2, torch.ones(1)), ("caught", torch.full((1,), 2.0))
            )
        self.assertEqual(buf2, bytearray(b"abc"))

        # A bytearray built inside the traced function cannot have exports, so
        # it still compiles instead of falling back for every empty extended
        # slice.
        def local():
            b = bytearray(b"abc")
            del b[0:0:2]
            return bytes(b)

        self.assertEqual(
            torch.compile(local, backend="eager", fullgraph=True)(), b"abc"
        )

    def test_setslice_single_frame(self):
        cnt = CompileCounter()

        def f(data, x):
            ba = bytearray(data)
            x = x + 1
            ba[0:1] = b"XY"
            del ba[::2]
            return bytes(ba), x * 2

        compiled = torch.compile(f, backend=cnt)

        self.assertEqual(compiled(b"abc", torch.ones(1)), f(b"abc", torch.ones(1)))
        # The slice store and delete are traced, so the tensor ops on either
        # side stay in the same frame.
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_setslice_error_messages(self):
        # The store runs the real operation on a copy, so the messages are
        # CPython's (bytearray_ass_subscript).
        def extended_size_mismatch(ba):
            ba[::2] = b"X"

        def step_zero(ba):
            ba[::0] = b"X"

        def del_step_zero(ba):
            del ba[::0]

        def non_iterable(ba):
            ba[0:1] = 5

        def bad_element(ba):
            ba[0:1] = [256]

        self.assertEqual(
            self._error_from_traced(extended_size_mismatch),
            (
                "ValueError",
                "attempt to assign bytes of size 1 to extended slice of size 2",
            ),
        )
        self.assertEqual(
            self._error_from_traced(step_zero),
            ("ValueError", "slice step cannot be zero"),
        )
        self.assertEqual(
            self._error_from_traced(del_step_zero),
            ("ValueError", "slice step cannot be zero"),
        )
        self.assertEqual(
            self._error_from_traced(non_iterable),
            (
                "TypeError",
                "can assign only bytes, buffers, or iterables of ints in range(0, 256)",
            ),
        )
        self.assertEqual(
            self._error_from_traced(bad_element),
            ("ValueError", "byte must be in range(0, 256)"),
        )

    def test_setslice_dynamic_value_graph_break(self):
        def f(x):
            ba = bytearray(b"abc")
            x = x + 1
            ba[0:1] = [x[0]]
            return bytes(ba), x * 2

        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "non-constant elements"
            ):
                torch.compile(f, backend="eager", fullgraph=True)(torch.tensor([89]))

    def test_setitem_dynamic_value_graph_break(self):
        def f(x):
            ba = bytearray(b"abc")
            x = x + 1
            ba[0] = x[0]
            return bytes(ba), x * 2

        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "non-constant byte value"
            ):
                torch.compile(f, backend="eager", fullgraph=True)(torch.tensor([89]))

    def test_setslice_zero_step_does_not_consume_iterable(self):
        class Values:
            def __init__(self, seen):
                self.seen = seen

            def __iter__(self):
                self.seen.append("iterated")
                return iter([90])

        def f(x):
            seen = []
            ba = bytearray(b"abc")
            try:
                ba[::0] = Values(seen)
            except ValueError:
                pass
            return tuple(seen), bytes(ba), x + 1

        cnt = CompileCounter()
        x = torch.ones(1)
        self.assertEqual(torch.compile(f, backend=cnt, fullgraph=True)(x), f(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_setslice_same_bytearray_twice(self):
        # Both arguments alias one buffer, so the slice stores must land in the
        # same tracking state and both be visible to the caller.
        def f(ba, alias, x):
            x = x + 1
            ba[0:1] = b"X"
            alias[1:2] = b"Z"
            return bytes(ba), bytes(alias), x * 2

        eager_buf = bytearray(b"abc")
        expected = f(eager_buf, eager_buf, torch.ones(1))

        buf = bytearray(b"abc")
        self.assertEqual(
            torch.compile(f, backend="eager", fullgraph=True)(buf, buf, torch.ones(1)),
            expected,
        )
        self.assertEqual(buf, eager_buf)

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


if __name__ == "__main__":
    run_tests()
