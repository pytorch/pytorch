# Owner(s): ["module: dynamo"]
"""Tests for ByteArrayVariable: bytearray support in Dynamo.

Ported from CPython's Lib/test/test_bytes.py (BaseBytesTest class).
"""

import operator
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


# CPython's test_translate builds this table with item assignment, so build
# it once at import time instead.
_rosetta = bytearray(range(256))
_rosetta[ord("o")] = ord("e")


class Indexable:
    def __init__(self, value=0):
        self.value = value

    def __index__(self):
        return self.value


class ByteArrayTest(torch._dynamo.test_case.TestCase):
    """bytearray-specific tests, ported from CPython ByteArrayTest."""

    def assertGraphBreakReason(self, reason):
        # A break inside a try block is counted as unimplemented.
        reasons = [
            *torch._dynamo.utils.counters["graph_break"],
            *torch._dynamo.utils.counters["unimplemented"],
        ]
        self.assertTrue(any(reason in r for r in reasons), reasons)

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
    def test_join(self):
        self.assertEqual(self.type2test(b"").join([]), b"")
        self.assertEqual(self.type2test(b"").join([b""]), b"")
        for lst in [[b"abc"], [b"a", b"bc"], [b"ab", b"c"], [b"a", b"b", b"c"]]:
            lst = list(map(self.type2test, lst))
            self.assertEqual(self.type2test(b"").join(lst), b"abc")
            self.assertEqual(self.type2test(b"").join(tuple(lst)), b"abc")
            self.assertEqual(self.type2test(b"").join(iter(lst)), b"abc")

        dot_join = self.type2test(b".:").join
        self.assertEqual(dot_join([b"ab", b"cd"]), b"ab.:cd")
        self.assertEqual(dot_join([bytearray(b"ab"), b"cd"]), b"ab.:cd")
        self.assertEqual(dot_join([b"ab", bytearray(b"cd")]), b"ab.:cd")
        self.assertEqual(dot_join(x for x in (b"a", b"b")), b"a.:b")

        # The result is a new bytearray, also for a single item.
        self.assertIsInstance(dot_join([b"ab", b"cd"]), self.type2test)
        ba = self.type2test(b"ab")
        self.assertIsNot(self.type2test(b"").join([ba]), ba)

    @make_dynamo_test
    def test_join_errors(self):
        with self.assertRaisesRegex(TypeError, "can only join an iterable"):
            self.type2test(b" ").join(None)
        with self.assertRaisesRegex(TypeError, "can only join an iterable"):
            self.type2test(b" ").join(0)
        dot_join = self.type2test(b".:").join
        with self.assertRaisesRegex(
            TypeError, "sequence item 1: expected a bytes-like object, str found"
        ):
            dot_join([bytearray(b"ab"), "cd", b"ef"])

    @parametrize(
        "error_case",
        [
            "first_type_error",
            "first_type_error_subclass",
            "first_value_error",
            "second_type_error",
            "next_type_error",
            "next_value_error",
        ],
    )
    @parametrize("outer", [False, True])
    def test_join_iterator_errors(self, error_case, outer):
        class MyTypeError(TypeError):
            pass

        class Items:
            def __init__(self, iter_error=None, next_error=None):
                self.iter_error = iter_error
                self.next_error = next_error

            def __iter__(self):
                if self.iter_error is not None:
                    raise self.iter_error
                return self

            def __next__(self):
                raise self.next_error

        class Outer:
            def __init__(self, inner):
                self.inner = inner

            def __iter__(self):
                return self.inner

        def join_error(make_items):
            try:
                bytearray(b",").join(make_items())
            except Exception as e:
                return type(e), str(e), type(e.__context__), e.__suppress_context__
            return None

        def f(make_items, outer, x):
            if outer:
                # Errors raised while another exception is handled chain to it.
                try:
                    raise ValueError("outer")
                except ValueError:
                    return join_error(make_items), x + 1
            return join_error(make_items), x + 1

        make_items = {
            "first_type_error": lambda: Items(iter_error=TypeError("first iter")),
            "first_type_error_subclass": lambda: Items(
                iter_error=MyTypeError("first iter")
            ),
            "first_value_error": lambda: Items(iter_error=ValueError("first iter")),
            "second_type_error": lambda: Outer(
                Items(iter_error=TypeError("second iter"))
            ),
            "next_type_error": lambda: Items(next_error=TypeError("next")),
            "next_value_error": lambda: Items(next_error=ValueError("next")),
        }[error_case]
        x = torch.ones(1)
        torch._dynamo.reset()
        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(make_items, outer, x), f(make_items, outer, x))

    def test_join_length_hint_first_iterator(self):
        seen = []

        class Second:
            def __init__(self):
                self.done = False

            def __iter__(self):
                return self

            def __length_hint__(self):
                seen.append("second hint")
                return 1

            def __next__(self):
                if self.done:
                    raise StopIteration
                self.done = True
                return b"A"

        class First:
            def __init__(self):
                self.second = Second()

            def __iter__(self):
                return self.second

            def __length_hint__(self):
                seen.append("first hint")
                return 1

            def __next__(self):
                raise StopIteration

        class Iterable:
            def __init__(self):
                self.first = First()

            def __iter__(self):
                return self.first

        def f(iterable, x):
            seen.clear()
            return bytearray(b",").join(iterable), list(seen), x + 1

        x = torch.ones(1)
        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(Iterable(), x), f(Iterable(), x))

    def test_join_length_hint(self):
        class Iterator:
            def __init__(self):
                self.seen = []
                self.done = False

            def __iter__(self):
                self.seen.append("iter")
                return self

            def __length_hint__(self):
                self.seen.append("hint")
                return 1

            def __next__(self):
                self.seen.append("next")
                if self.done:
                    raise StopIteration
                self.done = True
                return b"A"

        def f(x):
            it = Iterator()
            return bytearray(b",").join(it), it.seen, x + 1

        x = torch.ones(1)
        self.assertEqual(torch.compile(f, backend="eager", fullgraph=True)(x), f(x))

    @parametrize(
        "length_source,error_kind",
        [
            ("len", "type_error"),
            ("len", "type_error_subclass"),
            ("len", "value_error"),
            ("hint", "type_error"),
            ("hint", "type_error_subclass"),
            ("hint", "value_error"),
        ],
    )
    def test_join_length_hint_errors(self, length_source, error_kind):
        class MyTypeError(TypeError):
            pass

        class Items:
            def __init__(self, error):
                self.error = error
                self.items = [b"a", b"b"]

            def __iter__(self):
                return self

            def __next__(self):
                if not self.items:
                    raise StopIteration
                return self.items.pop(0)

        class LenRaises(Items):
            def __len__(self):
                raise self.error

        class HintRaises(Items):
            def __length_hint__(self):
                raise self.error

        def f(cls, error_type, x):
            try:
                return bytearray(b",").join(cls(error_type("x"))), x + 1
            except Exception as e:
                return type(e), str(e), x + 1

        cls = LenRaises if length_source == "len" else HintRaises
        error_type = {
            "type_error": TypeError,
            "type_error_subclass": MyTypeError,
            "value_error": ValueError,
        }[error_kind]
        x = torch.ones(1)
        # A TypeError, subclasses included, selects the default hint; any
        # other error propagates.
        torch._dynamo.reset()
        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(cls, error_type, x), f(cls, error_type, x))

    def test_join_separator_mutation_graph_break(self):
        class Items:
            def __init__(self, sep):
                self.sep = sep
                self.items = [b"A", b"B"]

            def __iter__(self):
                return self

            def __next__(self):
                self.sep.extend(b"")
                if not self.items:
                    raise StopIteration
                return self.items.pop(0)

        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(sep, x):
            x = x + 1
            return sep.join(Items(sep)), x * 2

        self.assertEqual(
            f(bytearray(b","), torch.ones(1)), (b"A,B", torch.full((1,), 4.0))
        )
        # Mutating the separator during join is left to eager.
        self.assertEqual(cnt.frame_count, 2)
        self.assertGraphBreakReason("separator mutated during iteration")

    def test_join_nonconstant_item_graph_break(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(sep, item, x):
            x = x + 1
            return sep.join([item, b"c"]), x * 2

        result = f(bytearray(b"-"), memoryview(b"ab"), torch.ones(1))
        self.assertEqual(result, (b"ab-c", torch.full((1,), 4.0)))
        self.assertEqual(cnt.frame_count, 2)
        self.assertGraphBreakReason("join with an unsupported item")

    @parametrize("item_kind", ["bytes_iterator", "str_iterator"])
    def test_join_iterator_item_error(self, item_kind):
        # Eager names the real iterator type in the error.
        def f(make_item, x):
            try:
                bytearray(b",").join([make_item()])
            except TypeError as e:
                return str(e), x + 1
            return None, x + 1

        make_item = (
            (lambda: iter(b"ab"))
            if item_kind == "bytes_iterator"
            else (lambda: iter("ab"))
        )
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        compiled = torch.compile(f, backend="eager")
        self.assertEqual(
            compiled(make_item, torch.ones(1)), f(make_item, torch.ones(1))
        )
        self.assertGraphBreakReason("join with an unsupported item")

    def test_join_buffer_subclass_item(self):
        if sys.version_info < (3, 12):
            self.skipTest("a Python __buffer__ needs PEP 688, Python 3.12+")
        log = []

        class BufStr(str):
            __slots__ = ()

            def __buffer__(self, flags):
                log.append("buffer")
                return memoryview(b"S")

        def f(item, x):
            return bytearray(b"-").join([item, b"c"]), x + 1

        compiled = torch.compile(f, backend="eager")
        for _ in range(2):
            self.assertEqual(compiled(BufStr("q"), torch.ones(1))[0], b"S-c")
        # __buffer__ runs on every call, as in eager.
        self.assertEqual(log, ["buffer", "buffer"])
        self.assertGraphBreakReason("join with an unsupported item")

    @parametrize("iterator_kind", ["next", "iter"])
    def test_join_separator_mutation_then_raise(self, iterator_kind):
        class NextRaises:
            def __init__(self, sep):
                self.sep = sep

            def __iter__(self):
                return self

            def __next__(self):
                self.sep.extend(b"x")
                raise ValueError("next")

        class IterRaises:
            def __init__(self, sep):
                self.sep = sep

            def __iter__(self):
                self.sep.extend(b"x")
                raise TypeError("iter")

        def f(sep, items, x):
            try:
                sep.join(items)
            except (TypeError, ValueError) as e:
                return type(e), x + 1
            return None, x + 1

        def run(fn, cls):
            sep = bytearray(b",")
            try:
                result = fn(sep, cls(sep), torch.ones(1))[0]
            except BufferError:
                result = BufferError
            return result, sep

        cls = NextRaises if iterator_kind == "next" else IterRaises
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        compiled = torch.compile(f, backend="eager")
        self.assertEqual(run(compiled, cls), run(f, cls))
        self.assertGraphBreakReason("separator mutated during iteration")

    @parametrize(
        "mutator_name",
        [
            "extend",
            "iadd",
            "imul",
            "append",
            "insert",
            "pop",
            "iadd_resize",
            "imul_resize",
        ],
    )
    def test_join_separator_mutators(self, mutator_name):
        class Items:
            def __init__(self, sep, mutate):
                self.sep = sep
                self.mutate = mutate
                self.items = [b"A", b"B"]

            def __iter__(self):
                return self

            def __next__(self):
                self.mutate(self.sep)
                if not self.items:
                    raise StopIteration
                return self.items.pop(0)

        def f(sep, mutate, x):
            try:
                return sep.join(Items(sep, mutate)), x + 1
            except BufferError as e:
                return str(e), x + 1

        # Resizing mutators raise BufferError in eager while join holds the
        # export; the others do not, but Dynamo still leaves them to eager.
        mutators = {
            "extend": lambda b: b.extend(b""),
            "iadd": lambda b: operator.iadd(b, b""),
            "imul": lambda b: operator.imul(b, 1),
            "append": lambda b: b.append(0x2C),
            "insert": lambda b: b.insert(0, 0x2C),
            "pop": lambda b: b.pop(),
            "iadd_resize": lambda b: operator.iadd(b, b"x"),
            "imul_resize": lambda b: operator.imul(b, 2),
        }
        mutate = mutators[mutator_name]
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        sep, compiled_sep = bytearray(b","), bytearray(b",")
        compiled = torch.compile(f, backend="eager")
        self.assertEqual(
            compiled(compiled_sep, mutate, torch.ones(1)),
            f(sep, mutate, torch.ones(1)),
        )
        self.assertEqual(compiled_sep, sep)
        self.assertGraphBreakReason("separator mutated during iteration")

    def test_join_other_iterables(self):
        class Sequence:
            def __getitem__(self, i):
                if i >= 2:
                    raise IndexError(i)
                return (b"a", b"b")[i]

        def f(x):
            sep = bytearray(b",")
            return (
                sep.join({b"a": 1, b"b": 2}),
                sep.join({b"a"}),
                sep.join(Sequence()),
                sep.join(reversed([b"a", b"b"])),
                sep.join(iter(bytearray())),
                x + 1,
            )

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.ones(1)), f(torch.ones(1)))

    @parametrize("length_case", ["normal", "value_error", "negative", "type_error"])
    def test_join_sequence_length(self, length_case):
        class Sequence:
            def __init__(self, log):
                self.log = log

            def __len__(self):
                self.log.append("len")
                if length_case == "value_error":
                    raise ValueError("sequence length failed")
                if length_case == "type_error":
                    raise TypeError("sequence length unavailable")
                return -1 if length_case == "negative" else 2

            def __getitem__(self, index):
                self.log.append(index)
                if index >= 2:
                    raise IndexError(index)
                return (b"a", b"b")[index]

        def f(x):
            log = []
            try:
                result = bytearray(b",").join(Sequence(log))
            except ValueError as exc:
                return str(exc), log, x + 1
            return result, log, x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.ones(1)), f(torch.ones(1)))

    @parametrize(
        "length_case", ["normal", "value_error", "negative", "type_error", "shrinks"]
    )
    def test_join_reversed_sequence_length(self, length_case):
        class Sequence:
            def __init__(self):
                self.calls = 0
                self.log = []

            def __len__(self):
                self.calls += 1
                self.log.append("len")
                if self.calls > 1:
                    if length_case == "value_error":
                        raise ValueError("sequence length failed")
                    if length_case == "type_error":
                        raise TypeError("sequence length unavailable")
                    if length_case == "negative":
                        return -1  # noqa: PLE0303
                    if length_case == "shrinks":
                        return 0
                return 2

            def __getitem__(self, index):
                self.log.append(index)
                return (b"a", b"b")[index]

        def f(x):
            seq = Sequence()
            try:
                result = bytearray(b",").join(reversed(seq))
            except ValueError as exc:
                result = str(exc)
            return result, seq.log, x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.ones(1)), f(torch.ones(1)))

    @parametrize("metaclass_kind", ["hides", "descriptor"])
    def test_join_sequence_length_ignores_metaclass_lookup(self, metaclass_kind):
        class Meta(type):
            def __getattribute__(cls, name):
                if metaclass_kind == "hides" and name == "__len__":
                    raise AttributeError(name)
                return super().__getattribute__(name)

            @property
            def __len__(cls):
                raise AssertionError("metaclass lookup must not run")

        class Sequence(metaclass=Meta):
            def __len__(self):
                raise ValueError("sequence length failed")

            def __getitem__(self, index):
                if index >= 2:
                    raise IndexError(index)
                return (b"a", b"b")[index]

        def f(seq, x):
            try:
                result = bytearray(b",").join(seq)
            except ValueError as exc:
                return str(exc), x + 1
            return result, x + 1

        counter = CompileCounter()
        compiled = torch.compile(f, backend=counter, fullgraph=True)
        self.assertEqual(
            compiled(Sequence(), torch.ones(1)), f(Sequence(), torch.ones(1))
        )
        self.assertEqual(counter.frame_count, 1)

    @parametrize("consumer", ["extend", "length_hint"])
    def test_sequence_iterator_with_custom_metaclass(self, consumer):
        class Meta(type):
            def __getattribute__(cls, name):
                return super().__getattribute__(name)

        class Sequence(metaclass=Meta):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                if index >= 2:
                    raise IndexError(index)
                return (97, 98)[index]

        def f(seq, x):
            iterator = iter(seq)
            if consumer == "extend":
                result = bytearray()
                result.extend(iterator)
            else:
                result = operator.length_hint(iterator)
            return result, x + 1

        counter = CompileCounter()
        compiled = torch.compile(f, backend=counter, fullgraph=True)
        self.assertEqual(
            compiled(Sequence(), torch.ones(1)), f(Sequence(), torch.ones(1))
        )
        self.assertEqual(counter.frame_count, 1)

    def test_reversed_sequence_iterator_length_hint(self):
        class Sequence:
            def __init__(self):
                self.items = [b"a", b"b", b"c"]
                self.calls = 0

            def __len__(self):
                self.calls += 1
                return len(self.items)

            def __getitem__(self, index):
                return self.items[index]

        def f(x):
            seq = Sequence()
            it = reversed(seq)
            hints = [operator.length_hint(it)]
            next(it)
            hints.append(operator.length_hint(it))
            seq.items.append(b"d")
            hints.append(operator.length_hint(it))
            seq.items[1:] = []
            hints.append(operator.length_hint(it))
            seq.items.extend([b"b", b"c"])
            hints.append(operator.length_hint(it))
            consumed = next(it), next(it)
            # Before 3.14 the final yield still permits another length query.
            hints.append(operator.length_hint(it))
            next(it, None)
            hints.append(operator.length_hint(it))
            return hints, consumed, seq.calls, x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.ones(1)), f(torch.ones(1)))

    @parametrize("size", [0, 1])
    def test_reversed_sequence_iterator_length_hint_at_end(self, size):
        class Sequence:
            def __init__(self):
                self.fail = False
                self.calls = 0

            def __len__(self):
                self.calls += 1
                if self.fail:
                    raise ValueError("length queried")
                return size

            def __getitem__(self, index):
                return b"a"

        def f(x):
            seq = Sequence()
            it = reversed(seq)
            if size:
                next(it)
            seq.fail = True
            try:
                hint = operator.length_hint(it)
            except ValueError as exc:
                hint = str(exc)
            return hint, seq.calls, x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.ones(1)), f(torch.ones(1)))

    @parametrize("error", [IndexError, StopIteration, ValueError])
    def test_reversed_sequence_iterator_length_hint_after_error(self, error):
        class Sequence:
            def __init__(self):
                self.calls = 0
                self.failed = False

            def __len__(self):
                self.calls += 1
                return 2

            def __getitem__(self, index):
                if not self.failed:
                    self.failed = True
                    raise error("getitem failed")
                return b"a"

        def f(x):
            seq = Sequence()
            it = reversed(seq)
            before = operator.length_hint(it)
            try:
                next(it, None)
            except error:
                pass
            after = operator.length_hint(it)
            result = bytearray(b",").join(it)
            return before, after, seq.calls, result, x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.ones(1)), f(torch.ones(1)))

    def test_sequence_iterator_length_hint(self):
        class Sequence:
            def __init__(self):
                self.items = [b"a", b"b"]

            def __len__(self):
                return len(self.items)

            def __getitem__(self, index):
                return self.items[index]

        def f(x):
            seq = Sequence()
            it = iter(seq)
            hints = [operator.length_hint(it)]
            next(it)
            hints.append(operator.length_hint(it))
            seq.items[:] = []
            hints.append(operator.length_hint(it))
            list(it)
            seq.items.extend([b"c", b"d"])
            hints.append(operator.length_hint(it))
            return hints, x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.ones(1)), f(torch.ones(1)))

    def test_sequence_iterator_length_hint_without_len(self):
        class Sequence:
            def __getitem__(self, index):
                if index >= 1:
                    raise IndexError(index)
                return b"a"

        def f(x):
            it = iter(Sequence())
            before = operator.length_hint(it, 8)
            list(it)
            return before, operator.length_hint(it, 8), x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.ones(1)), f(torch.ones(1)))

    def test_iterator_length_hint(self):
        def f(x):
            ba = bytearray(b"abc")
            it = iter(ba)
            hints = [operator.length_hint(it)]
            next(it)
            hints.append(operator.length_hint(it))
            ba.pop()
            ba.pop()
            ba.pop()
            hints.append(operator.length_hint(it))
            list(it)
            hints.append(operator.length_hint(it))
            ba.extend(b"xyz")
            hints.append(operator.length_hint(it))
            return hints, x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.ones(1)), f(torch.ones(1)))

    def test_list_reverse_iterator_length_hint(self):
        def f(x):
            items = [b"a", b"b", b"c"]
            it = reversed(items)
            hints = [operator.length_hint(it)]
            next(it)
            hints.append(operator.length_hint(it))
            # Growing the list does not change the hint.
            items.append(b"d")
            hints.append(operator.length_hint(it))
            # Shrinking it below the iterator position makes the hint zero.
            items[1:] = []
            hints.append(operator.length_hint(it))
            consumed = list(it)
            hints.append(operator.length_hint(it))
            # Once exhausted, growing the list does not revive the hint.
            items.extend([b"e", b"f"])
            hints.append(operator.length_hint(it))
            return hints, consumed, x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.ones(1)), f(torch.ones(1)))

    def test_input_bytearray_join(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(sep, ba, items, x):
            return sep.join([ba, b"x"]), sep.join(items), sep.join([ba]), x + 1

        def check(frames):
            *results, _ = fn(sep, ba, items, torch.ones(1))
            self.assertEqual(results, [sep.join([ba, b"x"]), sep.join(items), ba])
            self.assertIsNot(results[2], ba)
            for r in results:
                self.assertIs(type(r), bytearray)
            self.assertEqual(cnt.frame_count, frames)

        # Each step changes one input, which must fail its guard and recompile.
        sep, ba, items = bytearray(b"-"), bytearray(b"ab"), [b"ab", bytearray(b"cd")]
        check(1)
        sep = bytearray(b"+")
        check(2)
        ba = bytearray(b"yz")
        check(3)
        items = [b"ef", items[1]]
        check(4)
        items[1][0] = ord("x")
        check(5)
        ba[0] = ord("q")
        check(6)

    def test_join_result_mutated_in_graph(self):
        def f(sep, ba, x):
            out = sep.join([ba, b"c"])
            out += b"!"
            single = sep.join([ba])
            single.append(ord("?"))
            return out, single, x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        sep, ba = bytearray(b"-"), bytearray(b"ab")
        self.assertEqual(
            compiled(sep, ba, torch.ones(1)),
            f(bytearray(b"-"), bytearray(b"ab"), torch.ones(1)),
        )
        self.assertEqual((sep, ba), (b"-", b"ab"))

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

    @make_dynamo_test
    def test_replace(self):
        b = self.type2test(b"mississippi")
        self.assertEqual(b.replace(b"i", b"a"), b"massassappa")
        self.assertEqual(b.replace(b"ss", b"x"), b"mixixippi")
        self.assertEqual(b.replace(b"i", b"a", 2), b"massassippi")
        self.assertEqual(b.replace(b"i", b"a", 0), b"mississippi")
        self.assertEqual(b.replace(b"i", b"a", -1), b"massassappa")
        self.assertEqual(b.replace(b"", b"-"), b"-m-i-s-s-i-s-s-i-p-p-i-")
        self.assertEqual(b.replace(b"x", b"y"), b"mississippi")
        self.assertEqual(b.replace(bytearray(b"ss"), b"x"), b"mixixippi")
        self.assertEqual(b, b"mississippi")
        self.assertIsInstance(b.replace(b"i", b"a"), self.type2test)

    @make_dynamo_test
    def test_replace_errors(self):
        b = self.type2test(b"a b")
        with self.assertRaisesRegex(TypeError, "a bytes-like object is required"):
            b.replace(32, b"")
        with self.assertRaisesRegex(TypeError, "a bytes-like object is required"):
            b.replace(b"a", 32)
        with self.assertRaisesRegex(
            TypeError, "'bytes' object cannot be interpreted as an integer"
        ):
            b.replace(b"a", b"b", b"0")

    @make_dynamo_test
    def test_translate(self):
        b = self.type2test(b"hello")
        self.assertEqual(b.translate(_rosetta), b"helle")
        self.assertEqual(b.translate(_rosetta, b"l"), b"hee")
        self.assertEqual(b.translate(_rosetta, delete=b"l"), b"hee")
        self.assertEqual(b.translate(None, b"e"), b"hllo")
        self.assertEqual(b, b"hello")
        self.assertIsInstance(b.translate(_rosetta), self.type2test)
        with self.assertRaisesRegex(
            ValueError, "translation table must be 256 characters long"
        ):
            b.translate(bytes(range(255)))
        with self.assertRaisesRegex(TypeError, "a bytes-like object is required"):
            b.translate(_rosetta, 1)
        with self.assertRaisesRegex(
            TypeError, r"translate\(\) takes at least 1 positional argument \(0 given\)"
        ):
            b.translate()
        with self.assertRaisesRegex(TypeError, "a bytes-like object is required"):
            b.translate(None, None)

    def test_replace_translate_returns_new_bytearrays(self):
        def f(ba, table, x):
            return ba.replace(b"a", b"z"), ba.translate(table), x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        table = bytearray(range(256))
        ba = bytearray(b"ab")
        result = compiled(ba, table, torch.ones(1))
        self.assertEqual(result, f(bytearray(b"ab"), table, torch.ones(1)))
        self.assertIsNot(result[0], ba)
        self.assertIsNot(result[1], ba)
        self.assertIsNot(result[0], result[1])
        result[0][0] = ord("q")
        self.assertEqual(ba, b"ab")
        self.assertEqual(result[1], b"ab")

    def test_replace_translate_nonconstant_buffer_graph_break(self):
        class BytesSubclass(bytes):
            pass

        def replace(ba, arg):
            return ba.replace(arg, b"z")

        def translate(ba, arg):
            return ba.translate(arg)

        for fn, arg in ((replace, b"a"), (translate, _rosetta)):
            name = fn.__name__
            with self.subTest(method=name):
                args = bytearray(b"ab"), BytesSubclass(arg)
                torch._dynamo.reset()
                with self.assertRaisesRegex(
                    torch._dynamo.exc.Unsupported,
                    f"Dynamo does not know how to trace method `{name}` of class `bytearray`",
                ):
                    torch.compile(fn, backend="eager", fullgraph=True)(*args)
                self.assertEqual(torch.compile(fn, backend="eager")(*args), fn(*args))

    def test_input_bytearray_replace_translate(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(ba, table):
            return ba.replace(b"a", b"z"), ba.translate(table, b"b")

        self.assertEqual(
            fn(bytearray(b"ab"), bytearray(range(256))),
            (bytearray(b"zb"), bytearray(b"a")),
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
