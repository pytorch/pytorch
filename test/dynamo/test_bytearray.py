# Owner(s): ["module: dynamo"]
"""Tests for ByteArrayVariable: bytearray support in Dynamo.

Ported from CPython's Lib/test/test_bytes.py (BaseBytesTest class).
"""

import sys
import types

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


class IndexableInt(int):
    def __index__(self):
        raise AssertionError("int subclasses do not call __index__")


class ClearOnIndex:
    def __init__(self, b):
        self.b = b

    def __index__(self):
        del self.b[:]
        return 0


def setitem(b, key, value):
    b[key] = value


def delitem(b, key):
    del b[key]


global_bytearray = bytearray(b"ab")


def setitem_global():
    global_bytearray[0] = 1


# These captures must predate tracing, including under the Dynamo test wrapper.
@torch._dynamo.disable
def make_sourceless_store_class(binding, container_kind, operation="setitem"):
    value = bytearray(b"abc")
    if container_kind == "direct":
        payload = value
    elif container_kind == "list":
        payload = [value]
    elif container_kind == "dict":
        payload = {"buf": value}
    elif container_kind == "nested_list":
        payload = [[value]]
    else:
        payload = {"x": [value]}

    if binding == "class_attribute":

        class Holder:
            buf = payload

        holder = Holder
    elif binding == "module_attribute":
        holder = types.ModuleType("bytearray_holder")
        holder.buf = payload
    elif binding == "function_defaults":

        def captured_default(target=payload):
            return target

    def write_target(target):
        if container_kind == "list":
            target = target[0]
        elif container_kind == "nested_list":
            target = target[0][0]
        elif container_kind == "nested_dict":
            target = target["x"][0]
        elif container_kind == "dict":
            target = target["buf"]

        if operation == "read":
            return target[0]
        if operation == "setitem":
            target[0] = ord("Z")
        elif operation == "setslice":
            target[1:2] = b"XY"
        elif operation == "delitem":
            del target[0]
        else:
            del target[::2]
        return None

    class Store:
        if binding == "closure":

            def store(self):
                return write_target(payload)
        elif binding == "positional_default":

            def store(self, captured=payload):
                return write_target(captured)
        elif binding == "keyword_default":

            def store(self, *, captured=payload):
                return write_target(captured)
        elif binding == "function_defaults":

            def store(self):
                return write_target(captured_default.__defaults__[0])
        else:

            def store(self):
                return write_target(holder.buf)

    return Store, value


@torch._dynamo.disable
def make_sourceless_resize_store_class():
    value = bytearray(b"abc")

    class Store:
        def store(self):
            try:
                value[1:2] = b"XY"
            except BufferError as e:
                return str(e)
            return None

    return Store, value


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

    @make_dynamo_test
    def test_setitem(self):
        b = self.type2test(b"abc")
        b[0] = ord("x")
        b[-1] = Indexable(ord("z"))
        b[Indexable(1)] = 0xFF
        self.assertEqual(b, b"x\xffz")
        out_of_range = "bytearray index out of range"
        self.assertRaisesRegex(IndexError, out_of_range, setitem, b, 3, 1)
        self.assertRaisesRegex(IndexError, out_of_range, setitem, b, -4, 1)
        self.assertRaisesRegex(
            IndexError,
            "cannot fit 'int' into an index-sized integer",
            setitem,
            b,
            2**64,
            1,
        )
        # The value is checked before the index.
        self.assertRaisesRegex(
            ValueError, r"byte must be in range\(0, 256\)", setitem, b, 10, 256
        )
        self.assertRaisesRegex(
            TypeError, "cannot be interpreted as an integer", setitem, b, 0, b"a"
        )
        self.assertRaisesRegex(
            TypeError,
            "bytearray indices must be integers or slices, not str",
            setitem,
            b,
            "0",
            1,
        )
        self.assertEqual(b, b"x\xffz")

    @make_dynamo_test
    def test_setitem_value_index_mutates(self):
        b = self.type2test(b"abc")
        # The value is converted before the index is checked.
        self.assertRaisesRegex(
            IndexError, "bytearray index out of range", setitem, b, 0, ClearOnIndex(b)
        )
        self.assertEqual(b, b"")

    @make_dynamo_test
    def test_delitem(self):
        b = self.type2test(b"abcd")
        del b[0]
        del b[-1]
        self.assertEqual(b, b"bc")
        self.assertRaisesRegex(
            IndexError, "bytearray index out of range", delitem, b, 2
        )
        self.assertRaisesRegex(
            IndexError, "bytearray index out of range", delitem, b, -3
        )
        self.assertEqual(b, b"bc")

    @make_dynamo_test
    def test_setslice(self):
        b = self.type2test(b"abcdef")
        b[1:3] = b"XYZ"
        self.assertEqual(b, b"aXYZdef")
        b[:2] = [0x31, 0x32]
        self.assertEqual(b, b"12YZdef")
        b[::2] = (0x41, 0x42, 0x43, 0x44)
        self.assertEqual(b, b"A2BZCeD")
        b[5:] = self.type2test()
        self.assertEqual(b, b"A2BZC")
        b[1:] = b
        self.assertEqual(b, b"AA2BZC")
        self.assertRaisesRegex(
            ValueError,
            "attempt to assign bytes of size 1 to extended slice of size 3",
            setitem,
            b,
            slice(None, None, 2),
            b"x",
        )
        self.assertRaisesRegex(
            ValueError,
            "slice step cannot be zero",
            setitem,
            b,
            slice(None, None, 0),
            b"",
        )
        self.assertRaisesRegex(
            ValueError,
            "byte must be in range\\(0, 256\\)",
            setitem,
            b,
            slice(0, 1),
            [256],
        )
        self.assertRaisesRegex(
            TypeError, "can assign only bytes, buffers", setitem, b, slice(0, 1), 1
        )
        self.assertRaisesRegex(
            TypeError, "can assign only bytes, buffers", setitem, b, slice(0, 1), "a"
        )
        self.assertEqual(b, b"AA2BZC")

    @make_dynamo_test
    def test_delslice(self):
        b = self.type2test(b"abcdefg")
        del b[1:3]
        self.assertEqual(b, b"adefg")
        del b[::2]
        self.assertEqual(b, b"df")
        del b[10:]
        self.assertEqual(b, b"df")
        self.assertRaisesRegex(
            ValueError, "slice step cannot be zero", delitem, b, slice(None, None, 0)
        )

    @make_dynamo_test
    def test_setslice_delslice_zero_step_first(self):
        # PySlice_Unpack checks the step before the other bounds.
        b = self.type2test(b"abc")
        zero_step = "slice step cannot be zero"
        self.assertRaisesRegex(ValueError, zero_step, setitem, b, slice("a", 1, 0), b"")
        self.assertRaisesRegex(ValueError, zero_step, delitem, b, slice(1.5, None, 0))
        self.assertEqual(b, b"abc")

    @parametrize(
        "case",
        [
            "int_key",
            "int_slice_store",
            "enum_slice_store",
            "subclass_slice_store",
            "int_slice_list",
            "enum_slice_list",
            "subclass_slice_list",
            "int_slice_delete",
            "enum_slice_delete",
            "subclass_slice_delete",
        ],
    )
    def test_setslice_zero_step_from_input(self, case):
        import enum

        class Step(enum.IntEnum):
            ZERO = 0

        class MyInt(int):
            pass

        def store_key(key, x):
            ba = bytearray(b"abc")
            try:
                ba[key] = b""
            except ValueError as e:
                return str(e), x + 1
            return None, x + 1

        def store_step(step, x):
            ba = bytearray(b"abc")
            try:
                ba["a":1:step] = b""
            except ValueError as e:
                return str(e), x + 1
            return None, x + 1

        def store_list(step, x):
            ba = bytearray(b"abc")
            try:
                ba[::step] = [300]
            except ValueError as e:
                return str(e), x + 1
            return None, x + 1

        def delete_step(step, x):
            ba = bytearray(b"abc")
            try:
                del ba["a":1:step]
            except ValueError as e:
                return str(e), x + 1
            return None, x + 1

        cases = {
            "int_key": (store_key, slice("a", 1, 0)),
            "int_slice_store": (store_step, 0),
            "enum_slice_store": (store_step, Step.ZERO),
            "subclass_slice_store": (store_step, MyInt(0)),
            "int_slice_list": (store_list, 0),
            "enum_slice_list": (store_list, Step.ZERO),
            "subclass_slice_list": (store_list, MyInt(0)),
            "int_slice_delete": (delete_step, 0),
            "enum_slice_delete": (delete_step, Step.ZERO),
            "subclass_slice_delete": (delete_step, MyInt(0)),
        }
        fn, arg = cases[case]
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(
            compiled(arg, torch.ones(1)),
            ("slice step cannot be zero", torch.ones(1) + 1),
        )

    @parametrize(
        "operation,step,expected",
        [
            ("store", IndexableInt(2), (None, bytearray(b"xby"))),
            (
                "store",
                IndexableInt(0),
                ("slice step cannot be zero", bytearray(b"abc")),
            ),
            ("delete", IndexableInt(2), (None, bytearray(b"b"))),
            (
                "delete",
                IndexableInt(0),
                ("slice step cannot be zero", bytearray(b"abc")),
            ),
        ],
    )
    def test_int_subclass_slice_step_ignores_python_index(
        self, operation, step, expected
    ):
        def f(step, x):
            ba = bytearray(b"abc")
            try:
                if operation == "store":
                    ba[::step] = b"xy"
                else:
                    del ba[::step]
            except ValueError as e:
                return str(e), ba, x + 1
            return None, ba, x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        error, result, output = compiled(step, torch.ones(1))
        self.assertEqual((error, result), expected)
        self.assertEqual(output, torch.ones(1) + 1)

    @parametrize(
        "binding,container_kind",
        [
            ("closure", "direct"),
            ("closure", "list"),
            ("closure", "dict"),
            ("closure", "nested_list"),
            ("closure", "nested_dict"),
            ("positional_default", "direct"),
            ("positional_default", "list"),
            ("positional_default", "dict"),
            ("positional_default", "nested_list"),
            ("positional_default", "nested_dict"),
            ("keyword_default", "direct"),
            ("keyword_default", "list"),
            ("keyword_default", "dict"),
            ("keyword_default", "nested_list"),
            ("keyword_default", "nested_dict"),
        ],
    )
    def test_sourceless_bytearray_store_falls_back(self, binding, container_kind):
        Store, value = make_sourceless_store_class(binding, container_kind)

        def f(x):
            Store().store()
            return x + 1

        compiled = torch.compile(f, backend="eager")
        self.assertEqual(compiled(torch.ones(1)), torch.ones(1) + 1)
        self.assertEqual(value, bytearray(b"Zbc"))

        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "bytearray store to an existing bytearray",
        ):
            torch.compile(f, backend="eager", fullgraph=True)(torch.ones(1))

    @parametrize("binding", ["closure", "positional_default", "keyword_default"])
    @parametrize(
        "operation,expected",
        [
            ("setslice", bytearray(b"aXYc")),
            ("delitem", bytearray(b"bc")),
            ("delslice", bytearray(b"b")),
        ],
    )
    def test_sourceless_bytearray_slice_store_and_delete(
        self, binding, operation, expected
    ):
        Store, value = make_sourceless_store_class(binding, "direct", operation)

        def f(x):
            Store().store()
            return x + 1

        compiled = torch.compile(f, backend="eager")
        self.assertEqual(compiled(torch.ones(1)), torch.ones(1) + 1)
        self.assertEqual(value, expected)

    @parametrize(
        "binding",
        [
            "class_attribute",
            "module_attribute",
            "function_defaults",
        ],
    )
    @parametrize(
        "operation,expected",
        [
            ("setitem", bytearray(b"Zbc")),
            ("setslice", bytearray(b"aXYc")),
            ("delitem", bytearray(b"bc")),
            ("delslice", bytearray(b"b")),
        ],
    )
    def test_sourceless_bytearray_attribute_store(self, binding, operation, expected):
        Store, value = make_sourceless_store_class(binding, "direct", operation)

        def f(x):
            Store().store()
            return x + 1

        compiled = torch.compile(f, backend="eager")
        self.assertEqual(compiled(torch.ones(1)), torch.ones(1) + 1)
        self.assertEqual(value, expected)
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "bytearray store to an existing bytearray",
        ):
            torch.compile(f, backend="eager", fullgraph=True)(torch.ones(1))

    @parametrize(
        "binding",
        [
            "closure",
            "positional_default",
            "keyword_default",
            "class_attribute",
            "module_attribute",
            "function_defaults",
        ],
    )
    def test_sourceless_bytearray_read_still_compiles(self, binding):
        Store, _ = make_sourceless_store_class(binding, "direct", "read")

        def f(x):
            return Store().store(), x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.ones(1)), (ord("a"), torch.ones(1) + 1))

    def test_sourceless_bytearray_store_preserves_buffer_error(self):
        Store, value = make_sourceless_resize_store_class()
        view = memoryview(value)

        def f(x):
            return Store().store(), x + 1

        compiled = torch.compile(f, backend="eager")
        self.assertEqual(compiled(torch.ones(1)), f(torch.ones(1)))
        self.assertEqual(value, bytearray(b"abc"))
        view.release()

    @parametrize(
        "binding",
        [
            "closure",
            "positional_default",
            "keyword_default",
            "class_attribute",
            "module_attribute",
            "function_defaults",
        ],
    )
    def test_sourceless_bytearray_store_with_export(self, binding):
        Store, value = make_sourceless_store_class(binding, "direct", "setslice")

        def f(x):
            Store().store()
            return x + 1

        compiled = torch.compile(f, backend="eager")
        with memoryview(value):
            with self.assertRaisesRegex(BufferError, "Existing exports of data"):
                compiled(torch.ones(1))
        self.assertEqual(value, bytearray(b"abc"))

    @parametrize(
        "operation",
        [
            "empty",
            "literal",
            "encoding",
            "iterable",
            "copy",
            "slice",
            "repeat",
            "concat",
            "format",
            "split",
            "rsplit",
            "upper",
            "lower",
            "title",
        ],
    )
    def test_store_to_bytearray_operation_result(self, operation):
        def f(source, x):
            if operation == "empty":
                value = bytearray()
            elif operation == "literal":
                value = bytearray(b"abc")
            elif operation == "encoding":
                value = bytearray("abc", "ascii")
            elif operation == "iterable":
                value = bytearray(range(3))
            elif operation == "copy":
                value = bytearray(source)
            elif operation == "slice":
                value = source[:]
            elif operation == "repeat":
                value = source * 2
            elif operation == "concat":
                value = source + b"d"
            elif operation == "format":
                value = bytearray(b"%s") % b"abc"
            elif operation == "split":
                value = source.split(b"b")[0]
            elif operation == "rsplit":
                value = source.rsplit(b"b")[0]
            elif operation == "upper":
                value = source.upper()
            elif operation == "lower":
                value = source.lower()
            else:
                value = source.title()
            value[:1] = b"Z"
            del value[1:2]
            return value, x + 1

        source = bytearray(b"abc")
        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(compiled(source, torch.ones(1)), f(source, torch.ones(1)))
        self.assertEqual(source, bytearray(b"abc"))

    def test_new_bytearray_in_function_default_still_compiles(self):
        def f(x):
            value = bytearray(b"abc")

            def store(target=value):
                target[0] = ord("Z")

            store()
            return value, x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(
            compiled(torch.ones(1)), (bytearray(b"Zbc"), torch.ones(1) + 1)
        )

    def test_new_bytearray_captured_by_traced_function_still_compiles(self):
        def f(x):
            value = bytearray(b"abc")

            def store():
                value[0] = ord("Z")

            store()
            return value, x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(
            compiled(torch.ones(1)), (bytearray(b"Zbc"), torch.ones(1) + 1)
        )

    def test_new_bytearray_in_tracked_container_cell_still_compiles(self):
        def f(x):
            value = bytearray(b"abc")
            payload = [[value]]

            def store():
                payload[0][0][0] = ord("Z")

            store()
            return value, x + 1

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(
            compiled(torch.ones(1)), (bytearray(b"Zbc"), torch.ones(1) + 1)
        )

    @parametrize(
        "case",
        [
            "argument_item",
            "argument_slice_resize",
            "argument_slice_overwrite",
            "argument_delete_item",
            "argument_delete_slice",
            "global",
            "closure",
            "attribute",
        ],
    )
    def test_store_to_existing_bytearray(self, case):
        if case == "global":
            fn, args = setitem_global, ()
            target = global_bytearray
        elif case == "closure":
            ba = bytearray(b"ab")

            def fn():
                ba[0] = 1

            args = ()
            target = ba
        elif case == "attribute":
            holder = types.SimpleNamespace(ba=bytearray(b"ab"))

            def fn(holder):
                holder.ba[0] = 1

            args = (holder,)
            target = holder.ba
        else:
            arguments = {
                "argument_item": (setitem, (0, ord("x"))),
                "argument_slice_resize": (setitem, (slice(1, 2), b"XY")),
                "argument_slice_overwrite": (setitem, (slice(1, 3), b"YZ")),
                "argument_delete_item": (delitem, (0,)),
                "argument_delete_slice": (delitem, (slice(None, None, 2),)),
            }
            fn, args = arguments[case]
            args = (bytearray(b"abcd"), *args)
            target = args[0]

        initial = bytearray(target)
        try:
            fn(*args)
            expected = bytearray(target)
            target[:] = initial
            torch.compile(fn, backend="eager")(*args)
            self.assertEqual(target, expected)
        finally:
            target[:] = initial
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "bytearray store to an existing bytearray",
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(*args)

    @parametrize("escape", ["global", "list", "attribute", "alias"])
    def test_new_bytearray_escape_then_store(self, escape):
        global global_bytearray

        def f(holder, x):
            global global_bytearray
            ba = bytearray(b"abc")
            alias = ba
            if escape == "global":
                global_bytearray = ba
            elif escape == "list":
                holder.append(ba)
            elif escape == "attribute":
                holder.ba = ba
            ba[0] = 1
            return alias, ba, x + 1

        original = global_bytearray
        holder = [] if escape == "list" else types.SimpleNamespace()
        counter = CompileCounter()
        try:
            alias, result, tensor = torch.compile(f, backend=counter, fullgraph=True)(
                holder, torch.ones(1)
            )
            self.assertEqual(result, bytearray(b"\x01bc"))
            self.assertEqual(tensor, torch.ones(1) + 1)
            self.assertIs(alias, result)
            if escape == "global":
                self.assertIs(global_bytearray, result)
            elif escape == "list":
                self.assertIs(holder[0], result)
            elif escape == "attribute":
                self.assertIs(holder.ba, result)
            self.assertEqual(counter.frame_count, 1)
        finally:
            global_bytearray = original

    def test_store_to_existing_bytearray_with_export(self):
        # The tensor and the memoryview share the buffer: they must see the
        # store, and a resize must raise BufferError inside the try.
        def f(ba, t, view, x):
            ba[0] = 9
            try:
                ba[1:2] = b"XY"
            except BufferError as e:
                return t.sum(), view[0], str(e), x + 1
            return t.sum(), view[0], None, x + 1

        def run(fn):
            ba = bytearray(b"\x01\x02")
            t = torch.frombuffer(ba, dtype=torch.uint8)
            view = memoryview(ba)
            result = fn(ba, t, view, torch.ones(1))
            view.release()
            return result, ba

        self.assertEqual(run(torch.compile(f, backend="eager")), run(f))
        reasons = [
            *torch._dynamo.utils.counters["graph_break"],
            *torch._dynamo.utils.counters["unimplemented"],
        ]
        self.assertTrue(
            any("bytearray store to an existing bytearray" in r for r in reasons),
            reasons,
        )

    @parametrize("case", ["store_bytes", "store_bytearray", "store_list", "delete"])
    def test_setslice_python_index_bounds(self, case):
        class Bound:
            def __init__(self, name, value, log):
                self.name = name
                self.value = value
                self.log = log

            def __index__(self):
                self.log.append(self.name)
                return self.value

        def store(start, stop, step, value, x):
            ba = bytearray(b"abcd")
            ba[start:stop:step] = value
            return ba, x + 1

        def delete(start, stop, step, x):
            ba = bytearray(b"abcd")
            del ba[start:stop:step]
            return ba, x + 1

        def run(fn, *args):
            log = []
            bounds = [
                Bound(n, i, log) for n, i in (("start", 0), ("stop", 2), ("step", 1))
            ]
            return fn(*bounds, *args, torch.ones(1)), log

        # CPython unpacks the slice step first, and again when it converts a
        # value that is not a bytearray.
        fn, args = {
            "store_bytes": (store, (b"XY",)),
            "store_bytearray": (store, (bytearray(b"XY"),)),
            "store_list": (store, ([88, 89],)),
            "delete": (delete, ()),
        }[case]
        torch._dynamo.reset()
        self.assertEqual(run(torch.compile(fn, backend="eager"), *args), run(fn, *args))
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "bytearray slice store with Python __index__ bounds",
        ):
            run(torch.compile(fn, backend="eager", fullgraph=True), *args)

    @parametrize("position", ["start", "stop", "step"])
    def test_setslice_builtin_index_bounds(self, position):
        def store(bound, x):
            ba = bytearray(b"abc")
            if position == "start":
                ba[bound:2] = b"XY"
            elif position == "stop":
                ba[:bound] = b"XY"
            else:
                ba[::bound] = b"XYZ"
            return ba, x + 1

        def run(fn):
            index = {"start": 0, "stop": 2, "step": 1}[position]
            calls = [index, index]

            class Bound:
                __index__ = calls.pop

            return fn(Bound(), torch.ones(1)), calls

        self.assertEqual(run(torch.compile(store, backend="eager")), run(store))
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "bytearray slice store with Python __index__ bounds",
        ):
            run(torch.compile(store, backend="eager", fullgraph=True))

    @parametrize("index_type", ["float", "str"])
    def test_setslice_python_index_subclass_bounds(self, index_type):
        log = []

        class FloatIndex(float):
            def __index__(self):
                log.append(float(self))
                return int(self)

        class StrIndex(str):
            __slots__ = ()

            def __index__(self):
                log.append(self)
                return len(self)

        def store(start, stop, x):
            ba = bytearray(b"abcd")
            ba[start:stop] = [88, 89]
            return ba, x + 1

        def run(fn, start, stop):
            log.clear()
            return fn(start, stop, torch.ones(1)), list(log)

        start, stop = (
            (FloatIndex(0.0), FloatIndex(2.0))
            if index_type == "float"
            else (StrIndex(""), StrIndex("ab"))
        )
        torch._dynamo.reset()
        self.assertEqual(
            run(torch.compile(store, backend="eager"), start, stop),
            run(store, start, stop),
        )
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "bytearray slice store with Python __index__ bounds",
        ):
            run(torch.compile(store, backend="eager", fullgraph=True), start, stop)

    def test_store_symbolic_ints(self):
        def f(x):
            n = x.size(0)
            ba = bytearray(b"abcdefgh")
            ba[n] = n + 60
            ba[:n] = [n + 70, n + 71]
            del ba[::n]
            return ba

        compiled = torch.compile(f, backend="eager", fullgraph=True, dynamic=True)
        # Reuse the compiled callable to check successive input sizes.
        for size in (2, 3):
            with self.subTest(size=size):
                x = torch.ones(size)
                self.assertEqual(compiled(x), f(x))

    def test_setslice_input_values(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(values, x):
            ba = bytearray(b"xyz")
            ba[0:2] = values
            return ba, x + 1

        # Reuse the compiled callable across changes in value contents and types.
        for values, expected in (
            ([65, 66], b"ABz"),
            ([69, 70], b"EFz"),
            ((67, 68), b"CDz"),
            ((71, 72), b"GHz"),
            (b"IJ", b"IJz"),
            (bytearray(b"KL"), b"KLz"),
        ):
            with self.subTest(values=values):
                self.assertEqual(f(values, torch.ones(1))[0], bytearray(expected))

    @parametrize("case", ["range", "type", "mixed"])
    def test_setslice_value_conversion_order(self, case):
        class Logged:
            def __init__(self, value, log):
                self.value = value
                self.log = log

            def __index__(self):
                self.log.append(self.value)
                return self.value

        def f(make_value, x):
            ba = bytearray(b"xyz")
            log = []
            try:
                ba[0:2] = make_value(log)
            except (TypeError, ValueError) as e:
                return type(e), log, ba, x + 1
            return None, log, ba, x + 1

        # Items are converted in order and the first bad one stops it.
        make_values = {
            "range": lambda log: [256, Logged(66, log)],
            "type": lambda log: ("a", Logged(66, log)),
            "mixed": lambda log: [Logged(65, log), 66, Logged(67, log)],
        }[case]
        torch._dynamo.reset()
        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(
            compiled(make_values, torch.ones(1)), f(make_values, torch.ones(1))
        )

    def test_setitem_setslice_nonconstant_graph_break(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            ba = bytearray(b"ab")
            x = x + 1
            ba[0] = 0x7A
            x = x * 2
            ba[:1] = (i for i in range(2))
            x = x - 1
            ba[-1] = 3
            return ba, x * 3

        self.assertEqual(
            f(torch.ones(1)), (bytearray(b"\x00\x01\x03"), torch.full((1,), 9.0))
        )
        # The first store is traced. The generator value breaks the graph, and
        # in the resume function ba is existing, so the next store breaks too.
        self.assertEqual(cnt.frame_count, 3)
        reasons = list(torch._dynamo.utils.counters["graph_break"])
        for reason in (
            "bytearray store with an unsupported value",
            "bytearray store to an existing bytearray",
        ):
            self.assertTrue(any(reason in r for r in reasons), reasons)

    @parametrize("operation", ["item", "slice"])
    def test_store_nonconstant_value_fullgraph(self, operation):
        def setitem_tensor(x):
            ba = bytearray(b"ab")
            ba[0] = x[0].to(torch.int64)
            return ba

        def setslice_generator(x):
            ba = bytearray(b"ab")
            ba[:1] = (i for i in range(2))
            return ba

        fn = setitem_tensor if operation == "item" else setslice_generator
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "bytearray store with an unsupported value",
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(1))

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
