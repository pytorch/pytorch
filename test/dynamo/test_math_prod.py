# Owner(s): ["module: dynamo"]

import decimal
import fractions
import math
from unittest import mock

import torch
from torch._dynamo import polyfills
from torch._dynamo.exc import Unsupported
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


class MathProdTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("construct", [False, True])
    @parametrize(
        "rounding", [decimal.ROUND_HALF_EVEN, decimal.ROUND_DOWN, decimal.ROUND_UP]
    )
    def test_decimal_context(self, construct, rounding):
        def fn(x, values):
            if construct:
                values = [decimal.Decimal("1.23456789")] * 2
            return x + 1, math.prod(values)

        compiled = torch.compile(fn, backend="eager")
        x = torch.ones(1)
        values = [decimal.Decimal("1.23456789")] * 2
        with decimal.localcontext(prec=28):
            self.assertEqual(compiled(x, values), fn(x, values))
        with decimal.localcontext(prec=3, rounding=rounding) as ctx:
            ctx.clear_flags()
            expected = fn(x, values)
            expected_flags = ctx.flags.copy()
            for _ in range(2):
                ctx.clear_flags()
                self.assertEqual(compiled(x, values), expected)
                self.assertEqual(ctx.flags, expected_flags)
        with decimal.localcontext(prec=3) as ctx:
            ctx.traps[decimal.Inexact] = True
            with self.assertRaises(decimal.Inexact):
                compiled(x, values)

    @parametrize("construct", [False, True])
    def test_decimal_fullgraph_unsupported(self, construct):
        def fn(x, values):
            if construct:
                values = [decimal.Decimal("1.23456789")] * 2
            return x + 1, math.prod(values)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        if construct:
            message = r"Unsupported function call[\s\S]*decimal\.Decimal"
        else:
            message = r"'Decimal' implements '__rmul__' in C"
        with self.assertRaisesRegex(Unsupported, message):
            compiled(torch.ones(1), [decimal.Decimal("1.23456789")] * 2)

    @parametrize("start_type", [list, bytearray])
    @parametrize("count", [0, 1, 2])
    @parametrize("external", [False, True])
    def test_mutable_start(self, start_type, count, external):
        class Index:
            def __init__(self):
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return 2

        def fn(x, start, factor):
            if not external:
                start = start_type([1])
            out = math.prod([factor] * count, start=start)
            return out, start, x + 1

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            start, factor = start_type([1]), Index()
            actual = compiled(x, start, factor)
            expected = fn(x, start_type([1]), Index())
            self.assertEqual(actual, expected)
            self.assertEqual(start, start_type([1]))
            self.assertEqual(factor.calls, count)
            if count:
                self.assertIsNot(actual[0], actual[1])
            else:
                self.assertIs(actual[0], actual[1])

    @parametrize("start_type", [fractions.Fraction, tuple])
    def test_empty_start_identity(self, start_type):
        def fn(x, start):
            return math.prod([], start=start), start, x + 1

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        for _ in range(2):
            if start_type is fractions.Fraction:
                start = start_type(2, 3)
            else:
                start = start_type([1])
            actual = compiled(torch.ones(1), start)
            self.assertIs(actual[0], start)
            self.assertIs(actual[1], start)

    @parametrize("count", [0, 1, 2])
    def test_start_element_alias(self, count):
        def fn(x, start):
            out = math.prod([2] * count, start=start)
            return out, start, x + 1

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            inner = [1]
            start = [inner]
            actual = compiled(x, start)
            self.assertEqual(actual, fn(x, start))
            self.assertIs(actual[1], start)
            for element in actual[0]:
                self.assertIs(element, inner)
            if not count:
                self.assertIs(actual[0], start)

    def test_element_alias_without_start(self):
        def fn(x, inner):
            return math.prod([[inner]]), inner, x + 1

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            inner = [1]
            actual = compiled(x, inner)
            self.assertEqual(actual, fn(x, inner))
            self.assertIs(actual[0][0], inner)
            self.assertIs(actual[1], inner)

    @parametrize("reflected", [False, True])
    def test_multiplication_protocol(self, reflected):
        class Product:
            def __init__(self, value, events):
                self.value = value
                self.events = events

            def __mul__(self, other):
                self.events.append("mul")
                if reflected:
                    return NotImplemented
                return self.value * other

            def __imul__(self, other):
                raise AssertionError("math.prod must not call __imul__")

        class Factor:
            def __rmul__(self, other):
                other.events.append("rmul")
                return other.value * 3

        def fn(x):
            events = []
            start = Product(2, events)
            factor = Factor() if reflected else 3
            out = math.prod([factor, 4], start=start)
            return out, start.value, events, x + 1

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            self.assertEqual(compiled(x), fn(x))

    @parametrize("operation", ["prod", "multiply", "inplace"])
    @parametrize("index_result", [2.5, None, "2"])
    def test_invalid_index_result(self, operation, index_result):
        class BadIndex:
            def __index__(self):
                return index_result

        def fn(x):
            start = [1]
            try:
                if operation == "prod":
                    math.prod([BadIndex()], start=start)
                elif operation == "multiply":
                    start * BadIndex()
                else:
                    start *= BadIndex()
            except TypeError:
                return start, True, x + 1
            return start, False, x + 1

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        self.assertEqual(compiled(x), fn(x))

    def test_multiplication_exception_preserves_start(self):
        class Factor:
            def __rmul__(self, other):
                raise ValueError("multiply failed")

        def fn(x):
            start = [1]
            try:
                math.prod([2, Factor()], start=start)
            except ValueError:
                return start, x + 1

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            self.assertEqual(compiled(x), fn(x))

    @parametrize("source", ["constructor", "from_float", "from_decimal"])
    def test_fraction_mixed_construction(self, source):
        values = {
            "constructor": fractions.Fraction(1, 2),
            "from_float": fractions.Fraction.from_float(0.5),
            "from_decimal": fractions.Fraction.from_decimal(decimal.Decimal("0.5")),
        }

        def fn(x, value):
            left = fractions.Fraction(3, 5)
            right = fractions.Fraction.from_float(0.5)
            return x + 1, left * right * value, math.prod([left, right, value])

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            self.assertEqual(compiled(x, values[source]), fn(x, values[source]))

    @parametrize("fullgraph", [False, True])
    def test_fraction_replacement(self, fullgraph):
        def fn(x, values):
            return x + 1, math.prod(values)

        # General Fraction arithmetic can graph break when its integers become symbolic.
        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=fullgraph,
            dynamic=False if fullgraph else None,
        )
        x = torch.ones(1)
        values = [fractions.Fraction(2, 3)]
        for value in (fractions.Fraction(2, 3), fractions.Fraction(4, 5)):
            values[0] = value
            for _ in range(2):
                self.assertEqual(compiled(x, values), fn(x, values))

    @parametrize("container", [list, tuple])
    @parametrize("values", [(), (9,)])
    @parametrize("external", [False, True])
    def test_sequence_subclass_iteration(self, container, values, external):
        class Sequence(container):
            def __iter__(self):
                self.calls += 1
                return iter((2, 3))

        def fn(x, sequence):
            if not external:
                sequence = Sequence(values)
                sequence.calls = 0
            return math.prod(sequence), sequence.calls, x + 1

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            sequence, expected_sequence = Sequence(values), Sequence(values)
            sequence.calls = expected_sequence.calls = 0
            self.assertEqual(compiled(x, sequence), fn(x, expected_sequence))
            self.assertEqual(sequence.calls, 1 if external else 0)

    @parametrize(
        "container,external",
        [(list, False), (tuple, False), (range, False), (tuple, True), (range, True)],
    )
    def test_iterator_consumption(self, container, external):
        def fn(x, iterator):
            if not external:
                values = range(2, 5) if container is range else container([2, 3, 4])
                iterator = iter(values)
            out = math.prod(iterator, start=2)
            return out, math.prod(iterator), next(iterator, -1), x + 1

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            values = range(2, 5) if container is range else container([2, 3, 4])
            iterator = iter(values)
            self.assertEqual(compiled(x, iterator), fn(x, iter(values)))
            if external:
                self.assertEqual(list(iterator), [])

    def test_external_list_iterator_fallback(self):
        def fn(x, iterator):
            out = math.prod(iterator, start=2)
            return out, math.prod(iterator), next(iterator, -1), x + 1

        x = torch.ones(1)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            Unsupported, "'list_iterator' implements '__iter__' in C"
        ):
            compiled(x, iter([2, 3, 4]))

        compiled = torch.compile(fn, backend="eager")
        for _ in range(2):
            iterator = iter([2, 3, 4])
            self.assertEqual(compiled(x, iterator), fn(x, iter([2, 3, 4])))
            self.assertEqual(list(iterator), [])

    def test_iterator_progress_after_error(self):
        class Factor:
            def __rmul__(self, other):
                raise ValueError("multiply failed")

        def fn(x):
            iterator = iter([2, Factor(), 3])
            try:
                math.prod(iterator)
            except ValueError:
                return next(iterator), list(iterator), x + 1

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            self.assertEqual(compiled(x), fn(x))

    @parametrize(
        "case", ["missing", "positional", "keyword", "empty_keyword", "noniterable"]
    )
    def test_invalid_arguments(self, case):
        def fn(x):
            try:
                if case == "missing":
                    math.prod()
                elif case == "positional":
                    math.prod([2], 3)
                elif case == "keyword":
                    math.prod([2], invalid=3)
                elif case == "empty_keyword":
                    math.prod([], invalid=3)
                else:
                    math.prod(42)
            except TypeError as exc:
                return x + 1, str(exc)
            return x, "no error"

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            self.assertEqual(compiled(x), fn(x))

    @parametrize("tensor_start", [False, True])
    def test_tensor_product(self, tensor_start):
        def fn(x):
            start = x if tensor_start else 2
            return math.prod([x, x + 1], start=start)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        for size in (3, 4):
            x = torch.ones(size)
            self.assertEqual(compiled(x), fn(x))

    @parametrize("container", [list, tuple, torch.Size])
    def test_symbolic_product(self, container):
        def fn(x):
            values = container([x.shape[0], x.shape[1]])
            return x + 1, math.prod(values, start=2)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True, dynamic=True)
        for shape in ((3, 4), (5, 6)):
            x = torch.ones(shape)
            torch._dynamo.mark_dynamic(x, (0, 1))
            self.assertEqual(compiled(x), fn(x))
        self.assertEqual(counter.frame_count, 1)

    @parametrize("kind", ["shape", "shape_slice", "range", "reverse_range"])
    def test_static_product_constant_fold(self, kind):
        def fn(x):
            if kind == "shape":
                values = x.shape
            elif kind == "shape_slice":
                values = x.shape[1:]
            elif kind == "range":
                values = range(2, 8)
            else:
                values = range(8, 2, -2)
            return x + 1, math.prod(values, start=2)

        def unexpected_polyfill(*args, **kwargs):
            raise AssertionError(
                "Expected math.prod to fold without tracing the polyfill"
            )

        x = torch.ones(2, 3, 4)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True, dynamic=False)
        with mock.patch.object(polyfills, "math_prod", unexpected_polyfill):
            for _ in range(2):
                self.assertEqual(compiled(x), fn(x))
        self.assertEqual(counter.frame_count, 1)

    @parametrize("start_type", [list, str])
    def test_empty_range_start_identity(self, start_type):
        def fn(x, start):
            return math.prod(range(1, 3, -1), start=start), x + 1

        def unexpected_polyfill(*args, **kwargs):
            raise AssertionError("Expected the empty range to preserve start directly")

        x = torch.ones(1)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with mock.patch.object(polyfills, "math_prod", unexpected_polyfill):
            for _ in range(2):
                start = [1] if start_type is list else "".join(("mutable", "-start"))
                self.assertIs(compiled(x, start)[0], start)

    @parametrize(
        "values,start",
        [
            ([2, 3, 4], 1),
            ([2.5, 4], 2),
            ((2, 3), 2),
            ([False, 3], 1),
            ([2**100, 3, -2], 1),
            ([1e-300, 1e-300], 1.0),
            ([0.0, math.inf], 1),
            ([math.nan], 1),
            ([-0.0, 2], 1),
            ([], -0.0),
            ([1 + 2j, 3 - 4j], 1),
            ([fractions.Fraction(3, 5)] * 2, fractions.Fraction(2, 3)),
            ([2, fractions.Fraction(3, 5)], fractions.Fraction(2, 3)),
            ([], fractions.Fraction(2, 3)),
        ],
    )
    def test_supported_fullgraph(self, values, start):
        def fn(x, values):
            return x + 1, math.prod(values, start=start)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        actual, expected = compiled(x, values), fn(x, values)
        self.assertEqual(actual, expected)
        self.assertIs(type(actual[1]), type(expected[1]))
        if isinstance(expected[1], float) and expected[1] == 0:
            self.assertEqual(math.copysign(1, actual[1]), math.copysign(1, expected[1]))


instantiate_parametrized_tests(MathProdTests)

if __name__ == "__main__":
    run_tests()
