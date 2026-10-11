# Owner(s): ["module: dynamo"]

import decimal
import fractions
import math

import torch
from torch._dynamo.exc import Unsupported
from torch._dynamo.test_case import run_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TestCase,
)


class MathProdTests(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

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
        with self.assertRaises(Unsupported):
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
