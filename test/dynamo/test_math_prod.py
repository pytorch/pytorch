# Owner(s): ["module: dynamo"]

import decimal
import fractions
import math

import torch
from torch._dynamo.exc import Unsupported
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
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

    @parametrize(
        "values,start",
        [
            ([2, 3, 4], 1),
            ([2.5, 4], 2),
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
        self.assertEqual(compiled(x, values), fn(x, values))


instantiate_parametrized_tests(MathProdTests)

if __name__ == "__main__":
    run_tests()
