# Owner(s): ["module: dynamo"]

import cmath
import decimal
import fractions
import gc
import math
import operator
from unittest import mock

import torch
from torch._dynamo.exc import Unsupported
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


NAN_IDENTITY_OPS = {
    "dict_get": lambda a, b: {a: 1}.get(b, "missing"),
    "list_contains": lambda a, b: b in [a],
}


class DecimalTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_lazy_wrapper_identity(self):
        from torch._dynamo.source import LocalSource
        from torch._dynamo.variables.decimal import DecimalVariable
        from torch._dynamo.variables.lazy import LazyCache, LazyVariableTracker
        from torch._dynamo.variables.object_protocol import vt_identity_compare

        value = decimal.Decimal("NaN")
        source = LocalSource("k")
        inner = DecimalVariable(value, source=source)
        cache = LazyCache(value, source)
        cache.vt = inner
        wrapper = LazyVariableTracker(cache, source=source)
        # The guarded NaN value path also answers this, so block it.
        with mock.patch.object(
            DecimalVariable, "get_real_python_backed_value", side_effect=AssertionError
        ):
            self.assertTrue(vt_identity_compare(wrapper, inner).as_python_constant())

    @parametrize("name", list(NAN_IDENTITY_OPS))
    def test_nan_input_identity(self, name):
        op = NAN_IDENTITY_OPS[name]

        def fn(x, a, b):
            return x + 1, op(a, b)

        n1, n2 = decimal.Decimal("NaN"), decimal.Decimal("NaN")
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        # Each relation compiles once; repeating the sequence hits the cache.
        for args in [(n1, n1), (n1, n2)] * 2:
            self.assertEqual(compiled(x, *args), fn(x, *args))
        self.assertEqual(counter.frame_count, 2)

    def test_equal_valued_inputs_reuse_graph(self):
        def fn(x, a, b):
            return x + float(a), str(a), b in [a], {a: 1}.get(b), a

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        d1, d2 = decimal.Decimal("7.25"), decimal.Decimal("7.25")
        x = torch.ones(2)
        for a, b in ((d1, d1), (d1, d2), (d2, d1)):
            actual = compiled(x, a, b)
            self.assertEqual(actual[:4], fn(x, a, b)[:4])
            self.assertIs(actual[4], a)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(recompile_limit=64)
    def test_input_representation_guard(self):
        def fn(x, d):
            return x + 1, str(d), d.as_tuple(), d

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        values = [
            "0",
            "-0",
            "0E-100",
            "1.0",
            "1.00",
            "Infinity",
            "-Infinity",
            "NaN123",
            "sNaN123",
        ]
        for value in values:
            d = decimal.Decimal(value)
            actual = compiled(x, d)
            self.assertEqual(actual[:3], fn(x, d)[:3])
            self.assertIs(actual[3], d)
        self.assertEqual(counter.frame_count, len(values))

    @torch._dynamo.config.patch(recompile_limit=64)
    def test_fresh_inputs_comparison_and_hash(self):
        def fn(x, a, b):
            return x + 1, a == b, a < b, hash(a)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        for i in range(12):
            a = decimal.Decimal(f"{i}.5")
            b = decimal.Decimal(f"{(i * 7) % 5}.5")
            self.assertEqual(compiled(x, a, b), fn(x, a, b))
            del a, b
            gc.collect()
        self.assertEqual(counter.frame_count, 12)

    @parametrize("op", [operator.eq, operator.lt])
    @parametrize("other", [decimal.Decimal("1.5"), 1])
    def test_comparisons(self, op, other):
        def fn(x, d):
            try:
                return x + 1, op(d, other), op(other, d)
            except TypeError:
                return x + 2, None

        x = torch.ones(2)
        d = decimal.Decimal("1.5")
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, d), fn(x, d)
        )

    def test_input_alias_guards(self):
        def fn(x, a, b):
            return x + 1, a is b, a is not b, a, b

        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        x = torch.ones(2)
        a, b = (decimal.Decimal("1.5") for _ in range(2))
        for left, right in ((a, a), (a, b), (b, a), (b, b)):
            actual = compiled(x, left, right)
            self.assertEqual(actual[:3], fn(x, left, right)[:3])
            self.assertIs(actual[3], left)
            self.assertIs(actual[4], right)

    def test_comparison_unmodeled_operands(self):
        class Integer(int):
            pass

        class Reflected:
            def __eq__(self, other):
                return "reflected eq"

            def __gt__(self, other):
                return "reflected gt"

        class Rational(fractions.Fraction):
            def __eq__(self, other):
                return False

            def __gt__(self, other):
                return False

        def fn(x, d, other):
            return x + 1, d == other, d != other, d < other

        x = torch.ones(2)
        d = decimal.Decimal("0.5")
        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        others = (
            Integer(0),
            Integer(1),
            Reflected(),
            fractions.Fraction(1, 4),
            fractions.Fraction(1, 2),
            Rational(1, 2),
            Rational(3, 4),
        )
        for other in others:
            self.assertEqual(compiled(x, d, other), fn(x, d, other))

    def test_sourceless_quiet_nan_hash_does_not_escape(self):
        def fn(x):
            d = decimal.Decimal("NaN123")
            return x + 1, d, hash(d)

        x = torch.ones(2)
        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        for _ in range(3):
            actual = compiled(x)
            self.assertEqual(actual[0], x + 1)
            self.assertEqual(actual[1].as_tuple(), decimal.Decimal("NaN123").as_tuple())
            self.assertEqual(actual[2], hash(actual[1]))

    @parametrize("op", [operator.eq, operator.lt])
    @parametrize(
        "case",
        [
            ("1.5", 1.5, decimal.FloatOperation),
            ("sNaN123", decimal.Decimal(1), decimal.InvalidOperation),
        ],
    )
    @parametrize("trapped", [False, True])
    def test_localcontext_comparison_flags_and_traps(self, op, case, trapped):
        value, other, signal = case

        def fn(x, d):
            with decimal.localcontext(
                decimal.Context(traps=[signal] if trapped else [])
            ) as context:
                try:
                    result = op(d, other)
                except (decimal.DecimalException, TypeError):
                    result = "caught"
            return x + 1, result, context

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        d = decimal.Decimal(value)
        with decimal.localcontext() as ambient:
            before = str(ambient)
            expected = fn(x, d)
            for _ in range(2):
                actual = compiled(x, d)
                self.assertEqual(actual[:2], expected[:2])
                self.assertEqual(str(actual[2]), str(expected[2]))
                self.assertEqual(str(ambient), before)
        self.assertEqual(counter.frame_count, 1)

    @parametrize(
        "case",
        [
            ("1.5", lambda d: d == 1.5),
            ("1.5", lambda d: d < 1.5),
            ("NaN123", lambda d: d < 1),
            ("sNaN123", lambda d: d == 1),
            ("1", lambda d: d / 3),
            ("2", lambda d: d.sqrt()),
            ("1", lambda d: decimal.Decimal(1.1)),
            ("1", lambda d: decimal.Decimal("invalid")),
            ("1", lambda d: decimal.getcontext()),
            ("1", lambda d: decimal.localcontext()),
            ("1", lambda d: decimal.Context().flags),
        ],
    )
    def test_unsupported_ambient_operations_do_not_leak(self, case):
        value, operation = case

        def fn(x, d):
            return x + 1, operation(d)

        with decimal.localcontext() as ambient:
            ambient.prec = 2
            ambient.traps[decimal.FloatOperation] = False
            ambient.traps[decimal.InvalidOperation] = False
            ambient.clear_flags()
            before = str(ambient)
            with self.assertRaises(Unsupported):
                torch.compile(fn, backend="eager", fullgraph=True)(
                    torch.ones(2), decimal.Decimal(value)
                )
            self.assertEqual(str(ambient), before)

    @parametrize("case", [("1.5", lambda d: d < 1.5), ("1", lambda d: d / 3)])
    def test_ambient_fallback_preserves_flags_and_traps(self, case):
        value, operation = case

        def fn(x, d):
            try:
                result = operation(d)
            except decimal.DecimalException:
                result = "caught"
            return x + 1, result

        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        x = torch.ones(2)
        d = decimal.Decimal(value)
        with decimal.localcontext() as ambient:
            for precision, trapped in ((2, False), (6, True), (2, False)):
                ambient.prec = precision
                ambient.traps[decimal.FloatOperation] = trapped
                ambient.traps[decimal.InvalidOperation] = trapped
                ambient.clear_flags()
                expected = fn(x, d)
                expected_flags = dict(ambient.flags)
                ambient.clear_flags()
                actual = compiled(x, d)
                self.assertEqual(actual[0], expected[0])
                self.assertEqual(repr(actual[1]), repr(expected[1]))
                self.assertEqual(dict(ambient.flags), expected_flags)

    @parametrize("value", ["-0", "NaN123", (1, (1, 2, 3), -2)])
    def test_construction_and_reconstruction(self, value):
        def fn(x):
            d = decimal.Decimal(value)
            return x + 1, d, d, d.as_tuple()

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for _ in range(2):
            result = compiled(torch.ones(2))
            self.assertEqual(result[0], torch.full((2,), 2.0))
            self.assertIs(type(result[1]), decimal.Decimal)
            self.assertIs(result[1], result[2])
            self.assertEqual(result[1].as_tuple(), decimal.Decimal(value).as_tuple())
            self.assertEqual(result[3], decimal.Decimal(value).as_tuple())
        self.assertEqual(counter.frame_count, 1)

    def test_constructor_signature(self):
        def fn(x):
            a = decimal.Decimal()
            b = decimal.Decimal(value="1.25", context=None)
            c = decimal.Decimal("2.5", None)
            return x + float(a) + float(b) + float(c)

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_constructor_identity_with_context_and_keywords(self):
        def fn(x, d):
            context = decimal.Context()
            a = decimal.Decimal(value=d)
            b = decimal.Decimal(d, context)
            with decimal.localcontext(context):
                c = decimal.Decimal(value=d, context=None)
            return x + 1, a is d, b is d, c is d, a, b, c

        x = torch.ones(2)
        d = decimal.Decimal("1.23")
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x, d)
        self.assertEqual(actual[:4], (x + 1, True, True, True))
        for value in actual[4:]:
            self.assertIs(value, d)

    @parametrize(
        "call",
        [
            lambda: decimal.Decimal("1", value=2),
            lambda: decimal.Context(prec=0),
            lambda: decimal.localcontext(decimal.Context(), prec=0),
        ],
    )
    def test_invalid_arguments(self, call):
        def fn(x):
            try:
                call()
            except (TypeError, ValueError):
                return x + 1
            return x - 1

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_nonliteral_argument_does_not_execute_user_code(self):
        calls = []

        class Operand:
            def __float__(self):
                calls.append(1)
                return 1.0

        def fn(x, operand):
            with decimal.localcontext(decimal.Context()):
                result = decimal.Decimal(operand)
            return x + 1, result

        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2), Operand())
        self.assertEqual(calls, [])

    def test_explicit_context_constructor(self):
        def fn(x):
            context = decimal.Context(traps=[])
            a = decimal.Decimal(1.1, context)
            b = decimal.Decimal("invalid", context=context)
            return x + float(a), a, b, context

        x = torch.ones(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        expected = fn(x)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1].as_tuple(), expected[1].as_tuple())
        self.assertEqual(actual[2].as_tuple(), expected[2].as_tuple())
        self.assertEqual(str(actual[3]), str(expected[3]))

    @parametrize("value", ["sNaN", "-0"])
    def test_context_independent_methods(self, value):
        def fn(x):
            d = decimal.Decimal(value)
            return x + 1, (
                d.is_finite(),
                d.is_infinite(),
                d.is_nan(),
                d.is_snan(),
                d.is_qnan(),
                d.is_signed(),
                d.is_zero(),
                d.copy_abs().as_tuple(),
                d.copy_negate().as_tuple(),
                d.copy_sign(decimal.Decimal(-1)).as_tuple(),
            )

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("name", ["sqrt", "quantize"])
    def test_contextual_methods(self, name):
        def fn(x):
            with decimal.localcontext(decimal.Context(prec=6)) as context:
                d = decimal.Decimal("1.23456789")
                if name == "quantize":
                    result = d.quantize(decimal.Decimal("0.01"))
                else:
                    result = getattr(d, name)()
            return x + float(result), result, context

        x = torch.ones(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[:2], expected[:2])
        self.assertEqual(str(actual[2]), str(expected[2]))

    @parametrize("op", [float, int, bool])
    @parametrize("value", ["-1.25", "NaN"])
    def test_numeric_conversion(self, op, value):
        def fn(x):
            try:
                result = op(decimal.Decimal(value))
            except (ValueError, OverflowError):
                return x + 2, None
            return x + 1, result

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("op", [str, repr])
    def test_capitals_guard(self, op):
        def fn(x):
            return x + 1, op(decimal.Decimal("1e-20"))

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        with decimal.localcontext() as context:
            for capitals in (0, 1, 0):
                context.capitals = capitals
                self.assertEqual(compiled(x), fn(x))
        self.assertEqual(counter.frame_count, 2)

    def test_cmath_isclose_complex_operand(self):
        def fn(x):
            d = decimal.Decimal(1)
            return x + 1, cmath.isclose(d, 1 + 0j), cmath.isclose(1 + 1j, d)

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_isclose_conversion_order(self):
        class Number:
            def __init__(self, name, calls):
                self.name = name
                self.calls = calls

            def __float__(self):
                self.calls.append(self.name)
                return 1.0

        def fn(x, calls):
            result = math.isclose(
                abs_tol=Number("abs_tol", calls),
                b=Number("b", calls),
                rel_tol=Number("rel_tol", calls),
                a=decimal.Decimal(1),
            )
            return x + 1, result

        x = torch.ones(2)
        calls = []
        expected = fn(x, calls)
        self.assertEqual(calls, ["b", "rel_tol", "abs_tol"])
        calls.clear()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        for _ in range(2):
            self.assertEqual(compiled(x, calls), expected)
            self.assertEqual(calls, ["b", "rel_tol", "abs_tol"])
            calls.clear()

    def test_localcontext_arithmetic_and_cache(self):
        def fn(x, b):
            base = decimal.Context(prec=5, rounding=decimal.ROUND_UP)
            with decimal.localcontext(base) as context:
                a = decimal.Decimal(1.1)
                result = ((a**2 + b**2).sqrt(), 2 + a, 2 - a, 2 * a, 2 / b, 2**b)
                text = str(result[0])
                same = decimal.getcontext() is context
            return x + float(result[0]), result, text, same, context, base

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        b = decimal.Decimal(3)
        with decimal.localcontext() as ambient:
            ambient.prec = 2
            ambient.traps[decimal.FloatOperation] = True
            before = ambient.copy()
            expected = fn(x, b)
            previous = None
            for _ in range(3):
                actual = compiled(x, b)
                self.assertEqual(actual[:4], expected[:4])
                for actual_ctx, expected_ctx in zip(actual[4:], expected[4:]):
                    self.assertEqual(str(actual_ctx), str(expected_ctx))
                self.assertIsNot(actual[4], actual[5])
                self.assertIsNot(actual[4], previous)
                previous = actual[4]
                actual[4].clear_flags()
                self.assertEqual(str(ambient), str(before))
        self.assertEqual(counter.frame_count, 1)

    def test_default_context_guard(self):
        def fn(x):
            with decimal.localcontext(decimal.Context()):
                result = decimal.Decimal(1) / decimal.Decimal(3)
            return x + float(result), result

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        previous = decimal.DefaultContext.prec
        try:
            for precision in (2, 5, 2):
                decimal.DefaultContext.prec = precision
                self.assertEqual(compiled(x), fn(x))
        finally:
            decimal.DefaultContext.prec = previous
        self.assertEqual(counter.frame_count, 2)

    def test_context_state_reconstruction(self):
        def fn(x):
            base = decimal.Context(
                prec=3,
                rounding=decimal.ROUND_UP,
                Emin=-4,
                Emax=4,
                capitals=0,
                clamp=1,
                flags=[decimal.Inexact],
                traps=[],
            )
            with decimal.localcontext(base) as context:
                result = decimal.Decimal("9.99e4") * 10
            return x + 1, base, context, result

        x = torch.ones(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(str(actual[1]), str(expected[1]))
        self.assertEqual(str(actual[2]), str(expected[2]))
        self.assertEqual(actual[3].as_tuple(), expected[3].as_tuple())

    def test_cond_branch_local_context(self):
        def positive(x):
            with decimal.localcontext(decimal.Context(prec=3)):
                amount = float(decimal.Decimal(1) / 3)
            return x + amount

        def negative(x):
            return x - 1

        def fn(x, pred):
            return torch.cond(pred, positive, negative, (x,))

        x = torch.ones(2)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        for pred in (torch.tensor(True), torch.tensor(False)):
            self.assertEqual(
                compiled(x, pred), positive(x) if pred.item() else negative(x)
            )

    def test_context_methods_and_copy(self):
        def fn(x):
            context = decimal.Context(prec=4)
            alias = context
            other = context.copy()
            a = context.create_decimal("1.2345")
            b = other.create_decimal_from_float(1.2345)
            context.clear_flags()
            other.clear_traps()
            return x + float(a) + float(b), context, alias, other

        x = torch.ones(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        expected = fn(x)
        self.assertEqual(actual[0], expected[0])
        self.assertIs(actual[1], actual[2])
        for actual_ctx, expected_ctx in zip(actual[1:], expected[1:]):
            self.assertEqual(str(actual_ctx), str(expected_ctx))

    def test_nested_contexts(self):
        def fn(x):
            with decimal.localcontext(decimal.Context(prec=3)) as outer:
                a = decimal.Decimal(1) / 3
                with decimal.localcontext(
                    decimal.Context(prec=7), rounding=decimal.ROUND_UP
                ) as inner:
                    b = decimal.Decimal(1) / 3
                    innermost = decimal.getcontext() is inner
                c = decimal.Decimal(1) / 3
                restored = decimal.getcontext() is outer
            return x + float(b), a, b, c, outer, inner, innermost, restored

        x = torch.ones(2)
        with decimal.localcontext() as ambient:
            before = str(ambient)
            actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
            expected = fn(x)
            self.assertEqual(actual[:4], expected[:4])
            self.assertEqual(
                tuple(map(str, actual[4:6])), tuple(map(str, expected[4:6]))
            )
            self.assertEqual(actual[6:], (True, True))
            self.assertEqual(str(ambient), before)

    def test_nested_trap_restores_outer_context(self):
        def fn(x):
            with decimal.localcontext(decimal.Context(prec=5)) as outer:
                try:
                    with decimal.localcontext(
                        decimal.Context(prec=2, traps=[decimal.Inexact])
                    ):
                        decimal.Decimal(1) / 3
                except decimal.Inexact:
                    result = decimal.Decimal(1) / 3
                restored = decimal.getcontext() is outer
            return x + float(result), result, restored

        x = torch.ones(2)
        with decimal.localcontext() as ambient:
            before = str(ambient)
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )
            self.assertEqual(str(ambient), before)

    @parametrize("graph_break", [False, True])
    def test_unsupported_inside_localcontext_cleans_up(self, graph_break):
        def fn(x):
            with decimal.localcontext(decimal.Context(prec=2)):
                decimal.Decimal(1) / 3
                if graph_break:
                    torch._dynamo.graph_break()
                y = x + 1
            return y

        with decimal.localcontext() as ambient:
            before = str(ambient)
            with self.assertRaises(Unsupported):
                torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2))
            self.assertEqual(str(ambient), before)
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=False)(torch.ones(2)),
                torch.full((2,), 2.0),
            )
            self.assertEqual(str(ambient), before)

    @parametrize("nested", [False, True])
    def test_localcontext_manager_reuse(self, nested):
        def fn(x):
            manager = decimal.localcontext(decimal.Context(prec=2))
            with manager:
                a = decimal.Decimal(1) / 3
                if nested:
                    with manager:
                        a = decimal.Decimal(1) / 3
            with manager:
                b = decimal.Decimal(1) / 3
            return x + float(a), b

        x = torch.ones(2)
        with decimal.localcontext() as ambient:
            before = str(ambient)
            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            if nested:
                with self.assertRaises(Unsupported):
                    compiled(x)
            else:
                self.assertEqual(compiled(x), fn(x))
            self.assertEqual(str(ambient), before)


instantiate_parametrized_tests(DecimalTests)


if __name__ == "__main__":
    run_tests()
