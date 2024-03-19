# Owner(s): ["module: inductor"]
import sympy

from torch._inductor.codegen.cpp import cexpr
from torch._inductor.codegen.triton import texpr
from torch._inductor.codegen.wrapper import pexpr

from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.utils._sympy.functions import FloorDiv, ModularIndexing, Round, RoundDecimal


class TestIndexingSimplification(InductorTestCase):
    def test_indexing_simplification(self):
        sizevars = SizeVarAllocator()
        i0 = sympy.Symbol("i0", integer=True)
        i1 = sympy.Symbol("i1", integer=True)
        i2 = sympy.Symbol("i2", integer=True)
        r3 = sympy.Symbol("r3", integer=True)

        var_ranges = {i0: 3136, i1: 64, i2: 32, r3: 3}
        expr = (
            128 * i2
            + ModularIndexing(i1, 1, 64)
            + 64 * ModularIndexing(i1 + 64 * r3, 64, 2)
        )
        # check that `i1//64` is removed when i1 is always less than 64,
        # and the next simplificaton doesn't happen
        self.assertEqual(
            sizevars.simplify_with_ranges(expr, var_ranges),
            i1 + 128 * i2 + 64 * ModularIndexing(r3, 1, 2),
        )
        # all the modular indexing should be removed when the body cant be larger than the modulus
        var_ranges[r3] = 2
        self.assertEqual(
            sizevars.simplify_with_ranges(expr, var_ranges), i1 + 128 * i2 + 64 * r3
        )
        # if there are negative terms in ModularIndexing base, we cannot replace it with FloorDiv
        expr = ModularIndexing(i1 - 15, 1, 64)
        self.assertEqual(
            sizevars.simplify_with_ranges(expr, var_ranges),
            ModularIndexing(i1 - 15, 1, 64),
        )
        # small terms should be kept if the rest is not guaranteed to be divisible
        self.assertEqual(
            sizevars.simplify_with_ranges(FloorDiv(r3 + i2 + i1, 32), var_ranges),
            FloorDiv(r3 + i2 + i1, 32),
        )

        expr = ModularIndexing(2 * i2 + r3, 1, 64)
        # modular indexing is removed if base is smaller than modulo
        self.assertEqual(sizevars.simplify_with_ranges(expr, var_ranges), 2 * i2 + r3)

        # check the same thing but with symbolic divisor
        self.assertEqual(FloorDiv(r3 * i0, r3), i0)
        self.assertEqual(ModularIndexing(r3 * i0, r3, 10), ModularIndexing(i0, 1, 10))

        # (10*i) % 10 is always zero and should get optimized away
        self.assertEqual(
            ModularIndexing(i0 + i1 * 10, 1, 10), ModularIndexing(i0, 1, 10)
        )

        # ((20*i)//2) % 10 is always zero and should get optimized away
        self.assertEqual(
            ModularIndexing(i0 + i1 * 20, 2, 10), ModularIndexing(i0, 2, 10)
        )

        # the same things happens with symbolic divisor
        self.assertEqual(
            ModularIndexing(i0 + i1 * i2 * r3, i2, r3), ModularIndexing(i0, i2, r3)
        )

        # if there are negative terms, we cannot optimize away zero terms due to https://github.com/openai/triton/issues/619
        self.assertEqual(
            ModularIndexing(-i0 + i1 * 20, 2, 10), ModularIndexing(-i0 + i1 * 20, 2, 10)
        )
        self.assertEqual(
            ModularIndexing(-15 + i1 * 20, 2, 10), ModularIndexing(-15 + i1 * 20, 2, 10)
        )

        # Constant fold from divisor into base
        self.assertEqual(ModularIndexing(i0 * 4, 2, 10), ModularIndexing(i0 * 2, 1, 10))
        self.assertEqual(FloorDiv(i0 * 4, 2), i0 * 2)

        # Nested modular indexing is correctly simplified
        var_ranges = {sympy.Symbol("i1"): 13, sympy.Symbol("i2"): 121}
        expr = ModularIndexing(ModularIndexing(121 * i1 + i2, 1, 784), 1, 28)
        self.assertEqual(sizevars.simplify_with_ranges(expr, var_ranges), expr)
        expr = ModularIndexing(ModularIndexing(121 * i1 + i2, 1, 784) + 1, 1, 28)
        self.assertEqual(sizevars.simplify_with_ranges(expr, var_ranges), expr)
        var_ranges = {sympy.Symbol("i2"): 784}
        expr = ModularIndexing(ModularIndexing(i2, 1, 28), 7, 4)
        expected = FloorDiv(ModularIndexing(i2, 1, 28), 7)
        self.assertEqual(sizevars.simplify_with_ranges(expr, var_ranges), expected)
        expr = ModularIndexing(ModularIndexing(i2, 1, 28) + 1, 7, 4)
        self.assertEqual(sizevars.simplify_with_ranges(expr, var_ranges), expr)

    def test_indexing_join(self):
        sizevars = SizeVarAllocator()
        i0 = sympy.Symbol("i0", integer=True)
        i1 = sympy.Symbol("i1", integer=True)
        i2 = sympy.Symbol("i2", integer=True)

        # join two ModularIndexing calls into one larger one when possible
        expr1 = ModularIndexing(i0, 1, 32) + 32 * ModularIndexing(i0, 32, 4)
        self.assertEqual(
            sizevars.simplify_with_ranges(expr1, {}), ModularIndexing(i0, 1, 128)
        )

        # it should also work with a scale
        self.assertEqual(
            sizevars.simplify_with_ranges(2 * expr1, {}),
            2 * ModularIndexing(i0, 1, 128),
        )

        # it should work when divisor is not 1
        expr2 = ModularIndexing(i0, 3, 32) + 32 * ModularIndexing(i0, 32 * 3, 4)
        simplified = sizevars.simplify_with_ranges(expr2, {})
        self.assertEqual(simplified, ModularIndexing(i0, 3, 128))
        self.assertEqual(expr2.subs({i0: 39485}), simplified.subs({i0: 39485}))

        # it should not happen in this case as the modulus is wrong
        expr3 = ModularIndexing(i0, 1, 30) + 32 * ModularIndexing(i0, 32, 4)
        self.assertEqual(sizevars.simplify_with_ranges(expr3, {}), expr3)

        # check that it also works with a modulus>1
        expr4 = ModularIndexing(i0, 10, i1) + i1 * ModularIndexing(i0, i1 * 10, i2)
        res0 = expr4.subs({i0: 24056, i1: 13, i2: 19})
        simplified = sizevars.simplify_with_ranges(expr4, {})
        res1 = simplified.subs({i0: 24056, i1: 13, i2: 19})
        self.assertEqual(res0, res1)
        self.assertEqual(simplified, ModularIndexing(i0, 10, i1 * i2))

        # and also works with an offset
        self.assertEqual(
            sizevars.simplify_with_ranges(expr4 + 10, {}),
            ModularIndexing(i0, 10, i1 * i2) + 10,
        )

        # works for ModularIndexing + FloorDiv
        expr5 = 197 * FloorDiv(i0, 197) + ModularIndexing(i0, 1, 197)
        simplified = sizevars.simplify_with_ranges(expr5, {})
        self.assertEqual(simplified, i0)
        self.assertEqual(expr5.subs({i0: 39485}), simplified.subs({i0: 39485}))

        # works with a scale
        self.assertEqual(
            sizevars.simplify_with_ranges(2 * expr5, {}),
            2 * i0,
        )

        # divisor != 1
        expr6 = 197 * FloorDiv(i0, 197 * 3) + ModularIndexing(i0, 3, 197)
        simplified = sizevars.simplify_with_ranges(expr6, {})
        self.assertEqual(simplified, FloorDiv(i0, 3))
        self.assertEqual(expr6.subs({i0: 39485}), simplified.subs({i0: 39485}))


class ExprPrinterTests(InductorTestCase):
    def test_print_pow(self):
        s1 = sympy.Symbol("foo", integer=True)
        s2 = sympy.Symbol("bar", integer=True)
        s3 = sympy.Symbol("baz", integer=True)

        common_cases = [
            # expr, result
            # Test exprs.
            (
                s1 / (2 * s1 - 1) - 1 / (2 * s1 - 1),
                lambda c, L: f"((-1{L})*({c}/((-1{L}) + (2{L}*foo)))) + (foo*({c}/((-1{L}) + (2{L}*foo))))",
            ),
            (s1 / (s2 - s3), lambda c, L: f"foo*({c}/(bar + ((-1{L})*baz)))"),
            # Test Pow directly.
            (
                sympy.Pow(s1 + s2, 0),
                lambda _, L: f"1{L}",
            ),  # note: simplified before _print_Pow
            (
                sympy.Pow(s1 + s2, -3),
                lambda c, _: f"{c}/((bar + foo)*(bar + foo)*(bar + foo))",
            ),
        ]

        gpu_cases = common_cases + [
            (sympy.Pow(s1 + s2, 2), lambda c, L: "(bar + foo)*(bar + foo)")
        ]
        cpu_cases = common_cases + [
            (
                sympy.Pow(s1 + s2, 2),
                lambda c, L: "static_cast<long>((bar + foo)*(bar + foo))",
            )
        ]
        for expr, result in gpu_cases:
            self.assertEqual(texpr(expr), result(1, ""))
            self.assertEqual(pexpr(expr), result(1, ""))
        for expr, result in cpu_cases:
            self.assertEqual(cexpr(expr), result(1.0, "L"))  # 1.0 for FP div

    def test_print_floor(self):
        for integer in [True, False]:
            s1 = sympy.Symbol("s1", integer=integer)
            expr = sympy.floor(s1 / 2)
            if integer:
                self.assertEqual(pexpr(expr), "math.floor((1/2)*s1)")
                self.assertEqual(
                    cexpr(expr), "static_cast<long>(std::floor((1.0/2.0)*s1))"
                )
            else:
                self.assertExpectedInline(pexpr(expr), """math.floor((1/2)*s1)""")
                self.assertExpectedInline(
                    texpr(expr),
                    """libdevice.floor((1/2)*s1).to(tl.int64)""",
                )
                self.assertExpectedInline(cexpr(expr), """std::floor((1.0/2.0)*s1)""")

    def test_print_ceil(self):
        for integer in [True, False]:
            s1 = sympy.Symbol("s1", integer=integer)
            expr = sympy.ceiling(s1 / 2)
            if integer:
                self.assertExpectedInline(pexpr(expr), """math.ceil((1/2)*s1)""")
                self.assertExpectedInline(
                    cexpr(expr), """static_cast<long>(std::ceil((1.0/2.0)*s1))"""
                )
            else:
                self.assertExpectedInline(pexpr(expr), """math.ceil((1/2)*s1)""")
                self.assertExpectedInline(cexpr(expr), """std::ceil((1.0/2.0)*s1)""")

    def test_print_round(self):
        expr = Round(sympy.Symbol("x", integer=True) / 2)
        self.assertExpectedInline(pexpr(expr), """round((1/2)*x)""")
        self.assertExpectedInline(cexpr(expr), """std::lrint((1.0/2.0)*x)""")
        self.assertExpectedInline(
            texpr(expr), """libdevice.llrint((1/2)*x).to(tl.int64)"""
        )

    @parametrize("ndigits", [-1, 0, 1])
    def test_print_round_decimal(self, ndigits):
        expr = RoundDecimal(sympy.Symbol("x", integer=True) / 2, ndigits)
        self.assertEqual(pexpr(expr), f"round((1/2)*x, {ndigits})")
        self.assertEqual(
            cexpr(expr),
            f"static_cast<double>(std::nearbyint(1e{ndigits} * ((1.0/2.0)*x)) * 1e{-ndigits})",
        )
        self.assertEqual(
            texpr(expr),
            f"libdevice.nearbyint(1e{ndigits} * ((1/2)*x)) * 1e{-ndigits}",
        )

        expr = RoundDecimal(sympy.Symbol("x", integer=True), ndigits)
        if ndigits >= 0:
            for do_print in [pexpr, cexpr, texpr]:
                self.assertEqual(do_print(expr), "x")
        else:
            self.assertEqual(pexpr(expr), f"round(x, {ndigits})")
            for do_print in [cexpr, texpr]:
                with self.assertRaisesRegex(
                    ValueError, "only non-negative ndigits are currently supported"
                ):
                    do_print(expr)

    def test_print_floor_div(self):
        for integer in [True, False]:
            s1 = sympy.Symbol("s1", integer=integer)
            s2 = sympy.Symbol("s2", integer=integer)
            expr = FloorDiv(s1, s2)
            self.assertEqual(pexpr(expr), "(s1 // s2)")
            if integer:
                self.assertEqual(cexpr(expr), "c10::div_floor_integer(s1, s2)")
            else:
                self.assertEqual(
                    cexpr(expr),
                    "c10::div_floor_floating(static_cast<double>(s1), static_cast<double>(s2))",
                )

        for integer in [True, False]:
            s1 = sympy.Symbol("s1", integer=integer)
            s2 = sympy.S(-1)
            expr = FloorDiv(s1, s2)
            if integer:
                self.assertEqual(pexpr(expr), "(-1)*s1")
                self.assertEqual(cexpr(expr), "(-1L)*s1")
            else:
                self.assertEqual(pexpr(expr), "(s1 // (-1))")
                self.assertEqual(
                    cexpr(expr),
                    "c10::div_floor_floating(static_cast<double>(s1), static_cast<double>((-1L)))",
                )

    def test_print_Min_Max(self):
        cases = (
            (sympy.Min, "min"),
            (sympy.Max, "max"),
        )
        for f, s in cases:
            x = sympy.Symbol("x", integer=True)
            expr = f(-2, x)
            self.assertEqual(texpr(expr), f"tl.{s}imum(-2, x)")
            self.assertEqual(cexpr(expr), f"std::{s}(-2L, x)")

            expr = f(x, 2 * x, 3 * x)
            self.assertEqual(
                texpr(expr),
                f"tl.{s}imum(x, tl.{s}imum(2*x, 3*x))",
            )
            self.assertEqual(cexpr(expr), f"std::{s}({{x, 2L*x, 3L*x}})")


instantiate_parametrized_tests(ExprPrinterTests)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

    if HAS_CPU or HAS_CUDA:
        run_tests("sympy")
