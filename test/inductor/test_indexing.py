# Owner(s): ["module: inductor"]
import os
import sys
import unittest

import sympy

import torch
from torch._inductor.codegen.cpp import cexpr
from torch._inductor.codegen.triton import texpr
from torch._inductor.codegen.wrapper import pexpr
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.utils._sympy.functions import (
    FloorDiv,
    Mod,
    ModularIndexing,
    PythonMod,
    RoundDecimal,
    RoundToInt,
)


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


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
        var_ranges = {i1: 13, i2: 121}
        expr = ModularIndexing(ModularIndexing(121 * i1 + i2, 1, 784), 1, 28)
        self.assertEqual(sizevars.simplify_with_ranges(expr, var_ranges), expr)
        expr = ModularIndexing(ModularIndexing(121 * i1 + i2, 1, 784) + 1, 1, 28)
        self.assertEqual(sizevars.simplify_with_ranges(expr, var_ranges), expr)
        var_ranges = {i2: 784}
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

    def test_modular_indexing_pairs_merged(self):
        sizevars = SizeVarAllocator()
        x = sympy.Symbol("x", integer=True, positive=True)
        a = 1024
        b = 32
        expr1 = ModularIndexing(x, 1, a)
        expr2 = ModularIndexing(expr1, 1, b)
        expected = ModularIndexing(x, 1, b)

        actual = sizevars.combine_modular_indexing_pairs(expr2)
        self.assertEqual(expected, actual)
        self.assertNotEqual(expr2, actual)

    def test_modular_indexing_pairs_not_merged(self):
        sizevars = SizeVarAllocator()
        x = sympy.Symbol("x", integer=True, positive=True)
        a = 1024
        b = 3  # pick a 'b' that we can not merge
        expr1 = ModularIndexing(x, 1, a)
        expr2 = ModularIndexing(expr1, 1, b)

        actual = sizevars.combine_modular_indexing_pairs(expr2)
        self.assertEqual(expr2, actual)
        self.assertNotEqual(ModularIndexing(x, 1, b), actual)

    def test_expand_floor_div_skipped(self):
        sizevars = SizeVarAllocator()
        x = sympy.Symbol("x", integer=True, positive=True)
        y = sympy.Symbol("y", integer=True, positive=True)

        expr = FloorDiv(x, 2) + FloorDiv(y, 3)
        # The expression can not be simplified since there are multiple
        # FloorDiv. We return False in that case
        self.assertFalse(sizevars.expand_floor_div(expr))

    def test_expand_floor_div_applied(self):
        sizevars = SizeVarAllocator()
        x = sympy.Symbol("x", integer=True, positive=True)
        y = sympy.Symbol("y", integer=True, positive=True)

        expr = x * 5 + FloorDiv(y, 3)
        actual, denominator = sizevars.expand_floor_div(expr)
        self.assertNotEqual(expr, actual)
        expected = FloorDiv(x * 15 + y, 3)
        self.assertEqual(expected, FloorDiv(actual, denominator))

    @unittest.skipUnless(HAS_GPU, "Need GPU for this test")
    def test_int8_unpack(self):
        @torch.compile
        def f(x):
            first_elements = x >> 4
            second_elements = x & 15
            unpacked = torch.stack([first_elements, second_elements], dim=-1).view(
                *x.size()[:-1], -1
            )
            return unpacked * 2

        x = torch.randint(0, 255, (2, 4096, 5504), dtype=torch.uint8, device=GPU_TYPE)

        triton_code = run_and_get_triton_code(f, x)
        # Make sure the 2 load uses simpified indexing rather than something like
        # tl.load(in_ptr0 + ((5504*x1) + (x0 // 2)),
        self.assertEqual(2, triton_code.count("tl.load(in_ptr0 + (x2 // 2),"))
        if DO_PERF_TEST:
            ms = benchmarker.benchmark_gpu(lambda: f(x))
            print(f"{ms=:.03f}")


class ExprPrinterTests(InductorTestCase):
    def test_print_pow(self):
        s1 = sympy.Symbol("foo", integer=True)
        s2 = sympy.Symbol("bar", integer=True)
        s3 = sympy.Symbol("baz", integer=True)

        common_cases = [
            # expr, result
            # Test Pow directly.
            (
                sympy.Pow(s1 + s2, 0),
                lambda _, L: f"1{L}",
            ),  # note: simplified before _print_Pow
        ]

        gpu_cases = common_cases + [
            (sympy.Pow(s1 + s2, 2), lambda c, L: "(bar + foo)*(bar + foo)")
        ]
        cpu_cases = common_cases + [
            (
                sympy.Pow(s1 + s2, 2),
                lambda c, L: "static_cast<int64_t>((bar + foo)*(bar + foo))",
            )
        ]
        for expr, result in gpu_cases:
            self.assertEqual(texpr(expr), result(1, ""))
            self.assertEqual(pexpr(expr), result(1, ""))
        for expr, result in cpu_cases:
            self.assertEqual(
                cexpr(expr),
                result(1.0, "LL")
                if sys.platform in ["darwin", "win32"]
                else result(1.0, "L"),
            )  # 1.0 for FP div

    def test_print_floor(self):
        for integer in [True, False]:
            s1 = sympy.Symbol("s1", integer=integer)
            expr = sympy.floor(s1 / 2)
            if integer:
                self.assertEqual(pexpr(expr), "math.floor((1/2)*s1)")
                self.assertEqual(
                    cexpr(expr), "static_cast<int64_t>(std::floor((1.0/2.0)*s1))"
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
                    cexpr(expr), """static_cast<int64_t>(std::ceil((1.0/2.0)*s1))"""
                )
            else:
                self.assertExpectedInline(pexpr(expr), """math.ceil((1/2)*s1)""")
                self.assertExpectedInline(cexpr(expr), """std::ceil((1.0/2.0)*s1)""")

    def test_print_round(self):
        expr = RoundToInt(sympy.Symbol("x", integer=True) / 2)
        self.assertExpectedInline(pexpr(expr), """round((1/2)*x)""")
        self.assertExpectedInline(cexpr(expr), """std::lrint((1.0/2.0)*x)""")
        self.assertExpectedInline(texpr(expr), """libdevice.llrint((1/2)*x)""")

    def test_print_mod(self):
        x = sympy.Symbol("x", integer=True)
        expr = Mod(x - 1, 2)
        self.assertExpectedInline(pexpr(expr), """((-1) + x) % 2""")
        self.assertExpectedInline(cexpr(expr), """((-1L) + x) % 2L""")
        self.assertExpectedInline(texpr(expr), """((-1) + x) % 2""")

        expr = (x - 10) % x
        self.assertExpectedInline(pexpr(expr), """(-10) % x""")
        self.assertExpectedInline(cexpr(expr), """(-10L) % x""")
        self.assertExpectedInline(texpr(expr), """(-10) % x""")

    def test_print_mod_index(self):
        x = sympy.Symbol("x", integer=True)
        ks = sympy.Symbol("ks", integer=True)
        expr = ModularIndexing(x - 10, ks, ks)
        self.assertExpectedInline(pexpr(expr), """((((-10) + x) // ks) % ks)""")
        self.assertExpectedInline(
            cexpr(expr),
            """(static_cast<int64_t>(c10::div_floor_integer("""
            """static_cast<int64_t>((-10L) + x), static_cast<int64_t>(ks))) % static_cast<int64_t>(ks))""",
        )
        self.assertExpectedInline(texpr(expr), """((((-10) + x) // ks) % ks)""")

    def test_print_python_mod(self):
        x = sympy.Symbol("x", integer=True)
        expr = PythonMod(x - 10, x)
        self.assertExpectedInline(pexpr(expr), """((-10) + x) % x""")
        self.assertExpectedInline(cexpr(expr), """((-10L) + x) % x""")
        self.assertExpectedInline(
            texpr(expr), """triton_helpers.remainder_integer((-10) + x, x)"""
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

    def test_print_floor_div(self):
        s1 = sympy.Symbol("s1", integer=True)
        s2 = sympy.Symbol("s2", integer=True)
        expr = FloorDiv(s1, s2)
        self.assertEqual(pexpr(expr), "s1 // s2")
        self.assertEqual(
            cexpr(expr),
            "c10::div_floor_integer(static_cast<int64_t>(s1), static_cast<int64_t>(s2))",
        )

        s1 = sympy.Symbol("s1", integer=True)
        s2 = sympy.S(-1)
        expr = FloorDiv(s1, s2)
        self.assertEqual(pexpr(expr), "(-1)*s1")
        self.assertEqual(cexpr(expr), "(-1LL)*s1") if sys.platform in [
            "darwin",
            "win32",
        ] else "(-1L)*s1"

    def test_print_Min_Max(self):
        cases = (
            (sympy.Min, "min", "<"),
            (sympy.Max, "max", ">"),
        )
        for f, s, cmp in cases:
            x = sympy.Symbol("x", integer=True)
            expr = f(-2, x)
            self.assertEqual(
                texpr(expr), f"((-2) * ((-2) {cmp}= (x)) + (x) * ((x) {cmp} (-2)))"
            )
            self.assertEqual(
                cexpr(expr),
                f"std::{s}(static_cast<int64_t>(-2LL), static_cast<int64_t>(x))"
                if sys.platform in ["darwin", "win32"]
                else f"std::{s}(static_cast<int64_t>(-2L), static_cast<int64_t>(x))",
            )

            expr = f(x, 2 * x, 3 * x)
            self.assertEqual(
                texpr(expr),
                f"((x) * ((x) {cmp}= (((2*x) * ((2*x) {cmp}= (3*x)) + (3*x) * ((3*x) {cmp} (2*x))))) + (((2*x) * ((2*x) {cmp}= (3*x)) + (3*x) * ((3*x) {cmp} (2*x)))) * ((((2*x) * ((2*x) {cmp}= (3*x)) + (3*x) * ((3*x) {cmp} (2*x)))) {cmp} (x)))",  # noqa: B950 line too long
            )
            self.assertEqual(
                cexpr(expr),
                f"std::{s}({{x, 2LL*x, 3LL*x}})"
                if sys.platform in ["darwin", "win32"]
                else f"std::{s}({{x, 2L*x, 3L*x}})",
            )


instantiate_parametrized_tests(ExprPrinterTests)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests("sympy")
