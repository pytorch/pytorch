# Owner(s): ["module: inductor"]

import importlib
import math
import sys
import unittest

import torch
from torch._higher_order_ops import flex_gemm
from torch._higher_order_ops.flex_gemm import supported_flex_gemm_op_names
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM100OrLater, TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


try:
    import cutlass.cute as cute
except ImportError:
    cute = None

if cute is not None:

    @cute.jit
    def relu_epilogue(acc):
        return cute.where(acc > cute.full_like(acc, 0), acc, cute.full_like(acc, 0))

    @cute.jit
    def affine_aux_epilogue(acc, col_bias, row_scale, tile_bias):
        value = (acc + col_bias) * row_scale + tile_bias
        return cute.where(
            value > cute.full_like(value, 0), value, cute.full_like(value, 0)
        )

    @cute.jit
    def row_scale_epilogue(acc, row_scale):
        return acc * row_scale


class TestFlexGemmRuntimeImport(TestCase):
    def test_import_does_not_load_external_quack(self):
        sys.modules.pop("quack", None)
        importlib.import_module("torch._inductor.kernel.flex_gemm.runtime")
        self.assertNotIn("quack", sys.modules)


class FlexGemmTestCase(TestCase):
    def makeTensor(self, *shape, dtype=torch.bfloat16):
        return torch.testing.make_tensor(
            *shape, device="cuda", dtype=dtype, low=-0.1, high=0.1
        )

    def assertMatchesLowPrecisionEager(
        self,
        actual,
        low_precision_expected,
        high_precision_expected,
        reduction_size,
    ):
        actual_error = (actual.double() - high_precision_expected).abs().mean()
        eager_error = (
            (low_precision_expected.double() - high_precision_expected).abs().mean()
        )
        # Model the extra slack as fp32 accumulator rounding across K plus final output rounding.
        fp32_accumulation_eps = (
            math.sqrt(reduction_size) * torch.finfo(torch.float32).eps
        )
        result_rounding_eps = torch.finfo(actual.dtype).eps
        output_scale = high_precision_expected.abs().mean().item()
        rounding_atol = (fp32_accumulation_eps + result_rounding_eps) * output_scale
        self.assertLessEqual(
            actual_error.item(),
            eager_error.item() + rounding_atol,
            msg=(
                f"actual error {actual_error.item()} exceeded low precision eager "
                f"error {eager_error.item()} with fp32_accumulation_eps="
                f"{fp32_accumulation_eps}, result_rounding_eps="
                f"{result_rounding_eps}, output_scale={output_scale}, "
                f"and atol={rounding_atol}"
            ),
        )


@skipIfNoCuteDSL
@unittest.skipIf(not TEST_CUDA, "CUDA required")
class TestFlexGemmRuntime(FlexGemmTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.relu_epilogue = staticmethod(relu_epilogue)
        cls.affine_aux_epilogue = staticmethod(affine_aux_epilogue)
        cls.row_scale_epilogue = staticmethod(row_scale_epilogue)

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_with_c_alpha_beta_matches_reference(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(0)
        m, n, k = 128, 128, 64
        a = self.makeTensor(k, m).t()
        b = self.makeTensor(k, n)
        c = self.makeTensor(n, m).t()

        out = gemm_epilogue(
            a,
            b,
            self.relu_epilogue,
            "test_flex_gemm_relu_c",
            C=c,
            alpha=0.5,
            beta=1.25,
        )
        low_precision_expected = (
            (0.5 * (a @ b).float() + 1.25 * c.float()).relu().to(out.dtype)
        )
        high_precision_expected = (
            0.5 * (a.double() @ b.double()) + 1.25 * c.double()
        ).relu()
        self.assertMatchesLowPrecisionEager(
            out, low_precision_expected, high_precision_expected, k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_explicit_arg_kind_disambiguates_row_arg(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(2)
        m, n, k = 1, 128, 64
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)

        out = gemm_epilogue(
            a,
            b,
            self.row_scale_epilogue,
            "test_flex_gemm_row_scale",
            out_dtype=torch.float32,
            epilogue_args=(row_scale,),
            epilogue_arg_kinds=("row",),
        )
        self.assertMatchesLowPrecisionEager(
            out,
            (a @ b).float() * row_scale,
            (a.double() @ b.double()) * row_scale.double(),
            k,
        )

    def test_explicit_arg_kind_disambiguates_col_arg_shape(self):
        from torch._inductor.kernel.flex_gemm.runtime import _epilogue_arg_kinds

        a = torch.empty(128, 64)
        b = torch.empty(64, 1)
        col_bias = torch.empty(128, 1)

        self.assertEqual(
            _epilogue_arg_kinds(a, b, (col_bias,), ("col",)),
            ("col",),
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_infers_captured_aux_arg_kinds(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(4)
        m, n, k = 128, 128, 64
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        col_bias = self.makeTensor(m, 1, dtype=torch.float32)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)
        tile_bias = self.makeTensor(m, n, dtype=torch.float32)

        out = gemm_epilogue(
            a,
            b,
            self.affine_aux_epilogue,
            "test_flex_gemm_infer_aux",
            out_dtype=torch.float32,
            epilogue_args=(col_bias, row_scale, tile_bias),
        )
        low_precision_expected = (
            ((a @ b).float() + col_bias) * row_scale + tile_bias
        ).relu()
        high_precision_expected = (
            (a.double() @ b.double() + col_bias.double()) * row_scale.double()
            + tile_bias.double()
        ).relu()
        self.assertMatchesLowPrecisionEager(
            out, low_precision_expected, high_precision_expected, k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_validation_rejects_unsupported_epilogue_arg_combinations(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        a = self.makeTensor(128, 64)
        b = self.makeTensor(64, 128)
        c = self.makeTensor(128, 128)
        row_scale = self.makeTensor(1, 128, dtype=torch.float32)

        with self.assertRaisesRegex(NotImplementedError, "cannot be combined with C"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_c_args",
                C=c,
                epilogue_args=(row_scale,),
                epilogue_arg_kinds=("row",),
            )
        with self.assertRaisesRegex(NotImplementedError, "non-default alpha/beta"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_alpha_args",
                alpha=0.5,
                epilogue_args=(row_scale,),
                epilogue_arg_kinds=("row",),
            )
        with self.assertRaisesRegex(NotImplementedError, "tile/row/col"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_bad_kind",
                epilogue_args=(row_scale,),
                epilogue_arg_kinds=("diag",),
            )
        with self.assertRaisesRegex(RuntimeError, "row epilogue arg shape"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_bad_shape",
                epilogue_args=(row_scale.t(),),
                epilogue_arg_kinds=("row",),
            )
        bad_layout = self.makeTensor(256, 128)[::2, ::2]
        with self.assertRaisesRegex(NotImplementedError, "row- or column-major"):
            gemm_epilogue(
                bad_layout,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_bad_layout",
            )
        with self.assertRaisesRegex(NotImplementedError, "tuned=True"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_tuned",
                tuned=True,
            )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_reads_captured_aux_tensors(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(1)
        m, n, k = 128, 128, 64
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        col_bias = self.makeTensor(m, 1, dtype=torch.float32)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)
        tile_bias = self.makeTensor(m, n, dtype=torch.float32)

        out = gemm_epilogue(
            a,
            b,
            self.affine_aux_epilogue,
            "test_flex_gemm_affine_aux",
            out_dtype=torch.float32,
            epilogue_args=(col_bias, row_scale, tile_bias),
            epilogue_arg_kinds=("col", "row", "tile"),
        )
        low_precision_expected = (
            ((a @ b).float() + col_bias) * row_scale + tile_bias
        ).relu()
        high_precision_expected = (
            (a.double() @ b.double() + col_bias.double()) * row_scale.double()
            + tile_bias.double()
        ).relu()
        self.assertMatchesLowPrecisionEager(
            out, low_precision_expected, high_precision_expected, k
        )


@instantiate_parametrized_tests
class TestFlexGemmEpilogueHOP(FlexGemmTestCase):
    def assertFlexGemmGeneratedCode(self, code, *checks):
        file_check = (
            FileCheck()
            .check(
                "from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue as flex_gemm_epilogue"
            )
            .check("flex_gemm_epilogue(")
        )
        for check in checks:
            file_check = file_check.check(check)
        file_check.check("tuned=False").check("epilogue_source=").check_not(
            "tuned=True"
        ).check_not("from quack").check_not("import quack").check_not(
            "torch._vendor.quack"
        ).run(code)

    def test_supported_op_names_match_pr2b_scope(self):
        self.assertEqual(supported_flex_gemm_op_names(), "mm/addmm")

    def test_default_backend_eager_matches_reference(self):
        a = torch.randn(8, 16)
        b = torch.randn(16, 12)

        def epilogue_fn(acc):
            return acc.relu()

        actual = flex_gemm(torch.mm, (a, b), epilogue_fn)

        torch.testing.assert_close(actual, epilogue_fn(a @ b))

    def test_autograd_is_not_implemented(self):
        a = torch.randn(8, 16, requires_grad=True)
        b = torch.randn(16, 12, requires_grad=True)

        def epilogue_fn(acc):
            return acc.relu()

        actual = flex_gemm(torch.mm, (a, b), epilogue_fn)

        with self.assertRaisesRegex(RuntimeError, "flex_gemm"):
            actual.sum().backward()

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_compiled_matches_reference(self):
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            return (acc + 1).relu()

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.mm,
            (a, b),
            epilogue_fn,
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_alpha_clamp_compiled_matches_reference(self):
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            return torch.add(acc, 2.0, alpha=0.25).clamp(min=0.0)

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.mm,
            (a, b),
            epilogue_fn,
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_dynamic_shapes_compiled_matches_reference(self):
        def epilogue_fn(acc):
            return (acc + 1).relu()

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        for m in (128, 256):
            a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
            actual = compiled(a, b)
            self.assertMatchesLowPrecisionEager(
                actual,
                epilogue_fn(a @ b),
                epilogue_fn(a.double() @ b.double()),
                a.shape[1],
            )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_addmm_compiled_matches_reference(self):
        bias = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            return acc.relu()

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.addmm,
            (bias, a, b),
            epilogue_fn,
            gemm_kwargs={"beta": 0.5, "alpha": 1.5},
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.addmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.addmm(bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5)
            ),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_generated_code_calls_flex_gemm_adapter(self):
        def epilogue_fn(acc):
            return (acc + 1).relu()

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": False},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_addmm_generated_code_calls_flex_gemm_adapter(self):
        def epilogue_fn(acc):
            return acc.relu()

        def fn(bias, a, b):
            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                epilogue_fn,
                gemm_kwargs={"beta": 0.5, "alpha": 1.5},
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.addmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.addmm(bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5)
            ),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(code, "C=")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            (
                "unsupported_epilogue",
                lambda acc: acc.sum(dim=1, keepdim=True),
                {"backend": "QUACK"},
                "unsupported FlexGEMM epilogue",
            ),
            (
                "unknown_kernel_option",
                lambda acc: acc.relu(),
                {"backend": "QUACK", "split_k": 2},
                "unsupported FlexGEMM kernel options",
            ),
            (
                "tuned_option",
                lambda acc: acc.relu(),
                {"backend": "QUACK", "tuned": True},
                "tuned=True",
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_code_rejects_unsupported_cases(self, case):
        _, epilogue_fn, kernel_options, error = case

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options=kernel_options,
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        with self.assertRaisesRegex(Exception, error):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    def test_rejects_unsupported_quack_op(self):
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)

        with self.assertRaisesRegex(RuntimeError, "unsupported GEMM op"):
            flex_gemm(
                torch.ops.aten.bmm.default,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK"},
            )

    def test_rejects_unknown_backend(self):
        a = torch.randn(8, 16)
        b = torch.randn(16, 12)

        with self.assertRaisesRegex(RuntimeError, "unsupported FlexGEMM backend"):
            flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "CUTLASS"},
            )


if __name__ == "__main__":
    run_tests()
