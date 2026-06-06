# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._higher_order_ops import flex_gemm
from torch._higher_order_ops.flex_gemm import supported_flex_gemm_op_names
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM80OrLater, TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


class TestFlexGemmEpilogueHOP(TestCase):
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

    def test_mm_eager_matches_reference(self):
        a = torch.randn(8, 16)
        b = torch.randn(16, 12)

        def epilogue_fn(acc):
            return (acc + 1).relu()

        actual = flex_gemm(
            torch.mm, (a, b), epilogue_fn, kernel_options={"backend": "QUACK"}
        )

        torch.testing.assert_close(actual, epilogue_fn(a @ b))

    def test_addmm_eager_matches_reference(self):
        bias = torch.randn(8, 12)
        a = torch.randn(8, 16)
        b = torch.randn(16, 12)

        def epilogue_fn(acc):
            return acc.relu()

        actual = flex_gemm(
            torch.addmm,
            (bias, a, b),
            epilogue_fn,
            gemm_kwargs={"beta": 0.5, "alpha": 1.5},
            kernel_options={"backend": "QUACK"},
        )

        torch.testing.assert_close(
            actual, epilogue_fn(torch.addmm(bias, a, b, beta=0.5, alpha=1.5))
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM80OrLater, "SM80+ required")
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

        torch.testing.assert_close(actual, epilogue_fn(a @ b), atol=1e-2, rtol=1e-2)
        self.assertFlexGemmGeneratedCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM80OrLater, "SM80+ required")
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

        torch.testing.assert_close(
            actual,
            epilogue_fn(torch.addmm(bias, a, b, beta=0.5, alpha=1.5)),
            atol=1e-2,
            rtol=1e-2,
        )
        self.assertFlexGemmGeneratedCode(code, "C=")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM80OrLater, "SM80+ required")
    def test_generated_code_rejects_unsupported_epilogue(self):
        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: acc.log(),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        with self.assertRaisesRegex(Exception, "unsupported FlexGEMM epilogue"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM80OrLater, "SM80+ required")
    def test_generated_code_rejects_unknown_kernel_option(self):
        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK", "split_k": 2},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        with self.assertRaisesRegex(Exception, "unsupported FlexGEMM kernel options"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM80OrLater, "SM80+ required")
    def test_generated_code_rejects_tuned_option(self):
        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        with self.assertRaisesRegex(Exception, "tuned=True"):
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
