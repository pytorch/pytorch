# Owner(s): ["module: inductor"]

import torch
import torch._inductor.config as inductor_config
from torch._C import FileCheck
from torch._higher_order_ops import gemm_epilogue_fusion
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.triton_utils import requires_cuda_and_triton


class GemmEpilogueFusionTests(TestCase):
    def test_eager_matches_reference(self):
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)

        actual = gemm_epilogue_fusion(
            torch.ops.aten.mm.default,
            (a, b),
            lambda acc: acc.relu(),
        )

        torch.testing.assert_close(actual, (a @ b).relu())

    def test_addmm_eager_matches_reference(self):
        bias = torch.randn(2, 4)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)

        actual = gemm_epilogue_fusion(
            torch.ops.aten.addmm.default,
            (bias, a, b),
            lambda acc: acc.relu(),
        )

        torch.testing.assert_close(actual, torch.addmm(bias, a, b).relu())

    def test_bmm_eager_matches_reference(self):
        a = torch.randn(5, 2, 3)
        b = torch.randn(5, 3, 4)

        actual = gemm_epilogue_fusion(
            torch.ops.aten.bmm.default,
            (a, b),
            lambda acc: acc.relu(),
        )

        torch.testing.assert_close(actual, torch.bmm(a, b).relu())

    def test_baddbmm_eager_matches_reference(self):
        bias = torch.randn(5, 2, 4)
        a = torch.randn(5, 2, 3)
        b = torch.randn(5, 3, 4)

        actual = gemm_epilogue_fusion(
            torch.ops.aten.baddbmm.default,
            (bias, a, b),
            lambda acc: acc.relu(),
        )

        torch.testing.assert_close(actual, torch.baddbmm(bias, a, b).relu())

    def test_cutlass_backend_is_accepted(self):
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)

        actual = gemm_epilogue_fusion(
            torch.ops.aten.mm.default,
            (a, b),
            lambda acc: acc.relu(),
            kernel_options={"backend": "CUTLASS"},
        )

        torch.testing.assert_close(actual, (a @ b).relu())

    def test_exports_as_gemm_epilogue_fusion_region(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: acc.relu(),
            )

        gm, _ = torch._dynamo.export(fn, torch.randn(2, 3), torch.randn(3, 4))

        self.assertIn("gemm_epilogue_fusion", gm.code)
        self.assertIn("{'backend': 'TRITON'}", gm.code)
        self.assertIn("aten.mm.default", gm.gemm_epilogue_fusion_body_0.code)
        self.assertIn("relu", gm.gemm_epilogue_fusion_body_0.code)

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm=False)
    def test_cuda_inductor_forces_template_epilogue_fusion(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: acc.relu(),
            )

        a = torch.randn(128, 128, device="cuda")
        b = torch.randn(128, 128, device="cuda")

        actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

        torch.testing.assert_close(actual, fn(a, b))
        self.assertFalse(inductor_config.max_autotune_gemm)

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm=False)
    def test_cuda_inductor_addmm_epilogue_fuses(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
            )

        bias = torch.randn(128, 64, device="cuda", dtype=torch.float16)
        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_").check_not("extern_kernels.addmm").run(code)

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm=False)
    def test_cuda_inductor_bmm_epilogue_fuses(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.bmm.default,
                (a, b),
                lambda acc: acc.relu(),
            )

        a = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(4, 128, 32, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_").check_not("extern_kernels.bmm").run(code)

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm=False)
    def test_cuda_inductor_baddbmm_epilogue_fuses(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.baddbmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
            )

        bias = torch.randn(4, 64, 32, device="cuda", dtype=torch.float16)
        a = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(4, 128, 32, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_").check_not("extern_kernels.baddbmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_cutlass_backend_uses_cutlass_template_fusion(self):
        from torch._inductor.codegen.cutlass.utils import try_import_cutlass

        if not try_import_cutlass() or torch.version.cuda is None:
            self.skipTest("CUTLASS is not available")

        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "CUTLASS"},
            )

        a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        b = torch.randn(512, 512, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("async_compile.cuda").check("cutlass").check_not(
            "extern_kernels.mm"
        ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_relu_to_quack_hook(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("@cute.jit").check("def flex_gemm_quack_epilogue_").check(
            "gemm_epilogue("
        ).check_not("call_quack_gemm_epilogue").check_not(
            "torch.ops.flex_gemm"
        ).check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_generates_pointwise_epilogue(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: (acc + 1.0).relu(),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("@cute.jit").check(
            "tmp0 = (acc + cute.full_like(acc, 1.0))"
        ).check("gemm_epilogue(").check_not("call_quack_gemm_epilogue").check_not(
            "torch.ops.flex_gemm"
        ).check_not("extern_kernels.mm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_generates_cutedsl_math_epilogue(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: torch.abs(acc),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("from cutlass._mlir.dialects import math as mlir_math").check(
            "mlir_math.absf"
        ).check("gemm_epilogue(").check_not("call_quack_gemm_epilogue").run(code)


if __name__ == "__main__":
    run_tests(needs="filelock")
