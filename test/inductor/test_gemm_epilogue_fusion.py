# Owner(s): ["module: inductor"]

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch._C import FileCheck
from torch._higher_order_ops import (
    addmm_epilogue,
    baddbmm_epilogue,
    bmm_epilogue,
    gemm_epilogue_fusion,
    grouped_mm_epilogue,
    matmul_epilogue,
    mm_epilogue,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_MX_GEMM,
)
from torch.testing._internal.common_quantized import ceil_div, to_blocked
from torch.testing._internal.inductor_utils import _quantize_tensorwise
from torch.testing._internal.triton_utils import requires_cuda_and_triton


class GemmEpilogueFusionTests(TestCase):
    def _grouped_mm_inputs(self, op, device):
        dtype = torch.bfloat16
        match op:
            case "2d/2d":
                m, n, k, groups = 64, 128, 64, 2
                a = torch.randn(m, k * groups, device=device, dtype=dtype)
                b = torch.randn(n, k * groups, device=device, dtype=dtype).t()
                offs = torch.arange(
                    k, k * groups + 1, k, device=device, dtype=torch.int32
                )
            case "2d/3d":
                groups, m, n, k = 2, 64, 128, 64
                a = torch.randn(m * groups, k, device=device, dtype=dtype)
                b = torch.randn(groups, n, k, device=device, dtype=dtype).transpose(
                    -2, -1
                )
                offs = torch.arange(
                    m, m * groups + 1, m, device=device, dtype=torch.int32
                )
            case "3d/2d":
                groups, m, n, k = 2, 64, 128, 64
                a = torch.randn(groups, m, k, device=device, dtype=dtype)
                b = torch.randn(n * groups, k, device=device, dtype=dtype).t()
                offs = torch.arange(
                    n, n * groups + 1, n, device=device, dtype=torch.int32
                )
            case "3d/3d":
                groups, m, n, k = 2, 64, 128, 64
                a = torch.randn(groups, m, k, device=device, dtype=dtype)
                b = torch.randn(groups, n, k, device=device, dtype=dtype).transpose(
                    -2, -1
                )
                offs = None
            case _:
                raise AssertionError(f"invalid grouped_mm op shape: {op}")
        return a, b, offs

    def test_eager_matches_reference(self):
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)

        actual = gemm_epilogue_fusion(
            torch.ops.aten.mm.default,
            (a, b),
            lambda acc: acc.relu(),
        )

        torch.testing.assert_close(actual, (a @ b).relu())

    def test_shorthand_mm_eager_matches_reference(self):
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)

        actual = gemm_epilogue_fusion(
            torch.mm,
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

    def test_convenience_wrappers_eager_match_reference(self):
        bias = torch.randn(2, 4)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        batch_bias = torch.randn(5, 2, 4)
        batch_a = torch.randn(5, 2, 3)
        batch_b = torch.randn(5, 3, 4)

        torch.testing.assert_close(
            mm_epilogue(a, b, lambda acc: acc.relu()), (a @ b).relu()
        )
        torch.testing.assert_close(
            addmm_epilogue(bias, a, b, lambda acc: acc.relu(), alpha=0.5, beta=0.25),
            torch.addmm(bias, a, b, alpha=0.5, beta=0.25).relu(),
        )
        torch.testing.assert_close(
            bmm_epilogue(batch_a, batch_b, lambda acc: acc.relu()),
            torch.bmm(batch_a, batch_b).relu(),
        )
        torch.testing.assert_close(
            baddbmm_epilogue(
                batch_bias,
                batch_a,
                batch_b,
                lambda acc: acc.relu(),
                alpha=0.5,
                beta=0.25,
            ),
            torch.baddbmm(batch_bias, batch_a, batch_b, alpha=0.5, beta=0.25).relu(),
        )
        torch.testing.assert_close(
            matmul_epilogue(a, b, lambda acc: acc.relu()), torch.matmul(a, b).relu()
        )
        torch.testing.assert_close(
            matmul_epilogue(batch_a, batch_b, lambda acc: acc.relu()),
            torch.matmul(batch_a, batch_b).relu(),
        )

    def test_matmul_epilogue_rejects_non_gemm_shapes(self):
        with self.assertRaisesRegex(NotImplementedError, "2D mm and 3D bmm"):
            matmul_epilogue(torch.randn(3), torch.randn(3), lambda acc: acc.relu())

    def test_grouped_mm_epilogue_eager_matches_reference(self):
        for op in ("2d/2d", "2d/3d", "3d/2d", "3d/3d"):
            with self.subTest(op=op):
                a, b, offs = self._grouped_mm_inputs(op, "cpu")

                actual = grouped_mm_epilogue(a, b, lambda acc: acc.relu(), offs=offs)

                torch.testing.assert_close(actual, F.grouped_mm(a, b, offs=offs).relu())

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

    def test_make_fx_preserves_gemm_kwargs(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                gemm_kwargs={"alpha": 0.5, "beta": 0.25},
            )

        bias = torch.randn(2, 4)
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)

        actual = make_fx(fn)(bias, a, b)(bias, a, b)

        torch.testing.assert_close(
            actual, torch.addmm(bias, a, b, alpha=0.5, beta=0.25).relu()
        )

    def test_exports_as_gemm_epilogue_fusion_region(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: acc.relu(),
            )

        gm, _ = torch._dynamo.export(fn, torch.randn(2, 3), torch.randn(3, 4))

        self.assertIn("gemm_epilogue_fusion", gm.code)
        self.assertIn("{'backend': 'TRITON', 'SPLIT_K': False}", gm.code)
        self.assertIn("aten.mm.default", gm.gemm_epilogue_fusion_body_0.code)
        self.assertIn("relu", gm.gemm_epilogue_fusion_body_0.code)

    def test_shorthand_mm_exports_as_gemm_epilogue_fusion_region(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.mm,
                (a, b),
                lambda acc: acc.relu(),
            )

        gm, _ = torch._dynamo.export(fn, torch.randn(2, 3), torch.randn(3, 4))

        self.assertIn("gemm_epilogue_fusion", gm.code)
        self.assertIn("aten.mm.default", gm.code)
        self.assertIn("aten.mm.default", gm.gemm_epilogue_fusion_body_0.code)

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm=False)
    def test_cuda_inductor_grouped_mm_epilogue_fuses(self):
        if torch.cuda.get_device_capability() < (9, 0):
            self.skipTest("TRITON grouped_mm template requires SM90+")

        def fn(a, b, offs):
            return grouped_mm_epilogue(a, b, lambda acc: acc.relu(), offs=offs)

        for op in ("2d/2d", "2d/3d", "3d/2d", "3d/3d"):
            with self.subTest(op=op):
                a, b, offs = self._grouped_mm_inputs(op, "cuda")

                actual, (code,) = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b, offs
                )

                torch.testing.assert_close(actual, fn(a, b, offs), atol=1e-2, rtol=1e-2)
                FileCheck().check("triton_").check("grouped_mm").check_not(
                    "extern_kernels._grouped_mm"
                ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_grouped_mm_relu_to_quack_hook(self):
        def fn(a, b, offs):
            return grouped_mm_epilogue(
                a,
                b,
                lambda acc: acc.relu(),
                offs=offs,
                kernel_options={"backend": "QUACK"},
            )

        for op in ("2d/3d", "2d/2d"):
            with self.subTest(op=op):
                a, b, offs = self._grouped_mm_inputs(op, "cuda")
                if op == "2d/2d":
                    a = a.t().contiguous().t()
                    b = b.contiguous()

                actual, (code,) = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True), a, b, offs
                )

                torch.testing.assert_close(actual, fn(a, b, offs), atol=1e-2, rtol=1e-2)
                FileCheck().check("@cute.jit").check("gemm_epilogue(").check(
                    "offs="
                ).check_not("extern_kernels._grouped_mm").run(code)

    def _run_split_k_epilogue_test(self, backend, expected_kernel):
        if backend == "QUACK":
            try:
                import quack  # noqa: F401
            except ImportError:
                self.skipTest("QuACK is not available")

        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": backend, "SPLIT_K": True},
            )

        a = torch.randn(8, 8192, device="cuda", dtype=torch.float16)
        b = torch.randn(8192, 8, device="cuda", dtype=torch.float16)

        actual, codes = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check(expected_kernel).run("\n".join(codes))

    @requires_cuda_and_triton
    @inductor_config.patch("triton.num_decompose_k_splits", 2)
    @inductor_config.patch("triton.decompose_k_threshold", 0)
    def test_cuda_inductor_triton_epilogue_split_k_allows_decompose_k(self):
        self._run_split_k_epilogue_test("TRITON", "decompose_k_fp32_mm")

    @requires_cuda_and_triton
    @inductor_config.patch("triton.num_decompose_k_splits", 2)
    @inductor_config.patch("triton.decompose_k_threshold", 0)
    def test_cuda_inductor_quack_epilogue_split_k_uses_quack_bmm(self):
        self._run_split_k_epilogue_test("QUACK", "quack_gemm")

    @requires_cuda_and_triton
    @inductor_config.patch(max_autotune_gemm=False)
    @inductor_config.patch("triton.num_decompose_k_splits", 2)
    @inductor_config.patch("triton.decompose_k_threshold", 0)
    def test_cuda_inductor_triton_epilogue_disables_decompose_k(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: acc.relu(),
            )

        a = torch.randn(16, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, 16, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_").check_not("decompose_k_mm").run(code)

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
    def test_cuda_inductor_addmm_alpha_beta_epilogue_fuses(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                gemm_kwargs={"alpha": 0.5, "beta": 0.25},
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
    def test_cuda_inductor_scaled_mm_epilogue_fuses(self):
        if not PLATFORM_SUPPORTS_FP8:
            self.skipTest("FP8 is not supported")

        def fn(a, b, scale_a, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={"out_dtype": torch.bfloat16},
            )

        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        a, scale_a = _quantize_tensorwise(x, torch.float8_e4m3fn)
        w_fp8, scale_b = _quantize_tensorwise(w, torch.float8_e4m3fn)
        b = w_fp8.t()

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            scale_a,
            scale_b,
        )

        torch.testing.assert_close(
            actual, fn(a, b, scale_a, scale_b), atol=0.05, rtol=1e-2
        )
        FileCheck().check("triton_tem_fused__scaled_mm").check("maximum").check_not(
            "extern_kernels._scaled_mm"
        ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_scaled_mm_relu_to_quack_hook(self):
        if not PLATFORM_SUPPORTS_MX_GEMM:
            self.skipTest("MX GEMM is not supported")

        def fn(a, b, scale_a, scale_b):
            return gemm_epilogue_fusion(
                torch.ops.aten._scaled_mm.default,
                (a, b, scale_a, scale_b),
                lambda acc: acc.relu(),
                gemm_kwargs={"out_dtype": torch.bfloat16},
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 32, 128
        a = torch.eye(m, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = (
            torch.eye(n, k, device="cuda", dtype=torch.bfloat16)
            .to(torch.float8_e4m3fn)
            .t()
        )
        scale_a = to_blocked(
            torch.full(
                (m, ceil_div(k, 32)),
                1.0,
                device="cuda",
                dtype=torch.float8_e8m0fnu,
            )
        )
        scale_b = to_blocked(
            torch.full(
                (n, ceil_div(k, 32)),
                1.0,
                device="cuda",
                dtype=torch.float8_e8m0fnu,
            )
        )

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            scale_a,
            scale_b,
        )

        torch.testing.assert_close(
            actual, fn(a, b, scale_a, scale_b), atol=0.05, rtol=1e-2
        )
        FileCheck().check("@cute.jit").check("gemm_epilogue(").check("scale_a=").check(
            "scale_b="
        ).check_not("extern_kernels._scaled_mm").run(code)

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
    def test_cuda_inductor_quack_backend_routes_addmm_relu_to_quack_hook(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(128, 64, device="cuda", dtype=torch.float16)
        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("@cute.jit").check("gemm_epilogue(").check(
            "C=arg0_1"
        ).check_not("extern_kernels.addmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_addmm_alpha_beta_to_quack_hook(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                gemm_kwargs={"alpha": 0.5, "beta": 0.25},
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(128, 64, device="cuda", dtype=torch.float16)
        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("gemm_epilogue(").check("alpha=0.5").check(
            "beta=0.25"
        ).check_not("extern_kernels.addmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_bmm_relu_to_quack_hook(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.bmm.default,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(4, 128, 32, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("@cute.jit").check("gemm_epilogue(").check_not(
            "extern_kernels.bmm"
        ).run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_baddbmm_alpha_beta_to_quack_hook(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.baddbmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                gemm_kwargs={"alpha": 0.5, "beta": 0.25},
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(4, 64, 32, device="cuda", dtype=torch.float16)
        a = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(4, 128, 32, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("gemm_epilogue(").check("alpha=0.5").check(
            "beta=0.25"
        ).check_not("extern_kernels.baddbmm").run(code)

    @requires_cuda_and_triton
    def test_cuda_inductor_quack_backend_routes_baddbmm_relu_to_quack_hook(self):
        def fn(bias, a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.baddbmm.default,
                (bias, a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(4, 64, 32, device="cuda", dtype=torch.float16)
        a = torch.randn(4, 64, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(4, 128, 32, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        torch.testing.assert_close(actual, fn(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("@cute.jit").check("gemm_epilogue(").check(
            "C=arg0_1"
        ).check_not("extern_kernels.baddbmm").run(code)

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
    def test_cuda_inductor_quack_backend_honors_epilogue_add_alpha(self):
        def fn(a, b):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (a, b),
                lambda acc: torch.add(acc, 1.0, alpha=2.0).relu(),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("@cute.jit").check("gemm_epilogue(").check_not(
            "call_quack_gemm_epilogue"
        ).run(code)

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
