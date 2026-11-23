# Owner(s): ["module: inductor"]
import os
import unittest
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters
from torch._inductor.fx_passes import b2b_gemm
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import BF16X9_SUPPORTED
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    recover_orig_fp32_precision,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


@skipIfXpu(msg="Segmentation fault on CI machine")
class B2BGEMMTest(TestCase):
    device = GPU_TYPE

    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_left_assoc_good_shape(self):
        """
        left_assoc means the pattern is (subgraph(A @ B) @ C)
        good_shape means the sizes are good for b2b_gemm
        """

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            g = torch.nn.GELU()
            return torch.mm(g(torch.mm(m1, m2)), m3)

        def f_32(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            """
            When the optimization is applied,
            the Triton kernel is more precise than the above f,
            because it internally uses float32 for accumulation while the above f uses float16.
            To ensure a fair comparison,
            we promote the baseline f to float32 for precision comparison.
            This actually reduced some atol's in the tests from 0.2 to 0.1.
            """
            m1 = m1.to(torch.float32)
            m2 = m2.to(torch.float32)
            m3 = m3.to(torch.float32)
            return f(m1, m2, m3).to(torch.float16)

        f_opt = torch.compile(f)
        A = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        B = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        C = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        res = f_opt(A, B, C)
        self.assertTrue(torch.allclose(f_32(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertGreater(counters["inductor"]["b2b_gemm"], 0)

    @unittest.skipUnless(
        BF16X9_SUPPORTED, "requires CUDA 12.9+ and compute capability 10.0 or 10.3"
    )
    @recover_orig_fp32_precision
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_bfx9_rejects_mixed_dtype_b2b_gemm(self):
        def fn(a, b, c):
            return torch.mm(torch.mm(a, b).float(), c)

        a = torch.randn((256, 32), device=self.device, dtype=torch.float16)
        b = torch.randn((32, 256), device=self.device, dtype=torch.float16)
        c = torch.randn((256, 32), device=self.device, dtype=torch.float32)
        for precision, should_fuse in (("ieee", True), ("bfx9", False)):
            with self.subTest(fp32_precision=precision):
                torch.backends.cuda.matmul.fp32_precision = precision
                torch._dynamo.reset()
                counters.clear()
                actual = torch.compile(fn)(a, b, c)
                if should_fuse:
                    self.assertGreater(counters["inductor"]["b2b_gemm"], 0)
                else:
                    self.assertEqual(actual, fn(a, b, c))
                    self.assertEqual(counters["inductor"]["b2b_gemm"], 0)

    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_right_assoc_good_shape(self):
        """
        right_assoc means the pattern is (A @ subgraph(B @ C))
        good_shape means the sizes are good for b2b_gemm
        """

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            g = torch.nn.ReLU()
            return torch.mm(m1, g(torch.mm(m2, m3)))

        def f_32(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            m1 = m1.to(torch.float32)
            m2 = m2.to(torch.float32)
            m3 = m3.to(torch.float32)
            return f(m1, m2, m3).to(torch.float16)

        f_opt = torch.compile(f)
        A = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        B = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        C = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        res = f_opt(A, B, C)
        self.assertTrue(torch.allclose(f_32(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertGreater(counters["inductor"]["b2b_gemm"], 0)

    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_trivial_left_assoc_good_shape(self):
        """
        trivial_left_assoc means the pattern is ((A @ B) @ C)
        good_shape means the sizes are good for b2b_gemm
        """

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            return torch.mm(torch.mm(m1, m2), m3)

        def f_32(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            m1 = m1.to(torch.float32)
            m2 = m2.to(torch.float32)
            m3 = m3.to(torch.float32)
            return f(m1, m2, m3).to(torch.float16)

        f_opt = torch.compile(f)
        A = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        B = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        C = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        res = f_opt(A, B, C)
        self.assertTrue(torch.allclose(f_32(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertGreater(counters["inductor"]["b2b_gemm"], 0)

    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_trivial_right_assoc_good_shape(self):
        """
        trivial_right_assoc means the pattern is (A @ (B @ C))
        good_shape means the sizes are good for b2b_gemm
        """

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            return torch.mm(m1, torch.mm(m2, m3))

        def f_32(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            m1 = m1.to(torch.float32)
            m2 = m2.to(torch.float32)
            m3 = m3.to(torch.float32)
            return f(m1, m2, m3).to(torch.float16)

        f_opt = torch.compile(f)
        A = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        B = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        C = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        res = f_opt(A, B, C)
        self.assertTrue(torch.allclose(f_32(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertGreater(counters["inductor"]["b2b_gemm"], 0)

    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_bad_pattern_good_shape(self):
        """
        bad_pattern means the code does not contain the supported patterns
        """

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            mm1 = torch.mm(m1, m2)
            mm2 = torch.mm(mm1, m3)
            return torch.mm(mm1, mm2)

        f_opt = torch.compile(f)
        A = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        B = torch.randn((32, 256), device=GPU_TYPE, dtype=torch.float16)
        C = torch.randn((256, 32), device=GPU_TYPE, dtype=torch.float16)
        res, codes = run_and_get_code(f_opt, A, B, C)
        code = "\n".join(codes)
        self.assertTrue(torch.allclose(f(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertTrue("B2B_GEMM_LEFT_TRITON_ENTRANCE" not in code)
        self.assertTrue("B2B_GEMM_RIGHT_TRITON_ENTRANCE" not in code)

    @torch._dynamo.config.patch(recompile_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_good_pattern_bad_shape(self):
        """
        bad_shape means the sizes are not good for b2b_gemm
        """

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            return torch.mm(torch.mm(m1, m2), m3)

        f_opt = torch.compile(f)
        A = torch.randn((100, 100), device=GPU_TYPE, dtype=torch.float16)
        B = torch.randn((100, 100), device=GPU_TYPE, dtype=torch.float16)
        C = torch.randn((100, 100), device=GPU_TYPE, dtype=torch.float16)
        res, codes = run_and_get_code(f_opt, A, B, C)
        code = "\n".join(codes)
        self.assertTrue(torch.allclose(f(A, B, C), res, atol=0.1, rtol=0.01))
        self.assertTrue("B2B_GEMM_LEFT_TRITON_ENTRANCE" not in code)
        self.assertTrue("B2B_GEMM_RIGHT_TRITON_ENTRANCE" not in code)

    @unittest.skipIf(os.environ.get("DO_PERF_TEST") != "1", "Perf test not enabled")
    @torch._dynamo.config.patch(recompile_limit=32)
    def test_plain_b2b_gemm_performance(self):
        """compare torch.compile(f, b2b_gemm = off) with torch.compile(f, b2b_gemm = on)"""

        def run_with_b2b_gemm_off(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                return torch.mm(torch.mm(m1, m2), m3)

            f_opt = torch.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        @torch._inductor.config.patch(b2b_gemm_pass=True)
        def run_with_b2b_gemm_on(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                return torch.mm(torch.mm(m1, m2), m3)

            f_opt = torch.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        Ms = [128, 256, 300, 400, 512]
        Ns = [16, 20, 32, 40, 50, 64]
        speedups = []
        print("Perf Test for Plain B2B-GEMM:")
        print("Speedups".ljust(10), end="")
        for N in Ns:
            print(f"N = {N}".ljust(10), end="")
        print()
        for M in Ms:
            print(f"M = {M}".ljust(10), end="")
            for N in Ns:
                O, P = M, N
                A = torch.randn((M, N), device=GPU_TYPE, dtype=torch.float16)
                B = torch.randn((N, O), device=GPU_TYPE, dtype=torch.float16)
                C = torch.randn((O, P), device=GPU_TYPE, dtype=torch.float16)
                speedup = run_with_b2b_gemm_off(A, B, C) / run_with_b2b_gemm_on(A, B, C)
                print(f"{round(speedup, 3)}".ljust(10), end="")
                speedups.append(speedup)
            print()

        average_speedup = 1.0
        for s in speedups:
            average_speedup *= s
        average_speedup = average_speedup ** (1 / len(speedups))
        print(f"Average speedup: {round(average_speedup, 3)}")

        # flaky test assertion: disabled
        # self.assertTrue(average_speedup > 1)

    @unittest.skipIf(os.environ.get("DO_PERF_TEST") != "1", "Perf test not enabled")
    @torch._dynamo.config.patch(recompile_limit=32)
    def test_gelu_b2b_gemm_performance(self):
        """compare torch.compile(f, b2b_gemm = off) with torch.compile(f, b2b_gemm = on)"""

        def run_with_b2b_gemm_off(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                g = torch.nn.GELU()
                return torch.mm(g(torch.mm(m1, m2)), m3)

            f_opt = torch.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        @torch._inductor.config.patch(b2b_gemm_pass=True)
        def run_with_b2b_gemm_on(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                g = torch.nn.GELU()
                return torch.mm(g(torch.mm(m1, m2)), m3)

            f_opt = torch.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        Ms = [128, 256, 300, 400, 512]
        Ns = [16, 20, 32, 40, 50, 64]
        speedups = []
        print("Perf Test for GELU B2B-GEMM:")
        print("Speedups".ljust(10), end="")
        for N in Ns:
            print(f"N = {N}".ljust(10), end="")
        print()
        for M in Ms:
            print(f"M = {M}".ljust(10), end="")
            for N in Ns:
                O, P = M, N
                A = torch.randn((M, N), device=GPU_TYPE, dtype=torch.float16)
                B = torch.randn((N, O), device=GPU_TYPE, dtype=torch.float16)
                C = torch.randn((O, P), device=GPU_TYPE, dtype=torch.float16)
                speedup = run_with_b2b_gemm_off(A, B, C) / run_with_b2b_gemm_on(A, B, C)
                print(f"{round(speedup, 3)}".ljust(10), end="")
                speedups.append(speedup)
            print()

        average_speedup = 1.0
        for s in speedups:
            average_speedup *= s
        average_speedup = average_speedup ** (1 / len(speedups))
        print(f"Average speedup: {round(average_speedup, 3)}")

        # flaky test assertion: disabled
        # self.assertTrue(average_speedup > 1)

    @unittest.skipIf(os.environ.get("DO_PERF_TEST") != "1", "Perf test not enabled")
    @torch._dynamo.config.patch(recompile_limit=32)
    def test_gelu_mlp_b2b_gemm_performance(self):
        """compare torch.compile(f, b2b_gemm = off) with torch.compile(f, b2b_gemm = on)"""

        def run_with_b2b_gemm_off(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                g = torch.nn.GELU()
                return torch.mm(g(torch.mm(m1, m2)), m3)

            f_opt = torch.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        @torch._inductor.config.patch(b2b_gemm_pass=True)
        def run_with_b2b_gemm_on(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                g = torch.nn.GELU()
                return torch.mm(g(torch.mm(m1, m2)), m3)

            f_opt = torch.compile(f, dynamic=False)
            return benchmarker.benchmark(f_opt, (m1, m2, m3), {}, warmup=100, rep=500)

        Ms = [128, 256, 300, 400, 512]
        Ns = [16, 20, 32, 40, 50, 64]
        speedups = []
        print("Perf Test for GELU B2B-GEMM (MLP):")
        print("Speedups".ljust(10), end="")
        for N in Ns:
            print(f"N = {N}".ljust(10), end="")
        print()
        for M in Ms:
            print(f"M = {M}".ljust(10), end="")
            for N in Ns:
                O, P = N, N
                A = torch.randn((M, N), device=GPU_TYPE, dtype=torch.float16)
                B = torch.randn((N, O), device=GPU_TYPE, dtype=torch.float16)
                C = torch.randn((O, P), device=GPU_TYPE, dtype=torch.float16)
                speedup = run_with_b2b_gemm_off(A, B, C) / run_with_b2b_gemm_on(A, B, C)
                print(f"{round(speedup, 3)}".ljust(10), end="")
                speedups.append(speedup)
            print()

        average_speedup = 1.0
        for s in speedups:
            average_speedup *= s
        average_speedup = average_speedup ** (1 / len(speedups))
        print(f"Average speedup: {round(average_speedup, 3)}")

        # flaky test assertion: disabled
        # self.assertTrue(average_speedup > 1)


class B2BGEMMPrecisionTest(TestCase):
    def _make_inputs(self, device, is_left_assoc, dtype, value=None):
        m, n = (256, 32) if is_left_assoc else (32, 256)
        shapes = ((m, n), (n, m), (m, n))
        torch.manual_seed(314159)
        if value is None:
            return [torch.randn(shape, device=device, dtype=dtype) for shape in shapes]
        return [
            torch.full(shape, value, device=device, dtype=dtype) for shape in shapes
        ]

    @torch._inductor.config.patch(b2b_gemm_pass=True, force_disable_caches=True)
    def _compile_fused(self, fn, inputs, is_left_assoc):
        select_algorithm = b2b_gemm.autotune_select_algorithm

        def select_fused(name, choices, *args, **kwargs):
            if name == "b2b_gemm":
                choices = [
                    choice
                    for choice in choices
                    if isinstance(choice, b2b_gemm.TritonTemplateCaller)
                ]
                self.assertTrue(choices, "expected a fused B2B GEMM template")
                choices = choices[:1]
            return select_algorithm(name, choices, *args, **kwargs)

        counters.clear()
        with patch.object(b2b_gemm, "autotune_select_algorithm", select_fused):
            actual, codes = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), *inputs
            )
        code = "\n".join(codes)
        assoc = "LEFT" if is_left_assoc else "RIGHT"
        self.assertIn(f"B2B_GEMM_{assoc}_TRITON_ENTRANCE", code)
        self.assertGreater(counters["inductor"]["b2b_gemm"], 0)
        return actual, code

    def _assert_input_precision(self, code, inner, outer):
        for name, value in (
            ("INNER_MM_PRECISION", inner),
            ("OUTER_MM_PRECISION", outer),
        ):
            self.assertIn(f"input_precision={name}", code)
            value_pattern = "None" if value is None else f"['\"]{value}['\"]"
            self.assertRegex(
                code, rf"{name}\s*:\s*tl.constexpr\s*=\s*{value_pattern}(?:\s|$)"
            )

    def _check_fp32_accuracy(self, device, is_left_assoc, precision):
        def fn(a, b, c):
            return (a @ b) @ c if is_left_assoc else a @ (b @ c)

        torch.set_float32_matmul_precision("highest")
        inputs = self._make_inputs(device, is_left_assoc, torch.float32)
        expected = fn(*inputs)
        double_reference = fn(*(x.double() for x in inputs))
        if precision == "ieee":
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        else:
            torch.set_float32_matmul_precision(precision)
        actual, code = self._compile_fused(fn, inputs, is_left_assoc)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual, expected, atol=1e-3, rtol=1e-5)
        self.assertEqual(actual.double(), double_reference, atol=1e-3, rtol=1e-5)
        # Both logical GEMMs are below the MM policy's CUDA TF32 shape threshold.
        self._assert_input_precision(code, "ieee", "ieee")

    @recover_orig_fp32_precision
    @parametrize("is_left_assoc", (False, True))
    @parametrize("precision", ("highest", "ieee", "high", "medium"))
    def test_fp32_accuracy(self, device, is_left_assoc, precision):
        self._check_fp32_accuracy(device, is_left_assoc, precision)

    @recover_orig_fp32_precision
    @torch._inductor.config.patch(compile_threads=1)
    @parametrize("is_left_assoc", (False, True))
    @patch.dict(os.environ, {"TRITON_F32_DEFAULT": "tf32"})
    def test_fp32_highest_triton_override(self, device, is_left_assoc):
        from triton import knobs

        self.assertEqual(knobs.language.fp32_default, "tf32")
        self._check_fp32_accuracy(device, is_left_assoc, "highest")

    @recover_orig_fp32_precision
    @parametrize("precision", ("highest", "high"))
    def test_tf32_outer_mm_precision(self, device, precision):
        def fn(a, b, c):
            return a @ (b @ c)

        torch.manual_seed(314159)
        torch.set_float32_matmul_precision(precision)
        inputs = [
            torch.randn(shape, device=device, dtype=torch.float32) / 4
            for shape in ((16, 512), (512, 16), (16, 512))
        ]
        expected = fn(*inputs)
        actual, code = self._compile_fused(fn, inputs, is_left_assoc=False)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        tol = 1e-2 if precision == "high" else 1e-4
        self.assertEqual(actual, expected, atol=tol, rtol=tol)
        outer_precision = None if precision == "high" else "ieee"
        self._assert_input_precision(code, "ieee", outer_precision)

    @recover_orig_fp32_precision
    @parametrize("is_left_assoc", (False, True))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_low_precision_output(self, device, is_left_assoc, dtype):
        def fn(a, b, c):
            return (a @ b) @ c if is_left_assoc else a @ (b @ c)

        torch.set_float32_matmul_precision("highest")
        inputs = self._make_inputs(device, is_left_assoc, dtype)
        # Preserve the existing promoted-intermediate contract.
        expected = fn(*(x.float() for x in inputs)).to(dtype)
        actual, code = self._compile_fused(fn, inputs, is_left_assoc)
        self.assertEqual(actual.dtype, dtype)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(actual, expected, atol=0.1, rtol=0.01)
        self._assert_input_precision(code, None, None)

    @recover_orig_fp32_precision
    @parametrize("is_left_assoc", (False, True))
    @parametrize(
        "dtype,value",
        (
            (torch.float32, 2.0),
            (torch.bfloat16, 2.0),
            (torch.float16, 1.0),
            (torch.float16, 2.0),
        ),
    )
    def test_output_range(self, device, is_left_assoc, dtype, value):
        def fn(a, b, c):
            return (a @ b) @ c if is_left_assoc else a @ (b @ c)

        torch.set_float32_matmul_precision("highest")
        inputs = self._make_inputs(device, is_left_assoc, dtype, value)
        expected = fn(*inputs)
        actual, _ = self._compile_fused(fn, inputs, is_left_assoc)
        self.assertEqual(actual.dtype, dtype)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        if dtype == torch.float16 and value == 2.0:
            self.assertTrue(torch.isinf(actual).all())
        else:
            self.assertTrue(torch.isfinite(actual).all())
        if dtype != torch.float16:
            self.assertGreater(
                actual.abs().max().item(), torch.finfo(torch.float16).max
            )

    @recover_orig_fp32_precision
    @parametrize("is_left_assoc", (False, True))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_mixed_dtype_output(self, device, is_left_assoc, dtype):
        def fn(a, b, c):
            return (a @ b).float() @ c if is_left_assoc else a @ (b @ c).float()

        torch.set_float32_matmul_precision("highest")
        inputs = self._make_inputs(device, is_left_assoc, dtype)
        outer_input = 2 if is_left_assoc else 0
        inputs[outer_input] = inputs[outer_input].float()
        expected = fn(*(x.float() for x in inputs))
        actual, code = self._compile_fused(fn, inputs, is_left_assoc)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual, expected, atol=1e-3, rtol=1e-5)
        self._assert_input_precision(code, None, "ieee")


instantiate_device_type_tests(B2BGEMMPrecisionTest, globals(), only_for="cuda")


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
