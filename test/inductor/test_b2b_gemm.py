# Owner(s): ["module: inductor"]
import os
import unittest
from unittest.mock import Mock, patch

import torch
from torch._dynamo.source import ConstantSource
from torch._dynamo.utils import counters
from torch._inductor.fx_passes import b2b_gemm
from torch._inductor.ir import FixedLayout, InputBuffer, TensorBox
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.select_algorithm import TritonTemplateCaller
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
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

    def select_fused(self, name, choices, *args, **kwargs):
        self.assertEqual(name, "b2b_gemm")
        choice = next(c for c in choices if isinstance(c, TritonTemplateCaller))
        return choice.output_node(), None

    def check_tf32_policy(self, left_assoc, device_type, shape, sizevars, expected):
        m, n, o, p = shape
        tensors = [
            TensorBox.create(
                InputBuffer(
                    name=name,
                    layout=FixedLayout(torch.device(device_type), torch.float32, dims),
                )
            )
            for name, dims in (("a", [m, n]), ("b", [n, o]), ("c", [o, p]))
        ]
        template = (
            b2b_gemm.b2b_gemm_left_template
            if left_assoc
            else b2b_gemm.b2b_gemm_right_template
        )
        with (
            V.set_graph_handler(Mock(sizevars=sizevars)),
            patch.object(b2b_gemm, "build_subgraph_buffer", return_value=None),
            patch.object(b2b_gemm, "_make_unoptimized_b2b_gemm_choice"),
            patch.object(
                b2b_gemm, "autotune_select_algorithm", return_value=(None, None)
            ),
            patch.object(template, "maybe_append_choice") as append,
        ):
            b2b_gemm.tuned_b2b_gemm(left_assoc, None, *tensors)
        self.assertEqual(append.call_count, len(b2b_gemm.b2b_gemm_configs))
        for call in append.call_args_list:
            flags = call.kwargs
            self.assertEqual(
                (flags["ALLOW_TF32_INNER"], flags["ALLOW_TF32_OUTER"]), expected
            )

    @parametrize("left_assoc", (False, True))
    @parametrize(
        "device_type,cuda_tf32,xpu_tf32,shape,expected_left,expected_right",
        (
            ("xpu", False, True, (256, 32, 256, 32), (True, True), (True, True)),
            ("xpu", True, False, (256, 32, 256, 32), (False, False), (False, False)),
            ("cuda", False, True, (16, 512, 512, 512), (False, False), (False, False)),
            ("cuda", True, False, (16, 512, 512, 32), (True, False), (False, False)),
            ("cuda", True, False, (16, 32, 512, 512), (False, True), (True, False)),
            ("cuda", True, False, (16, 512, 32, 512), (False, False), (False, True)),
            ("cuda", True, False, (16, 512, 512, 512), (True, True), (True, True)),
            ("cuda", True, False, (15, 512, 512, 512), (False, False), (True, False)),
        ),
    )
    @recover_orig_fp32_precision
    def test_b2b_gemm_tf32_policy(
        self,
        left_assoc,
        device_type,
        cuda_tf32,
        xpu_tf32,
        shape,
        expected_left,
        expected_right,
    ):
        torch.backends.cuda.matmul.fp32_precision = "tf32" if cuda_tf32 else "ieee"
        expected = expected_left if left_assoc else expected_right
        with patch(
            "torch._dynamo.device_interface.XpuInterface.allow_tf32",
            return_value=xpu_tf32,
        ) as xpu_flag:
            self.check_tf32_policy(
                left_assoc, device_type, shape, SizeVarAllocator(), expected
            )
        if device_type == "xpu":
            self.assertTrue(xpu_flag.called)
        else:
            xpu_flag.assert_not_called()

    @parametrize("left_assoc", (False, True))
    @recover_orig_fp32_precision
    def test_b2b_gemm_tf32_policy_symbolic(self, left_assoc):
        shape_env = ShapeEnv()
        m = shape_env.create_symbol(
            16, source=ConstantSource("dimension"), dynamic_dim=DimDynamic.DYNAMIC
        )
        sizevars = SizeVarAllocator(shape_env)
        self.assertEqual(sizevars.optimization_hint(m), 16)
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        expected = (False, False) if left_assoc else (True, False)
        self.check_tf32_policy(
            left_assoc, "cuda", (m, 512, 512, 512), sizevars, expected
        )
        self.assertEqual(shape_env.guards, [])

    @parametrize("left_assoc", (False, True))
    @parametrize(
        "inner_dtype,outer_dtype,allow_tf32",
        (
            (torch.float32, torch.float32, False),
            (torch.float32, torch.float32, True),
            (torch.float16, torch.float16, False),
            (torch.bfloat16, torch.bfloat16, False),
            (torch.float16, torch.float32, False),
            (torch.float32, torch.float16, False),
            (torch.bfloat16, torch.float32, False),
            (torch.float32, torch.bfloat16, False),
        ),
    )
    @recover_orig_fp32_precision
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_precision(
        self, device, left_assoc, inner_dtype, outer_dtype, allow_tf32
    ):
        if allow_tf32 and not torch.cuda.is_tf32_supported():
            self.skipTest("TF32 is not supported")

        def fn(a, b, c):
            if left_assoc:
                return (a @ b).to(c.dtype) @ c
            return a @ (b @ c).to(a.dtype)

        torch.manual_seed(314159)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        m, n, o, p = (256, 32, 256, 32) if left_assoc else (32, 256, 32, 256)
        a_dtype, c_dtype = (
            (inner_dtype, outer_dtype) if left_assoc else (outer_dtype, inner_dtype)
        )
        a = torch.randn(m, n, device=device, dtype=a_dtype)
        b = torch.randn(n, o, device=device, dtype=inner_dtype)
        c = torch.randn(o, p, device=device, dtype=c_dtype)
        if allow_tf32:
            a.fill_(2)
            b.fill_(2)
            c.fill_(2)
        expected = fn(a.double(), b.double(), c.double()).to(outer_dtype)

        with patch(
            "torch._inductor.fx_passes.b2b_gemm.autotune_select_algorithm",
            self.select_fused,
        ):
            actual, codes = run_and_get_code(torch.compile(fn, fullgraph=True), a, b, c)
        assoc = "left" if left_assoc else "right"
        code = "\n".join(codes)
        self.assertIn(f"B2B_GEMM_{assoc.upper()}_TRITON_ENTRANCE", code)
        for role, dtype in (("INNER", inner_dtype), ("OUTER", outer_dtype)):
            tf32 = dtype != torch.float32
            self.assertIn(f"ALLOW_TF32_{role} : tl.constexpr = {tf32}", code)
        precise = outer_dtype == torch.float32
        atol, rtol = (1e-3, 1e-5) if precise else (0.2, 0.01)
        self.assertEqual(actual, expected, atol=atol, rtol=rtol, exact_dtype=True)

    @parametrize("left_assoc", (False, True))
    @parametrize("shape", ((32, 32), (64, 64), (128, 128), (256, 32)))
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_dynamic_rows(self, device, left_assoc, shape):
        def fn(a, b, c):
            return (a @ b) @ c if left_assoc else a @ (b @ c)

        torch.manual_seed(0)
        m, n = shape
        m, n, o, p = (m, n, m, n) if left_assoc else (n, m, n, m)
        a = torch.randn(m, n, device=device, dtype=torch.float16)
        b = torch.randn(n, o, device=device, dtype=torch.float16)
        c = torch.randn(o, p, device=device, dtype=torch.float16)
        torch._dynamo.mark_dynamic(a, 0, min=2, max=512)
        compiled = torch.compile(fn, fullgraph=True)
        with patch(
            "torch._inductor.fx_passes.b2b_gemm.autotune_select_algorithm",
            self.select_fused,
        ):
            actual, codes = run_and_get_code(compiled, a, b, c)
        if m != n:
            assoc = "LEFT" if left_assoc else "RIGHT"
            self.assertIn(f"B2B_GEMM_{assoc}_TRITON_ENTRANCE", "\n".join(codes))
        expected = fn(a.float(), b.float(), c.float()).half()
        self.assertEqual(actual, expected, atol=0.1, rtol=0.01)
        a = torch.randn(17, n, device=device, dtype=torch.float16)
        with torch._dynamo.config.patch(error_on_recompile=True):
            expected = fn(a.float(), b.float(), c.float()).half()
            self.assertEqual(compiled(a, b, c), expected, atol=0.1, rtol=0.01)


instantiate_device_type_tests(B2BGEMMTest, globals(), only_for=GPU_TYPE, allow_xpu=True)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
