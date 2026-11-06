# Owner(s): ["module: inductor"]


import unittest
from collections.abc import Callable
from unittest import mock

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils._triton import (
    has_triton_tensor_descriptor_host_tma,
    has_triton_tma_device,
)


aten = torch.ops.aten


@inductor_config.patch(
    {
        "triton.native_matmul": True,
        "test_configs.force_filter_reduction_configs": True,
        "coordinate_descent_tuning": False,
        "triton.autotune_pointwise": False,
    }
)
class TestNativeMatmul(TestCase):
    def _check_equal(
        self, f: Callable, example_inputs: tuple[torch.Tensor], tol: float = 1e-4
    ):
        compiled = torch.compile(f)
        actual = compiled(*example_inputs)
        expect = f(*example_inputs)
        self.assertTrue(same(expect, actual, tol=tol))

    def _check_code(
        self,
        f: Callable,
        example_inputs: tuple[torch.Tensor],
        kernel_count: int,
        dot_count: int,
        where_count: int | None = None,
    ):
        f = torch.compile(f)
        code = run_and_get_triton_code(f, *example_inputs)
        FileCheck().check_regex(r"triton.*mm.*\.run\(").run(code)

        FileCheck().check_count("@triton.jit", kernel_count, exactly=True).check_count(
            "tl.dot", dot_count, exactly=True
        ).run(code)
        if where_count is not None:
            self.assertEqual(code.count("tl.where("), where_count)

    def test_matmul(self):
        def f(x, y):
            z = x @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1, where_count=0)

    def test_mm_1d_expand(self):
        def f(x, y, M, K):
            z = x[:, None].expand(M, K) @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M,), (1,), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, M, K))
        self._check_code(f, (x, y, M, K), 1, 1, where_count=0)

    def test_mm_2_expand(self):
        def f(x, y, M, K):
            z = x[:, None].expand(M, K) @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((1,), (0,), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, M, K))
        self._check_code(f, (x, y, M, K), 1, 1, where_count=0)

    def test_matmul_fp16(self):
        def f(x, y):
            z = x @ y.to(x.dtype)
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), dtype=torch.float16, device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), dtype=torch.float32, device=GPU_TYPE)

        # _check_equal calls torch._dynamo.utils.same with kwarg tol=1e-4.
        # For fp16 dtype, torch.allclose() defaults to atol=1e-3 rtol=1e-5,
        # but same() uses the single value to assign both, resulting in
        # Accuracy failed: allclose not within tol=0.0001.
        self._check_equal(f, (x, y), tol=1e-3)
        self._check_code(f, (x, y), 1, 1, where_count=0)

    def test_reduction_mask_zeroout(self):
        def f(x, y):
            return (x + 1) @ (y - 2)

        M, K, N = 62, 62, 62
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1, where_count=2)

    def test_3mm_add(self):
        def f(x, y, z, w, r, t):
            return x @ y + z @ w + r @ t

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)
        w = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        z = rand_strided((K, N), (N, 1), device=GPU_TYPE)
        r = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        t = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, z, w, r, t))
        self._check_code(f, (x, y, z, w, r, t), 1, 3, where_count=0)

    def test_mm_complex(self):
        def f(x, y, z, w):
            return x[z] @ y + w + 3

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        z = torch.randint(M, (M, K), dtype=torch.long, device=GPU_TYPE)
        w = rand_strided((M, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, z, w))
        self._check_code(f, (x, y, z, w), 1, 1, where_count=1)

    def test_batchmatmul(self):
        def f(x, y):
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 256, 128, 128, 128
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1, where_count=0)

    def test_bmm_vertical_fusion(self):
        def f(x, y):
            z = torch.bmm(x, y)
            w = torch.nn.functional.relu(z)
            v = w + z * z
            return v

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    def test_bmm_horizontal_fusion(self):
        def f(x, y, z, w):
            bmm1 = torch.bmm(x, y)
            bmm2 = torch.bmm(z, w)
            return bmm1 - bmm2 + bmm1 * bmm2

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)
        z = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        w = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, z, w))
        self._check_code(f, (x, y, z, w), 1, 2, where_count=0)

    def test_bmm_fusion_complex1(self):
        # Out[O[b,i],j] += Input[I[b,i],r] * Weight[W[b],r,j] * Val[b,i]
        def f(inp, weight, val, I_idx, W_idx, O_idx, out):
            x = inp[I_idx]  # [B,I,R]
            y = weight[W_idx]  # [B,R,J]
            t = torch.einsum("bir,brj,bi->bij", x, y, val)
            out.index_add_(0, O_idx.reshape(-1), t.reshape(-1, J))
            return out

        B, I, R, J = 32, 16, 128, 16
        N_in = 128
        N_w = 64
        N_out = 128

        inp = rand_strided((N_in, R), (R, 1), device=GPU_TYPE)
        weight = rand_strided((N_w, R, J), (R * J, J, 1), device=GPU_TYPE)
        val = rand_strided((B, I), (I, 1), device=GPU_TYPE)

        I_idx = torch.randint(0, N_in, (B, I), device=GPU_TYPE, dtype=torch.int64)
        W_idx = torch.randint(0, N_w, (B,), device=GPU_TYPE, dtype=torch.int64)
        O_idx = torch.randint(0, N_out, (B, I), device=GPU_TYPE, dtype=torch.int64)

        out = torch.zeros((N_out, J), device=GPU_TYPE)

        self._check_equal(f, (inp, weight, val, I_idx, W_idx, O_idx, out))
        self._check_code(
            f, (inp, weight, val, I_idx, W_idx, O_idx, out), 1, 1, where_count=3
        )

    def test_bmm_fusion_complex2(self):
        # out[Ai[g],m,n] += Av[g,m,k,p] * B[Ak[g,p],k,n]
        def f(Av, B, Ai, Ak, out):
            Bg = B[Ak]  # [G,P,K,N]
            Cg = torch.einsum("gmkp,gpkn->gmn", Av, Bg)  # [G,M,N]
            out.index_add_(0, Ai, Cg)
            return out

        G, M, K, P, N = 128, 16, 64, 8, 16
        NB = 128
        NA = 128

        Av = rand_strided((G, M, K, P), (M * K * P, K * P, P, 1), device=GPU_TYPE)
        B = rand_strided((NB, K, N), (K * N, N, 1), device=GPU_TYPE)

        Ai = torch.randint(0, NA, (G,), device=GPU_TYPE, dtype=torch.int64)
        Ak = torch.randint(0, NB, (G, P), device=GPU_TYPE, dtype=torch.int64)

        out = torch.zeros((NA, M, N), device=GPU_TYPE)

        self._check_equal(f, (Av, B, Ai, Ak, out))
        self._check_code(f, (Av, B, Ai, Ak, out), 1, 1, where_count=2)

    def test_bmm_large_batch_reversed_pid(self):
        def f(x, y):
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 65537, 16, 16, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        f = torch.compile(f)
        code = run_and_get_triton_code(f, x, y)

        FileCheck().check("zoffset = tl.program_id(0)").check(
            "pid_mn = tl.program_id(2)"
        ).check("tl.program_id(1)").check(
            "pid_y, pid_x = tl.swizzle2d(pid_mn //"
        ).check("yoffset = pid_y * YBLOCK").check("xoffset = pid_x * XBLOCK").run(code)
        self.assertNotIn("tl.where(", code)

    def test_bmm_no_z_broadcast(self):
        def f(x, y):
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        f = torch.compile(f)
        code = run_and_get_triton_code(f, x, y)

        FileCheck().check_not("tl.arange(0, ZBLOCK)[:, None, None, None]").check(
            "tl.arange(0, ZBLOCK)"
        ).check("tl.arange(0, YBLOCK)[None, :, None, None]").check(
            "tl.arange(0, XBLOCK)[None, None, :, None]"
        ).check("tl.arange(0, R0_BLOCK)[None, None, None, :]").run(code)
        self.assertNotIn("tl.where(", code)


@unittest.skipUnless(
    HAS_GPU and has_triton_tma_device(), "Tensor descriptors are not supported"
)
@inductor_config.patch(
    {
        "triton.use_tensor_descriptor": True,
        "triton.enable_host_side_tma": False,
        "assume_aligned_inputs": True,
    }
)
class TestNativeMatmulDeviceTMA(TestNativeMatmul):
    pass


@unittest.skipUnless(
    HAS_GPU and has_triton_tensor_descriptor_host_tma(),
    "Host tensor descriptors are not supported",
)
@inductor_config.patch(
    {
        "triton.use_tensor_descriptor": True,
        "triton.enable_host_side_tma": True,
        "assume_aligned_inputs": True,
    }
)
class TestNativeMatmulHostTMA(TestNativeMatmul):
    pass


class TestNativeMatmulCodegen(TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            inductor_config.patch(
                {
                    "test_configs.force_filter_reduction_configs": True,
                    "coordinate_descent_tuning": False,
                    "triton.autotune_pointwise": False,
                }
            )
        )

    @inductor_config.patch(
        {
            "triton.native_matmul": True,
            "triton.persistent_reductions": False,
            "assume_aligned_inputs": True,
            "force_disable_caches": True,
        }
    )
    @parametrize("backend", ["pointer", "device_tma", "host_tma"])
    @parametrize("batch_shape", [(), (3,)])
    @parametrize("transpose_a", [False, True])
    @parametrize("transpose_b", [False, True])
    @parametrize("shape", [(40, 72, 136), (128, 128, 128)])
    def test_matmul_load_singletons(
        self, device, backend, batch_shape, transpose_a, transpose_b, shape
    ):
        use_tma = backend in ("device_tma", "host_tma")
        if use_tma and not has_triton_tma_device():
            self.skipTest("Tensor descriptors are not supported")
        if backend == "host_tma" and not has_triton_tensor_descriptor_host_tma():
            self.skipTest("Host tensor descriptors are not supported")

        def f(a, b):
            return a @ b

        m, k, n = shape
        a = torch.randn((*batch_shape, m, k), device=device, dtype=torch.float16)
        b = torch.randn((*batch_shape, k, n), device=device, dtype=torch.float16)
        if transpose_a:
            a = a.mT.contiguous().mT
        if transpose_b:
            b = b.mT.contiguous().mT
        self.enterContext(
            inductor_config.patch(
                {
                    "triton.use_tensor_descriptor": use_tma,
                    "triton.enable_host_side_tma": backend == "host_tma",
                }
            )
        )
        compiled = torch.compile(f, fullgraph=True)
        code = run_and_get_triton_code(compiled, a, b)
        self.assertEqual(compiled(a, b), f(a, b), atol=1e-3, rtol=1e-3)
        self.assertEqual(code.count("@triton.jit"), 1)
        self.assertEqual(code.count("tl.dot("), 1)
        self.assertNotIn("tl.reshape(tmp", code)
        self.assertNotIn("[:, :, None], [", code)
        self.assertNotIn("[:, :, :, None], [", code)
        if backend == "pointer":
            self.assertIn("tl.reshape(tl.load(in_ptr0 + (tl.squeeze(", code)
            self.assertIn("tl.reshape(tl.load(in_ptr1 + (tl.squeeze(", code)
        else:
            self.assertEqual(code.count(".load(["), 2)

    def _check_masked_values(self, f, inputs, where_count, descriptor=True):
        if descriptor and not has_triton_tma_device():
            self.skipTest("Tensor descriptors are not supported")
        with inductor_config.patch(
            {
                "triton.native_matmul": True,
                "triton.persistent_reductions": False,
                "triton.use_tensor_descriptor": descriptor,
                "triton.enable_host_side_tma": False,
                "assume_aligned_inputs": True,
                "force_disable_caches": True,
            }
        ):
            compiled = torch.compile(f, fullgraph=True)
            code = run_and_get_triton_code(compiled, *inputs)
            self.assertEqual(compiled(*inputs), f(*inputs), atol=1e-3, rtol=1e-3)
        self.assertEqual(code.count("@triton.jit"), 1)
        self.assertEqual(code.count("tl.dot("), 1)
        self.assertEqual(code.count(".load(["), 2 if descriptor else 0)
        self.assertEqual(code.count("tl.where("), where_count)

    @parametrize("descriptor", [False, True])
    def test_matmul_masked_value_prologue(self, device, descriptor):
        def f(a, b):
            a = (a.float() + 1 - 1).half()
            b = (b.float() + 2 - 2).half()
            return a @ b

        inputs = (
            torch.randn((33, 72), device=device, dtype=torch.float16),
            torch.randn((72, 136), device=device, dtype=torch.float16),
        )
        self._check_masked_values(f, inputs, where_count=0, descriptor=descriptor)

    @parametrize("repeat_both", [False, True])
    def test_tma_repeat_masked_values(self, device, repeat_both):
        def f(a, b):
            if repeat_both:
                b = b.repeat(9, 1)
            return a.repeat(1, 9) @ b

        inputs = (
            torch.full(
                (33, 8),
                1.0 if repeat_both else float("inf"),
                device=device,
                dtype=torch.float16,
            ),
            torch.ones(
                (8 if repeat_both else 72, 136), device=device, dtype=torch.float16
            ),
        )
        self._check_masked_values(f, inputs, where_count=2 if repeat_both else 1)

    def test_tma_retained_quotient_masked_values(self, device):
        def f(a, b):
            return a[:, :, None].expand(33, 36, 2).reshape(33, 72) @ b

        a = torch.full((33, 40), float("inf"), device=device, dtype=torch.float16)
        a = a[:, :36]
        b = torch.ones((72, 136), device=device, dtype=torch.float16)
        self._check_masked_values(f, (a, b), where_count=0)

    @inductor_config.patch(
        {
            "triton.native_matmul": True,
            "triton.use_tensor_descriptor": True,
            "triton.enable_host_side_tma": True,
            "assume_aligned_inputs": True,
        }
    )
    @parametrize("batch", [None, 4])
    def test_matmul_32bit_indexing(self, device, batch):
        if not (has_triton_tma_device() and has_triton_tensor_descriptor_host_tma()):
            self.skipTest("Host tensor descriptors are not supported")

        def f(a, b):
            return a @ b

        m = n = 1536 if batch is None else 1024
        k = 1024
        batch_shape = () if batch is None else (batch,)
        a = torch.randn((*batch_shape, m, k), device=device, dtype=torch.bfloat16)
        b = torch.randn((*batch_shape, k, n), device=device, dtype=torch.bfloat16)
        self.assertGreater((batch or 1) * m * n * k, torch.iinfo(torch.int32).max)
        compiled = torch.compile(f, fullgraph=True)
        code = run_and_get_triton_code(compiled, a, b)
        self.assertEqual(compiled(a, b), f(a, b), atol=0.016, rtol=0.016)
        self.assertNotIn(".to(tl.int64)", code)
        FileCheck().check_count(".load([", 2, exactly=True).check("tl.dot").run(code)
        FileCheck().check_not("tl.where(").run(code)

    @inductor_config.patch(
        {
            "triton.native_matmul": True,
            "triton.use_tensor_descriptor": True,
            "triton.enable_host_side_tma": False,
            "assume_aligned_inputs": True,
        }
    )
    def test_mm_device_tma_grouped_pid(self, device):
        if not has_triton_tma_device():
            self.skipTest("Tensor descriptors are not supported")

        def f(a, b):
            return a @ b

        # Cover a partial group of eight Y tiles and a K tail.
        a = torch.randn((1025, 72), device=device, dtype=torch.float16)
        b = torch.randn((72, 136), device=device, dtype=torch.float16)
        compiled = torch.compile(f, fullgraph=True)
        code = run_and_get_triton_code(compiled, a, b)
        self.assertEqual(compiled(a, b), f(a, b), atol=1e-3, rtol=1e-3)
        FileCheck().check_count(".load([", 2, exactly=True).check("tl.dot").run(code)
        FileCheck().check_not("tl.where(").run(code)
        FileCheck().check("tl.swizzle2d").check(
            "tl.cdiv(ynumel, YBLOCK), tl.cdiv(xnumel, XBLOCK), GROUP_M)"
        ).run(code)

    @inductor_config.patch(
        {
            "triton.native_matmul": True,
            "triton.use_tensor_descriptor": True,
            "triton.enable_host_side_tma": True,
            "assume_aligned_inputs": True,
        }
    )
    def test_bmm_host_tma_grouped_pid(self, device):
        if not (has_triton_tma_device() and has_triton_tensor_descriptor_host_tma()):
            self.skipTest("Host tensor descriptors are not supported")

        def f(a, b):
            return torch.bmm(a, b)

        a = torch.randn((3, 1025, 72), device=device, dtype=torch.bfloat16)
        b = torch.randn((3, 136, 72), device=device, dtype=torch.bfloat16)
        b = b.transpose(-1, -2)
        compiled = torch.compile(f, fullgraph=True)
        code = run_and_get_triton_code(compiled, a, b)
        self.assertEqual(compiled(a, b), f(a, b), atol=0.016, rtol=0.016)
        FileCheck().check_count(".load([", 2, exactly=True).check("tl.dot").run(code)
        FileCheck().check_not("tl.where(").run(code)
        FileCheck().check("zoffset = tl.program_id(0)").check(
            "pid_mn = tl.program_id(2)"
        ).check("tl.program_id(1)").check("tl.swizzle2d(pid_mn //").run(code)

    @inductor_config.patch(
        {
            "triton.native_matmul": True,
            "triton.use_tensor_descriptor": False,
            "combo_kernels": True,
            "combo_kernels_autotune": 0,
            "benchmark_combo_kernel": False,
            "combo_kernel_per_subkernel_blocks": False,
            "combo_kernel_peak_memory_increase_gb": None,
            "combo_kernel_peak_memory_pct_threshold": None,
            "fx_graph_cache": False,
        }
    )
    def test_grouped_pid_combo(self, device):
        def f(a, b, c, d):
            return a @ b, c @ d

        # Round-robin dispatch pads both X and Y for the smaller matmul.
        inputs = [
            torch.randn(shape, device=device, dtype=torch.bfloat16)
            for shape in ((256, 64), (64, 96), (512, 64), (64, 128))
        ]
        compiled = torch.compile(f, fullgraph=True)
        code = run_and_get_triton_code(compiled, *inputs)
        self.assertEqual(compiled(*inputs), f(*inputs), atol=0.016, rtol=0.016)
        FileCheck().check_count("if pid_y <", 2, exactly=True).run(code)
        FileCheck().check_count("tl.swizzle2d", 2, exactly=True).run(code)
        FileCheck().check("pid_offset = pid // 2").run(code)
        self.assertNotIn("tl.where(", code)

    @inductor_config.patch(
        {
            "triton.native_matmul": True,
            "triton.use_tensor_descriptor": False,
            "fx_graph_cache": False,
        }
    )
    def test_grouped_pid_y_grid_overflow(self, device):
        def f(a, b):
            return a @ b

        a = torch.randn((256, 32), device=device, dtype=torch.bfloat16)
        b = torch.randn((32, 48), device=device, dtype=torch.bfloat16)
        # Split 16 Y tiles into a 6x3 launch grid, leaving two padded Y tiles.
        with (
            mock.patch("torch._inductor.codegen.triton.get_max_y_grid", return_value=7),
            mock.patch(
                "torch._inductor.runtime.triton_heuristics.get_max_y_grid",
                return_value=7,
            ),
            mock.patch(
                "torch._inductor.runtime.triton_heuristics.triton_native_mm_configs",
                [({"x": 16, "y": 16, "r": 16}, 4, 2)],
            ),
        ):
            compiled = torch.compile(f, fullgraph=True)
            code = run_and_get_triton_code(compiled, a, b)
            self.assertEqual(compiled(a, b), f(a, b), atol=0.016, rtol=0.016)
        FileCheck().check("tl.swizzle2d").check(
            "tl.num_programs(1) * tl.num_programs(2)"
        ).run(code)
        self.assertNotIn("tl.where(", code)


if HAS_GPU:
    instantiate_device_type_tests(TestNativeMatmulCodegen, globals(), only_for=GPU_TYPE)
    torch.set_default_device(GPU_TYPE)

if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
