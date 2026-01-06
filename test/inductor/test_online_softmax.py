# Owner(s): ["module: inductor"]

import math
import os

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch._dynamo.utils import rmse, same
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, HAS_TRITON


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"
USE_LARGE_INPUT = os.environ.get("USE_LARGE_INPUT") == "1" or DO_PERF_TEST


def _prepare_softmax(x, dim):
    xmax = x.amax(dim=dim, keepdim=True)
    xsum = (x - xmax).exp().sum(dim=dim, keepdim=True)
    return xmax, xsum


class TestOnlineSoftmax(TestCase):
    def do_test_acc_and_perf(self, op):
        if DO_PERF_TEST:
            N = 32 * 1024
            V = 50304  # padded version for gpt2
        else:
            N, V = 1024, 2048  # small value to avoid OOM in CI

        def f(x):
            return op(x, dim=-1)

        x = torch.randn(N, V, dtype=torch.bfloat16, device=GPU_TYPE)
        opt_f = torch.compile(f)
        expected = f(x)
        actual = opt_f(x)

        self.assertTrue(same(expected, actual, tol=1e-2))

        if DO_PERF_TEST:
            from triton.testing import do_bench

            eager_ms = do_bench(lambda: f(x))
            opt_ms = do_bench(lambda: opt_f(x))
            print(f"{eager_ms=}")
            print(f"{opt_ms=}")

    def test_softmax(self):
        self.do_test_acc_and_perf(torch.softmax)

    def test_log_softmax(self):
        self.do_test_acc_and_perf(torch.log_softmax)

    @inductor_config.patch(use_fast_math=True)
    def test_prepare_softmax_perf(self):
        self.do_test_acc_and_perf(_prepare_softmax)

    def get_softmax_wrapper(self, V=50304, use_log_softmax=False, device=GPU_TYPE):
        N = 32 * 1024

        @torch.compile
        def f(x):
            if use_log_softmax:
                return torch.log_softmax(x, dim=-1)
            else:
                return torch.softmax(x, dim=-1)

        x = torch.randn(N, V, dtype=torch.bfloat16, device=device)
        out, source_codes = run_and_get_code(f, x)
        return source_codes[0]

    def test_codegen_3pass_softmax_due_to_disable(self):
        with inductor_config.patch(online_softmax=False):
            wrapper_code = self.get_softmax_wrapper()

        self.assertEqual(wrapper_code.count("for r0_offset in"), 3)

    @parametrize("V", [2048, 50304])
    @parametrize("use_log_softmax", [False, True])
    def test_codegen_online_softmax(self, use_log_softmax, V):
        wrapper_code = self.get_softmax_wrapper(use_log_softmax=use_log_softmax, V=V)

        self.assertEqual(wrapper_code.count("for r0_offset in"), 2)

    def test_no_online_softmax_for_cpu(self):
        code = self.get_softmax_wrapper(V=2048, device="cpu")

        # CPU need an explicit loop across different rows.
        # For GPU, this is parallelized by the hardware.
        self.assertEqual(code.count("for(int64_t"), 4)

    def test_codegen_softmax_persistent_reduction(self):
        """
        Persistent reduction has no for loops.
        """
        wrapper_code = self.get_softmax_wrapper(1024)
        self.assertEqual(wrapper_code.count("for r0_offset in"), 0)

    @inductor_config.patch("triton.persistent_reductions", False)
    def test_sdpa(self):
        """
        Make sure online softmax here does not conflict with the sdpa
        patterns.
        """
        q, k, v = (
            torch.randn((4, 2, 16, 32), device=GPU_TYPE, dtype=torch.bfloat16)
            for _ in range(3)
        )

        def f(q, k, v):
            return (
                torch.matmul(q, k.transpose(-2, -1))
                .div(math.sqrt(k.shape[-1]))
                .softmax(dim=-1)
                .matmul(v)
            )

        opt_f = torch.compile(f)
        ref = f(q, k, v)
        act, (code,) = run_and_get_code(opt_f, q, k, v)
        self.assertTrue(torch.allclose(ref, act, atol=1e-2, rtol=1e-2))
        self.assertTrue("aten._scaled_dot_product_" in code)

    @parametrize("nrow", [2, 2048])
    @parametrize("dim", [-1, 0, 1])
    def test_prepare_softmax(self, dim, nrow):
        x = torch.randn(nrow, 2048, dtype=torch.bfloat16, device=GPU_TYPE)
        act, (code,) = run_and_get_code(torch.compile(_prepare_softmax), x, dim)
        ref = _prepare_softmax(x, dim)
        self.assertTrue(same(ref, act, tol=1e-2))

        if nrow == 2048 and dim == 0:
            num_kernels = 2
            # Note: split reduction is not triggered for this shape on some xpu devices.
            #       check "num_splits" for more details
            if GPU_TYPE == "xpu":
                num_kernels = 1

            # split reduction is triggered. We have multiple kernels
            self.assertTrue(code.count("def triton") >= num_kernels)
        else:
            if nrow == 2 and dim == 0:
                # persistent reduction triggered
                expected_num_loop = 0
            else:
                # A single loop due to online softmax
                expected_num_loop = 1
            self.assertEqual(code.count("for r0_offset in"), expected_num_loop)

    def test_split_reduction(self):
        """
        We don't split online_softmax_reduce for now. Check
        'Split online_softmax_reduce' note in the code.

        When a split is promsing, we fallback for now.

        This is just a manual example rather than something we
        see in practice.
        """
        # tensor shape to trigger split reduction
        x = torch.randn(1, 2**20, dtype=torch.bfloat16, device=GPU_TYPE)
        ref = torch.softmax(x, dim=-1)
        act, (code,) = run_and_get_code(torch.compile(torch.softmax), x, dim=-1)
        self.assertTrue(torch.allclose(ref, act, atol=1e-3, rtol=1e-3))
        self.assertTrue(code.count("def triton") >= 2)
        self.assertTrue("online_softmax_reduce" not in code)

    @parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
    def test_prepare_softmax_acc_with_fp64(self, dtype):
        if USE_LARGE_INPUT:
            M, N = 32768, 50257
        else:
            M, N = 1024, 2048

        x = torch.randn(M, N, device=GPU_TYPE, dtype=dtype)

        ref_fp64 = _prepare_softmax(x.to(dtype=torch.float64), dim=-1)
        ref = _prepare_softmax(x, dim=-1)
        res, (code,) = run_and_get_code(torch.compile(_prepare_softmax), x, dim=-1)
        self.assertTrue("online_softmax_reduce" in code)

        # Max should be exactly equal
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[0].to(dtype=torch.float64), ref_fp64[0])

        ref_error = rmse(ref_fp64[1], ref[1]).item()
        res_error = rmse(ref_fp64[1], res[1]).item()

        # My local tests even shows a smaller res_error:
        #   ref_error=2.1065, res_error=2.1028
        # for bf16
        #   ref_error=0.2611, res_error=0.2609
        # for fp16
        #   ref_error=0.0001, res_error=0.0001
        # for fp32
        print(f"{ref_error=:.4f}, {res_error=:.4f}")

        self.assertTrue(
            res_error < ref_error + 0.1
        )  # Is this good enough to make CI stable

    @parametrize("fn", [torch.log_softmax, torch.softmax])
    @parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
    def test_softmax_acc_with_fp64(self, dtype, fn):
        if USE_LARGE_INPUT:
            M, N = 32768, 50257
        else:
            M, N = 1024, 2048

        x = torch.randn(M, N, device=GPU_TYPE, dtype=dtype)

        ref_fp64 = fn(x.to(dtype=torch.float64), dim=-1)
        ref = fn(x, dim=-1)
        res, (code,) = run_and_get_code(torch.compile(fn), x, dim=-1)
        self.assertTrue("online_softmax_reduce" in code)

        ref_error = rmse(ref_fp64, ref).item()
        res_error = rmse(ref_fp64, res).item()

        # For torch.softmax,
        # I get almost 0 for ref_error/res_error for all 3 dtypes. It's because
        # each value is very small since each row add up to 1.0
        #
        # For torch.log_softmax
        #   ref_error=0.0180399032, res_error=0.0180399031
        # for bf16
        #   ref_error=0.0022548872, res_error=0.0022548872
        # for fp16
        #   ref_error=0.0000003744, res_error=0.0000003748
        # for fp32
        print(f"{ref_error=:.10f}, {res_error=:.10f}")

        self.assertTrue(
            res_error < ref_error + 0.1
        )  # Is this good enough to make CI stable

    def test_softmin(self):
        """
        The rnumel==1 kind of reduction should be unrolled.
        """

        def f(x):
            return F.softmin(x, dim=0)

        x = torch.randn(1, device=GPU_TYPE)
        ref = f(x)
        act, (code,) = run_and_get_code(torch.compile(f), x)
        self.assertTrue(torch.allclose(ref, act))
        self.assertTrue("online_softmax_reduce" not in code)

    def test_causal_mask(self):
        def f(x):
            return x.softmax(dim=-1)

        x = torch.randn(2048, 2048, device=GPU_TYPE)
        mask = torch.tril(torch.ones(2048, 2048, device=GPU_TYPE))
        x.masked_fill_(mask == 0, float("-inf"))

        ref = f(x)
        act = torch.compile(f)(x)
        self.assertTrue(not ref.isnan().any())
        self.assertTrue(not act.isnan().any())
        self.assertTrue(torch.allclose(ref, act))

    def test_tb_speech_transformer_attn(self):
        """
        This is an example extracted from speech_transformer.
        Since online softmax use the max from partial elements of an entire
        row, if the input contains '-inf', it's possible that the
        max of those partial elements is '-inf' even if the entire row
        has non '-inf' value. In this cause, online softmax will need
        do things like 'float(-inf) - float(-inf)' which becomes 'nan'.
        We fixed this by interpreting 'float(-inf) - float(-inf)' as 0
        if we found both operands are 'float(-inf)'.
        """
        torch.manual_seed(1337)

        def f(x, mask):
            x = torch.where(mask, float("-inf"), x)
            xmax = x.amax(dim=-1, keepdim=True)
            xsum = (x - xmax).exp().sum(dim=-1, keepdim=True)
            return xsum

        x = torch.randn(8, 10, 22, 204, device=GPU_TYPE)
        mask = torch.randint(0, 2, (10, 204), device=GPU_TYPE) == 0
        mask = mask.view(1, 10, 1, 204)

        ref = f(x, mask)
        act = torch.compile(f)(x, mask)
        self.assertTrue(not ref.isnan().any())
        self.assertTrue(not act.isnan().any())
        self.assertTrue(torch.allclose(ref, act))

    @inductor_config.patch(split_reductions=False)
    def test_3d_tiled_online_softmax(self):
        def f(x, y):
            return (x * y).softmax(dim=-1)

        M, N, K = 32, 8, 1024

        x = torch.randn(K, N, M, device=GPU_TYPE).permute(2, 1, 0)
        y = torch.randn(K, M, N, device=GPU_TYPE).permute(1, 2, 0)

        opt_f = torch.compile(f)
        torch.testing.assert_close(f(x, y), opt_f(x, y), atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestOnlineSoftmax)

if __name__ == "__main__":
    if IS_LINUX and HAS_GPU and HAS_TRITON:
        run_tests()
