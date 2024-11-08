import torch
from torch._inductor import config, metrics
from torch._inductor.test_case import TestCase
from torch._dynamo.utils import same
from torch import nn
import os
from triton.testing import do_bench

# TODO: always test with large input. Skip the test if the GPU
# does not have enough memory
USE_LARGE_INPUT = os.environ.get("USE_LARGE_INPUT", "1") == "1"
DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"

@config.patch("AutoChunker.enable", True)
class AutoChunkerTest(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()

    def common_matmul_test(self, has_softmax, use_bias=False):
        M, K, N = 1024, 16, 1024

        if USE_LARGE_INPUT:
            M = 1024 * 32
            K = 32
            N = 1024 * 32

        dtype = torch.float32
        _input = torch.randn(
            M, K, dtype=dtype, requires_grad=True, device="cuda"
        )
        weight = torch.randn(
            K, N, dtype=dtype, requires_grad=True, device="cuda"
        )
        bias = torch.randn(N, dtype=dtype, requires_grad=True, device="cuda")

        def f(_input, weight, bias):
            out = (_input * 2) @ weight
            if use_bias:
                out = out + bias
            if has_softmax:
                out = out.softmax(dim=-1)
            _sum = out.sum()
            _sum.backward()
            return _sum

        expect = (f(_input, weight, bias), _input.grad, weight.grad, bias.grad if use_bias else None)

        _input.grad = None
        weight.grad =None
        bias.grad = None

        torch.cuda.reset_peak_memory_stats()
        opt_f = torch.compile(f)
        actual = (opt_f(_input, weight, bias), _input.grad, weight.grad, bias.grad if use_bias else None)
        peak_memory = torch.cuda.max_memory_allocated()

        print(f"Peak memory {peak_memory / 10 ** 9 :.6f} GB")

        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}")

        # Make sure AutoChunking is not skipped
        self.assertEqual(metrics.num_auto_chunking, 1)

        # Only assert peak memory saving for large input. For small input, the saving can
        # be largely distorted by other memory allocation such as the tensor used to clear L2
        # cache by triton perf benchmarking API.
        if USE_LARGE_INPUT:
            expected_bound = M * N * dtype.itemsize
            self.assertTrue(peak_memory < expected_bound, f"Actual peak_memory {peak_memory}, expected bound {expected_bound}")


    def test_matmul_trivial(self):
        self.common_matmul_test(has_softmax=False)

    def test_linear_trivial(self):
        self.common_matmul_test(has_softmax=False, use_bias=True)
   
    # Due to not able to generate an inplace version of a softmax like
    # kernel, having 2 chunks does not have large enough savings.
    # Use at least 4 chunks here.
    @config.patch("AutoChunker.num_chunk", 4)
    def test_matmul_softmax(self):
        self.common_matmul_test(has_softmax=True)

    @config.patch("AutoChunker.num_chunk", 4)
    def test_linear_softmax(self):
        self.common_matmul_test(has_softmax=True, use_bias=True)

    @config.patch("AutoChunker.num_chunk", config.AutoChunker.num_chunk or 16)
    def test_fused_linear_cel(self):
        B = 32
        T = 1024
        C = 768
        V = 50257

        dtype = torch.bfloat16
        
        class LinearAndCEL(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(C, V)
                self.ce = nn.CrossEntropyLoss()
        
            def forward(self, x, y):
                return self.ce(self.linear(x).view(B * T, V), y.view(-1))
        
        mod = LinearAndCEL().cuda().to(dtype)
        
        def f(x, y):
            x.grad = None
            mod.linear.weight.grad = None
            mod.linear.bias.grad = None

            x = x * 2
            loss = mod(x, y)
            loss.backward()
            return loss

        opt_f = torch.compile(f)
        
        x = torch.randn(B, T, C, dtype=dtype, requires_grad=True).cuda()
        x.retain_grad()
        y = torch.randint(0, V, (B, T)).cuda()

        expect = (f(x, y), x.grad, mod.linear.weight.grad, mod.linear.bias.grad)
        torch.cuda.reset_peak_memory_stats()
        actual = (opt_f(x, y), x.grad, mod.linear.weight.grad, mod.linear.bias.grad)
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"Peak memory {peak_memory / 10 ** 9 :.6f} GB")

        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}")

        if DO_PERF_TEST:
            ms_eager = do_bench(lambda: f(x, y))
            ms_opt = do_bench(lambda: opt_f(x, y))
            print(f"Eager v.s. Compile perf: {ms_eager:.3f}ms v.s. {ms_opt:.3f}ms")

        self.assertEqual(metrics.num_auto_chunking, 1)
        expected_bound = B * T * V * x.dtype.itemsize
        self.assertTrue(peak_memory < expected_bound, f"Actual peak_memory {peak_memory}, expected bound {expected_bound}")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
