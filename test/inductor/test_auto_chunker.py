# Owner(s): ["module: inductor"]
import os

import torch
from torch import nn
from torch._dynamo.utils import same
from torch._inductor import config, metrics
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CUDA_AND_TRITON


USE_LARGE_INPUT = os.environ.get("USE_LARGE_INPUT", "1") == "1"
DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"
DO_PROFILING = os.environ.get("DO_PROFILING") == "1"


class LinearAndCEL(nn.Module):
    def __init__(self, C, V):
        super().__init__()
        self.linear = nn.Linear(C, V)
        self.ce = nn.CrossEntropyLoss()
        self.V = V

    def forward(self, x, y):
        return self.ce(self.linear(x).view(-1, self.V), y.view(-1))


@config.patch("auto_chunker.enable", True)
@instantiate_parametrized_tests
class AutoChunkerTest(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()

    @largeTensorTest("8GB", device=GPU_TYPE, inductor=True)
    def common_matmul_test(self, has_softmax, use_bias=False, dynamic_shape=None):
        M, K, N = 1024, 16, 1024

        if USE_LARGE_INPUT:
            M = 1024 * 32
            K = 32
            N = 1024 * 32

        dtype = torch.float32
        _input = torch.randn(M, K, dtype=dtype, requires_grad=True, device=GPU_TYPE)
        weight = torch.randn(K, N, dtype=dtype, requires_grad=True, device=GPU_TYPE)
        bias = torch.randn(N, dtype=dtype, requires_grad=True, device=GPU_TYPE)

        def f(_input, weight, bias):
            out = (_input * 2) @ weight
            if use_bias:
                out = out + bias
            if has_softmax:
                out = out.softmax(dim=-1)
            _sum = out.sum()
            _sum.backward()
            return _sum

        expect = (
            f(_input, weight, bias),
            _input.grad,
            weight.grad,
            bias.grad if use_bias else None,
        )

        _input.grad = None
        weight.grad = None
        bias.grad = None

        torch.cuda.reset_peak_memory_stats()
        opt_f = torch.compile(f, dynamic=dynamic_shape)
        actual = (
            opt_f(_input, weight, bias),
            _input.grad,
            weight.grad,
            bias.grad if use_bias else None,
        )
        peak_memory = torch.cuda.max_memory_allocated()

        print(f"Peak memory {peak_memory / 10**9:.6f} GB")

        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}")

        # When the model is too trivial without softmax, no chunking happens because chunking
        # metadata propagation can not reach the backward.
        # Check this joint graph for example: https://gist.github.com/shunting314/5b684d7179680e337d178dcc77ff1b91
        self.assertEqual(metrics.num_auto_chunking, has_softmax and not dynamic_shape)

        # Only assert peak memory saving for large input. For small input, the saving can
        # be largely distorted by other memory allocation such as the tensor used to clear L2
        # cache by triton perf benchmarking API.
        if USE_LARGE_INPUT and metrics.num_auto_chunking > 0:
            expected_bound = M * N * dtype.itemsize
            self.assertTrue(
                peak_memory < expected_bound,
                f"Actual peak_memory {peak_memory}, expected bound {expected_bound}",
            )

    def test_matmul_trivial(self):
        self.common_matmul_test(has_softmax=False)

    def test_linear_trivial(self):
        self.common_matmul_test(has_softmax=False, use_bias=True)

    # Due to not able to generate an inplace version of a softmax like
    # kernel, having 2 chunks does not have large enough savings.
    # Use at least 4 chunks here.
    @config.patch("auto_chunker.num_chunk", config.auto_chunker.num_chunk or 4)
    def test_matmul_softmax(self):
        self.common_matmul_test(has_softmax=True)

    def test_matmul_softmax_dynamic_shape(self):
        self.common_matmul_test(has_softmax=True, dynamic_shape=True)

    @config.patch("auto_chunker.num_chunk", 4)
    def test_linear_softmax(self):
        self.common_matmul_test(has_softmax=True, use_bias=True)

    @config.patch("auto_chunker.num_chunk", config.auto_chunker.num_chunk or 16)
    @largeTensorTest("6GB", device=GPU_TYPE, inductor=True)
    def test_fused_linear_cel(self):
        B = 32
        T = 1024
        C = 768
        V = 50257

        dtype = torch.bfloat16

        mod = LinearAndCEL(C, V).cuda().to(dtype)

        def f(x, y):
            x.grad = None
            mod.linear.weight.grad = None
            mod.linear.bias.grad = None

            x = x * 2
            loss = mod(x, y)
            loss.backward()
            return loss

        opt_f = torch.compile(f)

        x = torch.randn(B, T, C, dtype=dtype, requires_grad=True, device="cuda")
        y = torch.randint(0, V, (B, T)).cuda()

        expect = (f(x, y), x.grad, mod.linear.weight.grad, mod.linear.bias.grad)
        torch.cuda.reset_peak_memory_stats()
        actual = (opt_f(x, y), x.grad, mod.linear.weight.grad, mod.linear.bias.grad)
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"Peak memory {peak_memory / 10**9:.6f} GB")

        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}")

        if DO_PERF_TEST:
            from triton.testing import do_bench

            ms_eager = do_bench(lambda: f(x, y))
            ms_opt = do_bench(lambda: opt_f(x, y))
            print(f"Eager v.s. Compile perf: {ms_eager:.3f}ms v.s. {ms_opt:.3f}ms")

        if DO_PERF_TEST and DO_PROFILING:
            # no need for warmup since we have done perf test previously
            with torch.profiler.profile() as p:
                for step in range(5):
                    with torch.profiler.record_function(f"Step {step}"):
                        opt_f(x, y)
            path = "/tmp/trace.json"
            print(f"Write the chrome trace to {path}")
            p.export_chrome_trace(path)

        self.assertEqual(metrics.num_auto_chunking, 1)
        expected_bound = B * T * V * x.dtype.itemsize
        self.assertTrue(
            peak_memory < expected_bound,
            f"Actual peak_memory {peak_memory}, expected bound {expected_bound}",
        )

    @config.patch("auto_chunker.num_chunk", config.auto_chunker.num_chunk or 16)
    @largeTensorTest("12GB", device=GPU_TYPE, inductor=True)
    @parametrize("gradient_accumulation_steps", [1, 2])
    def test_gradient_accumulation(self, gradient_accumulation_steps):
        B = 32
        T = 1024
        C = 768
        V = 50257

        dtype = torch.bfloat16

        mod = LinearAndCEL(C, V).cuda().to(dtype)

        def f(x, y):
            x.grad = None

            x = x * 2
            loss = mod(x, y) / gradient_accumulation_steps
            loss.backward()
            return loss

        def step(func, xs, ys):
            mod.linear.weight.grad = None
            mod.linear.bias.grad = None

            tot = 0
            for x, y in zip(xs, ys):
                loss = func(x, y)
                tot += loss.detach().item()
            return tot

        opt_f = torch.compile(f)

        xs = [
            torch.randn(B, T, C, dtype=dtype, requires_grad=True, device="cuda")
            for _ in range(gradient_accumulation_steps)
        ]
        ys = [
            torch.randint(0, V, (B, T)).cuda()
            for _ in range(gradient_accumulation_steps)
        ]

        expect = (
            step(f, xs, ys),
            *[x.grad for x in xs],
            mod.linear.weight.grad,
            mod.linear.bias.grad,
        )
        torch.cuda.reset_peak_memory_stats()
        actual = (
            step(opt_f, xs, ys),
            *[x.grad for x in xs],
            mod.linear.weight.grad,
            mod.linear.bias.grad,
        )
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"Peak memory {peak_memory / 10**9:.6f} GB")

        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}")

        self.assertEqual(metrics.num_auto_chunking, 1)
        expected_bound = B * T * V * xs[0].dtype.itemsize
        self.assertTrue(
            peak_memory < expected_bound,
            f"Actual peak_memory {peak_memory}, expected bound {expected_bound}",
        )

    def test_set_num_chunk_with_compile_options(self):
        B = 32
        T = 1024
        C = 768
        V = 50257

        dtype = torch.bfloat16

        options = {
            "auto_chunker.enable": True,
            "auto_chunker.num_chunk": 16,
            "auto_chunker.amplify_ratio_threshold": 10,
        }
        mod = torch.compile(LinearAndCEL(C, V).cuda().to(dtype), options=options)
        x = torch.randn(B, T, C, dtype=dtype, requires_grad=True, device="cuda")
        y = torch.randint(0, V, (B, T)).cuda()
        mod(x, y).backward()
        self.assertEqual(metrics.num_auto_chunking, 1)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CUDA_AND_TRITON:
        run_tests()
