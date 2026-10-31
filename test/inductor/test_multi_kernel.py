# Owner(s): ["module: inductor"]

import contextlib
import os
import re
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch._dynamo.testing import reset_rng_state
from torch._dynamo.utils import counters
from torch._inductor import config, test_operators
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.runtime.benchmarking import set_gpu_benchmark_lock_context
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.nn import functional as F
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfXpu,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    IS_BIG_GPU,
    requires_triton,
)
from torch.utils._triton import has_triton_stable_tma_api


class TransformerSnippet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)

    def forward(self, x1, x2):
        x1 = F.dropout(x1, 0.1)
        x2 = F.dropout(self.ln1(x2), 0.1)

        return self.ln2(x1 + x2)

    def example_inputs(self):
        return (torch.randn(2, 64).to(GPU_TYPE), torch.randn(2, 64).to(GPU_TYPE))


def _contains_multi_kernel_code(wrapper_code: str):
    return (
        re.search(r"multi_kernel_[^ ]* = async_compile.multi_kernel[(]", wrapper_code)
        is not None
    )


def _contains_size_hint_multi_kernel_code(wrapper_code: str):
    return (
        re.search(
            r"multi_kernel_[^ ]* = async_compile.size_hint_multi_kernel[(]",
            wrapper_code,
        )
        is not None
    )


def make_cpp_wrapper_test(orig_test, **extra_args):
    """
    Wrap an existing test into a new test with cpp-wrapper enabled.

    Make this as a free function rather than staticmethod in MultiKernelTest.
    Otherwise we get 'TypeError: 'staticmethod' object is not callable'
    error in py3.8. (py3.10 works)
    """

    @config.patch("cpp_wrapper", True)
    @config.patch("triton.autotune_at_compile_time", True)
    def fn(self):
        # The same kernel may have been compiled by previous tests with
        # cpp_wrapper disabled. Clear the cache so we go ahead to re-compile
        # the kernel with cpp_wrapper enabled.
        from torch._inductor import codecache

        codecache.PyCodeCache.cache_clear()
        return orig_test(self, **extra_args)

    return fn


@config.patch(
    {
        "triton.multi_kernel": int(os.environ.get("TORCHINDUCTOR_MULTI_KERNEL", "1")),
        "benchmark_kernel": True,
        "multi_kernel_hints": [64, 256, 4096],
    }
)
@instantiate_parametrized_tests
class MultiKernelTest(TestCase):
    @staticmethod
    def _benchmark_lock_call(events):
        def kernel(index):
            return SimpleNamespace(
                clone_args=lambda *args, **kwargs: (args, kwargs),
                run=lambda *args, **kwargs: events.append(f"run_{index}"),
                device_props=SimpleNamespace(type="cuda"),
            )

        multi_kernel_call = object.__new__(MultiKernelCall)
        multi_kernel_call._kernels = [kernel(0), kernel(1)]
        multi_kernel_call.arg_index = {
            0: [slice(0, 1)],
            1: [slice(0, 1)],
        }
        return multi_kernel_call

    def test_run_binds_the_picked_kernel(self):
        events = []
        call = self._benchmark_lock_call(events)
        call.picked_kernel, call._recorded = 1, True
        call.run("arg")
        self.assertIs(call.run, call._kernels[1].run)
        # Args split across slices still need filtering, so that call stays unbound.
        split = self._benchmark_lock_call(events)
        split.arg_index = {i: [slice(0, 1), slice(2, 3)] for i in (0, 1)}
        split.picked_kernel, split._recorded = 0, True
        split.run("a", "b", "c")
        self.assertNotIn("run", vars(split))
        self.assertEqual(events, ["run_1", "run_0"])

    def test_failed_subkernels_remap_args_and_skip_single_survivor_benchmark(self):
        # A failed optional sub-kernel comes back as its exception, as one missing under no_compilation() does.
        with torch.compiler.precompile.no_compilation():
            failed = AsyncCompile().triton(
                "triton_missing", "def triton_missing():\n    pass\n", optional=True
            )
        events = []
        call = self._benchmark_lock_call(events)
        k0, k1 = call._kernels
        call.multi_kernel_name = "multi_kernel_test"
        call._kernels = [k0, failed, k1]
        call.arg_index = {i: [slice(i, i + 1)] for i in range(3)}
        self.assertEqual(call.kernels, [k0, k1])
        self.assertEqual(call.arg_index, {0: [slice(0, 1)], 1: [slice(2, 3)]})
        self.assertEqual(call._get_filtered_args(("a", "b", "c"), 1), ["c"])

        single = self._benchmark_lock_call(events)
        single.multi_kernel_name = "multi_kernel_test"
        single._kernels[0] = failed
        single.arg_index = {0: [slice(0, 1)], 1: [slice(1, 2)]}
        single.picked_kernel, single._recorded = None, True
        with unittest.mock.patch.object(single, "benchmark_sub_kernels") as benchmark:
            single.run("a", "b")
        benchmark.assert_not_called()
        self.assertEqual(single.picked_kernel, 0)
        self.assertEqual(events, ["run_1"])

    def test_benchmark_sub_kernels_holds_gpu_lock_across_candidates(self):
        events = []
        multi_kernel_call = self._benchmark_lock_call(events)

        @contextlib.contextmanager
        def benchmark_lock():
            events.append("lock_enter")
            try:
                yield
            finally:
                events.append("lock_exit")

        def benchmark(fn, **kwargs):
            index = len([event for event in events if event.startswith("benchmark_")])
            events.append(f"benchmark_{index}")
            fn()
            return 2.0 - index

        previous = set_gpu_benchmark_lock_context(benchmark_lock)
        try:
            with unittest.mock.patch(
                "torch._inductor.codegen.multi_kernel.benchmarker.benchmark",
                side_effect=benchmark,
            ):
                timings = multi_kernel_call.benchmark_sub_kernels("arg")
        finally:
            set_gpu_benchmark_lock_context(previous)

        self.assertEqual(timings, [2.0, 1.0])
        self.assertEqual(
            events,
            [
                "lock_enter",
                "benchmark_0",
                "run_0",
                "benchmark_1",
                "run_1",
                "lock_exit",
            ],
        )

    def test_benchmark_sub_kernels_releases_gpu_lock_on_failure(self):
        events = []
        multi_kernel_call = self._benchmark_lock_call(events)

        @contextlib.contextmanager
        def benchmark_lock():
            events.append("lock_enter")
            try:
                yield
            finally:
                events.append("lock_exit")

        previous = set_gpu_benchmark_lock_context(benchmark_lock)
        try:
            with (
                unittest.mock.patch(
                    "torch._inductor.codegen.multi_kernel.benchmarker.benchmark",
                    side_effect=RuntimeError("benchmark failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "benchmark failed"),
            ):
                multi_kernel_call.benchmark_sub_kernels("arg")
        finally:
            set_gpu_benchmark_lock_context(previous)

        self.assertEqual(events, ["lock_enter", "lock_exit"])

    @requires_triton()
    @config.patch("triton.multi_kernel", 1)
    def test_subgraph_multi_kernel_names_unique(self):
        def f(pred, x, z):
            branch = torch.cond(
                pred, lambda x: x.sum(-1), lambda x: (x * 3).sum(-1), (x,)
            )
            return (z.relu() * 2).sum(-1), branch

        x = torch.randn(1024, 256, device=GPU_TYPE)
        z = torch.randn(4096, 256, device=GPU_TYPE)
        pred = torch.tensor(True, device=GPU_TYPE)
        expected = f(pred, x, z)
        actual, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), pred, x, z)
        self.assertEqual(expected, actual)
        # Subgraph wrappers are emitted into the parent's module, so a name
        # defined by both would shadow the parent's definition.
        names = re.findall(r"^(multi_kernel_\d+) = ", code, re.MULTILINE)
        self.assertGreater(len(names), 1)
        self.assertEqual(len(names), len(set(names)), names)

    def test_softmax(self, expect_multi_kernel=True):
        x = torch.rand(2, 1024).to(GPU_TYPE)
        ref = torch.softmax(x, -1)
        compiled_fn = torch.compile(torch.softmax)
        act, wrapper_code = run_and_get_code(compiled_fn, x, -1)

        # wrapper_code will contains 2 entries if cpp_wrapper=True.
        # One for the first pass and one for the second pass.
        # We mainly care about the wrapper for the final pass here.
        wrapper_code = wrapper_code[-1]
        self.assertEqual(ref, act)
        if expect_multi_kernel:
            self.assertTrue(_contains_multi_kernel_code(wrapper_code))
        else:
            self.assertFalse(_contains_multi_kernel_code(wrapper_code))

    @requires_triton()
    @config.patch(
        {
            "bundle_triton_into_fx_graph_cache": True,
            "use_static_triton_launcher": True,
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "compile_threads": 1,
            "triton.multi_kernel": 1,
        }
    )
    def test_warm_fx_graph_cache(self):
        # The warm compile rebuilds sub-kernels from the FX graph cache, and MultiKernelCall hashes their fn.cache_key.
        from torch._inductor import codecache

        x = torch.rand(2, 1024, device=GPU_TYPE)
        ref = torch.softmax(x, -1)
        for _ in range(2):
            codecache.PyCodeCache.cache_clear(purge=True)
            result, (code,) = run_and_get_code(torch.compile(torch.softmax), x, -1)
            self.assertEqual(result, ref)
            self.assertTrue(_contains_multi_kernel_code(code))
        # Otherwise a cache miss on the second compile would pass without restoring any sub-kernel.
        self.assertGreater(
            counters["inductor"]["triton_bundler_load_static_autotuner"], 0
        )

    @requires_triton()
    @unittest.skipIf(
        not (SM90OrLater and has_triton_stable_tma_api()) or torch.version.hip,
        "requires NVIDIA sm90+ with Triton tensor descriptors",
    )
    @config.patch(
        {
            "assume_aligned_inputs": True,
            "compile_threads": 2,
            "max_autotune_pointwise": True,
            "triton.autotune_tensor_descriptor": "auto",
            "triton.cudagraphs": False,
            "triton.multi_kernel": 0,
            "triton.use_tensor_descriptor": True,
        }
    )
    def test_multi_kernel_resolves_triton_futures_once(self):
        if not AsyncCompile.wait_process_pool_ready():
            self.skipTest("requires the parallel compile pool")

        def fn(x, weight):
            scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
            return (x.float() * scale).to(x.dtype) * weight

        # This decoder reduction shape reproduces duplicate resolution.
        x = torch.randn(4096, 4096, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.ones(
            4096, device=GPU_TYPE, dtype=torch.bfloat16, requires_grad=True
        )
        with torch.no_grad():
            ref = fn(x, weight)

        def forward_backward(x, weight):
            out = torch.compile(fn)(x, weight)
            out.float().pow(2).mean().backward()
            return out

        counters.clear()
        actual, codes = run_and_get_code(forward_backward, x, weight)
        self.assertEqual(actual, ref)
        self.assertTrue(any(_contains_multi_kernel_code(code) for code in codes))
        self.assertIn("'tma_variant': True", "\n".join(codes))

        submitted = (
            counters["inductor"]["async_compile_cache_miss"]
            - counters["inductor"]["async_compile_cache_hit"]
        )
        self.assertGreater(submitted, 0)
        self.assertEqual(
            counters["inductor"]["async_compile_triton_get_result"],
            submitted,
            "Triton future resolved more than once",
        )

    @requires_triton()
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    @skipIfXpu(msg="driver issue, torch-xpu-ops: 2295")
    def test_triton_gemm(self):
        def fn(x, y):
            return x @ y

        compiled_fn = torch.compile(
            fn,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )
        x = torch.randn(4096, 4096, device=GPU_TYPE)
        y = torch.randn(4096, 4096, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 0)
        act, wrapper_code = run_and_get_code(compiled_fn, x, y)
        ref = fn(x, y)

        # wrapper_code will contains 2 entries if cpp_wrapper=True.
        # One for the first pass and one for the second pass.
        # We mainly care about the wrapper for the final pass here.
        wrapper_code = wrapper_code[-1]
        # fp32 accumulation-order divergence between the triton template and
        # hipblasLt at K=4096; measured and derived in issue #196494.
        rocm_tol = {"atol": 2e-3, "rtol": 1e-3} if TEST_WITH_ROCM else {}
        self.assertEqual(ref, act, **rocm_tol)
        self.assertTrue(_contains_size_hint_multi_kernel_code(wrapper_code))

    @skipIfXpu(msg="driver issue, torch-xpu-ops: 2295")
    @requires_triton()
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_triton_relu_fused_gemm(self):
        def fn(x, y):
            return (x @ y).relu()

        compiled_fn = torch.compile(
            fn,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )
        x = torch.randn(4096, 4096, device=GPU_TYPE)
        y = torch.randn(4096, 4096, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 0)
        act, wrapper_code = run_and_get_code(compiled_fn, x, y)
        ref = fn(x, y)

        # wrapper_code will contains 2 entries if cpp_wrapper=True.
        # One for the first pass and one for the second pass.
        # We mainly care about the wrapper for the final pass here.
        wrapper_code = wrapper_code[-1]
        # fp32 accumulation-order divergence between the triton template and
        # hipblasLt at K=4096; measured and derived in issue #196494.
        rocm_tol = {"atol": 2e-3, "rtol": 1e-3} if TEST_WITH_ROCM else {}
        self.assertEqual(ref, act, **rocm_tol)
        self.assertTrue(_contains_size_hint_multi_kernel_code(wrapper_code))

    @parametrize("force_kernel", (0, 1))
    @unittest.mock.patch.dict(
        os.environ, {"TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE": "1"}
    )
    def test_softmax_force_non_persistent_reduction(self, force_kernel):
        """
        Force a specific sub-kernel being picked by mocking the benchmark result.
        """
        x = torch.rand(2, 1024).to(GPU_TYPE)
        mock_latency = [0.2, 0.2]
        mock_latency[force_kernel] = 0.1  # this make sure force_kernel will be picked

        def f(x):
            return torch.softmax(x, -1) + force_kernel

        orig_run = MultiKernelCall.run
        picked_kernel = None

        def mock_run(self, *args, **kwargs):
            out = orig_run(self, *args, **kwargs)
            nonlocal picked_kernel
            picked_kernel = self.picked_kernel
            return out

        with (
            unittest.mock.patch.object(MultiKernelCall, "run", mock_run),
            unittest.mock.patch.object(
                MultiKernelCall,
                "benchmark_sub_kernels",
                lambda *args, **kwargs: mock_latency,
            ),
        ):
            torch.compile(f)(x)
        self.assertEqual(picked_kernel, force_kernel)

    @config.patch("warn_mix_layout", True)
    def test_softmax_warn_mixed_layout(self):
        self.test_softmax()

    test_softmax_cpp_wrapper = make_cpp_wrapper_test(
        test_softmax, expect_multi_kernel=True
    )
    test_softmax_autotune_at_compile_time = config.patch(
        "triton.autotune_at_compile_time", True
    )(test_softmax)
    test_triton_gemm_autotune_at_compile_time = config.patch(
        "triton.autotune_at_compile_time", True
    )(test_triton_gemm)

    def test_layernorm(self):
        ln = nn.LayerNorm(1024).to(GPU_TYPE)
        x = torch.rand(2, 1024).to(GPU_TYPE)
        ref = ln(x)
        act = torch.compile(ln)(x)
        self.assertEqual(ref, act, atol=1e-4, rtol=1e-4)

    def test_inplace_update(self):
        """
        Inductor generate inplace kernel for mul.
        """

        def f(x, y):
            return x.sum(dim=-1, keepdims=True) * (y @ y)

        x = torch.rand(1024, 1024).to(GPU_TYPE)
        y = torch.rand(1024, 1024).to(GPU_TYPE)
        ref = f(x, y)
        act = torch.compile(f)(x, y)
        self.assertEqual(ref, act)

    def test_transformer_snippet(self):
        model = TransformerSnippet().to(GPU_TYPE)
        x = model.example_inputs()

        def f(*x):
            y = model(*x)
            return y

        reset_rng_state()
        ref = f(*x)

        opt_f = torch.compile(f)
        reset_rng_state()
        act = opt_f(*x)

        # don't compare tensor if using inductor random number generator.
        # inductor random number implementation is different to eager.
        # We should fallback to eager if we want to test accuracy.
        if config.fallback_random:
            self.assertEqual(ref, act, atol=1e-4, rtol=1e-4)

    def test_transformer_snippet_with_fallback_random(self):
        """
        Same as test_transformer_snippet but fallback the random number
        generator to eager so we can check accuracy.
        """
        with config.patch("fallback_random", True):
            self.test_transformer_snippet()

    def test_batchnorm_training(self):
        """
        For training, batchnorm will tracking running mean/variance during forward pass.
        The kernel generated by inductor currently will pass in those tensors twice as arguments:
        once for input and once for output. They are ruled out as in-out argument because
        they are considered as graph inputs.

        Multi-kernel previously assumes that we never pass the same argument multi times
        for a kernel. No matter if we change inductor behavior to assure that, it's better
        to make multi-kernel being able to handle those cases.
        """
        bn = nn.BatchNorm2d(3).to(GPU_TYPE)

        @torch.compile
        def f(x):
            bn(x).sum().backward()

        _, (wrapper_code, _) = run_and_get_code(
            f, torch.randn(2, 3, 8, 8, device=GPU_TYPE)
        )
        self.assertTrue(_contains_multi_kernel_code(wrapper_code))

    def test_pass_same_arg_multi_times(self):
        """
        A super simple example that simulate how BatchNorm update the running
        stats.

        Inductor currently pass the same tensor multiple times for the generated
        kernel: once for input and once for output.

        Here is a paster for the generated kernel (without multi-kernel enabled):
        https://gist.github.com/shunting314/f0b446b4b9a28f4940e31dcd3e809cf9
        """

        def f(x, y):
            x = x.sum(dim=1, keepdim=False)
            y.copy_(y * 0.9 + x * 0.1)

        x = torch.randn(8, 16, device=GPU_TYPE)
        y = torch.randn(8, device=GPU_TYPE)
        y_ref = y.clone()

        ref = f(x, y_ref)  # noqa: F841
        act = torch.compile(f)(x, y)  # noqa: F841
        self.assertEqual(y_ref, y)

    def test_reduction_scratch_buffer(self, force_multi_kernel=1):
        """
        The explicitly realized buffer in the test function will be passed in
        as a scratch buffer for the non-persistent reduction kernel but
        can be skipped for the persistent reduction kernel.

        This causes different argument lists for non-persistent reduction kernel and
        persistent reduction kernel.

        Check documentation around torch._inductor.config.triton.multi_kernel about
        how to interpret the force_multi_kernel argument.
        """

        def f(x):
            x = x.sum(dim=-1, keepdim=True) + x
            x = test_operators.realize(x)
            x = x.sum(dim=-1, keepdim=True) + x
            return x

        x = torch.rand(16, 16, device=GPU_TYPE)
        ref = f(x)
        with config.patch("triton.multi_kernel", force_multi_kernel):
            act = torch.compile(f)(x)
        self.assertEqual(ref, act)

    def test_split_scan(self, force_multi_kernel=1):
        def f(x):
            x = x.view(-1)
            return torch.cumsum(x, 0)

        x = make_tensor(10, 3, 352, 352, low=0, dtype=torch.float32, device=GPU_TYPE)
        expect = f(x)
        with config.patch("triton.multi_kernel", force_multi_kernel):
            actual = torch.compile(f)(x)
        self.assertEqual(expect, actual)

    def test_sort_disables_multi_kernel(self, force_multi_kernel=1):
        """
        Sort currently requires a persistent kernel, so multi-kernel is not
        possible. Make sure this falls back gracefully.
        """

        def f(x):
            return x.sort(-1).values

        x = torch.rand(32, 32, device=GPU_TYPE)
        expect = f(x)
        with config.patch("triton.multi_kernel", force_multi_kernel):
            actual = torch.compile(f)(x)
        self.assertEqual(expect, actual)

    # Use benchmarking to pick the faster kernel
    test_reduction_scratch_buffer_cpp_wrapper = make_cpp_wrapper_test(
        test_reduction_scratch_buffer, force_multi_kernel=1
    )
    # force pick persistent reduction. This can be a good test since this persistent
    # reduction uses less call arguments than the corresponding non-persistent
    # reduction.
    test_reduction_scratch_buffer_cpp_wrapper_persistent_reduction = (
        make_cpp_wrapper_test(test_reduction_scratch_buffer, force_multi_kernel=2)
    )
    # force pick non-persistent reduction
    test_reduction_scratch_buffer_cpp_wrapper_non_persistent_reduction = (
        make_cpp_wrapper_test(test_reduction_scratch_buffer, force_multi_kernel=3)
    )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
