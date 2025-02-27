# Owner(s): ["module: inductor"]

import os
import re
import unittest

import torch
from torch import nn
from torch._dynamo.testing import reset_rng_state
from torch._inductor import config, test_operators
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.nn import functional as F
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


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


def make_cpp_wrapper_test(orig_test, **extra_args):
    """
    Wrap an existing test into a new test with cpp-wrapper enabled.

    Make this as a free function rather than staticmethod in MultiKernelTest.
    Otherwise we get 'TypeError: 'staticmethod' object is not callable'
    error in py3.8. (py3.10 works)
    """

    @config.patch("cpp_wrapper", True)
    @skipIfXpu(msg="cpp wrapper doesn't currently work on the XPU stack")
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
    }
)
@instantiate_parametrized_tests
class MultiKernelTest(TestCase):
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

        Multi-kernel previously assumes that we never pass the same argument mutli times
        for a kernel. No mater if we change inductor behavior to assure that, it's better
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
        The explicited realized buffer in the test function will be passed in
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
