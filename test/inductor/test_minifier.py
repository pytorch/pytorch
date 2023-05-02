# Owner(s): ["module: inductor"]
import functools
import unittest

import torch
import torch._dynamo
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch._inductor.utils
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch.testing._internal.common_utils import IS_JETSON, IS_MACOS, TEST_WITH_ASAN

_HAS_TRITON = torch._inductor.utils.has_triton()
requires_cuda = functools.partial(unittest.skipIf, not _HAS_TRITON, "requires cuda")


class MinifierTests(MinifierTestBase):
    # Test that compile and accuracy errors after aot can be repro'd (both CPU and CUDA)
    def _test_after_aot(self, device, expected_error):
        # NB: The program is intentionally quite simple, just enough to
        # trigger one minification step, no more (dedicated minifier tests
        # should exercise minifier only)
        run_code = f"""\
@torch.compile()
def inner(x):
    x = torch.relu(x)
    x = torch.cos(x)
    return x

inner(torch.randn(20, 20).to("{device}"))
"""
        self._run_full_test(run_code, "aot", expected_error, isolate=False)

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "compile_error")
    def test_after_aot_cpu_compile_error(self):
        self._test_after_aot("cpu", "CppCompileError")

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_after_aot_cpu_accuracy_error(self):
        self._test_after_aot("cpu", "AccuracyError")

    @requires_cuda()
    @inductor_config.patch("triton.inject_relu_bug_TESTING_ONLY", "compile_error")
    def test_after_aot_cuda_compile_error(self):
        self._test_after_aot("cuda", "SyntaxError")

    @requires_cuda()
    @inductor_config.patch("triton.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_after_aot_cuda_accuracy_error(self):
        self._test_after_aot("cuda", "AccuracyError")

    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_constant_in_graph(self):
        run_code = """\
@torch.compile()
def inner(x):
    return torch.tensor(2) + torch.relu(x)

inner(torch.randn(2))
"""
        self._run_full_test(run_code, "aot", "AccuracyError", isolate=False)

    @requires_cuda
    def test_rmse_improves_over_atol(self):
        # From https://twitter.com/itsclivetime/status/1651135821045719041?s=20
        run_code = """
@torch.compile()
def inner(x):
    return x - torch.tensor(655, dtype=torch.half, device='cuda') * 100

inner(inner(torch.tensor(655 * 100, dtype=torch.half, device='cuda')))
"""

        # If we disable RMSE against fp64, this triggers accuracy error,
        # as the increased precision from torch.compile changes the result
        # of 655 * 100
        with dynamo_config.patch("same_two_models_use_fp64", False):
            self._run_full_test(run_code, "aot", "AccuracyError", isolate=False)

        # But using fp64, we see that the intended semantics is the increased
        # 655 * 100 precision, and so we report no problem
        self._run_full_test(run_code, "aot", None, isolate=False)


if __name__ == "__main__":
    import sys

    from torch._dynamo.test_case import run_tests

    # Skip CI tests on mac since CPU inductor does not seem to work due to C++ compile errors,
    # also skip on ASAN due to https://github.com/pytorch/pytorch/issues/98262
    # also skip on Py 3.11+ since unhandled exceptions can cause segfaults
    if not IS_MACOS and not TEST_WITH_ASAN and sys.version_info < (3, 11):
        run_tests()
