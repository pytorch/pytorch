# Owner(s): ["module: inductor"]
import unittest
from unittest.mock import patch

import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch._inductor import config
from torch.testing._internal.common_utils import IS_JETSON, IS_MACOS, TEST_WITH_ASAN
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu


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

    @requires_gpu
    @inductor_config.patch("triton.inject_relu_bug_TESTING_ONLY", "compile_error")
    def test_after_aot_gpu_compile_error(self):
        self._test_after_aot(GPU_TYPE, "SyntaxError")

    @requires_gpu
    @inductor_config.patch("triton.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_after_aot_gpu_accuracy_error(self):
        self._test_after_aot(GPU_TYPE, "AccuracyError")

    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_constant_in_graph(self):
        run_code = """\
@torch.compile()
def inner(x):
    return torch.tensor(2) + torch.relu(x)

inner(torch.randn(2))
"""
        self._run_full_test(run_code, "aot", "AccuracyError", isolate=False)

    @requires_gpu
    @patch.object(config, "joint_graph_constant_folding", False)
    def test_rmse_improves_over_atol(self):
        # From https://twitter.com/itsclivetime/status/1651135821045719041?s=20
        run_code = """
@torch.compile()
def inner(x):
    return x - torch.tensor(655, dtype=torch.half, device='GPU_TYPE') * 100

inner(torch.tensor(655 * 100, dtype=torch.half, device='GPU_TYPE'))
""".replace(
            "GPU_TYPE", GPU_TYPE
        )

        # If we disable RMSE against fp64, this triggers accuracy error,
        # as the increased precision from torch.compile changes the result
        # of 655 * 100
        with dynamo_config.patch("same_two_models_use_fp64", False):
            self._run_full_test(
                run_code,
                "aot",
                "AccuracyError",
                isolate=False,
                # NB: need this to avoid refusing to minify when fp64 doesn't work
                # (which it doesn't, due to the config patch above)
                minifier_args=["--strict-accuracy"],
            )

        # But using fp64, we see that the intended semantics is the increased
        # 655 * 100 precision, and so we report no problem
        self._run_full_test(run_code, "aot", None, isolate=False)

    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
    @inductor_config.patch("cpp.inject_log1p_bug_TESTING_ONLY", "accuracy")
    def test_accuracy_vs_strict_accuracy(self):
        run_code = """
@torch.compile()
def inner(x):
    y = torch.log1p(x)
    b = y > 0
    # Need to ensure suffix removal hits a boolean output
    b = torch.logical_not(b)
    b = torch.logical_not(b)
    x = torch.relu(x)
    return torch.where(b, x, x)

inner(torch.randn(20))
"""

        # Strict accuracy gets hung up on the boolean mask difference, which
        # will localize the error to sigmoid, even though it doesn't actually
        # matter to the end result
        res = self._run_full_test(
            run_code,
            "aot",
            "AccuracyError",
            isolate=False,
            minifier_args=["--strict-accuracy"],
        )
        self.assertExpectedInline(
            res.repro_module(),
            """\
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, arg0_1):
        log1p = torch.ops.aten.log1p.default(arg0_1);  arg0_1 = None
        return (log1p,)""",
        )

        # FP accuracy will refuse to promote the logical_not on the outputs,
        # and so you'll get to the relu (unless the minifier somehow tries
        # removing entire suffix except the log1p first!)
        res = self._run_full_test(run_code, "aot", "AccuracyError", isolate=False)
        self.assertExpectedInline(
            res.repro_module(),
            """\
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, arg0_1):
        relu = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        return (relu,)""",
        )

    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_offload_to_disk(self):
        # Just a smoketest, this doesn't actually test that memory
        # usage went down.  Test case is carefully constructed to hit
        # delta debugging.
        run_code = """\
@torch.compile()
def inner(x):
    x = torch.sin(x)
    x = torch.sin(x)
    x = torch.cos(x)
    x = torch.relu(x)
    return x

inner(torch.randn(20, 20))
"""
        self._run_full_test(
            run_code,
            "aot",
            "AccuracyError",
            isolate=False,
            minifier_args=["--offload-to-disk"],
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # Skip CI tests on mac since CPU inductor does not seem to work due to C++ compile errors,
    # also skip on ASAN due to https://github.com/pytorch/pytorch/issues/98262
    if not IS_MACOS and not TEST_WITH_ASAN:
        run_tests()
