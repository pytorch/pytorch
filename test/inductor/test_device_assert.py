# Owner(s): ["module: inductor"]
import os
import subprocess
import sys

import torch
import torch._inductor.config
from torch._inductor import metrics
from torch._inductor.compiler_bisector import BisectionResult, CompilerBisector
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import skipIfRocm
from torch.testing._internal.triton_utils import requires_cuda_and_triton


class TestTorchDeviceAssertTrigger(TestCase):
    def _run_assert_should_throw(self, device):
        def func():
            a = torch.tensor([1.0, -2.0], device=device)
            result = torch.all(a > 0)
            assert result, "should throw"

        def test_fn():
            torch._dynamo.reset()
            f_c = torch.compile(func)

            try:
                f_c()
                return False
            except Exception:
                return True

        bisect_result = CompilerBisector.do_bisect(test_fn)
        # do_bisect return None if all system is passed else return BisectionResult
        self.assertNotIsInstance(bisect_result, BisectionResult)

    def _run_assert_should_not_throw(self, device):
        def func():
            a = torch.tensor([1.0, 2.0], device=device)
            result = torch.all(a > 0)
            assert result, "should throw"

        def test_fn():
            torch._dynamo.reset()
            f_c = torch.compile(func)

            try:
                f_c()
                return True
            except Exception:
                return False

        bisect_result = CompilerBisector.do_bisect(test_fn)
        self.assertNotIsInstance(bisect_result, BisectionResult)

    def _run_assert_inline_expression_should_throw(self, device):
        def func():
            a = torch.tensor([1.0, -2.0], device=device)
            assert torch.all(a > 0), "should throw"

        def test_fn():
            torch._dynamo.reset()
            f_c = torch.compile(func)

            try:
                f_c()
                return False
            except Exception:
                return True

        bisect_result = CompilerBisector.do_bisect(test_fn)
        self.assertNotIsInstance(bisect_result, BisectionResult)

    def _run_assert_inline_expression_should_not_throw(self, device):
        def func():
            a = torch.tensor([1.0, 2.0], device=device)
            assert torch.all(a > 0), "should throw"

        def test_fn():
            torch._dynamo.reset()
            f_c = torch.compile(func)

            try:
                f_c()
                return True
            except Exception:
                return False

        bisect_result = CompilerBisector.do_bisect(test_fn)
        self.assertNotIsInstance(bisect_result, BisectionResult)

    @torch._inductor.config.patch(force_disable_caches=True)
    def test_assert_should_throw(self):
        device = "cpu"
        self._run_assert_should_throw(device)
        self._run_assert_inline_expression_should_throw(device)

    @torch._inductor.config.patch(force_disable_caches=True)
    def test_assert_should_not_throw(self):
        device = "cpu"
        self._run_assert_should_not_throw(device)
        self._run_assert_inline_expression_should_not_throw(device)

    @torch._inductor.config.patch(force_disable_caches=True, cpp_wrapper=True)
    def test_assert_should_throw_cpp_wrapper(self):
        device = "cpu"
        self._run_assert_should_throw(device)
        self._run_assert_inline_expression_should_throw(device)

    @torch._inductor.config.patch(force_disable_caches=True, cpp_wrapper=True)
    def test_assert_should_not_throw_cpp_wrapper(self):
        device = "cpu"
        self._run_assert_should_not_throw(device)
        self._run_assert_inline_expression_should_not_throw(device)

    @requires_cuda_and_triton
    @skipIfRocm
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_assert_fusion(self):
        torch._logging.set_logs(inductor_metrics=True)

        def func():
            a = torch.tensor([1.0, 2.0], device="cuda")
            result = torch.all(a > 0)
            assert result, "should throw"

        torch._dynamo.reset()
        f_c = torch.compile(func, backend="inductor")
        metrics.reset()
        self.assertEqual(metrics.generated_kernel_count, 0)
        f_c()
        self.assertEqual(metrics.generated_kernel_count, 1)
        torch._logging.set_logs()

    @requires_cuda_and_triton
    @skipIfRocm
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_run_assert_triton(self):
        should_throw = """
import torch
import torch._dynamo

def func_should_throw():
    a = torch.tensor([1.0, -2.0], device='cuda')
    result = torch.all(a > 0)
    assert result, "should throw"

def test_fn():
    torch._dynamo.reset()
    f_c = torch.compile(func_should_throw, backend="inductor")

    try:
        f_c()
        torch.cuda.synchronize()
        return False
    except Exception as e:
        return True

result = test_fn()
print(f"Test result: {result}")
"""

        should_not_throw = """
import torch
import torch._dynamo

def func_should_not_throw():
    a = torch.tensor([1.0, 2.0], device='cuda')
    result = torch.all(a > 0)
    assert result, "should throw"

def test_fn():
    torch._dynamo.reset()
    f_c = torch.compile(func_should_not_throw, backend="inductor")

    try:
        f_c()
        torch.cuda.synchronize()
        return True
    except Exception as e:
        return False

result = test_fn()
print(f"Test result: {result}")
"""
        for script in [should_not_throw, should_throw]:
            p = subprocess.run(
                [sys.executable, "-c", script],
                cwd=os.path.dirname(os.path.realpath(__file__)),
                capture_output=True,
                text=True,
            )

            output = p.stdout + "\n" + p.stderr

            self.assertIn("Test result: True", output)

            if p.returncode != 0:
                self.fail(
                    f"Subprocess failed with return code {p.returncode}. Output: {output}"
                )


if __name__ == "__main__":
    run_tests()
