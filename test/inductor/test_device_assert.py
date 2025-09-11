# Owner(s): ["module: inductor"]

import torch
import torch._inductor.config
from torch._inductor import metrics
from torch._inductor.compiler_bisector import BisectionResult, CompilerBisector
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
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
        @torch.compile(backend="inductor")
        def fn():
            a = torch.tensor([1.0, 2.0], device="cuda")
            result = torch.all(a > 0)
            assert result, "should throw"

        def should_not_throw(fn):
            try:
                fn()
                return True
            except Exception:
                return False

        self.assertEqual(should_not_throw(fn), True)

        _, code = run_and_get_code(fn)
        self.assertEqual(code[0].count("tl.device_assert"), 1)


if __name__ == "__main__":
    run_tests()
