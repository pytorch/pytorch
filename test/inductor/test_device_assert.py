# Owner(s): ["module: inductor"]
import torch
import torch._inductor.config
from torch._inductor.compiler_bisector import BisectionResult, CompilerBisector
from torch._inductor.test_case import run_tests, TestCase


class TestTorchDeviceAssertTrigger(TestCase):
    def _run_assert_should_throw(self):
        def func():
            a = torch.tensor([1.0, -2.0])
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

    def _run_assert_should_not_throw(self):
        def func():
            a = torch.tensor([1.0, 2.0])
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

    def _run_assert_inline_expression_should_throw(self):
        def func():
            a = torch.tensor([1.0, -2.0])
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

    def _run_assert_inline_expression_should_not_throw(self):
        def func():
            a = torch.tensor([1.0, 2.0])
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
        self._run_assert_should_throw()
        self._run_assert_inline_expression_should_throw()

    @torch._inductor.config.patch(force_disable_caches=True)
    def test_assert_should_not_throw(self):
        self._run_assert_should_not_throw()
        self._run_assert_inline_expression_should_not_throw()

    @torch._inductor.config.patch(force_disable_caches=True, cpp_wrapper=True)
    def test_assert_should_throw_cpp_wrapper(self):
        self._run_assert_should_throw()
        self._run_assert_inline_expression_should_throw()

    @torch._inductor.config.patch(force_disable_caches=True, cpp_wrapper=True)
    def test_assert_should_not_throw_cpp_wrapper(self):
        self._run_assert_should_not_throw()
        self._run_assert_inline_expression_should_not_throw()


if __name__ == "__main__":
    run_tests()
