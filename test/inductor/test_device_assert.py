# Owner(s): ["module: inductor"]

import torch
import torch._inductor.config
from torch._inductor import metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.triton_utils import requires_gpu_and_triton


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


@instantiate_parametrized_tests
class TestTorchDeviceAssertTrigger(TestCase):
    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_assert_should_throw(self, backend):
        def func():
            a = torch.tensor([1.0, -2.0], device="cpu")
            result = torch.all(a > 0)
            assert result, "should throw"  # noqa: S101

        def func_inline():
            a = torch.tensor([1.0, -2.0], device="cpu")
            assert torch.all(a > 0), "should throw"  # noqa: S101

        with self.assertRaisesRegex(RuntimeError, "should throw"):
            torch._dynamo.reset()
            f_c = torch.compile(func, backend=backend)
            f_c()

        with self.assertRaisesRegex(RuntimeError, "should throw"):
            torch._dynamo.reset()
            f_c = torch.compile(func_inline, backend=backend)
            f_c()

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_assert_should_not_throw(self, backend):
        def func():
            a = torch.tensor([1.0, 2.0], device="cpu")
            result = torch.all(a > 0)
            assert result, "should throw"  # noqa: S101

        def func_inline():
            a = torch.tensor([1.0, 2.0], device="cpu")
            assert torch.all(a > 0), "should throw"  # noqa: S101

        torch._dynamo.reset()
        f_c = torch.compile(func, backend=backend)
        f_c()

        torch._dynamo.reset()
        f_c = torch.compile(func_inline, backend=backend)
        f_c()

    @requires_gpu_and_triton
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_assert_fusion(self):
        torch._logging.set_logs(inductor_metrics=True)

        def func():
            a = torch.tensor([1.0, 2.0], device=device_type)
            result = torch.all(a > 0)
            assert result, "should throw"  # noqa: S101

        torch._dynamo.reset()
        f_c = torch.compile(func, backend="inductor")
        metrics.reset()
        self.assertEqual(metrics.generated_kernel_count, 0)
        f_c()
        self.assertEqual(metrics.generated_kernel_count, 1)
        torch._logging.set_logs()

    @requires_gpu_and_triton
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_run_assert_triton(self):
        @torch.compile(backend="inductor")
        def fn():
            a = torch.tensor([1.0, 2.0], device=device_type)
            result = torch.all(a > 0)
            assert result, "should throw"  # noqa: S101

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
