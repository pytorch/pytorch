# Owner(s): ["module: inductor"]

import torch
import torch._inductor.config
from torch._inductor import metrics
from unittest import skipIf

from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_TRITON


@instantiate_parametrized_tests
class TestTorchDeviceAssertTriggerGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC

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


class TestTorchDeviceAssertTriggerAccelerator(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @skipIf(not HAS_TRITON, "requires triton")
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_assert_fusion(self, device):
        torch._logging.set_logs(inductor_metrics=True)

        def func():
            a = torch.tensor([1.0, 2.0], device=device)
            result = torch.all(a > 0)
            assert result, "should throw"  # noqa: S101

        torch._dynamo.reset()
        f_c = torch.compile(func, backend="inductor")
        metrics.reset()
        self.assertEqual(metrics.generated_kernel_count, 0)
        f_c()
        self.assertEqual(metrics.generated_kernel_count, 1)
        torch._logging.set_logs()

    @skipIf(not HAS_TRITON, "requires triton")
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_run_assert_triton(self, device):
        @torch.compile(backend="inductor")
        def fn():
            a = torch.tensor([1.0, 2.0], device=device)
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


instantiate_device_type_tests(
    TestTorchDeviceAssertTriggerAccelerator,
    globals(),
    except_for="cpu",
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests()
