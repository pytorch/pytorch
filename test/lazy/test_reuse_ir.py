# Owner(s): ["oncall: jit"]

import unittest

import torch
import torch._lazy
import torch._lazy.config
import torch._lazy.ir_cache
import torch._lazy.metrics as metrics
import torch._lazy.ts_backend
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    IS_WINDOWS,
    run_tests,
    TestCase,
)


torch._lazy.ts_backend.init()
torch._lazy.config.set_reuse_ir(True)


@unittest.skipIf(IS_WINDOWS, "To be fixed")
class TestLazyReuseIrDevice(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def testAdd(self, device):
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for _ in range(10):
            z += x + y

        for _ in range(10):
            z_lazy += x_lazy + y_lazy
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        reused = metrics.counter_value("IrNodeReused_torch::lazy::AddTensor")
        if reused < 14:
            raise AssertionError(
                f"Expected at least 14 reused AddTensor nodes, got {reused}"
            )
        metrics.reset()
        torch._lazy.ir_cache.reset()

    def testAddSub(self, device):
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for i in range(10):
            if i < 5:
                z += x + y
            else:
                z += x - y

        for i in range(10):
            if i < 5:
                z_lazy += x_lazy + y_lazy
            else:
                z_lazy += x_lazy - y_lazy
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        reused = metrics.counter_value("IrNodeReused_torch::lazy::AddTensor")
        if reused < 8:
            raise AssertionError(
                f"Expected at least 8 reused AddTensor nodes, got {reused}"
            )
        metrics.reset()
        torch._lazy.ir_cache.reset()

    def testAddSubFallback(self, device):
        torch._lazy.config.set_force_fallback("aten::sub")
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for i in range(10):
            if i < 5:
                z += x + y
            else:
                z += x - y

        for i in range(10):
            if i < 5:
                z_lazy += x_lazy + y_lazy
            else:
                z_lazy += x_lazy - y_lazy
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        reused = metrics.counter_value("IrNodeReused_torch::lazy::AddTensor")
        if reused < 8:
            raise AssertionError(
                f"Expected at least 8 reused AddTensor nodes, got {reused}"
            )
        metrics.reset()
        torch._lazy.ir_cache.reset()
        torch._lazy.config.set_force_fallback("")

    def testBatchNorm(self, device):
        x = torch.randn(16, 3, 224, 224, device=device)
        weight = torch.randn(3, device=device)
        bias = torch.randn(3, device=device)

        for _ in range(10):
            # BatchNorm2d does extra checks on dimensions which SymInts don't support yet
            # so we call `torch.ops.aten.native_batch_norm` to bypass the checks.
            z, _, _ = torch.ops.aten.native_batch_norm(
                x, weight, bias, None, None, True, 0.1, 1e-5
            )
            z_legit, _, _ = torch.ops.aten._native_batch_norm_legit(
                x, weight, bias, True, 0.1, 1e-5
            )

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        weight_lazy = weight.detach().clone().to(device=device)
        bias_lazy = bias.detach().clone().to(device=device)
        for _ in range(10):
            z_lazy, _, _ = torch.ops.aten.native_batch_norm(
                x_lazy, weight_lazy, bias_lazy, None, None, True, 0.1, 1e-5
            )
            z_legit_lazy, _, _ = torch.ops.aten._native_batch_norm_legit(
                x_lazy, weight_lazy, bias_lazy, True, 0.1, 1e-5
            )
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        torch.testing.assert_close(z_legit.cpu(), z_legit_lazy.cpu())
        reused = metrics.counter_value("IrNodeReused_torch::lazy::NativeBatchNorm")
        if reused < 7:
            raise AssertionError(
                f"Expected at least 7 reused NativeBatchNorm nodes, got {reused}"
            )
        metrics.reset()
        torch._lazy.ir_cache.reset()


instantiate_device_type_tests(
    TestLazyReuseIrDevice,
    globals(),
    only_for=("cpu", "cuda", "xpu"),
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests()
