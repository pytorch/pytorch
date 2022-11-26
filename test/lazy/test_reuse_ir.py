# Owner(s): ["oncall: jit"]

import torch
import torch._lazy
import torch._lazy.config
import torch._lazy.ir_cache
import torch._lazy.ts_backend
import torch._lazy.metrics as metrics
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase
import os
import unittest

torch._lazy.ts_backend.init()
torch._lazy.config.set_reuse_ir(True)

def get_test_device():
    return 'cuda' if 'LTC_TS_CUDA' in os.environ else 'cpu'

@unittest.skipIf(IS_WINDOWS, "To be fixed")
class TestLazyReuseIr(TestCase):
    def testAdd(self):
        device = get_test_device()
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = 'lazy'
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for i in range(10):
            z += (x + y)

        for i in range(10):
            z_lazy += (x_lazy + y_lazy)
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        assert metrics.counter_value("IrNodeReused_torch::lazy::AddTensor") >= 14
        metrics.reset()
        torch._lazy.ir_cache.reset()

    def testAddSub(self):
        device = get_test_device()
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = 'lazy'
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for i in range(10):
            if i < 5:
                z += (x + y)
            else:
                z += (x - y)

        for i in range(10):
            if i < 5:
                z_lazy += (x_lazy + y_lazy)
            else:
                z_lazy += (x_lazy - y_lazy)
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        assert metrics.counter_value("IrNodeReused_torch::lazy::AddTensor") >= 8
        metrics.reset()
        torch._lazy.ir_cache.reset()

    def testAddSubFallback(self):
        torch._lazy.config.set_force_fallback("aten::sub")
        device = get_test_device()
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = 'lazy'
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for i in range(10):
            if i < 5:
                z += (x + y)
            else:
                z += (x - y)

        for i in range(10):
            if i < 5:
                z_lazy += (x_lazy + y_lazy)
            else:
                z_lazy += (x_lazy - y_lazy)
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        assert metrics.counter_value("IrNodeReused_torch::lazy::AddTensor") >= 8
        metrics.reset()
        torch._lazy.ir_cache.reset()
        torch._lazy.config.set_force_fallback("")

    def testBatchNorm(self):
        device = get_test_device()
        x = torch.randn(16, 3, 224, 224, device=device)
        weight = torch.randn(3, device=device)
        bias = torch.randn(3, device=device)

        for i in range(10):
            # BatchNorm2d does extra checks on dimensions which SymInts don't support yet
            # so we call `torch.ops.aten.native_batch_norm` to bypass the checks.
            z, _, _ = torch.ops.aten.native_batch_norm(x, weight, bias, None, None, True, 0.1, 1e-5)
            z_legit, _, _ = torch.ops.aten._native_batch_norm_legit(x, weight, bias, True, 0.1, 1e-5)

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        weight_lazy = weight.detach().clone().to(device=device)
        bias_lazy = bias.detach().clone().to(device=device)
        for i in range(10):
            z_lazy, _, _ = torch.ops.aten.native_batch_norm(x_lazy, weight_lazy, bias_lazy, None, None, True, 0.1, 1e-5)
            z_legit_lazy, _, _ = torch.ops.aten._native_batch_norm_legit(x_lazy, weight_lazy, bias_lazy, True, 0.1, 1e-5)
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        torch.testing.assert_close(z_legit.cpu(), z_legit_lazy.cpu())
        assert metrics.counter_value("IrNodeReused_torch::lazy::NativeBatchNorm") >= 7
        metrics.reset()
        torch._lazy.ir_cache.reset()


if __name__ == '__main__':
    run_tests()
