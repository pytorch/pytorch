# -*- coding: utf-8 -*-
# Owner(s): ["oncall: jit"]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.testing._internal.common_utils import run_tests, TestCase

import torch._lazy
import torch._lazy.metrics
import torch._lazy.ts_backend
torch._lazy.ts_backend.init()

class TestDynamicShapes(TestCase):
    def test_nonzero_narrow_copy(self):
        x_cpu = torch.rand(10)
        y_cpu = torch.nonzero(x_cpu)
        y0_cpu_size = y_cpu.sym_size()[0]
        b_cpu = torch.randn(10) # a base tensor so we can compute with symint
        _ = b_cpu.narrow_copy(0, 0, y0_cpu_size)

        # Same operations, but on LAZY tensor
        x_lazy = x_cpu.to("lazy")
        y_lazy = torch.nonzero(x_lazy)
        # y_lazy is a lazy tensor with the upper bound x_lazy.size(0)
        y0_lazy_size = y_lazy.sym_size()[0]
        b_lazy = b_cpu.to("lazy")
        _ = b_lazy.narrow_copy(0, 0, y0_lazy_size);
        # TODO: add python bindings to get upper bounds
        # ASSERT_EQ(y_lazy.sizes()[0], 10)


    @classmethod
    def setUpClass(cls) -> None:
        # Setup the dynamic shape mode
        cls.old_ssa_mode = torch._C._lazy._get_symbolic_shape_mode()
        torch._C._lazy._set_symbolic_shape_mode(True)
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        torch._C._lazy._set_symbolic_shape_mode(cls.old_ssa_mode)
        return super().tearDownClass()

if __name__ == '__main__':
    run_tests()
