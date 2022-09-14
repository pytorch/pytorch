# Owner(s): ["oncall: jit"]

import torch

from torch.testing._internal.common_utils import TestCase
import torch._lazy
import torch._lazy.ts_backend

torch._lazy.ts_backend.init()

class TestMetaKernel(TestCase):

    def test_mixed_precision_addmm(self):
        """Tests that the addmm meta kernel returns the correct output type"""
        input = torch.ones(2, 2, dtype=torch.float16).to("lazy")
        self.assertTrue(input.dtype == torch.float16)

        fc_nobias = torch.nn.Linear(2, 2, bias=False, dtype=float32).to("lazy")
        out_nobias = fc_nobias(input)
        self.assertTrue(out_nobias.dtype == torch.float16)

        fc_bias = torch.nn.Linear(2, 2, bias=True, dtype=float32).to("lazy")
        out_bias = fc_bias(input)
        self.assertTrue(out_bias.dtype == torch.float16)
