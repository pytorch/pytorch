# Owner(s): ["oncall: jit"]

import torch
import torch._lazy
import torch._lazy.ts_backend
from torch import float16, float32

from torch.testing._internal.common_utils import TestCase

torch._lazy.ts_backend.init()


class TestMetaKernel(TestCase):
    def test_addmm_invalid_dtype(self):
        """Tests that the addmm meta kernel returns the correct output type"""
        input = torch.ones(2, 2, dtype=torch.float16).to("lazy")
        self.assertTrue(input.dtype == torch.float16)

        fc_nobias = torch.nn.Linear(2, 2, bias=False, dtype=float32).to("lazy")

        with self.assertRaises(Exception):
            out_nobias = fc_nobias(input)

    def test_addmm(self):
        """Tests that the addmm meta kernel returns the correct output type"""
        input = torch.ones(2, 2, dtype=torch.float16).to("lazy")
        self.assertEqual(input.dtype, torch.float16)

        fc_nobias = torch.nn.Linear(2, 2, bias=False, dtype=float16).to("lazy")
        out_nobias = fc_nobias(input)
        self.assertEqual(out_nobias.dtype, torch.float16)

        fc_bias = torch.nn.Linear(2, 2, bias=True, dtype=float16).to("lazy")
        out_bias = fc_bias(input)
        self.assertEqual(out_bias.dtype, torch.float16)

    def test_add_invalid_device(self):
        with self.assertRaisesRegex(RuntimeError, ".*not a lazy tensor.*"):
            _ = torch.tensor([1], device="cpu") + torch.tensor([1], device="lazy")
