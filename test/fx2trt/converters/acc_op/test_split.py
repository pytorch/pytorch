# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests


class TestSplitConverter(AccTestCase):
    @parameterized.expand(
        [
            ("split_size", 3, 1),
            ("sections", [5, 2, 3], 1),
        ]
    )
    def test_split(self, _, split_size_or_sections, dim):
        class Split(nn.Module):
            def forward(self, x):
                return x.split(split_size_or_sections, dim)[0]

        inputs = [torch.randn(1, 10)]
        self.run_test(
            Split(),
            inputs,
            expected_ops={
                acc_ops.split
                if isinstance(split_size_or_sections, int)
                else acc_ops.slice_tensor
            },
            test_explicit_batch_dim=False,
        )

if __name__ == '__main__':
    run_tests()
