# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests


class TestTopKConverter(AccTestCase):
    @parameterized.expand(
        [
            ("top1", 1, -1),
            ("top2", 2, -1),
            ("none_dim", 1, None),
            ("smallest", 1, -1, False),
            ("top1_dim0", 1, 0, False),
        ]
    )
    def test_topk(self, _, k, dim, largest=True):
        class TopK(nn.Module):
            def __init__(self, k, dim):
                super().__init__()
                self.k = k
                self.dim = dim
                self.largest = largest

            def forward(self, x):
                if self.dim is not None:
                    out = torch.topk(
                        x, k=self.k, dim=self.dim, largest=self.largest, sorted=False
                    )
                else:
                    out = torch.topk(x, k=self.k, largest=self.largest, sorted=False)
                return out[0], out[1]

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            TopK(k, dim),
            inputs,
            expected_ops={acc_ops.topk},
            test_implicit_batch_dim=(dim != 0),
        )

if __name__ == '__main__':
    run_tests()
