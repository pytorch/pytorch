# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests


class TestGetitemConverter(AccTestCase):
    @parameterized.expand(
        [
            ("slice_batch_dim", slice(None, None, None)),
            ("slice_basic", (slice(None, None, None), slice(0, 3, 2))),
            ("slice_full", (slice(None, None, None), slice(0, 10, 3))),
            ("ellipsis", (slice(None, None, None), ..., slice(0, 3, 2))),
            (
                "slice_all_none",
                (slice(None, None, None), slice(None, None, None)),
            ),
            (
                "slice_start_none",
                (slice(None, None, None), slice(None, 2, 1)),
            ),
            ("slice_end_none", (slice(None, None, None), slice(1, None, 1))),
            (
                "slice_step_none",
                (slice(None, None, None), slice(0, 3, None)),
            ),
            ("slice_neg_idx", (slice(None, None, None), -1)),
            ("slice_neg_slice", (slice(None, None, None), slice(-8, -2, 3))),
            ("multi_dim", (slice(None, None, None), 0, 1)),
            (
                "slice_multi_dim",
                (slice(None, None, None), slice(0, 3, 2), slice(1, -1, 3)),
            ),
            (
                "none",
                (slice(None, None, None), None, slice(1, -1, 3), 1),
            ),
        ]
    )
    def test_getitem(self, _, idx):
        class Getitem(nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.idx = idx

            def forward(self, x):
                x = x + x
                return x[self.idx]

        inputs = [torch.randn(2, 10, 10, 10)]
        self.run_test(Getitem(idx), inputs, expected_ops={acc_ops.getitem})

if __name__ == '__main__':
    run_tests()
