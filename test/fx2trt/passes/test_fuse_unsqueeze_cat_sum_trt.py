import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.fx.experimental.fx2trt.passes.fuse_pass import (
    fuse_unsqueeze_cat_sum,
)
from torch.testing._internal.common_utils import run_tests


class TestFuseUnsqueezeCatSum(AccTestCase):
    def test_fuse_unsqueeze_cat_sum(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                x_unsqueeze = torch.unsqueeze(x, 0)
                y_unsqueeze = torch.unsqueeze(y, 0)
                return torch.sum(torch.cat((x_unsqueeze, y_unsqueeze), 0), 0)

        inputs = [torch.randn(2, 3), torch.randn(2, 3)]
        self.run_test(
            TestModule(), inputs, {acc_ops.add}, apply_passes=[fuse_unsqueeze_cat_sum]
        )

    def test_not_fuse_unsqueeze_cat_sum(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                x_unsqueeze = torch.unsqueeze(x, 2)
                y_unsqueeze = torch.unsqueeze(y, 2)
                return torch.sum(torch.cat((x_unsqueeze, y_unsqueeze), 1), 1)

        inputs = [torch.randn(2, 3), torch.randn(2, 3)]
        self.run_test(
            TestModule(),
            inputs,
            {acc_ops.unsqueeze, acc_ops.cat, acc_ops.sum},
            apply_passes=[fuse_unsqueeze_cat_sum],
        )

if __name__ == '__main__':
    run_tests()
