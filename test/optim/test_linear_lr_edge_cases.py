import torch

from torch.testing._internal.common_utils import TestCase


class TestLinearLREdgeCases(TestCase):
    def test_zero_total_iters_is_rejected(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
        with self.assertRaisesRegex(ValueError, "total_iters must be greater than 0"):
            torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=0)


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
