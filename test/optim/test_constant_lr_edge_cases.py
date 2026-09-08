import torch

from torch.testing._internal.common_utils import TestCase


class TestConstantLREdgeCases(TestCase):
    def test_zero_factor_is_rejected(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
        with self.assertRaisesRegex(ValueError, "factor must be greater than 0"):
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.0)


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
