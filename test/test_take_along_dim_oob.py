import torch
from torch.testing._internal.common_utils import TestCase


class TestTakeAlongDimOOB(TestCase):
    def test_explicit_dim_rejects_positive_oob(self):
        x = torch.tensor([10.0, 20.0, 30.0])
        with self.assertRaises(IndexError):
            torch.take_along_dim(x, torch.tensor([3]), dim=0)

    def test_explicit_dim_rejects_negative_oob(self):
        x = torch.tensor([10.0, 20.0, 30.0])
        with self.assertRaises(IndexError):
            torch.take_along_dim(x, torch.tensor([-4]), dim=0)

    def test_explicit_dim_keeps_valid_negative_indices(self):
        x = torch.tensor([10.0, 20.0, 30.0])
        self.assertEqual(
            torch.take_along_dim(x, torch.tensor([-1, -3]), dim=0),
            torch.tensor([30.0, 10.0]),
        )


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
