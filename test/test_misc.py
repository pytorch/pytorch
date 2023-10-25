# Owner(s): ["module: tests"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestBasic(TestCase):
    def test_size_stride(self) -> None:
        t = torch.rand(2, 3, dtype=torch.float32)
        self.assertEqual(t.size(0), 2)
        self.assertEqual(t.size(dim=None), torch.Size([2, 3]))
        self.assertEqual(t.stride(dim=None), torch.Size([3, 1]))
        self.assertEqual(t.t().stride(), torch.Size([1, 3]))


if __name__ == "__main__":
    run_tests()
