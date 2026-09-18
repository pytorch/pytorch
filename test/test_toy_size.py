# Owner(s): ["module: tests"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestToySize(TestCase):
    def test_numel(self):
        self.assertEqual(torch.Size([2, 3]).numel(), 6)


if __name__ == "__main__":
    run_tests()
