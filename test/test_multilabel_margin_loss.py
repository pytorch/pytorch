# Owner(s): ["module: nn"]

import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests


class TestMultiLabelMarginLossErrors(TestCase):
    def test_input_shape_error(self):
        loss = nn.MultiLabelMarginLoss()
        input = torch.rand(2, 3, 4)
        target = torch.zeros(2, 3, dtype=torch.long)

        with self.assertRaisesRegex(
            ValueError,
            "Expected input to be a scalar, non-empty 1D tensor, or non-empty 2D tensor",
        ):
            loss(input, target)

    def test_batched_target_shape_error(self):
        loss = nn.MultiLabelMarginLoss()
        input = torch.rand(1, 3)
        target = torch.zeros(2, 2, dtype=torch.long)

        with self.assertRaisesRegex(
            ValueError,
            r"Expected target to be 2D with shape \[1, 3\] to match input",
        ):
            loss(input, target)

    def test_unbatched_target_shape_error(self):
        loss = nn.MultiLabelMarginLoss()
        input = torch.rand(3)
        target = torch.zeros(2, 2, dtype=torch.long)

        with self.assertRaisesRegex(
            ValueError,
            "Expected target to have at most 1 dimension and 3 elements to match input",
        ):
            loss(input, target)


if __name__ == "__main__":
    run_tests()
