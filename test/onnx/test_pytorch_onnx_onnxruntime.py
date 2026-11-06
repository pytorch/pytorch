# Owner(s): ["module: onnx"]

import torch
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class TestONNXRuntime(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_split_tensor_multi(self):
        class SplitModel(torch.nn.Module):
            def forward(self, x):
                return torch.split(x, torch.ones(3))

        x = torch.randn(1, 2, 3, requires_grad=True)

        def run_model():
            SplitModel(x)

        self.assertRaises(TypeError, run_model)


if __name__ == "__main__":
    run_tests()
