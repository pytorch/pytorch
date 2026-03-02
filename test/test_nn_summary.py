# Owner(s): ["module: nn"]

import unittest

import torch
import torch.nn as nn
from torch.nn.utils import summary
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    from torchvision import models as torchvision_models

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


class TestSummary(TestCase):
    """Tests for torch.nn.utils.summary"""

    def test_simple_sequential(self):
        """Basic Sequential model produces valid summary."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )
        s = summary(model, input_size=(10,))
        self.assertEqual(s.model_name, "Sequential")
        self.assertGreater(len(s.layer_info), 0)
        # Should have 4 layers: Sequential + 3 children
        self.assertEqual(len(s.layer_info), 4)

    def test_param_count_linear(self):
        """Parameter counting is correct for Linear layer."""
        model = nn.Linear(10, 5, bias=True)
        s = summary(model, input_size=(10,))
        # Linear(10, 5) with bias: 10*5 + 5 = 55 params
        self.assertEqual(s.total_params, 55)
        self.assertEqual(s.trainable_params, 55)

    def test_param_count_conv2d(self):
        """Parameter counting is correct for Conv2d layer."""
        model = nn.Conv2d(3, 16, kernel_size=3, bias=True)
        s = summary(model, input_size=(3, 32, 32))
        # Conv2d(3, 16, 3): 3*16*3*3 + 16 = 432 + 16 = 448 params
        self.assertEqual(s.total_params, 448)

    def test_output_shape(self):
        """Output shapes are captured correctly."""
        model = nn.Linear(10, 5)
        s = summary(model, input_size=(10,))
        # Find the Linear layer info (root module)
        self.assertEqual(len(s.layer_info), 1)
        linear_info = s.layer_info[0]
        self.assertEqual(linear_info.output_size, (1, 5))  # batch=1, out=5

    def test_output_shape_conv(self):
        """Output shapes are correct for Conv2d."""
        model = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        s = summary(model, input_size=(3, 32, 32))
        conv_info = s.layer_info[0]
        # With padding=1 and kernel=3, output size = input size
        self.assertEqual(conv_info.output_size, (1, 16, 32, 32))

    def test_nested_modules(self):
        """Nested modules are captured with correct depth."""
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 5),
            ),
        )
        s = summary(model, input_size=(10,))
        # Should have: Sequential (depth 0), Sequential (depth 1), Linear (depth 2)
        depths = [layer.depth for layer in s.layer_info]
        self.assertIn(0, depths)
        self.assertIn(1, depths)
        self.assertIn(2, depths)

    def test_depth_parameter(self):
        """Depth parameter limits displayed layers in output."""
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 5),
            ),
        )
        s = summary(model, input_size=(10,), depth=1)
        # max_depth should be set correctly
        self.assertEqual(s.max_depth, 1)
        # All layers should still be in layer_info (filtering happens in __repr__)
        self.assertEqual(len(s.layer_info), 3)

    def test_frozen_parameters(self):
        """Frozen parameters are counted as non-trainable."""
        model = nn.Linear(10, 5)
        # Freeze the weights
        model.weight.requires_grad = False
        s = summary(model, input_size=(10,))
        # Total: 55, trainable: only bias (5)
        self.assertEqual(s.total_params, 55)
        self.assertEqual(s.trainable_params, 5)

    def test_input_data(self):
        """Summary works with input_data instead of input_size."""
        model = nn.Linear(10, 5)
        x = torch.randn(2, 10)  # batch size 2
        s = summary(model, input_data=x)
        self.assertEqual(s.total_params, 55)
        # Output shape should reflect batch size 2
        self.assertEqual(s.layer_info[0].output_size, (2, 5))

    def test_no_input_raises(self):
        """ValueError raised when neither input_size nor input_data provided."""
        model = nn.Linear(10, 5)
        with self.assertRaises(ValueError) as ctx:
            summary(model)
        self.assertIn("input_size", str(ctx.exception))
        self.assertIn("input_data", str(ctx.exception))

    def test_model_mode_restored(self):
        """Model's training mode is restored after summary."""
        model = nn.Linear(10, 5)
        model.train()
        self.assertTrue(model.training)
        summary(model, input_size=(10,))
        self.assertTrue(model.training)

        model.eval()
        self.assertFalse(model.training)
        summary(model, input_size=(10,))
        self.assertFalse(model.training)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_model(self):
        """Summary works on CUDA device."""
        model = nn.Linear(10, 5).cuda()
        s = summary(model, input_size=(10,))
        self.assertEqual(s.total_params, 55)

    @skipIfNoTorchVision
    def test_resnet18(self):
        """Integration test with ResNet18."""
        model = torchvision_models.resnet18(weights=None)
        s = summary(model, input_size=(3, 224, 224))
        # ResNet18 has ~11.7M parameters
        self.assertGreater(s.total_params, 11_000_000)
        self.assertLess(s.total_params, 12_000_000)


if __name__ == "__main__":
    run_tests()
