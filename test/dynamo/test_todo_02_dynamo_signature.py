# Owner(s): ["module: dynamo"]

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPrecompileDynamoSignature(TestCase):
    """Test torch.Precompile.dynamo() signature and basic functionality."""

    def test_dynamo_is_callable(self):
        """Test that torch.Precompile.dynamo is callable."""
        self.assertTrue(
            callable(getattr(torch.Precompile, "dynamo", None)),
            "torch.Precompile.dynamo should be callable",
        )

    def test_dynamo_returns_four_values(self):
        """Test that dynamo() returns 4 values: (gm, bytecode, guards, example_inputs)."""

        class SimpleModel(nn.Module):
            def forward(self, x):
                return x * 2

        model = SimpleModel()
        example_input = torch.randn(2, 3)

        result = torch.Precompile.dynamo(model, example_input)
        self.assertEqual(
            len(result), 4, "dynamo() should return 4 values: (gm, bytecode, guards, example_inputs)"
        )

    def test_dynamo_returns_graph_module(self):
        """Test that the first return value is an FX GraphModule."""

        class SimpleModel(nn.Module):
            def forward(self, x):
                return x * 2

        model = SimpleModel()
        example_input = torch.randn(2, 3)

        gm, bytecode, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
        self.assertTrue(
            hasattr(gm, "graph"), "First return value should be an FX GraphModule"
        )

    def test_dynamo_captures_simple_model(self):
        """Test that dynamo() can capture a simple model."""

        class SimpleModel(nn.Module):
            def forward(self, x):
                return x + 1

        model = SimpleModel()
        example_input = torch.randn(2, 3)

        gm, bytecode, guards, example_inputs = torch.Precompile.dynamo(model, example_input)

        # Verify the graph module can be executed
        output = gm(example_input)
        expected = example_input + 1
        self.assertTrue(torch.allclose(output, expected))


if __name__ == "__main__":
    run_tests()
