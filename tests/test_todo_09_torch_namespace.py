# Owner(s): ["oncall: pt2"]
"""
Test that torch.Precompile is accessible from the torch namespace.

This test verifies TODO 9: Export `Precompile` class from `torch` namespace.
"""

from torch.testing._internal.common_utils import run_tests, TestCase


class TestPrecompileTorchNamespace(TestCase):
    def test_precompile_in_torch_namespace(self):
        """Test that torch.Precompile is accessible."""
        import torch

        self.assertTrue(
            hasattr(torch, "Precompile"), "torch.Precompile should be accessible"
        )

    def test_precompile_dynamo_accessible(self):
        """Test that torch.Precompile.dynamo is accessible."""
        import torch

        self.assertTrue(
            hasattr(torch.Precompile, "dynamo"),
            "torch.Precompile.dynamo should be accessible",
        )

    def test_precompile_aot_autograd_accessible(self):
        """Test that torch.Precompile.aot_autograd is accessible."""
        import torch

        self.assertTrue(
            hasattr(torch.Precompile, "aot_autograd"),
            "torch.Precompile.aot_autograd should be accessible",
        )

    def test_precompile_in_all(self):
        """Test that Precompile is in torch.__all__."""
        import torch

        self.assertIn(
            "Precompile", torch.__all__, "Precompile should be in torch.__all__"
        )


if __name__ == "__main__":
    run_tests()
