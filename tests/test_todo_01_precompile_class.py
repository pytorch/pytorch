# Owner(s): ["module: dynamo"]

"""
Test for TODO 1: Verify torch.Precompile class stub exists and is accessible.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPrecompileClass(TestCase):
    """Test that torch.Precompile exists and is a class."""

    def test_precompile_exists(self):
        """Test that torch.Precompile exists."""
        self.assertTrue(
            hasattr(torch, "Precompile"), "torch.Precompile should exist"
        )

    def test_precompile_is_class(self):
        """Test that torch.Precompile is a class."""
        self.assertTrue(
            isinstance(torch.Precompile, type),
            "torch.Precompile should be a class",
        )


if __name__ == "__main__":
    run_tests()
