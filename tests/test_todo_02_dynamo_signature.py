# Test that Precompile.dynamo exists and is callable
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPrecompileDynamoSignature(TestCase):
    def test_dynamo_exists(self):
        """Test that Precompile.dynamo exists."""
        assert hasattr(torch.Precompile, "dynamo"), "Precompile.dynamo should exist"

    def test_dynamo_callable(self):
        """Test that Precompile.dynamo is callable."""
        assert callable(torch.Precompile.dynamo), "Precompile.dynamo should be callable"


if __name__ == "__main__":
    run_tests()
