# Owner(s): ["module: dynamo"]

"""
Test for TODO 7: Verify that regional_inductor compiler exists and is callable.
"""

from torch.fx.passes.regional_inductor import regional_inductor
from torch.testing._internal.common_utils import run_tests, TestCase


class TestRegionalInductorExists(TestCase):
    """Test that regional_inductor is importable and callable."""

    def test_regional_inductor_importable(self):
        """Test that regional_inductor can be imported from torch.fx.passes.regional_inductor."""
        # Import already done at module level, verify it's not None
        self.assertIsNotNone(
            regional_inductor, "regional_inductor should be importable"
        )

    def test_regional_inductor_callable(self):
        """Test that regional_inductor is callable."""
        self.assertTrue(
            callable(regional_inductor), "regional_inductor should be callable"
        )


if __name__ == "__main__":
    run_tests()
