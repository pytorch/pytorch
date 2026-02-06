# Owner(s): ["module: inductor"]
"""
Tests for decompose_functional_to_out pass with schema-based detection.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDecomposeFunctionalToOut(TestCase):
    """Tests for the decompose pass with auto-detection."""

    def setUp(self):
        self._setup_test_ops()

    def _setup_test_ops(self):
        """Create test custom ops with .out convention."""

        # Functional op
        @torch.library.custom_op("test_decompose::add_one", mutates_args=())
        def add_one(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        @add_one.register_fake
        def _(x):
            return torch.empty_like(x)

        # Out variant with .out overload
        @torch.library.custom_op("test_decompose::add_one.out", mutates_args=("out",))
        def add_one_out(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x + 1)

        @add_one_out.register_fake
        def _(out, x):
            pass

        self.functional_op = torch.ops.test_decompose.add_one.default
        self.out_op = torch.ops.test_decompose.add_one.out

    def test_find_out_variant(self):
        """Test that schema-based detection finds the out variant."""
        from torch._library._out_variant import check_out_variant, to_out_variant

        out_op = to_out_variant(self.functional_op)
        self.assertIsNotNone(out_op)
        self.assertEqual(out_op, self.out_op)

        self.assertTrue(check_out_variant(self.functional_op, self.out_op))

    def test_decompose_with_config_enabled(self):
        """Test decomposition when config is enabled."""

        def fn(x):
            return torch.ops.test_decompose.add_one(x)

        x = torch.randn(4, 4)

        with torch._inductor.config.patch(decompose_functional_to_out=True):
            compiled = torch.compile(fn, backend="inductor")
            result = compiled(x)

        expected = x + 1
        torch.testing.assert_close(result, expected)

    def test_no_decompose_when_disabled(self):
        """Test that decomposition is skipped when config is disabled."""

        def fn(x):
            return torch.ops.test_decompose.add_one(x)

        x = torch.randn(4, 4)

        # Default is disabled
        compiled = torch.compile(fn, backend="inductor")
        result = compiled(x)

        expected = x + 1
        torch.testing.assert_close(result, expected)


class TestOutVariantDetection(TestCase):
    """Tests for out variant detection utilities."""

    def test_no_out_variant_returns_none(self):
        """Test that None is returned when no out variant exists."""
        from torch._library._out_variant import to_out_variant

        # Create an op without an out variant
        @torch.library.custom_op("test_no_out::my_op", mutates_args=())
        def my_op(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        @my_op.register_fake
        def _(x):
            return torch.empty_like(x)

        result = to_out_variant(torch.ops.test_no_out.my_op.default)
        self.assertIsNone(result)

    def test_native_op_out_variant(self):
        """Test detection works for native ops too."""
        from torch._library._out_variant import to_out_variant

        # aten::add has an out variant
        out_op = to_out_variant(torch.ops.aten.add.Tensor)
        self.assertIsNotNone(out_op)
        self.assertEqual(out_op, torch.ops.aten.add.out)


if __name__ == "__main__":
    run_tests()
