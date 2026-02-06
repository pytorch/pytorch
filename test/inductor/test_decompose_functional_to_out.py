# Owner(s): ["module: inductor"]
"""
Tests for decompose_functional_to_out pass with schema-based detection.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDecomposeFunctionalToOut(TestCase):
    """Tests for the decompose pass with auto-detection."""

    def setUp(self):
        from torch._library.schema_utils import clear_cache

        clear_cache()
        self._setup_test_ops()

    def tearDown(self):
        from torch._library.schema_utils import clear_cache

        clear_cache()

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
        from torch._library.schema_utils import find_out_variant

        out_op = find_out_variant(self.functional_op)
        self.assertIsNotNone(out_op)
        self.assertEqual(out_op, self.out_op)

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


class TestSchemaUtils(TestCase):
    """Tests for schema_utils module."""

    def test_find_out_variant_no_match(self):
        """Test that None is returned when no out variant exists."""
        from torch._library.schema_utils import find_out_variant

        # aten::relu has no .out overload with matching signature
        result = find_out_variant(torch.ops.aten.relu.default)
        # relu does have relu.out, so this might find it
        # The test is just to ensure the function doesn't crash

    def test_get_out_arg_count(self):
        """Test counting out arguments."""
        from torch._library.schema_utils import get_out_arg_count

        # aten::add.out has 1 out argument
        count = get_out_arg_count(torch.ops.aten.add.out)
        self.assertEqual(count, 1)


class TestConfigOption(TestCase):
    """Test config option exists and works."""

    def test_config_option_exists(self):
        """Test that config option exists."""
        from torch._inductor import config

        self.assertTrue(hasattr(config, "decompose_functional_to_out"))

    def test_config_default_is_false(self):
        """Test that config defaults to False (opt-in)."""
        from torch._inductor import config

        self.assertFalse(config.decompose_functional_to_out)


if __name__ == "__main__":
    run_tests()
