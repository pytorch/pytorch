# Owner(s): ["module: inductor"]
"""
Minimal tests for decompose_functional_to_out pass.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._library.functional_to_out import (
    clear_registry,
    register_functional_to_out,
    get_out_variant,
)


class TestDecomposeFunctionalToOut(TestCase):
    """Tests for the decompose pass."""

    def setUp(self):
        clear_registry()
        self._setup_test_ops()

    def tearDown(self):
        clear_registry()

    def _setup_test_ops(self):
        """Create test custom ops."""

        @torch.library.custom_op("test_pass::add_one", mutates_args=())
        def add_one(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        @add_one.register_fake
        def _(x):
            return torch.empty_like(x)

        @torch.library.custom_op("test_pass::add_one_out", mutates_args=("out",))
        def add_one_out(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x + 1)

        @add_one_out.register_fake
        def _(out, x):
            pass

        self.functional_op = torch.ops.test_pass.add_one.default
        self.out_op = torch.ops.test_pass.add_one_out.default

    def test_decompose_single_output(self):
        """Test decomposition of a single-output functional op."""
        register_functional_to_out(
            functional_op=self.functional_op,
            out_op=self.out_op,
            out_arg_positions=(0,),
        )

        def fn(x):
            return torch.ops.test_pass.add_one(x)

        x = torch.randn(4, 4)

        with torch._inductor.config.patch(decompose_functional_to_out=True):
            compiled = torch.compile(fn, backend="inductor")
            result = compiled(x)

        expected = x + 1
        torch.testing.assert_close(result, expected)

    def test_no_decompose_when_disabled(self):
        """Test that decomposition is skipped when config is disabled."""
        register_functional_to_out(
            functional_op=self.functional_op,
            out_op=self.out_op,
            out_arg_positions=(0,),
        )

        def fn(x):
            return torch.ops.test_pass.add_one(x)

        x = torch.randn(4, 4)

        # Default is disabled
        compiled = torch.compile(fn, backend="inductor")
        result = compiled(x)

        expected = x + 1
        torch.testing.assert_close(result, expected)

    def test_no_decompose_unregistered_op(self):
        """Test that unregistered ops are not decomposed."""

        def fn(x):
            return torch.ops.test_pass.add_one(x)

        x = torch.randn(4, 4)

        with torch._inductor.config.patch(decompose_functional_to_out=True):
            compiled = torch.compile(fn, backend="inductor")
            result = compiled(x)

        expected = x + 1
        torch.testing.assert_close(result, expected)


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
