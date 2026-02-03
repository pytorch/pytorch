# Owner(s): ["module: inductor"]
"""
Minimal tests for functional_to_out registry API.
"""

import torch
from torch._library.functional_to_out import (
    clear_registry,
    FunctionalToOutMapping,
    get_out_variant,
    has_any_registered_mappings,
    register_functional_to_out,
    TensorSpec,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFunctionalToOutAPI(TestCase):
    """Minimal tests for registry API."""

    def setUp(self):
        clear_registry()
        self._setup_test_ops()

    def tearDown(self):
        clear_registry()

    def _setup_test_ops(self):
        """Create test custom ops."""

        @torch.library.custom_op("test_api::add_one", mutates_args=())
        def add_one(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        @add_one.register_fake
        def _(x):
            return torch.empty_like(x)

        @torch.library.custom_op("test_api::add_one_out", mutates_args=("out",))
        def add_one_out(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x + 1)

        @add_one_out.register_fake
        def _(out, x):
            pass

        self.functional_op = torch.ops.test_api.add_one.default
        self.out_op = torch.ops.test_api.add_one_out.default

    def test_register_and_lookup(self):
        """Test basic registration and lookup."""
        self.assertFalse(has_any_registered_mappings())

        register_functional_to_out(
            functional_op=self.functional_op,
            out_op=self.out_op,
            out_arg_positions=(0,),
        )

        self.assertTrue(has_any_registered_mappings())
        self.assertIsNotNone(get_out_variant(self.functional_op))

        mapping = get_out_variant(self.functional_op)
        self.assertEqual(mapping.functional_op, self.functional_op)
        self.assertEqual(mapping.out_op, self.out_op)

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises error."""
        register_functional_to_out(
            functional_op=self.functional_op,
            out_op=self.out_op,
            out_arg_positions=(0,),
        )

        with self.assertRaises(ValueError):
            register_functional_to_out(
                functional_op=self.functional_op,
                out_op=self.out_op,
                out_arg_positions=(0,),
            )

    def test_tensor_spec_allocate(self):
        """Test TensorSpec allocates correct tensor."""
        spec = TensorSpec(shape=(2, 3), dtype=torch.float32, device="cpu")
        tensor = spec.allocate()
        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_mapping_num_outputs(self):
        """Test FunctionalToOutMapping.num_outputs."""
        mapping = FunctionalToOutMapping(
            functional_op=self.functional_op,
            out_op=self.out_op,
            out_arg_positions=(0, 1, 2),
        )
        self.assertEqual(mapping.num_outputs, 3)


if __name__ == "__main__":
    run_tests()
