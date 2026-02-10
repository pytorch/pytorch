# Owner(s): ["oncall: distributed"]

import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
    _apply_to_modules,
    _build_prefix_set,
    _get_param_to_fqns,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPrefixSet(TestCase):
    """Tests for the prefix set optimization in _common_utils."""

    def test_build_prefix_set_simple(self):
        """Test that _build_prefix_set correctly builds prefix sets from FQNs."""
        fqns = ["layer1.weight", "layer1.bias", "layer2.weight"]
        prefix_set = _build_prefix_set(fqns)

        # Should contain all intermediate prefixes and the FQNs themselves
        expected = {
            "layer1.",
            "layer1.weight",
            "layer1.bias",
            "layer2.",
            "layer2.weight",
        }
        self.assertEqual(prefix_set, expected)

    def test_build_prefix_set_nested(self):
        """Test prefix set with deeply nested module hierarchy."""
        fqns = ["model.encoder.layer1.weight", "model.decoder.layer2.bias"]
        prefix_set = _build_prefix_set(fqns)

        expected = {
            "model.",
            "model.encoder.",
            "model.encoder.layer1.",
            "model.encoder.layer1.weight",
            "model.decoder.",
            "model.decoder.layer2.",
            "model.decoder.layer2.bias",
        }
        self.assertEqual(prefix_set, expected)

    def test_build_prefix_set_empty(self):
        """Test prefix set with empty FQN list."""
        prefix_set = _build_prefix_set([])
        self.assertEqual(prefix_set, set())

    def test_build_prefix_set_single_level(self):
        """Test prefix set with single-level names (no dots)."""
        fqns = ["weight", "bias"]
        prefix_set = _build_prefix_set(fqns)

        # Single-level names should just be themselves (no trailing dot)
        expected = {"weight", "bias"}
        self.assertEqual(prefix_set, expected)

    def test_apply_to_modules_with_prefix_set(self):
        """Test that _apply_to_modules correctly uses prefix set for filtering."""

        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer2 = nn.Linear(10, 10)
                self.layer3 = nn.Linear(10, 10)

        model = SimpleModel()

        # Collect visited module prefixes
        visited_prefixes = []

        def module_fn(module, prefix, tree_level):
            visited_prefixes.append(prefix)

        def return_fn():
            return visited_prefixes

        # Call with filter_fqns - should use prefix set internally
        param_fqns = [name for name, _ in model.named_parameters()]
        result = _apply_to_modules(
            model,
            module_fn,
            return_fn,
            param_fqns,
        )

        # Should visit the root and all layers
        self.assertIn("", result)  # root
        self.assertIn("layer1.", result)
        self.assertIn("layer2.", result)
        self.assertIn("layer3.", result)

    def test_get_param_to_fqns_consistency(self):
        """Test that _get_param_to_fqns produces correct mappings."""

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.ReLU(),
                    nn.Linear(10, 10),
                )
                self.layer2 = nn.Linear(10, 5)

        model = NestedModel()
        param_to_fqns = _get_param_to_fqns(model)

        # Each parameter should map to exactly one FQN (singleton list)
        for fqns in param_to_fqns.values():
            self.assertIsInstance(fqns, list)
            self.assertGreaterEqual(len(fqns), 1)
            for fqn in fqns:
                self.assertIsInstance(fqn, str)

        # Count should match the total number of parameters
        self.assertEqual(len(param_to_fqns), len(list(model.parameters())))

    def test_get_param_to_fqns_shared_params(self):
        """Test _get_param_to_fqns with shared parameters."""

        class SharedParamModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                # Share the weight with layer1
                self.layer2 = nn.Linear(10, 10)
                self.layer2.weight = self.layer1.weight

        model = SharedParamModel()

        # With dedup_shared_params=True (default)
        param_to_fqns = _get_param_to_fqns(model, dedup_shared_params=True)
        shared_weight = model.layer1.weight
        # Should only have the first FQN
        self.assertIn(shared_weight, param_to_fqns)
        fqns = param_to_fqns[shared_weight]
        self.assertEqual(len(fqns), 1)
        self.assertTrue(fqns[0] in ["layer1.weight", "layer2.weight"])

        # With dedup_shared_params=False
        param_to_fqns = _get_param_to_fqns(model, dedup_shared_params=False)
        shared_weight = model.layer1.weight
        self.assertIn(shared_weight, param_to_fqns)
        fqns = param_to_fqns[shared_weight]
        # Should have both FQNs
        self.assertEqual(len(fqns), 2)
        self.assertIn("layer1.weight", fqns)
        self.assertIn("layer2.weight", fqns)


if __name__ == "__main__":
    run_tests()
