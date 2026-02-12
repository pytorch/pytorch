# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Tests for the strategy validation library.

These tests verify that the validation logic correctly detects:
- Incorrect rules (DTensor claims valid but produces wrong results)
- Missing rules (ground truth valid but DTensor has no rule)
"""

import torch
from torch.distributed.tensor import Replicate
from torch.distributed.tensor._ops.strategy_validation import (
    extract_tensors_from_sample,
    get_1d_input_placements_for_tensor,
    get_1d_output_placements_for_tensor,
    is_fully_replicated,
    is_trivial_shard,
    normalize_combo_key,
    normalize_placement,
    normalize_placement_str,
    parse_placement,
    placement_tuple_to_str,
)
from torch.distributed.tensor.placement_types import Partial, Shard
from torch.testing._internal.common_methods_invocations import SampleInput
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPlacementUtilities(TestCase):
    """Test placement utility functions."""

    def test_parse_placement_replicate(self):
        p = parse_placement("R")
        self.assertIsInstance(p, Replicate)

    def test_parse_placement_shard(self):
        p = parse_placement("S(0)")
        self.assertIsInstance(p, Shard)
        self.assertEqual(p.dim, 0)

        p = parse_placement("S(2)")
        self.assertIsInstance(p, Shard)
        self.assertEqual(p.dim, 2)

    def test_parse_placement_partial(self):
        p = parse_placement("P(sum)")
        self.assertIsInstance(p, Partial)
        self.assertEqual(p.reduce_op, "sum")

        p = parse_placement("P(max)")
        self.assertIsInstance(p, Partial)
        self.assertEqual(p.reduce_op, "max")

    def test_placement_tuple_to_str(self):
        s = placement_tuple_to_str((Replicate(),))
        self.assertEqual(s, "(R)")

        s = placement_tuple_to_str((Shard(0), Replicate()))
        self.assertEqual(s, "(S(0), R)")

        s = placement_tuple_to_str((Partial("sum"),))
        self.assertEqual(s, "(P(sum))")

    def test_is_fully_replicated(self):
        self.assertTrue(is_fully_replicated((Replicate(),)))
        self.assertTrue(is_fully_replicated((Replicate(), Replicate())))
        self.assertFalse(is_fully_replicated((Shard(0),)))
        self.assertFalse(is_fully_replicated((Replicate(), Shard(0))))
        self.assertFalse(is_fully_replicated((Partial("sum"),)))


class TestPlacementNormalization(TestCase):
    """Test placement normalization for trivial shard deduplication."""

    def test_is_trivial_shard_size_1(self):
        """Shard on size-1 dimension is trivial."""
        self.assertTrue(is_trivial_shard(Shard(0), (1, 4)))
        self.assertTrue(is_trivial_shard(Shard(1), (4, 1)))
        self.assertTrue(is_trivial_shard(Shard(0), (1, 1, 1)))
        self.assertTrue(is_trivial_shard(Shard(2), (4, 4, 1)))

    def test_is_trivial_shard_normal_size(self):
        """Shard on non-size-1 dimension is not trivial."""
        self.assertFalse(is_trivial_shard(Shard(0), (4, 3)))
        self.assertFalse(is_trivial_shard(Shard(1), (4, 3)))

    def test_is_trivial_shard_replicate(self):
        """Replicate is not a trivial shard."""
        self.assertFalse(is_trivial_shard(Replicate(), (1, 4)))

    def test_is_trivial_shard_partial(self):
        """Partial is not a trivial shard."""
        self.assertFalse(is_trivial_shard(Partial("sum"), (1, 4)))

    def test_normalize_placement_trivial_shard(self):
        """Trivial shards should normalize to Replicate."""
        result = normalize_placement(Shard(0), (1, 4))
        self.assertIsInstance(result, Replicate)

        result = normalize_placement(Shard(1), (4, 1))
        self.assertIsInstance(result, Replicate)

    def test_normalize_placement_normal_shard(self):
        """Non-trivial shards should stay as Shard."""
        result = normalize_placement(Shard(0), (4, 3))
        self.assertIsInstance(result, Shard)
        self.assertEqual(result.dim, 0)

    def test_normalize_placement_replicate(self):
        """Replicate should stay as Replicate."""
        result = normalize_placement(Replicate(), (4, 3))
        self.assertIsInstance(result, Replicate)

    def test_normalize_placement_partial(self):
        """Partial should stay as Partial."""
        result = normalize_placement(Partial("sum"), (4, 3))
        self.assertIsInstance(result, Partial)
        self.assertEqual(result.reduce_op, "sum")

    def test_normalize_placement_str_trivial_shard(self):
        """Trivial shard strings should normalize to 'R'."""
        result = normalize_placement_str("S(0)", (1, 4))
        self.assertEqual(result, "R")

        result = normalize_placement_str("S(1)", (4, 1))
        self.assertEqual(result, "R")

    def test_normalize_placement_str_normal_shard(self):
        """Non-trivial shard strings should stay unchanged."""
        result = normalize_placement_str("S(0)", (4, 3))
        self.assertEqual(result, "S(0)")

    def test_normalize_combo_key_trivial_output(self):
        """Combo keys with trivial output shard should normalize output to R."""
        # P(max) -> S(0) on output [1,1,1] should become P(max) -> R
        combo = (("P(max)",), "S(0)")
        input_shapes = ((4, 3),)
        output_shape = (1, 1, 1)

        result = normalize_combo_key(combo, input_shapes, output_shape)
        self.assertEqual(result, (("P(max)",), "R"))

    def test_normalize_combo_key_trivial_input(self):
        """Combo keys with trivial input shard should normalize input to R."""
        # S(0),R -> R on input [1,4],[4,4] should become R,R -> R
        combo = (("S(0)", "R"), "R")
        input_shapes = ((1, 4), (4, 4))
        output_shape = (4, 4)

        result = normalize_combo_key(combo, input_shapes, output_shape)
        self.assertEqual(result, (("R", "R"), "R"))

    def test_normalize_combo_key_all_trivial(self):
        """All-size-1 tensor should normalize all shards to R."""
        # S(0),S(1) -> S(2) on shape [1,1,1] should become R,R -> R
        combo = (("S(0)", "S(1)"), "S(2)")
        input_shapes = ((1, 1, 1), (1, 1, 1))
        output_shape = (1, 1, 1)

        result = normalize_combo_key(combo, input_shapes, output_shape)
        self.assertEqual(result, (("R", "R"), "R"))

    def test_normalize_combo_key_no_change(self):
        """Normal-sized tensors should not change."""
        combo = (("S(0)", "S(1)"), "S(0)")
        input_shapes = ((4, 3), (4, 3))
        output_shape = (4, 3)

        result = normalize_combo_key(combo, input_shapes, output_shape)
        self.assertEqual(result, combo)

    def test_normalize_combo_key_partial_unchanged(self):
        """Partial placements should never be normalized."""
        combo = (("P(max)", "P(sum)"), "P(max)")
        input_shapes = ((1, 1), (1, 1))  # Even with all-size-1
        output_shape = (1, 1)

        result = normalize_combo_key(combo, input_shapes, output_shape)
        self.assertEqual(result, combo)  # Partials unchanged

    def test_normalize_deduplicates_equivalent_rules(self):
        """
        Verify that normalization deduplicates equivalent rules.

        For squeeze on [1,1,1,1] -> [1,1,1], all of these are equivalent:
        - P(max) -> S(0)
        - P(max) -> S(1)
        - P(max) -> S(2)
        - P(max) -> R

        After normalization, they should all become P(max) -> R.
        """
        input_shapes = ((1, 1, 1, 1),)
        output_shape = (1, 1, 1)

        normalized_rules = set()
        for shard_dim in [0, 1, 2]:
            combo = (("P(max)",), f"S({shard_dim})")
            normalized = normalize_combo_key(combo, input_shapes, output_shape)
            normalized_rules.add(normalized)

        # Also add the explicit R version
        combo_r = (("P(max)",), "R")
        normalized_r = normalize_combo_key(combo_r, input_shapes, output_shape)
        normalized_rules.add(normalized_r)

        # All should have normalized to the same rule
        self.assertEqual(len(normalized_rules), 1)
        self.assertEqual(normalized_rules.pop(), (("P(max)",), "R"))


class TestInputPlacements(TestCase):
    """Test input/output placement generation."""

    def test_get_1d_input_placements(self):
        t = torch.randn(4, 3)
        placements = get_1d_input_placements_for_tensor(t, include_partial=False)
        # Should have Replicate + 2 Shard (one per dim)
        self.assertEqual(len(placements), 3)
        self.assertIsInstance(placements[0], Replicate)
        self.assertIsInstance(placements[1], Shard)
        self.assertIsInstance(placements[2], Shard)

    def test_get_1d_input_placements_with_partial(self):
        t = torch.randn(4, 3)
        placements = get_1d_input_placements_for_tensor(t, include_partial=True)
        # Should have Replicate + 2 Shard + 4 Partial
        self.assertEqual(len(placements), 7)

    def test_get_1d_output_placements_float(self):
        t = torch.randn(4, 3)
        placements = get_1d_output_placements_for_tensor(t)
        # Should have Replicate + 2 Shard + 4 Partial
        self.assertEqual(len(placements), 7)

    def test_get_1d_output_placements_integer(self):
        t = torch.randint(0, 10, (4, 3))
        placements = get_1d_output_placements_for_tensor(t)
        # Should have Replicate + 2 Shard + 4 Partial (all types)
        self.assertEqual(len(placements), 7)

    def test_get_1d_output_placements_scalar(self):
        t = torch.tensor(1.0)
        placements = get_1d_output_placements_for_tensor(t)
        # Scalars should have Replicate + 4 Partial (sum, avg, min, max)
        self.assertEqual(len(placements), 5)
        self.assertIsInstance(placements[0], Replicate)
        partial_ops = {p.reduce_op for p in placements if isinstance(p, Partial)}
        self.assertEqual(partial_ops, {"sum", "avg", "min", "max"})

    def test_get_1d_output_placements_integer_scalar(self):
        t = torch.tensor(1)
        placements = get_1d_output_placements_for_tensor(t)
        # Integer scalars should have Replicate + 4 Partial (all types)
        self.assertEqual(len(placements), 5)
        partial_ops = {p.reduce_op for p in placements if isinstance(p, Partial)}
        self.assertEqual(partial_ops, {"sum", "avg", "min", "max"})

    def test_get_1d_input_placements_scalar_with_partial(self):
        t = torch.tensor(1.0)
        placements = get_1d_input_placements_for_tensor(t, include_partial=True)
        # Scalars should have Replicate + 4 Partial
        self.assertEqual(len(placements), 5)
        partial_ops = {p.reduce_op for p in placements if isinstance(p, Partial)}
        self.assertEqual(partial_ops, {"sum", "avg", "min", "max"})

    def test_get_1d_input_placements_boolean_no_partial(self):
        t = torch.tensor(True)
        placements = get_1d_input_placements_for_tensor(t, include_partial=True)
        # Boolean tensors should have no Partial placements since Partial
        # decomposition creates float values that lose boolean semantics
        partial_placements = [p for p in placements if isinstance(p, Partial)]
        self.assertEqual(len(partial_placements), 0)

    def test_get_1d_output_placements_boolean_no_partial(self):
        t = torch.tensor(True)
        placements = get_1d_output_placements_for_tensor(t)
        # Boolean outputs should have no Partial placements
        partial_placements = [p for p in placements if isinstance(p, Partial)]
        self.assertEqual(len(partial_placements), 0)

    def test_get_1d_input_placements_boolean_2d(self):
        t = torch.tensor([[True, False], [False, True]])
        placements = get_1d_input_placements_for_tensor(t, include_partial=True)
        # Boolean 2D: Replicate + 2 Shard, no Partial
        self.assertEqual(len(placements), 3)
        partial_placements = [p for p in placements if isinstance(p, Partial)]
        self.assertEqual(len(partial_placements), 0)


class TestExtractTensors(TestCase):
    """Test tensor extraction from samples."""

    def test_extract_single_input(self):
        sample = SampleInput(torch.randn(4, 3))
        tensors = extract_tensors_from_sample(sample)
        self.assertEqual(len(tensors), 1)
        self.assertEqual(tensors[0][0], "tensor_0")

    def test_extract_with_args(self):
        sample = SampleInput(
            torch.randn(4, 3),
            args=(
                torch.randn(
                    3,
                ),
            ),
        )
        tensors = extract_tensors_from_sample(sample)
        self.assertEqual(len(tensors), 2)

    def test_extract_with_kwargs(self):
        sample = SampleInput(torch.randn(4, 3), kwargs={"other": torch.randn(4, 3)})
        tensors = extract_tensors_from_sample(sample)
        self.assertEqual(len(tensors), 2)


if __name__ == "__main__":
    run_tests()
